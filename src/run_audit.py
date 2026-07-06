"""Run the seeded-defect QA experiment.

Serves the pages/ directory on localhost, then has the Browser Use agent audit
each page K times. Every run's report is scored deterministically against the
seeded-defect manifest (src/score.py) and logged with its full trajectory.

Usage:
    python -m src.run_audit                 # both pages, K runs each
    python -m src.run_audit defect.html 1   # one page, 1 run (pilot)
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.score import load_manifest, recall_by_class, score_report

PAGES_DIR = Path(__file__).parent.parent / "pages"
PORT = 8765

AUDIT_PROMPT = (
    "Open {url} and audit this single web page as a meticulous web QA tester. "
    "Follow this procedure step by step:\n"
    "1. Extract and READ the complete visible text of the page from top to "
    "bottom. Look for text that is garbled or meaningless, statements that "
    "contradict each other anywhere on the page (compare header, sections and "
    "footer), prices or dates or years that are inconsistent or impossible, "
    "and anything a visitor would consider wrong.\n"
    "2. Check every link: does its label honestly describe where it leads? "
    "Click each link to verify it works (a 404 or empty target is a defect), "
    "then navigate back.\n"
    "3. Check every image: does it load, and does it have an alt description?\n"
    "4. Check every form and button: click it and observe whether anything "
    "actually happens; a form that submits nowhere is a defect.\n"
    "Then report EVERY defect you found as a numbered list, one defect per "
    "line, naming the affected element and what is wrong with it. If you "
    "found no defects, answer 'no defects found'."
)


def prewarm_model() -> None:
    """Load the model into memory before the agent's first step, so cold-load
    time (weights + KV cache at num_ctx) is not billed to a step timeout."""
    import ollama

    client = ollama.Client(host=config.OLLAMA_HOST)
    client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": "Say OK"}],
        think=False,
        keep_alive="60m",
        options={"num_ctx": config.OLLAMA_NUM_CTX},
    )


def serve_pages() -> None:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(PAGES_DIR)
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def _trajectory_stats(history) -> dict:
    """Per-step instrumentation: duration, URL observed, actions, error."""
    steps = []
    for h in history.history:
        entry: dict = {}
        md = getattr(h, "metadata", None)
        if md is not None:
            entry["n"] = md.step_number
            entry["duration_s"] = round(md.duration_seconds, 2)
        state = getattr(h, "state", None)
        if state is not None and getattr(state, "url", None):
            entry["url"] = state.url
        out = getattr(h, "model_output", None)
        if out is not None and getattr(out, "action", None):
            names: list[str] = []
            for a in out.action:
                try:
                    names.extend(a.model_dump(exclude_unset=True, exclude_none=True).keys())
                except Exception:
                    pass
            entry["actions"] = names
        errors = [r.error for r in (h.result or []) if getattr(r, "error", None)]
        if errors:
            entry["error"] = str(errors[0])[:200]
        steps.append(entry)
    return {
        "n_steps": history.number_of_steps(),
        "agent_steps_duration_s": round(history.total_duration_seconds(), 2),
        "steps": steps,
    }


async def run_once(page: str) -> tuple[str, dict]:
    from browser_use import Agent, BrowserProfile

    from src.local_llm import ThinklessChatOllama

    llm = ThinklessChatOllama(
        model=config.OLLAMA_MODEL,
        host=config.OLLAMA_HOST,
        ollama_options={"temperature": config.AGENT_TEMPERATURE,
                        "num_ctx": config.OLLAMA_NUM_CTX},
    )
    agent = Agent(
        task=AUDIT_PROMPT.format(url=f"http://127.0.0.1:{PORT}/{page}"),
        llm=llm,
        browser_profile=BrowserProfile(headless=config.HEADLESS),
        use_vision=False,  # text-DOM observation channel; a measured limitation
        llm_timeout=config.OLLAMA_LLM_TIMEOUT_S,
        step_timeout=config.OLLAMA_LLM_TIMEOUT_S + 120,
    )
    t0 = time.monotonic()
    history = await agent.run(max_steps=config.MAX_STEPS)
    stats = _trajectory_stats(history)
    stats["duration_s"] = round(time.monotonic() - t0, 2)
    return history.final_result() or "", stats


async def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    pages = [sys.argv[1]] if len(sys.argv) > 1 else ["defect.html", "control.html"]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else config.K

    serve_pages()
    print("prewarming model ...", flush=True)
    prewarm_model()
    manifest = load_manifest(PAGES_DIR / "defects.json")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_path = config.RESULTS_DIR / f"audit_runs_{stamp}.jsonl"
    print(f"model={config.OLLAMA_MODEL}  k={k}  max_steps={config.MAX_STEPS}  "
          f"pages={pages}\nwriting {runs_path}", flush=True)

    with open(runs_path, "w", encoding="utf-8") as f:
        for page in pages:
            for r in range(k):
                print(f"[{page}] run {r + 1}/{k} ...", flush=True)
                try:
                    answer, stats = await run_once(page)
                    error = None
                except Exception as exc:  # local stack: report and move on
                    answer, stats, error = "", {}, str(exc)[:500]
                result = score_report(answer, manifest)
                record = {
                    "page": page,
                    "run": r,
                    "answer": answer,
                    "error": error,
                    "n_claims": result.n_claims,
                    "detected": result.detected,
                    "n_detected": result.n_detected,
                    "matched_lines": result.matched_lines,
                    "false_positive_lines": result.false_positive_lines,
                    "recall_by_class": {c: list(v) for c, v in
                                        recall_by_class(result, manifest).items()},
                    "n_steps": stats.get("n_steps"),
                    "duration_s": stats.get("duration_s"),
                    "agent_steps_duration_s": stats.get("agent_steps_duration_s"),
                    "steps": stats.get("steps", []),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                if page == "defect.html":
                    print(f"[{page}] run {r + 1} -> {result.n_detected}/12 detected, "
                          f"{len(result.false_positive_lines)} unmatched claims, "
                          f"{stats.get('n_steps')} steps, {stats.get('duration_s')}s",
                          flush=True)
                else:
                    print(f"[{page}] run {r + 1} -> {result.n_claims} claims "
                          f"(all are false positives on the control page), "
                          f"{stats.get('n_steps')} steps, {stats.get('duration_s')}s",
                          flush=True)

    print(f"\ndone -> {runs_path}")


if __name__ == "__main__":
    asyncio.run(main())
