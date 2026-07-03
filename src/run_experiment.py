"""Run the browser-agent reliability experiment.

For each WebVoyager-style task we run the Browser Use agent K independent times,
judge each run as success/failure, and write the raw outcomes plus the report's
reliability metrics to the results directory.

Usage:
    python -m src.run_experiment
Configuration is read from config.py / environment (see .env.example).
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python src/run_experiment.py` as well as `-m src.run_experiment`.
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src import metrics
from src.judge import judge

# Markers of a transient infrastructure failure (API down / rate limited), as
# opposed to the agent genuinely failing the task.
_INFRA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "DeadlineExceeded",
                  "connection refused", "ConnectError", "ReadTimeout", "timed out",
                  "WinError 10061")
# Markers of an exhausted/billing-blocked account. These are not transient and not
# agent failures: retrying is futile and counting them as task failures corrupts the
# metrics, so we abort the whole experiment when one appears.
_BILLING_MARKERS = ("credit balance is too low", "billing", "insufficient_quota",
                    "exceeded your current quota")


class CreditExhausted(RuntimeError):
    """The model API account is out of credit; abort rather than log bogus failures."""


def is_billing_error(exc: Exception) -> bool:
    return any(m.lower() in str(exc).lower() for m in _BILLING_MARKERS)


def is_infra_error(exc: Exception) -> bool:
    return any(m in str(exc) for m in _INFRA_MARKERS)


def _browser_use_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("browser-use")
    except PackageNotFoundError:
        return "unknown"


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def _agent_llm():
    """Build the Browser Use chat model for the configured provider."""
    if config.PROVIDER == "anthropic":
        if config.REPAIR_TOOL_CALLS:
            from src.repair_llm import RepairingChatAnthropic
            return RepairingChatAnthropic(model=config.AGENT_MODEL, api_key=config.API_KEY,
                                          temperature=config.AGENT_TEMPERATURE)
        from browser_use import ChatAnthropic
        return ChatAnthropic(model=config.AGENT_MODEL, api_key=config.API_KEY,
                             temperature=config.AGENT_TEMPERATURE)
    if config.PROVIDER == "ollama":
        from browser_use import ChatOllama
        return ChatOllama(model=config.AGENT_MODEL, host=config.OLLAMA_HOST,
                          ollama_options={"temperature": config.AGENT_TEMPERATURE,
                                          "think": False,
                                          "num_ctx": config.OLLAMA_NUM_CTX})
    from browser_use import ChatGoogle
    return ChatGoogle(model=config.AGENT_MODEL, api_key=config.API_KEY,
                      temperature=config.AGENT_TEMPERATURE)


async def run_once(task: dict) -> str:
    """Run the Browser Use agent on one task; return its final answer text."""
    from browser_use import Agent, BrowserProfile

    prompt = (
        f"Open {task['web']} and complete this task: {task['ques']} "
        f"When done, state the answer clearly and concisely."
    )
    agent = Agent(
        task=prompt,
        llm=_agent_llm(),
        browser_profile=BrowserProfile(headless=config.HEADLESS),
        # Local Ollama models are text-only; sending screenshots makes every
        # LLM call fail with 400 "does not support multimodal requests".
        # DOM-only observation is also what the study measures.
        use_vision=(config.PROVIDER != "ollama"),
        llm_timeout=(config.OLLAMA_LLM_TIMEOUT_S if config.PROVIDER == "ollama" else None),
        step_timeout=(config.OLLAMA_LLM_TIMEOUT_S + 120 if config.PROVIDER == "ollama" else 180),
    )
    history = await agent.run(max_steps=config.MAX_STEPS)
    return history.final_result() or ""


async def attempt_run(task: dict) -> tuple[str, int | None]:
    """Run + judge one attempt, retrying transient infra errors.

    Returns (answer, outcome) where outcome is 1 (success), 0 (genuine task
    failure), or None (invalid: persistent infra/quota error -> excluded).
    """
    for attempt in range(config.INFRA_RETRIES + 1):
        try:
            answer = await run_once(task)
            outcome = judge(task["ques"], task.get("reference_answer", ""), answer)
            return answer, outcome
        except Exception as exc:
            if is_billing_error(exc):
                raise CreditExhausted(str(exc)) from exc
            if is_infra_error(exc) and attempt < config.INFRA_RETRIES:
                print(f"    infra error, retry in {config.INFRA_BACKOFF_S:.0f}s "
                      f"({attempt + 1}/{config.INFRA_RETRIES})", flush=True)
                await asyncio.sleep(config.INFRA_BACKOFF_S)
                continue
            if is_infra_error(exc):
                return f"INVALID (infra): {exc}", None
            return f"ERROR: {exc}", 0  # genuine failure (agent crashed)


async def main() -> None:
    # The project path contains non-Latin characters; force UTF-8 stdout so the
    # final summary print does not crash on the Windows cp1252 console.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not config.API_KEY:
        key_name = "ANTHROPIC_API_KEY" if config.PROVIDER == "anthropic" else "GOOGLE_API_KEY"
        sys.exit(f"{key_name} is not set. Copy .env.example to .env and add your key.")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(config.TASKS_PATH)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs_path = config.RESULTS_DIR / f"runs_{stamp}.jsonl"

    per_task_outcomes: dict[str, list[int]] = {}
    invalid_runs = 0

    try:
      with open(runs_path, "w", encoding="utf-8") as runs_file:
        for task in tasks:
            tid = task["id"]
            per_task_outcomes[tid] = []
            for r in range(config.K):
                print(f"[{tid}] run {r + 1}/{config.K} ...", flush=True)
                answer, outcome = await attempt_run(task)  # outcome: 1, 0, or None
                if outcome is None:
                    invalid_runs += 1
                    label = "INVALID (infra/quota, excluded)"
                else:
                    per_task_outcomes[tid].append(outcome)
                    label = "SUCCESS" if outcome else "FAILURE"
                record = {
                    "task_id": tid,
                    "run": r,
                    "success": outcome,  # null = excluded from metrics
                    "answer": answer,
                }
                runs_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                runs_file.flush()
                print(f"[{tid}] run {r + 1} -> {label}")
    except CreditExhausted as exc:
        sys.exit(
            f"\nAborted: the Anthropic API account is out of credit ({exc}).\n"
            "Partial raw runs were written but no summary was produced, so the "
            "credit error is never recorded as a task failure. Top up billing at "
            "https://console.anthropic.com/settings/billing and re-run."
        )

    # Drop tasks that produced no valid runs at all (cannot estimate p_hat).
    valid_tasks = {t: o for t, o in per_task_outcomes.items() if o}
    dropped = [t for t, o in per_task_outcomes.items() if not o]
    if not valid_tasks:
        sys.exit("No valid runs were recorded (all infra/quota errors). "
                 "Check your API quota/billing, then re-run.")

    summary = metrics.summarize(valid_tasks, config.TAU)
    # Reproducibility block (report, Appendix A).
    summary["provider"] = config.PROVIDER
    summary["model"] = config.AGENT_MODEL
    summary["judge_model"] = config.JUDGE_MODEL
    summary["k"] = config.K
    summary["agent_temperature"] = config.AGENT_TEMPERATURE
    summary["judge_temperature"] = config.JUDGE_TEMPERATURE
    summary["max_steps"] = config.MAX_STEPS
    summary["browser_use_version"] = _browser_use_version()
    summary["repair_tool_calls"] = config.REPAIR_TOOL_CALLS
    if config.REPAIR_TOOL_CALLS and config.PROVIDER == "anthropic":
        from src.repair_llm import STATS as repair_stats
        summary["tool_call_repair"] = dict(repair_stats)
    summary["invalid_runs_excluded"] = invalid_runs
    summary["tasks_dropped_no_valid_runs"] = dropped
    summary["timestamp"] = stamp

    summary_path = config.RESULTS_DIR / f"summary_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Reliability summary (browser agent, DOM) ===")
    print(f"  tasks={summary['n_tasks']}  k={config.K}  tau={config.TAU}"
          f"  invalid_runs_excluded={invalid_runs}")
    print(f"  mean success rate  p_bar = {summary['p_bar'] * 100:.1f}%")
    print(f"  reliably-solved    C_tau = {summary['C_tau'] * 100:.1f}%")
    print(f"  overstatement gap  Delta = {summary['delta'] * 100:.1f} pts")
    print(f"  bands = {summary['bands']}")
    print(f"\nraw runs -> {runs_path}\nsummary  -> {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
