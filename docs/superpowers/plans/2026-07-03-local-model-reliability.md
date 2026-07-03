# Local-Model Reliability Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the seminar experiment around qwen3:8b (Ollama) on four moderately complex tasks — fix the context-truncation bug, run k=3 × N=4 = 12 runs, regenerate figures, and rewrite the LaTeX report around the local results only.

**Architecture:** The existing harness (`config.py` → `src/run_experiment.py` → `src/judge.py` → `src/metrics.py`) is kept as-is; changes are confined to (1) an `OLLAMA_NUM_CTX` config knob passed into `ChatOllama`, (2) extra infra-error markers, (3) a replaced task slice, (4) a parameterized figure script, (5) the report rewrite.

**Tech Stack:** Python 3.13, Browser Use 0.13.1, Ollama (qwen3:8b), matplotlib, LaTeX (tectonic binary already in scratchpad).

## Global Constraints

- Working directory for all commands: `C:\Users\abdulhaa\Projects\Personal\Project Code\Project Code\browser-reliability` (the git repo). The report lives one level up.
- The repo has the user's own staged-but-uncommitted changes. **Every `git commit` MUST list explicit paths (`git commit -m "..." -- <path> [<path>...]`)** so staged work is never swept into a plan commit.
- Do not read or print `.env` (contains an API key). Configure runs via process environment variables; `load_dotenv()` does not override already-set process env vars.
- Agent temperature stays 1.0; judge temperature 0.0; τ=0.9; MAX_STEPS=25; K=3.
- Experiment env for every run: `PROVIDER=ollama`, `OLLAMA_MODEL=qwen3:8b`, `JUDGE_MODEL=qwen3:8b`, `OLLAMA_NUM_CTX=16384`.
- All report numbers must trace to files in `results/`. No invented results.

---

### Task 1: `OLLAMA_NUM_CTX` knob + Ollama infra markers

**Files:**
- Modify: `config.py` (Ollama section, ~line 29)
- Modify: `src/run_experiment.py` (`_INFRA_MARKERS` ~line 28; `_agent_llm()` ~line 76)
- Modify: `.env.example` (document new knobs)
- Test: `tests/test_config_ollama.py` (new)

**Interfaces:**
- Produces: `config.OLLAMA_NUM_CTX: int` (default 16384); `is_infra_error(exc)` returns True for Ollama connection failures. Consumed by Task 3/4 run commands.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_ollama.py`:

```python
"""Config knob + infra-marker coverage for the Ollama path. No API key needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.run_experiment import is_infra_error


def test_ollama_num_ctx_default():
    assert config.OLLAMA_NUM_CTX == 16384


def test_ollama_connection_errors_are_infra():
    assert is_infra_error(Exception("[WinError 10061] connection refused by target"))
    assert is_infra_error(Exception("httpx.ConnectError: All connection attempts failed"))
    assert is_infra_error(Exception("httpx.ReadTimeout: timed out"))


def test_wrong_answer_is_not_infra():
    assert not is_infra_error(Exception("ValidationError: action field required"))


if __name__ == "__main__":
    test_ollama_num_ctx_default()
    test_ollama_connection_errors_are_infra()
    test_wrong_answer_is_not_infra()
    print("ok")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe tests\test_config_ollama.py`
Expected: FAIL with `AttributeError: module 'config' has no attribute 'OLLAMA_NUM_CTX'`

- [ ] **Step 3: Implement**

In `config.py`, directly under `OLLAMA_HOST = ...` (line 30), add:

```python
# Context window for the local model. Ollama's default (~4k tokens) silently
# truncates Browser Use's 10-20k-token prompts, which makes the agent lose the
# task entirely (see results/runs_20260630T202223Z.jsonl: every answer chases a
# phantom task from the truncated system prompt's worked example).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
```

In `src/run_experiment.py`, extend `_INFRA_MARKERS` (line 28) to:

```python
_INFRA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "DeadlineExceeded",
                  "connection refused", "ConnectError", "ReadTimeout", "timed out",
                  "WinError 10061")
```

In `_agent_llm()`, change the Ollama branch to:

```python
    if config.PROVIDER == "ollama":
        from browser_use import ChatOllama
        return ChatOllama(model=config.AGENT_MODEL, host=config.OLLAMA_HOST,
                          ollama_options={"temperature": config.AGENT_TEMPERATURE,
                                          "think": False,
                                          "num_ctx": config.OLLAMA_NUM_CTX})
```

In `.env.example`, under the Gemini block, add:

```
# --- Ollama (local, free) ---
# PROVIDER=ollama
# OLLAMA_MODEL=qwen3:8b
# Context window: Ollama's ~4k default truncates Browser Use prompts.
# OLLAMA_NUM_CTX=16384
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe tests\test_config_ollama.py` → Expected: `ok`
Run: `.venv\Scripts\python.exe tests\test_metrics.py` → Expected: passes as before (metrics untouched)

- [ ] **Step 5: Commit (explicit paths only)**

```bash
git add tests/test_config_ollama.py config.py src/run_experiment.py .env.example
git commit -m "fix: set num_ctx for Ollama; treat local connection errors as infra" -- tests/test_config_ollama.py config.py src/run_experiment.py .env.example
```

---

### Task 2: Replace the task slice

**Files:**
- Modify: `tasks/webvoyager_slice.jsonl` (replace all 3 lines with 4)
- Create: `tasks/smoke_mdn.jsonl` (1-task file for the smoke gate)

**Interfaces:**
- Produces: task ids `wv-mdn-flex`, `wv-arxiv-attention`, `wv-wiki-eiffel-architect`, `wv-github-react-license` — consumed by Tasks 4–6 (figures use these exact ids as dict keys).

- [ ] **Step 1: Write `tasks/webvoyager_slice.jsonl`** (4 lines, exact content)

```jsonl
{"id": "wv-mdn-flex", "web_name": "MDN", "web": "https://developer.mozilla.org", "ques": "On developer.mozilla.org, find the documentation for the CSS 'flex' property and report the three sub-properties (longhands) it is a shorthand for.", "reference_answer": "flex-grow, flex-shrink, flex-basis"}
{"id": "wv-arxiv-attention", "web_name": "arXiv", "web": "https://arxiv.org", "ques": "Go to arxiv.org, find the paper titled 'Attention Is All You Need', and report its arXiv identifier (the arXiv:XXXX.XXXXX number).", "reference_answer": "arXiv:1706.03762"}
{"id": "wv-wiki-eiffel-architect", "web_name": "Wikipedia", "web": "https://en.wikipedia.org/wiki/Eiffel_Tower", "ques": "Starting from the English Wikipedia article on the Eiffel Tower, find the engineer Gustave Eiffel whose company designed the tower, open his article, and report his year of birth.", "reference_answer": "1832"}
{"id": "wv-github-react-license", "web_name": "GitHub", "web": "https://github.com", "ques": "Go to github.com, open the repository facebook/react, and report which open-source license it is released under.", "reference_answer": "MIT License"}
```

- [ ] **Step 2: Write `tasks/smoke_mdn.jsonl`** (first line of the slice only)

```jsonl
{"id": "wv-mdn-flex", "web_name": "MDN", "web": "https://developer.mozilla.org", "ques": "On developer.mozilla.org, find the documentation for the CSS 'flex' property and report the three sub-properties (longhands) it is a shorthand for.", "reference_answer": "flex-grow, flex-shrink, flex-basis"}
```

- [ ] **Step 3: Validate both files parse**

Run: `.venv\Scripts\python.exe -c "import json,pathlib; [json.loads(l) for f in ('tasks/webvoyager_slice.jsonl','tasks/smoke_mdn.jsonl') for l in pathlib.Path(f).read_text().splitlines() if l.strip()]; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tasks/webvoyager_slice.jsonl tasks/smoke_mdn.jsonl
git commit -m "feat: graded 4-task slice for local-model study + smoke task" -- tasks/webvoyager_slice.jsonl tasks/smoke_mdn.jsonl
```

---

### Task 3: Smoke gate (1 run, verifies the truncation fix)

**Files:** none created except a `results/runs_*.jsonl` from the run.

**Interfaces:**
- Consumes: Task 1 knob, Task 2 smoke file.
- Produces: go/no-go decision + per-run wall-clock estimate for Task 4.

- [ ] **Step 1: Confirm Ollama is serving**

Run: `ollama ps` (start the app/service if it errors), then `ollama list` must show `qwen3:8b`.

- [ ] **Step 2: Run one smoke run (PowerShell, foreground, time it)**

```powershell
$env:PROVIDER='ollama'; $env:OLLAMA_MODEL='qwen3:8b'; $env:JUDGE_MODEL='qwen3:8b'
$env:OLLAMA_NUM_CTX='16384'; $env:K='1'; $env:TASKS_PATH='tasks/smoke_mdn.jsonl'
Measure-Command { .venv\Scripts\python.exe -m src.run_experiment } | Select-Object TotalMinutes
```

- [ ] **Step 3: Gate check**

Open the newest `results/runs_*.jsonl`. PASS iff the answer text addresses the CSS flex task (mentions flex/longhands/sub-properties — correctness not required, only task adherence; no "CS.AI papers"/"papers.md" phantom).
- If phantom text appears: re-run Step 2 once with `$env:OLLAMA_NUM_CTX='24576'`. If it still fails, STOP — report to the user, do not run Task 4 (spec fallbacks: gemma4 or think mode on).
- Note TotalMinutes → total estimate for Task 4 is 12 × that.

- [ ] **Step 4: No commit** (results are gitignored; nothing else changed).

---

### Task 4: Full experiment (12 runs, background)

**Files:** produces `results/runs_<stamp>.jsonl` + `results/summary_<stamp>.json`.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: the summary JSON consumed by Task 5 figures and Task 6 report (fields: `p_hat` per task id, `p_bar`, `C_tau`, `delta`, `bands`, `per_task_variance`, `invalid_runs_excluded`).

- [ ] **Step 1: Launch in background**

```powershell
$env:PROVIDER='ollama'; $env:OLLAMA_MODEL='qwen3:8b'; $env:JUDGE_MODEL='qwen3:8b'
$env:OLLAMA_NUM_CTX='16384'; $env:K='3'; $env:TASKS_PATH='tasks/webvoyager_slice.jsonl'
.venv\Scripts\python.exe -m src.run_experiment
```

(run_in_background; expect ~12 × smoke-run minutes total. Progress lines print `[task] run i/3 -> SUCCESS/FAILURE`.)

- [ ] **Step 2: On completion, sanity-check the summary**

Newest `results/summary_*.json` must show: `n_tasks: 4`, 12 total outcomes minus `invalid_runs_excluded`, `provider: "ollama"`, `model: "qwen3:8b"`. Spot-read 2–3 answers in the runs file for task adherence.

- [ ] **Step 3: Report the outcome to the user** — per-task p̂, bands, p̄, C_τ, Δ — before starting Task 5. A flat result (all 0/3 or 3/3) is valid but the user decides whether to proceed or adjust.

---

### Task 5: Parameterize and regenerate figures

**Files:**
- Modify: `make_report_figures.py` (lines 26–34: `RUN_FILES`, `TASK_LABELS`)
- Produces: `../figures/run_grid.pdf/.png`, `../figures/browser_reliability.pdf/.png` regenerated from the new run.

**Interfaces:**
- Consumes: Task 4's `results/runs_<stamp>.jsonl` (exact stamp known at execution time).
- Produces: figure files referenced by the report as `figures/run_grid.pdf` and `figures/browser_reliability.pdf`.

- [ ] **Step 1: Point the script at the new run and fix labels**

Replace lines 26–34 of `make_report_figures.py` with (insert the real stamp):

```python
# Local-model run (qwen3:8b, num_ctx fix in place; see config.OLLAMA_NUM_CTX).
RUN_FILES = [
    "results/runs_<STAMP-FROM-TASK-4>.jsonl",
]
TASK_LABELS = {
    "wv-mdn-flex": "MDN\nflex longhands",
    "wv-arxiv-attention": "arXiv\npaper id",
    "wv-wiki-eiffel-architect": "Wikipedia\ntwo-hop birth year",
    "wv-github-react-license": "GitHub\nrepo license",
}
```

Also update the two panel titles in `figure_reliability()` (lines 114 and 123): they hard-code the pilot's outcome. Change to neutral titles:

```python
    axL.set_title("(a) Per-task reliability", fontsize=11)
```
```python
    axR.set_title("(b) Mean vs. reliably-solved", fontsize=11)
```

(The Δ-arrow branch for delta ≥ 0.005 already exists at lines 127–134 and will render the gap when it is non-zero.)

- [ ] **Step 2: Regenerate and eyeball**

Run: `.venv\Scripts\python.exe make_report_figures.py`
Expected: `wrote figures to ...\figures`. Open both PNGs (Read tool) — 4 rows in the grid, labels match tasks, bar colors reflect bands (green ≥ 0.9, orange flaky, red ≤ 0.1).

- [ ] **Step 3: Commit**

```bash
git add make_report_figures.py
git commit -m "feat: figures from local qwen3:8b run, 4-task slice" -- make_report_figures.py
```

---

### Task 6: Rewrite the report around the local study

**Files:**
- Modify: `../seminar_report (1).tex` (backup first)
- Modify: `../bib.bib` only if a citation is added/dropped (WebVoyager/WebArena/Mind2Web/OSWorld/BrowserArena all stay — they support the framing).

**Interfaces:**
- Consumes: Task 4 summary numbers, Task 5 figures, plus the phantom-task baseline (`results/summary_20260630T202223Z.json`: p̄=0%, k=3 on the 3-task slice).

- [ ] **Step 1: Backup**

```powershell
Copy-Item "..\seminar_report (1).tex" "..\seminar_report_haiku_backup.tex"
```

- [ ] **Step 2: Rewrite** — required content changes, all numbers from the logged files:

1. **Title/abstract**: reframe as a local-model reliability study. Suggested title: "Reliable or Just Lucky? Run-to-Run Reliability of a Local Browser Agent, and Why Its Apparent 0\% Was a Context-Truncation Artifact". Abstract arc: single-pass metrics hide causes → local agent scored 0% on every run → cause was Ollama's ~4k default context truncating the framework prompt (phantom-task evidence) → one-line `num_ctx` fix → measured reliability on 4 moderately complex tasks with per-task p̂/bands/Δ from Task 4's summary.
2. **Delete every mention** of Claude Haiku, Anthropic, the tool-call serialization artifact, `AgentOutput` repair, and the three trivial tasks (example.com heading, Eiffel completion year, oxygen atomic number). The repair switch disappears from Setup and Appendix. Grep the .tex for `Haiku`, `Anthropic`, `AgentOutput`, `example.com`, `oxygen`, `59.3`, `serializ` — all must be gone.
3. **Setup**: agent = Browser Use 0.13.1 + qwen3:8b via Ollama (temp 1.0, think off, `num_ctx` 16384, 25-step budget, headless); judge = qwen3:8b at temp 0; hardware = RTX A3000 Laptop 6 GB VRAM / 32 GB RAM; k=3, N=4, τ=0.9. State the k=3 granularity limitation (p̂ ∈ {0, ⅓, ⅔, 1}).
4. **The artifact section** (replaces tool-call repair): default-context truncation, the phantom "20 CS.AI papers / papers.md" answers quoting the system prompt's worked example, and the 0% baseline (`summary_20260630T202223Z.json`, 9 runs over 3 of the 4 tasks) as the "before" evidence; the `OLLAMA_NUM_CTX` knob as the fix.
5. **Results**: Table 1 = before (0%, 3-task slice) vs after (Task 4 numbers, 4-task slice) — caption must state the slices differ (before-run lacked the Wikipedia two-hop task) so it is context, not a controlled comparison. Table 2 = per-task x/3 with bands. Text reports p̄, C_τ, Δ, and what landed in the flaky band.
6. **Figures**: same two \includegraphics paths (regenerated in Task 5); rewrite both captions to describe the local run (k=3, 4 tasks, actual counts).
7. **Discussion/limitations**: single small local model; k=3 coarseness; live-web drift; judge is the same small model (weaker judge risk — spot-check by hand); the truncation artifact is the second known instance of interface faults masquerading as incapability (cite the general framing, keep OSWorld/BrowserArena related-work discussion).
8. **Appendix**: full env (Python 3.13, Windows 11, Ollama, qwen3:8b, num_ctx, dates of runs), exact 4 tasks with references, run accounting (12 runs + any excluded), per-task numbers.

- [ ] **Step 3: Compile and verify**

```powershell
$sp="C:\Users\abdulhaa\AppData\Local\Temp\claude\C--Users-abdulhaa-Projects-Personal-Project-Code-Project-Code\74c8fc4c-021d-41fa-854c-fda12c328088\scratchpad"
Copy-Item "..\seminar_report (1).tex","..\bib.bib" "$sp\build\" -Force
Copy-Item -Recurse -Force "..\figures" "$sp\build\figures"
& "$sp\tectonic\tectonic.exe" "$sp\build\seminar_report (1).tex"
```

Expected: PDF written, no errors. Extract text (pypdf in `$sp\pylibs`) and verify: no `Haiku`/`Anthropic`/`example.com` strings; table numbers match the summary JSON exactly.

- [ ] **Step 4: Deliver** — copy the built PDF next to the .tex as `seminar_report_local.pdf` (the original PDF may be file-locked by a viewer) and summarize all numbers to the user. The report files are outside the git repo; no commit.

---

## Self-Review

- **Spec coverage:** stack fix → Task 1; smoke gate → Task 3; slice → Task 2; protocol k=3/N=4 → Task 4; figures + label mismatch → Task 5; report rebuild incl. backup and no-Haiku sweep → Task 6; metrics tests still pass → Task 1 Step 4. ✓
- **Placeholders:** `<STAMP-FROM-TASK-4>` in Task 5 is data produced by Task 4 at runtime, called out explicitly — not an unknown. No TBDs. ✓
- **Type consistency:** task ids in Task 2 match `TASK_LABELS` keys in Task 5 and the appendix list in Task 6; `OLLAMA_NUM_CTX` name consistent across config, env, and run commands. ✓
