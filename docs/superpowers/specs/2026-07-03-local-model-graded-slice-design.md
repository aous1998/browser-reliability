# Local-Model Graded-Slice Experiment — Design

**Date:** 2026-07-03
**Goal:** Extend the seminar study with a harder, graded task slice run on a local
Ollama model, producing genuine run-to-run reliability findings (flaky-band
results) for the report, at zero API cost, within a ~3–4 hour runtime budget.

## Background and motivation

The submitted pilot (Claude Haiku 4.5, 3 easy tasks, k=9, two configurations)
showed that the stack's apparent 59.3% flakiness was a tool-call serialization
artifact; with a repair, 100% on every run. The report names two next steps:
harder tasks where genuine reliability gaps appear, and other backbones.

A previous local attempt (qwen3:8b via Ollama, harder tasks, k=3,
`results/summary_20260630T202223Z.json`) scored 0% on every run. Inspection of
`runs_20260630T202223Z.jsonl` shows every answer describes a phantom task
("collect metadata for 20 CS.AI papers", "papers.md") that appears nowhere in
the task slice — it matches the worked example in Browser Use's system prompt.
Root cause hypothesis: Ollama's default context window (~4k tokens) silently
truncates Browser Use's 10–20k-token prompts, so the model never sees the real
task. The harness's `ChatOllama` sets only `temperature` and `think`, not
`num_ctx`. This is a second model–framework interface artifact, parallel to the
pilot's serialization fault.

## Decisions made (with user)

- **Goal:** richer report findings (flaky-band results), not higher scores.
- **Backend:** local only (Ollama). Available hardware: RTX A3000 Laptop 6 GB
  VRAM, 32 GB RAM. Installed models: qwen3:8b, gemma4:latest.
- **Runtime budget:** a few hours max (~3–4 h).
- **Report:** extend the LaTeX report with the new results afterwards.
- **Approach:** A — fix the stack, smoke-gate, graded difficulty ladder,
  qwen3:8b only.

## Design

### 1. Stack fix (prerequisite)

- Add `OLLAMA_NUM_CTX` knob to `config.py` (env-driven, default **16384**).
- Pass `num_ctx` through `ollama_options` in `_agent_llm()` in
  `src/run_experiment.py`.
- If the smoke gate still shows truncation symptoms, raise to 24576.
- Nothing else changes: judge flow, metrics, infra retry, and the
  Anthropic-only tool-call repair stay as they are. The repair does not apply
  to the Ollama path.
- Add Ollama-specific infra markers (`connection refused`, read timeout) to
  `_INFRA_MARKERS` so local-server hiccups are excluded rather than counted as
  task failures.

### 2. Smoke gate

- 1 run of the easy anchor task with the fix in place.
- Pass criteria: the final answer addresses the asked task (no phantom task),
  and per-run wall clock is noted.
- If a run exceeds ~12 minutes, reduce k from 5 to 4 to stay inside budget.
- Only after the gate passes does the full experiment start.

### 3. Task slice (`tasks/webvoyager_slice.jsonl`, replaced)

| # | id | difficulty | task | reference |
|---|----|-----------|------|-----------|
| 1 | smoke-wiki-oxygen | easy (pilot anchor) | atomic number of oxygen on Wikipedia | 8 |
| 2 | wv-arxiv-attention | medium | arXiv ID of "Attention Is All You Need" | arXiv:1706.03762 |
| 3 | wv-mdn-flex | medium | three longhands of CSS `flex` on MDN | flex-grow, flex-shrink, flex-basis |
| 4 | wv-github-react-license | harder | license of facebook/react on GitHub | MIT License |

Tasks 2–4 reuse the existing harder tasks so the earlier 0% run is a directly
comparable "before" data point. All four have stable ground truth. The easy
anchor calibrates against the Haiku pilot.

### 4. Protocol

- Agent: qwen3:8b, temperature 1.0 (unchanged — sampling variance is the
  phenomenon), think off, headless, 25-step budget.
- Judge: qwen3:8b, temperature 0, reference answers as above.
- k=5, N=4 → 20 runs, executed in background with progress logging.
- Outputs land in `results/` via the existing pipeline (runs JSONL + summary
  JSON); metrics code untouched.

### 5. Report integration (after results)

- New experiment section in the LaTeX report: (i) the context-truncation
  artifact as a second interface fault (0% phantom-task runs → recovered by a
  one-line configuration fix), (ii) per-task reliability of a small local
  backbone on the graded slice, with bands and the overstatement gap.
- `make_report_figures.py` parameterized to generate a second run-grid and
  reliability panel from the new summary (this also fixes the current
  label/task-ID mismatch, where TASK_LABELS reference wv-* ids but RUN_FILES
  point at smoke-* logs).
- The Haiku pilot sections stay exactly as submitted.

## Error handling

- Ollama connection errors → infra-marker retry/exclusion path.
- A task scoring 0/5 or 5/5 is a valid result; the ladder makes an all-flat
  outcome across all four tasks unlikely.
- If the smoke gate fails even at 24576 ctx, stop and reassess (fallback
  options: gemma4 despite CPU offload, or think mode on) rather than burn the
  budget.

## Testing

- `tests/test_metrics.py` must still pass (metrics unchanged).
- Smoke gate acts as the integration test for the config fix.

## Success criteria

- Smoke gate: real task attempted, no phantom answer.
- Full run: 20 valid runs, summary produced, at least some per-task p̂ strictly
  between 0 and 1 (flaky band) — the informative outcome the report needs.
  (Not guaranteed; an honest flat result is still reportable.)
- Report: compiles, new section consistent with logged numbers.
