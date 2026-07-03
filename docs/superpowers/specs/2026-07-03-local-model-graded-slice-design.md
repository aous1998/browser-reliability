# Local-Model Reliability Study — Design (v2)

**Date:** 2026-07-03
**Goal:** Rebuild the seminar study around a local Ollama model on a slice of
moderately complex tasks. The Claude Haiku pilot and the three trivial tasks
(example.com heading, Eiffel year, oxygen number) are removed from the report
entirely; the study, its results, and its figures come exclusively from local
runs. Zero API cost, ~3–4 hour runtime budget.

## Background and motivation

A previous local attempt (qwen3:8b via Ollama, harder tasks, k=3,
`results/summary_20260630T202223Z.json`) scored 0% on every run. Inspection of
`runs_20260630T202223Z.jsonl` shows every answer describes a phantom task
("collect metadata for 20 CS.AI papers", "papers.md") that appears nowhere in
the task slice — it matches the worked example in Browser Use's system prompt.
Root cause hypothesis: Ollama's default context window (~4k tokens) silently
truncates Browser Use's 10–20k-token prompts, so the model never sees the real
task. The harness's `ChatOllama` sets only `temperature` and `think`, not
`num_ctx`.

This gives the rebuilt report its narrative arc, same thesis as before but now
entirely local: a single-pass success rate (0%) hid that the failure was a
model–framework interface artifact (context truncation), not agent incapacity;
fixing the interface reveals the model's true, run-to-run variable reliability
on moderately complex tasks.

## Decisions made (with user)

- **Goal:** richer report findings (flaky-band results), not higher scores.
- **Backend:** local only (Ollama), qwen3:8b. Hardware: RTX A3000 Laptop 6 GB
  VRAM, 32 GB RAM. gemma4:latest available as fallback only.
- **Runtime budget:** a few hours max (~3–4 h).
- **Report:** full rebuild around local results — Haiku pilot and trivial
  tasks deleted, not kept as a comparison. (Report rewrite happens only after
  the runs finish; until then the current report stays untouched, and a backup
  copy of the Haiku-based .tex is kept before any rewrite.)
- **Tasks:** "complex but not too complex" — single-site, one-to-two-hop
  lookups with stable ground truth, calibrated so a small local model lands in
  the informative flaky band rather than at 0% or 100%.

## Design

### 1. Stack fix (prerequisite)

- Add `OLLAMA_NUM_CTX` knob to `config.py` (env-driven, default **16384**).
- Pass `num_ctx` through `ollama_options` in `_agent_llm()` in
  `src/run_experiment.py`.
- If the smoke gate still shows truncation symptoms, raise to 24576.
- Add Ollama-specific infra markers (`connection refused`, read timeout) to
  `_INFRA_MARKERS` so local-server hiccups are excluded rather than counted as
  task failures.
- Judge flow, metrics, and infra retry otherwise unchanged. The
  Anthropic-only tool-call repair is out of the Ollama path and out of the
  rebuilt report.

### 2. Smoke gate

- 1 run of `wv-mdn-flex` (the most self-contained lookup) with the fix in
  place.
- Pass criteria: the final answer addresses the asked task (no phantom task);
  per-run wall clock noted.
- If a run exceeds ~12 minutes, reduce k from 5 to 4 to stay inside budget.
- Only after the gate passes does the full experiment start.

### 3. Task slice (`tasks/webvoyager_slice.jsonl`, replaced)

Four moderately complex tasks, stable ground truth, no trivial anchors:

| # | id | difficulty | task | reference |
|---|----|-----------|------|-----------|
| 1 | wv-mdn-flex | medium | three longhands of CSS `flex` on MDN | flex-grow, flex-shrink, flex-basis |
| 2 | wv-arxiv-attention | medium | arXiv ID of "Attention Is All You Need" | arXiv:1706.03762 |
| 3 | wv-wiki-eiffel-architect | medium (two-hop) | birth year of the engineer whose company designed the Eiffel Tower (navigate Wikipedia: Eiffel Tower → Gustave Eiffel) | 1832 |
| 4 | wv-github-react-license | medium-hard | license of facebook/react on GitHub | MIT License |

Tasks 1, 2, and 4 reuse the existing harder tasks, so the earlier 0% run is a
directly comparable "before" data point for the truncation fix. Task 3 adds a
genuine two-hop navigation without being fragile.

### 4. Protocol

- Agent: qwen3:8b, temperature 1.0 (sampling variance is the phenomenon),
  think off, headless, 25-step budget.
- Judge: qwen3:8b, temperature 0, reference answers as above.
- k=5, N=4 → 20 runs, executed in background with progress logging.
- Outputs land in `results/` via the existing pipeline; metrics code
  untouched.

### 5. Report rebuild (after results)

- Back up the current Haiku-based `seminar_report (1).tex` before editing.
- Rewrite the report as a local-model study: title/abstract reframed; the
  serialization-artifact narrative replaced by the context-truncation
  artifact (0% phantom-task baseline → configuration fix → measured
  reliability); results, tables, and both figures regenerated from the new
  runs via a parameterized `make_report_figures.py` (also fixing the current
  label/task-ID mismatch).
- All numbers in the rewritten report must trace to logged results files; no
  claims about runs that were not made.

## Error handling

- Ollama connection errors → infra-marker retry/exclusion path.
- A task scoring 0/5 or 5/5 is a valid result; the difficulty mix makes an
  all-flat outcome across all four tasks unlikely.
- If the smoke gate fails even at 24576 ctx, stop and reassess (fallbacks:
  gemma4 despite CPU offload, or think mode on) rather than burn the budget.

## Testing

- `tests/test_metrics.py` must still pass (metrics unchanged).
- Smoke gate acts as the integration test for the config fix.

## Success criteria

- Smoke gate: real task attempted, no phantom answer.
- Full run: 20 valid runs, summary produced, ideally some per-task p̂ strictly
  between 0 and 1 (not guaranteed; an honest flat result is still reportable).
- Report: compiles, all content consistent with the new logged numbers, no
  remaining references to Claude Haiku or the trivial task slice.
