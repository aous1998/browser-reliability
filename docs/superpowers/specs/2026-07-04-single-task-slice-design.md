# Single-Task Slice — Design

**Date:** 2026-07-04
**Goal:** Replace the experiment slice with exactly one fresh, mid-complexity
WebVoyager task whose ground truth is verified current, then run the k=3
experiment on it. Supersedes the 4-task slice of the 2026-07-03 design for the
experiment scope (protocol and stack fix are unchanged).

## Motivation

The 2026-07-03 attempts on real WebVoyager tasks failed for task-quality
reasons, not model reasons:

- `Google Search--3`: reCAPTCHA walls (plus one infra hiccup: runs recorded
  `ERROR: No module named 'browser_use'`).
- `GitHub--26`: 0/3 — the "Resolve merge conflicts" GitHub Skills course no
  longer exists under that name.
- `GitHub--3`: stale reference answer (48GB) — the live pricing page shows
  different storage numbers today.

Decision (with user): slice = **one** task, **k=3** (12 runs → 3 runs), task
chosen fresh from the WebVoyager dataset (643 records, downloaded from the
MinorJerry/WebVoyager repo), never used in any prior run here, on a
CAPTCHA-free, text-DOM site with stable ground truth.

## The task

`Cambridge Dictionary--21`, verbatim from the dataset:

> Search for the word "ephemeral" on Cambridge Dictionary and find its
> translation into Spanish.

- Reference answer: **efímero** — verified live on
  dictionary.cambridge.org (english-spanish entry) on 2026-07-04. The dataset
  ships no reference answers; we supply the verified one, as before.
- Why mid-complex: search + navigate to the English–Spanish translation
  section (two hops), not a single lookup.
- Why safe: Cambridge Dictionary is unused across all prior runs/smokes;
  no CAPTCHA for a real-Chrome UA (a plain fetch 403s, real browser passes);
  pure-text DOM suits the vision-disabled qwen3 agent; dictionary content is
  stable.

Candidates considered and rejected: `Cambridge Dictionary--17` (count of
meanings — page shows extra dictionary sections that could confuse the count),
`Wolfram Alpha--20` (timeless math, but result pods render partly as images —
risky with vision off).

## Plan

1. **Slice:** `tasks/webvoyager_slice.jsonl` becomes the single CD--21 line
   (dataset record + `reference_answer`).
2. **Gate:** one K=1 run (catches env problems like yesterday's missing-module
   infra failure before the real runs). Pass = agent reaches the site and
   addresses the task; correctness not required.
3. **Experiment:** K=3 on the same slice. Protocol unchanged: qwen3:8b via
   Ollama, `num_ctx` 16384, agent temp 1.0, judge temp 0, 25-step budget,
   headless.
4. **Figures:** `make_report_figures.py` → `RUN_FILES` = new k=3 stamp,
   `TASK_LABELS` = the single task id (grid becomes 1 row × 3 runs).
5. **Out of scope:** the LaTeX report rewrite — happens after the numbers
   exist, per the 2026-07-03 plan, with n=1 and k=3 granularity stated as
   limitations.

## Success criteria

- Gate run addresses the task (no phantom answer, no import errors).
- 3 valid k=3 runs, summary JSON produced.
- Any outcome 0/3–3/3 is valid; flaky (1–2 of 3) is the most informative.

## Timing expectation

~8 min for an early-success run, ~20 min for a full-budget run; total
expected wall clock ~35–80 min for gate + experiment.
