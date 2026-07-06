# Can a Local Browser Agent Do Web QA? — Seeded-Defect Study Design

**Date:** 2026-07-05
**Status:** Approved (conversation, 2026-07-05); replaces the reliability /
measurement-artifacts study as the report's headline. The repo is repurposed
in place; reliability-specific code and old results are deleted.

## Research question

> How reliably can an LLM-driven browser agent detect **seeded defects** in a
> static web page — with what precision, recall, and run-to-run consistency
> per defect class — compared with a classical rule-based checker?

- **RQ1 (capability):** Which defect classes (mechanical / semantic / visual)
  does the agent detect, with what precision and recall?
- **RQ2 (reliability):** Is detection stable across k identical audits, or is
  the agent a flaky tester? (Per-defect detection probability over runs.)
- **RQ3 (added value):** What does the agent catch that a rule-based checker
  structurally cannot, and vice versa?

"Replace human user testing" is deliberately NOT the claim: no human baseline
is measurable in this setting. The classical checker is the honest baseline.

## Why this design is sound

Ground truth is *planted*, not judged: we author the pages, so every defect is
known in advance and scoring is a deterministic pattern match — no LLM judge,
no live-web drift, no ambiguity about what a "success" is. A defect-free
control page measures false positives (hallucinated defects).

## Materials

`pages/` — a small fictitious business site ("Café Passau"), self-contained
HTML/CSS, served by `python -m http.server` on localhost:

- **`defect.html`** — seeded with 12 defects in 3 classes:
  - *Mechanical* (rule-detectable): D01 broken link (404 target), D02 dead
    image (missing file), D03 image missing alt text, D04 form with no
    action/handler, D05 empty link (href="#") labeled as a real page.
  - *Semantic* (only meaning-aware audit can catch): D06 button label
    contradicts its target ("View menu" → opening-hours anchor), D07 price
    contradiction (same cake €3.50 in menu, €4.80 in footer banner), D08
    garbled keyboard-mash paragraph, D09 copyright year 1875, D10 opening
    hours contradiction (header "open daily", footer "closed Sundays").
  - *Visual* (invisible in DOM text): D11 white-on-white paragraph, D12
    button covered by an overlapping element (z-index).
- **`control.html`** — same site, all 12 defects fixed. Anything the agent
  reports here is a false positive.
- **`defects.json`** — manifest: id, class, description, and match patterns
  (regex alternatives, all case-insensitive) used by the scorer.

## Systems under test

1. **Agent:** Browser Use 0.13.1 + qwen3:8b via Ollama (ThinklessChatOllama,
   num_ctx 16384, temperature 1.0, text-only DOM observation, headless,
   MAX_STEPS 10 — the audit needs navigation + scroll + report, not 25 steps).
   Task prompt: open the page, inspect it as a QA tester, list every defect
   found as numbered lines, then finish.
2. **Baseline:** `src/baseline_checker.py` — rule-based static checks over
   fetched HTML (requests + html.parser): broken internal links, missing
   image files, missing alt, form without action, empty hrefs. Emits findings
   in the same report format; scored by the same scorer.

## Protocol

- k = 5 audits of `defect.html`, k = 5 of `control.html` (agent), interleaved
  is unnecessary (no live drift). One baseline pass per page (deterministic).
- Scorer: split agent's final answer into lines/bullets; a seeded defect
  counts as **detected** iff any line matches its patterns; a line matching
  no defect counts as a **false positive** item. Manual review of pilot output
  validates the patterns before the full run.
- Per-run record (JSONL): page, run index, final answer, per-defect hits,
  false-positive lines, steps, durations (reuse existing trajectory logging).

## Metrics

- Per-defect detection rate over k runs (the reliability core, RQ2).
- Recall per class and overall = detected/seeded (per run and pooled).
- Precision per run = matched claims / all claims; false-positive count on
  the control page.
- Agent vs baseline: per-class recall table (RQ3). Expected shape: baseline
  100% on mechanical / 0% elsewhere; agent partial everywhere except visual
  ≈ 0% (observation-channel limit) — but these are hypotheses, the runs decide.

## Risks

- qwen3:8b may produce unparseable report formats → pilot gates the scorer.
- Agent may "fix" the format mid-list → scorer matches over the whole final
  answer, not only clean bullets.
- Steps may time out on long DOM → the pages are deliberately small (~1 short
  scroll), keeping prompts a fraction of WebVoyager-site size.

## Deliverables

1. Pages, manifest, runner, scorer, baseline, metrics, figures — in repo.
2. 10 measured agent runs + baseline results.
3. Fully rewritten `seminar_report_revised.tex` with only measured numbers.
4. Cleaned repo (old study's code/results removed), updated README, commits.
