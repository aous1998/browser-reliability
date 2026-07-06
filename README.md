# Seeded-Defect Web-QA Harness

Can an LLM browser agent do web QA? This harness measures how reliably a
**local** browser agent (Browser Use + qwen3:8b via Ollama) detects **seeded
defects** in a static web page, compared with a classical rule-based checker.

Ground truth is planted, not judged: we author the pages, so every defect is
known in advance and scoring is a deterministic pattern match — no LLM judge,
no live-web drift.

## Design

- **`pages/defect.html`** — a small café site seeded with 12 defects in 3
  classes:
  - *mechanical* (D01–D05): broken link, dead image, missing alt, form
    without action, placeholder `#` link — what a linkchecker finds.
  - *semantic* (D06–D10): mislabeled button, price contradiction, garbled
    paragraph, absurd copyright year, contradictory opening hours — require
    understanding meaning.
  - *visual* (D11–D12): white-on-white text, overlay-covered button —
    invisible in DOM text, require rendering.
- **`pages/control.html`** — the same site with every defect fixed; anything
  reported here is a false positive.
- **`pages/defects.json`** — the manifest: defect ids, classes, and the regex
  patterns the scorer uses.

## Research questions

1. **Capability:** which defect classes can the agent detect (precision/recall)?
2. **Reliability:** is detection stable over k identical audits?
3. **Added value:** what does the agent catch that rules cannot, and vice versa?

## Run

```bash
cd browser-reliability
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
# Ollama must be running with the model pulled: ollama pull qwen3:8b

python -m src.baseline_checker            # rule-based baseline, both pages
python -m src.run_audit                   # agent: K audits of both pages
python -m src.run_audit defect.html 1     # pilot: one run, one page
python -m src.audit_metrics               # summary + report figures

python tests/test_score.py                # scorer sanity tests (no model)
```

## Layout

```
config.py                 env-driven configuration (.env supported)
pages/                    seeded defect page, control page, defects.json
src/run_audit.py          serves pages/, runs the agent K times, logs + scores
src/score.py              deterministic report-to-manifest scorer
src/baseline_checker.py   classical rule-based checker (the baseline)
src/audit_metrics.py      aggregates runs -> summary json + figures
src/local_llm.py          Ollama chat wrapper with qwen3 thinking disabled
results/                  per-run JSONL + summaries (gitignored)
tests/                    scorer tests
```

## Knobs (env / `.env`)

`OLLAMA_MODEL` (qwen3:8b) · `OLLAMA_NUM_CTX` (16384 — Ollama's ~4k default
silently truncates Browser Use's prompts) · `OLLAMA_LLM_TIMEOUT_S` (300) ·
`K` audits per page · `MAX_STEPS` step budget per audit (15 recommended) ·
`TEMPERATURE` (1.0, non-zero by design: run-to-run variation is measured) ·
`HEADLESS`.
