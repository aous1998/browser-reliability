# Browser-Agent Reliability Harness

The browser-agent half of the seminar study *"Reliable or Just Lucky?"* — it
measures the **run-to-run reliability** of a DOM-based browser agent by running
each task many times and comparing the leaderboard-style mean success rate
against the rate of tasks solved *reliably*.

It uses [Browser Use](https://github.com/browser-use/browser-use) driven by
Gemini, on a small slice of WebVoyager-style tasks. (The OSWorld desktop-agent
half from the report is intentionally **not** built here.)

## What it computes

For each task `t`, run the agent `k` times → binary outcomes `s_{t,r}`.

| symbol | meaning |
|--------|---------|
| `p_hat_t` | per-task success probability = mean of its `k` outcomes |
| `p_bar`   | mean success rate (the single number a leaderboard reports) |
| `C_tau`   | reliably-solved rate = fraction of tasks with `p_hat_t >= tau` |
| `Delta`   | overstatement gap = `p_bar - C_tau` |

Tasks are also bucketed into **reliably solved / flaky / reliably failed**.

## Setup

```bash
cd browser-reliability
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
playwright install chromium                              # browser for Browser Use
cp .env.example .env                                     # then add your GOOGLE_API_KEY
```

> Browser Use targets Python 3.11–3.13. If install fails on a newer interpreter,
> create the venv with a 3.12/3.13 Python.

## Run

```bash
python -m src.run_experiment          # runs all tasks k times, writes results/
python -m src.plot_results            # histogram of p_hat from the latest summary
```

Verify the metrics logic without any API key:

```bash
python tests/test_metrics.py
```

## Layout

```
config.py                  env-driven configuration
tasks/webvoyager_slice.jsonl  task slice (id, url, question, reference answer)
src/run_experiment.py      k-times runner + per-run model judge
src/judge.py               WebVoyager-style SUCCESS/FAILURE judge
src/metrics.py             p_hat, p_bar, C_tau, Delta, bands, variance
src/plot_results.py        per-task probability histogram
results/                   raw runs + summary json/png (gitignored)
```

## Knobs (env / `.env`)

`K` runs per task · `TAU` reliability threshold (0.9) · `MAX_STEPS` step budget ·
`GEMINI_MODEL` · `HEADLESS`.
