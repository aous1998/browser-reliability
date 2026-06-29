"""Plot the per-task success-probability histogram (Figure 'subresults').

Usage:
    python -m src.plot_results results/summary_<stamp>.json
If no path is given, the most recent summary in the results dir is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import config


def latest_summary() -> Path:
    summaries = sorted(config.RESULTS_DIR.glob("summary_*.json"))
    if not summaries:
        sys.exit("No summary_*.json found in results/. Run the experiment first.")
    return summaries[-1]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_summary()
    summary = json.loads(path.read_text(encoding="utf-8"))
    phats = list(summary["p_hat"].values())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(phats, bins=[i / 10 for i in range(11)], edgecolor="black")
    ax.axvline(summary["p_bar"], color="red", linestyle="--",
               label=f"mean $\\bar p$ = {summary['p_bar']:.2f}")
    ax.set_xlabel(r"per-task success probability $\hat p_t$")
    ax.set_ylabel("number of tasks")
    ax.set_title("Browser agent (DOM)")
    ax.legend()
    fig.tight_layout()

    out = path.with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
