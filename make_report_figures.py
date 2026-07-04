"""Generate the publication-quality figures used in the seminar report.

Loads the local qwen3:8b run logs and writes two figures to ../figures/:
  * browser_reliability.pdf  -- per-task p_hat (with 95% Wilson CIs, tau and
                                mean lines) next to the mean-vs-reliable gap.
  * run_grid.pdf             -- the per-run success/failure grid, showing the
                                same task succeeding on some runs and failing
                                on others.

Run from the browser-reliability/ directory:  python make_report_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src import metrics

ROOT = Path(__file__).parent
FIG_DIR = ROOT.parent / "figures"
# Local-model run: qwen3:8b via Ollama, k=3, single-task slice
# (Cambridge Dictionary--21; num_ctx fix in place, see config.OLLAMA_NUM_CTX).
RUN_FILES = [
    "results/runs_20260704T093033Z.jsonl",
]
TASK_LABELS = {
    "Cambridge Dictionary--21": "Cambridge Dictionary\nSpanish translation",
}
TAU = 0.9
GREEN, ORANGE, RED, GREY = "#2E8B57", "#E8A33D", "#C0392B", "#BDC3C7"

# A clean, report-matching look (Computer-Modern-ish serif, light spines).
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


def load_pooled() -> dict[str, list[int]]:
    pooled: dict[str, list[int]] = {}
    for rf in RUN_FILES:
        for line in (ROOT / rf).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d["success"] is None:
                continue
            pooled.setdefault(d["task_id"], []).append(int(d["success"]))
    return pooled


def wilson_ci(k_succ: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k_succ / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def band_color(p: float) -> str:
    if p >= TAU:
        return GREEN
    if p <= 1 - TAU:
        return RED
    return ORANGE


def figure_reliability(pooled: dict[str, list[int]]) -> None:
    order = list(TASK_LABELS)
    phats = [metrics.p_hat(pooled[t]) for t in order]
    ns = [len(pooled[t]) for t in order]
    succ = [sum(pooled[t]) for t in order]
    s = metrics.summarize(pooled, TAU)
    p_bar, c_tau, delta = s["p_bar"], s["C_tau"], s["delta"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    # --- Left: per-task p_hat with Wilson CIs, tau + mean lines ---------------
    x = range(len(order))
    colors = [band_color(p) for p in phats]
    lo = [p - wilson_ci(succ[i], ns[i])[0] for i, p in enumerate(phats)]
    hi = [wilson_ci(succ[i], ns[i])[1] - p for i, p in enumerate(phats)]
    axL.bar(x, phats, width=0.62, color=colors, edgecolor="black", linewidth=0.7,
            zorder=3)
    axL.errorbar(x, phats, yerr=[lo, hi], fmt="none", ecolor="black",
                 elinewidth=1.1, capsize=4, zorder=4)
    axL.axhline(TAU, color=GREEN, ls=":", lw=1.6,
                label=fr"reliability threshold $\tau={TAU}$", zorder=2)
    axL.axhline(p_bar, color=RED, ls="--", lw=1.6,
                label=fr"mean $\bar p={p_bar:.2f}$", zorder=2)
    for i, p in enumerate(phats):
        axL.text(i, p + hi[i] + 0.03, f"{succ[i]}/{ns[i]}", ha="center",
                 va="bottom", fontsize=9)
    axL.set_xticks(list(x))
    axL.set_xticklabels([TASK_LABELS[t] for t in order], fontsize=9.5)
    axL.set_ylim(0, 1.08)
    axL.set_ylabel(r"per-task success probability $\hat p_t$")
    axL.set_title("(a) Per-task reliability", fontsize=11)
    axL.legend(loc="lower right", fontsize=8.2, framealpha=0.9)

    # --- Right: mean vs reliably-solved -- the gap has closed -----------------
    bars = axR.bar([0, 1], [p_bar, c_tau], width=0.5,
                   color=[GREEN, GREEN], edgecolor="black", linewidth=0.7, zorder=3)
    axR.set_xticks([0, 1])
    axR.set_xticklabels([r"mean $\bar p$", r"reliable $C_\tau$"], fontsize=10)
    axR.set_ylim(0, 1.08)
    axR.set_title("(b) Mean vs. reliably-solved", fontsize=11)
    for b, v in zip(bars, [p_bar, c_tau]):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.1f}%",
                 ha="center", va="bottom", fontsize=9.5)
    if delta < 0.005:
        axR.text(0.5, 0.5, fr"$\Delta = 0$" + "\n(no gap)", fontsize=10,
                 va="center", ha="center", color=GREEN)
    else:
        axR.add_patch(FancyArrowPatch((0, p_bar), (0, c_tau),
                      arrowstyle="<->", mutation_scale=12, lw=1.3, color="black"))
        axR.text(0.06, p_bar / 2, fr"$\Delta = {delta*100:.1f}$ pts", fontsize=10,
                 va="center", ha="left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "browser_reliability.pdf")
    fig.savefig(FIG_DIR / "browser_reliability.png")
    plt.close(fig)


def figure_run_grid(pooled: dict[str, list[int]]) -> None:
    order = list(TASK_LABELS)
    k = max(len(pooled[t]) for t in order)
    fig, ax = plt.subplots(figsize=(7.0, 2.4))

    for row, t in enumerate(order):
        outcomes = pooled[t]
        for col, o in enumerate(outcomes):
            ax.add_patch(plt.Rectangle((col, row), 0.92, 0.92,
                         facecolor=GREEN if o else GREY,
                         edgecolor="white", linewidth=1.5))
        p = metrics.p_hat(outcomes)
        ax.text(k + 0.25, row + 0.46, fr"$\hat p_t = {p:.2f}$", va="center",
                fontsize=10)

    ax.set_xlim(0, k + 1.6)
    ax.set_ylim(0, len(order))
    ax.set_xticks([c + 0.46 for c in range(k)])
    ax.set_xticklabels(range(1, k + 1), fontsize=9)
    ax.set_yticks([r + 0.46 for r in range(len(order))])
    ax.set_yticklabels([TASK_LABELS[t].replace("\n", " ") for t in order],
                       fontsize=9.5)
    ax.set_xlabel("run index")
    ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # legend
    ax.add_patch(plt.Rectangle((0, len(order) + 0.25), 0.92, 0.6,
                 facecolor=GREEN, edgecolor="white"))
    ax.text(1.1, len(order) + 0.55, "success", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((3.2, len(order) + 0.25), 0.92, 0.6,
                 facecolor=GREY, edgecolor="white"))
    ax.text(4.3, len(order) + 0.55, "failure", va="center", fontsize=9)
    ax.set_ylim(len(order) + 1.1, 0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "run_grid.pdf")
    fig.savefig(FIG_DIR / "run_grid.png")
    plt.close(fig)


def main() -> None:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    FIG_DIR.mkdir(exist_ok=True)
    pooled = load_pooled()
    figure_reliability(pooled)
    figure_run_grid(pooled)
    print(f"wrote figures to {FIG_DIR}")
    for f in ("browser_reliability", "run_grid"):
        print("  -", (FIG_DIR / f).with_suffix(".pdf").name)


if __name__ == "__main__":
    main()
