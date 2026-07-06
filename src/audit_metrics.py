"""Aggregate audit runs into the report's numbers and figures.

Reads the newest results/audit_runs_*.jsonl (or a path given on the command
line) plus results/baseline_results.json, writes results/audit_summary.json,
and renders the report figures into ../figures/ (the report's figure dir):

  qa_detection_matrix.pdf  defect x run grid (agent) + baseline column
  qa_recall.pdf            per-class recall, agent (mean over k) vs baseline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "figures"
CLASS_ORDER = ("mechanical", "semantic", "visual")
CLASS_COLOR = {"mechanical": "#4878a8", "semantic": "#a85448", "visual": "#6a9a58"}


def load_runs(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(runs_path: Path) -> dict:
    manifest = json.load(open(ROOT / "pages" / "defects.json", encoding="utf-8"))
    defects = manifest["defects"]
    runs = load_runs(runs_path)
    agent_runs = [r for r in runs if r["page"] == "defect.html" and not r.get("error")]
    control_runs = [r for r in runs if r["page"] == "control.html" and not r.get("error")]
    baseline = json.load(open(RESULTS / "baseline_results.json", encoding="utf-8"))

    k = len(agent_runs)
    det_prob = {
        d["id"]: mean(1 if run["detected"][d["id"]] else 0 for run in agent_runs)
        for d in defects
    }
    per_class_recall = {}
    for cls in CLASS_ORDER:
        ids = [d["id"] for d in defects if d["class"] == cls]
        per_run = [mean(1 if run["detected"][i] else 0 for i in ids) for run in agent_runs]
        base = mean(1 if baseline["defect.html"]["detected"][i] else 0 for i in ids)
        per_class_recall[cls] = {
            "n_defects": len(ids),
            "agent_per_run": per_run,
            "agent_mean": mean(per_run),
            "baseline": base,
        }
    overall_per_run = [run["n_detected"] / len(defects) for run in agent_runs]
    precisions = [run["n_claims"] and
                  (run["n_claims"] - len(run["false_positive_lines"])) / run["n_claims"]
                  for run in agent_runs]

    summary = {
        "runs_file": runs_path.name,
        "k": k,
        "n_defects": len(defects),
        "detection_probability": det_prob,
        "per_class_recall": per_class_recall,
        "overall_recall_per_run": overall_per_run,
        "overall_recall_mean": mean(overall_per_run) if overall_per_run else 0.0,
        "precision_per_run": precisions,
        "precision_mean": mean(p for p in precisions if p is not None) if precisions else None,
        "control_claims_per_run": [r["n_claims"] for r in control_runs],
        "control_false_positive_lines": [r["false_positive_lines"] for r in control_runs],
        "baseline_overall_recall": baseline["defect.html"]["n_detected"] / len(defects),
        "baseline_control_claims": baseline["control.html"]["n_claims"],
        "per_run_detected": [run["detected"] for run in agent_runs],
        "n_errors": len([r for r in runs if r.get("error")]),
        "durations_s": [r.get("duration_s") for r in agent_runs + control_runs],
        "n_steps": [r.get("n_steps") for r in agent_runs + control_runs],
    }
    return summary


def fig_detection_matrix(summary: dict, defects: list[dict]) -> None:
    k = summary["k"]
    per_run = summary["per_run_detected"]
    baseline = json.load(open(RESULTS / "baseline_results.json", encoding="utf-8"))
    ids = [d["id"] for d in sorted(defects, key=lambda d: CLASS_ORDER.index(d["class"]))]
    byid = {d["id"]: d for d in defects}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for row, did in enumerate(ids):
        cls = byid[did]["class"]
        for col in range(k):
            hit = per_run[col][did]
            ax.add_patch(Rectangle((col, row), 0.92, 0.92,
                                   facecolor=CLASS_COLOR[cls] if hit else "#e6e6e6",
                                   edgecolor="white"))
        bhit = baseline["defect.html"]["detected"][did]
        ax.add_patch(Rectangle((k + 0.35, row), 0.92, 0.92,
                               facecolor="#333333" if bhit else "#e6e6e6",
                               edgecolor="white"))
    ax.set_xlim(-0.1, k + 1.45)
    ax.set_ylim(len(ids) - 0.02, -0.1)
    ax.set_xticks([c + 0.46 for c in range(k)] + [k + 0.81])
    ax.set_xticklabels([f"run {c + 1}" for c in range(k)] + ["baseline"])
    ax.set_yticks([r + 0.46 for r in range(len(ids))])
    ax.set_yticklabels([f"{did}  {byid[did]['class']}" for did in ids], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Seeded-defect detection: agent runs vs. rule-based baseline")
    legend = [Patch(facecolor=CLASS_COLOR[c], label=f"detected ({c})") for c in CLASS_ORDER]
    legend += [Patch(facecolor="#333333", label="detected (baseline)"),
               Patch(facecolor="#e6e6e6", label="missed")]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"qa_detection_matrix.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_recall(summary: dict) -> None:
    classes = list(CLASS_ORDER) + ["overall"]
    agent = [summary["per_class_recall"][c]["agent_mean"] for c in CLASS_ORDER]
    agent.append(summary["overall_recall_mean"])
    base = [summary["per_class_recall"][c]["baseline"] for c in CLASS_ORDER]
    base.append(summary["baseline_overall_recall"])

    x = range(len(classes))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([i - 0.19 for i in x], agent, width=0.36, label="agent (mean over runs)",
           color="#4878a8")
    ax.bar([i + 0.19 for i in x], base, width=0.36, label="rule-based baseline",
           color="#333333")
    # per-run agent recall as dots, showing run-to-run spread
    for i, c in enumerate(CLASS_ORDER):
        for v in summary["per_class_recall"][c]["agent_per_run"]:
            ax.plot(i - 0.19, v, "o", color="#1c3a55", markersize=3.5, alpha=0.7)
    for v in summary["overall_recall_per_run"]:
        ax.plot(len(classes) - 1 - 0.19, v, "o", color="#1c3a55", markersize=3.5, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall on seeded defects")
    ax.set_title("What each tester can see")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"qa_recall.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1:
        runs_path = Path(sys.argv[1])
    else:
        candidates = sorted(RESULTS.glob("audit_runs_*.jsonl"))
        if not candidates:
            sys.exit("no audit_runs_*.jsonl in results/")
        runs_path = candidates[-1]

    summary = summarize(runs_path)
    out = RESULTS / "audit_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest = json.load(open(ROOT / "pages" / "defects.json", encoding="utf-8"))
    FIGURES.mkdir(exist_ok=True)
    fig_detection_matrix(summary, manifest["defects"])
    fig_recall(summary)

    print(f"k={summary['k']}  overall recall mean={summary['overall_recall_mean']:.2f}  "
          f"baseline={summary['baseline_overall_recall']:.2f}")
    for cls in CLASS_ORDER:
        r = summary["per_class_recall"][cls]
        print(f"  {cls:10s} agent={r['agent_mean']:.2f} per-run={r['agent_per_run']} "
              f"baseline={r['baseline']:.2f}")
    print(f"  precision per run: {summary['precision_per_run']}")
    print(f"  control-page claims per run: {summary['control_claims_per_run']}")
    print(f"wrote {out}, figures -> {FIGURES}")


if __name__ == "__main__":
    main()
