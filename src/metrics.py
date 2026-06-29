"""Reliability metrics from the report (Measurement Protocol section).

Given binary run outcomes per task, compute:
  - p_hat_t      : per-task empirical success probability     (Eq. phat)
  - p_bar        : mean success rate (what a leaderboard reports)
  - C_tau        : reliably-solved rate (fraction with p_hat >= tau)
  - Delta        : overstatement gap = p_bar - C_tau           (Eq. gap)
  - bands        : reliably-solved / flaky / reliably-failed counts
  - variance     : mean per-task variance of the outcomes
"""
from __future__ import annotations

from statistics import mean, pvariance


def p_hat(outcomes: list[int]) -> float:
    """Empirical success probability for one task over its repeated runs."""
    if not outcomes:
        return 0.0
    return sum(outcomes) / len(outcomes)


def band(p: float, tau: float) -> str:
    """Sort a per-task probability into reliable / flaky / reliably-failed."""
    if p >= tau:
        return "reliably_solved"
    if p <= 1 - tau:
        return "reliably_failed"
    return "flaky"


def summarize(per_task_outcomes: dict[str, list[int]], tau: float) -> dict:
    """Turn {task_id: [s_1, ..., s_k]} into the report's headline numbers."""
    phats = {tid: p_hat(o) for tid, o in per_task_outcomes.items()}
    if not phats:
        raise ValueError("no tasks to summarize")

    p_bar = mean(phats.values())
    c_tau = mean(1.0 if p >= tau else 0.0 for p in phats.values())
    delta = p_bar - c_tau

    bands = {"reliably_solved": 0, "flaky": 0, "reliably_failed": 0}
    for p in phats.values():
        bands[band(p, tau)] += 1

    # Mean within-task variance of the binary outcomes (spread across runs).
    per_task_var = {
        tid: pvariance(o) if len(o) > 1 else 0.0
        for tid, o in per_task_outcomes.items()
    }
    mean_var = mean(per_task_var.values())

    return {
        "tau": tau,
        "n_tasks": len(phats),
        "p_hat": phats,
        "p_bar": p_bar,
        "C_tau": c_tau,
        "delta": delta,
        "bands": bands,
        "per_task_variance": per_task_var,
        "mean_per_task_variance": mean_var,
    }
