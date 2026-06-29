"""Sanity checks for the reliability metrics (run without any API key)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import metrics


def test_p_hat():
    assert metrics.p_hat([1, 1, 1]) == 1.0
    assert metrics.p_hat([1, 0, 0, 0]) == 0.25
    assert metrics.p_hat([]) == 0.0


def test_band():
    assert metrics.band(0.95, 0.9) == "reliably_solved"
    assert metrics.band(0.5, 0.9) == "flaky"
    assert metrics.band(0.05, 0.9) == "reliably_failed"


def test_summarize_gap():
    # 3 tasks: one always solved, one flaky (3/5), one always failed.
    outcomes = {
        "a": [1, 1, 1, 1, 1],
        "b": [1, 1, 1, 0, 0],
        "c": [0, 0, 0, 0, 0],
    }
    s = metrics.summarize(outcomes, tau=0.9)
    assert abs(s["p_bar"] - (1.0 + 0.6 + 0.0) / 3) < 1e-9
    assert abs(s["C_tau"] - 1 / 3) < 1e-9            # only "a" is reliable
    assert abs(s["delta"] - (s["p_bar"] - s["C_tau"])) < 1e-9
    assert s["bands"] == {"reliably_solved": 1, "flaky": 1, "reliably_failed": 1}


if __name__ == "__main__":
    test_p_hat()
    test_band()
    test_summarize_gap()
    print("all metric tests passed")
