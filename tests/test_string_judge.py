"""Deterministic string-judge coverage. No API key or network needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.judge import string_judge


def test_exact_reference_present():
    assert string_judge("efímero", "The Spanish translation of 'ephemeral' is 'efímero'.") == 1


def test_misjudged_run_from_20260704_is_success():
    # qwen3:8b judged this real answer FAILURE although it contains the
    # reference string verbatim (runs_20260704T093033Z.jsonl, run index 1).
    assert string_judge("efímero", "The Spanish translation of 'ephemeral' is 'efímero' and 'pasajero'.") == 1


def test_accent_insensitive():
    assert string_judge("efímero", "the translation is efimero") == 1


def test_case_insensitive():
    assert string_judge("MIT License", "It is released under the mit license.") == 1


def test_wrong_answer_fails():
    assert string_judge("efímero", "The Spanish translation is 'pasajero'.") == 0


def test_empty_answer_fails():
    assert string_judge("efímero", "") == 0


def test_multi_part_all_present_any_order():
    parts = ["flex-grow", "flex-shrink", "flex-basis"]
    ans = "The longhands are flex-basis, flex-grow and flex-shrink."
    assert string_judge("flex-grow, flex-shrink, flex-basis", ans, parts) == 1


def test_multi_part_one_missing_fails():
    parts = ["flex-grow", "flex-shrink", "flex-basis"]
    ans = "The longhands are flex-grow and flex-shrink."
    assert string_judge("flex-grow, flex-shrink, flex-basis", ans, parts) == 0


def test_date_parts_tolerate_reordered_phrasing():
    parts = ["27", "mar", "2023"]
    assert string_judge("27 Mar 2023", "v3 was submitted on March 27, 2023.", parts) == 1
    assert string_judge("27 Mar 2023", "v3 was submitted on Mon, 27 Mar 2023 17:46:54 UTC.", parts) == 1
    assert string_judge("27 Mar 2023", "v3 was submitted in April 2023.", parts) == 0


def test_cjk_reference():
    assert string_judge("怀旧", "The Chinese translation of nostalgia is 怀旧（之情）.") == 1
    assert string_judge("怀旧", "The Chinese translation is 可持续性.") == 0


def test_punctuation_and_spacing_noise():
    assert string_judge("durabilité", "Answers:\n- French:  durabilité!") == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
