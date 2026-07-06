"""Scorer sanity checks: realistic phrasings must map to the right defect."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.score import load_manifest, score_report, split_claims

MANIFEST = load_manifest(Path(__file__).parent.parent / "pages" / "defects.json")

# One plausible agent phrasing per defect, plus paraphrase variants.
PHRASINGS = {
    "D01": [
        "The 'Last year's menu' link points to menu-archive.html which returns a 404 error.",
        "Link 'Last year's menu' is broken - the page does not exist.",
    ],
    "D02": [
        "The terrace image (images/terrace.jpg) is broken and does not load.",
        "Image 'Our terrace above the river Inn' is missing - file not found.",
    ],
    "D03": [
        "The cake image has no alt text, which is an accessibility issue.",
        "First image in the About section is missing an alt attribute.",
    ],
    "D04": [
        "The newsletter form has no action attribute, so subscribing does nothing.",
        "The Subscribe button does not work - the form is not functional.",
    ],
    "D05": [
        "The Reservations link has href '#' and leads nowhere.",
        "'Reservations' in the navigation is a dead link that does not go to any page.",
    ],
    "D06": [
        "The 'View full menu' button links to the Visit section instead of the menu.",
        "Button labeled 'View full menu' is misleading: it goes to the opening hours anchor.",
    ],
    "D07": [
        "The Apfelstrudel price is contradictory: EUR 3.50 in the menu but 4.80 in the banner.",
        "Price mismatch for Apfelstrudel between the menu table and the promotional banner.",
    ],
    "D08": [
        "The About section contains a garbled paragraph of random characters.",
        "There is a gibberish text block ('Ihr asdkfj qwelrju...') in the About section.",
    ],
    "D09": [
        "The footer copyright year 1875 is wrong - the cafe was founded in 1998.",
        "Copyright says 1875, which is an impossible year for this business.",
    ],
    "D10": [
        "Opening hours contradict each other: header says open daily but footer says closed on Sundays.",
        "The header and footer give inconsistent opening hours regarding Sundays.",
    ],
    "D11": [
        "There is white text on a white background in the newsletter section, invisible to visitors.",
        "A hidden paragraph about a 20% subscriber discount cannot be seen.",
    ],
    "D12": [
        "The 'Order a cake box' button is covered by an overlay and cannot be clicked.",
        "The order button is unclickable because another element overlaps it.",
    ],
}


def test_each_phrasing_maps_to_its_defect() -> None:
    for did, lines in PHRASINGS.items():
        for line in lines:
            result = score_report(line, MANIFEST)
            assert result.detected[did], f"{did} NOT detected from: {line!r}"


def test_full_report_detects_all_and_no_false_positives() -> None:
    report = "\n".join(f"{i + 1}. {lines[0]}"
                       for i, lines in enumerate(PHRASINGS.values()))
    result = score_report(report, MANIFEST)
    assert result.n_detected == 12, f"only {result.n_detected}/12: {result.detected}"
    assert not result.false_positive_lines, result.false_positive_lines


def test_clean_report_has_no_detections() -> None:
    report = ("1. The page looks fine overall.\n"
              "2. Navigation works as expected and prices are displayed.")
    result = score_report(report, MANIFEST)
    assert result.n_detected == 0, result.detected


def test_split_claims_strips_noise() -> None:
    report = ("Here is my audit report:\n"
              "1. The terrace image is broken.\n"
              "- \n"
              "No defects found beyond these.\n")
    claims = split_claims(report)
    assert claims == ["The terrace image is broken."], claims


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all scorer tests passed")
