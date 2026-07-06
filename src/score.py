"""Deterministic scoring of a defect report against the seeded manifest.

A report (the agent's final answer, or the baseline checker's output) is split
into claim lines; a seeded defect counts as DETECTED iff at least one line
matches all of its `all` patterns (and none of its `not` patterns). Lines that
match no defect are false-positive claims. No LLM is involved anywhere:
scoring is a pure function of (report text, manifest).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Lines that are report scaffolding, not defect claims.
_NOISE = re.compile(
    r"^(here (is|are)|the following|i (found|audited|inspected|checked)|summary|"
    r"defects? (found|list|report)|report|audit|in total|overall|no (other|further|additional)|"
    r"none|n/?a)\b|^[\s\W]*$",
    re.IGNORECASE,
)
_NO_DEFECT = re.compile(r"no defects? (were )?(found|detected)", re.IGNORECASE)


@dataclass
class ScoreResult:
    detected: dict[str, bool]
    matched_lines: dict[str, list[str]] = field(default_factory=dict)
    false_positive_lines: list[str] = field(default_factory=list)
    n_claims: int = 0

    @property
    def n_detected(self) -> int:
        return sum(self.detected.values())

    @property
    def precision(self) -> float | None:
        """Fraction of claim lines that map to a seeded defect (None if no claims)."""
        if self.n_claims == 0:
            return None
        return (self.n_claims - len(self.false_positive_lines)) / self.n_claims


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_claims(report: str) -> list[str]:
    """Split a free-text report into individual defect-claim lines."""
    claims = []
    for raw in (report or "").splitlines():
        # Strip list markers: "1.", "2)", "-", "*", "•", "#### 3."
        line = re.sub(r"^[\s>#*•\-]*\d*[.):]?\s*", "", raw).strip()
        if len(line) < 8 or _NOISE.match(line) or _NO_DEFECT.search(line):
            continue
        # Scaffolding headers like "Attachments:" or "defect_report.md:", not claims.
        if line.endswith(":") and len(line.split()) < 4:
            continue
        claims.append(line)
    return claims


def _line_matches(line: str, defect: dict) -> bool:
    if any(re.search(p, line, re.IGNORECASE) for p in defect.get("not", [])):
        return False
    return all(re.search(p, line, re.IGNORECASE) for p in defect["all"])


def score_report(report: str, manifest: dict) -> ScoreResult:
    defects = manifest["defects"]
    claims = split_claims(report)
    detected = {d["id"]: False for d in defects}
    matched: dict[str, list[str]] = {d["id"]: [] for d in defects}
    false_pos: list[str] = []

    for line in claims:
        hit = False
        for d in defects:
            if _line_matches(line, d):
                detected[d["id"]] = True
                matched[d["id"]].append(line)
                hit = True
        if not hit:
            false_pos.append(line)

    return ScoreResult(
        detected=detected,
        matched_lines={k: v for k, v in matched.items() if v},
        false_positive_lines=false_pos,
        n_claims=len(claims),
    )


def recall_by_class(result: ScoreResult, manifest: dict) -> dict[str, tuple[int, int]]:
    """{class: (n_detected, n_seeded)} for one report."""
    out: dict[str, list[int]] = {}
    for d in manifest["defects"]:
        cls = d["class"]
        det, tot = out.setdefault(cls, [0, 0])
        out[cls][1] += 1
        if result.detected[d["id"]]:
            out[cls][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}
