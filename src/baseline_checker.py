"""Classical rule-based web checker: the non-LLM baseline.

Implements the standard mechanical checks a linkchecker/validator pipeline
gives you for free: broken internal links, missing image files, images
without alt text, forms without an action, placeholder '#' links. It is
deliberately blind to meaning (semantic defects) and rendering (visual
defects) -- that blindness is the point of the comparison.

Usage:
    python -m src.baseline_checker            # checks defect.html and control.html
"""
from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

PAGES_DIR = Path(__file__).parent.parent / "pages"


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []   # (href, text so far -- filled after)
        self.images: list[dict] = []
        self.forms: list[dict] = []
        self._link_stack: list[int] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a":
            self.links.append([a.get("href", ""), ""])  # type: ignore[arg-type]
            self._link_stack.append(len(self.links) - 1)
        elif tag == "img":
            self.images.append({"src": a.get("src", ""), "alt": a.get("alt")})
        elif tag == "form":
            self.forms.append({"action": a.get("action")})

    def handle_endtag(self, tag):
        if tag == "a" and self._link_stack:
            self._link_stack.pop()

    def handle_data(self, data):
        if self._link_stack:
            idx = self._link_stack[-1]
            self.links[idx][1] += data.strip() + " "  # type: ignore[index]


def _local_target_exists(page_path: Path, href: str) -> bool:
    """Resolve an internal href against the pages directory."""
    target = (page_path.parent / urlparse(href).path).resolve()
    return target.exists()


def check_page(page: str) -> list[str]:
    """Run all rule checks on a page in pages/; return report lines."""
    page_path = PAGES_DIR / page
    html = page_path.read_text(encoding="utf-8")
    c = _Collector()
    c.feed(html)
    findings: list[str] = []

    for href, text in c.links:
        text = text.strip() or "(no text)"
        if href in ("", "#"):
            findings.append(
                f"Broken link: '{text}' has placeholder href '#' and leads nowhere (dead link)."
            )
        elif not href.startswith(("http", "mailto:", "#")):
            if not _local_target_exists(page_path, href):
                findings.append(
                    f"Broken link: '{text}' points to {href}, which does not exist (404 error)."
                )

    for img in c.images:
        src = img["src"]
        if src and not src.startswith("http") and not _local_target_exists(page_path, src):
            findings.append(
                f"Broken image: {src} is missing, the image file does not exist."
            )
        if img["alt"] is None:
            findings.append(
                f"Accessibility: image {src or '(no src)'} has no alt attribute (missing alt text)."
            )

    for form in c.forms:
        # action="" is valid HTML5 (submit to self); only a missing attribute is a defect.
        if form["action"] is None:
            findings.append(
                "Form problem: a form has no action attribute, submitting does not work."
            )

    return findings


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.score import load_manifest, score_report

    manifest = load_manifest(PAGES_DIR / "defects.json")
    out = {}
    for page in ("defect.html", "control.html"):
        findings = check_page(page)
        report = "\n".join(findings)
        result = score_report(report, manifest)
        out[page] = {
            "findings": findings,
            "detected": result.detected,
            "n_detected": result.n_detected,
            "false_positive_lines": result.false_positive_lines,
            "n_claims": result.n_claims,
        }
        print(f"--- {page}: {len(findings)} findings, "
              f"{result.n_detected}/12 seeded defects detected ---")
        for f in findings:
            print("  " + f)

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nwrote results/baseline_results.json")


if __name__ == "__main__":
    main()
