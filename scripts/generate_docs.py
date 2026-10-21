"""Regenerate docs/rules.md, docs/sources.md and docs/comparison_reference.md from the code.

uv run python scripts/generate_docs.py          # write
uv run python scripts/generate_docs.py --check  # exit 1 if out of date
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer this checkout over an unrelated editable installation. An isolated
# wheel verification directory has no qavf/ tree and uses its installed package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qavf.compare import ALIGNMENT_STEPS, COMPARATORS, LEVELS, MEASURES  # noqa: E402
from qavf.docgen import comparison_markdown, rules_markdown, sources_markdown  # noqa: E402
from qavf.sources import SourceRegistry  # noqa: E402
from qavf.testing import discovered_rules  # noqa: E402

DOCS = ROOT / "docs"


def pages() -> dict[Path, str]:
    """Generated documentation pages keyed by path."""
    return {
        DOCS / "rules.md": rules_markdown(discovered_rules(entry_points=False), "Built-in rules"),
        DOCS / "sources.md": sources_markdown(
            SourceRegistry().discover(use_entry_points=False), "Built-in sources"
        ),
        DOCS / "comparison_reference.md": comparison_markdown(
            {
                "Levels": LEVELS.discover(use_entry_points=False),
                "Measures": MEASURES.discover(use_entry_points=False),
                "Comparators": COMPARATORS.discover(use_entry_points=False),
                "Alignment steps": ALIGNMENT_STEPS.discover(use_entry_points=False),
            },
            "Built-in comparison components",
        ),
    }


def main(argv: list[str]) -> int:
    stale = []
    for path, text in pages().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            stale.append(path)
            if "--check" not in argv:
                path.write_text(text, encoding="utf-8", newline="\n")
                print(f"wrote {path}")
    if "--check" in argv and stale:
        print("out of date: " + ", ".join(str(p) for p in stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
