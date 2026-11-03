"""The configuration guide, the check reference and ``qavf init`` show valid 0.7 forms."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from qavf.checks.catalog import CATALOG
from qavf.checks.legacy import first_legacy_key
from qavf.scaffold import render_config
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

ROOT = Path(__file__).resolve().parents[1]
_SECTIONS = ("datasets", "connections")


def _blocks(page: str) -> list[dict[str, Any]]:
    text = (ROOT / "docs" / page).read_text(encoding="utf-8")
    return [yaml.safe_load(block) for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)]


def _merged(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """The blocks as one configuration: sections merged, check lists concatenated."""
    merged: dict[str, Any] = {}
    for block in blocks:
        for key, value in block.items():
            if key in _SECTIONS:
                merged.setdefault(key, {}).update(value)
            elif key == "checks":
                merged.setdefault("checks", []).extend(value)
            else:
                merged[key] = value
    return merged


def test_the_configuration_guide_examples_form_one_valid_configuration() -> None:
    blocks = _blocks("checks.md")
    assert len(blocks) >= 5
    for block in blocks:
        assert first_legacy_key(block) is None
    settings = QAVFSettings.model_validate(_merged(blocks))
    assert {"volumes@houston", "stations@austin", "readings@archive"} <= set(settings.datasets)


def test_no_documented_configuration_uses_a_rule_code_key() -> None:
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL):
            assert not re.search(r"^\s*[A-Z]{3}-\d{3}:", block, re.MULTILINE), page.name
            assert first_legacy_key(yaml.safe_load(block)) is None, page.name


def test_the_check_reference_leads_with_check_names_and_hides_internal_settings() -> None:
    text = (ROOT / "docs" / "rules.md").read_text(encoding="utf-8")
    for name in CATALOG:
        assert f"\n### {name}\n" in text, name
    assert "| `group` |" not in text
    assert "GroupPlan" not in text


def test_the_init_example_is_a_valid_configuration() -> None:
    text = render_config(discovered_rules(entry_points=False)).split("# Available checks")[0]
    lines = []
    for line in text.splitlines():
        if line in ("datasets: {}", "connections: {}", "checks: []"):
            lines.append(line.split(":")[0] + ":")
        elif line.startswith(("#vars:", "#defaults:")) or (
            line.startswith("#  ") and "warehouse" not in line
        ):
            lines.append(line[1:])
        elif not line.startswith("#"):
            lines.append(line)
    raw = yaml.safe_load("\n".join(lines))
    raw.pop("connections")
    settings = QAVFSettings.model_validate(raw)
    assert set(settings.datasets) == {"traffic@north", "traffic@south"}
    assert [item.check for item in settings.checks] == ["sql", "compare"]
    column = settings.datasets["traffic@north"].schema_spec.columns["aadt"]
    assert column.stats is not None and column.conditions
