"""Checks relating many datasets: one selector string, and member names in reports.

Such checks derive datasets from their members (the consensus, the pooled data,
each member's assessed rows). Users see those by role or by member, never by their
internal names (``consensus~checks[0]``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from qavf import validate
from qavf.cli import main
from qavf.engine.derived_names import is_derived, shown_datasets, shown_name, shown_value
from qavf.reports.html import render_html

MEMBERS = ("m0", "m1", "m2")


def _keyed() -> dict[str, pl.DataFrame]:
    """Three members agreeing on every key, except ``m2`` (other values, a key less)."""
    frames = {name: pl.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]}) for name in MEMBERS}
    frames["m2"] = pl.DataFrame({"k": [1, 2], "v": [30.0, 60.0]})
    return frames


def _drifting() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(7)
    return {
        name: pl.DataFrame({"x": rng.normal(10.0 if name == "m2" else 0.0, 1.0, 300)})
        for name in MEMBERS
    }


CASES: dict[str, tuple[dict[str, Any], Any]] = {
    "compare consensus": ({"check": "compare", "keys": ["k"], "strategy": "consensus"}, _keyed),
    "totals consensus": (
        {"check": "totals_match", "keys": ["k"], "value": "v", "strategy": "consensus"},
        _keyed,
    ),
    "keys consensus": ({"check": "key_sets_match", "keys": ["k"], "strategy": "consensus"}, _keyed),
    "drift pooled": (
        {"check": "drift", "column": "x", "max_psi": 1.0, "strategy": "consensus"},
        _drifting,
    ),
    "compare whole group": ({"check": "compare", "keys": ["k"]}, _keyed),
}


def _report(case: str) -> Any:
    item, frames = CASES[case]
    config = {
        "datasets": {name: {} for name in MEMBERS},
        "checks": [{**item, "name": "members agree", "datasets": list(MEMBERS)}],
    }
    return validate(frames(), config=config)


# --------------------------------------------------------------- shown names
def test_derived_names_are_shown_by_role_or_member() -> None:
    assert shown_name("consensus~checks[3]") == "consensus"
    assert shown_name("full~checks[0]") == "consensus"
    assert shown_name("pooled~checks[0]") == "pooled"
    assert shown_name("counts@c~checks[0]") == "counts@c"
    assert shown_name("only in consensus~checks[0]: 'counts@c~checks[0]'") == (
        "only in consensus: 'counts@c'"
    )
    assert shown_datasets(["consensus~checks[0]", "m1~checks[0]"]) == ["m1"]
    assert shown_datasets(["m0", "pooled~checks[0]"]) == ["m0"]
    assert shown_datasets(["consensus~checks[0]"]) == ["consensus"]
    assert shown_value({"m1~checks[0]_total": ["pooled~checks[0]", 1]}) == {
        "m1_total": ["pooled", 1]
    }
    assert is_derived("m0~checks[0]") and not is_derived("m0") and not is_derived("a~b")


@pytest.mark.parametrize("case", sorted(CASES))
def test_results_name_members_not_derived_datasets(case: str) -> None:
    report = _report(case)
    assert any(result.group is not None for result in report.results)
    for result in report.results:
        shown = [
            result.label,
            result.message,
            result.check or "",
            *result.datasets,
            json.dumps(result.affected_records, default=str),
            json.dumps(result.metrics, default=str),
            json.dumps(result.parameters, default=str),
        ]
        leaked = [text[:160] for text in shown if "~checks" in text]
        assert not leaked, (result.label, leaked)
        assert set(result.datasets) <= {*MEMBERS, "consensus", "pooled"}
    groups = json.dumps([group.model_dump(mode="json") for group in report.groups])
    assert "~checks" not in groups


def test_a_consensus_pair_is_about_its_member() -> None:
    report = _report("compare consensus")
    paired = [r for r in report.results if r.group is not None and r.group.members == ["m2"]]
    assert paired and all(r.datasets == ["m2"] for r in paired)
    failed = next(r for r in paired if r.status.value == "FAIL")
    assert failed.label.startswith("compare · m2 · members agree (CMP-001)")


@pytest.mark.parametrize("case", ["compare consensus", "drift pooled"])
def test_the_html_report_hides_derived_datasets(case: str) -> None:
    html = render_html(_report(case))
    visible = html.split('id="qavf-audit"')[0] if 'id="qavf-audit"' in html else html
    body = [line for line in visible.splitlines() if "~checks" in line and "&#34;" not in line]
    assert not body, body[:3]


# ------------------------------------------------------------- one selector
def _copies(tmp_path: Path) -> dict[str, Any]:
    for site, scale in (("a", 1), ("b", 1), ("c", 2)):
        (tmp_path / f"{site}.csv").write_text(
            f"k,v\n1,{10 * scale}\n2,{20 * scale}\n", encoding="utf-8"
        )
    return {
        "datasets": {
            "counts": {
                "path": (tmp_path / "{site}.csv").as_posix(),
                "across": {"site": ["a", "b", "c"]},
            }
        },
        "checks": [
            {
                "check": "totals_match",
                "datasets": "counts@*",
                "keys": ["k"],
                "value": "v",
                "strategy": "consensus",
            }
        ],
    }


def test_one_string_selects_every_copy(tmp_path: Path) -> None:
    report = validate(config=_copies(tmp_path))
    (group,) = report.groups
    assert group.members == ["counts@a", "counts@b", "counts@c"]
    assert group.ranking[0].member == "counts@c"


def test_one_string_names_a_single_dataset_like_a_list() -> None:
    frames = {"a": pl.DataFrame({"k": [1, 2]}), "b": pl.DataFrame({"k": [1, 2]})}
    for selector in ("a", ["a"]):
        config = {
            "datasets": {"a": {}, "b": {}},
            "checks": [
                {"check": "coverage", "datasets": selector, "entity": "k", "reference": "b"}
            ],
        }
        report = validate(frames, config=config)
        (coverage,) = [r for r in report.results if r.rule_id == "SPA-003"]
        assert coverage.status.value == "PASS", coverage.message


def test_the_schema_accepts_one_selector_string(capsys) -> None:
    assert main(["schema", "--backend", "polars"]) == 0
    definitions = json.loads(capsys.readouterr().out)["$defs"]
    datasets = definitions["Check_compare"]["properties"]["datasets"]
    assert {"$ref": "#/$defs/DatasetSelector"} in datasets["anyOf"]
    assert definitions["DatasetSelector"]["type"] == "string"
