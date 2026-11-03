"""Checks relating many datasets: strategies, consensus, whole-group levels, reports.

``compare``, ``totals_match``, ``key_sets_match`` and ``drift`` take any number of
datasets and a pairing ``strategy``; the two-sided engine runs once per pair.
Two datasets with the default strategy report exactly as before.
"""

from __future__ import annotations

import json
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.checks.compiler import compile_checks
from qavf.reports.models import QAReport
from qavf.settings import QAVFSettings
from qavf.testing import rule_results

STRATEGIES = ("reference", "sequential", "all_pairs", "consensus")
SIZES = (2, 3, 5)
_VOLATILE = {"execution_time_s", "load_time_s"}


def _names(n: int) -> list[str]:
    return [f"m{i}" for i in range(n)]


def _keyed(n: int, odd: int | None) -> dict[str, pl.DataFrame]:
    """Members agreeing on every key and value, except ``odd`` (other values, a key less)."""
    frames = {}
    for index, name in enumerate(_names(n)):
        keys = [1, 2, 3, 4, 5, 6]
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        if index == odd:
            keys, values = keys[:-1], [v * 3 for v in values[:-1]]
        frames[name] = pl.DataFrame({"k": keys, "v": values})
    return frames


def _drifting(n: int, odd: int | None) -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(7)
    return {
        name: pl.DataFrame({"x": rng.normal(10.0 if index == odd else 0.0, 1.0, 400)})
        for index, name in enumerate(_names(n))
    }


CHECKS: dict[str, dict[str, Any]] = {
    "compare": {"keys": ["k"]},
    "totals_match": {"keys": ["k"], "value": "v"},
    "key_sets_match": {"keys": ["k"]},
    "drift": {"column": "x", "max_psi": 1.0},
}


def _frames(check: str, n: int, odd: int | None) -> dict[str, pl.DataFrame]:
    return _drifting(n, odd) if check == "drift" else _keyed(n, odd)


def _config(check: str, members: list[str], **settings: Any) -> dict[str, Any]:
    item = {"check": check, "datasets": members, **CHECKS[check], **settings}
    return {"datasets": {name: {} for name in members}, "checks": [item]}


def _expected_pairs(strategy: str, members: list[str]) -> list[tuple[str, str]]:
    if strategy == "reference":
        return [(members[0], other) for other in members[1:]]
    if strategy == "sequential":
        return list(pairwise(members))
    if strategy == "all_pairs":
        return list(combinations(members, 2))
    return [("consensus", member) for member in members]


# --------------------------------------------------------------------------- strategies


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("check", sorted(CHECKS))
def test_each_strategy_pairs_the_members(check: str, strategy: str, n: int) -> None:
    members = _names(n)
    odd = n - 1
    report = validate(_frames(check, n, odd), config=_config(check, members, strategy=strategy))
    results = [r for r in report.results if r.check_name == check]
    if n == 2 and strategy == "reference":
        assert report.groups == []
        assert all(result.group is None for result in results)
        assert any(result.failed for result in results)
        return
    (group,) = report.groups
    assert group.strategy == strategy
    assert group.members == members
    pairs = [(pair.reference, pair.other) for pair in group.pairs if not pair.whole_group_level]
    if check == "drift" and strategy == "consensus":
        assert pairs == [("pooled", member) for member in members]
    else:
        assert pairs == _expected_pairs(strategy, members)
    assert all(result.group is not None for result in results)
    rows = {row.member: row.cells for row in group.matrix}
    if n > 2 or strategy != "consensus":
        # Two members have no majority: a consensus of two leaves their differences
        # unassessed rather than siding with either.
        assert "FAIL" in rows[f"m{odd}"].values()
    if n > 2 and not (check == "drift" and strategy == "consensus"):
        assert group.ranking[0].member == f"m{odd}"
        assert group.ranking[0].failed > 0
        healthy = [rank for rank in group.ranking if rank.member != f"m{odd}"]
        if strategy in ("reference", "consensus"):
            assert all(rank.failed == 0 for rank in healthy)


def test_the_reference_member_can_be_chosen() -> None:
    members = _names(3)
    config = _config("compare", members, reference="m1")
    report = validate(_keyed(3, 2), config=config)
    (group,) = report.groups
    assert [(p.reference, p.other) for p in group.pairs] == [("m1", "m0"), ("m1", "m2")]


def test_sequential_attributes_each_pair_to_the_later_member() -> None:
    report = validate(
        _keyed(3, 1), config=_config("totals_match", _names(3), strategy="sequential")
    )
    (group,) = report.groups
    by_member = {rank.member: rank.failed for rank in group.ranking}
    assert by_member["m0"] == 0  # m0 is never the later member of a pair
    assert by_member["m1"] > 0 and by_member["m2"] > 0  # m1 differs from both neighbours


def test_all_pairs_is_refused_past_max_pairs() -> None:
    members = _names(9)
    with pytest.raises(ValidationError, match=r"36 pairs, more than max_pairs \(28\)"):
        QAVFSettings.model_validate(_config("key_sets_match", members, strategy="all_pairs"))
    with pytest.raises(ValidationError, match=r"6 pairs, more than max_pairs \(5\)"):
        QAVFSettings.model_validate(
            _config("key_sets_match", _names(4), strategy="all_pairs", max_pairs=5)
        )
    settings = QAVFSettings.model_validate(
        _config("key_sets_match", members, strategy="all_pairs", max_pairs=36)
    )
    assert len(compile_checks(settings)[0].params["pairs"]) == 36


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"strategy": "best"}, "unknown strategy 'best'"),
        ({"strategy": "sequential", "reference": "m1"}, "'reference' does not apply"),
        ({"reference": "nope"}, "the reference must be one of the datasets"),
        ({"max_pairs": 3}, "max_pairs applies to strategy all_pairs"),
        ({"consensus": {"numeric": "mean"}}, "consensus settings apply to strategy consensus"),
        ({"strategy": "consensus", "consensus": {"numeric": "mode"}}, "consensus: numeric"),
        ({"strategy": "consensus", "consensus": {"trim": 0.5}}, "consensus: trim"),
        ({"datasets": ["m0", "m0", "m1"]}, "datasets repeat m0"),
        ({"strategy": "consensus", "columns": {"v": "w"}}, "'columns' aligns one side only"),
    ],
)
def test_invalid_group_settings(settings: dict[str, Any], message: str) -> None:
    config = _config("compare", _names(3))
    config["checks"][0] |= settings
    with pytest.raises(ValidationError, match=message):
        QAVFSettings.model_validate(config)


def test_a_consensus_needs_keys_and_takes_no_side_overrides() -> None:
    config = _config("key_sets_match", _names(3), strategy="consensus")
    config["checks"][0].pop("keys")
    with pytest.raises(ValidationError, match="strategy consensus matches members on keys"):
        QAVFSettings.model_validate(config)
    config = _config("totals_match", _names(3), strategy="consensus", sides={"m0": {"value": "w"}})
    with pytest.raises(ValidationError, match="per-dataset overrides do not apply"):
        QAVFSettings.model_validate(config)


# ------------------------------------------------------------------- two members, as today


TODAY: dict[str, tuple[str, dict[str, Any]]] = {
    "compare": ("CMP-001", {"comparisons": [{"baseline": "m0", "candidate": "m1", "keys": ["k"]}]}),
    "totals_match": (
        "REC-001",
        {
            "reconciliations": [
                {
                    "left": {"dataset": "m0", "value": "v"},
                    "right": {"dataset": "m1", "value": "v"},
                    "on": ["k"],
                }
            ]
        },
    ),
    "key_sets_match": (
        "REC-004",
        {
            "pairs": [
                {
                    "left": {"dataset": "m0", "columns": ["k"]},
                    "right": {"dataset": "m1", "columns": ["k"]},
                }
            ]
        },
    ),
    "drift": (
        "STA-002",
        {"comparisons": [{"reference": "m0", "dataset": "m1", "column": "x", "max_psi": 1.0}]},
    ),
}


def _comparable(result: Any) -> dict[str, Any]:
    row = json.loads(result.model_dump_json())
    for field in (*_VOLATILE, "item_name", "item_ids", "check_name", "tags", "description"):
        row.pop(field, None)
    return row


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_two_datasets_with_the_default_strategy_report_as_today(check: str) -> None:
    frames = _frames(check, 2, 1)
    rule_id, params = TODAY[check]
    assert rule_id == next(
        r.rule_id
        for r in validate(frames, config=_config(check, _names(2))).results
        if r.check_name == check
    )
    settings = QAVFSettings.model_validate(_config(check, _names(2)))
    (compiled,) = compile_checks(settings)
    assert "group" not in compiled.params
    declared = {name: {} for name in frames}
    today = [_comparable(r) for r in rule_results(rule_id, frames, params, datasets=declared)]
    for config in (_config(check, _names(2)), _config(check, _names(2), strategy="reference")):
        report = validate(frames, config=config)
        assert report.groups == []
        now = [_comparable(r) for r in report.results_for(rule_id)]
        assert now == today


# ---------------------------------------------------------------------------- consensus


def _ties(n: int = 4) -> dict[str, pl.DataFrame]:
    """Key 1 agrees; key 2 splits the members two against two (a tie)."""
    frames = {}
    for index, name in enumerate(_names(n)):
        frames[name] = pl.DataFrame({"k": [1, 2], "c": ["x", "y" if index % 2 else "z"]})
    return frames


def test_a_tie_is_reported_as_not_assessed_for_that_key() -> None:
    config = _config("compare", _names(4), strategy="consensus", levels=["records"])
    report = validate(_ties(), config=config)
    consensus = next(r for r in report.results if (r.check or "").endswith(": consensus"))
    assert consensus.status.value == "ERROR"
    assert consensus.affected_count == 1
    assert consensus.affected_records == [{"k": 2, "no_consensus": "c: tie"}]
    assert consensus.group is not None and consensus.group.members == []
    # The tied key is left out of every pair: no member is judged on it.
    records = [r for r in report.results if r.group and r.group.level == "records"]
    assert records and all(not r.failed for r in records)


def test_ties_can_take_the_reference_members_value() -> None:
    config = _config(
        "compare",
        _names(4),
        strategy="consensus",
        levels=["records"],
        reference="m1",
        consensus={"ties": "reference"},
    )
    report = validate(_ties(), config=config)
    consensus = next(r for r in report.results if (r.check or "").endswith(": consensus"))
    assert consensus.status.value == "PASS"
    (group,) = report.groups
    failing = {rank.member for rank in group.ranking if rank.failed}
    assert failing == {"m0", "m2"}  # the reference m1 holds "y"; m0 and m2 hold "z"


def test_consensus_values_are_compared_with_each_member() -> None:
    frames = {
        "m0": pl.DataFrame({"k": [1, 2], "v": [10.0, 1.0]}),
        "m1": pl.DataFrame({"k": [1, 2], "v": [11.0, 1.0]}),
        "m2": pl.DataFrame({"k": [1, 2], "v": [30.0, 1.0]}),
    }
    config = _config("totals_match", _names(3), strategy="consensus")
    report = validate(frames, config=config)
    (group,) = report.groups
    # The median of 10, 11 and 30 is 11: m1 agrees, m0 and m2 differ on key 1.
    failed = {rank.member: rank.failed for rank in group.ranking}
    assert failed == {"m0": 1, "m1": 0, "m2": 1}
    config["checks"][0]["consensus"] = {"numeric": "mean"}
    report = validate(frames, config=config)
    assert {rank.member: rank.failed for rank in report.groups[0].ranking} == {
        "m0": 1,
        "m1": 1,
        "m2": 1,
    }


def test_drift_against_the_pooled_members() -> None:
    report = validate(_drifting(5, 4), config=_config("drift", _names(5), strategy="consensus"))
    (group,) = report.groups
    assert [(p.reference, p.other) for p in group.pairs] == [("pooled", m) for m in _names(5)]
    cells = {row.member: row.cells for row in group.matrix}
    assert cells["m4"]["drift"] == "FAIL"


def test_copies_across_sources_are_related_with_all_copies(tmp_path: Path) -> None:
    for site, scale in (("a", 1), ("b", 1), ("c", 2)):
        (tmp_path / f"{site}.csv").write_text(
            f"k,v\n1,{10 * scale}\n2,{20 * scale}\n", encoding="utf-8"
        )
    config = {
        "datasets": {
            "counts": {
                "path": (tmp_path / "{site}.csv").as_posix(),
                "across": {"site": ["a", "b", "c"]},
            }
        },
        "checks": [
            {
                "check": "totals_match",
                "datasets": ["counts@*"],
                "keys": ["k"],
                "value": "v",
                "strategy": "consensus",
            }
        ],
    }
    report = validate(config=config)
    (group,) = report.groups
    assert group.members == ["counts@a", "counts@b", "counts@c"]
    assert group.ranking[0].member == "counts@c"


# --------------------------------------------------------------------- whole-group levels


def test_whole_group_levels_assess_every_member_at_once() -> None:
    frames = {
        "m0": pl.DataFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0], "w": ["a", "b", "c"]}),
        "m1": pl.DataFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0], "w": ["a", "b", "c"]}),
        "m2": pl.DataFrame({"k": [1, 2], "v": [1, 2], "extra": [0, 0]}),
    }
    config = _config(
        "compare",
        _names(3),
        levels=["structure", "coverage", "records", {"level": "distribution", "columns": ["v"]}],
    )
    report = validate(frames, config=config)
    (group,) = report.groups
    assert group.whole_group_levels == ["structure", "coverage", "distribution"]
    cells = {row.member: row.cells for row in group.matrix}
    assert cells["m0"]["structure"] == cells["m1"]["structure"] == "PASS"
    assert cells["m2"]["structure"] == "FAIL"
    assert cells["m2"]["coverage"] == "FAIL" and cells["m0"]["coverage"] == "PASS"
    structure = next(
        r for r in report.results if r.group and r.group.level == "structure" and r.failed
    )
    problems = {(p["column"], p["problem"]) for p in structure.affected_records}
    assert problems == {("w", "missing"), ("v", "type"), ("extra", "extra")}
    coverage = next(
        r for r in report.results if r.group and r.group.level == "coverage" and r.failed
    )
    assert coverage.affected_records == [{"k": 3, "missing_from": "m2", "present_in": "m0, m1"}]
    # Records stay pair by pair; distribution compares each member with the pool.
    pairs = {(p.reference, p.other, p.whole_group_level) for p in group.pairs}
    assert ("m0", "m1", None) in pairs and ("m0", "m2", None) in pairs
    assert {("pooled", m, "distribution") for m in _names(3)} <= pairs


def test_two_members_keep_every_level_pair_by_pair() -> None:
    config = _config("compare", _names(2), strategy="sequential")
    report = validate(_keyed(2, 1), config=config)
    (group,) = report.groups
    assert group.whole_group_levels == []
    assert {r.group.level for r in report.results if r.group} >= {"structure", "coverage"}


# ------------------------------------------------------------------------------ reports


def test_the_group_report_round_trips_through_json(tmp_path: Path) -> None:
    report = validate(_keyed(3, 2), config=_config("compare", _names(3), strategy="all_pairs"))
    path = report.write_json(tmp_path / "report.json")
    loaded = QAReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.groups == report.groups
    payload = json.loads(path.read_text(encoding="utf-8"))
    (group,) = payload["groups"]
    assert set(group) >= {
        "name",
        "check",
        "strategy",
        "members",
        "levels",
        "matrix",
        "pairs",
        "ranking",
    }
    assert payload["results"][0]["group"] is None or "members" in payload["results"][0]["group"]


def test_reports_written_before_groups_still_load(tmp_path: Path) -> None:
    report = validate(_keyed(3, 2), config=_config("compare", _names(3)))
    payload = json.loads(report.model_dump_json())
    payload.pop("groups")
    for result in payload["results"]:
        result.pop("group")
    loaded = QAReport.model_validate(payload)
    assert loaded.groups == []
    assert all(result.group is None for result in loaded.results)


def test_the_html_report_shows_datasets_checked_together() -> None:
    from qavf.reports.html import render_html

    report = validate(_keyed(3, 2), config=_config("compare", _names(3), strategy="consensus"))
    html = render_html(report)
    assert 'id="groups"' in html
    assert "Datasets checked together (1)" in html
    assert "consensus" in html
