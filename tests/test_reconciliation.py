"""Reconciliation contracts: REC-001 to REC-004."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.rules.reconciliation_checks import Dimension, Reconciliation
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)

INPUT = pl.DataFrame(
    {
        "county": [1, 1, 2, 2, 3],
        "year": [2020, 2021, 2020, 2021, 2020],
        "nox": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)
OUTPUT = INPUT.group_by("county", maintain_order=True).agg(pl.col("nox").sum().alias("total"))
REC = {
    "left": {"dataset": "inp", "value": "nox"},
    "right": {"dataset": "out", "value": "total"},
    "on": ["county"],
}


def _run(frames, rules):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False, "parallel": False},
            "datasets": {name: {} for name in frames},
            "rules": rules,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def _rec(inp, out, **extra):
    return _run(
        {"inp": inp, "out": out}, {"REC-001": {"params": {"reconciliations": [{**REC, **extra}]}}}
    ).get("REC-001")


# ------------------------------------------------------------------ REC-001
def test_balanced_totals_pass_and_are_invariant_under_permutation() -> None:
    assert _rec(INPUT, OUTPUT).status is Status.PASS
    shuffled = INPUT.sample(fraction=1.0, shuffle=True, seed=3)
    assert _rec(shuffled, OUTPUT.reverse()).status is Status.PASS


def test_a_dropped_partition_fails() -> None:
    result = _rec(INPUT.filter(pl.col("year") != 2021), OUTPUT)
    assert result.status is Status.FAIL and result.affected_count == 2  # counties 1 and 2


def test_duplicated_rows_from_a_bad_join_fail() -> None:
    result = _rec(pl.concat([INPUT, INPUT.head(1)]), OUTPUT)
    assert result.status is Status.FAIL and result.affected_records[0]["problem"] == "totals differ"


def test_a_group_missing_altogether_never_passes() -> None:
    result = _rec(INPUT, OUTPUT.filter(pl.col("county") != 3))
    assert result.status is Status.FAIL
    assert result.affected_records[0]["problem"] == "missing in out"
    assert (
        _rec(INPUT, OUTPUT.filter(pl.col("county") != 3), missing_side="ignore").status
        is Status.PASS
    )


def test_null_keys_fail_unless_exempted() -> None:
    with_null = pl.concat(
        [INPUT, pl.DataFrame({"county": [None], "year": [2020], "nox": [9.0]}, schema=INPUT.schema)]
    )
    assert _rec(with_null, OUTPUT).status is Status.FAIL
    exempt = _rec(with_null, OUTPUT, null_keys="exempt")
    assert exempt.status is Status.PASS


def test_float_totals_use_a_tolerance_but_counts_are_exact() -> None:
    nudged = OUTPUT.with_columns(pl.col("total") * (1 + 1e-12))
    assert _rec(INPUT, nudged).status is Status.PASS
    counts = {
        "left": {"dataset": "inp"},
        "right": {"dataset": "out", "value": "n"},
        "on": ["county"],
    }
    exact = OUTPUT.select("county", pl.Series("n", [2, 2, 1]))
    report = _run(
        {"inp": INPUT, "out": exact}, {"REC-001": {"params": {"reconciliations": [counts]}}}
    )
    assert report.get("REC-001").status is Status.PASS
    off_by_one = exact.with_columns(pl.col("n") + pl.Series([0, 0, 1]))
    report = _run(
        {"inp": INPUT, "out": off_by_one}, {"REC-001": {"params": {"reconciliations": [counts]}}}
    )
    assert report.get("REC-001").status is Status.FAIL


def test_keys_may_have_other_names_and_grand_totals_work() -> None:
    renamed = OUTPUT.rename({"county": "fips"})
    spec = {**REC, "right": {"dataset": "out", "value": "total", "keys": ["fips"]}}
    report = _run(
        {"inp": INPUT, "out": renamed}, {"REC-001": {"params": {"reconciliations": [spec]}}}
    )
    assert report.get("REC-001").status is Status.PASS
    grand = {
        "left": {"dataset": "inp", "value": "nox"},
        "right": {"dataset": "out", "value": "total"},
    }
    report = _run(
        {"inp": INPUT, "out": OUTPUT}, {"REC-001": {"params": {"reconciliations": [grand]}}}
    )
    assert report.get("REC-001").status is Status.PASS


def test_side_keys_must_match_on() -> None:
    with pytest.raises(ValidationError, match="keys"):
        Reconciliation.model_validate(
            {**REC, "right": {"dataset": "out", "value": "t", "keys": ["a", "b"]}}
        )


# ------------------------------------------------------------------ REC-002
def _shares(frame, **spec):
    params = {"shares": [{"dataset": "s", "value": "share", "group_by": ["g"], **spec}]}
    return _run({"s": frame}, {"REC-002": {"params": params}}).get("REC-002")


def test_shares_sum_to_one_within_groups() -> None:
    good = pl.DataFrame({"g": ["a", "a", "b"], "share": [0.25, 0.75, 1.0]})
    assert _shares(good).status is Status.PASS
    bad = good.with_columns(pl.Series("share", [0.25, 0.70, 1.0]))
    result = _shares(bad)
    assert result.status is Status.FAIL and result.affected_records[0]["g"] == "a"
    percent = good.with_columns(pl.col("share") * 100)
    assert _shares(percent, total=100).status is Status.PASS


def test_a_missing_share_fails_unless_ignored() -> None:
    frame = pl.DataFrame({"g": ["a", "a", "a"], "share": [0.5, 0.5, None]})
    assert _shares(frame).status is Status.FAIL
    assert _shares(frame, missing="ignore").status is Status.PASS


# ------------------------------------------------------------------ REC-003
GRID = pl.DataFrame({"county": [1, 1, 2], "year": [2020, 2021, 2020]})


def _grid(frames, **spec):
    params = {"grids": [{"dataset": "d", **spec}]}
    return _run(frames, {"REC-003": {"params": params}})


def test_missing_grid_combinations_are_reported() -> None:
    dims = {"county": {"values": [1, 2]}, "year": {"values": [2020, 2021]}}
    result = _grid({"d": GRID}, dimensions=dims).get("REC-003")
    assert result.status is Status.FAIL and result.affected_records == [{"county": 2, "year": 2021}]


def test_grid_domains_can_come_from_reference_data() -> None:
    counties = pl.DataFrame({"county": [1, 2, 3]})
    dims = {"county": {"from_dataset": "c"}, "year": {"values": [2020]}}
    result = _grid({"d": GRID, "c": counties}, dimensions=dims).get("REC-003")
    assert [r["county"] for r in result.affected_records] == [3]


def test_unexpected_and_repeated_combinations() -> None:
    frame = pl.concat([GRID, pl.DataFrame({"county": [9, 1], "year": [2020, 2020]})])
    dims = {"county": {"values": [1, 2]}, "year": {"values": [2020, 2021]}}
    report = _grid({"d": frame}, dimensions=dims, unexpected=True, unique=True)
    checks = {r.check for r in report.results if r.failed}
    assert {"d grid (county, year): unexpected", "d grid (county, year): unique"} <= checks


def test_unusable_domains_are_unassessed() -> None:
    dims = {"county": {"values": ["x"]}, "year": {"values": [2020]}}
    [result] = _grid({"d": GRID}, dimensions=dims).results_for("REC-003")
    assert result.status is Status.ERROR
    with pytest.raises(ValidationError):
        Dimension.model_validate({"values": []})
    with pytest.raises(ValidationError):
        Dimension.model_validate({"values": [1], "from_dataset": "c"})


# ------------------------------------------------------------------ REC-004
def test_key_sets_must_match_in_both_directions() -> None:
    a = pl.DataFrame({"id": [1, 2, 3]})
    b = pl.DataFrame({"key": [2, 3, 4]})
    params = {
        "pairs": [
            {
                "left": {"dataset": "a", "columns": ["id"]},
                "right": {"dataset": "b", "columns": ["key"]},
            }
        ]
    }
    result = _run({"a": a, "b": b}, {"REC-004": {"params": params}}).get("REC-004")
    assert result.status is Status.FAIL
    assert sorted(r["side"] for r in result.affected_records) == ["only in a", "only in b"]
    same = _run({"a": a, "b": a.rename({"id": "key"})}, {"REC-004": {"params": params}})
    assert same.get("REC-004").status is Status.PASS
