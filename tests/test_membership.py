"""Typed membership, the coverage ledger and deterministic evidence.

Typed ``isin``/``notin`` membership found while running the shipped examples end to
end, the coverage ledger: passed checks, excluded rules and required checks, and
deterministic evidence, load timing and invariance.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from qavf.engine.runner import ConfigurationError, QARunner
from qavf.engine.severity import Status
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

REGISTRY = discovered_rules(entry_points=False)


def _run(frames, *, parallel: bool = False, **config):
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"parallel": parallel, **config.pop("engine", {})},
            **config,
        }
    )
    return QARunner(settings, REGISTRY).run(frames=frames)


def _inp003(frame: pl.DataFrame, limits: dict) -> dict:
    report = _run({"t": frame}, datasets={"t": {"schema": {"columns": {"v": limits}}}})
    return {r.check: r for r in report.results_for("INP-003")}


# =========================================================================
# isin / notin compare exactly across numeric types (Polars 2 is_in)
# =========================================================================
@pytest.mark.parametrize(
    ("values", "dtype", "limits", "bad"),
    [
        (
            [1.0, -999.0, 2.0],
            pl.Float64,
            {"notin": [-999, -9999]},
            1,
        ),  # int sentinels, float column
        ([1, 2, 3], pl.Int64, {"isin": [1.0, 2.0]}, 1),  # integral floats, int column
        ([1, 2, 3], pl.Int64, {"isin": [1, 2.5]}, 2),  # 2.5 can never match an integer
        ([1, 2], pl.UInt8, {"isin": [1, 300, -1]}, 1),  # out-of-range values never match
        ([2**53 + 1, 2**53], pl.Int64, {"isin": [2**53 + 1]}, 1),  # exact beyond 2**53
        ([2.0**53], pl.Float64, {"notin": [2**53 + 1]}, 0),  # not representable: no match
        ([Decimal("1.50"), Decimal("2.00")], pl.Decimal(10, 2), {"isin": [1.5]}, 1),
        (["a", "b"], pl.Categorical, {"isin": ["a"]}, 1),
        ([True, False], pl.Boolean, {"isin": [True]}, 1),
        ([date(2020, 1, 1), date(2020, 1, 2)], pl.Date, {"isin": [date(2020, 1, 1)]}, 1),
        (
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            pl.Datetime("us"),
            {"isin": [date(2020, 1, 1)]},
            1,
        ),
    ],
)
def test_membership_is_exact_and_typed(values, dtype, limits, bad) -> None:
    frame = pl.DataFrame({"v": pl.Series(values, dtype=dtype)})
    failed = [r for r in _inp003(frame, limits).values() if r.failed]
    if bad:
        assert [r.status for r in failed] == [Status.FAIL] and failed[0].affected_count == bad
    else:
        assert not failed


def test_long_membership_lists_use_the_fast_path() -> None:
    allowed = list(range(0, 2000, 2))
    frame = pl.DataFrame({"v": [0.0, 1.0, 4.0]})
    [result] = [r for r in _inp003(frame, {"isin": allowed}).values() if r.failed]
    assert result.status is Status.FAIL and result.affected_count == 1


def test_membership_values_of_the_wrong_kind_are_unassessed() -> None:
    frame = pl.DataFrame({"v": ["1", "2"]})
    [result] = [r for r in _inp003(frame, {"isin": [1, 2]}).values() if r.failed]
    assert result.status is Status.ERROR and "cannot apply" in result.message


def test_membership_on_time_zone_aware_columns_follows_the_bound_policy() -> None:
    frame = pl.DataFrame({"v": [datetime(2020, 1, 1, tzinfo=UTC)]})
    assert not any(r.failed for r in _inp003(frame, {"isin": [datetime(2020, 1, 1)]}).values())


# =========================================================================
# The report accounts for every check and rule
# =========================================================================
def test_passed_checks_are_listed_individually() -> None:
    frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    report = _run(
        {"t": frame},
        datasets={
            "t": {"schema": {"columns": {"a": {"ge": 0, "le": 1}, "b": {"isin": ["x", "y"]}}}}
        },
    )
    [summary] = [r for r in report.results_for("INP-003") if r.status is Status.PASS]
    assert summary.metrics["passed_checks"] == [
        "t.a: not null",
        "t.b: not null",
        "t.b: isin [x, y]",
    ]
    assert report.get("INP-003", "t.a: ge 0, le 1").status is Status.FAIL


def test_excluded_rules_are_recorded_with_the_reason() -> None:
    report = _run(
        {"t": pl.DataFrame({"a": [1]})},
        datasets={"t": {}},
        rules={"STA-001": {"enabled": False}},
        engine={"exclude_categories": ["temporal"]},
    )
    excluded = report.metadata.excluded_rules
    assert excluded["STA-001"] == "disabled in the configuration"
    assert "exclude_categories" in excluded["TMP-001"]
    assert "INP-001" not in excluded
    assert json.loads(report.model_dump_json())["metadata"]["excluded_rules"] == excluded


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"rules": {"INP-004": {"enabled": False}}}, "excluded"),
        ({"engine": {"exclude_categories": ["input"]}}, "excluded"),
        ({}, "skipped"),  # no primary key configured: INP-004 skips
    ],
)
def test_required_checks_that_were_not_assessed_make_the_run_incomplete(config, reason) -> None:
    engine = {"require": ["INP-004"], **config.pop("engine", {})}
    report = _run({"t": pl.DataFrame({"a": [1]})}, datasets={"t": {}}, engine=engine, **config)
    [required] = [r for r in report.results if r.check == "required: INP-004"]
    assert required.status is Status.ERROR and reason in required.message
    assert report.verdict() == "incomplete"


def test_required_checks_that_ran_add_nothing() -> None:
    datasets = {"t": {"schema": {"primary_key": ["a"], "columns": {"a": {"ge": 0}}}}}
    report = _run(
        {"t": pl.DataFrame({"a": [1, 2]})},
        datasets=datasets,
        engine={"require": ["INP-004", "INP-003[t.a: ge 0]"]},
    )
    assert not [r for r in report.results if (r.check or "").startswith("required:")]
    assert report.verdict() == "passed"


def test_a_required_single_check_must_exist() -> None:
    report = _run(
        {"t": pl.DataFrame({"a": [1]})},
        datasets={"t": {"schema": {"columns": {"a": {"ge": 0}}}}},
        engine={"require": ["INP-003[t.a: le 5]"]},
    )
    assert report.verdict() == "incomplete"


def test_unknown_required_rules_are_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="NOPE-1"):
        _run({}, engine={"require": ["NOPE-1"]})


# =========================================================================
# Evidence is deterministic; verdicts do not depend on row order
# =========================================================================
def _assertion_report(parallel: bool, frame: pl.DataFrame):
    return _run(
        {"t": frame},
        parallel=parallel,
        datasets={"t": {}},
        assertions=[
            {
                "name": "busy groups",
                "sql": "SELECT g, count(*) AS n FROM t GROUP BY g HAVING count(*) > 30",
            }
        ],
        engine={"max_affected_records": 10},
    )


def test_sql_assertion_evidence_is_the_same_on_every_run() -> None:
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({"g": rng.integers(0, 20_000, 800_000), "v": rng.normal(size=800_000)})
    samples = {
        json.dumps(_assertion_report(parallel, frame).get("INP-007").affected_records)
        for parallel in (False, True, False, True)
    }
    assert len(samples) == 1


def test_parallel_and_sequential_runs_agree_on_everything_but_timing() -> None:
    from qavf.demo import inject_defects, make_demo_data

    def stable(report):
        return [
            (
                r.rule_id,
                r.check,
                r.status,
                r.severity,
                r.message,
                r.affected_count,
                r.affected_records,
            )
            for r in report.results
        ]

    frames = inject_defects(make_demo_data(seed=0))
    import yaml

    from qavf.demo import demo_config

    config = yaml.safe_load(demo_config())
    config.pop("datasets", None)
    settings = {k: v for k, v in config.items() if k in ("rules", "assertions")}
    datasets = yaml.safe_load(demo_config())["datasets"]
    for spec in datasets.values():
        spec.pop("path", None)
        spec.pop("source", None)
    runs = [_run(dict(frames), parallel=p, datasets=datasets, **settings) for p in (False, True)]
    assert stable(runs[0]) == stable(runs[1])


def test_shuffling_rows_does_not_change_verdicts_or_counts() -> None:
    from qavf.demo import inject_defects, make_demo_data

    frames = inject_defects(make_demo_data(seed=0))
    shuffled = {
        name: frame.sample(fraction=1.0, shuffle=True, seed=7) for name, frame in frames.items()
    }
    import yaml

    from qavf.demo import demo_config

    config = yaml.safe_load(demo_config())
    datasets = config["datasets"]
    for spec in datasets.values():
        spec.pop("path", None)
        spec.pop("source", None)
    settings = {k: v for k, v in config.items() if k in ("rules", "assertions")}

    def summary(report):
        return sorted(
            (r.rule_id, r.check or "", r.status.value, r.affected_count) for r in report.results
        )

    assert summary(_run(frames, datasets=datasets, **settings)) == summary(
        _run(shuffled, datasets=datasets, **settings)
    )


def test_dataset_load_times_are_recorded(tmp_path) -> None:
    path = tmp_path / "d.csv"
    pl.DataFrame({"a": [1, 2]}).write_csv(path)
    report = _run({}, datasets={"d": {"path": str(path)}})
    [info] = report.metadata.datasets
    assert info.load_time_s is not None and info.load_time_s >= 0
