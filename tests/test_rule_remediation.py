"""Public counterexamples and resource invariants from the full package review."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from qavf import ResourceLimits, validate
from qavf.engine.context import DataContext
from qavf.engine.severity import Status
from qavf.engine.sqlite_store import Store
from qavf.rules import temporal_checks
from qavf.rules._outlier_batches import scan_target
from qavf.rules.helpers import normalize_missing
from qavf.rules.statistical_checks import OutlierTarget
from qavf.scalars import normalize_scalar
from qavf.settings import ColumnSpec
from qavf.stats import OutlierMethod


def assessed(data, schema, rule="INP-003", *, evidence=3):
    return validate(
        {"d": data},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "datasets": {"d": {"schema": schema}},
            "rules": {rule: {}},
        },
        resources=ResourceLimits(evidence_rows=evidence),
    )


@pytest.mark.parametrize("evidence", [0, 1, 5])
@pytest.mark.parametrize("missing", [None, float("nan"), -999.0])
def test_null_fraction_cannot_replace_a_failure_with_a_second_population(evidence, missing):
    calls = []
    first = pl.DataFrame({"x": [missing, -1.0, 3.0]}, schema={"x": pl.Float64})
    good = pl.DataFrame({"x": [1.0, 2.0, 3.0]})

    def changing(_):
        calls.append(1)
        return first if len(calls) == 1 else good

    plan = first.lazy().map_batches(changing, schema=first.schema)
    contract = {
        "columns": {
            "x": {
                "nullable": True,
                "missing_values": [-999.0],
                "max_null_fraction": 0.1,
                "ge": 0,
            }
        }
    }
    report = assessed(plan, contract, evidence=evidence)
    assert calls == [1]
    assert report.verdict() == "failed"
    [failure] = [r for r in report.results if r.status is Status.FAIL]
    assert failure.affected_count == 2
    assert failure.total_records == 3
    assert [r["_row"] for r in failure.affected_records] == list(range(min(evidence, 2)))


def test_null_fraction_and_forbidden_nan_count_the_same_row_once():
    frame = pl.DataFrame({"x": [float("nan"), -1.0, 2.0]})
    schema = {
        "columns": {
            "x": {
                "nullable": True,
                "max_null_fraction": 0.1,
                "ge": 0,
                "notin": [float("nan")],
            }
        }
    }
    eager = assessed(frame, schema)
    lazy = assessed(frame.lazy(), schema)
    a = next(r for r in eager.results if r.status is Status.FAIL)
    b = next(r for r in lazy.results if r.status is Status.FAIL)
    assert a.affected_count == b.affected_count == 2
    assert [r["_row"] for r in b.affected_records] == [0, 1]
    assert b.affected_records[0]["violates"] == "notin NaN; missing"


@pytest.mark.parametrize(
    "values,scope,where,expected",
    [
        ([None, None], [1, 1], None, 2),
        ([1.0, 2.0], [1, 1], None, 0),
        ([], [], None, 0),
        ([None, -1.0, 4.0], [0, 1, 1], "scope = 1", 1),
        ([None, -1.0], [1, 1], "false", 0),
    ],
)
def test_compact_null_fraction_matches_eager_empty_and_scoped_population(
    values, scope, where, expected
):
    frame = pl.DataFrame({"x": values, "scope": scope}, schema={"x": pl.Float64, "scope": pl.Int64})
    check = {"max_null_fraction": 0.1, "ge": 0, "min_applicable_rows": 0}
    if where is not None:
        check["where"] = where
    schema = {"columns": {"x": {"nullable": True, "checks": [check]}}}
    eager = assessed(frame, schema)
    lazy = assessed(frame.lazy(), schema)
    assert lazy.verdict() == eager.verdict()
    assert sum(r.affected_count for r in lazy.results if r.status is Status.FAIL) == expected
    timing = {"execution_time_s", "load_time_s"}
    assert [r.model_dump(exclude=timing) for r in lazy.results] == [
        r.model_dump(exclude=timing) for r in eager.results
    ]


@pytest.mark.parametrize(
    "dtype,first,second",
    [
        ("int", {"x": ["bad"], "y": ["1"]}, {"x": ["1"], "y": ["bad"]}),
        ("int", {"x": [1.5], "y": [1.0]}, {"x": [1.0], "y": [1.5]}),
        ("decimal", {"x": ["bad"], "y": ["1.00"]}, {"x": ["1.00"], "y": ["bad"]}),
    ],
)
def test_schema_conformance_uses_one_observation_for_all_columns(dtype, first, second):
    frames = pl.DataFrame(first), pl.DataFrame(second)
    calls = []

    def changing(_):
        calls.append(1)
        return frames[(len(calls) - 1) % 2]

    plan = frames[0].lazy().map_batches(changing, schema=frames[0].schema)
    report = assessed(
        plan,
        {
            "columns": {
                "x": {"dtype": dtype},
                "y": {"dtype": dtype},
            }
        },
        "INP-002",
    )
    assert calls == [1]
    assert report.verdict() == "failed"
    [failure] = [r for r in report.results if r.status is Status.FAIL]
    assert failure.check == f"d.x: dtype {dtype}"
    assert failure.affected_count == failure.total_records == 1


@pytest.mark.parametrize("evidence", [0, 2])
def test_text_conversion_preserves_explicit_whole_row_evidence(evidence, monkeypatch):
    observed = []
    original = DataContext.observe

    def observe(self, dataset, columns, *args, **kwargs):
        observed.append(list(columns))
        return original(self, dataset, columns, *args, **kwargs)

    monkeypatch.setattr(DataContext, "observe", observe)
    frame = pl.DataFrame({"id": [7, 8], "x": ["2", "4"], "payload": ["keepA", "keepB"]})
    report = assessed(
        frame.lazy(),
        {
            "primary_key": ["id"],
            "evidence_columns": "all",
            "columns": {"x": {"dtype": "int", "ge": 3}},
        },
        evidence=evidence,
    )
    [failure] = [r for r in report.results if r.status is Status.FAIL]
    assert failure.affected_count == 1
    if evidence:
        assert failure.affected_records[0]["payload"] == "keepA"
        assert failure.affected_records[0]["id"] == 7
        assert observed == [["id", "x", "payload"]]
    else:
        assert failure.affected_records == []
        assert observed == [["x"]]


@pytest.mark.parametrize("literal", [np.int8(1), np.int64(1), np.uint64(1), np.float32(1)])
def test_numpy_membership_literals_match_python_values(literal):
    report = assessed(pl.DataFrame({"x": [1]}), {"columns": {"x": {"isin": [literal]}}})
    assert report.verdict() == "passed"


def test_numpy_scalar_normalization_preserves_unsigned_boolean_and_float_identity():
    huge = np.uint64(2**64 - 1)
    spec = ColumnSpec(isin=[huge], ge=huge, missing_values=[np.float32(0.1)])
    assert type(spec.isin[0]) is int and spec.isin[0] == 2**64 - 1
    assert type(spec.ge) is int and spec.ge == 2**64 - 1
    assert spec.missing_values == [float(np.float32(0.1))]
    assert normalize_scalar(np.bool_(True)) is True
    assert normalize_scalar(Decimal("0.1")) == Decimal("0.1")
    report = assessed(pl.DataFrame({"x": [1]}), {"columns": {"x": {"isin": [np.bool_(True)]}}})
    assert report.verdict() == "incomplete"


def test_numpy_extended_precision_and_temporal_literals_never_silently_narrow():
    for scalar in [np.datetime64(1, "ns"), np.timedelta64(1, "ns")]:
        with pytest.raises(ValueError, match="temporal"):
            ColumnSpec(isin=[scalar])
    value = np.longdouble(1) + np.finfo(np.longdouble).eps
    if value != np.longdouble(float(value)):
        with pytest.raises(ValueError, match="exactly representable"):
            ColumnSpec(isin=[value])


@pytest.mark.parametrize("correction", ["none", "bh"])
@pytest.mark.parametrize("rows,byte_limit", [(0, 10000), (10, 0), (2, 10000)])
def test_corrected_trends_hydrate_only_admitted_evidence(monkeypatch, rows, byte_limit, correction):
    annotations = []
    original = temporal_checks.annotate

    def annotate(*args, **kwargs):
        annotations.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(temporal_checks, "annotate", annotate)
    frame = pl.DataFrame(
        {
            "g": [group for group in range(25) for _ in range(10)],
            "year": list(range(10)) * 25,
            "v": list(reversed(range(10))) * 25,
        }
    )
    report = validate(
        {"d": frame},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "rules": {
                "TMP-002": {
                    "params": {
                        "series": [
                            {
                                "dataset": "d",
                                "value": "v",
                                "group_by": ["g"],
                                "expected_trend": "increasing",
                            }
                        ],
                        "correction": correction,
                    }
                }
            },
        },
        resources=ResourceLimits(evidence_rows=rows, evidence_bytes=byte_limit),
    )
    [result] = report.results
    assert result.status is Status.FAIL
    assert result.affected_count == 25
    expected = rows if byte_limit else 0
    assert len(annotations) == len(result.affected_records) == expected
    assert [record["g"] for record in result.affected_records] == list(range(expected))


def test_deferred_evidence_spill_refusal_preserves_complete_failure_count(monkeypatch):
    from qavf.engine.resources import ResourceLimitError

    def unavailable(*args, **kwargs):
        raise ResourceLimitError("evidence storage is full")

    monkeypatch.setattr(DataContext, "work_store", unavailable)
    frame = pl.DataFrame({"year": [0, 1, 2], "v": [3, 2, 1]})
    report = validate(
        {"d": frame},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "rules": {
                "TMP-002": {
                    "params": {
                        "series": [{"dataset": "d", "value": "v", "expected_trend": "increasing"}],
                        "correction": "bh",
                    }
                }
            },
        },
    )
    [result] = report.results
    assert result.status is Status.FAIL
    assert result.affected_count == result.total_records == 1
    assert result.affected_records == []


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (pl.Datetime("ns"), ["1970-01-01T00:00:00.000000000", "1970-01-01T00:00:00.000000001"]),
        (pl.Duration("ns"), ["0", "1"]),
        (pl.Time, ["0", "1"]),
    ],
)
def test_corrected_trend_evidence_preserves_nanosecond_group_identities(dtype, expected):
    frame = pl.DataFrame(
        {
            "g": pl.Series([0] * 5 + [1] * 5, dtype=dtype),
            "year": list(range(5)) * 2,
            "v": list(reversed(range(5))) * 2,
        }
    )
    report = validate(
        {"d": frame},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "rules": {
                "TMP-002": {
                    "params": {
                        "series": [
                            {
                                "dataset": "d",
                                "value": "v",
                                "group_by": ["g"],
                                "expected_trend": "increasing",
                            }
                        ],
                        "correction": "bh",
                        "alpha": 0.1,
                    }
                }
            },
        },
    )
    [result] = report.results
    assert result.status is Status.FAIL
    assert result.affected_count == result.total_records == 2
    assert [record["g"] for record in result.affected_records] == expected


@pytest.mark.parametrize("method", list(OutlierMethod))
@pytest.mark.parametrize("transform", ["none", "log"])
def test_outlier_storage_contains_only_method_required_indices(monkeypatch, method, transform):
    statements = []
    original = Store.execute

    def execute(self, sql, parameters=()):
        statements.append(sql)
        return original(self, sql, parameters)

    monkeypatch.setattr(Store, "execute", execute)
    frame = pl.DataFrame({"x": [1.0] * 15 + [100.0]})
    target = OutlierTarget(dataset="d", value="x", transform=transform)
    with DataContext(frames={"d": frame}) as ctx:
        result = scan_target(
            ctx,
            target,
            method=method,
            threshold=None,
            min_group_size=3,
            evidence_limit=1,
            normalize=normalize_missing,
        )
    assert result.tested == 16
    assert any("CREATE INDEX values_order" in sql for sql in statements) == (transform == "log")
    assert any("CREATE TABLE deviations" in sql for sql in statements) == (
        method is OutlierMethod.MODIFIED_ZSCORE
    )
