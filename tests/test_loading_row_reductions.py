"""Row reductions preserve full counts while transferring bounded original evidence."""

from contextlib import nullcontext
from decimal import Decimal
from unittest.mock import patch

import polars as pl
import pytest

from qavf import Status, validate
from qavf.engine.context import DataContext


def run(tmp_path, backend, frame, rule, *, schema=None, params=None, limit=2):
    data = frame if backend == "eager" else frame.lazy()
    if backend in {"native", "file_lazy"}:
        path = tmp_path / "rows.parquet"
        if backend == "file_lazy":
            frame = frame.with_columns(pl.lit(1).cast(pl.Duration("ns")).alias("typed_unused"))
        frame.write_parquet(path)
        data = path
    shapes = []
    checked = DataContext._check_materialization

    def record(frame):
        if any(name.startswith("__qavf_row_total") for name in frame.columns):
            shapes.append(frame.shape)
        return frame

    def check(self, name, frame):
        checked(self, name, frame)
        record(frame)

    def no_get(*args, **kwargs):
        raise AssertionError("scan-backed row rules must not materialize the full source")

    with (
        patch.object(DataContext, "_check_materialization", check),
        patch.object(DataContext, "get", no_get)
        if backend != "eager" and not (backend == "lazy" and rule == "INP-004")
        else nullcontext(),
    ):
        report = validate(
            data,
            config={
                "plugins": {"entry_points": False},
                "engine": {
                    "parallel": False,
                    "run_unconfigured_rules": False,
                    "max_affected_records": limit,
                },
                "datasets": {"data": {"schema": schema or {}}},
                "rules": {rule: {"params": params or {}}},
            },
        )
    results = report.results_for(rule)
    assert all(result.status in (Status.PASS, Status.FAIL) for result in results), [
        result.message for result in results
    ]
    return results, shapes


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize("limit", [0, 2])
def test_duplicate_rows_and_keys_have_independent_exact_summaries(tmp_path, backend, limit):
    frame = pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, None, None, -999, -999],
            "value": ["a", "a", "b", "c", "d", "e", "e", "f", "f"],
            "payload": ["x" * 5000] * 9,
            "__QAVF_ROW": list("aabcdeeff"),
            "_row": [10, 10, 20, 21, 30, 40, 40, 50, 50],
        }
    )
    results, shapes = run(
        tmp_path,
        backend,
        frame,
        "INP-004",
        schema={
            "primary_key": ["id"],
            "unique_keys": [["value"]],
            "unique_rows": True,
            "columns": {"id": {"missing_values": [-999]}},
        },
        limit=limit,
    )
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    assert set(failures) == {
        "data",
        "data: null keys",
        "data: unique (value)",
        "data: duplicate rows",
    }
    assert failures["data: null keys"].affected_count == 4
    assert failures["data"].affected_count == 2
    assert failures["data: unique (value)"].affected_count == 3
    duplicate = failures["data: duplicate rows"]
    assert (duplicate.affected_count, duplicate.total_records) == (6, 9)
    assert len(duplicate.affected_records) == limit
    if limit:
        assert [r["qavf__row"] for r in duplicate.affected_records] == [0, 1]
        assert [r["_row"] for r in duplicate.affected_records] == [10, 10]
        assert all(r["payload"] == "x" * 5000 for r in duplicate.affected_records)
    assert shapes == [(1, 2 + bool(limit))]


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
@pytest.mark.parametrize("limit", [0, 2])
def test_coordinate_scalar_summary_preserves_missing_policy_and_original_rows(
    tmp_path, backend, limit
):
    frame = pl.DataFrame(
        {
            "latitude": [30.0, 97.0, 0.0, -999.0, float("inf"), float("nan"), 91.0],
            "longitude": [-96.0, 30.0, 0.0, 10.0, 0.0, 1.0, 200.0],
            "payload": ["x" * 5000] * 7,
            "problem": ["user problem"] * 7,
            "__QAVF_ROW": ["user row"] * 7,
        }
    )
    results, shapes = run(
        tmp_path,
        backend,
        frame,
        "GEO-001",
        schema={
            "columns": {"latitude": {"missing_values": [-999]}},
        },
        params={"points": [{"dataset": "data"}]},
        limit=limit,
    )
    [result] = results
    assert (result.affected_count, result.total_records) == (6, 7)
    assert {item["problem"]: item["count"] for item in result.metrics["problems"]} == {
        "missing coordinate": 2,
        "non-finite coordinate": 1,
        "latitude out of range (latitude and longitude swapped?)": 1,
        "latitude out of range": 1,
        "(0, 0): a missing location?": 1,
    }
    assert len(result.affected_records) == limit
    if limit:
        assert [r["_row"] for r in result.affected_records] == [1, 2]
        assert all(r["problem"] == "user problem" for r in result.affected_records)
        assert all(r["payload"] == "x" * 5000 for r in result.affected_records)
        assert "swapped" in result.affected_records[0]["qavf_problem"]
    assert shapes == [(1, 8 + bool(limit))]


@pytest.mark.parametrize("backend", ["eager", "lazy", "native"])
def test_coordinate_decimal_grid_and_empty_inputs(tmp_path, backend):
    frame = pl.DataFrame(
        {
            "latitude": pl.Series([Decimal("0.1"), Decimal("0.2")], dtype=pl.Decimal(10, 1)),
            "longitude": [1, 1],
        }
    )
    points = {
        "dataset": "data",
        "bbox": {
            "min_lat": Decimal("0.10000000000000000001"),
            "max_lat": 1,
            "min_lon": 0,
            "max_lon": 2,
        },
    }
    [result], _ = run(tmp_path, backend, frame, "GEO-001", params={"points": [points]})
    assert result.affected_count == 1
    assert result.affected_records[0]["_row"] == 0
    [empty], shapes = run(tmp_path, backend, frame.head(0), "GEO-001", params={"points": [points]})
    assert empty.status is Status.PASS
    assert shapes == [(1, 9)]


@pytest.mark.parametrize("backend", ["file_lazy", "native"])
@pytest.mark.parametrize("how", ["sum", "min", "max", "count", "none"])
def test_spatial_metrics_reduce_raw_inputs_before_group_comparison(tmp_path, backend, how):
    repeats = 1 if how == "none" else 200
    frame = pl.DataFrame(
        {
            "site": [site for year in range(2) for site in range(6) for _ in range(repeats)],
            "year": [year for year in range(2) for site in range(6) for _ in range(repeats)],
            "value": [
                100 if site == 5 else 1
                for year in range(2)
                for site in range(6)
                for _ in range(repeats)
            ],
            "activity": [10] * (12 * repeats),
            "payload": ["irrelevant" * 500] * (12 * repeats),
        }
    )

    def no_raw_projection(*args, **kwargs):
        raise AssertionError("supported spatial reductions must not load raw projected rows")

    with patch.object(DataContext, "project", no_raw_projection):
        results, _ = run(
            tmp_path,
            backend,
            frame,
            "SPA-002",
            params={
                "metrics": [
                    {
                        "dataset": "data",
                        "entity_column": "site",
                        "group_by": ["year"],
                        "value": "value",
                        "aggregate": how,
                        "normalize_by": {"dataset": "data", "value": "activity", "aggregate": how},
                    }
                ],
                "max_ratio": 2,
                "min_entities": 3,
            },
        )
    [result] = results
    assert result.total_records == 12
    assert result.affected_count == (0 if how == "count" else 2)
    assert result.status is (Status.PASS if how == "count" else Status.FAIL)
