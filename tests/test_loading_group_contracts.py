"""Independent grouped QA invariants across narrow execution boundaries."""

from contextlib import contextmanager
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.reconciliation_checks import SharesSumRule, TotalsReconcileRule
from qavf.rules.temporal_checks import CumulativeRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource


@contextmanager
def source_context(frame, backend, tmp_path, *, evidence=1, columns=None):
    options = {
        "specs": {"s": DatasetSpec.model_validate({"schema": {"columns": columns or {}}})},
        "evidence_limit": evidence,
    }
    if backend in {"eager", "lazy"}:
        options["frames"] = {"s": frame.lazy() if backend == "lazy" else frame}
    else:
        path = tmp_path / "input.parquet"
        frame.write_parquet(path)
        location = str(path).replace("'", "''")
        options["sources"] = {
            "s": (
                FileSource(FileParams(path=str(path)), SourceEnvironment())
                if backend == "file"
                else QuerySource(
                    QueryParams(sql=f"SELECT * FROM read_parquet('{location}')"),
                    SourceEnvironment(),
                )
            )
        }
    with DataContext(**options) as context:
        yield context


@pytest.mark.parametrize("backend", ["eager", "lazy", "file", "query"])
@pytest.mark.parametrize("evidence", [0, 1])
def test_invalid_time_count_is_complete_and_excludes_whole_null_equal_group(
    backend, evidence, tmp_path
):
    # Five invalid times exceed the evidence allowance. A sixth, valid row in
    # that same normalized null group must also be excluded from the trend test.
    frame = pl.DataFrame(
        {
            "group": ["MISSING"] * 6 + ["clean", "clean"],
            "year": [None] * 5 + [1.0, 1.0, 2.0],
            "value": [1] * 5 + [100, 10, 9],
        }
    )
    with source_context(
        frame, backend, tmp_path, evidence=evidence,
        columns={"group": {"missing_values": ["MISSING"]}},
    ) as context:
        results = CumulativeRule(
            {"series": [{"dataset": "s", "group_by": ["group"], "value": "value"}]}
        ).evaluate(context)
    invalid = [result for result in results if not result.assessed]
    assert len(invalid) == 1 and invalid[0].affected_count == 5
    assert len(invalid[0].affected) == evidence
    failures = [result for result in results if result.assessed and not result.passed]
    assert len(failures) == 1 and failures[0].affected_count == 1
    assert failures[0].total_records == 1


@pytest.mark.parametrize("backend", ["eager", "lazy", "file", "query"])
@pytest.mark.parametrize("dtype", [pl.Int64, pl.Decimal(38, 2)])
def test_null_only_period_is_never_replaced_by_measured_zero(backend, dtype, tmp_path):
    values = [None, None, 1, 2]
    if isinstance(dtype, pl.Decimal):
        values = [None, None, Decimal("1.00"), Decimal("2.00")]
    frame = pl.DataFrame(
        {
            "group": ["unknown", "unknown", "clean", "clean"],
            "year": [1, 2, 1, 2],
            "value": pl.Series(values, dtype=dtype),
        }
    )
    with source_context(frame, backend, tmp_path) as context:
        results = CumulativeRule(
            {"series": [{"dataset": "s", "group_by": ["group"], "value": "value"}]}
        ).evaluate(context)
    invalid = [result for result in results if not result.assessed]
    assert len(invalid) == 1 and invalid[0].affected_count == 2
    passed = [result for result in results if result.assessed and result.passed]
    assert len(passed) == 1 and passed[0].total_records == 1


@pytest.mark.parametrize("right_rows", [32, 33])
def test_reconciliation_row_count_scans_owned_spill_without_wide_reload(
    right_rows, tmp_path, monkeypatch
):
    frames = {
        "left": pl.DataFrame({"payload": ["x" * 4096] * 33}),
        "right": pl.DataFrame({"payload": ["y" * 4096] * right_rows}),
    }
    with DataContext(frames=frames, cache_max_bytes=0, temp_directory=tmp_path) as context:
        assert context.resource_usage()["frame_spill_bytes"] > 0

        def no_full_reload(*args, **kwargs):
            pytest.fail("row counts must not reload a complete owned spill")

        monkeypatch.setattr(context, "get", no_full_reload)
        results = TotalsReconcileRule(
            {"reconciliations": [{"left": {"dataset": "left"}, "right": {"dataset": "right"}}]}
        ).evaluate(context)
        usage = context.resource_usage()
    assert len(results) == 1 and results[0].assessed
    assert results[0].passed == (right_rows == 33)
    assert usage["frame_spill_read_count"] == 0
    assert usage["frame_spill_reduce_count"] == 2


def test_share_metadata_and_total_use_one_supplied_lazy_observation():
    observations = []

    def changing_values(frame):
        observations.append(1)
        return frame.with_columns(pl.col("value") * len(observations))

    lazy = pl.DataFrame({"value": [0, 1]}).lazy().map_batches(changing_values)
    with DataContext(frames={"s": lazy}) as context:
        results = SharesSumRule(
            {"shares": [{"dataset": "s", "value": "value", "total": 1, "atol": 0}]}
        ).evaluate(context)
    assert observations == [1]
    assert len(results) == 1 and results[0].assessed and results[0].passed
