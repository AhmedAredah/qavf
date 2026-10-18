"""Independent complete-input and evidence contracts for narrow rule execution."""

from contextlib import contextmanager
from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.geo_checks import CoordinateRule
from qavf.rules.statistical_checks import ColumnOutlierRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams, QueryParams
from qavf.sources.query import QuerySource

BACKENDS = ["lazy", "file", "native", "spill"]


@contextmanager
def input_context(frame, backend, tmp_path, *, evidence=2, row_budget=None, schema=None):
    """Exercise four distinct storage paths with the same independent input."""
    spec = DatasetSpec.model_validate(
        {
            **({"load_policy": {"row_budget": row_budget}} if row_budget else {}),
            **({"schema": schema} if schema else {}),
        }
    )
    options = {
        "specs": {"s": spec},
        "evidence_limit": evidence,
        "temp_directory": tmp_path / "owned",
    }
    if backend == "lazy":
        options["frames"] = {"s": frame.lazy()}
    elif backend == "spill":
        options.update(frames={"s": frame}, cache_max_bytes=0)
    else:
        path = tmp_path / "input.parquet"
        if backend == "file":
            # Force the exact typed-file reader, distinct from native SQL.
            frame = frame.with_columns(
                pl.Series("typed_only", range(frame.height), dtype=pl.Duration("ns"))
            )
        frame.write_parquet(path)
        location = str(path).replace("'", "''")
        source = (
            FileSource(FileParams(path=str(path)), SourceEnvironment())
            if backend == "file"
            else QuerySource(
                QueryParams(sql=f"SELECT * FROM read_parquet('{location}')"),
                SourceEnvironment(),
            )
        )
        options["sources"] = {"s": source}
    with DataContext(**options) as context:
        if backend == "spill" and (row_budget is None or frame.height <= row_budget):
            assert context.resource_usage()["frame_spill_bytes"] > 0
        yield context


def observe_materialization(context, monkeypatch):
    """Observe real result boundaries; output length alone cannot prove pushdown."""
    observed = []
    original = context._check_materialization

    def check(name, frame):
        observed.append(frame)
        return original(name, frame)

    def no_full_get(*args, **kwargs):
        pytest.fail("rule requested the complete wide input through ctx.get")

    monkeypatch.setattr(context, "_check_materialization", check)
    monkeypatch.setattr(context, "get", no_full_get)
    return observed


def payload_records(frame):
    """Count retained wide records in direct, struct, or list-of-struct results."""
    if "payload" in frame.columns:
        return frame.height
    count = 0
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Struct) and "payload" in {f.name for f in dtype.fields}:
            count += frame[name].is_not_null().sum()
        elif (
            isinstance(dtype, pl.List)
            and isinstance(dtype.inner, pl.Struct)
            and "payload" in {f.name for f in dtype.inner.fields}
        ):
            count += frame[name].list.len().fill_null(0).sum()
    return count


def failures(findings):
    assert not [f.message for f in findings if not f.assessed]
    return [f for f in findings if not f.passed]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("evidence", [0, 2])
def test_coordinate_counts_and_original_evidence_do_not_require_wide_input(
    backend, evidence, tmp_path, monkeypatch
):
    rows = 257
    bad = [63, 64, 128, 256]
    frame = pl.DataFrame(
        {
            "latitude": [91 if i in bad else 10 for i in range(rows)],
            "longitude": [100] * rows,
            "id": pl.Series(range(2**53 + 1, 2**53 + 1 + rows), dtype=pl.UInt64),
            "payload": [f"row-{i}:" + "x" * 4096 for i in range(rows)],
            "_row": [f"user-{i}" for i in range(rows)],
            "problem": ["user diagnostic"] * rows,
        }
    )
    with input_context(frame, backend, tmp_path, evidence=evidence) as context:
        observed = observe_materialization(context, monkeypatch)
        found = failures(CoordinateRule({"points": [{"dataset": "s"}]}).evaluate(context))
        assert len(found) == 1
        result = found[0]
        assert result.affected_count == len(bad)
        assert result.total_records == rows
        records = result.affected.to_dicts()
        assert len(records) == evidence
        assert [r["id"] for r in records] == [2**53 + 1 + i for i in bad[:evidence]]
        for record, index in zip(records, bad, strict=False):
            assert record["payload"] == f"row-{index}:" + "x" * 4096
            assert record["_row"] == f"user-{index}"
            assert record["problem"] == "user diagnostic"
            assert record["qavf__row"] == index
            assert record["qavf_problem"] == "latitude out of range"
        assert observed, "the actual materialization boundary must be observed"
        assert all(payload_records(value) <= evidence for value in observed)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("evidence", [0, 2])
def test_outlier_projection_preserves_grouped_truth_and_source_row_identity(
    backend, evidence, tmp_path, monkeypatch
):
    rows = 257
    bad = {63: 100.0, 128: 200.0, 256: 300.0}
    frame = pl.DataFrame(
        {
            "value": [bad.get(i, 1.0) for i in range(rows)],
            "group": [i % 2 for i in range(rows)],
            "payload": [f"row-{i}:" + "z" * 4096 for i in range(rows)],
            "__qavf_row": [-1] * rows,
        }
    )
    with input_context(frame, backend, tmp_path, evidence=evidence) as context:
        observed = observe_materialization(context, monkeypatch)
        found = failures(
            ColumnOutlierRule(
                {"targets": [{"dataset": "s", "value": "value", "group_by": ["group"]}]}
            ).evaluate(context)
        )
        assert len(found) == 1
        result = found[0]
        assert result.affected_count == len(bad)
        assert result.total_records == rows
        assert len(result.affected) == evidence
        for record in result.affected:
            assert "payload" not in record
            row_fields = [key for key in record if key.endswith("_row")]
            assert len(row_fields) == 1
            index = record[row_fields[0]]
            assert index in bad
            assert record["value"] == bad[index]
            assert record["group"] == index % 2
        assert observed, "the actual materialization boundary must be observed"
        assert all(payload_records(value) == 0 for value in observed)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rule_type", [CoordinateRule, ColumnOutlierRule])
def test_source_row_limit_is_not_bypassed_by_small_rule_output(backend, rule_type, tmp_path):
    frame = pl.DataFrame(
        {"latitude": [10] * 33, "longitude": [100] * 33, "value": [1.0] * 32 + [100.0]}
    )
    params = (
        {"points": [{"dataset": "s"}]}
        if rule_type is CoordinateRule
        else {"targets": [{"dataset": "s", "value": "value"}]}
    )
    with input_context(frame, backend, tmp_path, evidence=0, row_budget=32) as context:
        found = rule_type(params).evaluate(context)
    assert any(not result.assessed and "row_budget" in result.message for result in found)
    assert not any(result.assessed and result.passed for result in found)


@pytest.mark.parametrize("backend", BACKENDS)
def test_coordinate_decimal_boundary_and_missing_marker_keep_raw_evidence(backend, tmp_path):
    frame = pl.DataFrame(
        {
            "latitude": pl.Series(
                [Decimal("90"), Decimal("90.000000000000000001"), Decimal("-999")],
                dtype=pl.Decimal(38, 18),
            ),
            "longitude": [100, 100, 100],
        }
    )
    schema = {"columns": {"latitude": {"missing_values": [-999], "nullable": True}}}
    with input_context(frame, backend, tmp_path, evidence=3, schema=schema) as context:
        found = failures(CoordinateRule({"points": [{"dataset": "s"}]}).evaluate(context))
    assert len(found) == 1
    assert found[0].affected_count == 2
    assert [
        {key: value for key, value in record.items() if key != "typed_only"}
        for record in found[0].affected.to_dicts()
    ] == [
        {
            **frame.row(1, named=True),
            "_row": 1,
            "problem": "latitude out of range",
        },
        {
            **frame.row(2, named=True),
            "_row": 2,
            "problem": "missing coordinate",
        },
    ]


@pytest.mark.parametrize("rule_type", [CoordinateRule, ColumnOutlierRule])
def test_rule_count_and_evidence_consume_one_lazy_observation(rule_type):
    """A second positional read would observe a different row order here."""
    frame = pl.DataFrame(
        {
            "latitude": [10] * 31 + [91],
            "longitude": [100] * 32,
            "value": [1.0] * 31 + [100.0],
            "payload": [f"row-{i}" for i in range(32)],
        }
    )
    calls = []

    def changing_order(observation):
        calls.append(observation.height)
        return observation if len(calls) == 1 else observation.reverse()

    lazy = frame.lazy().map_batches(changing_order, streamable=False)
    params = (
        {"points": [{"dataset": "s"}]}
        if rule_type is CoordinateRule
        else {"targets": [{"dataset": "s", "value": "value"}]}
    )
    with DataContext(frames={"s": lazy}, evidence_limit=1) as context:
        found = failures(rule_type(params).evaluate(context))
    assert calls == [32]
    assert len(found) == 1 and found[0].affected_count == 1
    if rule_type is CoordinateRule:
        record = found[0].affected.row(0, named=True)
        assert record["_row"] == 31 and record["payload"] == "row-31"
    else:
        record = found[0].affected[0]
        row_fields = [key for key in record if key.endswith("_row")]
        assert len(row_fields) == 1 and record[row_fields[0]] == 31
        assert record["value"] == 100.0


@pytest.mark.parametrize("rule_type", [CoordinateRule, ColumnOutlierRule])
def test_zero_evidence_projects_unused_payload_before_lazy_execution(rule_type):
    frame = pl.DataFrame(
        {
            "latitude": [10] * 31 + [91],
            "longitude": [100] * 32,
            "value": [1.0] * 31 + [100.0],
            "payload": ["unnecessary"] * 32,
        }
    )
    seen_columns = []

    def observe_projection(observation):
        seen_columns.append(set(observation.columns))
        return observation

    lazy = frame.lazy().map_batches(
        observe_projection,
        projection_pushdown=True,
        validate_output_schema=False,
        streamable=False,
    )
    params = (
        {"points": [{"dataset": "s"}]}
        if rule_type is CoordinateRule
        else {"targets": [{"dataset": "s", "value": "value"}]}
    )
    with DataContext(frames={"s": lazy}, evidence_limit=0) as context:
        found = failures(rule_type(params).evaluate(context))
    assert len(found) == 1 and found[0].affected_count == 1
    assert seen_columns and all("payload" not in columns for columns in seen_columns)
    assert len(found[0].affected) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_outlier_invalid_evidence_preserves_raw_marker_and_observation_index(backend, tmp_path):
    frame = pl.DataFrame(
        {
            "value": [1.0] * 15 + [-999.0],
            "group": ["a"] * 16,
            "payload": ["unrelated"] * 16,
        }
    )
    schema = {"columns": {"value": {"missing_values": [-999], "nullable": True}}}
    with input_context(frame, backend, tmp_path, schema=schema, evidence=1) as context:
        found = ColumnOutlierRule(
            {"targets": [{"dataset": "s", "value": "value", "group_by": ["group"]}]}
        ).evaluate(context)
    invalid = [
        result for result in found if (result.check or "").endswith(": invalid observations")
    ]
    assert len(invalid) == 1
    assert not invalid[0].assessed and invalid[0].affected_count == 1
    record = invalid[0].affected.row(0, named=True)
    assert record["value"] == -999.0 and record["group"] == "a"
    assert "payload" not in record
    row_fields = [key for key in record if key.endswith("_row")]
    assert len(row_fields) == 1 and record[row_fields[0]] == 15
