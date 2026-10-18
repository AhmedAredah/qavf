"""Combined key and whole-row checks share a single volatile observation."""

import duckdb
import polars as pl
import pytest

from qavf import Status, validate
from qavf.engine.context import DataContext
from qavf.rules.input_checks.keys import DuplicateKeyRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.parameters import QueryParams
from qavf.sources.query import QuerySource

CONTRACT = {"primary_key": ["id"], "unique_keys": [["alternate"]], "unique_rows": True}


def config(tmp_path, *, spill=False):
    return {
        "plugins": {"entry_points": False},
        "engine": {
            "parallel": False,
            "run_unconfigured_rules": False,
            "max_affected_records": 2,
            "temp_directory": str(tmp_path / "owned"),
            **({"cache_max_bytes": 0} if spill else {}),
        },
        "datasets": {"data": {"schema": CONTRACT}},
        "rules": {"INP-004": {}},
    }


@pytest.mark.parametrize("spill", [False, True])
@pytest.mark.parametrize("duplicate_first", [False, True])
def test_lazy_primary_alternate_and_whole_rows_share_one_observation(
    tmp_path, spill, duplicate_first
):
    calls = []
    distinct = pl.DataFrame({"id": [1, 2], "alternate": [10, 20], "payload": ["x" * 4096] * 2})
    duplicates = distinct.head(1).vstack(distinct.head(1))

    def observe(_batch):
        calls.append(True)
        is_duplicate = duplicate_first if len(calls) == 1 else not duplicate_first
        return duplicates if is_duplicate else distinct

    report = validate(
        distinct.lazy().map_batches(observe, schema=distinct.schema),
        config=config(tmp_path, spill=spill),
    )
    results = report.results_for("INP-004")
    assert len(calls) == 1
    assert all(result.status in (Status.PASS, Status.FAIL) for result in results)
    if not duplicate_first:
        assert all(result.status is Status.PASS for result in results)
        return
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    assert set(failures) == {"data", "data: unique (alternate)", "data: duplicate rows"}
    for name in ("data", "data: unique (alternate)"):
        assert failures[name].affected_count == 1
        assert failures[name].metrics["rows_involved"] == 2
        assert len(failures[name].affected_records) == 1
    whole = failures["data: duplicate rows"]
    assert (whole.affected_count, whole.total_records) == (2, 2)
    assert [record["_row"] for record in whole.affected_records] == [0, 1]
    assert all(record["payload"] == "x" * 4096 for record in whole.affected_records)


def test_native_query_keys_and_whole_rows_share_one_observation(tmp_path, monkeypatch):
    observations = []
    original_connect = duckdb.connect

    def observed():
        observations.append(True)
        return len(observations)

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.create_function("observed", observed, [], "BIGINT", side_effects=True)
        return connection

    # Install a deterministic volatile SQL function on every real native connection.
    # Each two-row observation increments twice. Establish schema separately because
    # DuckDB 1.2 also evaluates Python UDFs during its schema-only binding operation.
    monkeypatch.setattr(duckdb, "connect", connect)
    source = QuerySource(
        QueryParams(
            sql="SELECT CASE WHEN observed()<=2 THEN i ELSE 1 END AS id, 'wide' AS payload "
            "FROM range(2) t(i)"
        ),
        SourceEnvironment(),
    )
    with DataContext(
        sources={"data": source},
        specs={
            "data": DatasetSpec.model_validate(
                {"schema": {"primary_key": ["id"], "unique_rows": True}}
            )
        },
        temp_directory=tmp_path / "owned",
    ) as context:
        assert context.has("data")
        observations.clear()
        findings = DuplicateKeyRule({}).evaluate(context)
        assert all(finding.assessed and finding.passed for finding in findings)
        assert len(observations) == 2


@pytest.mark.parametrize("lazy_snapshot", [False, True])
def test_existing_spill_uses_key_projections_and_scoped_row_reduction(
    tmp_path, monkeypatch, lazy_snapshot
):
    frame = pl.DataFrame({"id": [1, 2], "alternate": [10, 20], "payload": ["x" * 4096] * 2})
    observations = []

    def observe(batch):
        observations.append(True)
        return batch if len(observations) == 1 else batch.head(1).vstack(batch.head(1))

    supplied = frame.lazy().map_batches(observe, schema=frame.schema) if lazy_snapshot else frame
    with DataContext(
        frames={"data": supplied},
        specs={"data": DatasetSpec.model_validate({"schema": CONTRACT})},
        cache_max_bytes=0,
        temp_directory=tmp_path / "owned",
    ) as context:
        if lazy_snapshot:
            context.get("data")
        assert context.is_materialized("data")
        assert context.resource_usage()["frame_spill_bytes"] > 0
        projected = []
        original = context.project

        def project(name, columns):
            projected.append(list(columns))
            return original(name, columns)

        def no_full_reload(*args, **kwargs):
            pytest.fail("an existing IPC snapshot must not be fully reloaded")

        monkeypatch.setattr(context, "get", no_full_reload)
        monkeypatch.setattr(context, "project", project)
        findings = DuplicateKeyRule({}).evaluate(context)
        assert all(finding.assessed and finding.passed for finding in findings)
        assert projected == [["id"], ["alternate"]]
        assert len(observations) == int(lazy_snapshot)
