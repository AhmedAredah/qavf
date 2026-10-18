"""Public abstraction parity, coherent exact typed batches and bounded reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from qavf import Dataset, Loader, Query, ResourceLimits
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.resources import ResourceLimitError, RunResources
from qavf.engine.sqlite_store import Store
from qavf.keys import KeyTypeError
from qavf.redaction import redact
from qavf.regression import compare_frames, compare_outputs
from qavf.reports.html import iter_html, render_html, write_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata, SnapshotModel
from qavf.reports.presentation import HtmlLimits
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("descriptor", ["query", "loader"])
@pytest.mark.parametrize("mode", ["default", "resources", "dataset"])
def test_descriptor_comparison_does_not_depend_on_unrelated_options(descriptor, mode):
    frame = pl.DataFrame({"id": [1, 2], "value": [10, 20]})
    source = (
        Query("SELECT * FROM (VALUES (1,10),(2,20)) AS t(id,value)")
        if descriptor == "query"
        else Loader(lambda: frame)
    )
    result = compare_outputs(
        Dataset(source) if mode == "dataset" else source,
        frame,
        "id",
        resources=ResourceLimits() if mode == "resources" else None,
    )
    assert result.equal and result.execution == "typed_stream"


@pytest.mark.parametrize("container", [set, frozenset, tuple, list, np.array])
def test_redaction_matches_all_audit_container_types(container):
    value = container(["password=PRIORITY_FAKE_SECRET"])
    assert "PRIORITY_FAKE_SECRET" not in str(redact({"options": value}))


def test_nested_supported_secret_cannot_reappear_in_final_provenance():
    from qavf.engine.runner import QARunner

    class Simple(Rule):
        rule_id = "TEST-SECRET"
        name = description = category = "test"

        def evaluate(self, ctx):
            return Finding.ok("ok")

    registry = RuleRegistry()
    registry.register(Simple)
    settings = QAVFSettings(
        plugins={"entry_points": False},
        extensions={"example": {"options": {"password=PRIORITY_FAKE_SECRET"}}},
    )
    report = QARunner(settings, registry).run()
    assert "PRIORITY_FAKE_SECRET" not in report.model_dump_json()
    assert "PRIORITY_FAKE_SECRET" not in render_html(report)


def test_redaction_rejects_cycles_and_ambiguous_mapping_keys():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="cycles"):
        redact(cycle)
    with pytest.raises(ValueError, match="colliding"):
        redact({1: "ordinary", "1": "password=secret"})


def report(records):
    now = datetime.now(UTC)
    return QAReport(
        metadata=RunMetadata(
            run_id="test", project="test", qavf_version="test", started_at=now, finished_at=now
        ),
        results=[
            RuleResult(
                rule_id="T",
                name="test",
                description="test",
                category="test",
                status="FAIL",
                severity="ERROR",
                message="failed",
                affected_records=records,
                affected_count=len(records),
            )
        ],
    )


def test_sparse_html_scales_with_record_entries_and_streams_macroless_rows(tmp_path, monkeypatch):
    value = report([{f"field_{i}": i} for i in range(600)])

    def forbidden(*args, **kwargs):
        raise AssertionError("HTML must not dump a whole model graph")

    monkeypatch.setattr(SnapshotModel, "model_dump", forbidden)
    target = write_html(value, tmp_path / "report.html")
    assert target.stat().st_size < 150_000
    html = target.read_text()
    assert html.count("<td>") < 2000
    assert "field_599" in html
    assert max(map(len, iter_html(value))) < 100_000


def test_html_cell_and_byte_limits_disclose_omissions_and_keep_destination_atomic(tmp_path):
    value = report([{"a": i, "b": i, "c": i} for i in range(100)])
    html = render_html(value, limits=HtmlLimits(max_cells=30))
    assert "HTML evidence was shortened" in html
    target = tmp_path / "report.html"
    target.write_text("previous complete report")
    with pytest.raises(ResourceLimitError, match="HTML report"):
        write_html(value, target, limits=HtmlLimits(max_output_bytes=128))
    assert target.read_text() == "previous complete report"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("kind", ["list", "struct", "datetime", "duration", "int128", "decimal"])
def test_typed_files_compare_exactly_without_whole_projection(tmp_path, monkeypatch, kind):
    values = {
        "list": (pl.List(pl.Int128), [[1, 2], [3]], [[1, 3], [3]]),
        "struct": (
            pl.Struct({"stamp": pl.Datetime("ns")}),
            [{"stamp": 1001}, {"stamp": None}],
            [{"stamp": 1002}, {"stamp": None}],
        ),
        "datetime": (pl.Datetime("ns", "UTC"), [1001, 2000], [1002, 2000]),
        "duration": (pl.Duration("ns"), [1001, 2000], [1002, 2000]),
        "int128": (pl.Int128, [2**100, 2**100 + 2], [2**100 + 1, 2**100 + 2]),
        "decimal": (
            pl.List(pl.Decimal(38, 18)),
            [[Decimal("0.100000000000000001")], []],
            [[Decimal("0.100000000000000002")], []],
        ),
    }
    dtype, before, after = values[kind]
    left = pl.DataFrame({"id": [1, 2], "value": pl.Series(before, dtype=dtype)})
    right = pl.DataFrame({"id": [1, 2], "value": pl.Series(after, dtype=dtype)}).reverse()
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    left.write_parquet(a)
    right.write_parquet(b)

    def forbidden(*args, **kwargs):
        raise AssertionError("typed comparison must not materialize full projections")

    monkeypatch.setattr(DataContext, "project", forbidden)
    result = compare_outputs(a, b, "id", rtol=0, atol=0, name="same")
    expected = compare_frames(left, right, "id", rtol=0, atol=0, name="same")
    excluded = {"execution", "fallback_reason"}
    assert result.model_dump(exclude=excluded) == expected.model_dump(exclude=excluded)
    assert result.execution == ("native_stream" if kind == "decimal" else "typed_stream")


def test_indexed_typed_alignment_keeps_null_keys_and_counts_duplicates(tmp_path):
    frame = pl.DataFrame({"id": [None, 1, 2], "value": [[1], [2], [3]]})
    assert compare_outputs(frame.lazy(), frame.reverse().lazy(), "id").equal
    duplicated = frame.vstack(frame.head(1))
    with pytest.raises(ValueError, match=r"2 row.*duplicate key"):
        compare_outputs(duplicated.lazy(), frame.lazy(), "id")


def test_typed_alignment_preserves_lossless_mixed_numeric_key_policy():
    left = pl.DataFrame({"id": pl.Series([1, 2], dtype=pl.Int32), "value": [[1], [2]]})
    right = pl.DataFrame({"id": [2.0, 1.0], "value": [[2], [1]]})
    assert compare_outputs(left.lazy(), right.lazy(), "id").equal
    large = pl.DataFrame({"id": [2**54], "value": [[1]]})
    floated = large.with_columns(pl.col("id").cast(pl.Float64))
    with pytest.raises(KeyTypeError, match="cannot be compared exactly"):
        compare_outputs(large.lazy(), floated.lazy(), "id")


def test_typed_spill_failure_removes_all_owned_files(tmp_path):
    left = pl.DataFrame({"id": [1], "value": pl.Series([1001], dtype=pl.Datetime("ns"))})
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    left.write_parquet(a)
    left.write_parquet(b)
    storage = tmp_path / "work"
    with pytest.raises(ResourceLimitError, match="spill budget"):
        compare_outputs(a, b, "id", resources=ResourceLimits(max_spill=0, temp_dir=storage))
    assert list(storage.iterdir()) == []


def test_shared_work_store_caps_disk_and_rejects_unindexed_sort(tmp_path):
    resources = RunResources(0, 131_072, None, tmp_path)
    try:
        with Store(resources, "first") as first, Store(resources, "second") as second:
            for store in (first, second):
                store.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value BLOB)")
            first.executemany(
                "INSERT INTO items VALUES (?,?)", [(i, b"a" * 1024) for i in range(8)]
            )
            with pytest.raises(ResourceLimitError, match="temporary B-tree"):
                first.execute("SELECT value FROM items ORDER BY value")
            with pytest.raises(ResourceLimitError, match="spill budget"):
                second.executemany(
                    "INSERT INTO items VALUES (?,?)", [(i, b"b" * 4096) for i in range(100)]
                )
            assert resources.spill_budget.used <= resources.spill_budget.limit
        assert resources.spill_budget.used == 0
    finally:
        resources.close()
    assert list(tmp_path.iterdir()) == []
