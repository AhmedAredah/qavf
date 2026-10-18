"""Source scan counts, exact decisions and independent failure boundaries."""

from contextlib import contextmanager

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.input_checks import ColumnCheckRule
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError


@contextmanager
def context(frame, columns, backend, tmp_path, *, evidence_limit=0):
    dataset = {"schema": {"columns": columns}}
    frames = {"d": frame.lazy()}
    if backend == "native":
        path = tmp_path / "batch.parquet"
        frame.write_parquet(path)
        dataset["source"] = {"type": "file", "path": str(path)}
        frames = {}
    settings = QAVFSettings.model_validate(
        {"datasets": {"d": dataset}, "engine": {"max_affected_records": evidence_limit}}
    )
    with DataContext.from_settings(settings, frames=frames) as ctx:
        ctx.schema("d")
        yield ctx


def count_reads(ctx, backend, monkeypatch):
    reads = []
    method = "collect" if backend == "lazy" else "sql"
    execute = getattr(ctx, method)

    def counted(*args, **kwargs):
        reads.append(args)
        return execute(*args, **kwargs)

    monkeypatch.setattr(ctx, method, counted)
    monkeypatch.setattr(ctx, "get", lambda *args: pytest.fail("full dataset materialization"))
    return reads


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_mixed_contracts_share_one_source_reduction(backend, tmp_path, monkeypatch):
    frame = pl.DataFrame({"x": [1, 2, 3], "s": ["A", "B", "C"], "scope": [1, 1, 0]})
    columns = {
        "x": {
            "missing_values": [-999],
            "checks": [
                {"name": "range", "ge": 0, "le": 10},
                {"name": "membership", "isin": [1, 2, 3]},
                {"name": "unique", "unique": True},
                {"name": "selected", "where": "scope = 1", "ge": 0},
                {"name": "constant", "where": "TRUE", "ge": 0, "min_applicable_rows": 3},
            ],
        },
        "s": {"regex": "[A-Z]", "min_length": 1, "max_length": 1, "blank_as_missing": True},
    }
    with context(frame, columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        scans = []
        scan = ctx.scan

        def counted_scan(*args, **kwargs):
            scans.append(args)
            return scan(*args, **kwargs)

        monkeypatch.setattr(ctx, "scan", counted_scan)
        found = ColumnCheckRule().evaluate(ctx)
        assert all(f.passed for f in found), [f.message for f in found]
        assert len(reads) == 1
        assert len(scans) == (1 if backend == "lazy" else 0)
        assert not ctx.is_materialized("d")


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_large_contract_set_has_bounded_batch_width(backend, tmp_path, monkeypatch):
    columns = {
        "x": {
            "nullable": True,
            "missing_values": [-999],
            "checks": [{"name": f"check {i}", "ge": i - 129} for i in range(129)],
        }
    }
    with context(pl.DataFrame({"x": [0]}), columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        assert all(f.passed for f in ColumnCheckRule().evaluate(ctx))
        assert len(reads) == 3


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("size", [0, 3])
@pytest.mark.parametrize("scope,expected", [("TRUE", 3), ("FALSE", 0), ("NULL", 0)])
def test_constant_scope_counts_without_an_extra_scan(
    backend, size, scope, expected, tmp_path, monkeypatch
):
    columns = {"x": {"checks": [{"name": "scoped", "where": scope, "lt": 0}]}}
    with context(pl.DataFrame({"x": range(size)}), columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        found = ColumnCheckRule().evaluate(ctx)
        failing = [finding for finding in found if not finding.passed]
        assert all(f.assessed for f in found), [f.message for f in found]
        assert sum(f.affected_count or 0 for f in failing) == (expected if size else 0)
        assert len(reads) == 1


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_invalid_contract_isolated_before_shared_source_read(backend, tmp_path, monkeypatch):
    columns = {
        "x": {
            "nullable": True,
            "checks": [
                {"name": "before", "ge": 2},
                {"name": "invalid type", "regex": "a"},
                {"name": "after", "le": 0},
            ],
        }
    }
    with context(pl.DataFrame({"x": [1]}), columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        found = ColumnCheckRule().evaluate(ctx)
        assert [(f.check, f.assessed) for f in found] == [
            ("d.x: before", True),
            ("d.x: invalid type", False),
            ("d.x: after", True),
        ]
        assert [f.affected_count for f in found if f.assessed] == [1, 1]
        assert len(reads) == 1


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_data_dependent_scope_failure_cannot_poison_shared_checks(backend, tmp_path, monkeypatch):
    columns = {
        "x": {
            "nullable": True,
            "checks": [
                {"name": "before", "ge": 2},
                {"name": "unsafe cast", "where": "CAST(s AS INTEGER) > 0", "ge": 0},
                {"name": "after", "le": 0},
            ],
        }
    }
    frame = pl.DataFrame({"x": [1], "s": ["not a number"]})
    with context(frame, columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        found = ColumnCheckRule().evaluate(ctx)
        assert [(f.check, f.assessed) for f in found] == [
            ("d.x: before", True),
            ("d.x: unsafe cast", False),
            ("d.x: after", True),
        ]
        assert len(reads) == 2


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_shared_read_failure_is_not_retried(backend, tmp_path, monkeypatch):
    columns = {"x": {"missing_values": [-999], "ge": 0, "unique": True}}
    with context(pl.DataFrame({"x": [1]}), columns, backend, tmp_path) as ctx:
        calls = []

        def failed(*args, **kwargs):
            calls.append(args)
            raise SourceError("non-retryable source read")

        monkeypatch.setattr(ctx, "collect" if backend == "lazy" else "sql", failed)
        found = ColumnCheckRule().evaluate(ctx)
        assert len(found) == 2
        assert all(not f.assessed for f in found)
        assert len(calls) == 1


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_raw_missing_evidence_and_aliases_survive_fusion(backend, tmp_path):
    frame = pl.DataFrame(
        {
            "x": [-999.0, float("nan"), 3.0],
            "s": [" ", "A", "B"],
            "scope": [-1, 1, 1],
            "__QAVF_ROW": ["user row"] * 3,
            "__QAVF_ORIGINAL_X": ["user original"] * 3,
            "__QAVF_VIOLATES": ["user reason"] * 3,
        }
    )
    columns = {
        "x": {
            "missing_values": [-999],
            "checks": [
                {"name": "NaN", "notin": [float("nan")]},
                {"name": "scoped", "where": "scope = 1", "ge": 5, "severity": "warning"},
            ],
        },
        "s": {"nullable": True, "blank_as_missing": True, "regex": "[A-Z]"},
        "scope": {"nullable": True, "missing_values": [-1]},
    }
    with context(frame, columns, backend, tmp_path, evidence_limit=1) as ctx:
        found = [f for f in ColumnCheckRule().evaluate(ctx) if not f.passed]
        assert all(f.assessed for f in found), [f.message for f in found]
        assert [(f.check, f.affected_count) for f in found] == [
            ("d.x: not null", 2),
            ("d.x: NaN", 1),
            ("d.x: scoped", 1),
        ]
        assert str(found[-1].severity).lower().endswith("warning")
        for finding in found:
            assert isinstance(finding.affected, pl.DataFrame)
            assert finding.affected.height == 1
            assert finding.affected["__QAVF_ROW"].to_list() == ["user row"]
            assert finding.affected["__QAVF_ORIGINAL_X"].to_list() == ["user original"]
            assert finding.affected["__QAVF_VIOLATES"].to_list() == ["user reason"]
        assert found[0].affected["x"].to_list() == [-999.0]
        assert found[1].affected["x"].is_nan().all()


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_scalar_statistics_batch_without_materializing_a_series(backend, tmp_path, monkeypatch):
    columns = {
        "x": {
            "checks": [
                {"name": "minimum", "min": {"ge": 2**53 + 1, "le": 2**53 + 1}},
                {"name": "maximum", "max": {"ge": 2**53 + 3, "le": 2**53 + 3}},
                {"name": "distinct", "distinct_count": {"ge": 3, "le": 3}},
            ]
        }
    }
    frame = pl.DataFrame({"x": [2**53 + 1, 2**53 + 2, 2**53 + 3]})
    with context(frame, columns, backend, tmp_path) as ctx:
        reads = count_reads(ctx, backend, monkeypatch)
        monkeypatch.setattr(
            "qavf.rules.input_checks._column_values._statistic",
            lambda *args: pytest.fail("materialized a full series for a scalar statistic"),
        )
        found = ColumnCheckRule().evaluate(ctx)
        assert all(f.passed for f in found), [f.message for f in found]
        assert len(reads) == 1


def test_native_regex_binding_failure_isolated_without_reading_source(tmp_path, monkeypatch):
    # Engine-specific regexes now stream through the canonical Rust evaluator;
    # the sibling SQL batch still executes once without replay/materialization.
    columns = {
        "s": {
            "nullable": True,
            "checks": [
                {"name": "before", "isin": ["B"]},
                # Valid Rust regex, unsupported by DuckDB's RE2 backend.
                {"name": "unsupported regex", "regex": r"\u{41}"},
                {"name": "after", "min_length": 2},
            ],
        }
    }
    with context(pl.DataFrame({"s": ["A"]}), columns, "native", tmp_path) as ctx:
        reads = count_reads(ctx, "native", monkeypatch)
        found = ColumnCheckRule().evaluate(ctx)
        assert [(f.check, f.assessed) for f in found] == [
            ("d.s: before", True),
            ("d.s: after", True),
            (None, True),
        ]
        assert found[-1].metrics["passed_checks"] == ["d.s: unsupported regex"]
        assert len(reads) == 1
