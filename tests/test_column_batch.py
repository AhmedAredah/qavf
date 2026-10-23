"""One-observation, typed evidence and fallback oracles for compact column checks."""

from decimal import Decimal

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.input_checks import ColumnCheckRule
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError


def context(frame, columns, *, evidence=3, primary_key=None):
    settings = QAVFSettings.model_validate(
        {
            "datasets": {"d": {"schema": {"columns": columns, "primary_key": primary_key or []}}},
            "engine": {"max_affected_records": evidence},
        }
    )
    return DataContext.from_settings(settings, frames={"d": frame})


def changing(first, second, calls):
    def read(_):
        calls.append(1)
        return first if len(calls) == 1 else second

    return first.lazy().map_batches(read, schema=first.schema)


@pytest.mark.parametrize("evidence", [0, 1, 5])
def test_compact_checks_share_source_counts_and_raw_evidence(evidence, monkeypatch):
    first = pl.DataFrame({"id": [9, 8, 7, 6], "x": [-999, -2, None, 4], "s": ["A", "b", "C", "d"]})
    second = first.with_columns(pl.lit(0).alias("x"), pl.lit("A").alias("s"))
    calls = []
    columns = {
        "x": {"missing_values": [-999], "checks": [{"name": "positive", "ge": 0}]},
        "s": {"regex": "[A-Z]"},
    }
    with context(
        changing(first, second, calls), columns, evidence=evidence, primary_key=["id"]
    ) as ctx:
        monkeypatch.setattr(ctx, "observe", lambda *_: pytest.fail("unexpected full observation"))
        found = [f for f in ColumnCheckRule().evaluate(ctx) if not f.passed]
        assert calls == [1]
        assert [(f.check, f.affected_count, f.total_records) for f in found] == [
            ("d.x: not null", 2, 4),
            ("d.x: positive", 1, 4),
            ("d.s: regex [A-Z]", 2, 4),
        ]
        assert all(f.assessed for f in found)
        assert ctx.resource_usage()["peak_retained_total_bytes"] == 0
        if evidence:
            assert found[0].affected["id"].to_list() == [9, 7][:evidence]
            assert found[0].affected["x"].to_list() == [-999, None][:evidence]
            assert found[2].affected["_row"].to_list() == [1, 3][:evidence]
        else:
            assert all(f.affected is None for f in found)


@pytest.mark.parametrize("unsupported", ["global", "scope", "wide"])
def test_snapshot_fallback_still_observes_volatile_source_once(unsupported, monkeypatch):
    first, second = pl.DataFrame({"x": [-1, -1]}), pl.DataFrame({"x": [2, 3]})
    checks = [{"name": "positive", "ge": 0}]
    if unsupported == "global":
        checks.append({"name": "unique", "unique": True})
    elif unsupported == "scope":
        checks.append({"name": "scope", "where": "abs(x) > 0", "le": 0})
    else:
        checks = [{"name": f"check {index}", "ge": 0} for index in range(65)]
    calls, observations = [], []
    with context(
        changing(first, second, calls), {"x": {"nullable": True, "checks": checks}}
    ) as ctx:
        observe = ctx.observe

        def observed(*args, **kwargs):
            observations.append(1)
            return observe(*args, **kwargs)

        monkeypatch.setattr(ctx, "observe", observed)
        found = ColumnCheckRule().evaluate(ctx)
        assert calls == [1]
        assert observations == [1]
        assert all(f.assessed for f in found)
        assert found[0].affected_count == 2
        if unsupported == "wide":
            assert len(found) == 65
            assert all(f.affected_count == 2 for f in found)
        elif unsupported == "global":
            assert len(found) == 2 and "not unique" in found[1].message


@pytest.mark.parametrize("size", [0, 4])
def test_independent_preflight_errors_and_empty_scope_policy(size):
    frame = pl.DataFrame(
        {"x": [1] * size, "s": ["A"] * size}, schema={"x": pl.Int64, "s": pl.String}
    )
    columns = {
        "missing": {},
        "x": {
            "nullable": True,
            "checks": [
                {"name": "lower", "ge": 2},
                {"name": "bad type", "regex": "a"},
                {"name": "scope", "where": "FALSE", "le": 0, "min_applicable_rows": 1},
            ],
        },
        "s": {
            "nullable": True,
            "checks": [{"name": "missing dependency", "where": "absent IS NULL", "regex": "[A-Z]"}],
        },
    }
    with context(frame.lazy(), columns) as ctx:
        found = ColumnCheckRule().evaluate(ctx)
    by_check = {f.check: f for f in found}
    assert not by_check["d.x: bad type"].assessed
    assert not by_check["d.s: missing dependency"].assessed
    assert not by_check["d.x: scope"].assessed
    assert "min_applicable_rows" in by_check["d.x: scope"].message
    if size:
        assert by_check["d.x: lower"].affected_count == size
    else:
        assert "d.x: lower" not in by_check
    assert any("does not exist" in f.message for f in found)


def test_compact_source_failure_is_not_retried(monkeypatch):
    calls = []
    with context(pl.DataFrame({"x": [1]}).lazy(), {"x": {"ge": 0, "le": 4}}) as ctx:

        def fail(*_):
            calls.append(1)
            raise SourceError("controlled unavailable input")

        monkeypatch.setattr(ctx, "collect", fail)
        found = ColumnCheckRule().evaluate(ctx)
    assert calls == [1]
    assert len(found) == 2
    assert all(not f.assessed for f in found)
    assert all("controlled unavailable input" in f.message for f in found)


def test_compact_typed_evidence_does_not_round_trip_through_python():
    frame = pl.DataFrame(
        {
            "id": pl.Series(
                [1_700_000_000_000_000_001, 1_700_000_000_000_000_002], dtype=pl.Datetime("ns")
            ),
            "x": pl.Series(
                [Decimal("90.000000000000000001"), Decimal("90.000000000000000002")],
                dtype=pl.Decimal(38, 18),
            ),
            "big": pl.Series([2**100 + 1, 2**100 + 2], dtype=pl.Int128),
        }
    )
    columns = {"x": {"le": 90}, "big": {"le": 2**100}}
    with context(frame.lazy(), columns, primary_key=["id"]) as ctx:
        found = [f for f in ColumnCheckRule().evaluate(ctx) if not f.passed]
    assert all(f.assessed and f.affected_count == 2 for f in found)
    for finding in found:
        assert finding.affected["id"].to_list() == [
            "2023-11-14T22:13:20.000000001",
            "2023-11-14T22:13:20.000000002",
        ]
        assert finding.affected["x"].to_list() == frame["x"].to_list()
        assert finding.affected["big"].to_list() == [2**100 + 1, 2**100 + 2]


@pytest.mark.parametrize("limit", [0, 7])
def test_multi_batch_counts_and_first_rows_match_snapshot(tmp_path, limit):
    size = 150_003
    frame = pl.DataFrame({"id": range(size), "x": [(-1 if i % 3 else 2) for i in range(size)]})
    path = tmp_path / "rows.parquet"
    frame.write_parquet(path, row_group_size=2000)
    columns = {"x": {"nullable": True, "checks": [{"name": "positive", "ge": 0}]}}
    with context(pl.scan_parquet(path), columns, evidence=limit, primary_key=["id"]) as ctx:
        found = ColumnCheckRule().evaluate(ctx)
    assert len(found) == 1 and found[0].assessed
    assert found[0].affected_count == 100_002
    assert found[0].total_records == size
    if limit:
        assert found[0].affected["id"].to_list() == [1, 2, 4, 5, 7, 8, 10]
        assert found[0].affected["_row"].to_list() == [1, 2, 4, 5, 7, 8, 10]


def test_row_scope_units_and_null_fraction_reuse_canonical_policy():
    frame = pl.DataFrame(
        {"x": [-99.0, float("nan"), 3.0, 4.0], "u": ["kg", "kg", "lb", "kg"], "scope": [1, 1, 1, 0]}
    )
    columns = {
        "x": {
            "nullable": True,
            "missing_values": [-99],
            "unit": "kg",
            "unit_column": "u",
            "checks": [
                {"name": "missing", "where": "scope = 1", "max_null_fraction": 0.5},
                {"name": "forbidden", "notin": [float("nan")]},
            ],
        }
    }
    with context(frame.lazy(), columns) as ctx:
        found = [f for f in ColumnCheckRule().evaluate(ctx) if not f.passed]
    assert all(f.assessed for f in found), [f.message for f in found]
    assert [(f.check, f.affected_count) for f in found] == [
        ("d.x: units", 1),
        ("d.x: missing", None),
        ("d.x: forbidden", 1),
    ]
    assert found[0].affected["_row"].to_list() == [2]
    assert "missing fraction 0.6667" in found[1].message


@pytest.mark.parametrize("checks_count,observations", [(64, 0), (65, 1)])
def test_exact_compact_width_threshold(checks_count, observations, monkeypatch):
    calls, captures = [], []
    frame = pl.DataFrame({"x": [-1]})
    columns = {
        "x": {
            "nullable": True,
            "checks": [{"name": f"limit {i}", "ge": 0} for i in range(checks_count)],
        }
    }
    with context(changing(frame, pl.DataFrame({"x": [1]}), calls), columns, evidence=0) as ctx:
        observe = ctx.observe

        def capture(*args, **kwargs):
            captures.append(1)
            return observe(*args, **kwargs)

        monkeypatch.setattr(ctx, "observe", capture)
        found = ColumnCheckRule().evaluate(ctx)
    assert calls == [1]
    assert len(captures) == observations
    assert len(found) == checks_count
    assert all(f.assessed and f.affected_count == 1 for f in found)


def test_no_contracts_never_evaluate_source():
    calls = []
    frame = pl.DataFrame({"x": [1]})
    with context(changing(frame, frame, calls), {"x": {"nullable": True}}) as ctx:
        found = ColumnCheckRule().evaluate(ctx)
    assert calls == []
    assert len(found) == 1 and found[0].skipped


def test_late_source_failure_discards_counts_and_evidence_without_replay(monkeypatch):
    calls, collections = [], []
    frame = pl.DataFrame({"x": [-1, -2]})
    with context(changing(frame, frame, calls), {"x": {"ge": 0}}) as ctx:
        collect = ctx.collect

        def fail_after_read(*args, **kwargs):
            collections.append(1)
            collect(*args, **kwargs)
            raise SourceError("controlled post-read identity failure")

        monkeypatch.setattr(ctx, "collect", fail_after_read)
        found = ColumnCheckRule().evaluate(ctx)
    assert calls == [1] and collections == [1]
    assert all(not f.assessed and f.affected is None for f in found)
    assert all("post-read identity failure" in f.message for f in found)


def test_native_and_compact_lazy_evidence_have_identical_source_order(tmp_path):
    frame = pl.DataFrame(
        {"id": [7, 6, 5, 4, 3], "x": [-99, -2, 4, -3, None], "s": ["A", "b", "C", "d", "E"]}
    )
    columns = {"x": {"missing_values": [-99], "ge": 0}, "s": {"regex": "[A-Z]"}}
    path = tmp_path / "parity.parquet"
    frame.write_parquet(path)
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "d": {
                    "source": {"type": "file", "path": str(path)},
                    "schema": {"columns": columns, "primary_key": ["id"]},
                }
            },
            "engine": {"max_affected_records": 1},
        }
    )
    with DataContext.from_settings(settings) as native:
        native_findings = [f for f in ColumnCheckRule().evaluate(native) if not f.passed]
    with context(pl.scan_parquet(path), columns, evidence=1, primary_key=["id"]) as lazy:
        lazy_findings = [f for f in ColumnCheckRule().evaluate(lazy) if not f.passed]
    assert len(native_findings) == len(lazy_findings) == 3
    for left, right in zip(native_findings, lazy_findings, strict=True):
        assert left.assessed and right.assessed
        assert (left.check, left.affected_count, left.total_records) == (
            right.check,
            right.affected_count,
            right.total_records,
        )
        assert left.affected.to_dicts() == right.affected.to_dicts()


@pytest.mark.parametrize("evidence", [0, 2])
def test_invalid_sibling_missing_policy_does_not_normalize_raw_evidence(evidence):
    frame = pl.DataFrame({"good": [-1, -2], "bad": [1, 2]})
    columns = {"good": {"nullable": True, "ge": 0}, "bad": {"missing_values": ["oops"]}}
    with context(frame.lazy(), columns, evidence=evidence) as ctx:
        found = ColumnCheckRule().evaluate(ctx)
    assert len(found) == 2
    assert found[0].check == "d.good: ge 0"
    assert found[0].assessed and found[0].affected_count == 2
    assert found[1].check == "d.bad: not null" and not found[1].assessed
    assert "missing marker" in found[1].message
    if evidence:
        assert found[0].affected["good"].to_list() == [-1, -2]
        assert found[0].affected["bad"].to_list() == [1, 2]


@pytest.mark.parametrize("semantic_dependency", [False, True])
def test_primary_key_policy_only_applies_when_semantically_required(semantic_dependency):
    frame = pl.DataFrame({"good": [-1, -2], "id": [1, 2]})
    check = {"name": "positive", "ge": 0}
    if semantic_dependency:
        check["where"] = "id = 1"
    columns = {
        "good": {"nullable": True, "checks": [check]},
        "id": {"missing_values": ["oops"]},
    }
    with context(frame.lazy(), columns, primary_key=["id"]) as ctx:
        found = ColumnCheckRule().evaluate(ctx)
    assert len(found) == 2
    assert not found[1].assessed
    assert found[0].assessed is not semantic_dependency
    if semantic_dependency:
        assert "missing marker" in found[0].message
    else:
        assert found[0].affected_count == 2
        assert found[0].affected["id"].to_list() == [1, 2]
