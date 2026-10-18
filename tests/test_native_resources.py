"""Native resource, lazy provenance, bounded fallback and audit identity gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import ClassVar

import polars as pl
import pytest

from qavf import validate
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry
from qavf.engine.runner import QARunner
from qavf.settings import DatasetSpec, QAVFSettings
from qavf.sources.base import DataSource, SourceEnvironment, SourceParams
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams

QUERY = "SELECT sum(x) FROM (SELECT range x FROM range(300000) ORDER BY hash(range))"


def test_native_spill_works_without_precreating_directories(tmp_path):
    settings = QAVFSettings(
        engine={
            "parallel": False,
            "native_threads": 1,
            "memory_budget_bytes": 64 * 1024**2,
            "temp_directory": tmp_path,
        }
    )
    with DataContext.from_settings(settings, threads=1, execution_workers=2) as ctx:
        assert ctx.env.temp_directory.is_dir()
        assert int(ctx.sql(QUERY).item()) == 44999850000
        assert any(ctx.env.temp_directory.rglob("*"))
        directory = ctx._resources.directory
    assert not directory.exists()


@pytest.mark.parametrize("parallel,expected_slots", [(False, 2), (True, 2)])
def test_runner_budgets_for_actual_rule_concurrency(parallel, expected_slots):
    class Native(Rule):
        rule_id = "TEST-NATIVE"
        name = description = "Native query"
        category = "custom"

        def evaluate(self, ctx):
            assert ctx._resources.connection_slots == expected_slots
            assert int(ctx.sql(QUERY).item()) == 44999850000
            return Finding.ok("assessed")

    registry = RuleRegistry()
    registry.register(Native)
    settings = QAVFSettings(
        engine={"parallel": parallel, "native_threads": 1, "memory_budget_bytes": 64 * 1024**2}
    )
    report = QARunner(settings, registry).run()
    assert report.exit_code() == 0


def test_supplied_lazy_provenance_survives_release_and_close():
    report = validate(pl.DataFrame({"x": [1, 2]}).lazy(), schema={"columns": {"x": {"ge": 0}}})
    assert report.exit_code() == 0
    assert report.metadata.unavailable_datasets == {}
    assert len(report.metadata.datasets) == 1
    info = report.metadata.datasets[0]
    assert info.name == "data" and info.columns == ["x"]
    assert info.notes["execution"] == "supplied_lazy_scan"


def test_unused_supplied_lazy_is_not_unavailable_after_close():
    ctx = DataContext(
        frames={"lazy": pl.DataFrame({"x": [1]}).lazy()}, specs={"lazy": DatasetSpec()}
    )
    ctx.close()
    assert ctx.unavailable() == {}
    assert ctx.unused() == ["lazy"]


def test_path_plugin_module_and_plan_identity_stable_between_processes(tmp_path):
    path = tmp_path / "plugin.py"
    path.write_text("""from qavf.engine.registry import Rule,Finding
class Check(Rule):
 rule_id='STABLE'
 name=description='Stable'
 category='custom'
 def evaluate(self,ctx): return Finding.ok('done')
""")
    code = """import json,sys
from qavf.engine.registry import RuleRegistry
from qavf.engine.runner import _rule_info,config_digest
r=RuleRegistry();r.discover_path(sys.argv[1]);i=_rule_info(r.create('STABLE'))
print(json.dumps([i.module,config_digest({'rules':[i.model_dump(mode='python')]})]))
"""
    outputs = [
        json.loads(
            subprocess.check_output(
                [sys.executable, "-c", code, str(path)],
                env={**os.environ, "PYTHONHASHSEED": seed},
                text=True,
            )
        )
        for seed in ("1", "2")
    ]
    assert outputs[0] == outputs[1]


def test_generic_source_lazy_fallback_collects_only_budget_plus_one(monkeypatch):
    class LazyParams(SourceParams):
        type: str = "lazy-test"

    class LazySource(DataSource):
        kind: ClassVar[str] = "lazy-test"
        Params: ClassVar[type[SourceParams]] = LazyParams

        def load(self):
            return pl.DataFrame({"x": range(10000)}).lazy()

    source = LazySource(LazyParams(row_budget=3), SourceEnvironment())
    original = pl.LazyFrame.collect
    collected = []

    def observe(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        collected.append(result.height)
        return result

    with DataContext(sources={"d": source}) as ctx:
        monkeypatch.setattr(pl.LazyFrame, "collect", observe)
        assert not ctx.has("d")
        assert "row_budget" in ctx.unavailable_reason("d")
    assert collected == [4]


@pytest.mark.parametrize("mode", ["eager", "native"])
def test_sql_missing_normalization_preserves_raw_access_and_exact_types(tmp_path, mode):
    frame = pl.DataFrame(
        {
            "n": [1.0, float("nan"), -999.0, None],
            "s": ["yes", "\u2003", "missing", None],
            "d": [date(2020, 1, 1), date(2020, 1, 2), None, date(2020, 1, 3)],
            "t": [datetime(2020, 1, 1), datetime(2020, 1, 2), None, datetime(2020, 1, 3)],
            "m": pl.Series(
                [Decimal("1.01"), Decimal("2.02"), None, Decimal("3.03")], dtype=pl.Decimal(10, 2)
            ),
        }
    )
    spec = DatasetSpec(
        schema={
            "columns": {
                "n": {"missing_values": [-999]},
                "s": {"blank_as_missing": True, "missing_values": ["missing"]},
                "d": {"missing_values": [date(2020, 1, 2)]},
                "t": {"missing_values": [datetime(2020, 1, 2)]},
                "m": {"missing_values": [Decimal("2.02")]},
            }
        }
    )
    kwargs = {"frames": {"d": frame}}
    if mode == "native":
        path = tmp_path / "d.parquet"
        frame.write_parquet(path)
        kwargs = {"sources": {"d": FileSource(FileParams(path=str(path)), SourceEnvironment())}}
    with DataContext(specs={"d": spec}, **kwargs) as ctx:
        result = ctx.sql("SELECT * FROM d", normalize_missing=True)
        assert result.schema == frame.schema
        assert result.null_count().row(0) == (3, 3, 2, 2, 2)
        raw = ctx.sql("SELECT * FROM d")
        assert raw["n"].is_nan().sum() == 1
        assert raw["n"][2] == -999
        assert raw["s"][1] == "\u2003"


@pytest.mark.parametrize("mode", ["eager", "lazy", "parquet"])
def test_sql_cannot_silently_bind_case_distinct_columns(tmp_path, mode):
    from qavf.sources.base import SourceError

    frame = pl.DataFrame({"x": [0], "X": [10]})
    if mode == "parquet":
        path = tmp_path / "case.parquet"
        frame.write_parquet(path)
        kwargs = {"sources": {"d": FileSource(FileParams(path=str(path)), SourceEnvironment())}}
    else:
        kwargs = {"frames": {"d": frame if mode == "eager" else frame.lazy()}}
    with DataContext(**kwargs) as ctx, pytest.raises(SourceError, match="same identifier"):
        ctx.sql('SELECT "X" FROM d', normalize_missing=True)
    with DataContext(frames={"d": pl.DataFrame({"X": [10]})}) as ctx:
        assert ctx.sql('SELECT "X" FROM d').item() == 10
