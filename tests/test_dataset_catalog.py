"""Dataset observations survive data release without becoming an execution owner."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import polars as pl
import pytest

from qavf import DataContext
from qavf.engine.catalog import DatasetCatalog
from qavf.engine.resources import FrameStore
from qavf.settings import DatasetSpec
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams
from qavf.sources.parameters import QueryParams
from qavf.sources.query import QuerySource


def catalog(*, specs=None, sources=None, provided=(), canonical=()):
    return DatasetCatalog(specs or {}, sources or {}, provided, canonical, threading.RLock())


def test_catalog_retains_schema_not_input_frame_and_returns_detached_schema():
    owner = catalog(provided=["data"])
    frame = pl.DataFrame({"x": [1, 2]})
    reference = weakref.ref(frame)
    owner.record_frame("data", frame)
    del frame
    gc.collect()
    assert reference() is None
    schema = owner.schema("data")
    schema["x"] = pl.String
    assert owner.schema("data") == {"x": pl.Int64}
    assert owner.describe()[0].rows == 2


def test_empty_observed_schema_is_distinct_from_unknown_schema():
    owner = catalog()
    assert owner.schema("missing") is None
    owner.record_frame("empty", pl.DataFrame())
    owner.record_lazy("empty", {"should_not_replace": pl.String})
    assert owner.schema("empty") == {}
    info = owner.describe()[0]
    assert info.name == "empty" and info.rows == 0 and info.columns == []


def test_fingerprint_none_is_recorded_and_metadata_detaches_input_payloads():
    owner = catalog()
    assert not owner.has_fingerprint("data")
    owner.record_fingerprint("data", None)
    assert owner.has_fingerprint("data")
    owner.record_frame("data", pl.DataFrame({"x": [1]}))
    fingerprint = {"files": [{"sha256": "accepted"}]}
    notes = {"read": {"columns": ["x"]}}
    owner.record_fingerprint("data", fingerprint)
    owner.record_load("data", notes, 1.25)
    fingerprint["files"][0]["sha256"] = "changed"
    notes["read"]["columns"].append("changed")
    info = owner.describe()[0]
    assert info.fingerprint["files"][0]["sha256"] == "accepted"
    assert info.notes["read"]["columns"] == ["x"]
    with pytest.raises(TypeError):
        info.fingerprint["files"][0]["sha256"] = "mutated report"
    assert owner.describe()[0].fingerprint == info.fingerprint


def test_catalog_failure_and_declaration_queries_never_execute_sources(monkeypatch):
    source = QuerySource(QueryParams(sql="SELECT 1 AS x"), SourceEnvironment())
    owner = catalog(
        specs={"missing": DatasetSpec(), "unused": DatasetSpec()},
        sources={"native": source},
        provided=["unused"],
    )
    owner.record_native("native", {"x": pl.Int64}, 0.1)
    owner.record_native_plan("native")
    original = SourceError("first failure")
    owner.record_failure("broken", original)
    owner.record_failure("broken", SourceError("later failure"), replace=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Catalog reporting must not perform source I/O")

    for method in (
        "load",
        "scan",
        "scan_polars",
        "fingerprint",
        "prepare_identity",
        "verify_identity",
    ):
        monkeypatch.setattr(source, method, forbidden)
    assert owner.names([], []) == ["native"]
    assert owner.declared(["native"]) == ["missing", "native", "unused"]
    assert owner.unused() == ["unused"]
    assert owner.failure("broken") is original
    assert owner.unavailable([]) == {
        "broken": "first failure",
        "missing": (
            "dataset 'missing' is declared but has no source, and no in-memory frame was supplied"
        ),
    }
    assert owner.describe()[0].notes["execution"] == "native_scan"
    plans = owner.query_plans()
    plans[0]["dataset"] = "changed"
    assert owner.query_plans()[0]["dataset"] == "native"


def test_catalog_identity_state_changes_preserve_accepted_fingerprint():
    owner = catalog()
    owner.record_frame("data", pl.DataFrame({"x": [1]}))
    owner.record_fingerprint("data", {"sha256": "original"})
    owner.mark_identity_used(["data", "schema_probe"])
    assert owner.identity_names() == ["data", "schema_probe"]
    owner.identity_pending("data")
    pending = owner.describe()[0]
    assert pending.fingerprint["identity"] == "pending"
    assert pending.fingerprint["initial_identity"] == {"sha256": "original"}
    assert pending.notes["identity_verified"] is False
    failure = SourceError("input changed")
    owner.identity_failed("data", failure)
    invalid = owner.describe()[0]
    assert invalid.fingerprint["identity"] == "invalidated"
    assert invalid.fingerprint["reason"] == "input changed"
    assert owner.failure("data") is failure


def test_context_metadata_survives_release_and_close_for_all_execution_modes(monkeypatch):
    native = QuerySource(QueryParams(sql="SELECT 3 AS Value"), SourceEnvironment())
    ctx = DataContext(
        frames={
            "eager": pl.DataFrame({"VALUE": [1]}),
            "lazy": pl.DataFrame({"value": [2]}).lazy(),
            "unused_lazy": pl.DataFrame({"value": [4]}).lazy(),
        },
        specs={"missing": DatasetSpec(), "unused_lazy": DatasetSpec()},
        sources={"native": native},
        canonical_columns=["value"],
        cache_max_bytes=0,
    )
    with ctx:
        assert ctx.schema("lazy") == {"value": pl.Int64}
        assert ctx.schema("native") == {"value": pl.Int32}

        def no_spill_read(*args, **kwargs):
            raise AssertionError("Catalog queries must not promote spilled frames")

        monkeypatch.setattr(FrameStore, "__getitem__", no_spill_read)
        before = {item.name: item for item in ctx.describe()}
        assert set(before) == {"eager", "lazy", "native"}
        ctx.release(["eager", "lazy", "native"])
        assert {item.name: item for item in ctx.describe()} == before
        assert ctx.unused() == ["unused_lazy"]
    assert {item.name: item for item in ctx.describe()} == before
    assert ctx.declared() == ["eager", "lazy", "missing", "native", "unused_lazy"]
    assert list(ctx.unavailable()) == ["missing"]
    assert ctx.unused() == ["unused_lazy"]


def test_row_budget_failure_is_recorded_without_false_observation():
    ctx = DataContext(
        frames={"data": pl.DataFrame({"x": [1, 2]})},
        specs={"data": DatasetSpec(load_policy={"row_budget": 1})},
    )
    with ctx:
        assert not ctx.has("data")
        assert "row_budget" in ctx.unavailable()["data"]
        assert ctx.describe() == []
        assert ctx.unused() == []
    assert "row_budget" in ctx.unavailable()["data"]


def test_explicit_rename_resolves_canonical_ambiguity_and_is_recorded():
    owner = catalog(specs={"data": DatasetSpec(rename={"VALUE": "other"})}, canonical=["value"])
    frame = pl.DataFrame({"VALUE": [1], "Value": [2]})
    result = owner.prepare("data", frame)
    owner.record_frame("data", result)
    assert result.columns == ["other", "value"]
    assert owner.describe()[0].renamed == {"Value": "value", "VALUE": "other"}
    with pytest.raises(ValueError, match="ambiguous canonical"):
        catalog(canonical=["value"]).prepare("data", frame)


class _NoFingerprint(DataSource):
    kind = "catalog_probe"
    Params = SourceParams
    calls = 0

    def load(self):
        return pl.DataFrame({"x": [1]})

    def scan_polars(self):
        return pl.DataFrame({"x": [1]}).lazy()

    def fingerprint(self, **kwargs):
        self.calls += 1
        return None


def test_unavailable_fingerprint_is_not_retried_by_repeated_schema_and_reporting():
    source = _NoFingerprint(SourceParams(type="catalog_probe"), SourceEnvironment())
    with DataContext(sources={"data": source}) as ctx:
        for _ in range(3):
            assert ctx.schema("data") == {"x": pl.Int64}
            assert ctx.describe()[0].fingerprint is None
        assert ctx._catalog.sources["data"].calls == 1


def test_source_metadata_accepts_readonly_mappings_and_detaches_their_backing_values():
    payload = {"digest": "accepted", "paths": ["part"]}

    class ReadonlyMetadata(DataSource):
        kind = "readonly_metadata"
        Params = SourceParams

        def load(self):
            self.load_notes = {"details": MappingProxyType(payload)}
            return pl.DataFrame({"x": [1]})

        def fingerprint(self, **kwargs):
            return {"source": MappingProxyType(payload)}

    source = ReadonlyMetadata(SourceParams(type="readonly_metadata"), SourceEnvironment())
    with DataContext(sources={"data": source}) as ctx:
        assert ctx.get("data").item() == 1
        payload["digest"] = "changed"
        payload["paths"].append("changed")
        info = ctx.describe()[0]
        assert info.fingerprint["source"] == {"digest": "accepted", "paths": ["part"]}
        assert info.notes["details"] == {"digest": "accepted", "paths": ["part"]}
        assert ctx.unavailable() == {}
        assert ctx.get("data").item() == 1


def test_parallel_metadata_publication_preserves_each_dataset():
    owner = catalog()

    def publish(index):
        name = f"data_{index}"
        owner.record_frame(name, pl.DataFrame({"x": [index]}))
        owner.record_fingerprint(name, {"index": index})
        owner.mark_identity_used([name])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publish, range(20)))
    infos = owner.describe()
    assert len(infos) == len(owner.identity_names()) == 20
    assert all(info.rows == 1 and info.fingerprint["index"] == int(info.name[5:]) for info in infos)
