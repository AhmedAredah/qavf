"""Independent regressions for input abstraction, ownership, and registry publication."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from qavf import File, validate
from qavf.engine.compilation import build_registry
from qavf.engine.connections import merge_secret_bindings
from qavf.engine.context import DataContext
from qavf.engine.registry import Finding, Rule, RuleRegistry, default_registry
from qavf.inputs.normalize import prepare
from qavf.plugins.registry import Component, ComponentRegistry
from qavf.plugins.transactions import transaction, visibility
from qavf.profiling import profile_frame
from qavf.profiling.accumulator import ProfileAccumulator
from qavf.settings import QAVFSettings, SecretSpec
from qavf.sources import DataSource, FileParams, FileSource, SourceEnvironment, SourceParams
from qavf.sources.base import SourceError, SourceRegistry
from qavf.sources.frames import as_polars, supports_frame
from qavf.sources.sql import normalize_sql_paths
from qavf.sources.storage import create_secret_statements, storage_options


class ReviewComponent(Component):
    name = "review_component"


class ReviewRule(Rule):
    rule_id = "REVIEW-INPUT"
    name = "Review input"
    description = "Synthetic rule for registration isolation"
    category = "review"

    def evaluate(self, ctx):
        return Finding.ok("checked")


class ReviewSource(DataSource):
    kind = "review_source"
    Params = SourceParams

    def load(self):
        return pl.DataFrame({"x": [1]})


@pytest.mark.parametrize("fail", [False, True])
def test_entry_point_transaction_covers_other_registries_and_origin(fail):
    owner = ComponentRegistry("owner", "review.owner", Component, ())
    components = ComponentRegistry("secondary", "review.secondary", Component, ())
    sources, rules = SourceRegistry(), RuleRegistry()

    def load():
        components.register(ReviewComponent)
        sources.register(ReviewSource)
        rules.register(ReviewRule)
        if fail:
            raise RuntimeError("synthetic import failure")
        return ReviewComponent

    owner.entries = lambda: [SimpleNamespace(name="review", load=load)]
    owner.discover(use_entry_points=True)
    assert "review_component" not in components
    assert "review_source" not in sources
    assert "REVIEW-INPUT" not in rules
    with visibility(True):
        assert ("review_component" in components) is not fail
        assert ("review_source" in sources) is not fail
        assert ("REVIEW-INPUT" in rules) is not fail


@pytest.mark.parametrize("fail", [False, True])
def test_cross_registry_transaction_preserves_concurrent_registration(fail):
    registry = ComponentRegistry("review", "review", Component, ())
    staged, finish = threading.Event(), threading.Event()
    errors = []

    class Concurrent(Component):
        name = "concurrent"

    def work():
        try:
            with transaction():
                registry.register(ReviewComponent)
                staged.set()
                assert finish.wait(5)
                if fail:
                    raise RuntimeError("rollback")
        except RuntimeError:
            if not fail:
                errors.append("unexpected error")
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert staged.wait(5)
        assert "review_component" not in registry
        registry.register(Concurrent)
    finally:
        finish.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    assert "concurrent" in registry
    assert ("review_component" in registry) is not fail


def test_registry_conflict_commits_no_other_family():
    components = ComponentRegistry("review", "review", Component, ())
    rules = RuleRegistry()

    class Conflict(ReviewRule):
        name = "Concurrent conflicting rule"

    with pytest.raises(ValueError, match="concurrent registry conflict"), transaction():
        components.register(ReviewComponent)
        rules.register(ReviewRule)
        worker = threading.Thread(target=lambda: rules.register(Conflict))
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
    assert "review_component" not in components
    assert rules.get("REVIEW-INPUT") is Conflict


def test_failed_outer_plugin_restores_nested_discovery_state():
    owner = ComponentRegistry("owner", "owner", Component, ())
    secondary = ComponentRegistry("secondary", "secondary", Component, ())
    attempts = []

    def secondary_load():
        attempts.append("loaded")
        return ReviewComponent

    def owner_load():
        secondary.discover(use_entry_points=True)
        raise RuntimeError("rollback enclosing plugin")

    secondary.entries = lambda: [SimpleNamespace(name="secondary", load=secondary_load)]
    owner.entries = lambda: [SimpleNamespace(name="owner", load=owner_load)]
    owner.discover(use_entry_points=True)
    assert not secondary._entry_points_loaded
    assert not secondary.plugins
    secondary.discover(use_entry_points=True)
    assert attempts == ["loaded", "loaded"]
    with visibility(True):
        assert secondary.get("review_component") is ReviewComponent


def test_component_plugin_rule_is_available_only_in_enabled_runs(monkeypatch):
    import qavf.plugins.registry as plugin_registry

    owner = ComponentRegistry("owner", "owner", Component, ())
    monkeypatch.setattr(default_registry, "_rules", {})
    monkeypatch.setattr(default_registry, "_origins", {})

    def load():
        default_registry.register(ReviewRule)
        return ReviewComponent

    owner.entries = lambda: [SimpleNamespace(name="cross_family", load=load)]
    monkeypatch.setattr(plugin_registry, "registries", lambda: {"review": owner})
    enabled = QAVFSettings(plugins={"packages": [], "entry_points": True})
    disabled = QAVFSettings(plugins={"packages": [], "entry_points": False})
    assert "REVIEW-INPUT" in build_registry(enabled)
    assert "REVIEW-INPUT" not in build_registry(disabled)


def test_generic_arrow_stream_is_consumed_once_and_matches_table():
    table = pa.table({"x": [1, -1], "text": ["a", "b"]})

    class Stream:
        calls = 0

        def __arrow_c_stream__(self, requested_schema=None):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("stream requested twice")
            return table.__arrow_c_stream__(requested_schema)

    value = Stream()
    assert supports_frame(value) and value.calls == 0
    actual = as_polars(value)
    assert value.calls == 1
    assert actual.to_dicts() == pl.from_arrow(table).to_dicts()
    report = validate(Stream(), schema={"columns": {"x": {"ge": 0}}})
    assert report.get("INP-003").affected_count == 1


def test_arrow_adaptation_retains_chunk_buffers():
    chunks = [pa.array([1, 2]), pa.array([3, 4])]
    table = pa.table({"x": pa.chunked_array(chunks)})
    frame = as_polars(table)
    output = frame.to_arrow().column("x")
    assert output.num_chunks == 2
    assert [part.buffers()[1].address for part in output.chunks] == [
        part.buffers()[1].address for part in chunks
    ]


def synthetic_secret(value="one", **kwargs):
    return SecretSpec(type="s3", key_id=value, secret=value, **kwargs)


def test_source_bind_preserves_snapshot_of_owned_credentials():
    original = synthetic_secret()
    source = FileSource(
        FileParams(path="s3://private/data.parquet"),
        SourceEnvironment(secrets={"lake": original}),
    )
    bound = source.bind(SourceEnvironment(secrets={"other": synthetic_secret("other")}))
    assert set(bound.env.secrets) == {"lake"}
    assert bound.env.secrets["lake"] == original
    original.key_id = "mutated"
    assert bound.env.secrets["lake"].key_id == "one"
    assert source.env.secrets["lake"].key_id == "mutated"


def test_conflicting_operation_credentials_are_rejected():
    bindings = {}
    merge_secret_bindings(bindings, SourceEnvironment(secrets={"lake": synthetic_secret()}))
    with pytest.raises(SourceError, match="conflicting"):
        merge_secret_bindings(
            bindings, SourceEnvironment(secrets={"lake": synthetic_secret("two")})
        )
    with pytest.raises(SourceError, match="scopes"):
        merge_secret_bindings(
            bindings, SourceEnvironment(secrets={"other": synthetic_secret("two")})
        )


@pytest.mark.parametrize("owned", [False, True])
def test_native_scan_uses_source_credentials_not_run_credentials(tmp_path, monkeypatch, owned):
    import qavf.engine.connections as connections
    from qavf.sources.native import connect

    path = tmp_path / "data.csv"
    path.write_text("x\n1\n2\n")
    secrets = {"owned": synthetic_secret()} if owned else {}
    source = FileSource(FileParams(path=str(path)), SourceEnvironment(secrets=secrets))
    observed = []

    def local_connect(env, attach=()):
        observed.append(env)
        # Verify selection without invoking a real remote credential provider.
        return connect(replace(env, secrets={}), attach)

    monkeypatch.setattr(connections, "connect", local_connect)
    settings, frames = prepare(
        source,
        schema={"columns": {"x": {"ge": 0}}},
        config={"secrets": {"run": synthetic_secret("run")}},
        resources=None,
    )
    with DataContext.from_settings(settings, frames) as context:
        assert context.sql("SELECT sum(x) AS n FROM data")["n"].item() == 3
    assert observed and all(dict(env.secrets) == secrets for env in observed)


def test_native_connection_reuse_tracks_credentials_and_rejects_ambiguous_join(
    tmp_path, monkeypatch
):
    import qavf.engine.connections as connections
    from qavf.sources.native import connect

    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    sources = {
        name: FileSource(
            FileParams(path=str(path)), SourceEnvironment(secrets={"lake": synthetic_secret(name)})
        )
        for name in ("first", "second")
    }
    observed = []

    def local_connect(env, attach=()):
        observed.append(env.secrets["lake"].key_id)
        return connect(replace(env, secrets={}), attach)

    monkeypatch.setattr(connections, "connect", local_connect)
    settings, frames = prepare(sources, schema=None, config={}, resources=None)
    with DataContext.from_settings(settings, frames) as context:
        assert context.sql("SELECT * FROM first")["x"].item() == 1
        assert context.sql("SELECT * FROM second")["x"].item() == 1
        with pytest.raises(SourceError, match="conflicting storage secret"):
            context.sql("SELECT * FROM first JOIN second USING (x)")
    assert observed == ["first", "second"]


@pytest.mark.parametrize("parameter", ["path", "1"])
def test_shared_path_parameter_retains_scalar_meaning(tmp_path, parameter):
    (tmp_path / "d.csv").write_text("x\n1\n")
    sql = f"SELECT ${parameter} AS original, x FROM read_csv(${parameter})"
    normalized, values = normalize_sql_paths(sql, tmp_path, {parameter: "d.csv"})
    with duckdb.connect() as connection:
        assert connection.execute(normalized, values).fetchall() == [("d.csv", 1)]


def test_path_parameters_do_not_collide_with_user_internal_looking_names(tmp_path):
    (tmp_path / "d.csv").write_text("x\n1\n")
    sql = "SELECT $path, $__qavf_path_0, x FROM read_csv($PaTh)"
    normalized, values = normalize_sql_paths(
        sql, tmp_path, {"path": "d.csv", "__qavf_path_0": "keep"}
    )
    with duckdb.connect() as connection:
        assert connection.execute(normalized, values).fetchall() == [("d.csv", "keep", 1)]


def test_native_secret_names_are_injective_across_scopes_and_user_names():
    settings = QAVFSettings(
        secrets={
            "lake": synthetic_secret(scope=["s3://one", "s3://two"]),
            "lake__scope_0": synthetic_secret("other", scope=["s3://three"]),
        }
    )
    names = [
        statement.split('"')[1]
        for name, spec in settings.secrets.items()
        for statement in create_secret_statements(name, spec)
    ]
    assert len(names) == len(set(names)) == 3


@pytest.mark.parametrize("chain", ["env", "config", "instance", "env;config"])
def test_explicit_chain_never_silently_changes_at_polars_boundary(chain):
    secret = SecretSpec(type="s3", provider="credential_chain", chain=chain)
    with pytest.raises(SourceError, match="chain is not supported"):
        storage_options("s3://private/data.parquet", {"lake": secret})


def test_projected_numeric_schema_retains_native_reader(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1], "unused": pl.Series([1], dtype=pl.Datetime("ns"))}).write_parquet(path)
    settings, frames = prepare(
        File(path, columns=["x"]),
        schema={"columns": {"x": {"ge": 0}}},
        config={"plugins": {"entry_points": False}},
        resources=None,
    )
    with DataContext.from_settings(settings, frames) as context:
        assert context.execution_kind("data") == "native_sql"
        assert context.schema("data") == {"x": pl.Int64}


def test_projection_preserves_case_sensitive_physical_column_identity(tmp_path):
    path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1], "X": [2], "other": [3]}).write_parquet(path)
    settings, frames = prepare(
        File(path, columns=["X"]), schema=None, config={}, resources=None
    )
    with DataContext.from_settings(settings, frames) as context:
        assert context.get("data").to_dicts() == [{"X": 2}]
    settings, frames = prepare(
        File(path, columns=["other"]), schema=None, config={}, resources=None
    )
    with DataContext.from_settings(settings, frames) as context:
        assert context.execution_kind("data") == "native_sql"
        assert context.get("data").to_dicts() == [{"other": 3}]


def test_large_category_falls_back_to_exact_lengths_with_disclosure():
    text = "é" * 100_000
    profile = profile_frame(pl.DataFrame({"text": [text, None]}))
    column = profile["columns"]["text"]
    assert "isin" not in column
    assert column["min_length"] == column["max_length"] == 100_000
    assert column["nullable"]
    assert "byte limit" in column["description"]


def test_category_budget_is_shared_and_released_for_high_cardinality():
    accumulator = ProfileAccumulator(
        {"a": pl.String, "b": pl.String},
        max_categories=2,
        bounds=True,
        max_category_bytes=260,
        max_category_value_bytes=64,
    )
    accumulator.update(pl.DataFrame({"a": ["a", "a"], "b": ["b", "b"]}))
    assert accumulator.category_budget.used <= 260
    accumulator.update(pl.DataFrame({"a": ["c", "d"], "b": ["b", "b"]}))
    assert accumulator.columns["a"].categories is None
    assert accumulator.category_budget.used == accumulator.columns["b"].category_bytes


@pytest.mark.parametrize("budget", [0, -1, True])
def test_profile_category_byte_budget_admission(budget):
    if budget == 0 and type(budget) is int:
        column = profile_frame(pl.DataFrame({"x": ["a"]}), max_category_bytes=budget)["columns"][
            "x"
        ]
        assert "isin" not in column
    else:
        with pytest.raises(ValueError, match="non-negative integer"):
            profile_frame(pl.DataFrame({"x": ["a"]}), max_category_bytes=budget)
