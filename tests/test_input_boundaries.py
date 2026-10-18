"""Independent public acceptance across input contracts and ownership boundaries."""

from __future__ import annotations

import json

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from qavf import Dataset, File, Loader, Query, Status, validate
from qavf.engine.runner import QARunner
from qavf.settings import QAVFSettings, SchemaSpec, SourcePolicy
from qavf.sources import FileParams, FileSource, SourceEnvironment

SCHEMA = {"columns": {"x": {"checks": [{"name": "nonnegative", "ge": 0}]}}}
CONFIG = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}


def test_named_contracts_keep_each_schema_reference_and_expression():
    report = validate(
        {
            "child": Dataset(
                pl.DataFrame({"id": [1, 2], "x": [-1, 2]}),
                schema=SCHEMA,
                foreign_keys=[{"columns": ["id"], "ref_dataset": "parent"}],
                expressions=[{"name": "positive expression", "expr": "x >= 0"}],
            ),
            "parent": Dataset(
                pl.DataFrame({"id": [1]}),
                schema={"columns": {"id": {"checks": [{"name": "minimum id", "ge": 2}]}}},
            ),
        },
        config=CONFIG,
    )
    assert report.exit_code() == 1
    assert not report.incomplete
    for rule, check in (
        ("INP-003", "child.x: nonnegative"),
        ("INP-003", "parent.id: minimum id"),
        ("INP-005", "child(id) -> parent"),
    ):
        result = report.get(rule, check)
        assert result.status is Status.FAIL
        assert result.affected_count == 1
    expression = report.get("INP-006")
    assert expression.status is Status.FAIL
    assert expression.affected_count == 1


@pytest.mark.parametrize("form", ["mapping", "settings", "path"])
@pytest.mark.parametrize("conflict", ["schema", "rename"])
def test_explicit_duplicate_contract_rejected_before_loader_even_if_identical(
    tmp_path, form, conflict
):
    calls = []
    declaration = {conflict: SCHEMA if conflict == "schema" else {}}
    config = {**CONFIG, "datasets": {"data": declaration}}
    if form == "settings":
        config = QAVFSettings.model_validate(config)
    elif form == "path":
        path = tmp_path / "qavf.json"
        path.write_text(json.dumps(config))
        config = path
    supplied = Dataset(Loader(lambda: calls.append(True)), **declaration)
    with pytest.raises(ValueError, match=r"conflict|already|both|duplicate"):
        validate(supplied, config=config)
    assert calls == []


def test_omitted_configuration_defaults_do_not_conflict_with_inline_contract():
    settings = QAVFSettings.model_validate({**CONFIG, "datasets": {"data": {}}})
    before = settings.model_dump()
    report = validate(Dataset(pl.DataFrame({"x": [-1]}), schema=SCHEMA, rename={}), config=settings)
    assert report.get("INP-003").status is Status.FAIL
    assert settings.model_dump() == before


@pytest.mark.parametrize(
    "placement", ["schema_argument", "dataset_schema", "settings_schema", "engine"]
)
def test_mutated_models_are_revalidated_before_loader_invocation(placement):
    calls = []
    loader = Loader(lambda: calls.append(True))
    schema = SchemaSpec.model_validate(SCHEMA)
    schema.columns["x"].ge = float("nan")
    settings = QAVFSettings.model_validate({**CONFIG, "datasets": {"data": {"schema": SCHEMA}}})
    with pytest.raises(ValueError):
        if placement == "schema_argument":
            validate(loader, schema=schema, config=CONFIG)
        elif placement == "dataset_schema":
            validate(Dataset(loader, schema=schema), config=CONFIG)
        elif placement == "settings_schema":
            settings.datasets["data"].schema_spec = schema
            validate(loader, config=settings)
        else:
            settings.engine.max_workers = 0
            validate(loader, config=settings)
    assert calls == []


@pytest.mark.parametrize("invalid", ["suffix", "reader_options"])
def test_invalid_selected_file_is_rejected_before_converting_arrow_sibling(monkeypatch, invalid):
    conversions = []
    original = pl.from_arrow

    def observe(*args, **kwargs):
        conversions.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pl, "from_arrow", observe)
    config = {**CONFIG, "datasets": {"first": {"schema": SCHEMA}, "second": {}}}
    if invalid == "reader_options":
        config["datasets"]["second"]["source"] = {
            "type": "file",
            "path": "unused.csv",
            "options": {"header": "not a Boolean"},
        }
    supplied = {
        "first": pa.table({"x": [-1]}),
        "second": "unrecognized.suffix" if invalid == "suffix" else "unused.csv",
    }
    with pytest.raises(ValueError):
        validate(supplied, config=config)
    assert conversions == []


@pytest.mark.parametrize("policy", [None, {}, SourcePolicy()])
def test_omitted_policy_fields_keep_configured_source_budget(policy):
    report = validate(
        Dataset(pl.DataFrame({"x": [1, 2]}), schema=SCHEMA, load_policy=policy),
        config={
            **CONFIG,
            "datasets": {
                "data": {"source": {"type": "file", "path": "unused.parquet", "row_budget": 1}}
            },
        },
    )
    assert report.exit_code() == 1
    assert any("row_budget" in result.message for result in report.results)


@pytest.mark.parametrize("budget", [None, 2])
def test_inline_policy_cannot_silently_remove_or_change_declared_source_budget(tmp_path, budget):
    source = FileSource(
        FileParams(path="never-read.parquet", row_budget=1), SourceEnvironment(base_dir=tmp_path)
    )
    with pytest.raises(ValueError, match=r"conflict|already|both|duplicate"):
        validate(Dataset(source, schema=SCHEMA, load_policy={"row_budget": budget}), config=CONFIG)
    assert source.params.row_budget == 1


def test_preparation_preserves_lazy_input_and_borrowed_handles(monkeypatch):
    lazy = pl.DataFrame({"x": [-1]}).lazy()
    calls = []
    captured = []

    class Borrowed:
        def __deepcopy__(self, memo):
            raise AssertionError("borrowed handles must not be copied")

    handle = Borrowed()

    def load(*, external):
        calls.append(external)
        return lazy

    def collect_forbidden(*args, **kwargs):
        raise AssertionError("input preparation must not collect a lazy plan")

    def capture(self, *, frames):
        captured.append(frames)
        return None

    monkeypatch.setattr(pl.LazyFrame, "collect", collect_forbidden)
    monkeypatch.setattr(QARunner, "run", capture)
    validate(Dataset(lazy, schema=SCHEMA), config=CONFIG)
    validate(Dataset(Loader(load, kwargs={"external": handle}), schema=SCHEMA), config=CONFIG)
    assert captured[0]["data"] is lazy
    assert not calls


def test_dataset_wrapped_borrowed_query_remains_open_after_failure():
    with duckdb.connect() as connection:
        report = validate(
            Dataset(Query("SELECT $x AS x", connection, {"x": -1}), schema=SCHEMA), config=CONFIG
        )
        assert report.get("INP-003").status is Status.FAIL
        assert connection.execute("SELECT 42").fetchone() == (42,)


def test_raw_path_inherits_reader_contract_but_explicit_file_replaces_it(tmp_path):
    semicolon = tmp_path / "semicolon.data"
    comma = tmp_path / "comma.data"
    semicolon.write_text("x;extra\n1;2\n")
    comma.write_text("x,extra\n1,2\n")
    config = {
        **CONFIG,
        "datasets": {
            "data": {
                "schema": {**SCHEMA, "strict": True},
                "source": {
                    "type": "file",
                    "path": "unused.data",
                    "format": "csv",
                    "options": {"delim": ";", "header": True},
                    "columns": ["x"],
                },
            }
        },
    }
    assert validate(semicolon, config=config).exit_code() == 0
    explicit = File(comma, format="csv", options={"delim": ",", "header": True}, columns=["x"])
    assert validate(explicit, config=config).exit_code() == 0
    unprojected = File(comma, format="csv", options={"delim": ",", "header": True})
    report = validate(unprojected, config=config)
    assert report.get("INP-002").status is Status.FAIL


def test_descriptor_uses_config_root_while_direct_source_keeps_owned_root(tmp_path):
    config_root = tmp_path / "configured"
    source_root = tmp_path / "owned"
    config_root.mkdir()
    source_root.mkdir()
    pl.DataFrame({"x": [-1]}).write_parquet(config_root / "input.parquet")
    pl.DataFrame({"x": [1]}).write_parquet(source_root / "input.parquet")
    config = config_root / "qavf.json"
    config.write_text(json.dumps(CONFIG))
    described = validate(Dataset(File("input.parquet"), schema=SCHEMA), config=config)
    source = FileSource(FileParams(path="input.parquet"), SourceEnvironment(base_dir=source_root))
    supplied = validate(Dataset(source, schema=SCHEMA), config=config)
    assert described.get("INP-003").status is Status.FAIL
    assert supplied.exit_code() == 0
    assert described.metadata.datasets[0].source == str(config_root / "input.parquet")
    assert supplied.metadata.datasets[0].source == str(source_root / "input.parquet")
