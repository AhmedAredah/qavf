"""Simple input descriptions retain explicit QA, ownership and no-I/O preparation."""

import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import polars as pl
import pyarrow as pa
import pytest

from qavf import Dataset, File, Finding, QAVFSettings, Rule, Status, validate
from qavf.engine.registry import default_registry
from qavf.settings import SchemaSpec

SCHEMA = {"columns": {"x": {"ge": 0}}}


def test_file_accepts_extensionless_reader_settings_and_projects_explicitly(tmp_path):
    path = tmp_path / "readings"
    path.write_text("old;unused\n-1;99\n2;88\n")
    report = validate(
        Dataset(
            File(path, format="csv", options={"delim": ";", "header": True}, columns=["old"]),
            rename={"old": "x"},
            schema={**SCHEMA, "strict": True},
        )
    )
    assert report.get("INP-003", "data.x: ge 0").affected_count == 1
    assert not report.incomplete
    assert report.metadata.datasets[0].columns == ["x"]


def test_descriptor_contracts_snapshot_configuration_without_copying_data():
    frame = pl.DataFrame({"x": [-1, 1]})
    schema = {"columns": {"x": {"ge": 0}}}
    item = Dataset(frame, schema=schema)
    schema["columns"]["x"]["ge"] = -10
    assert item.data is frame
    assert validate(item).get("INP-003", "data.x: ge 0").affected_count == 1


def test_file_options_snapshot_caller_mapping():
    options = {"types": {"x": "BIGINT"}, "header": True}
    item = File("data.csv", options=options)
    options["types"]["x"] = "VARCHAR"
    options["header"] = False
    assert item.source_config()["options"] == {"types": {"x": "BIGINT"}, "header": True}


def test_file_and_dataset_construction_do_not_discover_or_read_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("descriptor construction attempted filesystem or lazy-frame evaluation")

    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(pl.LazyFrame, "collect", forbidden)
    item = Dataset(File("missing.parquet"), schema=SCHEMA)
    assert item.contract().schema_spec.columns["x"].ge == 0


@pytest.mark.parametrize(
    "options",
    [
        {"description": "readings"},
        {"required": False},
        {"load_policy": {"row_budget": 10}},
    ],
)
def test_metadata_or_loading_policy_does_not_imply_acceptance_contract(options):
    item = Dataset(pl.DataFrame({"x": [1]}), **options)
    with pytest.raises(ValueError, match="acceptance limits are not inferred"):
        validate(item)


@pytest.mark.parametrize("options", [{"schema": {}}, {"foreign_keys": []}, {"expressions": []}])
def test_explicit_empty_contracts_retain_existing_empty_config_semantics(options):
    report = validate(Dataset(pl.DataFrame({"x": [1]}), **options))
    assert not report.incomplete


def test_mixed_named_raw_and_described_inputs_do_not_guess_column_dictionaries():
    frame = pl.DataFrame({"x": [-1, 1]})
    report = validate({"checked": Dataset(frame, schema=SCHEMA), "reference": frame})
    assert report.get("INP-003", "checked.x: ge 0").status is Status.FAIL
    assert {dataset.name for dataset in report.metadata.datasets} == {"checked", "reference"}
    with pytest.raises(TypeError, match="mapping denotes named datasets"):
        validate({"x": [1, 2]}, schema=SCHEMA)


def test_invalid_sibling_type_does_not_consume_arrow_protocol():
    calls = []

    class ArrowInput:
        def __arrow_c_stream__(self, requested_schema=None):
            calls.append(1)
            return pa.table({"x": [1]}).__arrow_c_stream__(requested_schema)

    with pytest.raises(TypeError, match="unsupported input"):
        validate({"first": Dataset(ArrowInput(), schema=SCHEMA), "second": object()})
    assert calls == []


def test_resource_description_import_does_not_initialize_data_engines():
    script = (
        "import sys; from qavf import ResourceLimits; "
        "assert ResourceLimits(target_memory='1MiB'); "
        "assert 'duckdb' not in sys.modules and 'polars' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).parents[1])


@pytest.mark.parametrize("data", [None, Dataset(pl.DataFrame({"x": [1]}), schema=SCHEMA)])
def test_nested_or_absent_dataset_payload_is_rejected(data):
    with pytest.raises(TypeError, match=r"Dataset\.data"):
        Dataset(data, schema=SCHEMA)


@pytest.mark.parametrize("form", ["mapping", "settings"])
@pytest.mark.parametrize("invalid_contract", [False, True])
def test_free_form_tuple_semantics_survive_while_nested_models_revalidate(
    monkeypatch, form, invalid_contract
):
    calls = []

    class TupleConsumer(Rule):
        rule_id: ClassVar[str] = "FACADE-TUPLES"
        name: ClassVar[str] = "Tuple configuration"
        description: ClassVar[str] = "Plugin settings retain tuple and list distinctions."
        category: ClassVar[str] = "test"

        def evaluate(self, ctx):
            calls.append(ctx.extension("probe")["value"])
            return Finding.ok("plugin evaluated")

    monkeypatch.setattr(default_registry, "_rules", dict(default_registry._rules))
    default_registry.register(TupleConsumer)
    contract = SchemaSpec.model_validate(SCHEMA)
    expected = ("pair", {"limits": (1, 2), "records": [{"tags": ("a", "b")}]})
    raw = {
        "datasets": {"data": {"schema": contract}},
        "extensions": {"probe": {"value": expected}},
        "rules": {TupleConsumer.rule_id: {}},
        "plugins": {"entry_points": False},
        "engine": {"run_unconfigured_rules": False, "parallel": False},
    }
    config = QAVFSettings.model_validate(raw) if form == "settings" else raw
    if invalid_contract:
        contract.columns["x"].ge = float("nan")
        with pytest.raises(ValueError, match="finite"):
            validate(pl.DataFrame({"x": [1]}), config=config)
        assert calls == []
    else:
        report = validate(pl.DataFrame({"x": [1]}), config=config)
        assert report.verdict() == "passed"
        assert calls == [expected]
        assert isinstance(calls[0], tuple)
        assert isinstance(calls[0][1]["limits"], tuple)
        assert isinstance(calls[0][1]["records"], list)
        assert isinstance(calls[0][1]["records"][0]["tags"], tuple)
