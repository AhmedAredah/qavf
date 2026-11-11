"""Portable resource and backend settings (UX-03): the ``resources:`` section in the
vocabulary of ``ResourceLimits``, one value per limit across ``resources:``,
``engine`` and ``resources=``, and ``engine.backend`` with its precedence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from qavf import ResourceLimits, explain, validate
from qavf.backend_choice import BackendChoice, BackendRequest, ConfiguredBackend
from qavf.cli import main
from qavf.inputs.resources import ResourceSpec
from qavf.io import ProtectedInputs
from qavf.optional import OptionalDependencyError
from qavf.resolution import resolve
from qavf.settings import EngineSettings, QAVFSettings, load_settings

_ROOT = Path(__file__).resolve().parents[1]
_NO_POLARS = r"""
import importlib.abc, runpy, sys
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"polars", "pandas"}:
            raise ModuleNotFoundError("optional dependency absent", name=fullname)
sys.meta_path.insert(0, CoreOnly())
sys.argv = ["qavf", *sys.argv[1:]]
runpy.run_module("qavf.cli", run_name="__main__")
"""
DATA = "id,v\n1,2\n2,3\n"
DATASET = """datasets:
  data:
    path: data.csv
    schema: {columns: {id: {dtype: int}, v: {dtype: int}}}
"""


def _project(tmp_path: Path, text: str, name: str = "qavf.yaml") -> Path:
    (tmp_path / "data.csv").write_text(DATA, encoding="utf-8")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- the section
@pytest.mark.parametrize(
    ("name", "written", "engine_field", "value"),
    [
        ("target_memory", "2GiB", "memory_budget_bytes", 2 * 1024**3),
        ("cache_memory", "256MiB", "cache_max_bytes", 256 * 1024**2),
        ("max_spill", "20GB", "spill_max_bytes", 20 * 1000**3),
        ("max_spill_files", 64, "max_spill_files", 64),
        ("temp_dir", "scratch", "temp_directory", Path("scratch")),
        ("max_workers", 3, "max_workers", 3),
        ("native_threads", 2, "native_threads", 2),
        ("evidence_rows", 7, "max_affected_records", 7),
        ("evidence_bytes", "64KiB", "max_evidence_bytes", 65536),
        ("report_bytes", "1.5MiB", "max_report_bytes", 1572864),
        ("isolate", True, "execution_mode", "isolated"),
        ("wall_time", 30, "run_timeout_s", 30),
        ("address_space", "4GiB", "process_address_space_bytes", 4 * 1024**3),
        ("cpu_time", 60, "process_cpu_time_s", 60),
        ("file_size", "1GB", "process_file_bytes", 10**9),
    ],
)
def test_each_resource_limit_sets_its_engine_setting(name, written, engine_field, value):
    settings = QAVFSettings.model_validate({"resources": {name: written}})
    assert getattr(settings.engine, engine_field) == value
    assert ResourceLimits.ENGINE_FIELDS[name] == engine_field
    (place,) = settings.resource_provenance.places(engine_field)
    assert (place.source, place.written) == ("resources", f"resources.{name}")


def test_process_limits_select_isolation_and_the_section_keeps_its_text():
    settings = QAVFSettings.model_validate({"resources": {"wall_time": 5, "target_memory": "1GB"}})
    assert settings.engine.execution_mode == "isolated"
    assert settings.resources is not None and settings.resources.target_memory == "1GB"


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"target_memory": True}, "valid integer"),
        ({"target_memory": "lots"}, "target_memory must be a byte count"),
        ({"report_bytes": 0}, "report_bytes must be positive"),
        ({"memory": "1GiB"}, "Extra inputs are not permitted"),
        ({"isolate": False, "wall_time": 3}, "execution limits require isolate=True"),
    ],
)
def test_invalid_limits_are_reported_under_resources(section, message):
    with pytest.raises(ValidationError, match=message) as error:
        QAVFSettings.model_validate({"resources": section})
    assert error.value.errors()[0]["loc"][0] == "resources"


def test_engine_byte_settings_accept_sizes_but_not_booleans():
    assert EngineSettings(memory_budget_bytes="2GiB").memory_budget_bytes == 2 * 1024**3
    assert EngineSettings(spill_max_bytes="0").spill_max_bytes == 0
    for value in (True, "two gigs", -1):
        with pytest.raises(ValidationError):
            EngineSettings(memory_budget_bytes=value)


# ------------------------------------------------------------------ conflicts
def test_a_limit_in_resources_and_engine_must_agree_and_the_error_names_both(tmp_path):
    path = _project(
        tmp_path,
        "resources:\n  target_memory: 2GiB\nengine:\n  memory_budget_bytes: 1073741824\n",
    )
    with pytest.raises(ValueError, match="given different values") as error:
        load_settings(path)
    text = str(error.value)
    assert "resources.target_memory (qavf.yaml:2)" in text
    assert "engine.memory_budget_bytes (qavf.yaml:4)" in text


def test_the_same_limit_written_twice_with_one_value_is_accepted(tmp_path):
    path = _project(
        tmp_path,
        "resources: {target_memory: 2GiB, isolate: true}\n"
        "engine: {memory_budget_bytes: 2147483648, execution_mode: isolated}\n",
    )
    settings = load_settings(path)
    assert settings.engine.memory_budget_bytes == 2 * 1024**3
    (place,) = settings.resource_provenance.places("memory_budget_bytes")
    assert place.source == "resources"


def test_a_process_limit_conflicts_with_an_explicit_in_process_mode():
    with pytest.raises(ValidationError, match=r"engine.execution_mode"):
        QAVFSettings.model_validate(
            {"resources": {"wall_time": 3}, "engine": {"execution_mode": "in_process"}}
        )


def test_the_resources_argument_and_the_section_must_agree(tmp_path):
    path = _project(tmp_path, "resources:\n  target_memory: 2GiB\n" + DATASET)
    with pytest.raises(ValueError, match="given different values") as error:
        validate(config=path, resources=ResourceLimits(target_memory="1GiB"), backend="duckdb")
    text = str(error.value)
    assert "resources.target_memory (qavf.yaml:2)" in text
    assert "resources=ResourceLimits(target_memory=...)" in text
    report = validate(
        config=path, resources=ResourceLimits(target_memory=2 * 1024**3), backend="duckdb"
    )
    assert report.metadata.resource_usage["limits"]["target_memory"]["source"] == "resources"


def test_the_resources_argument_and_an_engine_setting_must_agree():
    config = {"engine": {"spill_max_bytes": 100}, "datasets": {"data": {}}}
    with pytest.raises(ValueError, match=r"engine.spill_max_bytes"):
        explain({"data": "x.csv"}, config=config, resources=ResourceLimits(max_spill=200))
    plan = explain({"data": "x.csv"}, config=config, resources=ResourceLimits(max_workers=2))
    limits = {limit.name: limit for limit in plan.limits}
    assert limits["max_workers"].source == "argument"
    assert limits["max_spill"].source == "engine"
    assert limits["target_memory"].source == "default"


# ---------------------------------------------------------------------- reuse
def test_include_vars_and_overrides_work_on_resources(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "limits.yaml").write_text(
        "resources:\n  max_spill: 20GiB\n  temp_dir: spill\n", encoding="utf-8"
    )
    path = _project(
        tmp_path,
        "include: [shared/limits.yaml]\n"
        "vars: {memory: 2GiB}\n"
        "resources:\n  target_memory: ${vars.memory}\n"
        "overrides:\n  resources: {max_spill: 30GiB}\n" + DATASET,
    )
    settings = load_settings(path)
    assert settings.engine.memory_budget_bytes == 2 * 1024**3
    assert settings.engine.spill_max_bytes == 30 * 1024**3
    assert Path(settings.engine.temp_directory).as_posix() == "shared/spill"
    (applied,) = settings.applied_overrides
    assert (applied.section, applied.instance, applied.fields) == (
        "resources",
        "resources",
        ("max_spill",),
    )
    assert "resources: max_spill" in applied.describe()
    origins = settings.origins
    assert origins is not None
    assert "via vars.memory" in origins.find(("engine", "memory_budget_bytes")).describe()
    assert "override resources" in origins.find(("engine", "spill_max_bytes")).describe()
    assert origins.find(("resources", "temp_dir")).file == "shared/limits.yaml"


def test_an_override_patches_resources_the_file_did_not_write():
    settings = QAVFSettings.model_validate({"overrides": {"resources": {"max_workers": 2}}})
    assert settings.engine.max_workers == 2


@pytest.mark.parametrize("patch", [{}, {"target_memory": None}, "2GiB"])
def test_an_empty_or_null_resources_override_is_refused(patch):
    with pytest.raises(ValidationError, match=r"overrides\.resources"):
        QAVFSettings.model_validate({"overrides": {"resources": patch}})


def test_a_resources_temp_dir_is_rebased_but_not_protected(tmp_path):
    path = _project(tmp_path, "resources: {temp_dir: work}\n" + DATASET)
    protected = ProtectedInputs.of_settings(load_settings(path), path)
    protected.check(tmp_path / "work")  # a working directory is not an input
    with pytest.raises(ValueError, match="would replace the input"):
        protected.check(tmp_path / "data.csv")


def test_resolve_shows_the_section_and_the_settings_it_set_with_origins(tmp_path, capsys):
    path = _project(
        tmp_path,
        "resources:\n  target_memory: 2GiB\nengine:\n  backend: duckdb\n" + DATASET,
    )
    assert main(["resolve", "-c", str(path), "--origins"]) == 0
    text = capsys.readouterr().out
    assert "# Backend: duckdb (engine.backend: duckdb at qavf.yaml:4:3)" in text
    assert "target_memory: 2GiB  # qavf.yaml:2:3" in text
    assert "memory_budget_bytes: 2147483648  # qavf.yaml:2:3 (from resources.target_memory)" in text
    document = json.loads(resolve(path).render("json", origins=True))
    assert document["backend"] == {
        "backend": "duckdb",
        "requested": "duckdb",
        "source": "config",
        "origin": "qavf.yaml:4:3",
    }


# -------------------------------------------------------------------- backend
@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_engine_backend_pins_the_backend(tmp_path, backend):
    path = _project(tmp_path, f"engine: {{backend: {backend}}}\n" + DATASET)
    report = validate(config=path)
    assert (report.metadata.backend, report.metadata.backend_source) == (backend, "config")
    plan = explain(config=path)
    assert (plan.backend, plan.backend_source) == (backend, "config")
    assert plan.backend_reason == f"{backend} (engine.backend: {backend} at qavf.yaml:1:10)"


@pytest.mark.parametrize(("requested", "expected"), [("polars", "polars"), ("duckdb", "duckdb")])
def test_an_argument_overrides_the_configured_backend(tmp_path, requested, expected):
    other = "duckdb" if requested == "polars" else "polars"
    path = _project(tmp_path, f"engine: {{backend: {other}}}\n" + DATASET)
    report = validate(config=path, backend=requested)
    assert (report.metadata.backend, report.metadata.backend_source) == (expected, "argument")
    assert explain(config=path, backend=requested).backend_source == "argument"


def test_an_explicit_auto_overrides_a_pinned_backend():
    choice = BackendChoice.resolve("auto", ConfiguredBackend("duckdb"))
    assert (choice.backend, choice.requested, choice.source) == ("polars", "auto", "argument")
    assert BackendChoice.resolve(None, ConfiguredBackend()).source == "auto"


def test_cli_backend_overrides_the_configuration_and_reports_record_it(tmp_path, capsys):
    path = _project(tmp_path, "engine: {backend: duckdb}\n" + DATASET)
    assert main(["run", "-c", str(path)]) == 0
    first = json.loads((tmp_path / "qa_output" / "qa_report.json").read_text(encoding="utf-8"))
    assert (first["metadata"]["backend"], first["metadata"]["backend_source"]) == (
        "duckdb",
        "config",
    )
    html = (tmp_path / "qa_output" / "qa_report.html").read_text(encoding="utf-8")
    assert "backend duckdb (config)" in html
    assert main(["run", "-c", str(path), "--backend", "polars"]) == 0
    second = json.loads((tmp_path / "qa_output" / "qa_report.json").read_text(encoding="utf-8"))
    assert (second["metadata"]["backend"], second["metadata"]["backend_source"]) == (
        "polars",
        "cli",
    )
    capsys.readouterr()
    assert main(["explain", "-c", str(path), "--backend", "polars"]) == 0
    assert "Backend: polars (--backend polars)" in capsys.readouterr().out


def test_a_pinned_core_backend_runs_the_native_capability_check(tmp_path):
    path = _project(
        tmp_path,
        "engine: {backend: duckdb}\n"
        + DATASET
        + "    checks:\n      - {check: outliers, column: v}\n",
    )
    with pytest.raises(OptionalDependencyError, match="outliers check"):
        validate(config=path)
    with pytest.raises(OptionalDependencyError, match="outliers check"):
        explain(config=path)


def test_a_pinned_polars_backend_without_the_extra_fails_at_load(tmp_path):
    path = _project(tmp_path, "engine:\n  backend: polars\n" + DATASET)
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    for command in (["run", "-c", str(path)], ["resolve", "-c", str(path)]):
        result = subprocess.run(
            [sys.executable, "-c", _NO_POLARS, *command],
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 2, result.stderr
        assert "engine.backend: polars (qavf.yaml:2:3)" in result.stderr
        assert "qavf[polars]" in result.stderr
        assert "Traceback" not in result.stderr
    assert not (tmp_path / "qa_output").exists()


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValidationError, match="backend"):
        EngineSettings(backend="spark")
    with pytest.raises(ValueError, match="backend must be"):
        BackendRequest.of("spark")


# ------------------------------------------------------------ explain / report
def test_explain_lists_every_limit_with_its_source_and_origin(tmp_path, capsys):
    path = _project(
        tmp_path,
        "resources:\n  target_memory: 2GiB\nengine:\n  max_workers: 2\n" + DATASET,
    )
    plan = explain(config=path, backend="duckdb")
    limits = {limit.name: limit for limit in plan.limits}
    assert set(limits) == set(ResourceLimits.ENGINE_FIELDS)
    memory = limits["target_memory"]
    assert (memory.value, memory.source, memory.written, memory.origin) == (
        2 * 1024**3,
        "resources",
        ("resources.target_memory",),
        "qavf.yaml:2:3",
    )
    assert (limits["max_workers"].source, limits["max_workers"].origin) == (
        "engine",
        "qavf.yaml:4:3",
    )
    assert limits["max_spill"].source == "default"
    assert main(["explain", "-c", str(path), "--backend", "duckdb"]) == 0
    text = capsys.readouterr().out
    assert (
        "target_memory (engine.memory_budget_bytes) = 2147483648 "
        "(resources.target_memory at qavf.yaml:2:3)" in text
    )
    assert "Backend: duckdb (--backend duckdb)" in text


def test_an_isolated_report_records_the_parent_provenance(tmp_path):
    path = _project(tmp_path, "engine: {backend: duckdb}\nresources: {wall_time: 120}\n" + DATASET)
    report = validate(config=path)
    assert report.metadata.resource_usage["execution_mode"] == "isolated"
    assert (report.metadata.backend, report.metadata.backend_source) == ("duckdb", "config")
    limits = report.metadata.resource_usage["limits"]
    assert limits["wall_time"] == {
        "value": 120,
        "source": "resources",
        "written": ["resources.wall_time"],
        "origin": "qavf.yaml:2:13",
    }


# ------------------------------------------------------- one shared vocabulary
def test_the_comparison_spec_and_the_project_share_one_resources_model(tmp_path):
    from qavf.cli import _comparison_spec

    section = {"target_memory": "64MiB", "max_spill": "1GiB", "evidence_rows": 5}
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "keys: [id]\nresources: {target_memory: 64MiB, max_spill: 1GiB, evidence_rows: 5}\n",
        encoding="utf-8",
    )
    limits = _comparison_spec(spec)["resources"]
    assert limits == ResourceSpec.model_validate(section).limits()
    project = QAVFSettings.model_validate({"resources": section})
    assert {
        field: getattr(project.engine, field) for field in limits.engine_overrides()
    } == limits.engine_overrides()
    assert QAVFSettings.model_fields["resources"].annotation == ResourceSpec | None
