"""CLI backend routing and discoverable contracts must not require Polars."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qavf import BackendCapabilities, backend_capabilities
from qavf.cli import main
from qavf.optional import OptionalDependencyError, unsupported

_ROOT = Path(__file__).resolve().parents[1]
_NO_OPTIONAL = r"""
import importlib.abc, runpy, sys
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"polars", "pandas"}:
            raise ModuleNotFoundError("optional dependency absent", name=fullname)
sys.meta_path.insert(0, CoreOnly())
sys.argv = ["qavf", *sys.argv[1:]]
try:
    runpy.run_module("qavf.cli", run_name="__main__")
finally:
    assert not {"polars", "pandas"} & sys.modules.keys()
"""


def _core_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", _NO_OPTIONAL, *args],
        text=True,
        capture_output=True,
        timeout=40,
        env=env,
    )


@pytest.mark.parametrize(
    "args",
    [("--help",), ("--version",), ("run", "--help"), ("compare", "--help"), ("schema", "--help")],
)
def test_help_and_version_without_optional_backend(args):
    result = _core_cli(*args)
    assert result.returncode == 0, result.stderr
    assert "qavf" in result.stdout


def test_cli_import_and_catalog_do_not_initialize_engines():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qavf.cli; from qavf import backend_capabilities; "
            "assert backend_capabilities('duckdb').backend == 'duckdb'; "
            "assert not {'polars', 'pandas', 'duckdb', 'pyarrow', 'pydantic'} & sys.modules.keys()",
        ],
        text=True,
        capture_output=True,
        timeout=20,
        cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr


def test_native_schema_and_rule_listing_without_optional_backend(tmp_path):
    destination = tmp_path / "qavf.schema.json"
    result = _core_cli("schema", "-o", str(destination))
    assert result.returncode == 0, result.stderr
    schema = json.loads(destination.read_text())
    assert schema["x-qavf-backend"] == "duckdb"
    assert "properties" in schema and "$defs" in schema
    rules = _core_cli("list-checks", "--format", "json")
    assert rules.returncode == 0, rules.stderr
    assert {rule["rule_id"] for rule in json.loads(rules.stdout)} == set(
        backend_capabilities("duckdb").rule_ids
    )


def test_native_run_writes_reports_and_exit_status_without_optional_backend(tmp_path):
    (tmp_path / "source.csv").write_text("v\n1\n-1\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "datasets": {
                    "data": {
                        "path": "source.csv",
                        "schema": {
                            "columns": {"v": {"dtype": "int", "ge": 0}},
                        },
                    }
                },
                "report": {"output_dir": "reports"},
            }
        )
    )
    result = _core_cli("run", "-c", str(config), "--no-parallel", "--fail-on", "ERROR")
    assert result.returncode == 1, result.stderr
    assert "QA FAILED" in result.stdout
    report = json.loads((tmp_path / "reports" / "qa_report.json").read_text())
    assert report["metadata"]["resource_usage"]["backend"] == "duckdb"
    assert report["summary"]["failed"] == 1
    assert (tmp_path / "reports" / "qa_report.html").is_file()


@pytest.mark.parametrize(
    "args", [("profile", "unused.csv"), ("run", "--backend", "polars", "-c", "unused.json")]
)
def test_advanced_cli_missing_backend_has_actionable_error(args, tmp_path):
    if args[0] == "run":
        config = tmp_path / "config.json"
        config.write_text('{"datasets":{"data":{"path":"unused.csv"}}}')
        args = (*args[:-1], str(config))
    result = _core_cli(*args)
    assert result.returncode == 2
    assert "qavf[polars]" in result.stderr
    assert "Traceback" not in result.stderr


def test_installed_backend_error_requests_selection_not_reinstallation(monkeypatch):
    monkeypatch.setattr("qavf.optional.has_polars", lambda: True)
    with pytest.raises(OptionalDependencyError, match="backend='polars'") as error:
        unsupported("custom rules")
    assert "install" not in str(error.value)


def test_capabilities_cli_and_public_api_are_json_ready(capsys):
    assert main(["capabilities", "--backend", "duckdb"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["duckdb"]["rule_ids"] == list(backend_capabilities().rule_ids)
    assert isinstance(backend_capabilities(), BackendCapabilities)
    first = backend_capabilities().to_dict()
    first["sources"][0]["supported"] = False
    assert backend_capabilities().sources[0].supported
    with pytest.raises(ValueError, match="backend"):
        backend_capabilities("auto")


def test_capability_catalog_covers_builtin_full_rules():
    from qavf.engine.compilation import build_registry
    from qavf.settings import QAVFSettings

    registry = build_registry(QAVFSettings(plugins={"entry_points": False}))
    assert set(backend_capabilities("polars").rule_ids) == {r.rule_id for r in registry}
