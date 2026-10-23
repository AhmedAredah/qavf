"""Output path collisions and optional dependency coverage."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import polars as pl
import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from qavf.cli import EXIT_USAGE, _check_report_destinations, main


@pytest.mark.parametrize("command", ["run", "compare-datasets", "default-json"])
def test_report_collision_preserves_existing_file(tmp_path, capsys, command):
    data = tmp_path / "data.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(data)
    target = tmp_path / "qa_report.json"
    target.write_text("previous report", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "datasets": {"data": {"path": str(data), "schema": {"columns": {"x": {"ge": 0}}}}},
        "plugins": {"entry_points": False},
        "report": {"output_dir": str(tmp_path)},
    }), encoding="utf-8")
    if command == "compare-datasets":
        arguments = [command, str(data), str(data), "--no-plugins"]
    else:
        arguments = ["run", "-c", str(config)]
    arguments += ["--html", str(target)]
    if command != "default-json":
        arguments += ["--json", str(target.parent / "." / target.name)]
    assert main(arguments) == EXIT_USAGE
    assert target.read_text(encoding="utf-8") == "previous report"
    assert "different output paths" in capsys.readouterr().err


def test_dependency_floor_recipe_covers_core_and_every_extra():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declarations = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        declarations.extend(extra)
    requirements = [Requirement(item) for item in declarations]
    requirements = [item for item in requirements if canonicalize_name(item.name) != "qavf"]
    pins = {}
    for line in (root / "tests/dependency-floors.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            item = Requirement(line)
            specifiers = list(item.specifier)
            assert len(specifiers) == 1 and specifiers[0].operator == "=="
            pins[canonicalize_name(item.name)] = specifiers[0].version
    assert set(pins) == {canonicalize_name(item.name) for item in requirements}
    for item in requirements:
        assert pins[canonicalize_name(item.name)] in item.specifier


def test_output_resolution_loop_is_a_usage_error(monkeypatch):
    def loop(self):
        raise RuntimeError("Symlink loop")

    monkeypatch.setattr(Path, "resolve", loop)
    with pytest.raises(ValueError, match="cannot resolve report output path"):
        _check_report_destinations(Path("loop.html"), Path("other.json"))
