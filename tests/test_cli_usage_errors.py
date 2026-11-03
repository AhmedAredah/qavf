"""CLI usage and configuration errors: exit 2 with the field path, without internals.

Also the standalone ``compare-datasets`` spec: its ``resources`` mapping and the
folder its relative paths resolve against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.inputs.resources import ResourceLimits

pytest.importorskip("polars")


def _pair(tmp_path: Path) -> tuple[str, str]:
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("id,v\n1,2\n2,3\n", encoding="utf-8")
    return str(tmp_path / "a.csv"), str(tmp_path / "b.csv")


def test_comparison_spec_accepts_a_resources_mapping(tmp_path: Path, capsys) -> None:
    baseline, candidate = _pair(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text("keys: [id]\nresources: {target_memory: 64MiB}\n", encoding="utf-8")
    assert main(["compare-datasets", baseline, candidate, "-c", str(spec)]) == EXIT_OK
    assert "COMPARISON PASSED" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        ("5", "resources: must be a mapping"),
        ("{target_memory: lots}", "target_memory must be a byte count"),
        ("{memory: 1GiB}", "resources.memory: unknown resource limit"),
    ],
)
def test_invalid_comparison_resources_are_usage_errors_with_the_field(
    tmp_path: Path, capsys, resources: str, expected: str
) -> None:
    baseline, candidate = _pair(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text(f"keys: [id]\nresources: {resources}\n", encoding="utf-8")
    assert main(["compare-datasets", baseline, candidate, "-c", str(spec)]) == EXIT_USAGE
    error = capsys.readouterr().err
    assert expected in error and "spec.yaml" in error
    assert "Traceback" not in error and ".py:" not in error


def test_resource_mapping_rebases_a_relative_temp_dir(tmp_path: Path) -> None:
    limits = ResourceLimits.from_mapping({"temp_dir": "spill"}, base_dir=tmp_path)
    assert limits.temp_dir == tmp_path / "spill"
    assert ResourceLimits.from_mapping({"temp_dir": "spill"}).temp_dir == "spill"


def test_comparison_spec_paths_resolve_from_the_spec_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from qavf.cli import _comparison_spec

    folder = tmp_path / "specs"
    folder.mkdir()
    spec = folder / "spec.yaml"
    spec.write_text(
        "keys: [id]\nlevels:\n  - level: spatial\n    layer: {path: zones/z.geojson, id: zone}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    options = _comparison_spec(spec)
    zones = options["levels"][0]["layer"]["path"]
    assert Path(zones).as_posix() == "specs/zones/z.geojson"


def test_usage_errors_omit_the_internal_location_unless_verbose(tmp_path: Path, capsys) -> None:
    baseline, candidate = _pair(tmp_path)
    spec = tmp_path / "spec.yaml"
    spec.write_text("keys: [id]\nbogus: 1\n", encoding="utf-8")
    arguments = ["compare-datasets", baseline, candidate, "-c", str(spec)]
    assert main(arguments) == EXIT_USAGE
    quiet = capsys.readouterr().err
    assert "bogus" in quiet and " at " not in quiet and ".py:" not in quiet
    assert main(["-v", *arguments]) == EXIT_USAGE
    assert ".py:" in capsys.readouterr().err


def test_config_validation_error_names_the_field_without_internals(tmp_path: Path, capsys) -> None:
    config = tmp_path / "qavf.yaml"
    config.write_text("project: x\nrun: {fail_on: NOPE}\n", encoding="utf-8")
    assert main(["run", "-c", str(config), "--backend", "duckdb"]) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "run.fail_on" in error and "main.py" not in error and ".py:" not in error
