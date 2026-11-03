"""Output commands never replace a declared input (CLI-01, CLI-03, CLI-05).

The declared inputs of a configuration are the file itself, every file it
includes and each setting typed as a file path (:mod:`qavf.config_paths`):
data and database files, accepted baselines with their manifests, zone layers,
crosswalks and plugin files. ``run``, ``schema -o``, ``init -o`` and
``migrate -o`` refuse a destination naming one of them, by any alias, even with
``--force``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from qavf.cli import main
from qavf.io import ProtectedInputs
from qavf.settings import load_settings

EXIT_USAGE = 2


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(root: Path) -> Path:
    """A root configuration including ``shared.yaml``, with a baseline and a zone layer."""
    _write(root / "d.csv", "id,x\n1,2\n2,3\n")
    _write(root / "baseline" / "counts.csv", "id,x\n1,2\n2,3\n")
    _write(root / "zones.geojson", '{"type": "FeatureCollection", "features": []}')
    _write(
        root / "shared.yaml",
        "datasets:\n  d:\n    path: d.csv\n    schema: {columns: {x: {dtype: int}}}\n"
        "    checks:\n      - {check: matches_baseline, path: baseline/counts.csv, keys: [id]}\n",
    )
    return _write(
        root / "root.yaml",
        "include: shared.yaml\nreport: {html: false, json: false}\n"
        "checks:\n  - check: coverage\n    datasets: [d]\n    entity: id\n"
        "    zone_layer: {path: zones.geojson, id: id}\n    enabled: false\n",
    )


def _snapshot(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _relative(target: Path, root: Path) -> Path:
    return Path(os.path.relpath(target, Path.cwd())) if target.drive == Path.cwd().drive else target


def _alias(kind: str, target: Path, root: Path) -> Path:
    if kind == "absolute":
        return target.resolve()
    if kind == "relative":
        return _relative(target, root)
    link = root / "alias" / target.name
    link.parent.mkdir(exist_ok=True)
    try:
        (os.link if kind == "hardlink" else os.symlink)(target, link)
    except (OSError, NotImplementedError):
        pytest.skip(f"{kind}s are not available here")
    return link


ALIASES = ["absolute", "relative", "hardlink", "symlink"]
INPUTS = [
    "shared.yaml",
    "baseline/counts.csv",
    "baseline/counts.csv.manifest.json",
    "zones.geojson",
]


def test_the_manifest_lists_includes_baselines_and_zone_layers(tmp_path: Path) -> None:
    settings = load_settings(_project(tmp_path))
    protected = {Path(path).resolve() for path in ProtectedInputs.of_settings(settings).locations}
    for name in ("root.yaml", "shared.yaml", "d.csv", *INPUTS):
        assert (tmp_path / name).resolve() in protected, name


@pytest.mark.parametrize("kind", ALIASES)
@pytest.mark.parametrize("name", INPUTS)
def test_run_never_writes_a_report_over_a_declared_input(
    tmp_path: Path, name: str, kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _project(tmp_path)
    target = tmp_path / name
    if not target.exists() and kind in ("hardlink", "symlink"):
        # The manifest is written when a baseline is accepted; it is protected before.
        pytest.skip("a link needs an existing file")
    destination = _alias(kind, target, tmp_path)
    before = _snapshot(tmp_path)
    assert main(["run", "-c", str(config), "--json", str(destination)]) == EXIT_USAGE
    assert "would replace the input" in capsys.readouterr().err
    assert _snapshot(tmp_path) == before


def test_run_still_writes_an_ordinary_report(tmp_path: Path) -> None:
    config = _project(tmp_path)
    report = tmp_path / "out" / "report.json"
    main(["run", "-c", str(config), "--json", str(report)])
    assert json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("name", ["root.yaml", "shared.yaml", "d.csv"])
def test_schema_never_replaces_the_configuration_or_its_inputs(
    tmp_path: Path, name: str, force: bool
) -> None:
    config = _project(tmp_path)
    before = _snapshot(tmp_path)
    arguments = ["schema", "-c", str(config), "-o", str(tmp_path / name), "--backend", "duckdb"]
    assert main([*arguments, *(["--force"] if force else [])]) == EXIT_USAGE
    assert _snapshot(tmp_path) == before


def test_schema_overwrites_an_existing_file_only_with_force(tmp_path: Path) -> None:
    existing = _write(tmp_path / "qavf.schema.json", "old")
    assert main(["schema", "-o", str(existing), "--backend", "duckdb"]) == EXIT_USAGE
    assert existing.read_text(encoding="utf-8") == "old"
    assert main(["schema", "-o", str(existing), "--backend", "duckdb", "--force"]) == 0
    assert json.loads(existing.read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["absolute", "relative", "hardlink"])
def test_init_never_replaces_a_profiled_input_even_with_force(tmp_path: Path, kind: str) -> None:
    data = _write(tmp_path / "input.json", json.dumps([{"x": 1}, {"x": 2}]))
    output = _alias(kind, data, tmp_path)
    before = _snapshot(tmp_path)
    arguments = ["init", "-d", f"data={data}", "-o", str(output), "--no-schema", "--force"]
    assert main(arguments) == EXIT_USAGE
    assert _snapshot(tmp_path) == before


OLD = "datasets:\n  d:\n    path: d.csv\n    schema: {columns: {x: {dtype: int}}}\n"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_migrate_writes_the_format_of_the_output_suffix(tmp_path: Path, suffix: str) -> None:
    _write(tmp_path / "d.csv", "x\n1\n")
    old = _write(tmp_path / "old.yaml", OLD)
    new = tmp_path / f"new{suffix}"
    assert main(["migrate", str(old), "-o", str(new)]) == 0
    if suffix == ".json":
        assert json.loads(new.read_text(encoding="utf-8"))["datasets"]["d"]["path"] == "d.csv"
    assert load_settings(new).datasets["d"].path == "d.csv"


def test_migrate_to_another_directory_rebases_relative_paths(tmp_path: Path) -> None:
    _write(tmp_path / "d.csv", "x\n1\n")
    old = _write(tmp_path / "p.yaml", OLD + "    rename: {path: route}\n")
    new = tmp_path / "sub" / "p.yaml"
    new.parent.mkdir()
    assert main(["migrate", str(old), "-o", str(new)]) == 0
    moved = load_settings(new)
    assert moved.datasets["d"].path == "../d.csv"
    assert moved.resolve(moved.datasets["d"].path).resolve() == (tmp_path / "d.csv").resolve()
    assert moved.datasets["d"].rename == {"path": "route"}


@pytest.mark.parametrize("name", ["p.yaml", "d.csv"])
def test_migrate_never_replaces_its_inputs(tmp_path: Path, name: str) -> None:
    _write(tmp_path / "d.csv", "x\n1\n")
    old = _write(tmp_path / "p.yaml", OLD)
    before = _snapshot(tmp_path)
    assert main(["migrate", str(old), "-o", str(tmp_path / name)]) == EXIT_USAGE
    assert _snapshot(tmp_path) == before


def test_migrate_refuses_an_unknown_output_format(tmp_path: Path) -> None:
    old = _write(tmp_path / "p.yaml", OLD)
    assert main(["migrate", str(old), "-o", str(tmp_path / "p.txt")]) == EXIT_USAGE
    assert not (tmp_path / "p.txt").exists()


def test_a_comparison_spec_declares_its_zone_layers(tmp_path: Path) -> None:
    from qavf.compare.spec import ComparisonSpec
    from qavf.io import declared_files

    spec = {
        "keys": ["id"],
        "levels": [{"level": "spatial", "layer": {"path": "zones.geojson"}}],
        "columns": {"path": "route"},
    }
    assert declared_files(spec, tmp_path, ComparisonSpec) == [str(tmp_path / "zones.geojson")]
