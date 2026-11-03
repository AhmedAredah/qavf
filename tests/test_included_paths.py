"""Relative file paths in an included file are relative to that file.

Files merge into the root configuration, whose directory relative paths resolve
against. A path written in an included file is therefore rewritten to the same
file seen from the root: sources, dataset shorthands, connections, baselines,
zone layers and plugin paths. Absolute paths, URLs and ``${...}`` references stay
as written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qavf import validate
from qavf.composition import IncludedPaths
from qavf.settings import load_settings


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SHARED = """
datasets:
  counts:
    path: data/counts.csv
    checks:
      - {check: matches_baseline, path: baselines/counts.csv, keys: [id]}
  sites:
    source: {type: file, path: data/sites.csv}
  remote:
    source: {type: file, path: 's3://bucket/x.parquet'}
  templated:
    path: '${DATA_DIR}/t.csv'
connections:
  warehouse: {type: duckdb, path: db/warehouse.duckdb}
plugins:
  paths: [plugins]
"""


def test_paths_in_an_included_file_are_relative_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_DIR", "/abs")
    _write(tmp_path / "shared" / "base.yaml", SHARED)
    config = _write(
        tmp_path / "qa.yaml",
        "include: [shared/base.yaml]\ndatasets:\n  own: {path: data/own.csv}\n",
    )
    settings = load_settings(config)
    assert settings.datasets["counts"].path == "shared/data/counts.csv"
    assert settings.datasets["sites"].source == {"type": "file", "path": "shared/data/sites.csv"}
    assert settings.datasets["remote"].source == {"type": "file", "path": "s3://bucket/x.parquet"}
    assert settings.datasets["templated"].path == "/abs/t.csv"
    assert settings.datasets["own"].path == "data/own.csv"
    assert settings.connections["warehouse"].path == "shared/db/warehouse.duckdb"
    assert [p.as_posix() for p in settings.plugins.paths] == ["shared/plugins"]
    baseline = settings.datasets["counts"].checks[0]
    assert baseline.settings["path"] == "shared/baselines/counts.csv"


def test_nested_includes_rebase_each_file_from_its_own_directory(tmp_path: Path) -> None:
    _write(tmp_path / "a" / "b" / "inner.yaml", "datasets:\n  inner: {path: x.csv}\n")
    _write(
        tmp_path / "a" / "outer.yaml",
        "include: [b/inner.yaml]\ndatasets:\n  outer: {path: ../up.csv}\n",
    )
    config = _write(tmp_path / "qa.yaml", "include: [a/outer.yaml]\n")
    settings = load_settings(config)
    assert settings.datasets["inner"].path == "a/b/x.csv"
    assert settings.datasets["outer"].path == "up.csv"


def test_zone_layer_paths_in_included_checks_are_rebased(tmp_path: Path) -> None:
    rebased = IncludedPaths(tmp_path / "shared", tmp_path).rebase(
        {
            "checks": [
                {
                    "check": "compare",
                    "datasets": ["a", "b"],
                    "levels": [{"level": "spatial", "layer": {"path": "zones.geojson"}}],
                }
            ]
        }
    )
    assert rebased["checks"][0]["levels"][0]["layer"]["path"] == "shared/zones.geojson"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    absolute = (tmp_path / "elsewhere" / "x.csv").as_posix()
    paths = IncludedPaths(tmp_path / "shared", tmp_path)
    assert paths.path(absolute) == absolute
    assert paths.path("https://host/x.csv") == "https://host/x.csv"
    assert paths.path("${vars.root}/x.csv") == "${vars.root}/x.csv"


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_validation_reads_data_next_to_the_included_file(tmp_path: Path, backend: str) -> None:
    _write(tmp_path / "shared" / "data" / "rows.csv", "id,x\n1,2\n2,3\n")
    _write(
        tmp_path / "shared" / "base.yaml",
        "datasets:\n  rows:\n    path: data/rows.csv\n"
        "    schema: {columns: {x: {dtype: int, ge: 0}}, primary_key: [id]}\n",
    )
    config = _write(tmp_path / "qa.yaml", "include: [shared/base.yaml]\n")
    report = validate(config=config, backend=backend)
    assert report.verdict() == "passed"
    assert {d.name for d in report.metadata.datasets} == {"rows"}
