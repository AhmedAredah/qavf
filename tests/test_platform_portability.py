"""Portable filesystem and child-process contracts; run on each real OS."""

import json

import polars as pl
import pytest

from qavf import DataContext, File, ResourceLimits, Status, validate
from qavf.sources.base import SourceEnvironment
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
@pytest.mark.parametrize("descriptor", ["path", "string", "file"])
def test_unicode_inputs_and_atomic_report_replacement(tmp_path, suffix, descriptor):
    directory = tmp_path / "résultats العربية with spaces"
    directory.mkdir()
    source = directory / f"entrée.{suffix}"
    frame = pl.DataFrame({"value": [1, -1, 3]})
    getattr(frame, f"write_{suffix}")(source)
    data = {"path": source, "string": str(source), "file": File(source)}[descriptor]
    report = validate(data, schema={"columns": {"value": {"ge": 0}}})
    assert report.exit_code() == 1
    assert any(result.status is Status.FAIL for result in report.results)
    assert all(result.status is not Status.ERROR for result in report.results)
    target = directory / "résumé.json"
    for _ in range(2):
        report.write_json(target)
        assert json.loads(target.read_text(encoding="utf-8"))
    target.rename(directory / "renamed.json")
    source.unlink()  # Windows detects leaked handles here.


def test_spill_cleanup_preserves_caller_directory(tmp_path):
    directory = tmp_path / "mémoire العربية"
    directory.mkdir()
    sentinel = directory / "caller.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    source = tmp_path / "input.csv"
    pl.DataFrame({"x": range(1000)}).write_csv(source)
    reader = FileSource(FileParams(path=str(source)), SourceEnvironment())
    with DataContext(sources={"data": reader}, temp_directory=directory,
                     cache_max_bytes=0) as ctx:
        assert ctx.get("data").shape == (1000, 1)
        assert ctx.resource_usage()["owned_spill_bytes"] > 0
        assert list(directory.rglob("*.arrow")), "fixture must create an actual owned spill"
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(directory.iterdir()) == [sentinel]


def test_isolated_validation_uses_unicode_paths(tmp_path):
    source = tmp_path / "entrée العربية.csv"
    pl.DataFrame({"value": [1, 2]}).write_csv(source)
    report = validate(source, schema={"columns": {"value": {"ge": 0}}},
                      resources=ResourceLimits(isolate=True, wall_time=15,
                                               temp_dir=tmp_path / "temporary é"))
    assert report.exit_code() == 0
    source.unlink()
