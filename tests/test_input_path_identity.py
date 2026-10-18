"""A path's exact spelling selects the audited input, including whitespace."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import Dataset, File, Status, validate
from qavf.sources import FileParams, FileSource, SourceEnvironment

SCHEMA = {"columns": {"x": {"checks": [{"name": "nonnegative", "ge": 0}]}}}
CONFIG = {"plugins": {"entry_points": False}, "engine": {"parallel": False}}


@pytest.mark.parametrize("path", [" input.csv", "input.csv ", " input.csv ", "\tinput.csv\t"])
def test_file_parameters_preserve_exact_nonblank_location(path):
    params = FileParams(path=path, format="csv")
    assert params.path == path
    assert FileParams.model_validate(params.model_dump()).path == path


@pytest.mark.parametrize("path", ["", " ", "\t\r\n"])
def test_file_parameters_reject_blank_locations(path):
    with pytest.raises(ValidationError, match="non-whitespace"):
        FileParams(path=path, format="csv")


def test_file_parameters_do_not_expand_their_configured_string_contract():
    with pytest.raises(ValidationError):
        FileParams(path=Path("input.csv"))


@pytest.mark.parametrize(
    "kind", ["raw", "path_config", "source_config", "direct_source", "file", "wrapped_file"]
)
@pytest.mark.parametrize("filename", [" input.parquet", "input.parquet "])
def test_facade_reads_exact_padded_path_instead_of_valid_trimmed_neighbor(
    tmp_path, monkeypatch, kind, filename
):
    monkeypatch.chdir(tmp_path)
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / filename.strip())
    pl.DataFrame({"x": [-1]}).write_parquet(tmp_path / filename)
    if (tmp_path / filename).samefile(tmp_path / filename.strip()):
        pytest.skip("Filesystem does not distinguish this whitespace-padded filename")
    declaration = {"schema": SCHEMA}
    if kind == "raw":
        # Explicit format belongs to the existing file contract and survives a
        # raw path replacement, even when the replacement suffix has whitespace.
        declaration.update(path="unused.parquet", format="parquet")
        report = validate(filename, config={**CONFIG, "datasets": {"data": declaration}})
    elif kind == "path_config":
        declaration.update(path=filename, format="parquet")
        report = validate(config={**CONFIG, "datasets": {"data": declaration}})
    elif kind == "source_config":
        declaration["source"] = {"type": "file", "path": filename, "format": "parquet"}
        report = validate(config={**CONFIG, "datasets": {"data": declaration}})
    elif kind == "direct_source":
        source = FileSource(
            FileParams(path=filename, format="parquet"), SourceEnvironment(base_dir=tmp_path)
        )
        report = validate(source, schema=SCHEMA, config=CONFIG)
        assert source.params.path == filename
    else:
        descriptor = File(filename, format="parquet")
        assert descriptor.path == filename
        report = (
            validate(descriptor, schema=SCHEMA, config=CONFIG)
            if kind == "file"
            else validate(Dataset(descriptor, schema=SCHEMA), config=CONFIG)
        )
    assert report.exit_code() == 1
    failure = report.get("INP-003", "data.x: nonnegative")
    assert failure.status is Status.FAIL
    assert failure.affected_count == failure.total_records == 1
    assert report.metadata.datasets[0].source == str(tmp_path / filename)


def test_unrecognized_exact_suffix_requires_format_instead_of_reading_trimmed_path(tmp_path):
    trimmed = tmp_path / "input.parquet"
    intended = tmp_path / "input.parquet "
    pl.DataFrame({"x": [1]}).write_parquet(trimmed)
    pl.DataFrame({"x": [-1]}).write_parquet(intended)
    with pytest.raises(ValueError, match="cannot infer the data format"):
        validate(intended, schema=SCHEMA, config=CONFIG)


def test_leading_whitespace_retains_valid_format_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pl.DataFrame({"x": [1]}).write_parquet("input.parquet")
    pl.DataFrame({"x": [-1]}).write_parquet(" input.parquet")
    report = validate(" input.parquet", schema=SCHEMA, config=CONFIG)
    assert report.exit_code() == 1
    assert report.metadata.datasets[0].source == str(tmp_path / " input.parquet")
