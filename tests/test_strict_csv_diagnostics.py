"""Strict CSV anchoring and record/physical-line diagnostics share one policy."""

from __future__ import annotations

import gzip
from functools import partial

import pytest

from qavf import Dataset, File, validate
from qavf.engine.severity import Status
from qavf.sources.csv_diagnostics import CsvPolicy


@pytest.fixture(params=["polars", "duckdb"], autouse=True)
def _validation_backend(request, monkeypatch):
    monkeypatch.setattr(request.module, "validate", partial(validate, backend=request.param))


def _errors(path, **options):
    source = File(str(path), options=options)
    report = validate({"t": Dataset(source, schema={"min_rows": 0})})
    return [result.message for result in report.results if result.status is Status.ERROR]


@pytest.mark.parametrize("strict_options", [{}, {"strict_mode": True}])
@pytest.mark.parametrize(
    "last,reason", [("3,4,5\n", "too many columns"), ("3\n", "missing columns")]
)
def test_ragged_prefix_cannot_be_sniffed_away(tmp_path, strict_options, last, reason):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,2\n" + last)
    errors = _errors(path, delim=",", header=True, **strict_options)
    assert errors
    assert all("CSV row 3, line 3" in error and reason in error for error in errors)
    assert all("expected 2 fields" in error.lower() for error in errors)


@pytest.mark.parametrize("explicit_layout", [False, True])
def test_unterminated_quote_is_not_an_unquoted_value(tmp_path, explicit_layout):
    path = tmp_path / "t.csv"
    path.write_text('a,b\n1,2\n3,"unfinished\nstill unfinished\n')
    options = {"delim": ",", "header": True, "strict_mode": True, "quote": '"'}
    if explicit_layout:
        options.update(auto_detect=False, columns={"a": "VARCHAR", "b": "VARCHAR"})
    errors = _errors(path, **options)
    assert errors
    assert all("CSV row 3, line 3" in error and "line 4" in error for error in errors)
    assert all("unterminated" in error and "closing quote" in error for error in errors)
    assert all(
        "unquoted value" not in error and "Set options {delim" not in error for error in errors
    )


def test_multiline_record_does_not_shift_physical_line(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('a,b\n1,"two\nlines"\n3,4,5\n')
    errors = _errors(path, delim=",", header=True, quote='"', strict_mode=True)
    assert errors and all("CSV row 3, line 4" in error for error in errors)


def test_explicit_skip_and_comment_preserve_physical_positions(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("metadata\n# comment\na;b\n1;2\n3;4;5\n")
    options = {"skip": 1, "comment": "#", "delim": ";", "header": True}
    issue = CsvPolicy.diagnose(str(path), CsvPolicy.options(options))
    assert issue is not None and (issue.row, issue.line) == (3, 5)
    path.write_text("metadata\n# comment\na;b\n1;2\n")
    report = validate(
        {
            "t": Dataset(
                File(str(path), options=options, acknowledge_lossy=True), schema={"min_rows": 1}
            )
        }
    )
    assert not any(result.status in {Status.FAIL, Status.ERROR} for result in report.results)


@pytest.mark.parametrize("suffix", [".gz", ".gzip", ".zst", ".zstd"])
def test_compressed_multiline_failure_has_uncompressed_line_number(tmp_path, suffix):
    import pyarrow as pa

    path = tmp_path / ("t.csv" + suffix)
    content = b'a,b\n1,"two\nlines"\n3,4,5\n'
    path.write_bytes(
        gzip.compress(content)
        if suffix.startswith(".g")
        else bytes(pa.compress(content, codec="zstd"))
    )
    errors = _errors(path)
    assert errors and all("CSV row 3, line 4" in error for error in errors)


def test_clean_input_never_runs_diagnostic_parser(tmp_path, monkeypatch):
    def no_diagnostic(*_args, **_kwargs):
        raise AssertionError("clean reads must not run a second full parser")

    monkeypatch.setattr(CsvPolicy, "diagnose", no_diagnostic)
    path = tmp_path / "t.csv"
    path.write_text('a,b\n1,"two\nlines"\n2,three\n')
    assert _errors(path, delim=",", header=True) == []


def test_policy_preserves_explicit_reader_choices():
    assert CsvPolicy.options({}) == {"skip": 0, "quote": '"'}
    assert CsvPolicy.options({"skip": 4}) == {"skip": 4, "quote": '"'}
    assert CsvPolicy.options({"strict_mode": False}) == {"strict_mode": False}


def test_diagnostics_preserve_quote_escape_and_padding(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a;b\n1;'escaped\\' quote'\n2\n")
    options = {"delim": ";", "quote": "'", "escape": "\\", "null_padding": True}
    assert CsvPolicy.diagnose(str(path), options) is None
    options["null_padding"] = False
    issue = CsvPolicy.diagnose(str(path), options)
    assert issue is not None and (issue.code, issue.row, issue.line) == ("MISSING COLUMNS", 3, 3)


def test_diagnostic_preview_is_bounded(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('a,b\n1,"' + "x" * 1000)
    issue = CsvPolicy.diagnose(str(path), {"delim": ","})
    assert issue is not None and len(issue.text) == 200


def test_comment_markers_inside_quotes_and_inline_comments_keep_locations(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('a,b\n1,2# ignored, fields\n3,"two\n# quoted text"\n4,5,6\n')
    issue = CsvPolicy.diagnose(str(path), {"delim": ",", "comment": "#", "quote": '"'})
    assert issue is not None and (issue.row, issue.line) == (4, 5)


def test_ambiguous_single_column_remains_valid(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text('"a,b"\n"one,two"\n')
    assert _errors(path, header=True) == []


@pytest.mark.parametrize(
    "record,reason",
    [('1,"unterminated\n', "quote"), ("1,2,3\n", "too many columns"), ("1\n", "missing columns")],
)
def test_default_strict_mode_anchors_first_record_dialect(tmp_path, record, reason):
    path = tmp_path / "default.csv"
    path.write_text("id,value\n" + record)
    errors = _errors(path)
    assert errors and all("CSV row 2, line 2" in error and reason in error for error in errors)


@pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
def test_default_strict_mode_keeps_unambiguous_common_dialects(tmp_path, delimiter):
    path = tmp_path / "data.csv"
    path.write_text(f"id{delimiter}value\n1{delimiter}2\n")
    report = validate(path, schema={"columns": {"id": {"dtype": "int"}, "value": {"dtype": "int"}}})
    assert report.verdict() == "passed"


def test_single_quote_dialect_is_available_as_an_explicit_option(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("id;value\n1;'two\nlines'\n")
    assert _errors(path, quote="'", header=True) == []


def test_disabling_quotes_explicitly_keeps_literal_quote_text(tmp_path):
    path = tmp_path / "literal.csv"
    path.write_text('id,value\n1,"literal\n')
    assert _errors(path, quote="", delim=",", header=True) == []


def test_first_record_probe_is_bounded_and_preserves_ambiguity(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text("a,b;c\n1,2;3\n")
    assert "delim" not in CsvPolicy.options({}, str(path))
    path.write_text("a" * (65 * 1024) + ",b\n")
    assert "delim" not in CsvPolicy.options({}, str(path))


def test_existing_literal_filename_with_glob_characters_is_read_exactly(tmp_path):
    path = tmp_path / "data[1].csv"
    path.write_text('id,value\n1,"unterminated\n')
    errors = _errors(path)
    assert errors and all("CSV row 2, line 2" in error for error in errors)


def test_explicit_headerless_csv_keeps_all_records(tmp_path):
    path = tmp_path / "headerless.csv"
    path.write_text("1,2\n3,4\n")
    report = validate(File(path, options={"header": False}), schema={"min_rows": 2, "max_rows": 2})
    assert report.verdict() == "passed"


def test_explicit_native_padding_keeps_nullable_values(tmp_path):
    path = tmp_path / "padding.csv"
    path.write_text("id,value\n1,2\n3\n")
    report = validate(
        File(path, options={"null_padding": True, "header": True}), schema={"min_rows": 2}
    )
    assert report.verdict() == "passed"


def test_unterminated_first_record_is_located(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text('id,"value\n')
    errors = _errors(path)
    assert errors and all("CSV row 1, line 1" in error and "quote" in error for error in errors)
