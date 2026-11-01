"""Strict CSV policy and failure-only, bounded-memory structural diagnostics.

The normal path anchors native reader options with a bounded first-record
probe. Structural diagnosis runs only after DuckDB reports a CSV failure and
stops at the first malformed record. Importing this module uses only the Python
standard library; zstd diagnostics load the core PyArrow dependency on demand.
"""

from __future__ import annotations

import bz2
import csv
import glob
import gzip
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class CsvIssue:
    """A malformed logical record and its one-based physical line range.

    ``row`` counts nonempty CSV records after explicitly skipped lines, including
    a header if present. It is independent of QAVF's zero-based data-row indexes.
    """

    file: str
    code: str
    row: int
    line: int
    end_line: int
    text: str
    expected: int | None = None
    actual: int | None = None

    def message(self, source: str | None = None) -> str:
        """Explain the first structural error without claiming a total count."""
        where = f"CSV row {self.row:,}, line {self.line:,}"
        if self.end_line > self.line:
            where += f" (through line {self.end_line:,})"
        if source is not None and source != self.file:
            where += f" of {self.file}"
        if self.code == "UNTERMINATED QUOTE":
            reason = "unterminated quoted field; the closing quote is missing"
        elif self.code in {"TOO MANY COLUMNS", "MISSING COLUMNS"}:
            reason = f"{self.code.lower()}: expected {self.expected} fields, found {self.actual}"
        else:
            reason = "invalid quoting after a closing quote"
        return f"{source or self.file}: {where}: {reason}. Record: {self.text!r}"


class CsvPolicy:
    """Keep strict reads anchored and translate structural failures precisely."""

    @staticmethod
    def options(
        options: Mapping[str, Any], location: str | list[str] | None = None
    ) -> dict[str, Any]:
        """Prevent strict dialect sniffing from silently discarding early rows.

        DuckDB's default strict mode can still infer a nonzero ``skip`` while
        sniffing malformed input. An explicit zero disables that inference.
        Deliberate preambles remain supported through an explicit ``skip``.
        Standard double quotes are the default, and a supplied local location
        allows a bounded first-record probe to pin an unambiguous delimiter.
        """
        effective = dict(options)
        if location is not None and effective.get("compression", "auto") == "auto":
            names = [location] if isinstance(location, str) else location
            compression = {
                {".gz": "gzip", ".gzip": "gzip", ".zst": "zstd", ".zstd": "zstd"}.get(
                    Path(name).suffix.lower()
                )
                for name in names
            }
            if len(compression) == 1 and None not in compression:
                effective["compression"] = next(iter(compression))
        if effective.get("strict_mode", True) is not False:
            effective.setdefault("skip", 0)
            effective.setdefault("quote", '"')
            if location is not None and not {"delim", "sep", "delimiter"} & effective.keys():
                delimiter = _CsvDelimiterProbe(effective).infer(location)
                if delimiter is not None:
                    effective["delim"] = delimiter
        return effective

    @staticmethod
    def protect_columns(options: Mapping[str, Any], governed: Collection[str]) -> dict[str, Any]:
        """Keep governed source spellings intact until declared validation.

        A native integer cast can round ``2.5`` or accept ``1_000`` before the
        configured grammar sees it. Named reader hints remain effective for
        ungoverned columns. Positional nontext hints cannot be matched safely
        without another layout probe, so they receive an explicit conflict.
        """
        effective = dict(options)
        wanted = {name.casefold() for name in governed}
        if not wanted:
            return effective
        for key in ("types", "dtypes", "column_types", "columns"):
            given = effective.get(key)
            if isinstance(given, Mapping):
                effective[key] = {
                    name: "VARCHAR" if str(name).casefold() in wanted else kind
                    for name, kind in given.items()
                }
            elif given is not None:
                kinds = given if isinstance(given, list | tuple) else [given]
                if any(str(kind).upper() not in {"VARCHAR", "TEXT", "STRING"} for kind in kinds):
                    raise ValueError(
                        f"CSV positional {key} conflicts with declared column validation: "
                        "use named reader types or VARCHAR so validation sees the original "
                        "text before numeric rounding or other conversion"
                    )
        return effective

    @staticmethod
    def diagnose(location: str | list[str], options: Mapping[str, Any]) -> CsvIssue | None:
        """Locate a structural failure in local input using its reader settings.

        Remote input, unsupported dialects and unreadable encodings return None
        so the original native error remains available. No reader options are
        changed to make a diagnostic succeed.
        """
        if options.get("strict_mode", True) is False:
            return None
        for name in [location] if isinstance(location, str) else location:
            if "://" in name:
                continue
            # Native readers receive escaped literal paths in pinned manifests.
            paths = [name] if Path(name).is_file() else sorted(glob.glob(name))
            for path in paths:
                try:
                    issue = _CsvDiagnosticReader(Path(path), options).first_issue()
                except (OSError, UnicodeError, LookupError, ValueError, TypeError, ImportError):
                    continue
                if issue is not None:
                    return issue
        return None


class _CsvDelimiterProbe:
    """Anchor an unambiguous first record with at most 64 KiB of decoded text.

    This is a bounded header probe, never a structural validation pass. A strict
    reader must not change a clearly delimited header into one text field merely
    to admit a later malformed record. Ambiguous or unsupported dialects retain
    native inference.
    """

    limit = 64 * 1024
    candidates = ",;\t|"

    def __init__(self, options: Mapping[str, Any]) -> None:
        self.options = options

    def infer(self, location: str | list[str]) -> str | None:
        found: set[str] = set()
        for name in [location] if isinstance(location, str) else location:
            if "://" in name:
                return None
            paths = [name] if Path(name).is_file() else sorted(glob.glob(name))
            if not paths:
                return None
            for path in paths:
                try:
                    with _CsvDiagnosticReader(Path(path), self.options)._open() as stream:
                        delimiter = self._first_record(stream)
                except (OSError, UnicodeError, LookupError, ValueError, TypeError, ImportError):
                    return None
                if delimiter is None:
                    return None
                found.add(delimiter)
                if len(found) > 1:
                    return None
        return next(iter(found)) if found else None

    def _first_record(self, stream: TextIO) -> str | None:
        remaining = self.limit
        skip = int(self.options.get("skip", 0))
        quote = self.options.get("quote", '"')
        escape = self.options.get("escape", quote)
        comment = str(self.options.get("comment", ""))
        if len(quote) > 1 or len(escape) > 1:
            return None
        quoted = False
        field_start = True
        separators: set[str] = set()
        for _ in range(skip):
            text = stream.readline(remaining + 1)
            remaining -= len(text)
            if remaining <= 0:
                return None
        while remaining > 0:
            text = stream.readline(remaining + 1)
            remaining -= len(text)
            if remaining < 0:
                return None
            if not text:
                return next(iter(separators)) if len(separators) == 1 else None
            if not quoted and (not text.strip() or (comment and text.startswith(comment))):
                continue
            index = 0
            while index < len(text):
                char = text[index]
                if quoted:
                    if (
                        escape
                        and char == escape
                        and index + 1 < len(text)
                        and (escape != quote or text[index + 1] == quote)
                    ):
                        index += 2
                        continue
                    if quote and char == quote:
                        quoted = False
                else:
                    if comment and text.startswith(comment, index):
                        return next(iter(separators)) if len(separators) == 1 else None
                    if char in "\r\n":
                        return next(iter(separators)) if len(separators) == 1 else None
                    if quote and field_start and char == quote:
                        quoted = True
                        field_start = False
                    else:
                        field_start = char in self.candidates
                        if field_start:
                            separators.add(char)
                index += 1
        return None


class _CsvDiagnosticReader:
    """Stream a failed input once, retaining only the current CSV record."""

    def __init__(self, path: Path, options: Mapping[str, Any]) -> None:
        self.path = path
        self.options = options

    def _open(self) -> TextIO:
        encoding = str(self.options.get("encoding", "utf-8"))
        if encoding.lower().replace("-", "") == "utf8":
            encoding = "utf-8-sig"
        compression = str(self.options.get("compression", "auto")).lower()
        suffix = self.path.suffix.lower()
        if compression == "gzip" or (compression == "auto" and suffix in {".gz", ".gzip"}):
            return gzip.open(self.path, "rt", encoding=encoding, newline="")
        if compression in {"bzip2", "bz2"} or (compression == "auto" and suffix == ".bz2"):
            return bz2.open(self.path, "rt", encoding=encoding, newline="")
        if compression in {"zstd", "zstandard"} or (
            compression == "auto" and suffix in {".zst", ".zstd"}
        ):
            import pyarrow as pa

            return TextIOWrapper(
                pa.input_stream(str(self.path), compression="zstd"),
                encoding=encoding,
                newline="",
            )
        return self.path.open(encoding=encoding, newline="")

    def first_issue(self) -> CsvIssue | None:
        with self._open() as stream:
            lines = _RecordLines(stream, str(self.options.get("comment", "")))
            for _ in range(int(self.options.get("skip", 0))):
                lines.skip()
            delimiter = next(
                (self.options[key] for key in ("delim", "sep", "delimiter") if key in self.options),
                None,
            )
            if delimiter is None:
                # Native sniffing has already failed. Infer only the delimiter
                # from the first nonempty record, rather than from ragged data.
                position = stream.tell()
                first = stream.readline()
                while first and (
                    not first.strip() or (lines.comment and first.startswith(lines.comment))
                ):
                    first = stream.readline()
                stream.seek(position)
                try:
                    delimiter = csv.Sniffer().sniff(first, delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
            quote = self.options.get("quote", '"')
            escape = self.options.get("escape", quote)
            if len(delimiter) != 1 or len(quote) > 1 or len(escape) > 1:
                return None  # Python's parser cannot reproduce this native dialect.
            lines.configure(delimiter, quote, escape)
            reader = csv.reader(
                lines,
                delimiter=delimiter,
                quotechar=quote or None,
                escapechar=escape if escape and escape != quote else None,
                doublequote=escape == quote,
                quoting=csv.QUOTE_MINIMAL if quote else csv.QUOTE_NONE,
                strict=True,
            )
            columns = self.options.get("columns")
            expected = len(columns) if isinstance(columns, dict) else None
            row = 0
            while True:
                lines.begin()
                try:
                    fields = next(reader)
                except StopIteration:
                    return None
                except csv.Error as error:
                    message = str(error)
                    if "field larger than field limit" in message:
                        # Do not mistake Python's process-global parser limit
                        # for a native input error or mutate it across threads.
                        return None
                    code = (
                        "UNTERMINATED QUOTE"
                        if "unexpected end of data" in message
                        else "INVALID QUOTING"
                    )
                    return self._issue(lines, row + 1, code)
                if not fields:
                    continue
                row += 1
                if expected is None:
                    expected = len(fields)
                elif len(fields) != expected:
                    if len(fields) < expected and self.options.get("null_padding") is True:
                        continue
                    code = "TOO MANY COLUMNS" if len(fields) > expected else "MISSING COLUMNS"
                    return self._issue(lines, row, code, expected, len(fields))

    def _issue(
        self,
        lines: _RecordLines,
        row: int,
        code: str,
        expected: int | None = None,
        actual: int | None = None,
    ) -> CsvIssue:
        return CsvIssue(
            str(self.path),
            code,
            row,
            lines.start,
            lines.line,
            lines.sample.rstrip("\r\n"),
            expected,
            actual,
        )


class _RecordLines:
    """Track physical lines and a bounded preview, including multiline records."""

    def __init__(self, stream: TextIO, comment: str) -> None:
        self.stream = stream
        self.comment = comment
        self.line = 0
        self.start = 1
        self.sample = ""
        self._at_start = True
        self._quoted = False
        self._field_start = True

    def configure(self, delimiter: str, quote: str, escape: str) -> None:
        self.delimiter, self.quote, self.escape = delimiter, quote, escape

    def begin(self) -> None:
        self.start = self.line + 1
        self.sample = ""
        self._at_start = True

    def skip(self) -> None:
        if self.stream.readline():
            self.line += 1

    def __iter__(self) -> _RecordLines:
        return self

    def __next__(self) -> str:
        while True:
            text = next(self.stream)
            self.line += 1
            if self._at_start and self.comment and text.startswith(self.comment):
                self.start = self.line + 1
                continue
            self._at_start = False
            remaining = 200 - len(self.sample)
            if remaining > 0:
                self.sample += text[:remaining]
            return self._without_comment(text) if self.comment else text

    def _without_comment(self, text: str) -> str:
        """Strip native inline comments only outside quoted fields.

        Keep the native line ending so removing an inline comment still ends a
        record. A comment marker inside a multiline quoted field is ordinary data.
        """
        index = 0
        while index < len(text):
            char = text[index]
            if self._quoted:
                if (
                    self.escape
                    and char == self.escape
                    and index + 1 < len(text)
                    and (self.escape != self.quote or text[index + 1] == self.quote)
                ):
                    index += 2
                    continue
                if self.quote and char == self.quote:
                    self._quoted = False
            else:
                if text.startswith(self.comment, index):
                    self._field_start = True
                    ending = (
                        "\r\n" if text.endswith("\r\n") else ("\n" if text.endswith("\n") else "")
                    )
                    return text[:index] + ending
                if self.quote and self._field_start and char == self.quote:
                    self._quoted = True
                    self._field_start = False
                else:
                    self._field_start = char == self.delimiter or char in "\r\n"
            index += 1
        return text
