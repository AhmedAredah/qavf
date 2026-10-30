"""Bounded probes of a CSV read, reusing the source's exact reader settings.

A :class:`CsvRead` describes how a source reads its files. A :class:`CsvProbe`
runs short queries against that read on its own connection: the header, the
first line as written, and the lines a failed read could not parse.
:class:`TypedColumns` converts declared columns inside the scan and locates the
values it could not convert; :class:`HeaderReview` explains names the reader had
to invent; :class:`ReadFailure` turns a reader's error into a located message.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import duckdb

from qavf.redaction import redact_text
from qavf.sources.base import RejectedRows, SourceEnvironment, SourceError, TypeViolations
from qavf.sources.errors import redacted_error
from qavf.sources.native import connect
from qavf.sources.sql import quote_identifier, sql_literal
from qavf.text_types import TextType, sql_stripped, sql_value

REJECTED_SAMPLE = 100
"""Failing lines or values located per file or column, for messages and evidence."""
REJECT_OPTIONS = frozenset(
    {"ignore_errors", "store_rejects", "rejects_table", "rejects_scan", "rejects_limit"}
)
"""Reader options that record or skip failing lines."""
_TYPE_OPTIONS = frozenset({"header", "types", "dtypes", "column_types"})


@dataclass(frozen=True)
class CsvRead:
    """Where and how a CSV source reads: the location and effective reader options."""

    location: str | list[str]
    options: Mapping[str, Any] = field(default_factory=dict)

    def reader(self, drop: frozenset[str] = frozenset(), **overrides: Any) -> str:
        """``read_csv($path, ...)`` with these options, less ``drop``, plus ``overrides``."""
        options = {k: v for k, v in self.options.items() if k not in drop} | overrides
        args = "".join(
            f", {quote_identifier(key)} = {sql_literal(value)}" for key, value in options.items()
        )
        return f"read_csv($path{args})"


class CsvProbe:
    """Short-lived queries against one CSV read, each on a connection of its own."""

    def __init__(self, env: SourceEnvironment, read: CsvRead) -> None:
        self.env = env
        self.read = read
        self._header: list[str] | None = None

    def header(self) -> list[str]:
        """Column names as the reader names them, read once without any rows.

        A header that cannot be read is empty: the read itself then reports why.
        """
        if self._header is None:
            query = f"SELECT * FROM {self.read.reader(REJECT_OPTIONS)} LIMIT 0"
            try:
                with self._connection() as con:
                    con.execute(query)
                    self._header = [str(column[0]) for column in con.description or ()]
            except duckdb.Error:
                self._header = []
        return list(self._header)

    def raw_header(self) -> list[str | None]:
        """The first line's fields exactly as written, before the reader names them."""
        reader = self.read.reader(REJECT_OPTIONS | _TYPE_OPTIONS, header=False, all_varchar=True)
        try:
            row = self.fetch_one(f"SELECT * FROM {reader} LIMIT 1")
        except SourceError:
            return []
        return [None if value is None else str(value) for value in row]

    def unreadable(self) -> RejectedRows | None:
        """The lines a strict read fails on, from one pass recording a bounded sample.

        ``count(COLUMNS(*))`` converts every column; ``count(*)`` would skip them.
        """
        reader = self.read.reader(
            REJECT_OPTIONS, store_rejects=True, rejects_limit=REJECTED_SAMPLE
        )
        try:
            with self._connection() as con:
                con.execute(f"SELECT count(COLUMNS(*)) FROM {reader}").fetchall()
                # The reject table is complete once the scan has been fetched.
                return read_rejects(
                    con, "reject_errors", "reject_scans", REJECTED_SAMPLE, REJECTED_SAMPLE
                )
        except duckdb.Error:
            return None

    def fetch_one(self, query: str) -> tuple[Any, ...]:
        """The single row of ``query``, which reads this probe's files through ``$path``."""
        try:
            with self._connection() as con:
                row = con.execute(query).fetchone()
        except duckdb.Error as exc:
            raise redacted_error(exc) from None
        return tuple(row or ())

    def reading(self, options: Mapping[str, Any]) -> CsvProbe:
        """A probe of the same files under other options, keeping the read header.

        Typing columns as text changes no column name, so the header carries over.
        """
        probe = CsvProbe(self.env, CsvRead(self.read.location, dict(options)))
        probe._header = self._header
        return probe

    @contextmanager
    def _connection(self) -> Iterator[_PathConnection]:
        con = connect(self.env)
        try:
            yield _PathConnection(con, self.read.location)
        finally:
            con.close()


class _PathConnection:
    """A probe's connection, binding ``$path`` to the probed files where a query reads
    them (DuckDB rejects a named parameter that a query does not use)."""

    def __init__(self, con: Any, location: str | list[str]) -> None:
        self._con = con
        self._location = location

    def execute(self, query: str) -> Any:
        if "$path" in query:
            return self._con.execute(query, {"path": self._location})
        return self._con.execute(query)

    @property
    def description(self) -> Any:
        return self._con.description


class TypedColumns:
    """Declared columns read as text and converted in the scan that reads them."""

    def __init__(self, columns: Mapping[str, TextType]) -> None:
        self.columns = dict(columns)

    def __bool__(self) -> bool:
        return bool(self.columns)

    def wrap(self, sql: str) -> str:
        """``sql`` with each declared column replaced by its converted value.

        Each column is stripped once, in an inner projection: DuckDB evaluates a
        repeated expression again at every reference.
        """
        q = quote_identifier
        stripped = self._stripped_names()
        strips = ", ".join(f"{sql_stripped(q(c))} AS {q(stripped[c])}" for c in self.columns)
        replaced = ", ".join(
            f"{sql_value(q(c), text, q(stripped[c]))[0]} AS {q(c)}"
            for c, text in self.columns.items()
        )
        hidden = ", ".join(q(name) for name in stripped.values())
        return (
            f"SELECT * EXCLUDE ({hidden}) REPLACE ({replaced}) "
            f"FROM (SELECT *, {strips} FROM ({sql}) AS raw_text) AS stripped_text"
        )

    def violations(
        self, probe: CsvProbe, context: Sequence[str] = ()
    ) -> dict[str, TypeViolations]:
        """Count values that do not convert; locate them only when there are any.

        Counting streams. Positions need every row numbered, which holds the scan,
        so that pass runs only on failure and keeps a bounded sample per column.
        """
        shown = [c for c in dict.fromkeys(context) if c not in self.columns]
        stripped = self._stripped_names(shown)
        reader = probe.read.reader()
        counts = probe.fetch_one(self._counting(reader, stripped, shown))
        samples: tuple[Any, ...] = (None,) * len(self.columns)
        if any(counts):
            samples = probe.fetch_one(self._locating(reader, stripped, shown))
        return {
            column: TypeViolations(int(count or 0), tuple(dict(e) for e in sample or ()))
            for column, count, sample in zip(self.columns, counts, samples, strict=True)
        }

    def _counting(self, reader: str, stripped: Mapping[str, str], shown: list[str]) -> str:
        counts = ", ".join(
            f"count(*) FILTER (WHERE {self._invalid(column, stripped)})" for column in self.columns
        )
        return f"SELECT {counts} FROM ({self._text(stripped, shown)} FROM {reader}) AS raw_text"

    def _locating(self, reader: str, stripped: Mapping[str, str], shown: list[str]) -> str:
        row = quote_identifier(_fresh("__qavf_row", [*self.columns, *shown, *stripped.values()]))
        samples = ", ".join(
            self._sample(column, row, stripped, shown) for column in self.columns
        )
        return (
            f"SELECT {samples} FROM ({self._text(stripped, shown)}, "
            f"row_number() OVER () - 1 AS {row} FROM {reader}) AS raw_text"
        )

    def _sample(self, column: str, row: str, stripped: Mapping[str, str], shown: list[str]) -> str:
        q = quote_identifier
        context = [f"{sql_literal(c)}: {q(c)}" for c in shown]
        fields = ", ".join([f"'_row': {row}", *context, f"'value': {q(column)}"])
        return (
            f"arg_min({{{fields}}}, {row}, {REJECTED_SAMPLE}) "
            f"FILTER (WHERE {self._invalid(column, stripped)})"
        )

    def _text(self, stripped: Mapping[str, str], shown: list[str]) -> str:
        q = quote_identifier
        columns = [q(c) for c in [*self.columns, *shown]]
        columns += [f"{sql_stripped(q(c))} AS {q(stripped[c])}" for c in self.columns]
        return f"SELECT {', '.join(columns)}"

    def _invalid(self, column: str, stripped: Mapping[str, str]) -> str:
        q = quote_identifier
        return sql_value(q(column), self.columns[column], q(stripped[column]))[1]

    def _stripped_names(self, others: Sequence[str] = ()) -> dict[str, str]:
        taken = [*self.columns, *others]
        names: dict[str, str] = {}
        for index, column in enumerate(self.columns):
            names[column] = _fresh(f"__qavf_text_{index}", taken)
            taken.append(names[column])
        return names


class HeaderReview:
    """Header names the reader had to invent, explained as INP-002 problems.

    Only synthesized names (``v_1``, ``column3``) prompt a look at the raw first
    line, so a clean header costs nothing beyond the cached header read.
    """

    _GENERIC = re.compile(r"column\d+")
    _DEDUPED = re.compile(r"(.+)_\d+")
    _DELIMITERS = ",;|\t"

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)

    def problems(self, probe: CsvProbe) -> list[dict[str, Any]]:
        """Problems the raw first line confirms. A name pattern alone never decides:
        ``column0`` and a quoted ``"rate,count"`` are legitimate names."""
        if not self.names or not self._suspicious():
            return []
        raw = probe.raw_header()
        if raw == self.names:
            return []  # the reader kept the names exactly as written
        if len(self.names) == 1 and any(mark in self.names[0] for mark in self._DELIMITERS):
            return [self._unsplit()]
        if all(self._GENERIC.fullmatch(name) for name in self.names):
            return [self._unrecognized()]
        return self._renamed(raw)

    def _suspicious(self) -> bool:
        """Whether a name looks invented by the reader, so the raw line is worth reading."""
        unsplit = len(self.names) == 1 and any(m in self.names[0] for m in self._DELIMITERS)
        return unsplit or self._synthesized()

    def _synthesized(self) -> bool:
        folded = {name.casefold() for name in self.names}
        return any(
            self._GENERIC.fullmatch(name)
            or ((match := self._DEDUPED.fullmatch(name)) and match[1].casefold() in folded)
            for name in self.names
        )

    def _renamed(self, raw: list[str | None]) -> list[dict[str, Any]]:
        if len(raw) != len(self.names):
            return []
        problems: list[dict[str, Any]] = []
        seen: set[str] = set()
        for original, name in zip(raw, self.names, strict=True):
            if original is None or not original.strip():
                problems.append({"column": name, "problem": f"blank header, read as {name!r}"})
            elif original.casefold() in seen:
                problems.append(
                    {"column": name, "problem": f"duplicate header {original!r}, read as {name!r}"}
                )
            if original is not None:
                seen.add(original.casefold())
        return problems

    def _unsplit(self) -> dict[str, Any]:
        return {
            "column": self.names[0],
            "problem": "the header was read as one column: rows could not be split into "
            "fields, usually because a line has extra or missing fields; set options "
            "{delim: ...} to locate it",
        }

    def _unrecognized(self) -> dict[str, Any]:
        shown = ", ".join(self.names[:3]) + (", …" if len(self.names) > 3 else "")
        return {
            "column": None,
            "problem": f"the header row was not recognized (columns read as {shown}): an "
            "early line may have extra or missing fields; set options {header: true} to "
            "locate it",
        }


class ReadFailure:
    """A reader's error, explained in terms of the input and the setting to change."""

    _CSV = re.compile(r"CSV Error|sniffing|CSV parsing dialect", re.IGNORECASE)
    _SNIFF = re.compile(r"sniffing|CSV parsing dialect", re.IGNORECASE)
    _EXCEL_CELL = re.compile(
        r"Failed to parse cell '(?P<cell>[A-Z]+[0-9]+)': Could not convert string "
        r"'(?P<text>.*?)' to (?P<type>[A-Z]+)"
    )
    _EXCEL_TYPES: ClassVar[dict[str, str]] = {
        "DOUBLE": "number",
        "DATE": "date",
        "TIMESTAMP": "date and time",
        "BOOLEAN": "boolean",
    }
    _ADVICE: ClassVar[dict[str, str]] = {
        "CAST": (
            "The reader chose the column's type from the first rows: declare the column's "
            "dtype in the schema (placeholders such as 'n/a' in missing_values), read it as "
            "text with text_columns, or let the reader see every row with options "
            "{sample_size: -1}."
        ),
        "TOO MANY COLUMNS": (
            "The line has more fields than the header: quote values that contain the "
            "delimiter, or leave such lines out and have them reported with options "
            "{store_rejects: true} and acknowledge_lossy: true."
        ),
        "MISSING COLUMNS": (
            "The line has fewer fields than the header: fix the line, or leave such lines "
            "out and have them reported with options {store_rejects: true} and "
            "acknowledge_lossy: true."
        ),
        "UNQUOTED VALUE": "A value has a quote character inside an unquoted field: quote it.",
        "INVALID ENCODING": (
            "The file is not UTF-8: set options {encoding: latin-1}, or the file's encoding."
        ),
        "LINE SIZE OVER MAXIMUM": "Set options {max_line_size: ...} above the longest line.",
    }

    def __init__(self, source: str, error: SourceError) -> None:
        self.source = redact_text(source)
        self.message = str(error)

    def excel_cell(self) -> SourceError | None:
        cell = self._EXCEL_CELL.search(self.message)
        if cell is None:
            return None
        kind = self._EXCEL_TYPES.get(cell["type"], "value")
        return SourceError(
            f"{self.source}: cell {cell['cell']} holds {cell['text']!r}, which is not a {kind} "
            "like the column's first rows. Correct the cell, or read every column as text "
            "with options {all_varchar: true} and declare each column's dtype in the schema "
            "(dates then read as day numbers)"
        )

    def is_csv(self) -> bool:
        return bool(self._CSV.search(self.message))

    def undetected_layout(self) -> SourceError | None:
        if not self._SNIFF.search(self.message):
            return None
        return SourceError(
            f"{self.source}: the reader could not detect the CSV layout (delimiter, quoting, "
            "header) from its first rows, often because an early line has extra or missing "
            "fields. Set options {delim: ','} (and quote or header) so the line can be located"
        )

    def located(self, rejected: RejectedRows) -> SourceError:
        first = rejected.sample[0]
        files = {row["file"] for row in rejected.sample}
        count = f"{rejected.count:,}" if rejected.exact else f"at least {rejected.count:,}"
        rows = "row" if rejected.exact and rejected.count == 1 else "rows"
        where = f"line {first['line']:,}" + (f" of {first['file']}" if len(files) > 1 else "")
        column = f", column {first['column']!r}" if first["column"] else ""
        advice = self._ADVICE.get(first["error"], "")
        return SourceError(
            (
                f"{count} {rows} of {self.source} cannot be read. The first is {where}{column} "
                f"({first['error'].lower()}): {first['text']!r}. {advice}"
            ).rstrip(" .")  # callers end the sentence
        )


def read_rejects(con: Any, errors: str, scans: str, limit: int, sample: int) -> RejectedRows:
    """Count rejected lines and locate the first ``sample`` of them.

    A line with several errors counts once and shows its root error: a structural
    error (wrong field count, encoding) causes any type error on the same line.
    With a per-file ``limit`` (0 for none), a file that reached it makes the count
    a lower bound.
    """
    e, s = quote_identifier(errors), quote_identifier(scans)
    per_file = con.execute(f"SELECT count(DISTINCT line) FROM {e} GROUP BY scan_id, file_id")
    counts = [int(row[0]) for row in per_file.fetchall()]
    exact = not limit or all(count < limit for count in counts)
    rows = con.execute(
        f"SELECT s.file_path, e.line, e.column_name, e.error_type, e.csv_line "
        f"FROM {e} AS e JOIN {s} AS s USING (scan_id, file_id) "
        f"QUALIFY row_number() OVER (PARTITION BY e.scan_id, e.file_id, e.line "
        f"ORDER BY e.error_type = 'CAST', e.column_idx NULLS LAST) = 1 "
        f"ORDER BY s.file_path, e.line LIMIT {int(sample)}"
    ).fetchall()
    located = tuple(
        {
            "file": redact_text(str(file)),
            "line": int(line),
            "column": column,
            "error": str(kind),
            "text": _clipped(text),
        }
        for file, line, column, kind, text in rows
    )
    return RejectedRows(sum(counts), exact, located)


def _clipped(text: object, width: int = 200) -> str:
    value = str(text or "").strip("\r\n")
    return value if len(value) <= width else value[: width - 1] + "…"


def _fresh(name: str, taken: Sequence[str]) -> str:
    """``name``, lengthened until it differs from every ``taken`` name in any case."""
    folded = {item.casefold() for item in taken}
    while name.casefold() in folded:
        name += "_"
    return name
