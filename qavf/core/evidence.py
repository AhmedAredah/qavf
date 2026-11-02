"""Lossless, byte-bounded admission at the DuckDB/Python evidence boundary."""

from __future__ import annotations

import bisect
import functools
import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from qavf.audit_codec import evidence_json_size
from qavf.sources.sql import quote_identifier as qi


class EvidenceRecords(list):
    """A compatible record list that remembers omitted database evidence.

    ``truncated`` is set when more records exist than were kept; ``omitted`` only
    when a record was left out because of the evidence byte budget.
    """

    def __init__(self, records=(), *, truncated=False, omitted=False):
        super().__init__(records)
        self.truncated = truncated
        self.omitted = omitted


def _identity(value):
    return value


def _decode_temporal(parser, value):
    """Parse temporal text into a Python value when it is representable."""
    if value is None or parser is None:
        return value
    try:
        return parser(value)
    except ValueError:
        # Infinity and years outside Python's range stay exact text.
        return value


def _decode_list(child, values):
    """Decode each element of a list evidence value."""
    return None if values is None else [child.decode(v) for v in values]


def _decode_struct(children, value):
    """Decode each field of a struct evidence value."""
    return (
        None
        if value is None
        else {name: child.decode(value[name]) for name, child in children.items()}
    )


def _decode_map(mapping_key, key, value, entries):
    """Decode a map evidence value as a dict, or parallel arrays for non-JSON keys."""
    if entries is None:
        return None
    # Non-JSON scalar keys need the parallel-array representation;
    # rebuilding a Python dict would reject structs or lose types.
    if not mapping_key:
        return {
            "key": [key.decode(entry["key"]) for entry in entries],
            "value": [value.decode(entry["value"]) for entry in entries],
        }
    return {key.decode(k): value.decode(v) for k, v in entries.items()}


@dataclass(frozen=True, slots=True)
class EvidenceProjection:
    """The SQL representation and decoder of one lossless evidence value.

    Python's native datetime conversion loses nanoseconds and maps infinity to
    a finite endpoint. Convert temporal leaves in SQL, including nested leaves,
    before DuckDB can perform that conversion. Ordinary representable temporal
    values retain the established Python date/datetime/time representation.
    """

    sql: str
    decode: Callable[[Any], Any] = _identity

    @classmethod
    def build(cls, expression, dtype, depth=0):
        kind = dtype.id
        if kind in {
            "date",
            "timestamp",
            "timestamp_s",
            "timestamp_ms",
            "timestamp_ns",
            "timestamp with time zone",
            "time",
            "time with time zone",
            "time_ns",
            "interval",
        }:
            temporal = f"CAST({expression} AS VARCHAR)"
            if kind.startswith("timestamp"):
                temporal = f"replace({temporal}, ' ', 'T')"
            if kind == "timestamp_ns":
                # DuckDB 1.3.1's VARCHAR conversion writes NUL bytes instead of
                # leading fractional zeros for some nanosecond timestamps.
                # Format seconds separately, then append the exact tick residue.
                ticks = f"epoch_ns({expression})"
                fraction = f"(({ticks} % 1000000000 + 1000000000) % 1000000000)"
                temporal = (
                    f"CASE WHEN isfinite({expression}) THEN "
                    f"strftime({expression}, '%Y-%m-%dT%H:%M:%S') || '.' || "
                    f"lpad(CAST({fraction} AS VARCHAR), 9, '0') ELSE {temporal} END"
                )
            parser = (
                date.fromisoformat
                if kind == "date"
                else datetime.fromisoformat
                if kind.startswith("timestamp") and kind != "timestamp_ns"
                else time.fromisoformat
                if kind in {"time", "time with time zone"}
                else None
            )
            return cls(temporal, functools.partial(_decode_temporal, parser))
        if kind in {"list", "array"}:
            item = f"__qavf_evidence_item_{depth}"
            child = cls.build(item, dict(dtype.children)["child"], depth + 1)
            return cls(
                f"list_transform({expression}, lambda {item}: {child.sql})",
                functools.partial(_decode_list, child),
            )
        if kind == "struct":
            children = {
                name: cls.build(f"({expression}).{qi(name)}", child, depth + 1)
                for name, child in dtype.children
            }
            fields = ", ".join(f"{qi(name)} := {child.sql}" for name, child in children.items())
            return cls(
                f"CASE WHEN {expression} IS NULL THEN NULL ELSE struct_pack({fields}) END",
                functools.partial(_decode_struct, children),
            )
        if kind == "map":
            item = f"__qavf_evidence_entry_{depth}"
            types = dict(dtype.children)
            key = cls.build(f"{item}.key", types["key"], depth + 1)
            value = cls.build(f"{item}.value", types["value"], depth + 1)
            mapping_key = types["key"].id in {
                "varchar",
                "enum",
                "boolean",
                "tinyint",
                "smallint",
                "integer",
                "bigint",
                "hugeint",
                "utinyint",
                "usmallint",
                "uinteger",
                "ubigint",
                "uhugeint",
                "float",
                "double",
                "bignum",
            }
            entries = (
                f"list_transform(map_entries({expression}), lambda {item}: "
                f"struct_pack(key := {key.sql}, value := {value.sql}))"
            )
            return cls(
                f"map_from_entries({entries})" if mapping_key else entries,
                functools.partial(_decode_map, mapping_key, key, value),
            )
        return cls(expression)


class SqlEvidenceReader:
    """Stream a bounded prefix without transferring oversized cells to Python.

    A native JSON-size gate rejects whole oversized records before fetching.
    The canonical audit codec then performs exact admission (its type tags can
    be larger than native JSON). Both stages stop at the first omitted record;
    counts are calculated independently by the check executor.
    """

    def __init__(self, connection, *, max_records, max_bytes):
        self.connection = connection
        self.max_records = max_records
        self.max_bytes = max_bytes

    @classmethod
    def _minimum_bytes(cls, expression, dtype, depth=0):
        """Cheap native lower bound: reject wide leaves without serializing them."""
        kind = dtype.id
        if kind == "varchar":
            return f"coalesce(strlen({expression}), 0)"
        if kind == "blob":
            return f"coalesce(octet_length({expression}), 0)"
        if kind in {"list", "array"}:
            item = f"__qavf_size_item_{depth}"
            child = cls._minimum_bytes(item, dict(dtype.children)["child"], depth + 1)
            return (
                f"coalesce(list_sum(list_transform({expression}, lambda {item}: {child} + 1)), 0)"
            )
        if kind == "struct":
            return (
                "("
                + " + ".join(
                    cls._minimum_bytes(f"({expression}).{qi(name)}", child, depth + 1)
                    for name, child in dtype.children
                )
                + ")"
            )
        if kind == "map":
            item = f"__qavf_size_entry_{depth}"
            children = dict(dtype.children)
            key = cls._minimum_bytes(f"{item}.key", children["key"], depth + 1)
            value = cls._minimum_bytes(f"{item}.value", children["value"], depth + 1)
            return (
                f"coalesce(list_sum(list_transform(map_entries({expression}), lambda {item}: "
                f"{key} + {value} + 1)), 0)"
            )
        return "1"

    def read(self, query):
        records = EvidenceRecords()
        if not self.max_records or self.max_bytes < 2:
            records.truncated = True
            records.omitted = self.max_bytes < 2
            return records
        relation = self.connection.sql(query)
        names, types = relation.columns, relation.types
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Evidence column names must be unique ignoring case")
        projected = {
            name: EvidenceProjection.build(f"__qavf_sample.{qi(name)}", dtype)
            for name, dtype in zip(names, types, strict=True)
        }
        fields = ", ".join(f"{qi(name)} := {item.sql}" for name, item in projected.items())
        minimum = " + ".join(
            self._minimum_bytes(f"__qavf_sample.{qi(name)}", dtype)
            for name, dtype in zip(names, types, strict=True)
        )
        # CASE is evaluated inside DuckDB: a rejected string/blob never crosses
        # the Python boundary. The original query's deterministic order remains
        # the evidence prefix; do not sort or materialize its entire result.
        guarded = (
            f"SELECT CASE WHEN __qavf_bytes <= {self.max_bytes} THEN __qavf_payload END, "
            f"__qavf_bytes > {self.max_bytes} FROM ("
            f"SELECT __qavf_payload, CASE WHEN __qavf_minimum > {self.max_bytes} "
            f"THEN {self.max_bytes + 1} ELSE strlen(to_json(__qavf_payload)) END AS __qavf_bytes "
            f"FROM (SELECT struct_pack({fields}) AS __qavf_payload, {minimum} AS __qavf_minimum "
            f"FROM ({query}) __qavf_sample LIMIT {self.max_records + 1}) __qavf_values"
            f") __qavf_sized"
        )
        cursor = self.connection.execute(guarded)
        used = 2
        while (row := cursor.fetchone()) is not None:
            payload, oversized = row
            if oversized or len(records) >= self.max_records:
                records.truncated = True
                records.omitted = bool(oversized)
                break
            decoded = {name: item.decode(payload[name]) for name, item in projected.items()}
            size = evidence_json_size(decoded) + bool(records)
            if used + size > self.max_bytes:
                records.truncated = records.omitted = True
                break
            records.append(decoded)
            used += size
        return records


class FileRows:
    """The file, and the row within it, of evidence rows from a multi-file dataset.

    Evidence names a row by its 0-based position in the dataset as read. Rows per
    file are counted once, only when such evidence exists; a count that does not
    add up leaves evidence unlocated, never misplaced.
    """

    def __init__(self, connection, files, counting_sql, row_column):
        self.connection = connection
        self.files = list(files)
        self.counting_sql = counting_sql
        self.row_column = row_column
        self._starts = None

    def starts(self):
        """Each file's first row position, plus the total; ``()`` when unknown."""
        if self._starts is None:
            self._starts = self._count()
        return self._starts

    def _count(self):
        try:
            counted = self.connection.execute(self.counting_sql).fetchall()
        except Exception:
            return ()
        by_path = {os.path.normcase(os.path.abspath(str(name))): int(n) for name, n in counted}
        rows = [by_path.get(os.path.normcase(os.path.abspath(path)), 0) for path in self.files]
        if sum(rows) != sum(by_path.values()):
            return ()
        return tuple(itertools.accumulate(rows, initial=0))

    def locate(self, position):
        starts = self.starts()
        index = bisect.bisect_right(starts, position) - 1
        if not starts or not 0 <= index < len(self.files):
            return None, None
        return self.files[index], position - starts[index]

    def annotate(self, records, row_column):
        """``records`` with ``_file`` and ``_file_row``; unchanged when unlocatable."""
        if not records or not all(row_column in record for record in records):
            return records
        if not self.starts():
            return records
        located = []
        for record in records:
            file, row = self.locate(int(record[row_column]))
            located.append(annotated(record, _file=file, _file_row=row))
        return located


def annotated(record, **fields):
    """An evidence record: ``fields`` first, then the data, never overwriting it.

    An annotation whose name the record already uses is prefixed with ``qavf_``.
    """
    taken = set(record)
    result = {}
    for name, value in fields.items():
        while name in taken:
            name = f"qavf_{name}"
        taken.add(name)
        result[name] = value
    result.update(record)
    return result
