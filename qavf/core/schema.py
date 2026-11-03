"""Schema conformance (INP-002) as SQL, reported as the full backend reports it.

Each column read as text but declared another type reports the values that do
not convert (they are read as missing). Missing and undeclared columns and the
row-count limits are one structural result per dataset.
"""

from __future__ import annotations

import re

from qavf import input_rules as wording
from qavf.core.key_policy import FLOATS
from qavf.engine.severity import Status
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit

_POLARS_NAMES = {
    "BIGINT": "Int64",
    "INTEGER": "Int32",
    "SMALLINT": "Int16",
    "TINYINT": "Int8",
    "HUGEINT": "Int128",
    "UBIGINT": "UInt64",
    "UINTEGER": "UInt32",
    "USMALLINT": "UInt16",
    "UTINYINT": "UInt8",
    "DOUBLE": "Float64",
    "FLOAT": "Float32",
    "REAL": "Float32",
    "VARCHAR": "String",
    "BOOLEAN": "Boolean",
    "DATE": "Date",
    "TIME": "Time",
    "BLOB": "Binary",
    "TIMESTAMP": "Datetime(time_unit='us', time_zone=None)",
    "TIMESTAMP_MS": "Datetime(time_unit='ms', time_zone=None)",
    "TIMESTAMP_NS": "Datetime(time_unit='ns', time_zone=None)",
}
_DECIMAL = re.compile(r"DECIMAL\((\d+),\s*(\d+)\)")


def dtype_name(dtype: str) -> str:
    """A storage type as reports name it, in the full backend's vocabulary."""
    if dtype in _POLARS_NAMES:
        return _POLARS_NAMES[dtype]
    decimal = _DECIMAL.fullmatch(dtype)
    if decimal:
        return f"Decimal(precision={decimal[1]}, scale={decimal[2]})"
    return dtype


def evidence_name(name: str, taken) -> str:
    """A readable evidence field name that never replaces a data column."""
    taken = set(taken)
    while name in taken:
        name = f"qavf_{name}"
    return name


class SchemaConformance:
    """INP-002 for one loaded dataset: value conformance, then its structure."""

    RULE = "INP-002"

    def __init__(self, checks):
        self.checks = checks
        self.out = checks.out
        self.data = checks.data
        self.schema = checks.schema
        self.parameters = self.schema.model_dump(mode="python", exclude_defaults=True)

    @property
    def available(self):
        return list(getattr(self.data, "original_columns", ()) or self.data.columns)

    def run(self):
        name = self.data.name
        try:
            failed = self._values() | self._structure()
        except Exception as exc:
            self.out.report(
                self.RULE,
                name,
                Status.ERROR,
                f"Not assessed: {name}: {type(exc).__name__}: {exc}.",
                check=name,
                datasets=(name,),
            )
            return
        if not failed:
            identity = {"check_id": "schema.contract", "datasets": (name,)}
            self.out.passed(self.RULE, name, name, {**identity, "parameters": self.parameters})

    def _kind(self, column):
        """How the full backend assesses a column's values: ``read``, ``text``, ``whole``."""
        spec, raw = self.schema.columns[column], self.data.types.get(column, "")
        value = self.checks.values.get(column)
        if value is None or spec.dtype is None or spec.dtype == "str":
            return None
        if raw.startswith(("VARCHAR", "CHAR")):
            return "read" if self.data.reader == "csv" else "text"
        if spec.dtype == "int" and raw in FLOATS:
            return "whole"
        if value.converted != "NULL" and not value.invalid.startswith("(FALSE)"):
            # Stored values the declared type cannot hold exactly (scale, precision).
            return "text"
        return None

    def _values(self):
        kinds = {c: self._kind(c) for c in self.schema.columns if c in self.data.columns}
        kinds = {column: kind for column, kind in kinds.items() if kind is not None}
        if not kinds:
            return False
        values = self.checks.values
        counts = self.out.connection.execute(
            "SELECT "
            + ", ".join(f"count(*) FILTER (WHERE {values[c].invalid})" for c in kinds)
            + f" FROM {qi(self.data.table)}"
        ).fetchone()
        failed = False
        for (column, kind), count in zip(kinds.items(), counts, strict=True):
            if count:
                failed = True
                self._report_values(column, kind, count)
        return failed

    def _report_values(self, column, kind, count):
        data, spec = self.data, self.schema.columns[column]
        rows = data.rows
        noun = wording.described(str(spec.dtype))
        if kind == "read" and spec.format:
            noun += f" written as {spec.format!r}"
        if kind == "read":
            message = wording.typed_read_values(count, rows, data.name, column, noun)
            identity = {"check_id": "schema.contract", "parameters": self.parameters}
            context = [c for c in self.schema.primary_key if c in data.columns]
        else:
            message = (
                wording.text_values(
                    count, rows, data.name, column, noun, dtype_name(data.types[column])
                )
                if kind == "text"
                else wording.non_integral_values(count, rows, data.name, column)
            )
            identity = {
                "check_id": "column.dtype",
                "columns": (column,),
                "parameters": {"dtype": spec.dtype},
            }
            context = [c for c in self.schema.primary_key if c in data.columns and c != column]
        label = "not an int" if kind == "whole" else f"not {noun}"
        self.out.report(
            self.RULE,
            data.name,
            Status.FAIL,
            message,
            check=f"{data.name}.{column}: dtype {spec.dtype}",
            datasets=(data.name,),
            records=self._evidence(column, context, label),
            count=count,
            total=rows,
            unit="rows",
            **identity,
        )

    def _evidence(self, column, context, label):
        data = self.data
        limit = self.out.evidence_limit
        if not limit:
            return None
        shown = list(dict.fromkeys([*context, column]))
        reason = evidence_name("violates", shown)
        return self.out.records(
            f"SELECT {qi(data.row_column)}, "
            f"{', '.join(qi(c) for c in shown)}, {lit(label)} AS {qi(reason)} "
            f"FROM {qi(data.table)} WHERE {self.checks.values[column].invalid} "
            f"ORDER BY {qi(data.row_column)} LIMIT {limit}"
        )

    def _structure(self):
        data, schema = self.data, self.schema
        available = self.available
        problems = []
        for column, spec in schema.columns.items():
            if column not in available:
                if spec.required:
                    problem = {"column": column, "problem": "missing required column"}
                    hint = wording.absent_column_hint(column, available)
                    if hint:
                        problem["hint"] = hint.removeprefix("; ")
                    problems.append(problem)
                continue
            mismatch = self._mismatch(column)
            if mismatch:
                problems.append({"column": column, "problem": mismatch})
        if schema.strict:
            problems.extend(
                {"column": column, "problem": "undeclared column"}
                for column in available
                if column not in schema.columns
            )
        if data.rows < schema.min_rows:
            shortfall = wording.row_count_problem(data.rows, schema.min_rows, at_least=True)
            problems.append({"column": None, "problem": shortfall})
        if schema.max_rows is not None and data.rows > schema.max_rows:
            excess = wording.row_count_problem(data.rows, schema.max_rows, at_least=False)
            problems.append({"column": None, "problem": excess})
        if not problems:
            return False
        self.out.report(
            self.RULE,
            data.name,
            Status.FAIL,
            wording.schema_violations(len(problems), data.name),
            check=data.name,
            check_id="schema.contract",
            datasets=(data.name,),
            parameters=self.parameters,
            records=problems,
            unit="problems",
        )
        return True

    def _mismatch(self, column):
        """A declared type that the column's storage cannot hold, as a problem."""
        spec = self.schema.columns[column]
        value = self.checks.values.get(column)
        if value is None or spec.dtype is None or self._kind(column) is not None:
            return None
        if value.converted != "NULL":
            return None
        return f"expected {spec.dtype}, found {dtype_name(self.data.types[column])}"
