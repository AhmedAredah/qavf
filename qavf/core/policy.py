"""Compile canonical values with independent missing/applicability states.

Lexical admission, physical literal admission and decimal scale resolution are
separate policies. They share no dependency on an optional dataframe engine.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from fractions import Fraction
from typing import Any

from qavf.applicability import NotApplicablePolicy
from qavf.number_grammar import COMPARISON_WHITESPACE, exact_decimal
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit
from qavf.text_types import sql_stripped, sql_value

_INTEGER_BITS = {
    "TINYINT": (8, True),
    "SMALLINT": (16, True),
    "INTEGER": (32, True),
    "BIGINT": (64, True),
    "HUGEINT": (128, True),
    "UTINYINT": (8, False),
    "USMALLINT": (16, False),
    "UINTEGER": (32, False),
    "UBIGINT": (64, False),
    "UHUGEINT": (128, False),
}


def native_literal(value: Any) -> str:
    """A typed scalar literal, including SQL's non-finite float syntax."""
    # A decimal SQL token such as 0.09999999999999999 can round differently on
    # DECIMAL->DOUBLE conversion. The direct text->DOUBLE parser preserves the
    # binary float's shortest round-trip representation.
    if isinstance(value, float):
        return f"CAST({lit(str(value))} AS DOUBLE)"
    return lit(value)


def decimal_places_condition(expression: str, dtype: str, places: int) -> str:
    """SQL form of the shared exact/fractional-quantum decimal-place contract."""
    if dtype in _INTEGER_BITS:
        return "TRUE"
    if dtype.startswith("DECIMAL"):
        return (
            f"length(regexp_extract(CAST({expression} AS VARCHAR), "
            f"'\\.([0-9]*?)0*$', 1)) <= {places}"
        )
    if places >= 324:
        return "TRUE"
    scale = 10.0 ** min(places, 308)
    extra = 10.0 ** max(0, places - 308)
    value = f"CAST(({expression}) AS DOUBLE)"
    fraction = f"abs({value} - floor({value}))"
    scaled = f"({fraction} * {scale!r} * {extra!r})"
    gap = f"abs({scaled} - round({scaled}, 0))"
    tolerance = f"least(1e-7, {4 * 2.220446049250313e-16!r} * greatest(abs({scaled}), 1.0))"
    threshold = 2.0**52 / scale / extra
    return f"({fraction} = 0 OR {fraction} >= {threshold!r} OR {gap} <= {tolerance})"


def native_bound_condition(expression: str, dtype: str, operator: str, bound: Any) -> str:
    """An exact inequality on the storage grid, with no implicit numeric rounding."""
    symbols = {"ge": ">=", "le": "<=", "gt": ">", "lt": "<"}
    if dtype in _INTEGER_BITS:
        threshold = math.ceil(bound) if operator in ("ge", "lt") else math.floor(bound)
        bits, signed = _INTEGER_BITS[dtype]
        minimum, maximum = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
        if threshold < minimum:
            return "TRUE" if operator in ("ge", "gt") else "FALSE"
        if threshold > maximum:
            return "FALSE" if operator in ("ge", "gt") else "TRUE"
        return f"({expression}) {symbols[operator]} {threshold}"
    if dtype in ("FLOAT", "REAL", "DOUBLE"):
        requested = Fraction(bound)
        try:
            threshold = float(bound)
        except OverflowError:
            threshold = math.inf if requested > 0 else -math.inf
        if math.isinf(threshold):
            rounded = 1 if threshold > 0 else -1
        else:
            actual = Fraction(threshold)
            rounded = (actual > requested) - (actual < requested)
        lower = operator in ("ge", "gt")
        if lower and (rounded < 0 or (operator == "gt" and rounded == 0)):
            threshold = math.nextafter(threshold, math.inf)
        elif not lower and (rounded > 0 or (operator == "lt" and rounded == 0)):
            threshold = math.nextafter(threshold, -math.inf)
        symbol = ">=" if lower else "<="
        return f"CAST(({expression}) AS DOUBLE) {symbol} {native_literal(threshold)}"
    decimal = re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", dtype)
    if decimal:
        precision, scale = map(int, decimal.groups())
        requested = Fraction(bound) * 10**scale
        lower = operator in ("ge", "gt")
        if lower:
            coefficient = math.ceil(requested) if operator == "ge" else math.floor(requested) + 1
        else:
            coefficient = math.floor(requested) if operator == "le" else math.ceil(requested) - 1
        maximum = 10**precision - 1
        if coefficient > maximum:
            return "FALSE" if lower else "TRUE"
        if coefficient < -maximum:
            return "TRUE" if lower else "FALSE"
        threshold = Decimal((int(coefficient < 0), tuple(map(int, str(abs(coefficient)))), -scale))
        return f"({expression}) {'>=' if lower else '<='} {native_literal(threshold)}"
    return f"({expression}) {symbols[operator]} {native_literal(bound)}"


@dataclass(frozen=True, slots=True)
class NativeValueDomain:
    """Exact configured-literal admission for a DuckDB physical type.

    Wrong-kind literals are foreign, while right-kind but unrepresentable
    values simply cannot match. No implicit SQL coercion may change equality.
    """

    dtype: str

    def members(self, values) -> tuple[list[Any], list[Any]]:
        usable: list[Any] = []
        foreign: list[Any] = []
        decimal = re.fullmatch(r"DECIMAL\((\d+),\s*(\d+)\)", self.dtype)
        for value in values:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            numeric = isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
            if self.dtype in _INTEGER_BITS:
                if not numeric:
                    foreign.append(value)
                    continue
                if not isinstance(value, int) and (not math.isfinite(value) or value != int(value)):
                    continue
                bits, signed = _INTEGER_BITS[self.dtype]
                integer = int(value)
                lower, upper = (-(2 ** (bits - 1)), 2 ** (bits - 1)) if signed else (0, 2**bits)
                if lower <= integer < upper:
                    usable.append(integer)
            elif self.dtype in ("FLOAT", "REAL", "DOUBLE"):
                if not numeric:
                    foreign.append(value)
                    continue
                try:
                    converted_float = float(value)
                    if self.dtype in ("FLOAT", "REAL"):
                        converted_float = struct.unpack("f", struct.pack("f", converted_float))[0]
                    if converted_float == value:
                        usable.append(converted_float)
                except (ValueError, OverflowError):
                    pass
            elif decimal:
                if not numeric:
                    foreign.append(value)
                    continue
                precision, scale = map(int, decimal.groups())
                converted_decimal = exact_decimal(value, precision, scale)
                if converted_decimal is not None:
                    usable.append(converted_decimal)
            elif self.dtype.startswith(("VARCHAR", "CHAR", "ENUM")):
                (usable if isinstance(value, str) else foreign).append(value)
            elif self.dtype == "BOOLEAN":
                (usable if isinstance(value, bool) else foreign).append(value)
            elif self.dtype == "DATE":
                if isinstance(value, datetime):
                    if value.tzinfo is not None:
                        foreign.append(value)
                    elif value.time() == time():
                        usable.append(value.date())
                else:
                    (usable if isinstance(value, date) else foreign).append(value)
            elif self.dtype.startswith("TIMESTAMP"):
                if not isinstance(value, date) or (
                    isinstance(value, datetime)
                    and value.tzinfo
                    and "WITH TIME ZONE" not in self.dtype
                ):
                    foreign.append(value)
                else:
                    usable.append(value)
            else:
                foreign.append(value)
        return usable, foreign


@dataclass(frozen=True, slots=True)
class DecimalTextSQL:
    """Exact decimal text conversion after data-dependent scale selection."""

    raw: str
    spec: Any

    @property
    def text(self) -> str:
        value = sql_stripped(f"CAST({self.raw} AS VARCHAR)", self.spec.trim_whitespace)
        pattern = lit(self.spec.number_grammar.pattern("decimal"))
        admitted = f"CASE WHEN regexp_full_match({value}, {pattern}) THEN {value} END"
        return self.spec.number_grammar.sql_canonical(admitted)

    @property
    def fraction(self) -> str:
        return f"length(regexp_extract({self.text}, '\\.([0-9]*?)0*$', 1))"

    @property
    def integer(self) -> str:
        return f"length(regexp_extract({self.text}, '^[+-]?0*([0-9]*)', 1))"

    def parsed(self, scale: int) -> str:
        precision = self.spec.decimal_precision or 38
        return (
            f"CASE WHEN {self.fraction} <= {scale} AND {self.integer} + {scale} <= {precision} "
            f"THEN try_cast({self.text} AS DECIMAL({precision}, {scale})) END"
        )


class DecimalColumnScale:
    """Resolve every textual decimal column in a single aggregate scan."""

    @staticmethod
    def resolve_many(connection, table, types, columns) -> dict[str, int]:
        fixed = {
            name: spec.decimal_scale
            for name, spec in columns.items()
            if spec.dtype == "decimal"
            and spec.decimal_scale is not None
            and types.get(name, "").startswith(("VARCHAR", "CHAR", "ENUM"))
        }
        selected = {
            name: DecimalTextSQL(qi(name), spec)
            for name, spec in columns.items()
            if spec.dtype == "decimal"
            and spec.decimal_scale is None
            and types.get(name, "").startswith(("VARCHAR", "CHAR", "ENUM"))
        }
        if not selected:
            return fixed
        aggregates = []
        for policy in selected.values():
            # Ignore configured text sentinels/N/A while inferring representable scale.
            exempt = [NotApplicablePolicy.from_spec(policy.spec).sql(policy.raw)]
            markers = [v for v in policy.spec.missing_values if isinstance(v, str)]
            if markers:
                exempt.append(f"{policy.raw} IN ({', '.join(lit(v) for v in markers)})")
            scope = "NOT (" + " OR ".join(f"({e})" for e in exempt) + ")"
            # Impossible outliers must not lower the grid for representable rows.
            scope += f" AND {policy.integer} + {policy.fraction} <= 38"
            fraction = f"coalesce(max({policy.fraction}) FILTER (WHERE {scope}), 0)"
            integer = f"coalesce(max({policy.integer}) FILTER (WHERE {scope}), 0)"
            aggregates.append(f"greatest(0, least({fraction}, 38 - {integer}))")
        values = connection.execute(f"SELECT {', '.join(aggregates)} FROM {qi(table)}").fetchone()
        return fixed | dict(zip(selected, map(int, values), strict=True))


@dataclass(frozen=True)
class ColumnValue:
    value: str
    invalid: str
    missing: str
    na: str
    dtype: str

    @classmethod
    def compile(cls, column, dtype, spec, decimal_scale=None):
        raw = qi(column)
        na = NotApplicablePolicy.from_spec(spec).sql(raw)
        text = dtype.startswith(("VARCHAR", "CHAR", "ENUM"))
        invalid = "FALSE"
        value = raw
        physical = dtype
        text_spec = spec.text_type()
        if text and text_spec is not None:
            value, invalid = sql_value(raw, text_spec)
            physical = {
                "int": "BIGINT",
                "float": "DOUBLE",
                "bool": "BOOLEAN",
                "date": "DATE",
                "datetime": text_spec.sql_datetime_type,
            }[spec.dtype]
        elif text and spec.dtype == "decimal":
            if decimal_scale is None:
                decimal_scale = spec.decimal_scale
            if decimal_scale is None:
                raise ValueError("textual decimal conversion requires resolved column scale")
            policy = DecimalTextSQL(raw, spec)
            value = policy.parsed(decimal_scale)
            markers = [v for v in spec.missing_values if isinstance(v, str)]
            present = f"{raw} IS NOT NULL"
            if markers:
                present += f" AND {raw} NOT IN ({', '.join(lit(v) for v in markers)})"
            if spec.blank_as_missing:
                present += f" AND {sql_stripped(raw, spec.trim_whitespace)} != ''"
            invalid = f"({present}) AND ({value}) IS NULL"
            physical = f"DECIMAL({spec.decimal_precision or 38},{decimal_scale})"
        elif (
            spec.dtype == "decimal"
            and dtype.startswith("DECIMAL")
            and spec.decimal_precision is not None
        ):
            precision, scale = spec.decimal_precision, spec.decimal_scale
            # Cast only after exact fixed-point admission; TRY_CAST alone rounds.
            valid = (
                f"abs({raw}) < {10 ** (precision - scale)} AND "
                f"{decimal_places_condition(raw, dtype, scale)}"
            )
            # DuckDB 1.3 rejects valid near-boundary DECIMAL->narrower DECIMAL
            # casts before rescaling. The exact text bridge avoids that bug;
            # the admission predicate above still prevents any rounding.
            value = (
                f"CASE WHEN {valid} THEN try_cast(CAST({raw} AS VARCHAR) "
                f"AS DECIMAL({precision},{scale})) END"
            )
            invalid = f"{raw} IS NOT NULL AND ({value}) IS NULL"
            physical = f"DECIMAL({precision},{scale})"
        elif (
            spec.dtype == "datetime"
            and dtype.startswith("TIMESTAMP")
            and text_spec.governed_datetime
        ):
            target = text_spec.sql_datetime_type
            unit = "ns" if dtype == "TIMESTAMP_NS" else "ms" if dtype == "TIMESTAMP_MS" else "us"
            ticks = {"ns": 1, "us": 1000, "ms": 1_000_000}
            ratio = ticks[spec.datetime_unit] // ticks[unit]
            exact = f"epoch_{unit}({raw}) % {ratio} = 0" if ratio > 1 else "TRUE"
            canonical = f"CAST({raw} AS TIMESTAMP)" if "WITH TIME ZONE" in dtype else raw
            value = f"CASE WHEN {exact} THEN try_cast({canonical} AS {target}) END"
            invalid = f"{raw} IS NOT NULL AND ({value}) IS NULL"
            physical = target
        elif spec.dtype == "int" and dtype in ("FLOAT", "DOUBLE", "REAL"):
            # Spreadsheet and dataframe integer-valued floats already have a
            # physical representation. Assess whole-number conformance without
            # changing their storage or rounding invalid fractions.
            invalid = (
                f"{raw} IS NOT NULL AND NOT isnan({raw}) "
                f"AND (isinf({raw}) OR {raw} != trunc({raw}))"
            )
        elif spec.dtype is not None:
            accepted = {
                "str": text,
                "int": dtype in _INTEGER_BITS,
                "float": dtype in _INTEGER_BITS or dtype.startswith(("FLOAT", "DOUBLE", "DECIMAL")),
                "decimal": dtype.startswith("DECIMAL"),
                "bool": dtype == "BOOLEAN",
                "date": dtype == "DATE",
                "datetime": dtype.startswith("TIMESTAMP") or dtype == "DATE",
            }[spec.dtype]
            if not accepted:
                invalid = f"{raw} IS NOT NULL"
                value = "NULL"
        missing = [f"({value}) IS NULL"]
        if physical in ("FLOAT", "DOUBLE", "REAL"):
            missing.append(f"isnan({value})")
        textual_markers = [v for v in spec.missing_values if isinstance(v, str)]
        typed_markers = [v for v in spec.missing_values if not isinstance(v, str)]
        if text and textual_markers:
            missing.append(f"{raw} IN ({', '.join(lit(v) for v in textual_markers)})")
        # Text markers describe raw placeholders in a declared non-text column;
        # typed markers describe its canonical values, regardless of input storage.
        compare = typed_markers + (
            textual_markers if spec.dtype in (None, "str") and not text else []
        )
        if spec.timezone:
            from qavf.text_policies import temporal_literal

            compare = [temporal_literal(value, spec) for value in compare]
        usable, foreign = NativeValueDomain(physical).members(compare)
        if foreign:
            raise ValueError(f"missing_values {foreign[:3]!r} cannot apply to {physical}")
        if usable:
            missing.append(f"({value}) IN ({', '.join(native_literal(v) for v in usable)})")
        if text and spec.blank_as_missing:
            missing.append(f"{sql_stripped(raw, spec.trim_whitespace)} = ''")
        is_missing = " OR ".join(f"({item})" for item in missing)
        normalized = f"CASE WHEN ({is_missing}) OR ({na}) THEN NULL ELSE {value} END"
        return cls(
            normalized,
            f"({invalid}) AND NOT ({na})",
            f"({is_missing}) AND NOT ({na})",
            na,
            physical,
        )


def normalized(expression, methods):
    ordered = [
        method
        for method in ("NFC", "NFD", "NFKC", "NFKD", "trim", "casefold", "case")
        if method in methods and not (method == "case" and "casefold" in methods)
    ]
    for method in ordered:
        if method == "case":
            expression = f"__qavf_unicode_lower({expression})"
        elif method == "trim":
            expression = f"trim({expression}, {lit(COMPARISON_WHITESPACE)})"
        else:
            expression = f"__qavf_unicode_normalize({expression}, {lit(method)})"
    return expression
