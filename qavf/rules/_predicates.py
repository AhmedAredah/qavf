"""Typed native Polars predicates for built-in column validation.

Predicates are evaluated as reductions and bounded samples by input rules.
No all-failure table or third-party schema engine is required.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

import polars as pl

from qavf.value_policy import _SHORT_LIST as _SHORT_LIST
from qavf.value_policy import _fits as _fits
from qavf.value_policy import _text as _text
from qavf.value_policy import is_nan as is_nan
from qavf.value_policy import is_temporal as is_temporal
from qavf.value_policy import is_text as is_text
from qavf.value_policy import members as members
from qavf.value_policy import membership as membership
from qavf.value_policy import temporal_ticks

RowTest = Callable[[str], pl.Expr]
"""Builds a boolean expression (``True`` = valid) for the named column."""


def compatibility_error(name: str, value: Any, dtype: pl.DataType) -> str | None:
    """Why limit ``name`` = ``value`` cannot apply to a ``dtype`` column; ``None`` if it can."""
    if name in ("ge", "le", "gt", "lt"):
        temporal_bound = isinstance(value, date | datetime)
        if temporal_bound and not is_temporal(dtype):
            return f"date bound {name} {value} cannot apply to a {dtype} column"
        if not temporal_bound and not dtype.is_numeric():
            return f"numeric bound {name} {value} cannot apply to a {dtype} column"
        if isinstance(value, float) and math.isnan(value):
            return f"bound {name} is NaN"
        aware_bound = isinstance(value, datetime) and value.tzinfo is not None
        aware_column = isinstance(dtype, pl.Datetime) and bool(dtype.time_zone)
        if aware_bound and not aware_column:
            return (
                f"time-zone-aware bound {name} {value} cannot apply to a {dtype} column "
                f"without a time zone; give the bound without a time zone"
            )
    if name in ("regex", "min_length", "max_length") and not is_text(dtype):
        return f"text limit {name} cannot apply to a {dtype} column"
    if name in ("integral", "max_decimals") and not dtype.is_numeric():
        return f"numeric limit {name} cannot apply to a {dtype} column"
    if name in ("isin", "notin"):
        _, foreign = members(value, dtype)
        if foreign:
            shown = ", ".join(repr(v) for v in foreign[:3])
            return f"{name} value(s) {shown} cannot apply to a {dtype} column"
    return None


def format_value(value: Any, limit: int = 6) -> str:
    """Compact rendering of a limit value for labels."""
    if isinstance(value, list):
        shown = ", ".join(format_value(v) for v in value[:limit])
        return f"[{shown}{', …' if len(value) > limit else ''}]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _temporal_compare(operator: str, value: Any, dtype: pl.DataType) -> RowTest:
    requested = temporal_ticks(value, dtype)
    lower = operator in ("ge", "gt")
    threshold = (
        (math.ceil(requested) if operator == "ge" else math.floor(requested) + 1)
        if lower
        else (math.floor(requested) if operator == "le" else math.ceil(requested) - 1)
    )

    def test(column: str) -> pl.Expr:
        if threshold > 2**63 - 1:
            return pl.lit(not lower)
        if threshold < -(2**63):
            return pl.lit(lower)
        ticks = pl.col(column).cast(pl.Int64)
        return ticks >= threshold if lower else ticks <= threshold

    return test


def row_tests(constraints: Mapping[str, Any], dtype: pl.DataType) -> dict[str, tuple[str, RowTest]]:
    """Polars predicates (``True`` = valid) for the row-level limits in ``constraints``.

    NaN values in ``isin``/``notin`` lists are ignored here: NaN counts as
    missing, and an explicitly forbidden NaN is enforced by the caller.
    """
    tests: dict[str, tuple[str, RowTest]] = {}
    for name, raw in constraints.items():
        label = f"{name} {format_value(raw)}"
        if name in ("ge", "le", "gt", "lt"):
            bound = raw
            if dtype.is_integer() and isinstance(raw, int | float | Decimal):
                bound = math.ceil(raw) if name in ("ge", "lt") else math.floor(raw)
            if is_temporal(dtype):
                tests[name] = (label, _temporal_compare(name, bound, dtype))
            elif dtype.is_float():
                operator, threshold = float_bound(name, bound)

                def floating(column: str, op: str = operator, v: float = threshold) -> pl.Expr:
                    value = pl.col(column).cast(pl.Float64)
                    return value >= v if op == "ge" else value <= v

                tests[name] = (label, floating)
            elif dtype.is_decimal():
                operator, decimal = decimal_bound(name, bound, dtype)
                if isinstance(decimal, bool):

                    def constant(column: str, valid: bool = decimal) -> pl.Expr:
                        return pl.lit(valid)

                    tests[name] = (label, constant)
                else:
                    tests[name] = (label, _compare(operator, decimal))
            else:
                tests[name] = (label, _compare(name, bound))
        elif name in ("isin", "notin"):
            values, _ = members(raw, dtype)
            tests[name] = (label, membership(values, dtype, negate=name == "notin"))
        elif name == "regex":
            tests[name] = (f"regex {raw}", _full_match(raw))
        elif name in ("min_length", "max_length"):
            tests[name] = (label, _length(raw, at_least=name == "min_length"))
        elif name == "integral":
            tests[name] = ("integral", _decimals(0, dtype))
        elif name == "max_decimals":
            tests[name] = (label, _decimals(raw, dtype))
    return tests


def _decimals(places: int, dtype: pl.DataType) -> RowTest:
    """At most ``places`` decimal places. Integers always pass; decimals are checked
    exactly on their digits; floats within :data:`_DECIMALS_RTOL` of the rounding."""

    def test(column: str) -> pl.Expr:
        if dtype.is_integer():
            return pl.lit(True)
        if dtype.is_decimal():
            fraction = pl.col(column).cast(pl.String).str.extract(r"\.(\d*?)0*$", 1)
            return fraction.fill_null("").str.len_chars() <= places
        # Once the decimal quantum is smaller than binary spacing, every
        # representable value satisfies it. This also avoids overflow in scaling.
        if places >= 324:
            return pl.lit(True)
        scale = 10.0 ** min(places, 308)
        extra = 10.0 ** max(0, places - 308)
        value = pl.col(column).cast(pl.Float64)
        # Remove the integral part first. Scaling the whole value can erase
        # .125 from a large-but-nonintegral float before checking two places.
        fraction = (value - value.floor()).abs()
        scaled = fraction * scale * extra
        gap = (scaled - scaled.round(0)).abs()
        epsilon = 2.220446049250313e-16
        tolerance = pl.min_horizontal(
            pl.lit(1e-7), 4 * epsilon * pl.max_horizontal(scaled.abs(), pl.lit(1.0))
        )
        return (fraction == 0) | (fraction >= (2.0**52 / scale / extra)) | (gap <= tolerance)

    return test


def decimal_bound(
    operator: str,
    bound: int | float | Decimal,
    dtype: pl.DataType,
) -> tuple[str, Decimal | bool]:
    """Exact inequality on the Decimal storage grid, without float coercion."""
    assert isinstance(dtype, pl.Decimal)
    scale = dtype.scale or 0
    requested = Fraction(bound) * 10**scale
    if operator in ("ge", "gt"):
        coefficient = math.ceil(requested) if operator == "ge" else math.floor(requested) + 1
        inclusive = "ge"
    else:
        coefficient = math.floor(requested) if operator == "le" else math.ceil(requested) - 1
        inclusive = "le"
    maximum = 10 ** (dtype.precision or 38) - 1
    if coefficient > maximum:
        return inclusive, inclusive == "le"
    if coefficient < -maximum:
        return inclusive, inclusive == "ge"
    with localcontext() as context:
        context.prec = 100
        return inclusive, Decimal(coefficient).scaleb(-scale)


def float_bound(operator: str, bound: int | float | Decimal) -> tuple[str, float]:
    """Directed Float64 threshold equivalent to the exact numeric inequality.

    Float32 data is promoted losslessly to Float64 by consumers. Strict bounds
    become inclusive thresholds at the adjacent representable value.
    """
    requested = Fraction(bound)
    try:
        value = float(bound)
    except OverflowError:
        value = math.inf if requested > 0 else -math.inf
    if math.isinf(value):
        rounded = 1 if value > 0 else -1
    else:
        actual = Fraction(value)
        rounded = (actual > requested) - (actual < requested)
    if operator in ("ge", "gt"):
        if rounded < 0 or (operator == "gt" and rounded == 0):
            value = math.nextafter(value, math.inf)
        return "ge", value
    if rounded > 0 or (operator == "lt" and rounded == 0):
        value = math.nextafter(value, -math.inf)
    return "le", value


def _compare(operator: str, bound: Any) -> RowTest:
    def test(column: str) -> pl.Expr:
        expr = pl.col(column)
        match operator:
            case "ge":
                return expr >= bound
            case "le":
                return expr <= bound
            case "gt":
                return expr > bound
            case _:
                return expr < bound

    return test


def _full_match(pattern: str) -> RowTest:
    anchored = f"^(?:{pattern})$"

    def test(column: str) -> pl.Expr:
        return _text(column).str.contains(anchored)

    return test


def _length(limit: int, at_least: bool) -> RowTest:
    def test(column: str) -> pl.Expr:
        length = _text(column).str.len_chars()
        return length >= limit if at_least else length <= limit

    return test
