"""Opt-in Unicode and temporal policies shared by every execution backend.

Ordinary numeric and ISO microsecond parsing stays in native engine expressions.
These scalar policies cover contracts the engines cannot express identically.
They never inspect a complete source column or import a dataframe library.
"""

from __future__ import annotations

import re
import unicodedata
import weakref
from collections.abc import Mapping
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

from qavf.duckdb_results import settle
from qavf.number_grammar import COMPARISON_WHITESPACE, trim_text

_REGISTERED: weakref.WeakSet = weakref.WeakSet()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def text_type_requires_sql_functions(spec) -> bool:
    """Whether a declared text parser needs anything beyond native SQL operators."""
    dtype = _field(spec, "dtype")
    if dtype == "datetime":
        return _field(spec, "datetime_unit", "us") != "us" or _field(spec, "timezone") is not None
    if dtype == "bool":
        words = [*_field(spec, "true_values", ["true"]), *_field(spec, "false_values", ["false"])]
        whitespace = _field(spec, "trim_whitespace", "ascii")
        return any(not trim_text(str(word), whitespace).lower().isascii() for word in words)
    return False


def requires_sql_functions(settings) -> bool:
    """Cheap configuration-only capability detection; ordinary calls register no UDFs."""
    datasets = _field(settings, "datasets", settings)
    for dataset in datasets.values():
        schema = _field(dataset, "schema_spec") or _field(dataset, "schema")
        for column in _field(schema, "columns", {}).values():
            if text_type_requires_sql_functions(column):
                return True
            for check in (column, *_field(column, "checks", [])):
                if any(method != "trim" for method in _field(check, "normalize", [])):
                    return True
        source = _field(dataset, "source")
        if any(
            text_type_requires_sql_functions(column)
            for column in _field(source, "declared_columns", {}).values()
        ):
            return True
    return False


def normalize_scalar(value: str, methods) -> str:
    """Canonical Unicode form, whitespace, then case; order is engine independent."""
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        if form in methods:
            value = unicodedata.normalize(form, value)
    if "trim" in methods:
        value = value.strip(COMPARISON_WHITESPACE)
    if "casefold" in methods:
        value = value.casefold()
    elif "case" in methods:
        value = value.lower()
    return value


@lru_cache(maxsize=64)
def _zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def _localize(value: datetime, zone: str, ambiguous: str, nonexistent: str) -> datetime | None:
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    tz = _zone(zone)
    candidates = []
    for fold in (0, 1):
        aware = value.replace(tzinfo=tz, fold=fold)
        utc = aware.astimezone(UTC)
        if utc.astimezone(tz).replace(tzinfo=None) == value and utc not in candidates:
            candidates.append(utc)
    if not candidates:
        # Both policies reject the value as a type violation; no guessed shift.
        return None
    if len(candidates) > 1:
        if ambiguous not in ("earliest", "latest"):
            return None
        chosen = min(candidates) if ambiguous == "earliest" else max(candidates)
    else:
        chosen = candidates[0]
    return chosen.replace(tzinfo=None)


_ISO = re.compile(
    r"(?P<base>[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[ T][0-9]{2}:[0-9]{2}"
    r"(?::[0-9]{2})?)?)(?:\.(?P<fraction>[0-9]{1,9}))?"
    r"(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?"
)


def datetime_text(
    value: str, unit: str, zone: str, ambiguous: str, nonexistent: str, fmt: str = ""
) -> str | None:
    """Lossless ISO text at the declared resolution, UTC when a zone is declared.

    Fractions are carried separately from Python's microsecond datetime. Values
    between ticks of the selected unit are rejected, never rounded or truncated.
    Offset text requires an explicit timezone contract. Local DST gaps and folds
    are validated by a UTC round trip, independent of the process timezone.
    """
    if value is None:
        return None
    try:
        fraction = ""
        if fmt:
            parsed = datetime.strptime(value, fmt)
            fraction = f"{parsed.microsecond:06d}".rstrip("0")
        else:
            match = _ISO.fullmatch(value)
            if match is None:
                return None
            fraction = match["fraction"] or ""
            # A fraction belongs to seconds, never dates/minutes.
            if fraction and len(match["base"]) != 19:
                return None
            offset = match["offset"] or ""
            if offset and not zone:
                return None
            if offset and len(match["base"]) < 16:
                return None
            parsed = datetime.fromisoformat(match["base"] + offset.replace("Z", "+00:00"))
        digits = {"ms": 3, "us": 6, "ns": 9}[unit]
        if any(char != "0" for char in fraction[digits:]):
            return None
        if zone:
            localized = _localize(parsed, zone, ambiguous, nonexistent)
            if localized is None:
                return None
            parsed = localized
        elif parsed.tzinfo is not None:
            return None
        suffix = fraction[:digits].ljust(digits, "0")
        return parsed.strftime("%Y-%m-%d %H:%M:%S") + "." + suffix
    except (ValueError, OverflowError):
        return None


def temporal_literal(value: Any, spec) -> Any:
    """Configured temporal literals in the UTC-naive canonical value domain."""
    if not spec.timezone or not isinstance(value, date):
        return value
    dt = value if isinstance(value, datetime) else datetime.combine(value, datetime.min.time())
    dt = datetime(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second,
        dt.microsecond,
        tzinfo=dt.tzinfo,
        fold=dt.fold,
    )
    result = _localize(dt, spec.timezone, spec.ambiguous, spec.nonexistent)
    if result is None:
        raise ValueError("configured datetime literal is ambiguous or nonexistent in timezone")
    # pandas.Timestamp is a datetime subclass; preserve its nanosecond remainder
    # without importing pandas for ordinary validation.
    nanosecond = getattr(value, "nanosecond", 0)
    if nanosecond:
        import sys

        pandas = sys.modules.get("pandas")
        if pandas is not None:
            result = pandas.Timestamp(result) + pandas.Timedelta(nanosecond, unit="ns")
    return result


def _unicode_lower(value):
    """Lower-case a SQL text value, preserving NULL."""
    return value.lower() if value is not None else None


def _unicode_normalize(value, method):
    """Apply one normalization method to a SQL text value, preserving NULL."""
    return normalize_scalar(value, [method]) if value is not None else None


def register_sql_functions(connection) -> None:
    """Register shared scalar policies once on a connection, without importing Polars."""
    if connection in _REGISTERED:
        return
    functions = (
        ("__qavf_unicode_lower", _unicode_lower, [str]),
        (
            "__qavf_unicode_normalize",
            _unicode_normalize,
            [str, str],
        ),
        ("__qavf_datetime_text", datetime_text, [str, str, str, str, str, str]),
    )
    for name, function, parameters in functions:
        settle(connection)
        connection.create_function(name, function, parameters, str, null_handling="special")
    _REGISTERED.add(connection)
