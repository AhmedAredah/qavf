"""Logical-type capabilities of the SQL and Polars file readers."""

from __future__ import annotations

import polars as pl
from polars.datatypes import DataTypeClass


def needs_typed_reader(dtype: pl.DataType | DataTypeClass) -> bool:
    """Whether DuckDB's Arrow conversion can erase this Polars logical type.

    Temporal units and zones, encoded categories, fixed array shape and 128-bit
    integers must survive before a QA predicate or comparison sees the values.
    Check containers recursively, including empty columns with only a schema.
    """
    if dtype in (pl.Int128, pl.Categorical, pl.Time):
        return True
    if isinstance(dtype, pl.Datetime | pl.Duration | pl.Enum | pl.Array):
        return True
    if isinstance(dtype, pl.List):
        return needs_typed_reader(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(needs_typed_reader(field.dtype) for field in dtype.fields)
    return False


def needs_typed_sql_binding(dtype: pl.DataType | DataTypeClass) -> bool:
    """Whether registering an already typed frame can narrow its SQL representation.

    DuckDB's in-memory Arrow binding preserves naive timestamp units, unlike
    its broader file-reader capability boundary. Zoned timestamps lose their
    declared zone and can lose nanoseconds, so they still require typed checks.
    """
    if isinstance(dtype, pl.Datetime):
        return dtype.time_zone is not None
    if isinstance(dtype, pl.List):
        return needs_typed_sql_binding(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(needs_typed_sql_binding(field.dtype) for field in dtype.fields)
    return needs_typed_reader(dtype)
