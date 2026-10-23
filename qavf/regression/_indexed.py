"""Exact typed comparison through bounded batches and disk-indexed row alignment.

The work database stores canonical keys and tagged values, never approximate
hashes. Original schemas are retained separately; temporal cells are normalized
losslessly before crossing Python's microsecond datetime boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from io import BytesIO
from typing import Any, Protocol

import polars as pl

from qavf.audit_codec import decode_audit, iter_audit_json
from qavf.engine.resources import ResourceLimitError
from qavf.engine.sqlite_store import Store
from qavf.key_codec import canonical_key
from qavf.keys import FLOAT_EXACT_LIMIT, KeyTypeError, key_casts
from qavf.regression._comparison import ComparisonAccumulator
from qavf.regression._plan import ComparisonPlan
from qavf.regression.models import ComparisonResult
from qavf.typed_values import encode_temporal, has_object


class BatchInput(Protocol):
    def schema(self) -> dict[str, pl.DataType]: ...
    def batches(self, batch_rows: int = 65536) -> Iterator[pl.DataFrame]: ...


class MappedInput:
    """Present a stored baseline projection under the plan's candidate names."""

    def __init__(self, source: BatchInput, plan: ComparisonPlan) -> None:
        self._source, self._mapping = source, dict(plan.column_map)

    def schema(self) -> dict[str, pl.DataType]:
        schema = self._source.schema()
        return {self._mapping.get(name, name): dtype for name, dtype in schema.items()}

    def batches(self, batch_rows: int = 65536) -> Iterator[pl.DataFrame]:
        for batch in self._source.batches(batch_rows):
            present = {old: new for old, new in self._mapping.items() if old in batch.columns}
            yield batch.rename(present)


class ProjectedInput:
    """Present only the selected columns of a wider batch input."""

    def __init__(self, source: BatchInput, columns: Sequence[str]) -> None:
        self._source, self._columns = source, list(columns)

    def schema(self) -> dict[str, pl.DataType]:
        schema = self._source.schema()
        return {name: schema[name] for name in self._columns}

    def batches(self, batch_rows: int = 65536) -> Iterator[pl.DataFrame]:
        for batch in self._source.batches(batch_rows):
            yield batch.select(self._columns)


def supported(plan: ComparisonPlan) -> bool:
    """Opaque Object cells have no portable typed reader/writer contract."""

    return not any(
        has_object(schema[column])
        for schema in (plan.baseline_schema, plan.candidate_schema)
        for column in plan.shared
    )


def _row_bytes(row: tuple[Any, ...], limit: int | None) -> bytes:
    encoded = BytesIO()
    for chunk in iter_audit_json(row):
        if limit is not None and encoded.tell() + len(chunk) > limit:
            raise ResourceLimitError("one comparison row exceeds the bounded work allowance")
        encoded.write(chunk)
    return encoded.getvalue()


def _key_types(
    plan: ComparisonPlan,
) -> tuple[dict[str, pl.DataType], dict[str, pl.DataType], set[str]]:
    schemas = [
        encode_temporal(pl.DataFrame(schema={key: schema[key] for key in plan.keys})).schema
        for schema in (plan.baseline_schema, plan.candidate_schema)
    ]
    left_casts: dict[str, pl.DataType] = {}
    right_casts: dict[str, pl.DataType] = {}
    mixed: set[str] = set()
    for key in plan.keys:
        left, right = schemas[0][key], schemas[1][key]
        a, b = key_casts(key, left, right, exact_float=lambda: True)
        if left != right and (
            (left.is_integer() and right.is_float()) or (left.is_float() and right.is_integer())
        ):
            mixed.add(key)
        if a is not None:
            left_casts[key] = a
        if b is not None:
            right_casts[key] = b
    return left_casts, right_casts, mixed


def _stage(
    store: Store,
    table: str,
    source: BatchInput,
    plan: ComparisonPlan,
    casts: Mapping[str, pl.DataType],
    mixed: set[str],
    limit: int | None,
) -> tuple[int, dict[str, pl.DataType]]:
    store.execute(f"CREATE TABLE {table} (ordinal INTEGER PRIMARY KEY, key BLOB, payload BLOB)")
    store.execute(f"CREATE INDEX {table}_key ON {table}(key)")
    schema = dict(encode_temporal(pl.DataFrame(schema=source.schema())).schema)
    schema.update(casts)
    ordinal = 0
    indexes = [list(schema).index(key) for key in plan.keys]

    def rows(batch: pl.DataFrame):
        nonlocal ordinal
        for key in plan.keys:
            values = batch[key]
            if values.dtype.is_float() and values.is_nan().any():
                raise ValueError(f"{plan.name}: {table} key column {key!r} contains NaN")
            if key in mixed:
                for value in values:
                    if value is not None and (
                        abs(value) > FLOAT_EXACT_LIMIT or int(value) != value
                    ):
                        raise KeyTypeError(
                            f"{plan.name}: key column {key!r} cannot be compared exactly: "
                            "fractional values or magnitudes beyond 2**53"
                        )
        batch = encode_temporal(batch)
        if casts:
            batch = batch.cast(pl.Schema(casts))
        for row in batch.iter_rows():
            encoded_key = canonical_key(tuple(row[index] for index in indexes), max_bytes=limit)
            yield ordinal, encoded_key, _row_bytes(row, limit)
            ordinal += 1

    for batch in source.batches(1024):
        store.executemany(f"INSERT INTO {table} VALUES (?,?,?)", rows(batch))
    duplicate = store.execute(
        f"SELECT coalesce(sum(n),0) FROM "
        f"(SELECT count(*) AS n FROM {table} GROUP BY key HAVING count(*) > 1)"
    ).fetchone()[0]
    if duplicate:
        raise ValueError(
            f"{plan.name}: {table} has {duplicate} row(s) with duplicate key "
            f"({', '.join(plan.keys)}); keys must identify rows uniquely"
        )
    return ordinal, schema


def _frames(
    rows: Iterator[Any],
    left_schema: Mapping[str, pl.DataType],
    right_schema: Mapping[str, pl.DataType] | None,
    limit: int,
) -> Iterator[tuple[pl.DataFrame, pl.DataFrame | None]]:
    left: list[Any] = []
    right: list[Any] = []
    used = 0

    def batch():
        return (
            pl.DataFrame(left, schema=dict(left_schema), orient="row"),
            pl.DataFrame(right, schema=dict(right_schema), orient="row")
            if right_schema is not None
            else None,
        )

    for row in rows:
        size = sum(len(value) for value in row)
        if left and (used + size > limit or len(left) >= 16_384):
            yield batch()
            left.clear()
            right.clear()
            used = 0
        left.append(decode_audit(json.loads(row[0])))
        if right_schema is not None:
            right.append(decode_audit(json.loads(row[1])))
        used += size
    if left:
        yield batch()


def compare_indexed(
    store: Store,
    baseline: BatchInput,
    candidate: BatchInput,
    plan: ComparisonPlan,
    *,
    fallback_reason: str | None = None,
) -> ComparisonResult:
    """Align one exact snapshot of each input using disk-resident indexes."""
    memory = store.resources.memory_bytes
    allowance = min(8 << 20, memory // 8) if memory is not None else 8 << 20
    left_casts, right_casts, mixed = _key_types(plan)
    row_limit = memory // 4 if memory is not None else None
    nleft, left_schema = _stage(store, "baseline", baseline, plan, left_casts, mixed, row_limit)
    nright, right_schema = _stage(
        store, "candidate", candidate, plan, right_casts, mixed, row_limit
    )
    accumulator = ComparisonAccumulator(plan, nleft, nright)
    for table, other, kind, schema in (
        ("baseline", "candidate", "row_removed", left_schema),
        ("candidate", "baseline", "row_added", right_schema),
    ):
        rows = store.execute(
            f"SELECT a.payload FROM {table} AS a LEFT JOIN {other} AS b ON a.key=b.key "
            "WHERE b.ordinal IS NULL ORDER BY a.ordinal"
        )
        for frame, _ in _frames(rows, schema, None, allowance):
            accumulator.unmatched(kind, frame)
    rows = store.execute(
        "SELECT a.payload,b.payload FROM baseline AS a JOIN candidate AS b ON a.key=b.key "
        "ORDER BY a.ordinal"
    )
    for left, right in _frames(rows, left_schema, right_schema, allowance):
        assert right is not None
        accumulator.matched(left, right)
    return accumulator.finish(execution="typed_stream", fallback_reason=fallback_reason)
