"""Outlier rule execution using one coherent projection and indexed group state."""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.key_codec import canonical_key
from qavf.rules.helpers import annotate, fresh_internal
from qavf.statistics.ordered import OutlierModel, fit_outliers, scaled
from qavf.stats import DEFAULT_THRESHOLDS, OutlierMethod, checked_float_series, minimum_samples
from qavf.typed_values import encode_temporal

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation
    from qavf.engine.observation_plan import ObservationView
    from qavf.engine.sqlite_store import Store
    from qavf.rules.statistical_checks import OutlierTarget


@dataclass(slots=True)
class TargetScan:
    tested: int = 0
    outliers: int = 0
    invalid: int = 0
    nonpositive: int = 0
    untested: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    untested_records: list[dict[str, Any]] = field(default_factory=list)


def _rows(store: Store, key: bytes, *, log: bool = False) -> Iterator[tuple[float, int]]:
    where = " AND value > 0" if log else ""
    cursor = store.execute(
        "SELECT value, frequency FROM values_by_group WHERE identity=?" + where + " ORDER BY value",
        (key,),
    )
    try:
        while batch := cursor.fetchmany(2048):
            store.resources.session.check()
            for value, count in batch:
                yield (math.log(value) if log else value), count
    finally:
        cursor.close()


def _median(rows: Iterator[tuple[float, int]], size: int) -> float:
    left, right = (size - 1) // 2, size // 2
    seen = 0
    a = b = None
    for value, count in rows:
        if a is None and seen + count > left:
            a = value
        if seen + count > right:
            b = value
            break
        seen += count
    if a is None or b is None:
        raise ValueError("group observation counts changed during median evaluation")
    if a == b:
        return a
    return a / 2 + b / 2 if a < 0 < b else a + (b - a) / 2


def _fit_groups(
    store: Store,
    target: OutlierTarget,
    method: OutlierMethod,
    threshold: float | None,
    minimum: int,
) -> None:
    log = target.transform == "log"
    if log:
        # Log must preserve globally distinct observations, not just values
        # within one group. An index supplies this scan without a temp sort.
        cursor = store.execute("SELECT value FROM values_by_group WHERE value>0 ORDER BY value")
        prior_value = prior_log = None
        for (value,) in cursor:
            transformed = math.log(value)
            if value != prior_value and transformed == prior_log:
                raise ValueError(
                    "log transformation cannot preserve distinct observations at Float64 precision"
                )
            prior_value, prior_log = value, transformed
        cursor.close()
    needed = max(
        minimum,
        minimum_samples(method, DEFAULT_THRESHOLDS[method] if threshold is None else threshold),
    )
    expression = "CASE WHEN value>0 THEN frequency ELSE 0 END" if log else "frequency"
    value_expr = "CASE WHEN value>0 THEN value END" if log else "value"
    groups = store.execute(
        f"SELECT identity,sum(frequency),sum({expression}),min({value_expr}),max({value_expr}) "
        f"FROM values_by_group GROUP BY identity HAVING sum({expression})>=? ORDER BY identity",
        (min(needed, 2**63 - 1),),
    )
    for key, raw_size, size, lo, hi in groups:
        store.resources.session.check()
        original_median = _median(_rows(store, key), raw_size)
        if log:
            lo, hi = math.log(lo), math.log(hi)

        def deviations(centre: float, origin: float, unit: float, key: bytes = key):
            store.execute("DELETE FROM deviations")
            store.executemany(
                "INSERT INTO deviations VALUES(?,?) ON CONFLICT(value) DO UPDATE SET "
                "frequency=frequency+excluded.frequency",
                (
                    (abs(scaled(value, origin, unit) - centre), count)
                    for value, count in _rows(store, key, log=log)
                ),
            )
            cursor = store.execute("SELECT value,frequency FROM deviations ORDER BY value")
            try:
                yield from cursor
            finally:
                cursor.close()

        def group_rows(key: bytes = key) -> Iterator[tuple[float, int]]:
            return _rows(store, key, log=log)

        model = fit_outliers(
            group_rows,
            size=size,
            minimum=lo,
            maximum=hi,
            method=method,
            threshold=threshold,
            min_group_size=minimum,
            original_median=original_median,
            deviations=deviations,
        )
        store.execute("INSERT INTO models VALUES(?,?)", (key, json.dumps(asdict(model))))
    groups.close()


def scan_target(
    ctx: DataContext,
    target: OutlierTarget,
    method: OutlierMethod,
    threshold: float | None,
    min_group_size: int,
    evidence_limit: int,
    normalize: Callable,
    observation: Observation | ObservationView | None = None,
) -> TargetScan:
    """Fit each whole group, then score bounded batches against indexed models."""
    columns = list(dict.fromkeys((*target.group_by, target.value)))
    row_name = fresh_internal("row", ctx.schema(target.dataset))
    result = TargetScan()
    scope = (
        ctx.observe(target.dataset, columns) if observation is None else nullcontext(observation)
    )
    with (
        scope as observation,
        ctx.work_store("outlier group statistics") as store,
    ):
        store.execute(
            "CREATE TABLE values_by_group(identity BLOB,value REAL,frequency INTEGER NOT NULL,"
            "PRIMARY KEY(identity,value)) WITHOUT ROWID"
        )
        store.execute("CREATE INDEX values_order ON values_by_group(value)")
        store.execute(
            "CREATE TABLE deviations(value REAL PRIMARY KEY,frequency INTEGER NOT NULL) "
            "WITHOUT ROWID"
        )
        store.execute(
            "CREATE TABLE models(identity BLOB PRIMARY KEY,model TEXT NOT NULL) WITHOUT ROWID"
        )
        offset = 0
        for original in observation.batches():
            source = normalize(ctx, target.dataset, original, columns)
            values = checked_float_series(source[target.value])
            valid = values.is_finite().fill_null(False)
            invalid = (~valid).arg_true()
            result.invalid += len(invalid)
            for index in invalid.head(max(0, ctx.evidence_limit - len(result.invalid_records))):
                record = original.row(int(index), named=True)
                record[row_name] = offset + int(index)
                result.invalid_records.append(record)
            finite = source.with_columns(values.alias(target.value)).filter(valid)
            if target.transform == "log":
                result.nonpositive += finite.filter(pl.col(target.value) <= 0).height
            # Batch-local grouping reduces duplicate key encoding and index
            # updates, while retained work stays proportional to batch size.
            count_name = fresh_internal("frequency", finite.columns)
            reduced = finite.group_by(columns).len(name=count_name)
            reduced = encode_temporal(reduced, target.group_by)
            group_indices = [reduced.columns.index(name) for name in target.group_by]
            value_index = reduced.columns.index(target.value)
            count_index = reduced.columns.index(count_name)
            store.executemany(
                "INSERT INTO values_by_group VALUES(?,?,?) ON CONFLICT(identity,value) "
                "DO UPDATE SET frequency=frequency+excluded.frequency",
                (
                    (
                        canonical_key([row[i] for i in group_indices]),
                        row[value_index],
                        row[count_index],
                    )
                    for row in reduced.iter_rows()
                ),
            )
            offset += original.height
        _fit_groups(store, target, method, threshold, min_group_size)
        best: list[tuple[float, int, dict[str, Any]]] = []
        offset = 0
        for original in observation.batches():
            source = normalize(ctx, target.dataset, original, columns)
            values = checked_float_series(source[target.value])
            # The model cache is per batch and cleared before the next;
            # a high-cardinality input cannot accumulate all group models.
            models: dict[bytes, OutlierModel | None] = {}
            keys = (
                encode_temporal(source.select(list(target.group_by))) if target.group_by else None
            )
            key_rows = keys.iter_rows() if keys is not None else iter([()] * original.height)
            for index, (value, key_values) in enumerate(zip(values, key_rows, strict=True)):
                if (
                    value is None
                    or not math.isfinite(value)
                    or (target.transform == "log" and value <= 0)
                ):
                    continue
                key = canonical_key(key_values)
                if key not in models:
                    row = store.execute(
                        "SELECT model FROM models WHERE identity=?", (key,)
                    ).fetchone()
                    if len(models) >= 1024:
                        models.clear()
                    if row is None:
                        models[key] = None
                    else:
                        data = json.loads(row[0])
                        data["method"] = OutlierMethod(data["method"])
                        models[key] = OutlierModel(**data)
                model = models[key]
                if model is None or not model.testable:
                    result.untested += 1
                    if len(result.untested_records) < ctx.evidence_limit:
                        record = original.row(index, named=True)
                        record[row_name] = offset + index
                        result.untested_records.append(record)
                    continue
                result.tested += 1
                transformed = math.log(value) if target.transform == "log" else value
                score, flagged = model.score(transformed)
                if (
                    not flagged
                    or (target.direction == "high" and score <= 0)
                    or (target.direction == "low" and score >= 0)
                ):
                    continue
                relative = None
                if target.min_relative_deviation > 0:
                    median = model.original_median
                    ratio = value / median if median else math.inf
                    relative = abs((value - median) / median if 0.5 <= ratio <= 2 else ratio - 1)
                    if relative < target.min_relative_deviation:
                        continue
                result.outliers += 1
                ordinal = offset + index
                if evidence_limit and (
                    len(best) < evidence_limit or (abs(score), -ordinal) > best[0][:2]
                ):
                    record = original.row(index, named=True)
                    record[row_name] = ordinal
                    fields = {
                        "score": score,
                        "lower_bound": model.data_units(model.lower),
                        "upper_bound": model.data_units(model.upper),
                        "group_median": model.data_units(model.median),
                    }
                    if target.transform == "log":
                        fields["log_value"] = transformed
                    if relative is not None:
                        fields["relative_deviation"] = relative
                    item = (abs(score), -ordinal, annotate(record, **fields))
                    if len(best) < evidence_limit:
                        heapq.heappush(best, item)
                    else:
                        heapq.heapreplace(best, item)
            offset += original.height
        result.records = [
            row for _, _, row in sorted(best, key=lambda item: item[:2], reverse=True)
        ]

    return result
