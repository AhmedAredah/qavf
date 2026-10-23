"""Exact sorted-batch reductions with one group's numeric state at a time."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import polars as pl

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.engine.cache_ownership import retained_size
from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.numeric import checked_float, finite_float
from qavf.rules.helpers import fresh_internal


def supports_exact(dtype: pl.DataType, how: str) -> bool:
    return (how == "sum" and dtype.is_float()) or (
        how in {"mean", "weighted_mean"} and (dtype.is_numeric() or dtype == pl.Null)
    )


@dataclass(slots=True)
class _State:
    values: ExactFloatAccumulator = field(default_factory=ExactFloatAccumulator)
    weighted_units: int = 0
    weight_units: int = 0
    rows: int = 0
    missing: int = 0
    negative_weight: bool = False

    def add(self, value, weight, *, weighted: bool, nonfinite_missing: bool) -> None:
        self.rows += 1
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.missing += 1
            return
        converted = checked_float(value) if not isinstance(value, float) else value
        if not math.isfinite(converted):
            if nonfinite_missing:
                self.missing += 1
                return
            raise ValueError("non-finite observations cannot be aggregated")
        if not weighted:
            self.values.update((converted,))
            return
        if weight is None or (isinstance(weight, float) and math.isnan(weight)):
            return
        supplied_weight = checked_float(weight) if not isinstance(weight, float) else weight
        if not math.isfinite(supplied_weight):
            raise ValueError("weighted means require finite observations and weights")
        if supplied_weight < 0:
            self.negative_weight = True
            return
        numerator, denominator = converted.as_integer_ratio()
        weight_numerator, weight_denominator = supplied_weight.as_integer_ratio()
        value_exponent = denominator.bit_length() - 1
        weight_exponent = weight_denominator.bit_length() - 1
        self.weighted_units += numerator * weight_numerator << (
            2148 - value_exponent - weight_exponent
        )
        self.weight_units += weight_numerator << (1074 - weight_exponent)

    def result(self, how: str, rational: bool):
        if how == "weighted_mean":
            value = (
                Fraction(self.weighted_units, self.weight_units << 1074)
                if self.weight_units and not self.negative_weight
                else None
            )
        else:
            value = self.values.mean() if how == "mean" else self.values.total()
        if value is None:
            return None
        return str(value) if rational else finite_float(value)


def exact_grouped(
    ctx: DataContext,
    observation: Observation,
    plan: pl.LazyFrame,
    keys: Sequence[str],
    value: str,
    alias: str,
    how: str,
    *,
    weight: str | None = None,
    rational: bool = False,
    presence_alias: str | None = None,
    missing_alias: str | None = None,
    nonfinite_missing: bool = False,
    batch_rows: int = 8192,
) -> pl.DataFrame:
    """Reduce sorted observations without retaining member rows or a Python group map.

    Sorting remains native and may use native working memory; its managed IPC
    output is charged to the shared spill budget. Python state is one exact
    accumulator and one bounded output batch. The returned G-row aggregate is
    subject to the context's result limit; this is not a constant-memory claim
    when every input observation forms its own group.
    """
    selected = list(dict.fromkeys([*keys, value, *([weight] if weight is not None else [])]))
    schema = dict(plan.collect_schema())
    if how == "weighted_mean" and weight is None:
        raise ValueError("weighted_mean requires a weight column")
    group_id = fresh_internal(
        "exact_group", [*schema, alias, presence_alias or "", missing_alias or ""]
    )
    ordered = plan.select(selected)
    if keys:
        ordered = ordered.sort(list(keys), nulls_last=False).with_columns(
            pl.struct(list(keys)).rle_id().cast(pl.UInt64).alias(group_id)
        )
    else:
        ordered = ordered.with_columns(pl.lit(0, dtype=pl.UInt64).alias(group_id))
    output_schema = {group_id: pl.UInt64, alias: pl.String if rational else pl.Float64}
    if presence_alias:
        output_schema[presence_alias] = pl.UInt64
    if missing_alias:
        output_schema[missing_alias] = pl.UInt64
    frames: list[pl.DataFrame] = []
    retained_output_bytes = 0
    rows: list[tuple] = []
    current = None
    state = _State()
    started = False

    def flush() -> None:
        nonlocal retained_output_bytes
        if not rows:
            return
        frame = pl.DataFrame(rows, schema=output_schema, orient="row")
        # These new primitive columns own their buffers; cumulative admission
        # avoids collecting every group before checking the configured target.
        retained_output_bytes += retained_size(frame)
        ctx.check_retained_bytes("exact aggregate output", retained_output_bytes)
        frames.append(frame)
        rows.clear()

    def finish() -> None:
        result = [0 if current is None else current, state.result(how, rational)]
        if presence_alias:
            result.append(state.rows)
        if missing_alias:
            result.append(state.missing)
        rows.append(tuple(result))
        if len(rows) >= batch_rows:
            flush()

    transfer = list(dict.fromkeys([group_id, value, *([weight] if weight is not None else [])]))
    indices = {name: index for index, name in enumerate(transfer)}
    with observation.derived(lambda _: ordered) as ordered_observation:
        for batch in ordered_observation.batches(batch_rows):
            for row in batch.select(transfer).iter_rows():
                key = row[indices[group_id]]
                if started and key != current:
                    finish()
                    state = _State()
                current = key
                started = True
                state.add(
                    row[indices[value]],
                    row[indices[weight]] if weight is not None else None,
                    weighted=how == "weighted_mean",
                    nonfinite_missing=nonfinite_missing,
                )
        if started or not keys:
            finish()
        flush()
        result = pl.concat(frames) if frames else pl.DataFrame(schema=output_schema)
        if keys:
            if result.is_empty():
                # Preserve key dtypes without a native join over an empty rank map.
                return pl.DataFrame(
                    schema={
                        **{key: schema[key] for key in keys},
                        **{
                            name: dtype for name, dtype in output_schema.items() if name != group_id
                        },
                    }
                )
            representatives = (
                ordered_observation.scan()
                .select([group_id, *keys])
                .unique(subset=group_id, maintain_order=True)
            )
            return ordered_observation.collect(
                representatives.join(result.lazy(), on=group_id, how="left")
                .sort(group_id)
                .drop(group_id)
            )
        result = result.drop(group_id)
        ctx.check_materialization("exact grouped result", result)
        return result
