"""Bounded baseline staging and exact round-trip validation before publication."""

from __future__ import annotations

import gzip
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.engine.source_read import SourceReadView
from qavf.engine.spill import SpillFile
from qavf.inputs.normalize import Configuration, prepare
from qavf.inputs.resources import ResourceLimits
from qavf.output_format import TableOutput, table_output
from qavf.regression._indexed import compare_indexed
from qavf.regression._plan import comparison_plan
from qavf.regression.models import ComparisonResult
from qavf.sources.base import SourceError


@dataclass(frozen=True, slots=True)
class StagedBaseline:
    columns: dict[str, str]
    rows: int


class _Projection:
    """Expose just the comparison plan's shared columns from an owned observation."""

    def __init__(self, observation: Observation, columns: Sequence[str]) -> None:
        self.observation = observation
        self.columns = list(columns)

    def schema(self) -> dict[str, pl.DataType]:
        schema = self.observation.schema()
        return {column: schema[column] for column in self.columns}

    def batches(self, batch_rows: int = 65536) -> Iterator[pl.DataFrame]:
        for frame in self.observation.batches(batch_rows):
            yield frame.select(self.columns)


def _key_summary(observation: Observation, keys: Sequence[str]) -> int:
    actual = observation.schema()
    absent = [key for key in keys if key not in actual]
    if absent:
        raise ValueError(f"baseline lacks key column(s) {', '.join(absent)}")
    missing = []
    for index, key in enumerate(keys):
        mask = pl.col(key).is_null()
        if actual[key].is_float():
            mask = mask | pl.col(key).is_nan()
        missing.append(mask.sum().alias(f"missing_{index}"))
    summary = observation.reduce(lambda plan: plan.select(pl.len().alias("rows"), *missing)).row(0)
    for key, count in zip(keys, summary[1:], strict=True):
        if count:
            raise ValueError(f"baseline key column {key!r} has {count} missing value(s)")
    return int(summary[0])


def _write(
    observation: Observation, keys: Sequence[str], stream: BinaryIO, output: TableOutput
) -> None:
    sorted_plan = observation.scan().sort(list(keys))
    if output.format == "csv":
        sorted_plan.sink_csv(
            stream, separator=output.separator, maintain_order=True, engine="streaming"
        )
    elif output.format == "parquet":
        sorted_plan.sink_parquet(stream, maintain_order=True, engine="streaming")
    elif output.format == "ndjson":
        sorted_plan.sink_ndjson(stream, maintain_order=True, engine="streaming")
    else:
        # Polars has no lazy JSON-array sink. Preserve exact native JSON encoding
        # batchwise, joining array interiors without retaining previous batches.
        stream.write(b"[")
        first = True
        for batch in observation.batches(build=lambda plan: plan.sort(list(keys))):
            encoded = batch.write_json().encode("utf-8")[1:-1]
            if not encoded:
                continue
            if not first:
                stream.write(b",")
            stream.write(encoded)
            first = False
        stream.write(b"]")


def _roundtrip(
    context: DataContext, original: Observation, staged: Path, keys: Sequence[str]
) -> None:
    def compare(view: SourceReadView) -> ComparisonResult:
        with view.observe() as restored:
            _key_summary(restored, keys)
            plan = comparison_plan(
                original.schema(),
                restored.schema(),
                keys,
                rtol=0,
                atol=0,
                max_differences=0,
                name="baseline serialization",
                schema="compatible",
            )
            with context.work_store("baseline-roundtrip") as store:
                return compare_indexed(
                    store,
                    _Projection(original, plan.shared),
                    _Projection(restored, plan.shared),
                    plan,
                )

    try:
        result = context.read_source({"type": "file", "path": str(staged)}, compare)
    except ValueError as exc:
        if "duplicate key" in str(exc):
            raise ValueError("baseline has rows with a duplicated key") from exc
        raise ValueError(f"baseline format does not preserve the accepted values: {exc}") from exc
    if not result.equal:
        raise ValueError(
            "baseline format does not preserve the accepted values: " + result.summary()
        )


@contextmanager
def stage_baseline(
    data: object,
    staged: Path,
    keys: Sequence[str],
    *,
    resources: ResourceLimits | None = None,
    config: Configuration = None,
) -> Iterator[StagedBaseline]:
    """Keep the source snapshot and capped payload alive until the caller commits."""
    output = table_output(staged)
    settings, frames = prepare(
        data, schema=None, config={} if config is None else config, resources=resources
    )
    if len(settings.datasets) != 1:
        raise ValueError("baseline acceptance requires exactly one dataset")
    if settings.engine.execution_mode == "isolated":
        raise ValueError(
            "isolated baseline acceptance is not supported; "
            "execution limits apply to validation/comparison"
        )
    name = next(iter(settings.datasets))
    with (
        DataContext.from_settings(
            settings, frames, threads=settings.engine.native_threads, execution_workers=1
        ) as context,
        context.observe(name) as observation,
    ):
        rows = _key_summary(observation, keys)
        with SpillFile(staged, context._resources.spill_budget) as payload:
            with payload.writer() as raw:
                if output.compressed:
                    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                        _write(observation, keys, cast(BinaryIO, compressed), output)
                else:
                    _write(observation, keys, raw, output)
                raw.flush()
                with staged.open("rb+") as flushed:
                    os.fsync(flushed.fileno())
            _roundtrip(context, observation, staged, keys)
            if context.verify_inputs():
                raise SourceError("input identity changed during baseline acceptance")
            yield StagedBaseline(
                {name: str(dtype) for name, dtype in observation.schema().items()}, rows
            )
