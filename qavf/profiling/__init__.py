"""Exact draft profiles over common inputs without collecting complete native tables."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.sql_read import SqlReadSession
from qavf.inputs.normalize import Configuration, prepare
from qavf.inputs.resources import ResourceLimits
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError
from qavf.sources.formats import infer_format
from qavf.sources.sql import quote_identifier, sql_literal

from .accumulator import ProfileAccumulator, ProfileResult
from .values import validate_margin

if TYPE_CHECKING:
    from qavf._input_types import ValidationInput


def _options(
    margin: float,
    max_categories: int,
    bounds: bool,
    max_category_bytes: int,
    max_category_value_bytes: int,
) -> float:
    margin = validate_margin(margin)
    if type(max_categories) is not int or max_categories < 0:
        raise ValueError("max_categories must be a non-negative integer")
    for name, value in (
        ("max_category_bytes", max_category_bytes),
        ("max_category_value_bytes", max_category_value_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(bounds, bool):
        raise ValueError("bounds must be a boolean")
    return margin


def profile_frame(
    frame: pl.DataFrame,
    *,
    margin: float = 0.0,
    max_categories: int = 20,
    bounds: bool = True,
    max_category_bytes: int = 1 << 20,
    max_category_value_bytes: int = 64 << 10,
) -> dict[str, Any]:
    """Infer a draft contract from a caller-owned frame using bounded profile state.

    Numeric bounds retain exact integer/Decimal values. Observed infinite sides
    remain unbounded; NaN/null are missing. Datetime bounds round outwards only
    when required by configuration datetime precision. Review inferred limits
    before relying on them: profiling describes observations, including defects.
    """
    margin = _options(margin, max_categories, bounds, max_category_bytes, max_category_value_bytes)
    accumulator = ProfileAccumulator(
        dict(frame.schema),
        max_categories=max_categories,
        bounds=bounds,
        max_category_bytes=max_category_bytes,
        max_category_value_bytes=max_category_value_bytes,
    )
    for batch in frame.iter_slices(65536):
        accumulator.update(batch)
    return accumulator.finish(margin).schema


def profile_input(
    data: ValidationInput,
    *,
    margin: float = 0.0,
    max_categories: int = 20,
    bounds: bool = True,
    max_category_bytes: int = 1 << 20,
    max_category_value_bytes: int = 64 << 10,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
) -> ProfileResult:
    """Profile one path, File, frame, Query, Loader or Dataset through shared inputs.

    Native sources stream bounded batches in one complete source operation.
    Typed/lazy inputs use an exact observation under the run's memory/spill
    policy. Retained profile state scales with columns and category limits,
    never with input row count. Category payloads share ``max_category_bytes``
    across all columns, with ``max_category_value_bytes`` per value. Oversized
    enumerations fall back to exact lengths/missingness with disclosure.
    Source backends may require native working
    storage; caller-owned frames/opaque loaders retain their ownership costs.
    Isolated execution is currently supported by validation/comparison only.
    """
    margin = _options(margin, max_categories, bounds, max_category_bytes, max_category_value_bytes)
    settings, frames = prepare(
        data, schema=None, config={} if config is None else config, resources=resources
    )
    if len(settings.datasets) != 1:
        raise ValueError("profiling requires exactly one dataset")
    if settings.engine.execution_mode == "isolated":
        raise ValueError(
            "isolated profiling is not supported; use validation/comparison for execution limits"
        )
    name = next(iter(settings.datasets))
    codes = _code_columns(settings, frames, name)
    if codes:
        settings = _with_text_columns(settings, name, codes)
    with DataContext.from_settings(
        settings, frames, threads=settings.engine.native_threads, execution_workers=1
    ) as context:
        actual = context.schema(name)

        def fresh() -> ProfileAccumulator:
            return ProfileAccumulator(
                actual,
                max_categories=max_categories,
                bounds=bounds,
                max_category_bytes=max_category_bytes,
                max_category_value_bytes=max_category_value_bytes,
            )

        if context.execution_kind(name) == "native_sql":

            def consumer_factory() -> Callable[[SqlReadSession], ProfileAccumulator]:
                def consume(session: SqlReadSession) -> ProfileAccumulator:
                    accumulator = fresh()
                    for batch in session.batches(f"SELECT * FROM {quote_identifier(name)}"):
                        accumulator.update(batch)
                    return accumulator

                return consume

            accumulator = context.consume_sql(consumer_factory, datasets=(name,))
        else:
            accumulator = fresh()
            with context.observe(name) as observation:
                for batch in observation.batches():
                    accumulator.update(batch)
        if context.verify_inputs():
            raise SourceError("input identity changed during profiling")
        return accumulator.finish(margin)


def _code_columns(settings: QAVFSettings, frames: Any, name: str) -> list[str]:
    """Integer CSV columns with values written with leading zeros, such as ``007``.

    Those values are codes, not numbers: a sniffed integer type would drop the
    zeros and draft numeric limits around the damaged values.
    """
    source = settings.datasets[name].source_config()
    if source is None or source.get("type") != "file" or name in frames:
        return []
    try:
        fmt = source.get("format") or infer_format(str(source["path"]))
    except ValueError:
        return []
    if fmt != "csv":
        return []
    threads = settings.engine.native_threads
    with DataContext.from_settings(settings, frames, threads=threads, execution_workers=1) as ctx:
        integers = [column for column, dtype in ctx.schema(name).items() if dtype.is_integer()]
    if not integers:
        return []
    probe = _with_text_columns(settings, name, integers)
    q = quote_identifier
    leading_zero = sql_literal("[+-]?0[0-9]+")
    with DataContext.from_settings(probe, frames, threads=threads, execution_workers=1) as ctx:
        found = ctx.sql(
            "SELECT "
            + ", ".join(
                f"coalesce(bool_or(regexp_full_match(trim({q(c)}), {leading_zero})), false)"
                f" AS {q(c)}"
                for c in integers
            )
            + f" FROM {q(name)}",
            datasets=(name,),
        )
    return [column for column in integers if found[column][0]]


def _with_text_columns(settings: QAVFSettings, name: str, columns: list[str]) -> QAVFSettings:
    spec = settings.datasets[name]
    source = dict(spec.source_config() or {})
    source["text_columns"] = list(dict.fromkeys([*source.get("text_columns", []), *columns]))
    datasets = dict(settings.datasets)
    datasets[name] = spec.model_copy(update={"path": None, "format": None, "source": source})
    return settings.model_copy(update={"datasets": datasets})


__all__ = ["ProfileResult", "profile_frame", "profile_input"]
