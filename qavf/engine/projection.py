"""Selected-column reads and scoped reductions across context backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import polars as pl

from qavf.sources.base import DataSource
from qavf.sources.sql import quote_identifier

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


def selected_columns(columns: Iterable[str]) -> list[str]:
    """Keep user order while eliminating duplicate selections."""
    selected = [columns] if isinstance(columns, str) else list(dict.fromkeys(columns))
    if not selected or not all(isinstance(column, str) for column in selected):
        raise ValueError("projection requires at least one column name")
    return selected


def repeatable_file_source(source: DataSource | None) -> bool:
    """The built-in local reader has stable values under its verified identity.

    Caller owns the surrounding identity operation. Exact type is intentional:
    subclass overrides can produce volatile values despite a local filename.
    """
    from qavf.sources.file import FileSource
    from qavf.sources.formats import is_remote

    return type(source) is FileSource and not is_remote(source.params.path)


def repeatable_scan(ctx: DataContext, name: str) -> bool:
    ctx._check_cancelled()
    ctx._require_available(name)
    source = ctx._catalog.sources.get(name)
    if not repeatable_file_source(source) or name in ctx._lazy or name in ctx._frames:
        return False
    # FileSource pins its manifest and verifies stat/content identity at every
    # subsequent operation boundary; capability checks must establish it too.
    return ctx._source_operation([name], lambda: True)


def project(ctx: DataContext, name: str, columns: Iterable[str]) -> pl.DataFrame:
    selected = selected_columns(columns)
    ctx._check_cancelled()
    ctx._require_available(name)
    if name in ctx._frames:
        result = ctx._frames.project(name, selected)
        ctx._check_materialization(name, result)
        ctx._check_cancelled()
        return result
    kind = ctx.execution_kind(name)
    if kind == "lazy":
        return ctx.collect(name, ctx.scan(name).select(selected))
    if kind == "native_sql":
        projection = ", ".join(quote_identifier(column) for column in selected)
        return ctx.sql(f"SELECT {projection} FROM {quote_identifier(name)}", datasets=[name])
    return ctx.get(name).select(selected)


def reduce_frame(
    ctx: DataContext, name: str, build: Callable[[pl.LazyFrame], pl.LazyFrame]
) -> pl.DataFrame:
    ctx._check_cancelled()
    ctx._require_available(name)
    if name in ctx._frames:
        result = ctx._frames.reduce(name, build)
        ctx._check_materialization(name, result)
        ctx._check_cancelled()
        return result
    return ctx.collect(name, build(ctx.scan(name)))
