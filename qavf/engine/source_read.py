"""One bound ad-hoc source, sharing its parent run's policy and resource owner."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.projection import selected_columns
from qavf.engine.sql_read import consume_native
from qavf.sources.base import DataSource, SourceError, SourceRegistry
from qavf.sources.sql import quote_identifier

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation
    from qavf.engine.sqlite_store import Store


class SourceReadView:
    """Schema/projection/provenance of one bound source, valid inside its callback."""

    def __init__(self, ctx: DataContext, source: DataSource) -> None:
        self._ctx = ctx
        self._source = source
        self._plan: pl.LazyFrame | None = None
        self._frame: pl.DataFrame | None = None
        self._kind: str | None = None
        self._closed = False
        self._alias = "__qavf_adhoc_source"
        while self._alias.lower() in {name.lower() for name in ctx.declared()}:
            self._alias += "_"

    def _ready(self) -> str:
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        if self._kind is None:
            if self._source.snapshot_connection() is None:
                self._plan = self._source.scan_polars()
            if self._plan is not None:
                budget = self._source.params.row_budget
                if budget is not None:
                    count = (
                        self._plan.limit(budget + 1)
                        .select(pl.len())
                        .collect(engine="streaming")
                        .item()
                    )
                    if count > budget:
                        raise SourceError(
                            f"ad-hoc source has more than {budget:,} rows (row_budget)"
                        )
                self._kind = "lazy"
            elif self._source.snapshot_connection() is None and self._source.scan() is not None:
                self._kind = "native"
            else:
                self._kind = "eager"
                self._ctx._resources.eager_fallbacks.add("<ad-hoc>")
                self._frame = self._ctx._load_with_retries("<ad-hoc>", self._source)
        return self._kind

    def schema(self) -> dict[str, pl.DataType]:
        kind = self._ready()
        if kind == "lazy":
            assert self._plan is not None
            return dict(self._plan.collect_schema())
        if kind == "native":
            return consume_native(
                self._ctx,
                lambda: (
                    lambda session: session.schema(f"SELECT * FROM {quote_identifier(self._alias)}")
                ),
                sources={self._alias: self._source},
            )
        assert self._frame is not None
        return dict(self._frame.schema)

    def project(self, columns: Iterable[str]) -> pl.DataFrame:
        selected = selected_columns(columns)
        kind = self._ready()
        if kind == "lazy":
            assert self._plan is not None
            result = self._plan.select(selected).collect(engine="streaming")
        elif kind == "native":
            projection = ", ".join(quote_identifier(column) for column in selected)
            result = consume_native(
                self._ctx,
                lambda: (
                    lambda session: session.frame(
                        f"SELECT {projection} FROM {quote_identifier(self._alias)}"
                    )
                ),
                sources={self._alias: self._source},
            )
        else:
            assert self._frame is not None
            result = self._frame.select(selected)
        self._ctx._check_materialization("ad-hoc projection", result)
        self._ctx._check_cancelled()
        return result

    def fingerprint(self, *, hash_contents: bool = True) -> dict[str, Any] | None:
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        return self._source.fingerprint(hash_contents=hash_contents)

    def observe(
        self, columns: Iterable[str] | None = None
    ) -> contextlib.AbstractContextManager[Observation]:
        """Own an exact projection of this same bound source for repeated passes."""
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        from qavf.engine.observation import observe_source

        return observe_source(self._ctx, self._source, columns)

    def work_store(self, label: str) -> Store:
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        return self._ctx.work_store(label)

    def close(self) -> None:
        self._closed = True
        self._plan = None
        self._frame = None


def read_source[T](
    ctx: DataContext, config: Mapping[str, Any], consumer: Callable[[SourceReadView], T]
) -> T:
    registry = ctx._source_registry or SourceRegistry().discover()
    source = registry.build(config, ctx.env)

    def attempt() -> T:
        # Ad-hoc baselines use content fingerprints; establish that identity before
        # schema/projection reads, on the exact source instance used for provenance.
        source.prepare_identity(hash_contents=True)
        view = SourceReadView(ctx, source)
        try:
            return consumer(view)
        finally:
            view.close()

    return ctx._operations.run([source], attempt)
