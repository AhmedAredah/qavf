"""One bound ad-hoc source, sharing its parent run's policy and resource owner."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from qavf.engine.projection import selected_columns
from qavf.sources.base import DataSource

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation
    from qavf.engine.sqlite_store import Store


@dataclass(frozen=True, slots=True)
class SourceReadServices:
    """Explicit read capabilities; a view does not depend on DataContext internals."""

    check_cancelled: Callable[[], None]
    declared: Callable[[], list[str]]
    check_materialization: Callable[[str, pl.DataFrame], None]
    batch_observation: Callable[[DataSource], Observation]
    lazy_observation: Callable[[pl.LazyFrame, int], Observation]
    eager_frame: Callable[[DataSource], pl.DataFrame]
    native_schema: Callable[[DataSource, str], dict[str, pl.DataType]]
    native_frame: Callable[[DataSource, str, list[str]], pl.DataFrame]
    observation: Callable[
        [DataSource, Iterable[str] | None], contextlib.AbstractContextManager[Observation]
    ]
    work_store: Callable[[str], Store]


class SourceReadView:
    """Schema/projection/provenance of one bound source, valid inside its callback."""

    def __init__(self, services: SourceReadServices, source: DataSource) -> None:
        self._services = services
        self._source = source
        self._plan: pl.LazyFrame | None = None
        self._frame: pl.DataFrame | None = None
        self._kind: str | None = None
        self._closed = False
        self._budget_observation: Observation | None = None
        self._alias = "__qavf_adhoc_source"
        while self._alias.lower() in {name.lower() for name in services.declared()}:
            self._alias += "_"

    def _ready(self) -> str:
        self._services.check_cancelled()
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        if self._kind is None:
            if self._source.batch_schema() is not None:
                self._budget_observation = self._services.batch_observation(self._source)
                self._plan = self._budget_observation.scan()
                self._kind = "lazy"
                return self._kind
            if self._source.snapshot_connection() is None:
                self._plan = self._source.scan_polars()
            if self._plan is not None:
                budget = self._source.params.row_budget
                if budget is not None:
                    self._budget_observation = self._services.lazy_observation(self._plan, budget)
                    self._plan = self._budget_observation.scan()
                self._kind = "lazy"
            elif self._source.snapshot_connection() is None and self._source.scan() is not None:
                self._kind = "native"
            else:
                self._kind = "eager"
                self._frame = self._services.eager_frame(self._source)
        return self._kind

    def schema(self) -> dict[str, pl.DataType]:
        kind = self._ready()
        if kind == "lazy":
            assert self._plan is not None
            return dict(self._plan.collect_schema())
        if kind == "native":
            return self._services.native_schema(self._source, self._alias)
        assert self._frame is not None
        return dict(self._frame.schema)

    def project(self, columns: Iterable[str]) -> pl.DataFrame:
        selected = selected_columns(columns)
        kind = self._ready()
        if kind == "lazy":
            assert self._plan is not None
            result = self._plan.select(selected).collect(engine="streaming")
        elif kind == "native":
            result = self._services.native_frame(self._source, self._alias, selected)
        else:
            assert self._frame is not None
            result = self._frame.select(selected)
        self._services.check_materialization("ad-hoc projection", result)
        self._services.check_cancelled()
        return result

    def fingerprint(self, *, hash_contents: bool = True) -> dict[str, Any] | None:
        self._services.check_cancelled()
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        return self._source.fingerprint(hash_contents=hash_contents)

    def observe(
        self, columns: Iterable[str] | None = None
    ) -> contextlib.AbstractContextManager[Observation]:
        """Own an exact projection of this same bound source for repeated passes."""
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        if self._budget_observation is not None:
            return self._budget_observation.borrow(columns)
        return self._services.observation(self._source, columns)

    def work_store(self, label: str) -> Store:
        if self._closed:
            raise RuntimeError("ad-hoc source read has ended")
        return self._services.work_store(label)

    def close(self) -> None:
        self._closed = True
        self._plan = None
        self._frame = None
        if self._budget_observation is not None:
            self._budget_observation.close()
            self._budget_observation = None


def read_source[T](
    ctx: DataContext, config: Mapping[str, Any], consumer: Callable[[SourceReadView], T]
) -> T:
    source = ctx.build_source(config)

    def attempt() -> T:
        # Ad-hoc baselines use content fingerprints; establish that identity before
        # schema/projection reads, on the exact source instance used for provenance.
        source.prepare_identity(hash_contents=True)
        view = SourceReadView(ctx.source_read_services(), source)
        try:
            return consumer(view)
        finally:
            view.close()

    return ctx.run_source_operation([source], attempt)
