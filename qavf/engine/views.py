"""Dataset views: named slices of a dataset, loaded from the dataset itself.

A view (``datasets.counts.views.recent: {where: year >= 2020}``) is used as
``counts.recent`` wherever a check names a dataset. The engine gives each view
a :class:`ViewSource` that derives its rows from the run's own copy of the
parent dataset, so a view sees exactly the data its parent was checked on
(including a frame supplied in memory), after the parent's renames. Rules need
no knowledge of views: to them a view is one more dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import polars as pl

from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams

if TYPE_CHECKING:
    from qavf.settings import ViewSpec


class FrameProvider(Protocol):
    """Where a view reads its parent dataset (the run's data context)."""

    def get(self, name: str) -> pl.DataFrame: ...


class ViewParams(SourceParams):
    """Parameters of :class:`ViewSource`."""

    type: Literal["view"] = "view"
    dataset: str
    """The dataset the view slices."""
    where: str | None = None
    """SQL condition selecting the view's rows."""
    columns: list[str] | None = None
    """Columns the view keeps (all when omitted)."""


class ViewSource(DataSource[ViewParams]):
    """The rows of a view, filtered from its parent dataset as the run loaded it."""

    kind: ClassVar[str] = "view"
    Params: ClassVar[type[SourceParams]] = ViewParams
    provider: FrameProvider | None = None

    @classmethod
    def of(
        cls, parent: str, spec: ViewSpec, env: SourceEnvironment, provider: FrameProvider
    ) -> ViewSource:
        source = cls(ViewParams(dataset=parent, where=spec.where, columns=spec.columns), env)
        source.provider = provider
        return source

    def bind(self, env: SourceEnvironment) -> ViewSource:
        bound = super().bind(env)
        assert isinstance(bound, ViewSource)
        bound.provider = self.provider
        return bound

    def load(self) -> pl.DataFrame:
        if self.provider is None:
            raise SourceError(f"view of {self.params.dataset!r} is not attached to a run")
        frame = self.provider.get(self.params.dataset)
        try:
            if self.params.where is not None:
                frame = frame.filter(pl.sql_expr(self.params.where))
            if self.params.columns is not None:
                frame = frame.select(self.params.columns)
        except (pl.exceptions.PolarsError, ValueError) as exc:
            raise SourceError(f"view of {self.params.dataset!r}: {exc}") from None
        return frame

    def describe(self) -> str:
        where = f" where {self.params.where}" if self.params.where else ""
        return f"view of {self.params.dataset}{where}"


__all__ = ["FrameProvider", "ViewParams", "ViewSource"]
