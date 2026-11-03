"""Datasets derived from several members of a check: consensus and pooled data.

A check relating many datasets with ``strategy: consensus`` compares each member
with a consensus member; ``compare`` with more than two members compares each
member's distribution with the pooled data of all of them. Like views
(:mod:`qavf.engine.views`), these are datasets of the run's data context, derived
from the members as the run loaded them, so the two-sided comparison engine reads
them like any other dataset:

* ``consensus_full``: the consensus of every key, with why a key has none
  (:data:`~qavf.compare.consensus.NO_CONSENSUS`);
* ``consensus``: the keys that have a consensus, compared with each member;
* ``member``: a member without the keys that have no consensus (not assessed);
* ``pooled``: every member's rows together.

The derived datasets of a configuration are named after the check item's place
(``consensus~checks[2]``), so they never clash with configured datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import polars as pl

from qavf.compare.consensus import NO_CONSENSUS, ConsensusBuilder, ConsensusError, pooled
from qavf.compare.pairing import ConsensusSettings
from qavf.engine.derived_names import derived_name
from qavf.sources.base import DataSource, SourceEnvironment, SourceError, SourceParams

if TYPE_CHECKING:
    from qavf.engine.views import FrameProvider

DerivedRole = Literal["consensus_full", "consensus", "member", "pooled"]


class DerivedParams(SourceParams):
    """Parameters of :class:`DerivedSource`."""

    type: Literal["derived"] = "derived"
    role: DerivedRole
    members: tuple[str, ...] = ()
    """The members the dataset is derived from, in the check's order."""
    keys: tuple[str, ...] = ()
    """Columns matching records across members (consensus)."""
    columns: tuple[str, ...] | None = None
    """Columns given a consensus value (all common non-key columns when omitted)."""
    ignore: tuple[str, ...] = ()
    """Columns left out of the consensus."""
    total: Literal["none", "sum", "rows"] = "none"
    """Combine each member's totals per key (``sum`` of ``value``, or ``rows``) rather
    than its rows (``totals_match``)."""
    value: str | None = None
    consensus: ConsensusSettings = ConsensusSettings()
    reference: str | None = None
    """The member whose value breaks a tie under ``ties: reference``."""
    full: str | None = None
    """The ``consensus_full`` dataset a ``consensus`` or ``member`` dataset reads."""
    member: str | None = None
    """The member a ``member`` dataset restricts."""

    def parents(self) -> tuple[str, ...]:
        """The datasets this one is derived from (kept loaded while it is used)."""
        if self.role == "consensus":
            return (self.full,) if self.full else ()
        if self.role == "member":
            return tuple(name for name in (self.member, self.full) if name)
        return self.members


class DerivedSource(DataSource[DerivedParams]):
    """The rows of a derived dataset, built from its members as the run loaded them."""

    kind: ClassVar[str] = "derived"
    Params: ClassVar[type[SourceParams]] = DerivedParams
    provider: FrameProvider | None = None

    @classmethod
    def of(
        cls, params: DerivedParams, env: SourceEnvironment, provider: FrameProvider
    ) -> DerivedSource:
        source = cls(params, env)
        source.provider = provider
        return source

    def bind(self, env: SourceEnvironment) -> DerivedSource:
        bound = super().bind(env)
        assert isinstance(bound, DerivedSource)
        bound.provider = self.provider
        return bound

    def load(self) -> pl.DataFrame:
        if self.provider is None:
            raise SourceError(f"{self.describe()} is not attached to a run")
        try:
            return self._derive(self.provider)
        except (ConsensusError, pl.exceptions.PolarsError) as exc:
            raise SourceError(f"{self.describe()}: {exc}") from None

    def _derive(self, provider: FrameProvider) -> pl.DataFrame:
        p = self.params
        if p.role == "pooled":
            return pooled([provider.get(member) for member in p.members])
        if p.role == "consensus_full":
            builder = ConsensusBuilder(
                p.consensus,
                p.keys,
                p.members,
                reference=p.reference,
                columns=p.columns,
                ignore=p.ignore,
            )
            frames = [provider.get(member) for member in p.members]
            if p.total != "none":
                frames = builder.aggregated(frames, p.value if p.total == "sum" else None)
            return builder.build(frames)
        assert p.full is not None
        full = provider.get(p.full)
        if p.role == "consensus":
            return full.filter(pl.col(NO_CONSENSUS).is_null()).drop(NO_CONSENSUS)
        assert p.member is not None
        unassessed = full.filter(pl.col(NO_CONSENSUS).is_not_null()).select(list(p.keys))
        return provider.get(p.member).join(unassessed, on=list(p.keys), how="anti")

    def describe(self) -> str:
        p = self.params
        if p.role == "member":
            return f"{p.member} without the keys that have no consensus"
        if p.role == "pooled":
            return f"pooled rows of {', '.join(p.members)}"
        return f"consensus of {', '.join(p.members)}"


__all__ = [
    "DerivedParams",
    "DerivedRole",
    "DerivedSource",
    "derived_name",
]
