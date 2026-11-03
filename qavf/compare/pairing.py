"""Pairing many datasets for checks that relate two: strategies and group plans.

A check relating datasets (``compare``, ``totals_match``, ``key_sets_match``,
``drift``) takes ``datasets: [a, b, c, ...]`` and a ``strategy`` that pairs them;
the two-sided engine then runs once per pair, unchanged:

* ``reference`` (default): the reference member (``reference:``, default the
  first) with each other member;
* ``sequential``: each member with the next, in list order;
* ``all_pairs``: every pair, refused above ``max_pairs``;
* ``consensus``: each member with a consensus built from all members
  (:mod:`qavf.compare.consensus`).

A :class:`GroupPlan` travels with the rule's parameters and tells which pair (by
label) concerns which members, so each finding can say which member it is about
(:class:`qavf.reports.models.GroupRole`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Any, Literal

from pydantic import Field, model_validator

from qavf.engine.severity import Severity
from qavf.parameters import RuleParams
from qavf.reports.models import GroupRole

Strategy = Literal["reference", "sequential", "all_pairs", "consensus"]
STRATEGIES: tuple[str, ...] = ("reference", "sequential", "all_pairs", "consensus")
DEFAULT_MAX_PAIRS = 28
"""``all_pairs`` refuses more pairs than this by default (8 datasets)."""
CONSENSUS = "consensus"
"""How the consensus member is named in pair labels and reports."""
POOLED = "pooled"
"""How the pooled members are named in pair labels and reports."""


class ConsensusSettings(RuleParams):
    """How the consensus of the members is built, key by key (``consensus:``)."""

    numeric: Literal["median", "mean", "trimmed_mean"] = "median"
    """Consensus of a number: the median, the mean or the trimmed mean of the members."""
    trim: float = Field(default=0.1, ge=0, lt=0.5)
    """Fraction trimmed from each end for ``trimmed_mean``."""
    categorical: Literal["majority", "plurality"] = "majority"
    """Consensus of any other value: held by more than half of the members
    (``majority``), or by more members than any other value (``plurality``)."""
    min_agreement: float = Field(default=0.5, gt=0, le=1)
    """Share of the members holding a key that must agree for a consensus value; for
    numbers, the share that reports one."""
    ties: Literal["report", "reference"] = "report"
    """A tie leaves the key without a consensus value (``report``: not assessed for
    that key), or takes the reference member's value (``reference``)."""


class PairRole(RuleParams):
    """One pair of a group: its label in findings and the members it concerns."""

    label: str
    """The pair's name, the label (prefix) of its findings."""
    reference: str
    """The reference side, as reports show it (a member, ``consensus`` or ``pooled``)."""
    other: str
    """The other side, as reports show it."""
    members: tuple[str, ...]
    """The members whose results these are (both, for ``all_pairs``)."""
    level: str | None = None
    """For a whole-group level run as pairs (``distribution`` against the pooled
    members): that level's name."""


class GroupPlan(RuleParams):
    """How one check relates its members: pairs, consensus and whole-group levels."""

    name: str
    """The check item's name."""
    check: str
    """The check's name (``compare``, ``totals_match``, ...)."""
    strategy: Strategy
    """How the members are paired."""
    members: tuple[str, ...] = Field(min_length=2)
    """Every member, in the check's order."""
    reference: str | None = None
    """The reference member (``reference`` and ``consensus`` strategies)."""
    pairs: tuple[PairRole, ...] = ()
    """The pairs, in the order of the rule's list entries."""
    consensus: ConsensusSettings | None = None
    """How the consensus is built (strategy ``consensus``)."""
    consensus_dataset: str | None = None
    """The derived dataset holding the consensus and, per key, why none was found."""
    keys: tuple[str, ...] = ()
    """Columns identifying a record on every member (whole-group levels, consensus)."""
    coverage: str | None = None
    """Name of the whole-group ``coverage`` level (``compare`` with more than two members)."""
    coverage_severity: Severity | None = None
    """Severity of the whole-group ``coverage`` level's failures."""
    structure: str | None = None
    """Name of the whole-group ``structure`` level."""
    structure_severity: Severity | None = None
    """Severity of the whole-group ``structure`` level's failures."""
    where: str | None = None
    """Condition selecting the rows whole-group levels assess."""
    ignore_columns: tuple[str, ...] = ()
    """Columns whole-group structure ignores."""

    @model_validator(mode="after")
    def _known_reference(self) -> GroupPlan:
        if self.reference is not None and self.reference not in self.members:
            raise ValueError("the reference must be one of the datasets")
        return self

    def whole_group_levels(self) -> tuple[str, ...]:
        """Levels assessed over all members at once rather than pair by pair."""
        levels = [name for name in (self.coverage, self.structure) if name is not None]
        levels.extend(pair.level for pair in self.pairs if pair.level is not None)
        return tuple(dict.fromkeys(levels))

    def whole_role(self, level: str | None = None) -> GroupRole:
        """The role of a result about every member (a level assessed over the whole
        group, or the check failing as a whole)."""
        return GroupRole(
            group=self.name,
            check=self.check,
            strategy=self.strategy,
            group_members=list(self.members),
            members=list(self.members),
            level=level,
            whole_group=level is not None,
        )

    def member_role(self, member: str, level: str) -> GroupRole:
        """The role of a whole-group level's result about one member."""
        return GroupRole(
            group=self.name,
            check=self.check,
            strategy=self.strategy,
            group_members=list(self.members),
            members=[member],
            level=level,
            whole_group=True,
        )

    def pair_role(self, pair: PairRole | None, level: str | None) -> GroupRole:
        """The role of a pair's result (``pair=None``: a result of no pair)."""
        if pair is None:
            return GroupRole(group=self.name, check=self.check, strategy=self.strategy, level=level)
        return GroupRole(
            group=self.name,
            check=self.check,
            strategy=self.strategy,
            group_members=list(self.members),
            members=list(pair.members),
            pair=[pair.reference, pair.other],
            level=pair.level or level,
            whole_group=pair.level is not None,
        )

    def role_for(self, check: str | None) -> PairRole | None:
        """The pair a finding labelled ``check`` belongs to (longest label first)."""
        if check is None:
            return None
        for pair in sorted(self.pairs, key=_label_length, reverse=True):
            if check == pair.label or check.startswith(f"{pair.label}: "):
                return pair
        return None


def _label_length(pair: PairRole) -> int:
    return len(pair.label)


class PairingError(ValueError):
    """Datasets or a strategy that cannot be paired; ``field`` names the setting."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


@dataclass(frozen=True, slots=True)
class DatasetGroup:
    """The members of a check relating datasets, and how they pair."""

    members: tuple[str, ...]
    strategy: str = "reference"
    reference: str | None = None
    max_pairs: int = DEFAULT_MAX_PAIRS
    consensus: ConsensusSettings | None = None

    @classmethod
    def of(cls, settings: Mapping[str, Any]) -> DatasetGroup:
        """The group a check item's settings describe.

        Raises:
            PairingError: for missing, repeated or too many datasets, an unknown
                strategy or a setting that does not apply to it.
        """
        datasets = settings.get("datasets")
        if not isinstance(datasets, list | tuple) or not all(isinstance(n, str) for n in datasets):
            raise PairingError("datasets", "give the datasets to relate as a list of names")
        if len(datasets) < 2:
            raise PairingError("datasets", "give at least two datasets")
        if len(set(datasets)) != len(datasets):
            repeated = sorted({name for name in datasets if datasets.count(name) > 1})
            raise PairingError("datasets", f"datasets repeat {', '.join(repeated)}")
        strategy = settings.get("strategy", "reference")
        if strategy not in STRATEGIES:
            raise PairingError(
                "strategy", f"unknown strategy {strategy!r}; use one of {', '.join(STRATEGIES)}"
            )
        reference = settings.get("reference")
        if reference is not None:
            if strategy not in ("reference", CONSENSUS):
                raise PairingError(
                    "reference", f"'reference' does not apply to strategy {strategy}"
                )
            if reference not in datasets:
                raise PairingError("reference", "the reference must be one of the datasets")
        max_pairs = settings.get("max_pairs", DEFAULT_MAX_PAIRS)
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
            raise PairingError("max_pairs", "max_pairs must be a positive whole number")
        if strategy != "all_pairs" and "max_pairs" in settings:
            raise PairingError("max_pairs", "max_pairs applies to strategy all_pairs")
        consensus = settings.get("consensus")
        if consensus is not None and strategy != CONSENSUS:
            raise PairingError("consensus", "consensus settings apply to strategy consensus")
        group = cls(
            members=tuple(datasets),
            strategy=strategy,
            reference=reference if reference is not None else datasets[0],
            max_pairs=max_pairs,
            consensus=_consensus(consensus) if strategy == CONSENSUS else None,
        )
        if strategy == "all_pairs" and len(group.pairs()) > max_pairs:
            raise PairingError(
                "datasets",
                f"all_pairs relates {len(datasets)} datasets in {len(group.pairs())} pairs, more "
                f"than max_pairs ({max_pairs}); raise max_pairs or use strategy reference, "
                "sequential or consensus",
            )
        return group

    @property
    def plain(self) -> bool:
        """Two members with the default strategy: today's single pair, exactly."""
        return len(self.members) == 2 and self.strategy == "reference"

    def pairs(self) -> list[tuple[str, str]]:
        """``(reference, other)`` member pairs (not for ``consensus``)."""
        if self.strategy == "sequential":
            return list(pairwise(self.members))
        if self.strategy == "all_pairs":
            return list(combinations(self.members, 2))
        if self.strategy == CONSENSUS:
            return []
        reference = self.reference or self.members[0]
        return [(reference, member) for member in self.members if member != reference]

    def concerned(self, reference: str, other: str) -> tuple[str, ...]:
        """The members whose results a pair's findings are."""
        return (reference, other) if self.strategy == "all_pairs" else (other,)


def _consensus(value: Any) -> ConsensusSettings:
    if value is None:
        return ConsensusSettings()
    if isinstance(value, ConsensusSettings):
        return value
    if not isinstance(value, Mapping):
        raise PairingError("consensus", "give the consensus settings as a mapping")
    try:
        return ConsensusSettings.model_validate(dict(value))
    except ValueError as exc:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            detail = "; ".join(
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors()
            )
            raise PairingError("consensus", detail) from None
        raise PairingError("consensus", str(exc)) from None


def pair_label(name: str, reference: str, other: str) -> str:
    """A pair's label within its check: ``<check name>: <reference> vs <other>``."""
    return f"{name}: {reference} vs {other}"


def member_names(roles: Sequence[PairRole]) -> list[str]:
    """Every member the pairs concern, in order of appearance."""
    return list(dict.fromkeys(member for role in roles for member in role.members))


__all__ = [
    "CONSENSUS",
    "DEFAULT_MAX_PAIRS",
    "POOLED",
    "STRATEGIES",
    "ConsensusSettings",
    "DatasetGroup",
    "GroupPlan",
    "PairRole",
    "PairingError",
    "Strategy",
    "member_names",
    "pair_label",
]
