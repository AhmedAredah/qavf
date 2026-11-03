"""Reports of checks relating many datasets: member matrix, pairs and ranking.

Each result of such a check carries a :class:`~qavf.reports.models.GroupRole`
saying which member it is about. :func:`group_reports` rolls those results up,
per check item, into:

* a **matrix**: one row per member, one cell per level (for ``compare``; the
  check itself otherwise), each cell the worst status of that member's results;
* the **pairs**: each pair compared, with its worst status and counts (the
  per-pair detail of a ``compare`` is its comparison in the report's comparison
  section, named by the pair's ``comparison``);
* a **ranking** of the members by failures, then results not assessed, then
  changed records, so the problem source comes first.

The roll-up is derived from the results alone, so both backends and reports
read back from JSON produce the same one, and reports written before groups
existed simply have none.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qavf.engine.severity import Status
from qavf.reports.models import (
    GroupReport,
    GroupRole,
    MemberRank,
    MemberRow,
    PairSummary,
    RuleResult,
)

WORST_FIRST: tuple[Status, ...] = (Status.FAIL, Status.ERROR, Status.PASS, Status.SKIPPED)
"""Statuses from worst to best: a cell shows the first one any of its results has."""
CHANGED_UNITS = frozenset({"rows", "records", "keys"})
"""Units whose failed counts are records that differ (the ranking's ``changed_records``)."""
CHECK_LEVEL = "check"
"""The matrix column of results that belong to no comparison level."""


def worst(statuses: Iterable[Status]) -> Status | None:
    """The worst of ``statuses`` (``None`` for none)."""
    present = set(statuses)
    return next((status for status in WORST_FIRST if status in present), None)


class GroupRollup:
    """Builds the :class:`GroupReport` of one check item from its results."""

    def __init__(self, rule_id: str, role: GroupRole) -> None:
        self.rule_id = rule_id
        self.role = role
        self.results: list[RuleResult] = []

    def add(self, result: RuleResult) -> None:
        self.results.append(result)

    def members(self) -> list[str]:
        named = [member for result in self.results for member in _role(result).members]
        return list(dict.fromkeys([*self.role.group_members, *named]))

    def levels(self) -> list[str]:
        return list(
            dict.fromkeys(_level(result) for result in self.results if _role(result).members)
        )

    def matrix(self, members: Sequence[str]) -> list[MemberRow]:
        rows = []
        for member in members:
            cells: dict[str, list[Status]] = {}
            for result in self.results:
                if member in _role(result).members:
                    cells.setdefault(_level(result), []).append(result.status)
            rows.append(
                MemberRow(
                    member=member,
                    cells={level: s for level, statuses in cells.items() if (s := worst(statuses))},
                )
            )
        return rows

    def pairs(self) -> list[PairSummary]:
        grouped: dict[tuple[str, ...], list[RuleResult]] = {}
        for result in self.results:
            role = _role(result)
            if len(role.pair) == 2:
                grouped.setdefault(tuple(role.pair), []).append(result)
        summaries = []
        for (reference, other), results in grouped.items():
            status = worst(result.status for result in results) or Status.SKIPPED
            first = _role(results[0])
            summaries.append(
                PairSummary(
                    reference=reference,
                    other=other,
                    comparison=_comparison(results[0]),
                    status=status,
                    results=len(results),
                    failed=sum(result.status is Status.FAIL for result in results),
                    not_assessed=sum(result.status is Status.ERROR for result in results),
                    whole_group_level=first.level if first.whole_group else None,
                )
            )
        summaries.sort(key=self.pair_order)
        return summaries

    def pair_order(self, pair: PairSummary) -> tuple[int, int]:
        """Pairs in the members' order (the consensus and the pool first)."""
        members = self.role.group_members
        return (
            members.index(pair.reference) if pair.reference in members else -1,
            members.index(pair.other) if pair.other in members else -1,
        )

    def ranking(self, members: Sequence[str]) -> list[MemberRank]:
        scores = []
        for member in members:
            mine = [result for result in self.results if member in _role(result).members]
            failed = [result for result in mine if result.status is Status.FAIL]
            scores.append(
                (
                    member,
                    worst(result.status for result in mine) or Status.SKIPPED,
                    len(failed),
                    sum(result.status is Status.ERROR for result in mine),
                    sum(r.affected_count for r in failed if (r.unit or "") in CHANGED_UNITS),
                )
            )
        scores.sort(key=_rank_order)
        return [
            MemberRank(
                member=member,
                rank=index + 1,
                status=status,
                failed=failed,
                not_assessed=unassessed,
                changed_records=changed,
            )
            for index, (member, status, failed, unassessed, changed) in enumerate(scores)
        ]

    def report(self) -> GroupReport:
        members = self.members()
        whole = [
            level
            for level in dict.fromkeys(
                _role(result).level for result in self.results if _role(result).whole_group
            )
            if level is not None
        ]
        return GroupReport(
            name=self.role.group,
            check=self.role.check,
            rule_id=self.rule_id,
            strategy=self.role.strategy,
            members=members,
            levels=self.levels(),
            whole_group_levels=whole,
            matrix=self.matrix(members),
            pairs=self.pairs(),
            ranking=self.ranking(members),
            group_results=[
                f"{result.check or result.name}: {result.status.value}"
                for result in self.results
                if not _role(result).members and not _role(result).whole_group
            ],
        )


def _role(result: RuleResult) -> GroupRole:
    assert result.group is not None
    return result.group


def _level(result: RuleResult) -> str:
    role = _role(result)
    return role.level or (role.check if role.check else CHECK_LEVEL)


def _comparison(result: RuleResult) -> str:
    """The pair's label: the check label up to the pair (``<name>: <a> vs <b>``)."""
    role = _role(result)
    return f"{role.group}: {role.pair[0]} vs {role.pair[1]}"


def _rank_order(score: tuple[str, Status, int, int, int]) -> tuple[int, int, int, int, str]:
    member, status, failed, unassessed, changed = score
    return (-failed, -unassessed, -changed, WORST_FIRST.index(status), member)


def group_reports(results: Iterable[RuleResult]) -> list[GroupReport]:
    """The roll-up of every check relating many datasets, in the order of its results."""
    rollups: dict[tuple[str, str], GroupRollup] = {}
    for result in results:
        if result.group is None:
            continue
        key = (result.rule_id, result.group.group)
        rollup = rollups.get(key)
        if rollup is None:
            rollup = rollups[key] = GroupRollup(result.rule_id, result.group)
        rollup.add(result)
    return [rollup.report() for rollup in rollups.values()]


__all__ = [
    "CHANGED_UNITS",
    "WORST_FIRST",
    "GroupReport",
    "MemberRank",
    "MemberRow",
    "PairSummary",
    "group_reports",
    "worst",
]
