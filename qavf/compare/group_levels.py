"""Comparison levels assessed over every member of a group at once.

``compare`` with more than two members assesses three levels over the whole
group instead of pair by pair:

* :class:`GroupCoverage`: a key x member presence matrix; each member is reported
  with the keys other members hold and it lacks;
* :class:`GroupStructure`: column names and types across members; each member is
  reported with the columns most members have and it lacks, the columns few
  members have and it has, and types that differ from the most common one;
* ``distribution``: each member against the pooled data, run by the two-sided
  engine (:mod:`qavf.engine.derived` supplies the pooled dataset).

Each finding is about one member (:meth:`GroupPlan.member_role`), so the report's
matrix and ranking show which member differs.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Sequence
from typing import Any

import polars as pl

from qavf.compare.pairing import GroupPlan
from qavf.engine.ports import LevelReads
from qavf.engine.registry import Finding
from qavf.redaction import redact_text

_MEMBER = "__qavf_member"
MISSING_FROM = "missing_from"
"""Evidence field listing the members a key is missing from."""


class GroupLevel:
    """A level assessed over all members of a group, reported per member."""

    level = ""

    def __init__(self, plan: GroupPlan, label: str, severity: Any = None) -> None:
        self.plan = plan
        self.label = label
        self.severity = severity

    def check(self, member: str, what: str) -> str:
        return f"{self.plan.name}: {self.label}: {what} {member}"

    def finding(self, finding: Finding, member: str) -> Finding:
        """``finding`` with this level's role for ``member``."""
        return dataclasses.replace(finding, group=self.plan.member_role(member, self.label))

    def unassessed(self, reason: str) -> list[Finding]:
        return [
            self.finding(
                Finding.unassessed(
                    f"{self.plan.name}: {self.label}: not assessed: {redact_text(reason)}",
                    check=f"{self.plan.name}: {self.label}",
                    severity=self.severity,
                    datasets=(member,),
                ),
                member,
            )
            for member in self.plan.members
        ]

    def evaluate(self, ctx: LevelReads) -> list[Finding]:
        unavailable = [member for member in self.plan.members if not ctx.has(member)]
        if unavailable:
            return self.unassessed(f"dataset {unavailable[0]!r} is not available")
        try:
            return self.assess(ctx)
        except Exception as exc:  # a level that raises is not assessed, like pair levels
            return self.unassessed(f"{type(exc).__name__}: {exc}")

    def assess(self, ctx: LevelReads) -> list[Finding]:
        raise NotImplementedError


class GroupCoverage(GroupLevel):
    """Keys missing from any member: a key x member presence matrix."""

    level = "coverage"

    def keys_of(self, ctx: LevelReads, member: str) -> pl.DataFrame:
        keys = list(self.plan.keys)
        if self.plan.where is None:
            return ctx.project(member, keys).unique()
        columns = sorted(set(keys) | set(pl.sql_expr(self.plan.where).meta.root_names()))
        frame = ctx.project(member, columns).filter(pl.sql_expr(self.plan.where))
        return frame.select(keys).unique()

    def assess(self, ctx: LevelReads) -> list[Finding]:
        keys = list(self.plan.keys)
        if not keys:
            return self.unassessed("coverage needs the check's keys")
        members = list(self.plan.members)
        frames = []
        for index, member in enumerate(members):
            frames.append(
                self.keys_of(ctx, member).with_columns(pl.lit(index, dtype=pl.Int32).alias(_MEMBER))
            )
        stacked = pl.concat(frames, how="vertical_relaxed")
        presence = stacked.group_by(keys).agg(pl.col(_MEMBER).sort().alias(_MEMBER))
        total = presence.height
        partial = presence.filter(pl.col(_MEMBER).list.len() < len(members))
        findings = []
        for index, member in enumerate(members):
            missing = partial.filter(~pl.col(_MEMBER).list.contains(index))
            count = missing.height
            records = []
            for row in missing.head(ctx.evidence_limit).to_dicts():
                present = row.pop(_MEMBER)
                row[MISSING_FROM] = member
                row["present_in"] = ", ".join(members[i] for i in present)
                records.append(row)
            common: dict[str, Any] = {
                "check": self.check(member, "keys missing from"),
                "datasets": (member,),
                "severity": self.severity,
                "total_records": total,
                "unit": "keys",
                "metrics": {"members": len(members), "keys": total, "missing": count},
            }
            message = (
                f"{self.plan.name}: {self.label}: {count:,} of {total:,} key(s) held by "
                f"other members are missing from {member}"
            )
            if count:
                finding = Finding.fail(message, affected=records, affected_count=count, **common)
            else:
                finding = Finding.ok(
                    f"{self.plan.name}: {self.label}: {member} holds every key of the group",
                    affected_count=0,
                    **common,
                )
            findings.append(self.finding(finding, member))
        return findings


class GroupStructure(GroupLevel):
    """Column names and types across members, against what most members have."""

    level = "structure"

    def assess(self, ctx: LevelReads) -> list[Finding]:
        ignored = set(self.plan.ignore_columns)
        schemas = {
            member: {
                name: str(dtype)
                for name, dtype in ctx.schema(member).items()
                if name not in ignored
            }
            for member in self.plan.members
        }
        counts = Counter(name for schema in schemas.values() for name in schema)
        majority = len(schemas) / 2
        expected = [name for name, count in counts.items() if count > majority]
        types = {name: _modal(schemas, name) for name in expected}
        findings = []
        for member, schema in schemas.items():
            problems = self.problems(schema, expected, types, counts)
            common: dict[str, Any] = {
                "check": self.check(member, "columns of"),
                "datasets": (member,),
                "severity": self.severity,
                "total_records": len(set(expected) | set(schema)),
                "unit": "columns",
                "metrics": {"expected_columns": len(expected), "members": len(schemas)},
            }
            if problems:
                message = (
                    f"{self.plan.name}: {self.label}: {member} differs from most members in "
                    f"{len(problems)} column(s)"
                )
                finding = Finding.fail(
                    message,
                    affected=problems[: ctx.evidence_limit],
                    affected_count=len(problems),
                    **common,
                )
            else:
                finding = Finding.ok(
                    f"{self.plan.name}: {self.label}: {member} has the columns and types of "
                    "most members",
                    affected_count=0,
                    **common,
                )
            findings.append(self.finding(finding, member))
        return findings

    @staticmethod
    def problems(
        schema: dict[str, str],
        expected: Sequence[str],
        types: dict[str, str],
        counts: Counter[str],
    ) -> list[dict[str, Any]]:
        problems: list[dict[str, Any]] = []
        for name in expected:
            if name not in schema:
                problems.append({"column": name, "problem": "missing", "expected": types[name]})
            elif schema[name] != types[name]:
                problems.append(
                    {
                        "column": name,
                        "problem": "type",
                        "dtype": schema[name],
                        "expected": types[name],
                    }
                )
        for name, dtype in schema.items():
            if name not in types:
                problems.append(
                    {"column": name, "problem": "extra", "dtype": dtype, "members": counts[name]}
                )
        return problems


def _modal(schemas: dict[str, dict[str, str]], column: str) -> str:
    """The most common type of ``column`` among the members that have it (first on ties)."""
    seen = Counter(schema[column] for schema in schemas.values() if column in schema)
    return seen.most_common(1)[0][0]


def group_levels(plan: GroupPlan) -> list[GroupLevel]:
    """The whole-group levels the plan configures (``coverage`` and ``structure``)."""
    levels: list[GroupLevel] = []
    if plan.structure is not None:
        levels.append(GroupStructure(plan, plan.structure, plan.structure_severity))
    if plan.coverage is not None:
        levels.append(GroupCoverage(plan, plan.coverage, plan.coverage_severity))
    return levels


__all__ = ["MISSING_FROM", "GroupCoverage", "GroupLevel", "GroupStructure", "group_levels"]
