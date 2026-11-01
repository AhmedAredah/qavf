"""Backend-independent SQL for grouped allowed-value occurrence contracts.

The execution backend owns conversion, admission and result materialization.
This planner only compares already converted values and never collects groups.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qavf.sources.sql import quote_identifier as qi

if TYPE_CHECKING:
    from qavf.settings import GroupedValueSpec


@dataclass(frozen=True)
class GroupedValuePlan:
    """Exact aggregate counts plus a relation of violating group/domain pairs."""

    violations_sql: str
    total_sql: str
    occurrences_column: str
    reason_column: str

    @classmethod
    def build(
        cls,
        spec: GroupedValueSpec,
        source_sql: str,
        domain_literals: Sequence[str],
        expected_group_literals: Sequence[Sequence[str]] | None = None,
        *,
        eligible_sql: str = "TRUE",
        where_sql: str | None = None,
        value_present_sql: str | None = None,
    ) -> GroupedValuePlan:
        """Build spillable SQL; supplied literals must already have canonical types.

        Groups are observed before the row condition, then unioned with explicit
        expected groups. NULL group components compare equal. The eligibility
        predicate excludes N/A groups; the value predicate excludes N/A/missing
        domain values. Every allowed value is checked even in empty groups.
        """
        fields = [*spec.group_by, spec.column]
        taken = {name.casefold() for name in fields}

        def alias(base: str) -> str:
            name = base
            while name.casefold() in taken:
                name = "qavf_" + name
            taken.add(name.casefold())
            return name

        count, reason = alias("occurrences"), alias("reason")

        def relation_alias(name: str) -> str:
            candidate = "__qavf_grouped_" + name
            while candidate.casefold() in source_sql.casefold():
                candidate += "_"
            return qi(candidate)

        source_name, group_name, domain_name, counts_name = (
            relation_alias(name) for name in ("source", "groups", "domain", "counts")
        )
        groups = ", ".join(qi(c) for c in spec.group_by)
        column = qi(spec.column)
        members = ", ".join(f"({value})" for value in domain_literals)
        # A typed empty domain still has the source column's type and no members.
        domain = (
            f"SELECT DISTINCT {column} FROM (VALUES {members}) allowed({column})"
            if members
            else f"SELECT {column} FROM {source_name} WHERE FALSE"
        )
        observed = f"SELECT DISTINCT {groups} FROM {source_name} WHERE {eligible_sql}"
        expected = expected_group_literals or ()
        if expected:
            rows = ", ".join("(" + ", ".join(values) + ")" for values in expected)
            observed += f" UNION SELECT * FROM (VALUES {rows}) expected({groups})"
        scope = where_sql if where_sql is not None else (spec.where or "TRUE")
        present = value_present_sql or f"{column} IS NOT NULL"
        common = (
            f"WITH {source_name} AS ({source_sql}), {group_name} AS ({observed}), "
            f"{domain_name} AS ({domain}), {counts_name} AS (SELECT {groups}, {column}, "
            f"count(*)::UBIGINT AS {qi(count)} FROM {source_name} WHERE ({eligible_sql}) "
            f"AND coalesce(({scope}), FALSE) AND ({present}) GROUP BY {groups}, {column}) "
        )
        match = " AND ".join(
            f"expected.{qi(c)} IS NOT DISTINCT FROM actual.{qi(c)}" for c in fields
        )
        expected_fields = ", ".join(f"expected.{qi(c)}" for c in fields)
        actual_count = f"coalesce(actual.{qi(count)}, 0)::UBIGINT"
        bounds = [f"{actual_count} < {spec.min_occurrences}"]
        if spec.max_occurrences is not None:
            bounds.append(f"{actual_count} > {spec.max_occurrences}")
        violations = (
            f"SELECT {expected_fields}, {actual_count} AS {qi(count)}, "
            f"'occurrence count' AS {qi(reason)} FROM "
            f"(SELECT groups.*, domain.{column} FROM {group_name} groups "
            f"CROSS JOIN {domain_name} domain) expected "
            f"LEFT JOIN {counts_name} actual ON {match} WHERE {' OR '.join(bounds)}"
        )
        unexpected = (
            f"NOT EXISTS (SELECT 1 FROM {domain_name} domain WHERE domain.{column} "
            f"IS NOT DISTINCT FROM actual.{column})"
        )
        total = f"SELECT count(*) FROM {group_name} CROSS JOIN {domain_name}"
        if not spec.allow_other_values:
            violations += (
                f" UNION ALL SELECT {', '.join('actual.' + qi(c) for c in fields)}, "
                f"actual.{qi(count)}, 'unexpected value' AS {qi(reason)} FROM {counts_name} actual "
                f"WHERE {unexpected}"
            )
            total = (
                f"SELECT ({total}) + (SELECT count(*) FROM {counts_name} actual WHERE {unexpected})"
            )
        return cls(
            common + violations,
            common + total,
            count,
            reason,
        )
