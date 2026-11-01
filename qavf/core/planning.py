"""Conservative column projection and snapshot lifetime planning."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass


@functools.lru_cache(maxsize=1024)
def expression_columns(expression):
    """Return parser-proven dependencies, or None when projection is unsafe."""
    from qavf.sources.sql import _parser

    try:
        tree = json.loads(
            _parser()
            .execute("SELECT json_serialize_sql(?)", [f"SELECT {expression}"])
            .fetchone()[0]
        )
    except Exception:
        return None
    if tree.get("error"):
        return None
    columns = set()
    unsafe = False

    def visit(node):
        nonlocal unsafe
        if isinstance(node, dict):
            if node.get("class") in {"STAR", "SUBQUERY"}:
                unsafe = True
            if node.get("class") == "COLUMN_REF":
                columns.update(node["column_names"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(tree)
    return None if unsafe else frozenset(columns)


@dataclass(frozen=True)
class NativePlan:
    """Input dependency components retain exactly one coherent snapshot per run.

    Arbitrary SQL is deliberately conservative: all inputs and columns stay
    available. Declared local checks and FK components have shorter lifetimes.
    """

    projections: dict
    groups: tuple[tuple[str, ...], ...]
    source_parents: frozenset[str] = frozenset()

    @classmethod
    def build(cls, settings, collector, inputs=None):
        from collections.abc import Mapping

        from qavf.inputs.descriptors import Query
        from qavf.sources.sql import _parser

        names = list(settings.datasets)
        source_dependencies = {name: set() for name in names}
        for name, value in (inputs or {}).items():
            sql = (
                value.sql
                if isinstance(value, Query)
                and (value.connection is None or isinstance(value.connection, str))
                else value.get("sql")
                if isinstance(value, Mapping) and value.get("type") == "query"
                else None
            )
            if sql:
                try:
                    referenced = {table.casefold() for table in _parser().get_table_names(sql)}
                    source_dependencies[name] = {
                        other for other in names if other.casefold() in referenced
                    }
                except Exception:
                    # A backend expression can still be assessed at load time.
                    # Retain all potential parents if parsing cannot prove more.
                    source_dependencies[name] = set(names) - {name}
        source_parents = frozenset(
            parent for values in source_dependencies.values() for parent in values
        )

        def ordered(component):
            remaining, result = set(component), []
            while remaining:
                ready = [
                    name
                    for name in names
                    if name in remaining and not source_dependencies[name].intersection(remaining)
                ]
                if not ready:  # cyclic references are reported as load errors
                    ready = [name for name in names if name in remaining]
                result.extend(ready)
                remaining.difference_update(ready)
            return tuple(result)

        conditions = [
            condition
            for spec in settings.datasets.values()
            if spec.schema_spec is not None
            for condition in (
                *(
                    check.where
                    for column in spec.schema_spec.columns.values()
                    for check in column.checks
                ),
                *(key.where for key in spec.schema_spec.alternate_keys),
                *(group.where for group in spec.schema_spec.grouped_values),
            )
            if condition
        ]
        unproven_conditions = any(expression_columns(condition) is None for condition in conditions)
        if unproven_conditions:
            source_parents = frozenset(names)
        global_sql = (
            unproven_conditions
            or bool(settings.assertions and collector.enabled("INP-007"))
            or any(
                spec.expressions and collector.enabled("INP-006", name)
                for name, spec in settings.datasets.items()
            )
        )
        if global_sql:
            return cls({name: None for name in names}, (ordered(names),), source_parents)
        projections = {}
        edges = {name: set() for name in names}
        for name, spec in settings.datasets.items():
            columns = set()
            conditions = []
            schema = spec.schema_spec
            if schema:
                if schema.unique_rows and collector.enabled("INP-004", name):
                    projections[name] = None
                    continue
                for column, policy in schema.columns.items():
                    # Type/null checks are INP-002 even if value rules are disabled.
                    if collector.enabled("INP-002", name) or collector.enabled("INP-003", name):
                        columns.add(column)
                        if policy.unit_column:
                            columns.add(policy.unit_column)
                        conditions.extend(
                            check.where for check in policy.all_checks() if check.where
                        )
                    if policy.na_values and collector.enabled("INP-009", name):
                        columns.add(column)
                if collector.enabled("INP-004", name):
                    columns.update(schema.primary_key)
                    for key in schema.alternate_keys:
                        columns.update(key.columns)
                        if key.where:
                            conditions.append(key.where)
                    for group in schema.grouped_values:
                        columns.update(group.group_by)
                        columns.add(group.column)
                        if group.where:
                            conditions.append(group.where)
                if collector.enabled("INP-009", name):
                    for group in schema.na_groups:
                        columns.update(group)
                evidence = schema.evidence_columns
                if evidence == "all":
                    projections[name] = None
                    continue
                if isinstance(evidence, list):
                    columns.update(evidence)
            for condition in conditions:
                dependencies = expression_columns(condition)
                if dependencies is None:
                    columns = None
                    break
                columns.update(dependencies)
            projections[name] = columns
        for name, spec in settings.datasets.items():
            if not collector.enabled("INP-005", name):
                continue
            for key in spec.foreign_keys:
                parent = key.ref_dataset
                if parent not in edges:
                    continue
                edges[name].add(parent)
                edges[parent].add(name)
                if projections[name] is not None:
                    projections[name].update(key.columns)
                if projections[parent] is not None:
                    projections[parent].update(key.target_columns)
        for name, dependencies in source_dependencies.items():
            for parent in dependencies:
                edges[name].add(parent)
                edges[parent].add(name)
                projections[parent] = None
        groups = []
        visited = set()
        for name in names:
            if name in visited:
                continue
            component, pending = set(), [name]
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(edges[current] - component)
            visited.update(component)
            groups.append(ordered(component))
        return cls(projections, tuple(groups), source_parents)
