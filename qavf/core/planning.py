"""Conservative column projection and snapshot lifetime planning."""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

from qavf.sql_dependencies import PredicateDependencies


@functools.lru_cache(maxsize=1024)
def expression_columns(expression):
    """Return parser-proven dependencies, or None when projection is unsafe."""
    from qavf.sources.sql import _parser

    try:
        row = _parser().execute("SELECT json_serialize_sql(?)", [f"SELECT {expression}"]).fetchone()
        if row is None:
            return None
        tree = json.loads(row[0])
    except Exception:
        return None
    if tree.get("error"):
        return None
    collector = _ColumnCollector()
    collector.visit(tree)
    return None if collector.unsafe else frozenset(collector.columns)


class _ColumnCollector:
    """Walk a serialized SQL tree, gathering column references and unsafe nodes."""

    def __init__(self):
        self.columns = set()
        self.unsafe = False

    def visit(self, node):
        """Record one node's column references, then visit its children."""
        if isinstance(node, dict):
            if node.get("class") in {"STAR", "SUBQUERY"}:
                self.unsafe = True
            if node.get("class") == "COLUMN_REF":
                self.columns.update(node["column_names"])
            for value in node.values():
                self.visit(value)
        elif isinstance(node, list):
            for value in node:
                self.visit(value)


def _dependency_order(names, source_dependencies, component):
    """Order a component's datasets so query sources follow their parents."""
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


@dataclass(frozen=True)
class NativePlan:
    """Input dependency components retain exactly one coherent snapshot per run.

    SQL whose reads the parser cannot prove is deliberately conservative: all
    inputs and columns stay available, and :attr:`fallback` says why. Declared
    local checks, parser-proven expressions and FK components have shorter
    lifetimes and narrower projections.
    """

    projections: dict
    groups: tuple[tuple[str, ...], ...]
    source_parents: frozenset[str] = frozenset()
    reasons: Mapping[str, str] = field(default_factory=dict)
    """Why a dataset keeps every column (absent: it is projected)."""
    fallback: tuple[str, ...] = ()
    """Why every dataset is read in full into one component (empty: it is not)."""

    @classmethod
    def build(cls, settings, switches, inputs=None):
        """Plan ``settings``; ``switches`` says which rules report (see
        :class:`~qavf.core.results.RuleSwitches`)."""
        return _NativePlanner(settings, switches, inputs).plan()

    def describe(self, name):
        """Data-free retention facts of one dataset, for explanations."""
        columns = self.projections.get(name)
        component = next((group for group in self.groups if name in group), (name,))
        return {
            "retained_columns": None if columns is None else tuple(sorted(columns)),
            "retention": self.reasons.get(name),
            "component": tuple(component),
        }


class _NativePlanner:
    """Build a :class:`NativePlan` from the configured contracts and SQL."""

    def __init__(self, settings, switches, inputs):
        self.settings = settings
        self.switches = switches
        self.names = list(settings.datasets)
        self.reads = PredicateDependencies(self.names)
        self.source_dependencies = _source_dependencies(self.names, inputs or {})
        self.ordered = functools.partial(_dependency_order, self.names, self.source_dependencies)

    def plan(self):
        fallback = self._fallback()
        if fallback:
            reason = "SQL whose reads are not proven runs over every dataset"
            return NativePlan(
                {name: None for name in self.names},
                (self.ordered(self.names),),
                frozenset(self.names),
                dict.fromkeys(self.names, reason),
                tuple(fallback),
            )
        projections, reasons = {}, {}
        for name, spec in self.settings.datasets.items():
            projections[name], reason = self._contract_columns(name, spec)
            if reason is not None:
                reasons[name] = reason
        edges: dict[str, set[str]] = {name: set() for name in self.names}
        parents = {parent for values in self.source_dependencies.values() for parent in values}
        for name, spec in self.settings.datasets.items():
            if not self.switches.enabled("INP-005", name):
                continue
            for key in spec.foreign_keys:
                parent = key.ref_dataset
                if parent not in edges:
                    continue
                edges[name].add(parent)
                edges[parent].add(name)
                child_columns, parent_columns = projections[name], projections[parent]
                if child_columns is not None:
                    child_columns.update(key.columns)
                if parent_columns is not None:
                    parent_columns.update(key.target_columns)
        read = [
            (name, parent, "a query input reads it")
            for name, values in self.source_dependencies.items()
            for parent in sorted(values)
        ]
        read += [
            (name, other, f"a subquery of {name} reads it")
            for name in self.names
            for predicate in self._predicates(name)
            for other in sorted(self.reads.of(predicate, name).datasets)
        ]
        for name, parent, why in read:
            edges[name].add(parent)
            edges[parent].add(name)
            projections[parent] = None
            reasons.setdefault(parent, why)
            parents.add(parent)
        return NativePlan(projections, self._groups(edges), frozenset(parents), reasons)

    def _fallback(self):
        """Why the whole project must stay one fully read component."""
        found = []
        if self.settings.assertions and self.switches.enabled("INP-007"):
            found.append("SQL assertions may read any dataset and column")
        for name in self.names:
            for predicate in self._predicates(name):
                dependencies = self.reads.of(predicate, name)
                if not dependencies.proven:
                    found.append(f"{name}: {dependencies.reason}")
        return found

    def _predicates(self, name):
        """Every SQL predicate evaluated against ``name``'s rows."""
        spec = self.settings.datasets[name]
        found = list(_conditions(spec.schema_spec)) if spec.schema_spec is not None else []
        if self.switches.enabled("INP-006", name):
            found += [check.violation_condition() for check in spec.expressions_of(name)]
        return found

    def _contract_columns(self, name, spec):
        """The columns ``name``'s own checks read, or ``None`` with the reason."""
        enabled = self.switches.enabled
        columns: set[str] = set()
        conditions: list[str] = []
        schema = spec.schema_spec
        if schema:
            if schema.unique_rows and enabled("INP-004", name):
                return None, "unique_rows compares whole rows"
            for column, policy in schema.columns.items():
                # Type/null checks are INP-002 even if value rules are disabled.
                if enabled("INP-002", name) or enabled("INP-003", name):
                    columns.add(column)
                    if policy.unit_column:
                        columns.add(policy.unit_column)
                    conditions.extend(check.where for check in policy.all_checks() if check.where)
                if policy.na_values and enabled("INP-009", name):
                    columns.add(column)
            if enabled("INP-004", name):
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
            if enabled("INP-009", name):
                for group in schema.na_groups:
                    columns.update(group)
            evidence = schema.evidence_columns
            if evidence == "all":
                return None, "evidence_columns: all shows whole rows"
            if isinstance(evidence, list):
                columns.update(evidence)
        for condition in conditions:
            dependencies = self.reads.of(condition, name).columns
            if dependencies is None:
                return None, "a where condition may read any column"
            columns.update(dependencies)
        if enabled("INP-006", name) and spec.expressions_of(name):
            if self.settings.engine.max_affected_records:
                return None, "expression evidence shows whole rows"
            for check in spec.expressions_of(name):
                dependencies = self.reads.of(check.violation_condition(), name).columns
                if dependencies is None:
                    return None, "an expression may read any column"
                columns.update(dependencies)
        return columns, None

    def _groups(self, edges):
        groups = []
        visited = set()
        for name in self.names:
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
            groups.append(self.ordered(component))
        return tuple(groups)


def _conditions(schema):
    """The ``where`` conditions of a schema's column checks, keys and groups."""
    for column in schema.columns.values():
        for check in column.conditions:
            if check.where:
                yield check.where
    for key in schema.alternate_keys:
        if key.where:
            yield key.where
    for group in schema.grouped_values:
        if group.where:
            yield group.where


def _source_dependencies(names, inputs):
    """The datasets each query input reads, by parsing its SQL."""
    from qavf.inputs.descriptors import Query
    from qavf.sources.sql import _parser

    found: dict[str, set[str]] = {name: set() for name in names}
    for name, value in inputs.items():
        sql = (
            value.sql
            if isinstance(value, Query)
            and (value.connection is None or isinstance(value.connection, str))
            else value.get("sql")
            if isinstance(value, Mapping) and value.get("type") == "query"
            else None
        )
        if not sql:
            continue
        try:
            referenced = {table.casefold() for table in _parser().get_table_names(sql)}
            found[name] = {other for other in names if other.casefold() in referenced}
        except Exception:
            # A backend expression can still be assessed at load time.
            # Retain all potential parents if parsing cannot prove more.
            found[name] = set(names) - {name}
    return found
