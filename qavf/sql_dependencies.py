"""What a dataset's SQL predicate reads, as far as DuckDB's parser proves it.

Row expressions and ``where`` conditions are SQL fragments evaluated against
one dataset. Planning narrows inputs only on parser-proven facts: the columns a
predicate names outside any subquery belong to its own dataset, and the base
tables its subqueries name are datasets it reads. Anything the parser cannot
attribute (a table function, a qualified or unknown relation, unparseable
text) is reported with a reason, and callers keep their conservative plan.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SqlDependencies:
    """The data one predicate reads.

    ``columns`` are its own dataset's columns (``None``: possibly every column,
    e.g. under ``*`` or a correlated subquery). ``datasets`` are the other
    datasets it reads through subqueries. ``reason`` explains why the reads
    could not be proven; the other fields are then meaningless.
    """

    columns: frozenset[str] | None = frozenset()
    datasets: frozenset[str] = frozenset()
    reason: str | None = None

    @property
    def proven(self) -> bool:
        return self.reason is None


class _PredicateTree:
    """Walk a serialized predicate, separating its own reads from subqueries'."""

    def __init__(self) -> None:
        self.columns: set[str] = set()
        self.every_column = False
        self.subquery = False
        self.opaque: list[str] = []

    def visit(self, node: Any, nested: bool = False) -> None:
        if isinstance(node, list):
            for value in node:
                self.visit(value, nested)
            return
        if not isinstance(node, dict):
            return
        kind, cls = node.get("type"), node.get("class")
        if cls == "SUBQUERY":
            self.subquery = True
            nested = True
        elif not nested and cls == "STAR":
            self.every_column = True
        elif not nested and cls == "COLUMN_REF":
            # ``a.b`` may be a qualified column or a struct field: keep both names.
            self.columns.update(str(name) for name in node.get("column_names", ()))
        if kind == "TABLE_FUNCTION":
            self.opaque.append("a table function")
        elif kind == "BASE_TABLE" and (
            node.get("catalog_name") or str(node.get("schema_name", "")).lower() not in ("", "main")
        ):
            self.opaque.append(f"the qualified relation {node.get('table_name')!r}")
        for value in node.values():
            self.visit(value, nested)


class PredicateDependencies:
    """Prove the reads of predicates evaluated against configured datasets."""

    def __init__(self, datasets: Iterable[str]) -> None:
        self.datasets = tuple(datasets)

    def of(self, predicate: str, dataset: str) -> SqlDependencies:
        """The reads of ``predicate`` evaluated as a row filter of ``dataset``."""
        return _dependencies(predicate, dataset, self.datasets)


@functools.lru_cache(maxsize=1024)
def _dependencies(predicate: str, dataset: str, datasets: tuple[str, ...]) -> SqlDependencies:
    from qavf.sources.sql import _BaseTables, _parser

    tree: dict[str, Any]
    try:
        row = (
            _parser()
            .execute("SELECT json_serialize_sql(?)", [f"SELECT 1 WHERE ({predicate}\n)"])
            .fetchone()
        )
        tree = json.loads(row[0]) if row is not None else {"error": True}
    except Exception:
        tree = {"error": True}
    statements = tree.get("statements") or []
    if tree.get("error") or len(statements) != 1:
        return SqlDependencies(reason=f"the predicate {predicate!r} could not be parsed")
    walker = _PredicateTree()
    walker.visit(statements[0])
    if walker.opaque:
        return SqlDependencies(reason=f"the predicate {predicate!r} reads {walker.opaque[0]}")
    tables = _BaseTables()
    tables.visit(statements[0])
    known = {name.casefold(): name for name in datasets}
    unknown = sorted(name for name in tables.names if name.casefold() not in known)
    if unknown:
        return SqlDependencies(
            reason=f"the predicate {predicate!r} reads {unknown[0]!r}, which is not a dataset"
        )
    read = {known[name.casefold()] for name in tables.names}
    # A subquery's unqualified columns may bind to the outer row: keep it whole.
    every = walker.every_column or walker.subquery
    return SqlDependencies(
        columns=None if every else frozenset(walker.columns),
        datasets=frozenset(read - {dataset}),
    )


__all__ = ["PredicateDependencies", "SqlDependencies"]
