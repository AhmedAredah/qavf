"""Native relation binding without discarding verified file-reader pushdown."""

from __future__ import annotations

import duckdb

from qavf.engine.connections import Connection
from qavf.engine.projection import repeatable_file_source
from qavf.sources.base import DataSource, ScanPlan
from qavf.sources.native import OwnedRelation


def bind_scan(
    connection: Connection, source: DataSource, plan: ScanPlan
) -> duckdb.DuckDBPyRelation | OwnedRelation:
    """Bind a source inside its caller-owned identity and connection operation.

    DuckDB can materialize parameterized relations before downstream projection.
    Only the built-in local file reader supplies repeatable, generated SQL that
    may stay lazy across schema, budget and result reads. Its path literals are
    escaped when the reader plan is built; this boundary never substitutes
    parameters into SQL. All other sources retain their existing binding call.
    In particular, empty bindings do not promise a stable observation: DuckDB
    versions differ in whether they materialize or re-evaluate such queries.
    """
    if repeatable_file_source(source) and not plan.parameters:
        return connection.sql(plan.sql)
    return connection.sql(plan.sql, params=dict(plan.parameters))
