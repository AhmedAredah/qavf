"""Named source attachments on the core's single owned SQL connection."""

from __future__ import annotations

from qavf.inputs.descriptors import Query
from qavf.optional import unsupported
from qavf.sources.base import SourceEnvironment
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import referenced_connections


class SourceConnections:
    """Lazily attach only requested sources while preserving shared budgets."""

    def __init__(self, connection, settings):
        self.connection = connection
        self.settings = settings
        self.attached = set()
        self.environment = SourceEnvironment(
            base_dir=settings.base_dir,
            connections=settings.connections,
            secrets=settings.secrets,
        )
        if settings.secrets:
            from qavf.sources.native import configure_sources

            configure_sources(connection, self.environment)

    def resolve(self, value, spec):
        from dataclasses import replace

        from qavf.sources.native import configure_sources
        from qavf.sources.parameters import DatabaseParams, QueryParams

        if isinstance(value, Query):
            query = value
            attach = referenced_connections(query.sql, self.settings.connections)
            if isinstance(query.connection, str):
                attach.append(query.connection)
        elif value.get("type") == "query":
            query_params = QueryParams.model_validate(
                {
                    key: item
                    for key, item in value.items()
                    if key not in {"connection", "parameters"}
                }
            )
            query = Query(query_params.sql, parameters=value.get("parameters", {}))
            attach = query_params.connections or referenced_connections(
                query.sql, self.settings.connections
            )
            if value.get("connection"):
                attach.append(value["connection"])
        elif value.get("type") == "database":
            database = DatabaseParams.model_validate(value)
            parts = [
                database.connection,
                *([database.db_schema] if database.db_schema else []),
                database.table,
            ]
            sql = database.query or "SELECT * FROM " + ".".join(qi(part) for part in parts if part)
            query, attach = Query(sql), [database.connection]
        else:
            unsupported(f"source type {value.get('type')!r}")
        missing = [name for name in dict.fromkeys(attach) if name not in self.attached]
        if missing:
            configure_sources(self.connection, replace(self.environment, secrets={}), missing)
            self.attached.update(missing)
        return Query(query.sql, connection=self.connection, parameters=query.parameters)
