"""Disposable PostgreSQL corpus setup; no validation logic lives in fixtures."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .cleanup import cleanup_owned
from .contracts import corpus
from .policy import IntegrationUnavailable, dependency, extension, settings


@dataclass(frozen=True)
class PostgresCorpus:
    dsn: str = field(repr=False)
    schema: str

    def table(self, clean: bool) -> str:
        return f"{self.schema}.{'clean' if clean else 'invalid'}"

    def connection_config(self, *, snapshot: bool = False) -> dict:
        return {
            "warehouse": {
                "type": "postgres",
                "uri": self.dsn,
                "read_only": True,
                "snapshot": snapshot,
            }
        }


@contextmanager
def postgres_corpus() -> Iterator[PostgresCorpus]:
    config = settings(os.environ, ("QAVF_POSTGRES_DSN",))
    psycopg = dependency("psycopg")
    extension("postgres")
    schema = "qavf_integration_" + uuid.uuid4().hex
    try:
        connection = psycopg.connect(
            config["QAVF_POSTGRES_DSN"], connect_timeout=5, autocommit=True
        )
    except Exception as exc:
        raise IntegrationUnavailable(
            f"required PostgreSQL service unavailable ({type(exc).__name__})"
        ) from None
    created = False
    try:
        try:
            connection.execute(f'CREATE SCHEMA "{schema}"')
            created = True
            for clean in (True, False):
                table = f'"{schema}".{"clean" if clean else "invalid"}'
                connection.execute(
                    f"CREATE TABLE {table} (ordinal INTEGER, id BIGINT, amount NUMERIC(24,4), "
                    "stamp TIMESTAMPTZ, label TEXT)"
                )
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f"INSERT INTO {table} VALUES (%s,%s,%s,%s,%s)", corpus(clean).iter_rows()
                    )
        except Exception as exc:
            raise IntegrationUnavailable(
                f"required PostgreSQL fixture setup failed ({type(exc).__name__})"
            ) from None
        yield PostgresCorpus(config["QAVF_POSTGRES_DSN"], schema)
    finally:
        actions = []
        if created:
            actions.append(lambda: connection.execute(f'DROP SCHEMA "{schema}" CASCADE'))
        actions.append(connection.close)
        cleanup_owned("PostgreSQL", actions, sys.exception())
