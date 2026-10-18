"""Native connection ownership, error classification, and bounded SQL results."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import duckdb
import polars as pl

from qavf.redaction import redact_text
from qavf.sources.base import ConnectionBudget, SourceEnvironment, SourceError, SourceNotFoundError
from qavf.sources.errors import redacted_error
from qavf.sources.memory import reservation_is_stricter
from qavf.sources.sql import sql_literal, validate_query
from qavf.sources.storage import create_secret_statements

logger = logging.getLogger(__name__)


class OwnedConnection:
    """A native handle with exactly-once release of its run-wide reservation."""

    def __init__(
        self, connection: duckdb.DuckDBPyConnection, owner: ConnectionBudget, token: object
    ) -> None:
        self._connection = connection
        self._owner = owner
        self._token = token
        self._closed = False
        self._close_lock = threading.Lock()
        try:
            register = getattr(owner, "register_connection", None)
            if register is not None:
                register(token, self)
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._connection.close()
            finally:
                try:
                    unregister = getattr(self._owner, "unregister_connection", None)
                    if unregister is not None:
                        unregister(self._token)
                finally:
                    self._owner.release(self._token)

    def __enter__(self) -> OwnedConnection:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def connect(
    env: SourceEnvironment, attach: Iterable[str] = ()
) -> duckdb.DuckDBPyConnection | OwnedConnection:
    """Open an in-memory DuckDB connection and attach the named connections.

    Each name in ``attach`` is attached read-only (unless configured otherwise)
    under its configured name. Relative file paths in SQL are not resolved here;
    pass SQL through :func:`qavf.sources.sql.resolve_sql_paths` first.

    Raises:
        SourceError: for an unknown connection or a failed attachment.
    """
    owner = env.connection_budget
    token, spill_bytes, memory_bytes = (
        owner.acquire() if owner else (None, env.spill_limit_bytes, None)
    )
    try:
        if memory_bytes is not None and memory_bytes < 1:
            raise SourceError("native memory reservation has no capacity for a connection")
        native = duckdb.connect()
    except BaseException:
        if owner:
            owner.release(token)
        raise
    con = OwnedConnection(native, owner, token) if owner else native
    try:
        if env.threads:
            con.execute(f"SET threads = {int(env.threads)}")
        # Validate the explicit setting with DuckDB itself, including its native
        # aliases/sentinels. A run target may tighten that cap, never loosen it.
        if env.memory_limit:
            con.execute(f"SET memory_limit = {sql_literal(env.memory_limit)}")
        if memory_bytes is not None and reservation_is_stricter(env.memory_limit, memory_bytes):
            con.execute(f"SET memory_limit = {sql_literal(str(memory_bytes) + 'B')}")
        if env.temp_directory is not None:
            location = env.temp_directory / f"qavf-{uuid.uuid4().hex}"
            con.execute(f"SET temp_directory = {sql_literal(str(location))}")
        if spill_bytes is not None:
            con.execute(f"SET max_temp_directory_size = {sql_literal(str(spill_bytes) + 'B')}")
        for name, secret in env.secrets.items():
            try:
                for statement in create_secret_statements(name, secret):
                    con.execute(statement)
            except duckdb.Error as exc:
                # The statement carries credentials: report only the redacted
                # native error, never the SQL that failed.
                error = redacted_error(exc)
                if secret.provider == "credential_chain":
                    # Newer DuckDB validates the chain on creation. Without
                    # credentials, only reads that need this secret can fail;
                    # unrelated local datasets on the connection still load.
                    logger.warning(
                        "storage secret %r found no usable credentials: %s",
                        redact_text(name),
                        error,
                    )
                    continue
                raise SourceError(
                    f"could not create storage secret {redact_text(name)!r}: {error}"
                ) from None
        for name in dict.fromkeys(attach):
            spec = env.connections.get(name)
            if spec is None:
                raise SourceError(f"unknown connection {name!r}; define it under 'connections'")
            if spec.file_based and spec.read_only and not env.resolve(str(spec.path)).exists():
                raise SourceNotFoundError(
                    f"database file of connection {name!r} not found: {env.resolve(str(spec.path))}"
                )
            try:
                con.execute(spec.attach_sql(name, env))
            except duckdb.Error as exc:
                # Classify the native exception before adding context: wrapping
                # it first would erase HTTP status/type information used by the
                # shared retry policy.
                error = redacted_error(exc)
                raise SourceError(
                    f"could not attach connection {redact_text(name)!r}: {error}",
                    retryable=error.retryable,
                ) from None
        return con
    except BaseException:
        con.close()
        raise


def fetch(
    con: duckdb.DuckDBPyConnection | OwnedConnection,
    sql: str,
    params: Mapping[str, Any] | None = None,
    *,
    keep_open: bool = False,
) -> pl.DataFrame:
    try:
        result = con.execute(sql, dict(params or {})).pl()
        if result is None:
            raise SourceError("the SQL statement returned no result set")
        return result
    except duckdb.IOException as exc:
        # DuckDB messages repeat the location (credentials and signed tokens included).
        message = str(exc)
        if "No files found" in message or "No such file" in message or "does not exist" in message:
            raise redacted_error(exc, SourceNotFoundError) from None
        raise redacted_error(exc) from None
    except duckdb.Error as exc:
        raise redacted_error(exc) from None
    finally:
        if not keep_open:
            con.close()


def budgeted(sql: str, budget: int | None) -> str:
    """Limit result rows; blocking operators can still scan the entire input."""
    query = validate_query(sql)
    if budget is None:
        return query
    return f"SELECT * FROM (\n{query}\n) AS __qavf_budget LIMIT {int(budget) + 1}"


def check_budget(frame: pl.DataFrame, budget: int | None, what: str) -> pl.DataFrame:
    """Raise when ``frame`` exceeds the row budget (it holds at most budget + 1 rows)."""
    if budget is not None and frame.height > budget:
        raise SourceError(f"{what} has more than {budget:,} rows (row_budget)")
    return frame
