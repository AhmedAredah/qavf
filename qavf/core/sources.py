"""DuckDB-owned snapshots over files and zero-copy Arrow-compatible inputs."""

from __future__ import annotations

import functools
import glob
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import duckdb

from qavf.bounded_batches import BoundedBatches
from qavf.core.admission import (
    FileIdentity,
    SourceNames,
    SourceProjection,
    arrow_batch,
    arrow_schema,
    eager_columns,
    eager_rows,
)
from qavf.duckdb_results import settle
from qavf.inputs.descriptors import BatchSource, File, Loader, Query
from qavf.optional import unsupported
from qavf.redaction import redact_text
from qavf.run_control import CANCELLED_MESSAGE
from qavf.sources.base import SourceError, SourceNotFoundError
from qavf.sources.errors import redacted_error, verify_postcondition
from qavf.sources.execution import RetryPolicy, RetryWait
from qavf.sources.formats import READERS, infer_format, is_remote
from qavf.sources.local_files import discover_local_files
from qavf.sources.sql import normalize_sql_paths
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    table: str
    columns: tuple[str, ...]
    types: dict[str, str]
    rows: int
    row_column: str
    original_columns: tuple[str, ...] = ()
    original_types: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    owned_relations: tuple[tuple[str, str], ...] = ()
    reader: str | None = None
    """The file format read, for file sources."""
    files: tuple[str, ...] = ()
    """The files of a multi-file read, in reading order."""
    file_count_sql: str | None = None
    """Counts the rows of each file of a multi-file read, when evidence needs it."""


def evidence_row_name(columns: Sequence[str]) -> str:
    """The row-position column, named ``_row`` as evidence documents it.

    A source column already called ``_row`` keeps its name; the position is then
    ``qavf__row``, as on the Polars backend.
    """
    preferred = "_row" if "_row" not in {c.casefold() for c in columns} else "qavf__row"
    return SourceNames.fresh(preferred, columns)


class InputReader:
    """Own QAVF snapshots; borrowed frames and connections remain caller-owned."""

    def __init__(self, connection, settings, projections=None, query_resolver=None, control=None):
        self.connection = connection
        self.settings = settings
        self.projections = projections or {}
        self.query_resolver = query_resolver
        self.control = control
        """The run's :class:`~qavf.run_control.RunControl`: a cancelled run starts no
        further attempt and does not wait out a retry delay."""
        self._identities = {}

    def _check_cancelled(self):
        if self.control is not None and self.control.cancelled:
            raise SourceError(CANCELLED_MESSAGE)

    def _pause(self, seconds):
        """Wait before a retry, returning as soon as the run is cancelled."""
        signal = None if self.control is None else self.control.token
        RetryWait(signal, self._check_cancelled)(seconds)

    def load(self, name, value, spec) -> LoadedDataset:
        from qavf.settings import SourcePolicy

        source_policy = (
            {k: value[k] for k in SourcePolicy.model_fields if k in value}
            if isinstance(value, Mapping)
            else {}
        )
        value = self._normalize(value, spec)
        policy = SourcePolicy.model_validate(
            {
                **source_policy,
                **spec.effective_load_policy().model_dump(exclude_unset=True),
            }
        )
        identity = self._identity(value)
        retry = RetryPolicy(
            retries=policy.retries,
            delay_s=policy.retry_delay_s,
            safe=isinstance(value, File | Query)
            or (isinstance(value, Loader | BatchSource) and value.replayable and value.retry_safe),
        )
        for attempt in range(retry.retries + 1):
            self._check_cancelled()
            owned: list[tuple[str, str]] = []
            attempt_identity = identity
            try:
                current = value
                if isinstance(value, Loader):
                    current = self._normalize(value.function(**value.kwargs), spec)
                    attempt_identity = self._identity(current)
                loaded = self._load_once(name, current, spec, attempt_identity, owned, policy)
                if attempt_identity is not None:
                    attempt_identity.verify()
                    self._identities[loaded.table] = attempt_identity
                return loaded
            except BaseException as exc:
                self._drop(owned)
                if isinstance(exc, duckdb.Error):
                    exc = redacted_error(exc)
                if attempt_identity is not None:
                    verify_postcondition(attempt_identity.verify, exc)
                pause = retry.pause(exc, attempt)
                if pause is None:
                    raise exc from None
                self._pause(pause)
        raise AssertionError("unreachable retry state")

    def _identity(self, value):
        if isinstance(value, File) and not is_remote(os.fspath(value.path)):
            paths = discover_local_files(self.settings.resolve(value.path))
            if not paths:
                location = redact_text(str(self.settings.resolve(value.path)))
                raise SourceNotFoundError(f"file not found: {location}")
            return FileIdentity(paths, self.settings.report.hash_inputs)
        return None

    def _normalize(self, value, spec):
        if isinstance(value, str | os.PathLike):
            configured = dict(spec.source_config() or {})
            configured.update(type="file", path=os.fspath(value))
            value = configured
        if isinstance(value, Mapping):
            from qavf.sources.parameters import FileParams, QueryParams

            config = dict(value)
            kind = config.setdefault("type", "file")
            if kind == "file":
                # Validate the complete public source model before adapting it.
                params = FileParams.model_validate(config)
                expected_types = spec._declared_types(config)
                if any(
                    name not in expected_types or kind.model_dump() != expected_types[name]
                    for name, kind in params.declared_columns.items()
                ):
                    unsupported("source declared_columns without a matching validation schema")
                allowed = File.__dataclass_fields__.keys()
                value = File(**{k: v for k, v in params.model_dump().items() if k in allowed})
            elif kind == "query":
                QueryParams.model_validate(
                    {k: v for k, v in config.items() if k not in {"connection", "parameters"}}
                )
                query = Query(
                    config["sql"],
                    connection=config.get("connection"),
                    parameters=config.get("parameters", {}),
                )
                if isinstance(query.connection, duckdb.DuckDBPyConnection):
                    if config.get("connections"):
                        raise ValueError(
                            "a borrowed query connection cannot request named attachments"
                        )
                    return query
                if self.query_resolver is not None:
                    return self.query_resolver(config, spec)
                if config.get("connections"):
                    unsupported("named query connections")
                value = query
            elif self.query_resolver is not None:
                return self.query_resolver(config, spec)
            else:
                unsupported(f"source type {kind!r}")
        if isinstance(value, Query) and (
            value.connection is None or isinstance(value.connection, str)
        ):
            if self.query_resolver is not None:
                value = self.query_resolver(value, spec)
            elif isinstance(value.connection, str):
                unsupported("named query connections")
        return value

    def _load_once(self, name, value, spec, identity, owned, policy):
        con = self.connection
        table = "__qavf_input_" + uuid4().hex
        metadata: dict[str, Any] = {"loaded_at": datetime.now(UTC).isoformat()}
        source_options = source_path = source_format = None
        files: tuple[str, ...] = ()
        file_count_sql = None
        budget = policy.row_budget
        try:
            if isinstance(value, BatchSource):
                return self._batches(name, value, spec, table, owned, metadata, policy)
            if isinstance(value, File):
                if value.max_rejected_rows or any(
                    (value.options or {}).get(key) for key in ("ignore_errors", "store_rejects")
                ):
                    unsupported("lossy file readers and rejection capture")
                path = os.fspath(value.path)
                if not is_remote(path):
                    path = str(self.settings.resolve(path))
                matched = identity.manifest.paths if identity else discover_local_files(path)
                if value.min_files is not None:
                    if matched is None:
                        unsupported("remote file manifest cardinality")
                    if len(matched) < value.min_files:
                        raise SourceError(
                            f"Expected at least {value.min_files} files, found {len(matched)}"
                        )
                fmt = value.format or infer_format(path)
                if fmt == "excel":
                    unsupported("Excel input")
                source_path, source_options, source_format = path, dict(value.options or {}), fmt
                if fmt == "csv":
                    from qavf.sources.csv_diagnostics import CsvPolicy

                    source_options = CsvPolicy.options(source_options, path)
                    source_options = self._csv_options(value, spec, path, matched, source_options)
                if fmt == "ndjson":
                    source_options.setdefault("format", "newline_delimited")
                locations = [glob.escape(str(p)) for p in matched] if matched is not None else path
                if fmt == "parquet":
                    self._parquet_precision(locations)
                sql = self._reader_sql(fmt, locations, source_options)
                if matched is not None and len(matched) > 1:
                    files = tuple(str(p) for p in matched)
                    counted = self._reader_sql(
                        fmt, locations, {**source_options, "filename": "__qavf_file"}
                    )
                    file_count_sql = (
                        f"SELECT __qavf_file, count(*) FROM ({counted}) files GROUP BY 1"
                    )
                desc = con.execute(f"DESCRIBE {sql}").fetchall()
                if fmt == "csv":
                    self._csv_header(locations, source_options, [r[0] for r in desc])
                if value.columns:
                    absent = set(value.columns) - {row[0] for row in desc}
                    if absent:
                        raise SourceError(
                            f"Requested exact column spelling is unavailable: {sorted(absent)!r}"
                        )
                    sql = f"SELECT {', '.join(qi(c) for c in value.columns)} FROM ({sql})"
                    by_name = {row[0]: row for row in desc}
                    desc = [by_name[c] for c in value.columns]
                metadata.update(source=path, effective_options=source_options)
                if identity:
                    metadata["fingerprint"] = identity.fingerprint
            elif isinstance(value, Query):
                query, parameters = normalize_sql_paths(
                    value.sql,
                    self.settings.base_dir,
                    value.parameters,
                    known_tables=self.settings.datasets,
                )
                borrowed = value.connection
                if isinstance(borrowed, duckdb.DuckDBPyConnection) and borrowed is not con:
                    # Limit before crossing the connection boundary; never publish a prefix.
                    query = self._bounded(query.rstrip().removesuffix(";"), budget)
                    relation = borrowed.sql(query, params=dict(parameters))
                    SourceNames.validate(relation.columns)
                    desc = list(zip(relation.columns, map(str, relation.types), strict=True))
                    projected = self._projection(name, spec, desc)
                    if self.projections.get(name) is not None:
                        fields = ", ".join(qi(source) for source, _ in projected.selected)
                        relation = relation.project(fields or "1 AS __qavf_empty_projection")
                    metadata["source"] = "query"
                    return self._stream_borrowed(
                        name, relation, spec, desc, table, owned, metadata, budget
                    )
                else:
                    settle(con)
                    rel = con.sql(query, params=parameters)
                    SourceNames.validate(rel.columns)
                    rel.create_view(table + "_query")
                    owned.append(("VIEW", table + "_query"))
                    sql = f"SELECT * FROM {qi(table + '_query')}"
                    desc = con.execute(f"DESCRIBE {sql}").fetchall()
                metadata["source"] = "query"
            else:
                source_kind = type(value).__name__
                columns = eager_columns(value)
                if columns is not None:
                    SourceNames.validate(columns)
                self._check_budget(name, eager_rows(value), budget)
                from qavf.core.temporal_admission import TemporalAdmission

                value, temporal = TemporalAdmission(
                    spec, self.settings.engine.resolved_canonical_columns()
                ).eager(value, budget)
                if temporal:
                    metadata["temporal_normalization"] = temporal
                    self._check_budget(name, eager_rows(value), budget)
                try:
                    settle(con)
                    con.register(table + "_arrow", value)
                except (duckdb.Error, TypeError) as exc:
                    raise TypeError(
                        "input must be a file, Query, Arrow table/record batch, or pandas frame"
                    ) from exc
                owned.append(("registered", table + "_arrow"))
                sql = f"SELECT * FROM {qi(table + '_arrow')}"
                desc = con.execute(f"DESCRIBE {sql}").fetchall()
                metadata["source"] = source_kind
            projection = self._projection(name, spec, desc)
            row_column = evidence_row_name(projection.original_columns)
            columns_sql = "".join(f", {qi(a)} AS {qi(b)}" for a, b in projection.selected)
            sql = self._bounded(sql, budget)
            # A materialized position never resolves to a user column called rowid.
            con.execute(
                f"CREATE TEMP TABLE {qi(table)} AS SELECT "
                f"row_number() OVER () - 1 AS {qi(row_column)}{columns_sql} FROM ({sql})"
            )
            owned.append(("TABLE", table))
            rows = con.execute(f"SELECT count(*) FROM {qi(table)}").fetchone()[0]
            self._check_budget(name, rows, budget)
            inputs = [entry for entry in owned if entry != ("TABLE", table)]
            self._drop(inputs)
            owned[:] = [("TABLE", table)]
            loaded = self._result(name, table, projection, desc, rows, row_column, metadata, owned)
            return replace(loaded, reader=source_format, files=files, file_count_sql=file_count_sql)
        except duckdb.Error as exc:
            if source_format == "csv" and source_path is not None:
                from qavf.sources.csv_diagnostics import CsvPolicy

                issue = CsvPolicy.diagnose(source_path, source_options or {})
                if issue is not None:
                    raise SourceError(issue.message(source_path)) from None
            raise redacted_error(exc) from None

    def _batches(self, name, value, spec, table, owned, metadata, policy):
        import pyarrow as pa

        from qavf.core.temporal_admission import TemporalAdmission

        declared = arrow_schema(value.schema)
        temporal_policy = TemporalAdmission(spec, self.settings.engine.resolved_canonical_columns())
        native_schema, temporal = temporal_policy.schema(declared)
        if temporal:
            metadata["temporal_normalization"] = temporal
        legacy_schema = not isinstance(value.schema, pa.Schema)
        empty = pa.Table.from_batches([], schema=native_schema)
        registration = table + "_batch"
        settle(self.connection)
        self.connection.register(registration, empty)
        owned.append(("registered", registration))
        desc = self.connection.execute(f"DESCRIBE {qi(registration)}").fetchall()
        projection = self._projection(name, spec, desc)
        row_column = evidence_row_name(projection.original_columns)
        selected = "".join(f", {qi(a)} AS {qi(b)}" for a, b in projection.selected)
        self.connection.execute(
            f"CREATE TEMP TABLE {qi(table)} AS SELECT 0::BIGINT AS {qi(row_column)}"
            f"{selected} FROM {qi(registration)}"
        )
        owned.append(("TABLE", table))
        rows = 0
        budget = policy.row_budget
        failure = None
        manager = value.factory(**value.kwargs)
        if not callable(getattr(manager, "__enter__", None)) or not callable(
            getattr(manager, "__exit__", None)
        ):
            close = getattr(manager, "close", None)
            if callable(close):
                close()
            raise SourceError("BatchSource.factory must return an owned context manager")
        with manager as batches:
            try:
                for batch in batches:
                    frame = arrow_batch(batch)
                    if legacy_schema:
                        from qavf.optional import require_polars

                        frame = (
                            require_polars("BatchSource with Polars dtype declarations")
                            .from_arrow(frame)
                            .to_arrow()
                        )
                    if not frame.schema.equals(declared, check_metadata=False):
                        raise SourceError("source batch schema/order differs from its declaration")
                    self._check_budget(name, rows + frame.num_rows, budget)
                    if temporal:
                        frame = frame.cast(native_schema, safe=True)
                    settle(self.connection)
                    self.connection.register(registration, frame)
                    self.connection.execute(
                        f"INSERT INTO {qi(table)} SELECT row_number() OVER () - 1 + {rows} "
                        f"AS {qi(row_column)}{selected} FROM {qi(registration)}"
                    )
                    rows += frame.num_rows
            except BaseException as exc:
                # A context manager may not suppress an admission error and expose a prefix.
                failure = exc
                raise
        if failure is not None:
            raise failure
        settle(self.connection)
        self.connection.unregister(registration)
        owned.remove(("registered", registration))
        metadata["source"] = "BatchSource"
        return self._result(name, table, projection, desc, rows, row_column, metadata, owned)

    def _stream_borrowed(self, name, relation, spec, desc, table, owned, metadata, budget):
        """Copy a borrowed query into the owned table one bounded Arrow batch at a time.

        Only one batch crosses the connection boundary at once: each is registered,
        inserted and unregistered before the next is read. The caller's connection is
        only read; an early end (row budget, cancellation, failure) closes the reader
        so it holds no unfinished result. ``desc`` is the query's own schema, which
        the projection and reported types describe; stored types are those of the
        Arrow bridge, as for a complete copy.
        """
        import pyarrow as pa

        con = self.connection
        registration = table + "_stream"
        admit = functools.partial(self._admit_rows, name, budget)
        with BoundedBatches(relation, admit, self._check_cancelled) as stream:
            settle(con)
            con.register(registration, stream.schema.empty_table())
            owned.append(("registered", registration))
            projection = self._projection(name, spec, desc)
            row_column = evidence_row_name(projection.original_columns)
            selected = "".join(f", {qi(a)} AS {qi(b)}" for a, b in projection.selected)
            con.execute(
                f"CREATE TEMP TABLE {qi(table)} AS SELECT 0::BIGINT AS {qi(row_column)}"
                f"{selected} FROM {qi(registration)}"
            )
            owned.append(("TABLE", table))
            settle(con)
            con.unregister(registration)
            rows = 0
            for batch in stream:
                settle(con)
                con.register(registration, pa.Table.from_batches([batch]))
                con.execute(
                    f"INSERT INTO {qi(table)} SELECT row_number() OVER () - 1 + {rows} "
                    f"AS {qi(row_column)}{selected} FROM {qi(registration)}"
                )
                settle(con)
                con.unregister(registration)
                rows += batch.num_rows
            batches = stream.batches
        owned.remove(("registered", registration))
        metadata["transfer"] = {
            "kind": "bounded_arrow_batches",
            "batch_rows": BoundedBatches.BATCH_ROWS,
            "batches": batches,
        }
        return self._result(name, table, projection, desc, rows, row_column, metadata, owned)

    @classmethod
    def _admit_rows(cls, name, budget, rows):
        cls._check_budget(name, rows, budget)

    def _projection(self, name, spec, desc):
        return SourceProjection(
            [r[0] for r in desc],
            spec.rename,
            self.settings.engine.resolved_canonical_columns(),
            self.projections.get(name),
        )

    @staticmethod
    def _result(name, table, projection, desc, rows, row_column, metadata, owned):
        types = {projection.renamed[r[0]]: r[1] for r in desc}
        columns = tuple(target for _, target in projection.selected)
        metadata["renamed"] = {a: b for a, b in projection.renamed.items() if a != b}
        metadata["retained_columns"] = list(columns)
        original_types = dict(types)
        for source, note in metadata.get("temporal_normalization", {}).items():
            original_types[projection.renamed[source]] = note["source_type"]
        return LoadedDataset(
            name,
            table,
            columns,
            {c: types[c] for c in columns},
            rows,
            row_column,
            projection.original_columns,
            original_types,
            metadata,
            tuple(owned),
        )

    @staticmethod
    def _bounded(sql, budget):
        return sql if budget is None else f"SELECT * FROM ({sql}) LIMIT {budget + 1}"

    @staticmethod
    def _check_budget(name, rows, budget):
        if budget is not None and rows is not None and rows > budget:
            raise SourceError(f"dataset {name!r} has more than {budget} rows (row_budget={budget})")

    @staticmethod
    def _reader_sql(fmt, locations, options):
        return (
            f"SELECT * FROM {READERS[fmt]}({lit(locations)}"
            + "".join(f", {qi(key)}={lit(item)}" for key, item in options.items())
            + ")"
        )

    def _parquet_precision(self, locations):
        lossy = self.connection.execute(
            f"SELECT name FROM parquet_schema({lit(locations)}) "
            "WHERE contains(logical_type, 'isAdjustedToUTC=1') "
            "AND contains(logical_type, 'NANOS=NanoSeconds') LIMIT 1"
        ).fetchone()
        if lossy:
            raise SourceError(
                "timezone-aware nanosecond Parquet timestamps would lose precision in DuckDB; "
                "load with PyArrow and provide an explicit datetime/timezone contract, "
                "or use an owned Arrow BatchSource"
            )

    def _csv_header(self, locations, options, names):
        """A raw first-record probe only when the reader may have invented names."""
        if options.get("header") is False:
            return
        if isinstance(locations, list) and len(locations) > 1:
            for location in locations:
                columns = [
                    row[0]
                    for row in self.connection.execute(
                        "DESCRIBE " + self._reader_sql("csv", location, options)
                    ).fetchall()
                ]
                self._csv_header(location, options, columns)
            return
        folded = {name.casefold() for name in names}
        suspicious = any(
            re.fullmatch(r"column\d+", name, re.IGNORECASE)
            or ((match := re.fullmatch(r"(.+)_\d+", name)) and match[1].casefold() in folded)
            for name in names
        )
        if not suspicious:
            return
        raw_options = {
            key: value
            for key, value in options.items()
            if key not in {"header", "types", "dtypes", "column_types", "columns"}
        }
        raw_options.update(header=False, all_varchar=True)
        raw = self.connection.execute(
            self._reader_sql("csv", locations, raw_options) + " LIMIT 1"
        ).fetchone()
        if raw is None or list(raw) == list(names):
            return
        if len(raw) == len(names):
            SourceNames.validate(raw)

    def _csv_options(self, file, spec, path, matched, options):
        """Keep governed text lossless and retain ordinary reader hints elsewhere."""
        from qavf.sources.csv_diagnostics import CsvPolicy

        original = {new: old for old, new in spec.rename.items()}
        columns = spec.schema_spec.columns if spec.schema_spec else {}
        governed = [
            original.get(c, c) for c, cs in columns.items() if cs.dtype is not None or cs.na_values
        ]
        declared = list(dict.fromkeys([*file.text_columns, *governed]))
        # A declared column absent from the file is reported by the schema check;
        # typing it here would make the reader reject the whole file.
        if declared:
            locations = [glob.escape(str(p)) for p in matched] if matched is not None else path
            present = {
                row[0].casefold()
                for row in self.connection.execute(
                    "DESCRIBE " + self._reader_sql("csv", locations, options)
                ).fetchall()
            }
            declared = [name for name in declared if name.casefold() in present]
        options = CsvPolicy.protect_columns(options, governed)
        if declared and not options.get("all_varchar") and "columns" not in options:
            existing = options.get("types", options.get("dtypes", options.get("column_types", {})))
            if isinstance(existing, dict):
                types = dict(existing)
                known = {name.casefold() for name in types}
                for name in declared:
                    if name.casefold() not in known:
                        types[name] = "VARCHAR"
                        known.add(name.casefold())
                options.pop("dtypes", None)
                options.pop("column_types", None)
                options["types"] = types
        spec._preserve_applicability(
            {"type": "file", "path": path, "format": "csv", "options": options}
        )
        return options

    def verify(self, loaded):
        """Verify a pinned source at the report boundary, before releasing it."""
        identity = self._identities.get(loaded.table)
        if identity is not None:
            identity.verify()

    def release(self, loaded):
        """Drop only owned input handles; callers remove dependent check views first."""
        self._drop(loaded.owned_relations)

    def _drop(self, owned):
        for kind, name in reversed(owned):
            try:
                if kind == "registered":
                    settle(self.connection)
                    self.connection.unregister(name)
                else:
                    self.connection.execute(f"DROP {kind} IF EXISTS {qi(name)}")
            except duckdb.Error:
                pass
