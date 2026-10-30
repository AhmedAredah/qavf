"""File discovery, exact source identity, and native or typed file readers."""

from __future__ import annotations

import glob
import hashlib
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import duckdb
import polars as pl

from qavf.redaction import redact, redact_text
from qavf.sources.base import (
    DataSource,
    RejectedRows,
    ScanPlan,
    SourceError,
    SourceIntegrityError,
    SourceNotFoundError,
    SourceParams,
    TypeViolations,
)
from qavf.sources.csv_probe import (
    REJECT_OPTIONS,
    REJECTED_SAMPLE,
    CsvProbe,
    CsvRead,
    HeaderReview,
    ReadFailure,
    TypedColumns,
    read_rejects,
)
from qavf.sources.errors import redacted_error
from qavf.sources.formats import (
    GLOB_PATTERN,
    READERS,
    FileFormat,
    infer_format,
    is_remote,
    suffixes_of,
)
from qavf.sources.identity import IdentitySession
from qavf.sources.identity import sha256_file as _sha256
from qavf.sources.native import budgeted, check_budget, connect, fetch
from qavf.sources.parameters import FileParams
from qavf.sources.sql import quote_identifier, sql_literal
from qavf.sources.storage import storage_options
from qavf.sources.typed import needs_typed_reader

#: Matched files listed by name in a glob fingerprint (all are always counted).
_LISTED_FILES = 1000


def file_fingerprint(path: Path, hash_contents: bool = False) -> dict[str, Any]:
    """Size, modification time (UTC) and optionally the SHA-256 of a local file."""
    stat = path.stat()
    info: dict[str, Any] = {
        # "content": the digest identifies the bytes; "metadata": only size and
        # modification time, which an edit can leave unchanged.
        "identity": "content" if hash_contents else "metadata",
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }
    if hash_contents:
        info["sha256"] = _sha256(path)
    return info


class FileSource(DataSource[FileParams]):
    """Tabular files with native SQL scans or lossless typed Polars plans."""

    kind: ClassVar[str] = "file"
    Params: ClassVar[type[SourceParams]] = FileParams
    _file_rows: tuple[tuple[str, int], ...] | None
    """Rows per file of a multi-file read, once counted (see :meth:`file_rows`)."""
    @property
    def _identity_session(self) -> IdentitySession:
        with self._state_lock:
            if not hasattr(self, "_identity_state"):
                self._identity_state = IdentitySession()
            return self._identity_state

    @contextmanager
    def verified_operation(self) -> Iterator[None]:
        """Keep full content checks at the outer boundaries of a verified operation."""
        with self._identity_session.operation(self.verify_identity):
            yield

    def identity_telemetry(self) -> dict[str, int]:
        # Do not wait on _state_lock: a timed-out reader may still be hashing.
        state = getattr(self, "_identity_state", None)
        return state.telemetry() if state is not None else {}

    def prepare_identity(self, hash_contents: bool = False) -> None:
        with self._state_lock:
            if hash_contents and not hasattr(self, "_content_hashes"):
                self.fingerprint(hash_contents=True)

    def _hash_file(self, path: Path) -> str:
        with self._identity_session.hashing() as record_bytes:
            return _sha256(path, record_bytes=record_bytes)

    def _location(self) -> str:
        path = self.params.path
        return path if is_remote(path) else str(self.env.resolve(path))

    def _files(self) -> tuple[Path, ...] | None:
        with self._state_lock:
            return self._discover_files()

    def _discover_files(self) -> tuple[Path, ...] | None:
        if hasattr(self, "_manifest"):
            return self._manifest
        location = self._location()
        if is_remote(location):
            self._manifest: tuple[Path, ...] | None = None
            return None
        paths: tuple[Path, ...]
        if Path(location).is_file():
            paths = (Path(location),)
        elif GLOB_PATTERN.search(location):
            paths = tuple(
                sorted(
                    {
                        Path(p)
                        for p in glob.iglob(location, recursive=True, include_hidden=True)
                        if Path(p).is_file()
                    }
                )
            )
        else:
            paths = ()
        if not paths:
            raise SourceNotFoundError(f"file not found: {redact_text(location)}")
        if self.params.min_files is not None and len(paths) < self.params.min_files:
            raise SourceError(
                f"{redact_text(location)} matched {len(paths)} file(s), "
                f"expected at least {self.params.min_files}"
            )
        self._manifest = paths
        self._manifest_stats = {path: self._file_stat(path) for path in paths}
        return paths

    @staticmethod
    def _file_stat(path: Path) -> tuple[int, int, int, int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino

    def _assert_unchanged(self) -> None:
        for path, original in getattr(self, "_manifest_stats", {}).items():
            try:
                current = self._file_stat(path)
            except OSError:
                raise SourceIntegrityError(
                    f"input file disappeared during validation: {redact_text(str(path))}"
                ) from None
            if current != original:
                raise SourceIntegrityError(
                    f"input file changed during validation: {redact_text(str(path))}"
                )

    def verify_identity(self) -> None:
        self._files()
        with self._state_lock:
            self._assert_unchanged()
            if self._identity_session.active:
                self._identity_session.coalesced()
                return
            for path, expected in getattr(self, "_content_hashes", {}).items():
                try:
                    actual = self._hash_file(path)
                except OSError:
                    raise SourceIntegrityError(
                        f"input file unavailable during validation: {redact_text(str(path))}"
                    ) from None
                if actual != expected:
                    raise SourceIntegrityError(
                        f"input file content changed during validation: {redact_text(str(path))}"
                    )
            self._assert_unchanged()

    def _needs_typed_parquet(self) -> bool:
        if (self.params.format or infer_format(self.params.path)) != "parquet":
            return False
        files = self._files()
        if files is None:
            # Remote Parquet retains its Arrow schema through Polars too; no extra
            # ungoverned metadata connection is opened just to choose a backend.
            return True
        with self._state_lock:
            if not hasattr(self, "_typed_parquet"):
                required = False
                union_by_name = bool(self.params.options.get("union_by_name"))
                selected = set(self.params.columns) if self.params.columns else None
                combined: dict[str, pl.DataType] = {}
                folded_names: dict[str, str] = {}
                try:
                    for index, path in enumerate(files):
                        # Polars can inspect its Int128 extension; Arrow cannot.
                        # Resolve metadata without expanding the pinned path
                        # again or copying the entire file through a Python
                        # file-like object on versions that read it eagerly.
                        schema = pl.scan_parquet(path, glob=False).collect_schema()
                        required |= any(
                            needs_typed_reader(dtype)
                            for name, dtype in schema.items()
                            if selected is None or name in selected
                        )
                        if index and not union_by_name and schema.keys() != combined.keys():
                            raise SourceError(
                                "Parquet files require matching column names unless "
                                "union_by_name is enabled; explicitly normalize the "
                                "input schemas before combining them"
                            )
                        for name, dtype in schema.items():
                            # SQL identifiers are case-insensitive. Distinct
                            # Parquet names must not be collapsed or renamed by
                            # a SQL union, including within a single file.
                            previous_name = folded_names.setdefault(name.casefold(), name)
                            required |= previous_name != name and (
                                selected is None or name in selected or previous_name in selected
                            )
                            if name in combined and combined[name] != dtype:
                                raise SourceError(
                                    "Parquet files require matching logical types for "
                                    f"column {name!r}; explicitly normalize the input "
                                    "schemas before combining them"
                                )
                            combined[name] = dtype
                except (OSError, pl.exceptions.PolarsError) as exc:
                    raise redacted_error(exc) from None
                self._typed_parquet = required
                # Every file must be checked even after selecting a typed reader:
                # a later file may have a conflicting schema. Keep only the union
                # needed by that reader, never all individual footer schemas.
                self._parquet_union = combined if required and union_by_name else None
            return self._typed_parquet

    def scan_polars(self) -> pl.LazyFrame | None:
        """Preserve exact logical types without materializing the source."""
        self.verify_identity()
        if not self._needs_typed_parquet():
            return None
        params = FileParams.model_validate(self.params.model_dump())
        allowed = {"union_by_name", "hive_partitioning", "filename"}
        unsupported = set(params.options) - allowed
        if unsupported:
            raise SourceError(
                "typed Parquet reads do not support reader option(s): "
                + ", ".join(sorted(unsupported))
            )
        files = self._files()
        location = self._location() if files is None else [str(path) for path in files]
        kwargs: dict[str, Any] = {"glob": files is None}
        if files is None:
            options = storage_options(self._location(), self.env.secrets)
            if options is not None:
                kwargs["storage_options"] = options
        if "hive_partitioning" in params.options:
            kwargs["hive_partitioning"] = params.options["hive_partitioning"]
        filename = params.options.get("filename", False)
        if filename:
            kwargs["include_file_paths"] = "filename" if filename is True else str(filename)
        if params.options.get("union_by_name"):
            if files is None:
                raise SourceError("remote typed Parquet union_by_name needs explicit sources")
            kwargs["schema"] = self._parquet_union
            kwargs["missing_columns"] = "insert"
        try:
            plan = pl.scan_parquet(location, **kwargs)
            if params.columns:
                plan = plan.select(params.columns)
        except pl.exceptions.PolarsError as exc:
            raise redacted_error(exc) from None
        self.verify_identity()
        self.load_notes = {
            "execution": "typed_parquet",
            "logical_types": "Polars schema preserved",
        }
        if files is not None:
            self.load_notes["files_matched"] = len(files)
        if params.columns:
            self.load_notes["columns_loaded"] = list(params.columns)
        if params.options:
            self.load_notes["reader_options"] = redact(params.options)
        return plan

    def _load_typed_parquet(self) -> pl.DataFrame:
        plan = self.scan_polars()
        assert plan is not None
        if self.params.row_budget is not None:
            plan = plan.limit(self.params.row_budget + 1)
        try:
            frame = plan.collect(engine="streaming")
        except pl.exceptions.PolarsError as exc:
            raise redacted_error(exc) from None
        self.verify_identity()
        return check_budget(frame, self.params.row_budget, redact_text(self._location()))

    def _plan(self) -> tuple[ScanPlan, dict[str, Any], bool, bool]:
        # Revalidate nested options, which cannot be made immutable by frozen=True alone.
        params = FileParams.model_validate(self.params.model_dump())
        files = self._files()
        location: str | list[str] = (
            self._location() if files is None else [str(path) for path in files]
        )
        notes: dict[str, Any] = {}
        if files is not None and GLOB_PATTERN.search(self._location()):
            notes["files_matched"] = len(files)
        fmt: FileFormat = params.format or infer_format(params.path)
        if files is not None and fmt != "excel":
            # These are already-expanded literal paths, not user glob patterns.
            location = [glob.escape(str(path)) for path in files]
        if fmt == "excel" and isinstance(location, list):
            if len(location) != 1:
                raise SourceError(
                    "Excel sources require one workbook; configure separate datasets "
                    "for multiple workbooks"
                )
            location = location[0]
        options = dict(params.options)
        count_rejects = fmt == "csv" and (
            options.get("ignore_errors") is True or options.get("store_rejects") is True
        )
        if count_rejects:
            options.pop("ignore_errors", None)
            options["store_rejects"] = True
        if fmt == "ndjson":
            options.setdefault("format", "newline_delimited")
        if (
            fmt == "csv"
            and suffixes_of(params.path)[0][-1:] == [".tsv"]
            and not {"delim", "sep", "delimiter"} & set(options)
        ):
            options["delim"] = "\t"
        trim_excel = fmt == "excel" and not ({"stop_at_empty", "range"} & set(options))
        if trim_excel:
            options["stop_at_empty"] = False
        typed = TypedColumns({})
        if fmt == "csv" and (params.text_columns or params.declared_columns):
            typed = self._read_as_text(location, options, notes)
        if params.options:
            notes["reader_options"] = redact(dict(params.options))
        if options != params.options:
            notes["effective_reader_options"] = redact(options)
        if params.columns:
            notes["columns_loaded"] = list(params.columns)
        args = "".join(
            f", {quote_identifier(key)} = {sql_literal(value)}" for key, value in options.items()
        )
        # SQL identifier resolution is case-insensitive, including quoted names.
        # Match decoded column names as string values instead: an unavailable
        # physical spelling must fail, never select a differently cased column.
        # A separate expression preserves the caller's projection order.
        select = (
            ", ".join(
                f"COLUMNS(__qavf_exact_column -> __qavf_exact_column = {sql_literal(c)})"
                for c in params.columns
            )
            if params.columns
            else "*"
        )
        if params.columns:
            loaded = set(params.columns)
            typed = TypedColumns({c: t for c, t in typed.columns.items() if c in loaded})
        with self._state_lock:
            self._planned_read = (fmt, location, dict(options))
            self._typed = typed
            if fmt == "csv":
                self._probe = self._header_probe(location, options).reading(options)
        if type(self) is FileSource and files is not None:
            # Only the built-in local reader embeds its pinned manifest. All
            # other source kinds retain their existing parameterized binding.
            sql = f"SELECT {select} FROM {READERS[fmt]}({sql_literal(location)}{args})"
            parameters: dict[str, Any] = {}
        else:
            sql = f"SELECT {select} FROM {READERS[fmt]}($path{args})"
            parameters = {"path": location}
        if typed:
            sql = typed.wrap(sql)  # converted in the same streaming scan
        plan = ScanPlan(sql, parameters) if parameters else ScanPlan(sql)
        return plan, notes, count_rejects, trim_excel

    def _read_as_text(
        self, location: str | list[str], options: dict[str, Any], notes: dict[str, Any]
    ) -> TypedColumns:
        """Read ``text_columns`` and ``declared_columns`` as VARCHAR; return the
        declared ones, by header name, for conversion in the scan.

        Columns the reader is told to type another way (explicit per-column types,
        an explicit ``columns`` layout, positional ``types``) are left alone.
        """
        key = next((k for k in ("types", "dtypes", "column_types") if k in options), "types")
        given = options.get(key) or {}
        if "columns" in options or not isinstance(given, dict):
            return TypedColumns({})
        text = {name.casefold() for name in self.params.text_columns}
        declared = {name.casefold(): kind for name, kind in self.params.declared_columns.items()}
        taken = {str(name).casefold() for name in given}
        matched = [
            name
            for name in self._header_probe(location, options).header()
            if (name.casefold() in text or name.casefold() in declared)
            and name.casefold() not in taken
        ]
        if matched and options.get("all_varchar") is not True:
            options[key] = {**dict.fromkeys(matched, "VARCHAR"), **given}
        as_text = [name for name in matched if name.casefold() in text]
        if as_text:
            notes["text_columns"] = as_text
        typed = TypedColumns(
            {name: declared[name.casefold()] for name in matched if name.casefold() in declared}
        )
        if typed:
            notes["declared_columns"] = list(typed.columns)
        return typed

    def _header_probe(self, location: str | list[str], options: dict[str, Any]) -> CsvProbe:
        """The probe that reads the header under the user's options, created once."""
        with self._state_lock:
            if getattr(self, "_header_reader", None) is None:
                self._header_reader = CsvProbe(self.env, CsvRead(location, dict(options)))
            return self._header_reader

    def _csv_probe(self) -> CsvProbe | None:
        """The probe of the read as planned; ``None`` before planning or for non-CSV."""
        with self._state_lock:
            return getattr(self, "_probe", None)

    def type_violations(self, context: Sequence[str] = ()) -> dict[str, TypeViolations]:
        """Values the typed read could not convert, found once per source."""
        with self._state_lock:
            typed = getattr(self, "_typed", None)
            cached = getattr(self, "_violations", None)
        probe = self._csv_probe()
        if not typed or probe is None:
            return {}
        if cached is None:
            cached = typed.violations(probe, context)
            with self._state_lock:
                self._violations = cached
        return cached

    def file_rows(self) -> tuple[tuple[str, int], ...] | None:
        """Rows per file of a multi-file read, counted once, in reading order."""
        files = self._files()
        if files is None or len(files) < 2:
            return None
        with self._state_lock:
            if hasattr(self, "_file_rows"):
                return self._file_rows
        self._plan()
        with self._state_lock:
            fmt, location, options = self._planned_read
        counted = self._count_file_rows(fmt, location, options)
        by_path = {_same_file(name): rows for name, rows in counted}
        located = tuple((str(path), by_path.get(_same_file(str(path)), 0)) for path in files)
        if sum(rows for _, rows in located) != sum(rows for _, rows in counted):
            located = ()  # names did not match: leave evidence unlocated, never misplaced
        with self._state_lock:
            self._file_rows = located or None
            return self._file_rows

    def _count_file_rows(
        self, fmt: FileFormat, location: str | list[str], options: dict[str, Any]
    ) -> list[tuple[str, int]]:
        probe = {k: v for k, v in options.items() if k not in REJECT_OPTIONS}
        probe["filename"] = "__qavf_file"
        args = "".join(f", {quote_identifier(k)} = {sql_literal(v)}" for k, v in probe.items())
        con = connect(self.env)
        try:
            rows = con.execute(
                f"SELECT __qavf_file, count(*) FROM {READERS[fmt]}($path{args}) "
                "GROUP BY __qavf_file",
                {"path": location},
            ).fetchall()
        except duckdb.Error as exc:
            raise redacted_error(exc) from None
        finally:
            con.close()
        return [(str(name), int(count)) for name, count in rows]

    def rejected_rows(self) -> RejectedRows | None:
        with self._state_lock:
            return getattr(self, "_rejected", None)

    def explain_failure(self, error: SourceError) -> SourceError | None:
        """Name the Excel cell or locate the CSV lines behind a read error.

        CSV lines are found once, with one bounded pass that keeps no data; clean
        reads never pay for it.
        """
        failure = ReadFailure(self.describe(), error)
        explained = failure.excel_cell()
        if explained is not None or not failure.is_csv() or not self._is_csv():
            return explained
        explained = failure.undetected_layout()
        probe = self._csv_probe()
        if explained is not None or probe is None:
            return explained
        with self._state_lock:
            if not hasattr(self, "_diagnosis"):
                self._diagnosis = probe.unreadable()
            rejected: RejectedRows | None = self._diagnosis
        return failure.located(rejected) if rejected is not None and rejected.sample else None

    def header_problems(self) -> list[dict[str, Any]]:
        """Duplicate, blank or unrecognized CSV header names (see :class:`HeaderReview`)."""
        probe = self._csv_probe()
        if probe is None or probe.read.options.get("header") is False:
            return []
        with self._state_lock:
            header = getattr(self, "_header_reader", probe)
        return HeaderReview(header.header()).problems(probe)

    def _is_csv(self) -> bool:
        try:
            return (self.params.format or infer_format(self.params.path)) == "csv"
        except ValueError:
            return False

    def scan(self) -> ScanPlan | None:
        self.verify_identity()
        if self._needs_typed_parquet():
            return None
        if (self.params.format or infer_format(self.params.path)) == "excel":
            return None  # per-workbook blank-row handling requires the bounded load path
        plan, notes, count_rejects, trim_excel = self._plan()
        if count_rejects or trim_excel:
            return None
        self.load_notes = notes
        return plan

    def load(self) -> pl.DataFrame:
        self.verify_identity()
        if self._needs_typed_parquet():
            return self._load_typed_parquet()
        files = self._files()
        if (
            (self.params.format or infer_format(self.params.path)) == "excel"
            and files
            and len(files) > 1
        ):
            frames: list[pl.DataFrame] = []
            total = 0
            for path in files:
                values = self.params.model_dump()
                values.update(path=str(path), min_files=None)
                if self.params.row_budget is not None:
                    values["row_budget"] = max(1, self.params.row_budget - total)
                child = FileSource(FileParams.model_validate(values), self.env)
                frame = child.load()
                total += frame.height
                if self.params.row_budget is not None and total > self.params.row_budget:
                    raise SourceError(
                        f"Excel source has more than {self.params.row_budget:,} rows (row_budget)"
                    )
                frames.append(frame)
            self._assert_unchanged()
            self.load_notes = {**child.load_notes, "files_matched": len(files)}
            try:
                return pl.concat(frames, how="vertical", rechunk=False)
            except pl.exceptions.PolarsError as exc:
                raise redacted_error(exc) from None
        plan, notes, count_rejects, trim_excel = self._plan()
        con = connect(self.env)
        try:
            query = plan.sql
            if trim_excel:
                # Trim inside DuckDB before applying the output-row cap. Trimming
                # a capped prefix would conceal valid rows after an interior gap.
                columns = con.sql(query, params=dict(plan.parameters)).columns
                row_name = "__qavf_excel_row"
                while row_name in columns:
                    row_name += "_"
                row_id = quote_identifier(row_name)
                selected = ", ".join(quote_identifier(name) for name in columns)
                filled = " OR ".join(f"{quote_identifier(name)} IS NOT NULL" for name in columns)
                query = (
                    f"WITH __qavf_sheet AS (SELECT *, row_number() OVER () AS {row_id} "
                    f"FROM ({query}) AS sheet) SELECT {selected} FROM __qavf_sheet "
                    f"WHERE {row_id} <= (SELECT max({row_id}) FROM __qavf_sheet WHERE {filled})"
                )
            frame = fetch(
                con, budgeted(query, self.params.row_budget), plan.parameters, keep_open=True
            )
            check_budget(frame, self.params.row_budget, redact_text(self._location()))
            if count_rejects:
                rejected = read_rejects(
                    con,
                    str(self.params.options.get("rejects_table", "reject_errors")),
                    str(self.params.options.get("rejects_scan", "reject_scans")),
                    int(self.params.options.get("rejects_limit", 0)),
                    REJECTED_SAMPLE,
                )
                with self._state_lock:
                    self._rejected = rejected
                if rejected.exact:
                    notes["rejected_rows"] = rejected.count
                else:
                    notes["rejected_rows_lower_bound"] = rejected.count
                notes["rejected_rows_exact"] = rejected.exact
            self._assert_unchanged()
            self.load_notes = notes
            return frame
        except duckdb.Error as exc:
            raise redacted_error(exc) from None
        finally:
            con.close()

    def describe(self) -> str:
        return self._location()

    def fingerprint(self, hash_contents: bool = False) -> dict[str, Any] | None:
        # Establish one identity atomically. Concurrent first readers must not
        # hash different bytes and overwrite each other's accepted digest.
        with self._state_lock:
            return self._fingerprint_locked(hash_contents)

    def _fingerprint_locked(self, hash_contents: bool) -> dict[str, Any] | None:
        files = self._files()
        if files is None:
            return None
        self.verify_identity()
        if not GLOB_PATTERN.search(self._location()):
            result = self._file_fingerprint(files[0], hash_contents)
            self._assert_unchanged()
            if hash_contents:
                with self._state_lock:
                    self._content_hashes = {files[0]: result["sha256"]}
            return result
        listing = hashlib.sha256()
        total, latest = 0, None
        hashes = {}
        for path in files:
            info = self._file_fingerprint(path, hash_contents)
            if hash_contents:
                hashes[path] = info["sha256"]
            total += info["bytes"]
            latest = max(latest or info["modified"], info["modified"])
            fields = (str(path), str(info["bytes"]), info["modified"], info.get("sha256", ""))
            listing.update(("\t".join(fields) + "\n").encode("utf-8"))
        self._assert_unchanged()
        if hash_contents:
            with self._state_lock:
                self._content_hashes = hashes
        return {
            "files": len(files),
            "identity": "content" if hash_contents else "metadata",
            "bytes": total,
            "modified": latest,
            "listing_sha256": listing.hexdigest(),
            "paths": [str(path) for path in files[:_LISTED_FILES]],
            "paths_truncated": len(files) > _LISTED_FILES,
        }

    def _file_fingerprint(self, path: Path, hash_contents: bool) -> dict[str, Any]:
        """Reuse a digest verified here or by the enclosing operation's boundaries."""
        expected = getattr(self, "_content_hashes", {}).get(path)
        if hash_contents and expected is not None:
            # The standalone call verified every stored digest. Within an
            # operation, its outer exit verifies again before publishing a result.
            return {**file_fingerprint(path), "identity": "content", "sha256": expected}
        info = file_fingerprint(path)
        if hash_contents:
            info.update(identity="content", sha256=self._hash_file(path))
        return info


def _same_file(path: str) -> str:
    """A path in one canonical spelling, to match the names a reader reports."""
    return os.path.normcase(os.path.abspath(path))
