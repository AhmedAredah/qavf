"""File discovery, exact source identity, and native or typed file readers."""

from __future__ import annotations

import glob
import hashlib
import re
from collections.abc import Iterator
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
    _header: list[str]
    """CSV column names, read once when ``text_columns`` are typed."""

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
        if fmt == "csv" and params.text_columns:
            self._read_as_text(location, options, notes)
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
        with self._state_lock:
            self._reader = (location, dict(options))
        if type(self) is FileSource and files is not None:
            # Only the built-in local reader embeds its pinned manifest. All
            # other source kinds retain their existing parameterized binding.
            sql = f"SELECT {select} FROM {READERS[fmt]}({sql_literal(location)}{args})"
            plan = ScanPlan(sql)
        else:
            sql = f"SELECT {select} FROM {READERS[fmt]}($path{args})"
            plan = ScanPlan(sql, {"path": location})
        return plan, notes, count_rejects, trim_excel

    def _read_as_text(
        self, location: str | list[str], options: dict[str, Any], notes: dict[str, Any]
    ) -> None:
        """Type ``text_columns`` as VARCHAR in ``options`` unless the reader is told
        the types another way (``all_varchar``, explicit ``columns``, positional
        ``types``). Explicit per-column types take precedence."""
        key = next((k for k in ("types", "dtypes", "column_types") if k in options), "types")
        given = options.get(key) or {}
        if options.get("all_varchar") is True or "columns" in options or not isinstance(
            given, dict
        ):
            return
        wanted = {name.casefold() for name in self.params.text_columns}
        taken = {str(name).casefold() for name in given}
        matched = [
            name
            for name in self._csv_header(location, options)
            if name.casefold() in wanted and name.casefold() not in taken
        ]
        if matched:
            options[key] = {**dict.fromkeys(matched, "VARCHAR"), **given}
            notes["text_columns"] = matched

    def _csv_header(self, location: str | list[str], options: dict[str, Any]) -> list[str]:
        """Column names under the reader options, read once without any rows.

        DuckDB rejects types for absent columns, so only present names are typed.
        A header that cannot be read leaves the types alone: the read itself then
        reports the problem.
        """
        with self._state_lock:
            if hasattr(self, "_header"):
                return list(self._header)
        args = _reader_args(options, drop=_REJECT_OPTIONS)
        try:
            con = connect(self.env)
            try:
                probe = f"SELECT * FROM {READERS['csv']}($path{args}) LIMIT 0"
                con.execute(probe, {"path": location})
                header = [str(column[0]) for column in con.description or ()]
            finally:
                con.close()
        except duckdb.Error:
            header = []
        with self._state_lock:
            self._header = header
        return list(header)

    def rejected_rows(self) -> RejectedRows | None:
        with self._state_lock:
            return getattr(self, "_rejected", None)

    def explain_failure(self, error: SourceError) -> SourceError | None:
        """Locate the CSV lines behind a read error, once, with one bounded pass.

        The pass records at most :data:`REJECTED_SAMPLE` failing lines per file
        and keeps no data. Clean reads never pay for it.
        """
        message = str(error)
        if not _CSV_ERROR.search(message) or not self._is_csv():
            return None
        if _SNIFF_ERROR.search(message):
            return SourceError(
                f"{redact_text(self.describe())}: the reader could not detect the CSV layout "
                "(delimiter, quoting, header) from its first rows, often because an early "
                "line has extra or missing fields. Set options {delim: ','} (and quote or "
                "header) so the line can be located."
            )
        with self._state_lock:
            planned = getattr(self, "_reader", None)
            if planned is None:
                return None
            if not hasattr(self, "_diagnosis"):
                self._diagnosis = self._diagnose(*planned)
            rejected: RejectedRows | None = self._diagnosis
        if rejected is None or not rejected.sample:
            return None
        return SourceError(_unreadable_message(redact_text(self.describe()), rejected))

    def header_problems(self) -> list[dict[str, Any]]:
        """Duplicate, blank or unrecognized CSV header names.

        Only names the reader synthesized (``v_1``, ``column3``) prompt a check, so
        a clean header costs nothing beyond the cached header read.
        """
        with self._state_lock:
            planned = getattr(self, "_reader", None)
        if planned is None or not self._is_csv() or planned[1].get("header") is False:
            return []
        location, options = planned
        names = self._csv_header(location, options)
        if not names:
            return []
        if len(names) == 1 and any(mark in names[0] for mark in ",;|\t"):
            return [
                {
                    "column": names[0],
                    "problem": "the header was read as one column: rows could not be split "
                    "into fields, usually because a line has extra or missing fields; set "
                    "options {delim: ...} to locate it",
                }
            ]
        if all(_GENERIC_NAME.fullmatch(name) for name in names):
            return [
                {
                    "column": None,
                    "problem": f"the header row was not recognized (columns read as "
                    f"{', '.join(names[:3])}{', …' if len(names) > 3 else ''}): an early line "
                    "may have extra or missing fields; set options {header: true} to locate it",
                }
            ]
        folded = {name.casefold() for name in names}
        if not any(
            _GENERIC_NAME.fullmatch(name)
            or ((match := _DEDUPED_NAME.fullmatch(name)) and match[1].casefold() in folded)
            for name in names
        ):
            return []
        raw = self._raw_header(location, options)
        if len(raw) != len(names):
            return []
        problems: list[dict[str, Any]] = []
        seen: set[str] = set()
        for original, name in zip(raw, names, strict=True):
            if original is None or not original.strip():
                problems.append({"column": name, "problem": f"blank header, read as {name!r}"})
            elif original.casefold() in seen:
                problems.append(
                    {"column": name, "problem": f"duplicate header {original!r}, read as {name!r}"}
                )
            if original is not None:
                seen.add(original.casefold())
        return problems

    def _is_csv(self) -> bool:
        try:
            return (self.params.format or infer_format(self.params.path)) == "csv"
        except ValueError:
            return False

    def _diagnose(self, location: str | list[str], options: dict[str, Any]) -> RejectedRows | None:
        probe = {
            key: value for key, value in options.items() if key not in _REJECT_OPTIONS
        }
        probe.update(store_rejects=True, rejects_limit=REJECTED_SAMPLE)
        try:
            con = connect(self.env)
            try:
                # count(COLUMNS(*)) converts every column: count(*) would skip them.
                con.execute(
                    f"SELECT count(COLUMNS(*)) FROM {READERS['csv']}($path{_reader_args(probe)})",
                    {"path": location},
                ).fetchall()  # the scan, and so the reject table, completes on fetch
                return _rejected_rows(
                    con, "reject_errors", "reject_scans", REJECTED_SAMPLE, REJECTED_SAMPLE
                )
            finally:
                con.close()
        except duckdb.Error:
            return None

    def _raw_header(self, location: str | list[str], options: dict[str, Any]) -> list[str | None]:
        """The first line's fields exactly as written, before the reader names them."""
        probe = {
            key: value
            for key, value in options.items()
            if key not in _REJECT_OPTIONS | {"header", "types", "dtypes", "column_types"}
        }
        probe.update(header=False, all_varchar=True)
        try:
            con = connect(self.env)
            try:
                row = con.execute(
                    f"SELECT * FROM {READERS['csv']}($path{_reader_args(probe)}) LIMIT 1",
                    {"path": location},
                ).fetchone()
            finally:
                con.close()
        except duckdb.Error:
            return []
        return [None if value is None else str(value) for value in row or ()]

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
                rejected = _rejected_rows(
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


REJECTED_SAMPLE = 100
"""Failing lines located per file, for messages and evidence."""
_REJECT_OPTIONS = frozenset(
    {"ignore_errors", "store_rejects", "rejects_table", "rejects_scan", "rejects_limit"}
)
_CSV_ERROR = re.compile(r"CSV Error|sniffing|CSV parsing dialect", re.IGNORECASE)
_SNIFF_ERROR = re.compile(r"sniffing|CSV parsing dialect", re.IGNORECASE)
_GENERIC_NAME = re.compile(r"column\d+")
_DEDUPED_NAME = re.compile(r"(.+)_\d+")
_ADVICE = {
    "CAST": (
        "The reader chose the column's type from the first rows: read the column as text "
        "with text_columns, list placeholders such as 'n/a' in options {nullstr: [...]}, "
        "or let the reader see every row with options {sample_size: -1}."
    ),
    "TOO MANY COLUMNS": (
        "The line has more fields than the header: quote values that contain the "
        "delimiter, or leave such lines out and have them reported with options "
        "{store_rejects: true} and acknowledge_lossy: true."
    ),
    "MISSING COLUMNS": (
        "The line has fewer fields than the header: fix the line, or leave such lines out "
        "and have them reported with options {store_rejects: true} and "
        "acknowledge_lossy: true."
    ),
    "UNQUOTED VALUE": "A value has a quote character inside an unquoted field: quote it.",
    "INVALID ENCODING": (
        "The file is not UTF-8: set options {encoding: latin-1}, or the file's encoding."
    ),
    "LINE SIZE OVER MAXIMUM": "Set options {max_line_size: ...} above the longest line.",
}


def _reader_args(options: dict[str, Any], drop: frozenset[str] = frozenset()) -> str:
    return "".join(
        f", {quote_identifier(key)} = {sql_literal(value)}"
        for key, value in options.items()
        if key not in drop
    )


def _rejected_rows(
    con: Any, errors: str, scans: str, limit: int, sample: int
) -> RejectedRows:
    """Count rejected lines and locate the first ``sample`` of them.

    A line with several errors counts once and shows its root error. With a
    per-file ``limit`` (0 for none), a file that reached it makes the count a
    lower bound.
    """
    e, s = quote_identifier(errors), quote_identifier(scans)
    per_file = con.execute(
        f"SELECT count(DISTINCT line) FROM {e} GROUP BY scan_id, file_id"
    ).fetchall()
    count = sum(int(row[0]) for row in per_file)
    exact = not limit or all(int(row[0]) < limit for row in per_file)
    rows = con.execute(
        f"SELECT s.file_path, e.line, e.column_name, e.error_type, e.csv_line "
        f"FROM {e} AS e JOIN {s} AS s USING (scan_id, file_id) "
        # A line's structural error (wrong field count, encoding) causes any type
        # error on the same line, so it is the one shown.
        f"QUALIFY row_number() OVER (PARTITION BY e.scan_id, e.file_id, e.line "
        f"ORDER BY e.error_type = 'CAST', e.column_idx NULLS LAST) = 1 "
        f"ORDER BY s.file_path, e.line LIMIT {int(sample)}"
    ).fetchall()
    located = tuple(
        {
            "file": redact_text(str(file)),
            "line": int(line),
            "column": column,
            "error": str(kind),
            "text": _clipped(text),
        }
        for file, line, column, kind, text in rows
    )
    return RejectedRows(count, exact, located)


def _clipped(text: object, width: int = 200) -> str:
    value = str(text or "").strip("\r\n")
    return value if len(value) <= width else value[: width - 1] + "…"


def _unreadable_message(source: str, rejected: RejectedRows) -> str:
    first = rejected.sample[0]
    files = {row["file"] for row in rejected.sample}
    count = f"{rejected.count:,}" if rejected.exact else f"at least {rejected.count:,}"
    rows = "row" if rejected.exact and rejected.count == 1 else "rows"
    where = f"line {first['line']:,}" + (f" of {first['file']}" if len(files) > 1 else "")
    column = f", column {first['column']!r}" if first["column"] else ""
    advice = _ADVICE.get(first["error"], "")
    return (
        f"{count} {rows} of {source} cannot be read. The first is {where}{column} "
        f"({first['error'].lower()}): {first['text']!r}. {advice}"
    ).rstrip(" .")  # callers end the sentence
