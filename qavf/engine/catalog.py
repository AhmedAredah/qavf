"""Dataset declarations and durable, data-free provenance for one run.

The catalog never loads, scans, hashes, caches, or releases input data. Execution
publishes observations after its policy checks; reporting can inspect them after
all data and connections have been released. The context shares this lock when
publishing cache entries and metadata atomically.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl

from qavf.audit_codec import freeze_evidence
from qavf.redaction import redact_text
from qavf.reports.models import DatasetInfo
from qavf.settings import DatasetSpec, check_canonical_columns
from qavf.sources.base import DataSource, SourceError


@dataclass(slots=True)
class _Observation:
    schema: dict[str, pl.DataType] | None = None
    rows: int | None = None
    renamed: dict[str, str] = field(default_factory=dict)
    fingerprint: dict[str, Any] | None = None
    fingerprint_recorded: bool = False
    load_seconds: float | None = None
    loaded_at: datetime | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    failure: SourceError | None = None
    identity_used: bool = False


class DatasetCatalog:
    """Own declarations and observations without retaining materialized frames."""

    def __init__(
        self,
        specs: Mapping[str, DatasetSpec],
        sources: Mapping[str, DataSource],
        provided_names: Iterable[str],
        canonical_columns: Iterable[str],
        lock: threading.RLock,
    ) -> None:
        self.specs = dict(specs)
        self.sources = dict(sources)
        self.provided_names = frozenset(provided_names)
        self._canonical = check_canonical_columns(canonical_columns)
        self._observations: dict[str, _Observation] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._lock = lock

    def _entry(self, name: str) -> _Observation:
        entry = self._observations.get(name)
        if entry is None:
            entry = self._observations[name] = _Observation()
        return entry

    def names(self, materialized: Iterable[str], lazy: Iterable[str]) -> list[str]:
        return sorted(set(materialized) | self.sources.keys() | set(lazy))

    def declared(self, accessible: Iterable[str]) -> list[str]:
        return sorted(set(accessible) | self.specs.keys() | self.provided_names)

    def failure(self, name: str) -> SourceError | None:
        with self._lock:
            entry = self._observations.get(name)
            return entry.failure if entry is not None else None

    def record_failure(self, name: str, failure: SourceError, *, replace: bool = True) -> None:
        with self._lock:
            entry = self._entry(name)
            if replace or entry.failure is None:
                entry.failure = failure

    def schema(self, name: str) -> dict[str, pl.DataType] | None:
        with self._lock:
            entry = self._observations.get(name)
            return dict(entry.schema) if entry is not None and entry.schema is not None else None

    def record_frame(self, name: str, frame: pl.DataFrame) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.schema, entry.rows = dict(frame.schema), frame.height

    def record_load(self, name: str, notes: Mapping[str, Any], seconds: float) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.notes = dict(freeze_evidence(notes))
            entry.loaded_at = datetime.now(UTC)
            entry.load_seconds = round(seconds, 6)

    def record_lazy(self, name: str, schema: Mapping[str, pl.DataType]) -> None:
        with self._lock:
            entry = self._entry(name)
            if entry.schema is None:
                entry.schema = dict(schema)
            if entry.loaded_at is None:
                entry.loaded_at = datetime.now(UTC)
            source = self.sources.get(name)
            entry.notes = {
                **freeze_evidence(source.load_notes if source is not None else {}),
                "execution": "typed_lazy_scan" if source is not None else "supplied_lazy_scan",
            }

    def record_native(self, name: str, schema: Mapping[str, pl.DataType], seconds: float) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.identity_used = True
            if entry.schema is None:
                entry.schema = dict(schema)
            if entry.loaded_at is None:
                entry.loaded_at = datetime.now(UTC)
            if entry.load_seconds is None:
                entry.load_seconds = round(seconds, 6)
            entry.notes = {
                **freeze_evidence(self.sources[name].load_notes),
                "execution": "native_scan",
                "row_count": "not materialized",
                "load_time_scope": (
                    "native schema binding; query execution is recorded in rule timing"
                ),
            }

    def record_native_plan(self, name: str) -> None:
        with self._lock:
            self._plans.setdefault(
                name,
                {"dataset": name, "execution": "native_scan", "projection_pushdown": True},
            )

    def record_snapshot(
        self, name: str, frame: pl.DataFrame, connection: str, members: list[str], started: datetime
    ) -> None:
        with self._lock:
            self.record_frame(name, frame)
            entry = self._entry(name)
            entry.identity_used = True
            entry.loaded_at = started
            notes = self.sources[name].load_notes
            entry.notes = {
                **freeze_evidence(notes),
                "snapshot": {
                    "connection": connection,
                    "datasets": list(members),
                    "started_at": started.isoformat(),
                },
                "retries_used": notes.get("retries_used", 0),
            }

    def has_fingerprint(self, name: str) -> bool:
        with self._lock:
            entry = self._observations.get(name)
            return entry is not None and entry.fingerprint_recorded

    def record_fingerprint(self, name: str, value: dict[str, Any] | None) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.fingerprint, entry.fingerprint_recorded = freeze_evidence(value), True

    def mark_identity_used(self, names: Iterable[str]) -> None:
        with self._lock:
            for name in names:
                self._entry(name).identity_used = True

    def identity_names(self) -> list[str]:
        with self._lock:
            return sorted(name for name, entry in self._observations.items() if entry.identity_used)

    def identity_pending(self, name: str) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.notes.update(
                identity_verified=False,
                identity_verification="pending: timed-out work remains active",
            )
            if entry.fingerprint is not None:
                entry.fingerprint = {"identity": "pending", "initial_identity": entry.fingerprint}

    def identity_failed(self, name: str, failure: SourceError) -> None:
        with self._lock:
            entry = self._entry(name)
            entry.failure = failure
            entry.notes["identity_verified"] = False
            if entry.fingerprint is not None:
                entry.fingerprint = {
                    "identity": "invalidated",
                    "initial_identity": entry.fingerprint,
                    "reason": str(failure),
                }

    def query_plans(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(plan) for plan in self._plans.values()]

    def describe(self) -> list[DatasetInfo]:
        with self._lock:
            result = []
            for name, entry in sorted(self._observations.items()):
                if entry.schema is None:
                    continue
                source = self.sources.get(name)
                counters = source.identity_telemetry() if source is not None else {}
                result.append(
                    DatasetInfo(
                        name=name,
                        rows=entry.rows,
                        columns=list(entry.schema),
                        source=redact_text(source.describe()) if source is not None else None,
                        dtypes={key: str(value) for key, value in entry.schema.items()},
                        renamed=dict(entry.renamed),
                        fingerprint=entry.fingerprint,
                        load_time_s=entry.load_seconds,
                        loaded_at=entry.loaded_at,
                        notes={**entry.notes, **({"identity_work": counters} if counters else {})},
                    )
                )
            return result

    def unused(self) -> list[str]:
        with self._lock:
            used = {
                name
                for name, entry in self._observations.items()
                if entry.schema is not None or entry.failure is not None
            }
            return sorted((self.sources.keys() | self.provided_names) - used - self._plans.keys())

    def unavailable(self, materialized: Iterable[str]) -> dict[str, str]:
        with self._lock:
            reasons = {
                name: redact_text(str(entry.failure))
                for name, entry in self._observations.items()
                if entry.failure is not None
            }
            available = set(materialized) | self.sources.keys() | self.provided_names
            for name in self.specs.keys() - available:
                entry = self._observations.get(name)
                if entry is None or entry.schema is None:
                    reasons[name] = (
                        f"dataset {name!r} is declared but has no source, and no in-memory "
                        f"frame was supplied"
                    )
            return dict(sorted(reasons.items()))

    def prepare(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        """Resolve declared column names and record only the resulting rename map."""
        spec = self.specs.get(name)
        explicit = spec.rename if spec is not None else {}
        matches: defaultdict[str, list[str]] = defaultdict(list)
        for column in frame.columns:
            target = self._canonical.get(column.lower())
            if target and column not in explicit:
                matches[target].append(column)
        ambiguous = {target: columns for target, columns in matches.items() if len(columns) > 1}
        if ambiguous:
            details = "; ".join(
                f"{', '.join(columns)} all match canonical column {target!r}"
                for target, columns in sorted(ambiguous.items())
            )
            raise ValueError(
                f"dataset {name!r}: ambiguous canonical column names: {details} "
                f"(case-insensitively). Rename all but one under 'datasets.{name}.rename'"
            )
        renames = {column: target for target, [column] in matches.items() if column != target}
        renames.update({old: new for old, new in explicit.items() if old in frame.columns})
        if not renames:
            return frame
        result = [renames.get(column, column) for column in frame.columns]
        clashes = sorted(column for column, count in Counter(result).items() if count > 1)
        if clashes:
            raise ValueError(
                f"dataset {name!r}: rename produces duplicate column(s) {', '.join(clashes)}; "
                f"adjust 'rename' (or 'canonical_columns') so each column keeps a unique name"
            )
        with self._lock:
            self._entry(name).renamed = renames
        return frame.rename(renames)
