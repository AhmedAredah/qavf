"""Source execution behind the data context: verified, retried source operations.

:class:`SourceExecution` runs every read of a configured or ad-hoc source under the
run's one source policy (:class:`~qavf.sources.execution.SourceOperations`: identity,
retries, cancellation, serialized access) and records what the read established in
the dataset catalog: that the source's identity was used, and its fingerprint for
provenance. The data context delegates its eager loads, lazy and native source
operations to it; :class:`LatestAttempt` owns the result of each retried attempt.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

from qavf.redaction import exception_details, redact_text
from qavf.sources.base import DataSource, SourceError, SourceIntegrityError
from qavf.sources.collection import collect_result
from qavf.sources.execution import SourceOperations

if TYPE_CHECKING:
    from qavf.engine.catalog import DatasetCatalog
    from qavf.engine.resources import RunResources

logger = logging.getLogger(__name__)


class _Closeable(Protocol):
    def close(self) -> None: ...


class LatestAttempt[T: _Closeable]:
    """Own one result per source attempt (an observation, a snapshot stage), closing
    every earlier attempt first."""

    def __init__(self, open_attempt: Callable[[], T]) -> None:
        self._open_attempt = open_attempt
        self._attempts: list[T] = []

    def __call__(self) -> T:
        """Discard earlier attempts and own a fresh one."""
        self.close()
        current = self._open_attempt()
        self._attempts.append(current)
        return current

    def close(self) -> None:
        """Release every attempt still owned."""
        for previous in self._attempts:
            previous.close()
        self._attempts.clear()


class SourceExecution:
    """Runs source operations under the run's policy and records their identity.

    Args:
        catalog: The run's dataset catalog (identity use and fingerprints).
        operations: The run's source policy.
        lock: The catalog lock publications are made under.
        resources: The run's resources (eager fallbacks are recorded on them).
        check_materialization: Admits a completed eager result against the memory target.
        hash_inputs: Fingerprints include a content digest of local files.
    """

    def __init__(
        self,
        catalog: DatasetCatalog,
        operations: SourceOperations,
        lock: threading.RLock,
        resources: RunResources,
        check_materialization: Callable[[str, pl.DataFrame], None],
        hash_inputs: bool,
    ) -> None:
        self._catalog = catalog
        self._operations = operations
        self._lock = lock
        self._resources = resources
        self._check_materialization = check_materialization
        self.hash_inputs = hash_inputs

    def run[T](self, names: Iterable[str], operation: Callable[[], T]) -> T:
        """``operation`` over the configured sources of ``names``, as one verified,
        retried operation; their fingerprints are recorded once it succeeds."""
        participants = {
            name: self._catalog.sources[name] for name in names if name in self._catalog.sources
        }
        verified = _VerifiedOperation(self, participants, operation)
        result = self._operations.run(participants.values(), verified)
        with self._lock:
            self._catalog.mark_identity_used(participants)
            for name, fingerprint in verified.fingerprints.items():
                self._catalog.record_fingerprint(name, fingerprint)
        return result

    def load(self, name: str, source: DataSource) -> pl.DataFrame:
        """Load ``source`` eagerly as dataset ``name`` (``<ad-hoc>`` for one that is
        not configured): eager and native paths use the same identity, retry and
        cancellation policy."""
        record_fingerprint = name in self._catalog.sources and not self._catalog.has_fingerprint(
            name
        )
        load = _EagerLoad(self, name, source, record_fingerprint)
        result = self._operations.run([source], load)
        with self._lock:
            if name in self._catalog.sources:
                self._catalog.mark_identity_used([name])
                if record_fingerprint:
                    self._catalog.record_fingerprint(name, load.fingerprint)
        return result

    def fingerprint(self, name: str, source: DataSource) -> dict[str, Any] | None:
        """The source's fingerprint; provenance must never break a run."""
        try:
            return source.fingerprint(hash_contents=self.hash_inputs)
        except SourceIntegrityError:
            raise
        except Exception as exc:
            logger.warning(
                "Could not fingerprint dataset %r: %s", redact_text(name), exception_details(exc)
            )
            return None

    def first_fingerprint(self, name: str, source: DataSource) -> dict[str, Any] | None:
        """Fingerprint a participant unless one is already recorded."""
        if not self._catalog.has_fingerprint(name):
            return self.fingerprint(name, source)
        return None

    def _collect(self, name: str, source: DataSource) -> pl.DataFrame:
        """One eager attempt: load, admit the result and its row budget."""
        source.prepare_identity(hash_contents=self.hash_inputs)
        self._resources.eager_fallbacks.add(name)
        result = collect_result(source.load(), source.params.row_budget)
        self._check_materialization(name, result)
        budget = getattr(source.params, "row_budget", None)
        if budget is not None and result.height > budget:
            raise SourceError(f"dataset {name!r} has more than {budget:,} rows (row_budget)")
        return result


class _VerifiedOperation[T]:
    """One attempt of a source operation: identities established before the first
    read, fingerprints taken after it (the enclosing policy verifies identity again)."""

    def __init__(
        self,
        execution: SourceExecution,
        participants: dict[str, DataSource],
        operation: Callable[[], T],
    ) -> None:
        self._execution = execution
        self._participants = participants
        self._operation = operation
        self.fingerprints: dict[str, dict[str, Any] | None] = {}

    def __call__(self) -> T:
        for source in self._participants.values():
            source.prepare_identity(hash_contents=self._execution.hash_inputs)
        value = self._operation()
        for name, source in self._participants.items():
            if value is not None and not self._execution._catalog.has_fingerprint(name):
                self.fingerprints[name] = self._execution.fingerprint(name, source)
        return value


class _EagerLoad:
    """One attempt of an eager load, keeping the fingerprint of the last one."""

    def __init__(
        self, execution: SourceExecution, name: str, source: DataSource, record_fingerprint: bool
    ) -> None:
        self._execution = execution
        self._name = name
        self._source = source
        self._record_fingerprint = record_fingerprint
        self.fingerprint: dict[str, Any] | None = None

    def __call__(self) -> pl.DataFrame:
        result = self._execution._collect(self._name, self._source)
        if self._record_fingerprint:
            self.fingerprint = self._execution.fingerprint(self._name, self._source)
        return result


__all__ = ["LatestAttempt", "SourceExecution"]
