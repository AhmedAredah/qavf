"""One retry and identity policy for complete eager/native source operations."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterable, Iterator

import duckdb

from qavf.sources.base import DataSource, SourceError, SourceIntegrityError, SourceNotFoundError
from qavf.sources.errors import redacted_error


class SourceOperations:
    """One run's source policy, including cancellable serialized execution.

    The reentrant guard permits SQL to materialize an eager participant while
    already executing under the same policy. Waiting callers remain cancellable.
    No source is published before both identity checks have completed.
    """

    def __init__(self, *, concurrent: bool, check_cancelled: Callable[[], None]) -> None:
        self._lock = None if concurrent else threading.RLock()
        self.check_cancelled = check_cancelled

    @contextlib.contextmanager
    def guard(self) -> Iterator[None]:
        self.check_cancelled()
        if self._lock is None:
            yield
            return
        while not self._lock.acquire(timeout=0.05):
            self.check_cancelled()
        try:
            self.check_cancelled()
            yield
        finally:
            self._lock.release()

    def run[T](
        self,
        sources: Iterable[DataSource],
        operation: Callable[[], T],
        *,
        on_retry: Callable[[int, float, SourceError], None] | None = None,
    ) -> T:
        with self.guard():
            return execute_with_source_policy(
                sources, operation, check_cancelled=self.check_cancelled, on_retry=on_retry
            )


def execute_with_source_policy[T](
    sources: Iterable[DataSource],
    operation: Callable[[], T],
    *,
    check_cancelled: Callable[[], None] | None = None,
    on_retry: Callable[[int, float, SourceError], None] | None = None,
) -> T:
    """Execute one coherent operation with the strictest participating policy.

    The operation owns per-attempt result/connection cleanup and row-budget checks.
    For native SQL it includes binding and execution, not separate count/sample
    calls. For eager input it includes conversion and materialization checks.
    Unknown failures and detected identity changes are never implicitly retried.
    """
    participants = tuple({id(source): source for source in sources}.values())
    retries = min((source.params.retries for source in participants), default=0)
    delay = max((source.params.retry_delay_s for source in participants), default=0.0)
    safe = all(source.retry_safe for source in participants)
    for attempt in range(retries + 1):
        if check_cancelled is not None:
            check_cancelled()
        try:
            with contextlib.ExitStack() as verification:
                for source in participants:
                    verification.enter_context(source.verified_operation())
                result = operation()
            if check_cancelled is not None:
                check_cancelled()
            if attempt:
                for source in participants:
                    source.load_notes["retries_used"] = attempt
            return result
        except (SourceError, duckdb.Error) as exc:
            error = exc if isinstance(exc, SourceError) else redacted_error(exc)
            if (
                attempt >= retries
                or not safe
                or isinstance(error, SourceNotFoundError | SourceIntegrityError)
                or not error.retryable
            ):
                raise error from None
            pause = delay * 2**attempt
            if on_retry is not None:
                on_retry(attempt + 1, pause, error)
            until = time.monotonic() + pause
            while time.monotonic() < until:
                if check_cancelled is not None:
                    check_cancelled()
                time.sleep(min(0.05, max(0.0, until - time.monotonic())))
    raise AssertionError("source operation exhausted without a result or exception")
