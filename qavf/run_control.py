"""Progress events and cancellation for a validation run, shared by both backends.

``validate(..., on_event=callback, cancel=token)`` reports each step as it
finishes and stops the run when ``token.cancel()`` is called from any thread.
A cancelled run still returns a report: completed steps keep their results and
the rest are ``ERROR`` "not assessed", so the verdict is ``incomplete``.

A step is one rule on the full backend and one rule stage of one dataset on the
native backend. Imports neither Polars nor Pydantic.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from qavf.reports.models import QAReport, RuleResult

logger = logging.getLogger(__name__)

CANCELLED_MESSAGE = "Not assessed: the run was cancelled."

EventKind = Literal["run_started", "step_started", "step_finished", "run_finished"]


class CancellationToken:
    """Thread-safe request to stop a run; cancelling twice is harmless."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._listeners: list[Callable[[], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation and interrupt the work in progress."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener()
            except Exception:  # Interrupting is best effort; the run still stops.
                logger.debug("A cancellation listener failed", exc_info=True)

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` on cancellation, at once if already cancelled."""
        with self._lock:
            if not self._event.is_set():
                self._listeners.append(listener)
                return
        listener()

    def remove_listener(self, listener: Callable[[], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    @contextlib.contextmanager
    def listening(self, listener: Callable[[], None]) -> Iterator[None]:
        """Call ``listener`` on cancellation while the block runs."""
        self.add_listener(listener)
        try:
            yield
        finally:
            self.remove_listener(listener)


@dataclass(frozen=True)
class RunEvent:
    """One progress event.

    ``total`` is the number of steps planned, or ``None`` when the backend does
    not know it in advance (the native backend). ``results`` holds the step's
    results on ``step_finished``; ``report`` is set on ``run_finished``.
    """

    kind: EventKind
    completed: int
    total: int | None = None
    rule_ids: tuple[str, ...] = ()
    dataset: str | None = None
    results: tuple[RuleResult, ...] = ()
    report: QAReport | None = None
    cancelled: bool = False

    def to_dict(self) -> dict[str, object]:
        """A compact JSON-ready summary (results by status, not the full results)."""
        statuses: dict[str, int] = {}
        for result in self.results:
            statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        record: dict[str, object] = {
            "event": self.kind,
            "completed": self.completed,
            "total": self.total,
            "cancelled": self.cancelled,
        }
        if self.rule_ids:
            record["rule_ids"] = list(self.rule_ids)
        if self.dataset is not None:
            record["dataset"] = self.dataset
        if self.kind == "step_finished":
            record["statuses"] = statuses
        if self.report is not None:
            record["verdict"] = self.report.verdict(self.report.metadata.fail_on)
        return record


class RunControl:
    """One run's event sink and cancellation state.

    Events are delivered one at a time, in order, from whichever thread finished
    the step. A failing callback is logged and never stops the run.
    """

    def __init__(
        self,
        on_event: Callable[[RunEvent], None] | None = None,
        cancel: CancellationToken | None = None,
    ) -> None:
        self._on_event = on_event
        self.token = cancel or CancellationToken()
        self._lock = threading.Lock()
        self._completed = 0
        self._total: int | None = None

    @property
    def cancelled(self) -> bool:
        return self.token.cancelled

    def started(self, total: int | None) -> None:
        with self._lock:
            self._total = total
            self._emit(RunEvent("run_started", 0, total, cancelled=self.cancelled))

    def step_started(self, rule_ids: Sequence[str], dataset: str | None = None) -> None:
        with self._lock:
            self._emit(
                RunEvent(
                    "step_started",
                    self._completed,
                    self._total,
                    tuple(rule_ids),
                    dataset,
                    cancelled=self.cancelled,
                )
            )

    def step_finished(
        self,
        rule_ids: Sequence[str],
        dataset: str | None,
        results: Sequence[RuleResult],
    ) -> None:
        with self._lock:
            self._completed += 1
            self._emit(
                RunEvent(
                    "step_finished",
                    self._completed,
                    self._total,
                    tuple(rule_ids),
                    dataset,
                    tuple(results),
                    cancelled=self.cancelled,
                )
            )

    def finished(self, report: QAReport) -> None:
        with self._lock:
            self._emit(
                RunEvent(
                    "run_finished",
                    self._completed,
                    self._total,
                    report=report,
                    cancelled=self.cancelled,
                )
            )

    def _emit(self, event: RunEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            logger.warning("A progress callback failed; the run continues", exc_info=True)


def control_for(
    on_event: Callable[[RunEvent], None] | None,
    cancel: CancellationToken | None,
    execution_mode: str,
) -> RunControl | None:
    """The run's control, or ``None`` when neither hook was requested."""
    if on_event is None and cancel is None:
        return None
    if execution_mode == "isolated":
        raise ValueError(
            "progress events and cancellation require engine.execution_mode: in_process"
        )
    return RunControl(on_event, cancel)


__all__ = ["CANCELLED_MESSAGE", "CancellationToken", "RunControl", "RunEvent", "control_for"]
