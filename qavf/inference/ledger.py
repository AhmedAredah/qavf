"""Corrections of a comparison's families, recorded and replayed for an inference family.

A ``compare`` item decides many small families (the groups of a measure, the
buckets of a temporal measure, ...), each corrected where it is judged. When the
item belongs to an inference family (``inference: {family: ...}``), those
corrections must use the p-values of every member. The item is then evaluated
twice: a :class:`RecordingLedger` keeps the raw p-values of every correction
(in order) while the first evaluation measures them, and a :class:`ReplayingLedger`
hands each correction its family-adjusted values while the second judges them.
Comparisons are deterministic (seeded resampling), so both evaluations correct the
same p-values in the same order; a difference is an error, never a silent mix.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from qavf.inference.multiple import correct
from qavf.inference.results import Correction, TestResult


class FamilyReplayError(RuntimeError):
    """The judging evaluation corrected other p-values than the measuring one."""


@dataclass(frozen=True, slots=True)
class RecordedCorrection:
    """One correction's raw p-values and the family size it counts."""

    p_values: np.ndarray
    family_size: int
    method: Correction


class CorrectionLedger:
    """Routes the corrections made while it is active (see :func:`family_correct`)."""

    def correct(
        self,
        results: Sequence[TestResult],
        method: Correction,
        alpha: float,
        family_size: int | None,
    ) -> tuple[list[TestResult], int]:
        """The corrected results and the family size their summary reports."""
        corrected = correct(results, method, alpha, family_size=family_size)
        return corrected, _size(results, family_size)

    @contextmanager
    def active(self) -> Iterator[None]:
        """Make this ledger route the corrections of the current thread."""
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)


class RecordingLedger(CorrectionLedger):
    """Corrects as usual and records each correction's raw p-values."""

    def __init__(self) -> None:
        self.recorded: list[RecordedCorrection] = []

    def correct(
        self,
        results: Sequence[TestResult],
        method: Correction,
        alpha: float,
        family_size: int | None,
    ) -> tuple[list[TestResult], int]:
        self.recorded.append(
            RecordedCorrection(
                np.asarray(_raw(results), dtype=float), _size(results, family_size), method
            )
        )
        return super().correct(results, method, alpha, family_size)

    @property
    def p_values(self) -> np.ndarray:
        """Every recorded raw p-value, in order."""
        if not self.recorded:
            return np.empty(0)
        return np.concatenate([entry.p_values for entry in self.recorded])

    @property
    def family_size(self) -> int:
        """The tests every recorded correction counts (untested ones as ``p = 1``)."""
        return sum(entry.family_size for entry in self.recorded)

    @property
    def methods(self) -> frozenset[str]:
        """The corrections the comparison's families asked for."""
        return frozenset(entry.method for entry in self.recorded)


class ReplayingLedger(CorrectionLedger):
    """Hands each correction, in recorded order, its family-adjusted p-values."""

    def __init__(
        self,
        recorded: Sequence[RecordedCorrection],
        adjusted: np.ndarray,
        method: Correction,
        family_size: int,
    ) -> None:
        self._recorded = list(recorded)
        self._adjusted = adjusted
        self._method: Correction = method
        self._size = family_size
        self._call = 0
        self._offset = 0

    def correct(
        self,
        results: Sequence[TestResult],
        method: Correction,
        alpha: float,
        family_size: int | None,
    ) -> tuple[list[TestResult], int]:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
        raw = np.asarray(_raw(results), dtype=float)
        if self._call >= len(self._recorded) or not np.array_equal(
            self._recorded[self._call].p_values, raw
        ):
            raise FamilyReplayError(
                "the comparison's tests differed between measuring and judging its "
                "inference family; its inputs must not change during the run"
            )
        self._call += 1
        adjusted = self._adjusted[self._offset : self._offset + len(raw)]
        self._offset += len(raw)
        out = list(results)
        for index, value in zip(_tested(results), adjusted, strict=True):
            out[index] = (
                results[index]
                .model_copy(
                    update={
                        "adjusted_p": float(value),
                        "correction": self._method,
                        "family_size": self._size,
                    }
                )
                .judged(alpha)
            )
        return out, self._size

    def complete(self) -> bool:
        """Whether every recorded correction was replayed."""
        return self._call == len(self._recorded)


_ACTIVE: ContextVar[CorrectionLedger | None] = ContextVar("qavf_correction_ledger", default=None)


def family_correct(
    results: Sequence[TestResult],
    method: Correction,
    alpha: float,
    family_size: int | None = None,
) -> tuple[list[TestResult], int]:
    """:func:`~qavf.inference.multiple.correct` through the active ledger, if any,
    with the family size the judged family reports."""
    ledger = _ACTIVE.get() or CorrectionLedger()
    return ledger.correct(results, method, alpha, family_size)


def _tested(results: Sequence[TestResult]) -> list[int]:
    return [i for i, result in enumerate(results) if result.assessed and result.p_value is not None]


def _raw(results: Sequence[TestResult]) -> list[float]:
    return [float(results[i].p_value or 0.0) for i in _tested(results)]


def _size(results: Sequence[TestResult], family_size: int | None) -> int:
    return len(_tested(results)) if family_size is None else family_size


__all__ = [
    "CorrectionLedger",
    "FamilyReplayError",
    "RecordedCorrection",
    "RecordingLedger",
    "ReplayingLedger",
    "family_correct",
]
