"""Severity and status vocabulary shared by the whole framework."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Severity(StrEnum):
    """How serious a failed rule is, ordered ``INFO < WARNING < ERROR < CRITICAL``.

    Members are strings so they round-trip through YAML/JSON unchanged, but they
    compare by rank rather than alphabetically::

        >>> Severity.CRITICAL > Severity.ERROR
        True
        >>> max([Severity.INFO, Severity.WARNING])
        <Severity.WARNING: 'WARNING'>
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def _missing_(cls, value: object) -> Severity | None:
        """Accept case-insensitive names such as ``"warning"``."""
        if isinstance(value, str):
            return cls.__members__.get(value.strip().upper())
        return None

    @property
    def rank(self) -> int:
        """Numeric rank used for ordering (higher is more severe)."""
        return _RANKS[self.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank < other.rank

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank <= other.rank

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank > other.rank

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.rank >= other.rank


_RANKS: dict[str, int] = {"INFO": 10, "WARNING": 20, "ERROR": 30, "CRITICAL": 40}


class Status(StrEnum):
    """Outcome of executing a single rule."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    """The rule did not apply (missing dataset, nothing configured)."""
    ERROR = "ERROR"
    """The rule itself raised; the data was not actually assessed."""


def highest_severity(severities: Iterable[Severity]) -> Severity | None:
    """Return the most severe member of ``severities``, or ``None`` when empty."""
    return max(severities, default=None, key=lambda s: s.rank)


def meets_threshold(severity: Severity | None, threshold: Severity) -> bool:
    """Whether ``severity`` is at least as severe as ``threshold``."""
    return severity is not None and severity >= threshold
