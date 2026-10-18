"""Small Python definitions of contracts, independent of QAVF/backend expressions."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from math import isnan
from typing import Any


@dataclass(frozen=True)
class MissingPolicy:
    """Transport-independent interpretation of explicitly declared missing values."""

    markers: tuple[object, ...] = ()
    blank_text: bool = False

    def missing(self, value: object) -> bool:
        return (
            value is None
            or (isinstance(value, float) and isnan(value))
            or (self.blank_text and isinstance(value, str) and not value.strip())
            or value in self.markers
        )


def key_counts(
    rows: Iterable[tuple[Any, ...]], policies: tuple[MissingPolicy, ...]
) -> tuple[Counter[tuple[Any, ...]], int, int]:
    """Return nonmissing frequencies, all missing-key rows, and partly missing rows."""
    counts: Counter[tuple[Any, ...]] = Counter()
    null_rows = partial_rows = 0
    for row in rows:
        missing = [policy.missing(value) for value, policy in zip(row, policies, strict=True)]
        if any(missing):
            null_rows += 1
            partial_rows += not all(missing)
        else:
            counts[row] += 1
    return counts, null_rows, partial_rows
