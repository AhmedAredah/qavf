"""Mergeable exact accumulation of stored binary floating-point observations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction


@dataclass(slots=True)
class ExactFloatAccumulator:
    """Retain an integer total and count, independent of input/chunk ordering.

    Every Float64 is an integer multiple of 2**-1074. Storage grows only with
    the bit length of the total and count; no observation buffer is retained.
    Null/NaN observations are missing; infinities cannot be aggregated.
    """

    units: int = 0
    count: int = 0

    def update(self, values: Iterable[float | None]) -> None:
        units, count = self.units, self.count
        for value in values:
            if value is None or math.isnan(value):
                continue
            if not math.isfinite(value):
                raise ValueError("non-finite observations cannot be aggregated")
            numerator, denominator = float(value).as_integer_ratio()
            units += numerator << (1074 - (denominator.bit_length() - 1))
            count += 1
        self.units, self.count = units, count

    def total(self) -> Fraction | None:
        return Fraction(self.units, 1 << 1074) if self.count else None

    def update_counted(self, values: Iterable[tuple[float, int]]) -> None:
        """Accumulate exact value frequencies without expanding repeated observations."""
        units, count = self.units, self.count
        for value, frequency in values:
            if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 0:
                raise ValueError("observation frequencies must be nonnegative integers")
            if not math.isfinite(value):
                raise ValueError("non-finite observations cannot be aggregated")
            numerator, denominator = float(value).as_integer_ratio()
            units += (numerator << (1074 - (denominator.bit_length() - 1))) * frequency
            count += frequency
        self.units, self.count = units, count

    def mean(self) -> Fraction | None:
        return Fraction(self.units, (1 << 1074) * self.count) if self.count else None
