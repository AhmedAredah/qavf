"""Exact period coverage with one domain and bounded missing-label evidence."""

from __future__ import annotations

from typing import Any

import polars as pl


class PeriodDomain:
    """Reuse expected-period ranks; never construct a group's complete gap list.

    Calendar labels stay in typed storage, including distinct DST fold instants.
    Numeric domains retain Python's exact mixed-number equality. Per-group state
    contains only observed ranks; evidence visits at most that many ranks plus
    the requested missing prefix, independently of the number of missing periods.
    """

    def __init__(self, expected: set[Any] | pl.Series) -> None:
        self._ranks: dict[Any, int] | None = None
        self._physical: pl.Series | None = None
        self._labels: tuple[Any, ...] | pl.Series
        if isinstance(expected, pl.Series):
            self._labels = expected.unique().sort()
            self._physical = self._labels.to_physical().cast(pl.Int64)
        else:
            ordered = sorted(expected)
            self._ranks = {value: index for index, value in enumerate(ordered)}
            self._labels = tuple(int(value) if value == int(value) else value for value in ordered)

    def __len__(self) -> int:
        return len(self._labels)

    def prefix(self, limit: int) -> list[Any]:
        return [self._labels[index] for index in range(min(max(0, limit), len(self)))]

    def missing(self, periods: pl.Series, limit: int) -> tuple[int, list[Any]]:
        if self._physical is not None:
            observed = periods.drop_nulls().to_physical().cast(pl.Int64)
            ranks = self._physical.search_sorted(observed)
            inside = ranks < len(self)
            ranks, observed = ranks.filter(inside), observed.filter(inside)
            matched = set(ranks.filter(self._physical.gather(ranks) == observed))
        else:
            assert self._ranks is not None
            matched = {self._ranks[value] for value in periods if value in self._ranks}
        count = len(self) - len(matched)
        if not count or limit <= 0:
            return count, []
        sample = []
        for rank in range(len(self)):
            if rank not in matched:
                sample.append(self._labels[rank])
                if len(sample) >= min(count, limit):
                    break
        return count, sample
