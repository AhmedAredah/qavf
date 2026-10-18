"""One result conversion policy for ordinary and transactional source reads."""

from __future__ import annotations

from typing import Any

import polars as pl

from qavf.sources.frames import as_polars


def collect_result(value: Any, row_budget: int | None) -> pl.DataFrame:
    """Bound a lazy result before collection; eager inputs already own their allocation."""
    if isinstance(value, pl.LazyFrame):
        if row_budget is not None:
            value = value.limit(row_budget + 1)
        value = value.collect(engine="streaming")
    return as_polars(value)
