"""Complete distribution comparisons over bounded batches and indexed counts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from qavf.inference.effects import population_stability
from qavf.inference.independent import ks_asymptotic_p_value, ks_two_sample
from qavf.statistics.ordered import bin_index, power_of_two, quantile_cutpoints, scaled
from qavf.stats import checked_float_series

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation
    from qavf.engine.observation_plan import ObservationView
    from qavf.engine.sqlite_store import Store


@dataclass(slots=True)
class Distribution:
    store: Store
    numeric: bool
    sizes: list[int]
    original_sizes: list[int]

    def rows(self, side: int | None = None) -> Iterator[tuple[Any, int, int]]:
        where = f" WHERE {'reference' if side == 0 else 'current'} > 0" if side is not None else ""
        cursor = self.store.execute(
            "SELECT value, reference, current FROM counts" + where + " ORDER BY value"
        )
        try:
            while batch := cursor.fetchmany(2048):
                self.store.resources.session.check()
                yield from batch
        finally:
            cursor.close()

    def metrics(self) -> dict[str, Any]:
        return {
            "reference_values": self.sizes[0],
            "values": self.sizes[1],
            "reference_excluded": self.original_sizes[0] - self.sizes[0],
            "excluded": self.original_sizes[1] - self.sizes[1],
            "execution": "indexed_counts",
        }

    def psi(self, bins: int) -> tuple[float, int, str]:
        reference_unique = self.store.execute(
            "SELECT count(*) FROM counts WHERE reference > 0"
        ).fetchone()[0]
        pairs: Iterable[tuple[int, int]]
        if self.numeric and reference_unique > bins:
            low, high = self.store.execute("SELECT min(value), max(value) FROM counts").fetchone()
            unit = power_of_two(max(abs(low), abs(high)))
            # Validate every observation; selecting only quantile ranks would
            # miss information lost at other points in a wide dynamic range.
            for value, _, _ in self.rows():
                scaled(value, 0.0, unit)
            cutpoints = quantile_cutpoints(
                ((scaled(v, 0.0, unit), r) for v, r, _ in self.rows(0)), self.sizes[0], bins
            )
            expected = [0] * (len(cutpoints) + 1)
            actual = [0] * len(expected)
            for value, ref, cur in self.rows():
                bucket = int(bin_index(cutpoints, scaled(value, 0.0, unit)))
                expected[bucket] += ref
                actual[bucket] += cur
            pairs = zip(expected, actual, strict=True)
            count = len(expected)
            policy = "reference_quantiles_with_support_tails"
        else:
            pairs = ((r, c) for _, r, c in self.rows())
            count = self.store.execute("SELECT count(*) FROM counts").fetchone()[0]
            policy = "distinct_values"
        sizes = (self.sizes[0], self.sizes[1])
        return population_stability(pairs, sizes), count, policy

    def ks(self, stack: Any = None) -> tuple[float, float, str]:
        """Match the documented SciPy auto policy with an exact streamed EDF.

        Small samples use the shared two-sample KS test (SciPy's ``ks_2samp``
        unchanged). For larger samples its two-sided asymptotic probability is
        computed from complete cumulative counts. Neither large observation
        arrays nor a concatenated EDF exist. ``stack`` is accepted for callers
        that pass ``scipy.stats``; SciPy is loaded by :mod:`qavf.inference`.
        """
        if max(self.sizes) <= 10_000:
            arrays = [np.empty(size, dtype=np.float64) for size in self.sizes]
            offsets = [0, 0]
            for value, ref, cur in self.rows():
                for side, count in enumerate((ref, cur)):
                    arrays[side][offsets[side] : offsets[side] + count] = value
                    offsets[side] += count
            result = ks_two_sample(*arrays)
            if not result.assessed:
                raise ValueError(result.reason)
            assert result.statistic is not None and result.p_value is not None
            return result.statistic, result.p_value, "scipy_auto_bounded_sample"
        ref_total = cur_total = 0
        statistic = 0.0
        for _, ref, cur in self.rows():
            ref_total += ref
            cur_total += cur
            statistic = max(statistic, abs(ref_total / self.sizes[0] - cur_total / self.sizes[1]))
        probability = ks_asymptotic_p_value(statistic, (self.sizes[0], self.sizes[1]))
        return statistic, probability, "scipy_auto_asymptotic_complete_edf"


@contextmanager
def observed_distribution(
    ctx: DataContext,
    reference: str,
    current: str,
    column: str,
    normalize: Callable,
    observations: Mapping[str, Observation | ObservationView] | None = None,
) -> Iterator[Distribution]:
    """Normalize each bounded batch once and retain only indexed frequencies."""
    names = (reference, current)
    scope = (
        ctx.projected_observations({name: [column] for name in names})
        if observations is None
        else nullcontext(observations)
    )
    with (
        scope as observations,
        ctx.work_store("distribution counts") as store,
    ):
        numeric = all(observations[name].schema()[column].is_numeric() for name in names)
        dtype = "REAL" if numeric else "TEXT"
        store.execute(
            f"CREATE TABLE counts(value {dtype} PRIMARY KEY, reference INTEGER NOT NULL,"
            " current INTEGER NOT NULL) WITHOUT ROWID"
        )
        result = Distribution(store, numeric, [0, 0], [0, 0])
        for side, name in enumerate(names):
            for batch in observations[name].batches():
                result.original_sizes[side] += batch.height
                values = normalize(ctx, name, batch, [column])[column]
                if numeric:
                    values = checked_float_series(values)
                values = values.drop_nulls()
                if values.dtype.is_float():
                    values = values.filter(values.is_finite())
                if not numeric:
                    values = values.cast(pl.String)
                result.sizes[side] += values.len()
                counts = values.rename("value").value_counts()
                payload = (
                    (value, count if side == 0 else 0, count if side else 0)
                    for value, count in counts.iter_rows()
                )
                store.executemany(
                    "INSERT INTO counts VALUES(?,?,?) ON CONFLICT(value) DO UPDATE SET "
                    "reference=reference+excluded.reference,current=current+excluded.current",
                    payload,
                )
        yield result
