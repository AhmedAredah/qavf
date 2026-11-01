"""Exact decision statistics with bounded Python storage and spillable SQL order."""

from __future__ import annotations

import math
from fractions import Fraction

from qavf._float_accumulator import ExactFloatAccumulator
from qavf.core.key_policy import FLOATS, INTEGERS
from qavf.sources.sql import quote_identifier as qi


class ExactAggregates:
    """Keep statistical decisions exact; round only optional presentation values."""

    def __init__(self, connection, table: str, expression: str, dtype: str, scope: str):
        self.con, self.table, self.expression = connection, qi(table), expression
        self.dtype, self.scope = dtype, scope

    def query(self, expression: str):
        return self.con.execute(f"SELECT {expression} FROM {self.table} WHERE {self.scope}")

    def values(self):
        cursor = self.query(self.expression)
        while batch := cursor.fetchmany(8192):
            for (value,) in batch:
                if value is not None:
                    yield value

    def statistic(self, kind: str):
        numeric = self.dtype in INTEGERS | FLOATS or self.dtype.startswith("DECIMAL")
        if kind in {"sum", "mean"} and numeric:
            if self.dtype in FLOATS:
                accumulator = ExactFloatAccumulator()
                accumulator.update(self.values())
                return accumulator.total() if kind == "sum" else accumulator.mean()
            # Native integer/decimal SUM remains exact and avoids Python row iteration.
            try:
                total, count = self.query(
                    f"sum({self.expression}), count({self.expression})"
                ).fetchone()
            except Exception:
                # Fixed-width SQL overflow has an exact, bounded-storage fallback.
                total, count = Fraction(), 0
                for value in self.values():
                    total += Fraction(value)
                    count += 1
            return None if not count else Fraction(total) / (count if kind == "mean" else 1)
        if kind == "median" and numeric:
            count = self.query(f"count({self.expression})").fetchone()[0]
            if not count:
                return None
            cursor = self.con.execute(
                f"SELECT {self.expression} FROM {self.table} WHERE ({self.scope}) "
                f"AND ({self.expression}) IS NOT NULL ORDER BY {self.expression} "
                f"LIMIT {2 if count % 2 == 0 else 1} OFFSET {(count - 1) // 2}"
            )
            values = [Fraction(row[0]) for row in cursor.fetchall()]
            return sum(values, Fraction()) / len(values)
        if kind == "std" and numeric:
            from statistics import StatisticsError, stdev

            try:
                value = stdev(self.values())
            except StatisticsError:
                return None
            if not math.isfinite(value):
                raise ValueError("standard deviation cannot be represented as a finite value")
            return value
        expression = (
            f"count(DISTINCT {self.expression})"
            if kind == "distinct_count"
            else f"{kind}({self.expression})"
        )
        return self.query(expression).fetchone()[0]


def observed_number(value):
    """Preserve nonintegral exact outcomes using the audit codec's rational spelling."""
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    return value
