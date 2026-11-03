"""Column names a comparison's options refer to, checked against the known schemas.

The checks run inside the comparison engine once the schemas are known and before
any level reads data: a derived column must use existing or earlier derived columns,
and ``tolerances`` and ``ignore_columns`` must name a column of at least one aligned
side. A misspelled name would otherwise be filtered out silently.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

import polars as pl

from qavf.compare.spec import ComparisonSpec, Side


class ComparisonConfigurationError(ValueError):
    """A comparison option names a column that does not exist; nothing is assessed."""


def _hint(name: str, known: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.6)
    return f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""


class ComparisonColumnNames:
    """Checks the column names of one comparison's ``derive``, ``tolerances`` and
    ``ignore_columns`` against schemas the engine already has."""

    def __init__(self, spec: ComparisonSpec) -> None:
        self.spec = spec

    def check_derive(self, side: Side, columns: Iterable[str]) -> None:
        """Each derived column uses columns of ``side`` or earlier derived columns.

        ``columns`` are the side's columns before ``derive`` (after renaming).
        """
        available = set(columns)
        problems = []
        for name, expression in self.spec.derive.items():
            used = pl.sql_expr(expression).meta.root_names()
            for missing in sorted(set(used) - available):
                problems.append(
                    f"derived column {name!r} uses {missing!r}, which is neither a "
                    f"{side} column nor an earlier derived column{_hint(missing, available)}"
                )
            available.add(name)
        if problems:
            raise ComparisonConfigurationError("; ".join(problems))

    def check_options(self, known: Iterable[str]) -> None:
        """``tolerances`` and ``ignore_columns`` name columns of at least one side.

        ``known`` holds every aligned column of either side; a column present on
        only one side is accepted (structure reports it as added or removed).
        """
        names = set(known)
        problems = [
            f"{setting} names column {name!r}, which neither aligned dataset has"
            f"{_hint(name, names)}"
            for setting, configured in (
                ("tolerances", self.spec.tolerances),
                ("ignore_columns", self.spec.ignore_columns),
            )
            for name in configured
            if name not in names
        ]
        if problems:
            raise ComparisonConfigurationError("; ".join(problems))
