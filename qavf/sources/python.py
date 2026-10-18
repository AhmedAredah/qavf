"""Python callable sources and their bounded result conversion."""

from __future__ import annotations

import importlib
from typing import ClassVar

import polars as pl

from qavf.redaction import redact_text
from qavf.sources.base import DataSource, SourceError, SourceParams
from qavf.sources.errors import redacted_error
from qavf.sources.frames import as_polars
from qavf.sources.native import check_budget
from qavf.sources.parameters import PythonParams


class PythonSource(DataSource[PythonParams]):
    """Whatever a Python function returns: the escape hatch for anything else."""

    kind: ClassVar[str] = "python"
    Params: ClassVar[type[SourceParams]] = PythonParams

    def load(self) -> pl.DataFrame | pl.LazyFrame:
        module_name, _, attribute = self.params.callable.partition(":")
        if not module_name or not attribute:
            raise SourceError(
                "'callable' must look like 'package.module:function', got "
                f"{redact_text(self.params.callable)!r}"
            )
        try:
            function = getattr(importlib.import_module(module_name), attribute)
            result = function(**self.params.kwargs)
            # Conversion (e.g. a pandas column of mixed types, or a lazy frame that
            # fails to collect) is part of loading and fails the same way.
            if isinstance(result, pl.LazyFrame):
                return (
                    result.limit(self.params.row_budget + 1)
                    if self.params.row_budget is not None
                    else result
                )
            return check_budget(as_polars(result), self.params.row_budget, self.describe())
        except Exception as exc:
            raise redacted_error(
                SourceError(
                    f"{self.params.callable} failed: {type(exc).__name__}: {exc}",
                    retryable=bool(getattr(exc, "retryable", False)),
                )
            ) from None

    def describe(self) -> str:
        return f"python: {self.params.callable}"
