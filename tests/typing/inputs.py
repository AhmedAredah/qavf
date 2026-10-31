"""Run with mypy --warn-unused-ignores to verify the installed public input facade.

Expected-error ignores become errors themselves if an overly broad input union
starts accepting integers or column mappings as dataset inputs.
"""

from pathlib import Path
from typing import ClassVar

import polars as pl

from qavf import Dataset, File, Loader, Query, compare_datasets, validate


def accepted(frame: pl.DataFrame, path: Path) -> None:
    schema = {"columns": {"value": {"ge": 0}}}
    validate(frame, schema=schema)
    validate(frame.lazy(), schema=schema)
    validate(path, schema=schema)
    validate(File(path), schema=schema)
    validate(Query("select 1 as value"), schema=schema)
    validate(Loader(lambda: frame), schema=schema)
    validate({"sample": Dataset(frame, schema=schema)}, config={})
    compare_datasets(frame, path, keys="id")


def rejected() -> None:
    validate(42, config={})  # type: ignore[arg-type]
    validate({"value": [1, 2]}, config={})  # type: ignore[dict-item]
    compare_datasets(42, 43)  # type: ignore[arg-type]


class FrameLookalike:
    columns: ClassVar[list[str]] = []
    dtypes: ClassVar[list[str]] = []
    schema: ClassVar[dict[str, str]] = {}
    num_rows: int = 0

    def to_numpy(self) -> list[int]:
        return []


class InterchangeProvider:
    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True) -> object:
        raise NotImplementedError


class ArrowProvider:
    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        raise NotImplementedError


def contracts() -> None:
    from qavf import explain, profile_input

    validate(InterchangeProvider(), schema={})
    validate(ArrowProvider(), schema={})
    explain(InterchangeProvider(), schema={})
    profile_input(ArrowProvider())
    validate(FrameLookalike(), schema={})  # type: ignore[arg-type]
    explain(FrameLookalike(), schema={})  # type: ignore[arg-type]
    profile_input(FrameLookalike())  # type: ignore[arg-type]
    explain(42, schema={})  # type: ignore[arg-type]
    profile_input(42)  # type: ignore[arg-type]
