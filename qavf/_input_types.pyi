"""Static input surface, without importing optional frame libraries at runtime."""

from collections.abc import Mapping
from os import PathLike
from typing import Any, Protocol

import polars as pl

from qavf.inputs.descriptors import BatchSource, Dataset, File, Loader, Query
from qavf.sources.base import DataSource

class ArrowTableStream(Protocol):
    def __arrow_c_stream__(self, requested_schema: Any = ...) -> Any: ...

class InterchangeFrame(Protocol):
    def __dataframe__(self, nan_as_null: bool = ..., allow_copy: bool = ...) -> Any: ...

type FrameInput = pl.DataFrame | pl.LazyFrame | InterchangeFrame | ArrowTableStream
type DatasetInput = (
    str
    | PathLike[str]
    | File
    | Dataset
    | Query
    | Loader
    | BatchSource
    | DataSource[Any]
    | FrameInput
)
type ValidationInput = DatasetInput | Mapping[str, DatasetInput] | None
