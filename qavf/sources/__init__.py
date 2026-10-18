"""Pluggable data sources: files, SQL, databases and Python callables via DuckDB."""

from qavf.sources.base import (
    DataSource,
    ScanPlan,
    SourceEnvironment,
    SourceError,
    SourceIntegrityError,
    SourceNotFoundError,
    SourceParams,
    SourceRegistry,
)
from qavf.sources.file import FileSource
from qavf.sources.formats import infer_format
from qavf.sources.frames import as_polars
from qavf.sources.parameters import DatabaseParams, FileParams, PythonParams, QueryParams
from qavf.sources.python import PythonSource
from qavf.sources.query import DatabaseSource, QuerySource
from qavf.sources.sql import validate_query

__all__ = [
    "DataSource",
    "DatabaseParams",
    "DatabaseSource",
    "FileParams",
    "FileSource",
    "PythonParams",
    "PythonSource",
    "QueryParams",
    "QuerySource",
    "ScanPlan",
    "SourceEnvironment",
    "SourceError",
    "SourceIntegrityError",
    "SourceNotFoundError",
    "SourceParams",
    "SourceRegistry",
    "as_polars",
    "infer_format",
    "validate_query",
]
