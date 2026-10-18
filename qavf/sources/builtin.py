"""Explicit built-in source inventory for registry discovery.

Public source classes and parameters are available from :mod:`qavf.sources`.
Each implementation and shared primitive lives in its owning source module.
"""

from qavf.sources.base import DataSource
from qavf.sources.file import FileSource
from qavf.sources.python import PythonSource
from qavf.sources.query import DatabaseSource, QuerySource

BUILTIN_SOURCES: tuple[type[DataSource], ...] = (
    FileSource,
    QuerySource,
    DatabaseSource,
    PythonSource,
)
