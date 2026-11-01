"""Convenience reading and writing of tabular files."""

from __future__ import annotations

import gzip
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

from qavf.output_format import table_output
from qavf.redaction import redact_text
from qavf.sources.base import SourceEnvironment
from qavf.sources.local_files import discover_local_files
from qavf.sources.parameters import FileParams

if TYPE_CHECKING:
    import polars as pl

    from qavf.settings import QAVFSettings

logger = logging.getLogger(__name__)


@contextmanager
def atomic_stream(path: str | Path) -> Iterator[BinaryIO]:
    """Stream to a temporary file, committing only after a successful close.

    Peak writer memory is independent of output size. The previous destination
    survives serialization, compression, or fsync errors.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        _check_publication(target)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write(path: str | Path, data: bytes | str) -> Path:
    """Atomically write a small bytes/text artifact; use atomic_stream for large data."""
    target = Path(path)
    with atomic_stream(target) as stream:
        if isinstance(data, str):
            for start in range(0, len(data), 65536):
                stream.write(data[start : start + 65536].encode("utf-8"))
        else:
            stream.write(data)
    return target


def read_table(path: str | Path, fmt: str | None = None) -> pl.DataFrame:
    """Read a CSV, Parquet, JSON/NDJSON or Excel file (or glob).

    Parquet logical types use a typed reader when SQL conversion is lossy.

    Raises:
        qavf.sources.SourceNotFoundError: if the file does not exist.
        qavf.sources.SourceError: if it cannot be read.
    """
    from qavf.optional import require_polars

    require_polars("read_table returning a Polars DataFrame")
    from qavf.sources.file import FileSource

    params = FileParams.model_validate({"type": "file", "path": str(path), "format": fmt})
    return FileSource(params, SourceEnvironment()).load()


def write_table(frame: pl.DataFrame, path: str | Path, fmt: str | None = None) -> Path:
    """Write ``frame`` in the format its name says, creating parent directories.

    ``.csv`` and ``.txt`` are comma-separated, ``.tsv`` tab-separated;
    ``.parquet``, ``.json`` and ``.ndjson``/``.jsonl`` as named. A ``.gz`` suffix
    on a CSV or JSON name writes gzip-compressed data. Parquet preserves typed
    values; text formats obey their format's representation limits. Baseline
    acceptance verifies an exact round trip before publishing a baseline.

    Raises:
        ValueError: for a name or format that cannot be written as it says (Excel,
            zstd compression, compressed Parquet, an unknown suffix), before
            anything is created.
    """
    target = Path(path)
    output = table_output(target, fmt)

    def write(stream: BinaryIO | gzip.GzipFile) -> None:
        match output.format:
            case "csv":
                frame.write_csv(cast(BinaryIO, stream), separator=output.separator)
            case "parquet":
                frame.write_parquet(cast(BinaryIO, stream))
            case "json":
                frame.write_json(cast(IOBase, stream))
            case _:
                frame.write_ndjson(cast(BinaryIO, stream))

    with atomic_stream(target) as stream:
        if output.compressed:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                write(compressed)
        else:
            write(stream)
    logger.debug("Wrote %d rows to %s", frame.height, redact_text(str(target)))
    return target


class ProtectedInputs:
    """The local files a run reads; a report destination may never replace one.

    Report publication replaces its target atomically, so a destination that names
    an input would destroy that input completely. Destinations are compared after
    resolving symbolic links, and with ``samefile`` where both exist (hard links,
    case-insensitive file systems). SQL queries may read further files that are not
    known here; only declared file sources, database files and named inputs are.
    """

    def __init__(self, paths: Iterable[str | Path | None]) -> None:
        self.locations = tuple(given for given in paths if given is not None)
        self.paths = [path for given in self.locations for path in _local(given)]

    @contextmanager
    def publication(self) -> Iterator[None]:
        """Check the frozen inputs and current aliases at atomic publication.

        The initial manifest remains protected even if a pattern later changes.
        Fresh matches are added before publication so newly selected inputs are
        protected too. Nested publication scopes preserve their outer guards.
        """
        token = _PUBLICATION_CHECKS.set((*_PUBLICATION_CHECKS.get(), self.check))
        try:
            yield
        finally:
            _PUBLICATION_CHECKS.reset(token)

    @classmethod
    def of_settings(cls, settings: QAVFSettings, config: Path | None) -> ProtectedInputs:
        """The configuration file, its file datasets and its database files."""
        found: list[str | Path | None] = [config]
        for spec in settings.datasets.values():
            source = spec.source_config()
            if source is not None and source.get("type") == "file":
                found.append(_resolved(settings, str(source["path"])))
        for connection in settings.connections.values():
            if connection.file_based and connection.path:
                found.append(_resolved(settings, connection.path))
        return cls(found)

    def check(self, *destinations: Path | None) -> None:
        """Refuse a destination that is one of the inputs."""
        current = [path for given in self.locations for path in _local(given)]
        sources = dict.fromkeys((*self.paths, *current))
        for destination in destinations:
            if destination is None:
                continue
            for source in sources:
                if _same_file(destination, source):
                    raise ValueError(
                        f"report path {destination} would replace the input {source}; "
                        "choose another output path"
                    )


def _resolved(settings: QAVFSettings, path: str) -> str:
    return path if "://" in path else str(settings.resolve(path))


def _local(path: str | Path) -> list[Path]:
    """The local files ``path`` names: none for a URL, every match for a glob."""
    files = discover_local_files(path)
    if files is None:
        return []
    return list(files) or ([Path(path)] if not any(c in str(path) for c in "*?[") else [])


_PUBLICATION_CHECKS: ContextVar[tuple[Callable[..., None], ...]] = ContextVar(
    "qavf_publication_checks", default=()
)


def _check_publication(target: Path) -> None:
    for check in _PUBLICATION_CHECKS.get():
        check(target)


def _same_file(destination: Path, source: Path) -> bool:
    try:
        if os.path.normcase(str(destination.resolve())) == os.path.normcase(str(source.resolve())):
            return True
        return destination.exists() and source.exists() and os.path.samefile(destination, source)
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop on Python 3.12
        return False
