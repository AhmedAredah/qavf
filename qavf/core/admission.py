"""Small, backend-independent source admission and identity policies."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from qavf.sources.base import SourceError, SourceIntegrityError
from qavf.sources.identity import sha256_file
from qavf.sources.local_files import LocalManifest


class SourceNames:
    """Reject names a case-insensitive SQL catalog cannot preserve faithfully."""

    @staticmethod
    def validate(names: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise SourceError("input has a blank or non-string column name")
            folded = name.casefold()
            if folded in seen:
                raise SourceError(
                    f"duplicate or case-colliding input column {name!r}; "
                    "DuckDB column names must be unique ignoring case"
                )
            seen.add(folded)
        return tuple(names)

    @staticmethod
    def fresh(preferred: str, taken: Sequence[str]) -> str:
        folded = {name.casefold() for name in taken}
        while preferred.casefold() in folded:
            preferred += "_"
        return preferred


class SourceProjection:
    """Resolve simultaneous renames before choosing retained source columns."""

    def __init__(self, columns, rename, canonical, required=None):
        SourceNames.validate(columns)
        canonical = {name.casefold(): name for name in canonical}
        self.renamed = {
            name: rename.get(name, canonical.get(name.casefold(), name)) for name in columns
        }
        self.original_columns = SourceNames.validate(tuple(self.renamed.values()))
        wanted = None if required is None else {name.casefold() for name in required}
        self.selected = tuple(
            (source, target)
            for source, target in self.renamed.items()
            if wanted is None or target.casefold() in wanted
        )


class FileIdentity:
    """Pin local file discovery and optionally content across admission/reporting."""

    def __init__(self, paths: tuple[Path, ...], hash_contents: bool = False):
        self.manifest = LocalManifest.capture(paths, self.stat)
        self.hashes = {path: sha256_file(path) for path in paths} if hash_contents else {}
        self.manifest.verify(self.stat)
        infos = []
        for path, stat in zip(paths, self.manifest.stats, strict=True):
            item = {
                "path": str(path),
                "identity": "content" if hash_contents else "metadata",
                "bytes": stat[0],
                "modified": datetime.fromtimestamp(stat[1] / 1e9, UTC).isoformat(),
            }
            if hash_contents:
                item["sha256"] = self.hashes[path]
            infos.append(item)
        if len(infos) == 1:
            self.fingerprint = {key: val for key, val in infos[0].items() if key != "path"}
        else:
            listing = hashlib.sha256()
            for item in infos:
                listing.update(
                    (
                        "\t".join(
                            str(item.get(k, "")) for k in ("path", "bytes", "modified", "sha256")
                        )
                        + "\n"
                    ).encode()
                )
            self.fingerprint = {
                "files": len(paths),
                "bytes": sum(item["bytes"] for item in infos),
                "identity": "content" if hash_contents else "metadata",
                "modified": max((item["modified"] for item in infos), default=None),
                "listing_sha256": listing.hexdigest(),
                "paths": [str(path) for path in paths[:1000]],
                "paths_truncated": len(paths) > 1000,
            }

    def verify(self):
        self.manifest.verify(self.stat)
        for path, expected in self.hashes.items():
            if sha256_file(path) != expected:
                raise SourceIntegrityError(f"input file content changed during validation: {path}")
        self.manifest.verify(self.stat)

    @staticmethod
    def stat(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def eager_columns(value):
    """Inspect names without reading or consuming an arbitrary input object."""
    module = type(value).__module__.split(".")[0]
    if module == "pyarrow" or any(
        cls.__module__.split(".")[0] == "pyarrow" for cls in type(value).__mro__
    ):
        import pyarrow as pa

        if isinstance(value, pa.RecordBatchReader):
            raise TypeError(
                "borrowed one-pass Arrow RecordBatchReader inputs are not supported; "
                "use BatchSource with an owned context-managed factory"
            )
        if isinstance(value, pa.Table | pa.RecordBatch):
            return value.schema.names
    if module in {"pandas", "polars"} and hasattr(value, "columns"):
        return list(value.columns)
    return None


def eager_rows(value):
    """Known cardinalities allow rejection before registration or copying."""
    module = type(value).__module__.split(".")[0]
    if module == "pyarrow" and hasattr(value, "num_rows"):
        return value.num_rows
    if module in {"pandas", "polars"} and hasattr(value, "shape"):
        return value.shape[0]
    return None


def arrow_schema(schema):
    """Keep Arrow streams usable in the core-only dependency installation."""
    import pyarrow as pa

    if isinstance(schema, pa.Schema):
        result = schema
    elif isinstance(schema, Mapping) and all(isinstance(t, pa.DataType) for t in schema.values()):
        result = pa.schema(schema)
    else:
        from qavf.optional import require_polars

        pl = require_polars("BatchSource with Polars dtype declarations")
        result = pl.DataFrame(schema=schema).to_arrow().schema
    SourceNames.validate(result.names)
    return result


def arrow_batch(batch):
    """Convert an eager batch only, never accept a hidden one-pass stream."""
    import pyarrow as pa

    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])
    if isinstance(batch, pa.Table):
        return batch
    module = type(batch).__module__.split(".")[0]
    if module == "pandas":
        return pa.Table.from_pandas(batch, preserve_index=False)
    if module == "polars" and hasattr(batch, "to_arrow"):
        return batch.to_arrow()
    raise SourceError("BatchSource must yield eager Arrow, pandas, or Polars batches")
