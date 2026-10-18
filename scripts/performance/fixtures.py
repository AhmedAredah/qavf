"""Deterministic transaction fixtures and independent, integer-only truth.

Preparation writes bounded chunks directly to each format. The only Python row
records are the control ledger (at most 1,024 groups), never the fact table.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

FIXTURE_VERSION = 1
PERIOD = 1_000
FEATURES = 8
TEXT_COLUMNS = 4


def content_fingerprint(
    directory: Path, dataset: str, input_format: str, declared: dict[str, Any]
) -> dict[str, Any]:
    """Independently establish the expected public file-glob fingerprint in setup.

    QAVF's listing identity is SHA-256 over ordered tab-separated path, byte
    count, UTC modification timestamp, and file SHA-256 records. Compute this
    with the standard library, never a QAVF identity helper or timed operation.
    """
    manifest_files = declared["files"][input_format]
    expected = {directory / item["path"]: item for item in manifest_files}
    actual = set((directory / input_format / dataset).glob(f"*.{input_format}"))
    if len(expected) != len(manifest_files) or actual != expected.keys():
        raise ValueError(f"fixture file coverage differs from manifest: {dataset}")
    paths = sorted(actual)
    listing = hashlib.sha256()
    total, latest = 0, ""
    for path in paths:
        before = path.stat()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"fixture changed while preparing its identity: {path}")
        if after.st_size != expected[path]["bytes"] or digest != expected[path]["sha256"]:
            raise ValueError(f"fixture content differs from manifest: {path}")
        modified = datetime.fromtimestamp(after.st_mtime, UTC).isoformat()
        listing.update(f"{path}\t{after.st_size}\t{modified}\t{digest}\n".encode())
        total += after.st_size
        latest = max(latest, modified)
    return {
        "files": len(paths),
        "identity": "content",
        "bytes": total,
        "modified": latest,
        "listing_sha256": listing.hexdigest(),
        "paths": [str(path) for path in paths[:1000]],
        "paths_truncated": len(paths) > 1000,
    }


def residue_count(rows: int, residue: int, period: int = PERIOD) -> int:
    """Count indices in ``range(rows)`` with the requested remainder."""
    return max(0, (rows - 1 - residue) // period + 1)


def _floor_sum(n: int, modulus: int, slope: int, intercept: int) -> int:
    """Sum floor((slope*k + intercept)/modulus), using Euclidean reduction."""
    answer = 0
    while True:
        quotient, slope = divmod(slope, modulus)
        answer += (n - 1) * n * quotient // 2
        quotient, intercept = divmod(intercept, modulus)
        answer += n * quotient
        top = slope * n + intercept
        if top < modulus:
            return answer
        n, intercept = divmod(top, modulus)
        modulus, slope = slope, modulus


def group_truth(rows: int, seed: int, groups: int) -> list[dict[str, int]]:
    """Exact totals from the generator equation, without reading generated files."""
    truth = []
    for group in range(groups):
        count = residue_count(rows, group, groups)
        slope, intercept = 37 * groups, 37 * group + seed % 100_000
        cents = (
            100 * count
            + count * intercept
            + slope * count * (count - 1) // 2
            - 100_000 * _floor_sum(count, 100_000, slope, intercept)
        )
        truth.append({"group_id": group, "amount_cents": cents, "row_count": count})
    return truth


def _facts(start: int, stop: int, seed: int, dimension: int, groups: int) -> pl.DataFrame:
    frame = pl.DataFrame({"row_id": pl.arange(start, stop, eager=True, dtype=pl.Int64)})
    row, remainder = pl.col("row_id"), pl.col("row_id") % PERIOD
    frame = frame.with_columns(
        pl.when(remainder == 999).then(row - 1).otherwise(row).alias("transaction_id"),
        pl.when(remainder == 0)
        .then(None)
        .when(remainder == 1)
        .then(-999)
        .when(remainder == 2)
        .then(1_000_000_000 + row // PERIOD % 7)
        .when(remainder < 803)
        .then(0)
        .otherwise(1 + row // PERIOD % (dimension - 1))
        .cast(pl.Int64)
        .alias("customer_id"),
        (row % groups).alias("group_id"),
        (100 + (row * 37 + seed % 100_000) % 100_000).alias("amount_cents"),
        pl.when(remainder == 3)
        .then(None)
        .when(remainder == 4)
        .then(-999)
        .when(remainder == 5)
        .then(-1)
        .otherwise(1 + (row + seed % 100) % 100)
        .cast(pl.Int64)
        .alias("quantity"),
        pl.when(remainder == 6)
        .then(pl.lit("INVALID"))
        .when(row % 3 == 0)
        .then(pl.lit("OPEN"))
        .when(row % 3 == 1)
        .then(pl.lit("CLOSED"))
        .otherwise(pl.lit("HOLD"))
        .alias("status"),
        pl.when(remainder == 7)
        .then(pl.lit("broken"))
        .otherwise(pl.concat_str(pl.lit("C"), (row % dimension).cast(pl.String).str.zfill(6)))
        .alias("customer_code"),
        *[
            (((row * (index * 2 + 3) + seed % 10_000) % 10_000) / 100).alias(f"measure_{index:02d}")
            for index in range(FEATURES)
        ],
        *[
            pl.concat_str(
                pl.lit(f"batch-{index}|regional shipment|"),
                (row % 97).cast(pl.String),
                pl.lit("|reference-"),
                row.cast(pl.String),
                pl.lit("|" + "loading bay inspection; " * (index + 1)),
            ).alias(f"description_{index:02d}")
            for index in range(TEXT_COLUMNS)
        ],
    )
    return frame


def _write_dataset(
    directory: Path, name: str, rows: int, partitions: int, chunk_rows: int, make_chunk: Any
) -> dict[str, Any]:
    schema: dict[str, str] = {}
    files: dict[str, list[dict[str, Any]]] = {"parquet": [], "csv": []}
    actual_partitions = min(partitions, rows)
    for partition in range(actual_partitions):
        start, stop = (
            rows * partition // actual_partitions,
            rows * (partition + 1) // actual_partitions,
        )
        paths = {fmt: directory / fmt / name / f"part-{partition:04d}.{fmt}" for fmt in files}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        try:
            with paths["csv"].open("wb") as csv:
                for offset in range(start, stop, chunk_rows):
                    chunk = make_chunk(offset, min(offset + chunk_rows, stop))
                    schema = {column: str(dtype) for column, dtype in chunk.schema.items()}
                    arrow = chunk.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(
                            paths["parquet"], arrow.schema, compression="zstd"
                        )
                    writer.write_table(arrow, row_group_size=chunk_rows)
                    chunk.write_csv(csv, include_header=offset == start)
        finally:
            if writer is not None:
                writer.close()
        for fmt, path in paths.items():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[fmt].append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "rows": stop - start,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    return {"rows": rows, "schema": schema, "partitions": actual_partitions, "files": files}


def prepare_fixture(
    directory: Path, *, rows: int, seed: int, partitions: int = 4, chunk_rows: int = 65_536
) -> dict[str, Any]:
    """Create a new fixture directory; refuse to overwrite a previous fixture."""
    for name, value in (("rows", rows), ("partitions", partitions), ("chunk_rows", chunk_rows)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if rows < 32:
        raise ValueError("rows must be at least 32 so every workload has meaningful coverage")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"fixture directory must be empty: {directory}")
    dimension, groups = min(4096, max(16, rows // 100)), min(1024, max(8, rows // 128))
    truth = group_truth(rows, seed, groups)
    ledger = pl.DataFrame(
        [
            {
                "group_id": item["group_id"],
                "ledger_cents": item["amount_cents"] + (item["group_id"] % 17 == 0),
                "row_count": item["row_count"],
            }
            for item in truth
            if item["group_id"] % 31 != 0
        ]
        + [{"group_id": groups + 7, "ledger_cents": 0, "row_count": 0}]
    )
    datasets = {
        "transactions": _write_dataset(
            directory,
            "transactions",
            rows,
            partitions,
            chunk_rows,
            lambda start, stop: _facts(start, stop, seed, dimension, groups),
        ),
        "customers": _write_dataset(
            directory,
            "customers",
            dimension,
            1,
            chunk_rows,
            lambda start, stop: pl.DataFrame(
                {"customer_id": pl.arange(start, stop, eager=True, dtype=pl.Int64)}
            ),
        ),
        "ledger": _write_dataset(
            directory,
            "ledger",
            ledger.height,
            1,
            chunk_rows,
            lambda start, stop: ledger.slice(start, stop - start),
        ),
    }
    orphans = residue_count(rows, 2)
    missing_groups = sum(item["group_id"] % 31 == 0 for item in truth)
    changed_groups = sum(
        item["group_id"] % 31 != 0 and item["group_id"] % 17 == 0 for item in truth
    )
    manifest = {
        "version": FIXTURE_VERSION,
        "rows": rows,
        "seed": seed,
        "chunk_rows": chunk_rows,
        "partitions_requested": partitions,
        "generator": "periodic-integer-transactions-v1",
        "dimension_rows": dimension,
        "group_count": groups,
        "datasets": datasets,
        "truth": {
            "duplicate_keys": residue_count(rows, 999),
            "quantity_missing": residue_count(rows, 3) + residue_count(rows, 4),
            "quantity_negative": residue_count(rows, 5),
            "invalid_status": residue_count(rows, 6),
            "invalid_code": residue_count(rows, 7),
            "null_customer_rows": residue_count(rows, 0) + residue_count(rows, 1),
            "orphan_rows": orphans,
            "orphan_keys": min(7, orphans),
            "foreign_key_distinct_keys": (
                1 + min(dimension - 1, residue_count(rows, 803)) + min(7, orphans)
            ),
            "reconciliation_total_failures": missing_groups + changed_groups + 1,
            "reconciliation_count_failures": missing_groups + 1,
            "group_totals": truth,
        },
        "notes": [
            "80% of each full 1,000-row block shares customer 0; tail keys vary by block.",
            "Each full block has one duplicate transaction key and fixed injected defects.",
            "Integer cents avoid a floating-point aggregate being its own correctness oracle.",
            "Input generation and input hashing here are preparation, never validation timing.",
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
