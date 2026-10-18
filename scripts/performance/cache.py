"""Real owned-frame spill, repeated reads, promotion and reuse through DataContext.

The cache allocation deliberately fits one input frame, but not two. This is a
tracked cache-retention budget, not a claim that process RSS is bounded by it.
Actual IPC files and QAVF's public resource counters must agree for a run to pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from qavf.engine.context import DataContext
from qavf.sources import FileParams, FileSource, SourceEnvironment

from .models import PreparedWorkload, WorkloadOptions
from .storage import read_json, write_json, write_parquet_chunks


def _expected(rows: int, seed: int) -> dict[str, int]:
    total = rows * (rows - 1) // 2
    complete, tail = divmod(rows, 17)
    return {
        "rows": rows,
        "id_min": 0,
        "id_max": rows - 1,
        "id_sum": total,
        "trip_sum": total + rows * (seed % 1_000_000),
        "load_sum": 3 * total + 7 * rows,
        "category_sum": complete * 136 + tail * (tail - 1) // 2,
        "all_values_exact": True,
    }


def prepare_fixture(
    directory: Path,
    *,
    rows: int,
    seed: int,
    partitions: int = 4,
    chunk_rows: int = 65536,
) -> dict[str, Any]:
    """Write two scalar Parquet datasets in bounded, independently reproducible chunks."""
    if rows < 1 or chunk_rows < 1 or partitions < 1:
        raise ValueError("rows, chunk_rows and partitions must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"a": "a.parquet", "b": "b.parquet"}
    for relative in paths.values():

        def frames():
            for start in range(0, rows, chunk_rows):
                frame = pl.DataFrame(
                    {"id": pl.Series(range(start, min(rows, start + chunk_rows)), dtype=pl.Int64)}
                )
                yield frame.with_columns(
                    (pl.col("id") + seed % 1_000_000).alias("trip"),
                    (pl.col("id") * 3 + 7).alias("load"),
                    (pl.col("id") % 17).alias("category"),
                )

        write_parquet_chunks(directory / relative, frames())
    manifest = {
        "case": "cache_spill",
        "fixture_version": 1,
        "rows": rows,
        "seed": seed,
        "chunk_rows": chunk_rows,
        "files": paths,
        "frame_data_bytes": rows * 32,
        "expected": _expected(rows, seed),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def _summary(frame: pl.DataFrame, seed: int) -> dict[str, Any]:
    return frame.select(
        pl.len().alias("rows"),
        pl.col("id").min().alias("id_min"),
        pl.col("id").max().alias("id_max"),
        pl.col("id").sum().alias("id_sum"),
        pl.col("trip").sum().alias("trip_sum"),
        pl.col("load").sum().alias("load_sum"),
        pl.col("category").sum().alias("category_sum"),
        pl.all_horizontal(
            pl.col("id") == pl.int_range(0, pl.len(), dtype=pl.Int64),
            pl.col("trip") == pl.col("id") + seed % 1_000_000,
            pl.col("load") == pl.col("id") * 3 + 7,
            pl.col("category") == pl.col("id") % 17,
        )
        .fill_null(False)
        .all()
        .alias("all_values_exact"),
    ).row(0, named=True)


def _snapshot(ctx: DataContext, directory: Path) -> dict[str, Any]:
    paths = list(directory.rglob("*.arrow"))
    return {
        "usage": ctx.resource_usage(),
        "physical_spill_files": len(paths),
        "physical_spill_bytes": sum(path.stat().st_size for path in paths),
    }


def build_workload(fixture: Path, work_dir: Path, options: WorkloadOptions) -> PreparedWorkload:
    """Exercise the public context lifecycle and verify real spill/reuse transitions."""
    if options.backend != "context" or options.input_format != "parquet":
        raise ValueError("cache_spill supports only backend='context', input_format='parquet'")
    manifest = read_json(fixture / "manifest.json")
    rows, seed = manifest["rows"], manifest["seed"]
    expected = _expected(rows, seed)
    if expected != manifest["expected"]:
        raise ValueError("cache fixture manifest has an inconsistent oracle")
    # 3/4 of the public cache budget belongs to frames; two other caches share 1/4.
    frame_bytes = rows * 32
    cache_bytes = options.cache_max_bytes if options.cache_max_bytes is not None else rows * 48
    spill_bytes = (
        options.spill_max_bytes if options.spill_max_bytes is not None else max(4 << 20, rows * 256)
    )
    effective_cache = (
        min(cache_bytes, options.memory_budget_bytes // 2)
        if options.memory_budget_bytes
        else cache_bytes
    )
    frame_allocation = effective_cache * 3 // 4
    if not frame_bytes <= frame_allocation < 2 * frame_bytes:
        raise ValueError("cache_spill requires a cache allocation that fits one frame but not two")
    if spill_bytes // 2 <= frame_bytes:
        raise ValueError("cache_spill needs a frame spill allocation larger than one frame")
    temporary = work_dir / "context-temporary"
    env = SourceEnvironment(base_dir=fixture.resolve(), threads=options.threads)

    def run() -> dict[str, Any]:
        sources = {
            name: FileSource(FileParams(path=relative), env)
            for name, relative in manifest["files"].items()
        }
        snapshots: dict[str, Any] = {}
        summaries: dict[str, Any] = {}
        with DataContext(
            sources=sources,
            env=env,
            cache_max_bytes=cache_bytes,
            spill_max_bytes=spill_bytes,
            memory_budget_bytes=options.memory_budget_bytes,
            temp_directory=temporary,
            hash_inputs=options.hash_inputs,
            connection_slots=1,
        ) as ctx:
            summaries["load_a"] = _summary(ctx.get("a"), seed)
            snapshots["load_a"] = _snapshot(ctx, temporary)
            summaries["load_b"] = _summary(ctx.get("b"), seed)
            snapshots["load_b"] = _snapshot(ctx, temporary)
            metadata = [info.model_dump(mode="json") for info in ctx.describe()]
            for name in ("read_a_1", "read_a_2"):
                summaries[name] = _summary(ctx.get("a"), seed)
                snapshots[name] = _snapshot(ctx, temporary)
            ctx.release(["b"])
            snapshots["release_b"] = _snapshot(ctx, temporary)
            summaries["promote_a"] = _summary(ctx.get("a"), seed)
            snapshots["promote_a"] = _snapshot(ctx, temporary)
            summaries["hit_a"] = _summary(ctx.get("a"), seed)
            snapshots["hit_a"] = _snapshot(ctx, temporary)
            final_metadata = [info.model_dump(mode="json") for info in ctx.describe()]
        return {
            "summaries": summaries,
            "snapshots": snapshots,
            "metadata": metadata,
            "final_metadata": final_metadata,
            "closed_metadata": [info.model_dump(mode="json") for info in ctx.describe()],
            "physical_files_after_close": len(list(temporary.rglob("*.*"))),
            "owned_directories_after_close": len(list(temporary.glob("qavf-*"))),
        }

    def verify(result: dict[str, Any]) -> dict[str, Any]:
        stages = {"load_a", "load_b", "read_a_1", "read_a_2", "promote_a", "hit_a"}
        if set(result["summaries"]) != stages or any(
            value != expected for value in result["summaries"].values()
        ):
            raise AssertionError("cache/spill changed exact dataset values")
        snapshots = result["snapshots"]
        load = snapshots["load_b"]
        physical_bytes = load["physical_spill_bytes"]
        if load["physical_spill_files"] != 1 or physical_bytes <= 0:
            raise AssertionError("cache workload did not create an owned IPC spill file")
        transitions = {
            "load_a": (0, 0, 0, 0),
            "load_b": (1, 0, 0, physical_bytes),
            "read_a_1": (1, 1, 0, physical_bytes),
            "read_a_2": (1, 2, 0, physical_bytes),
            "release_b": (1, 2, 0, physical_bytes),
            "promote_a": (1, 3, 1, 0),
            "hit_a": (1, 3, 1, 0),
        }
        for name, (writes, reads, promotions, retained) in transitions.items():
            snapshot = snapshots[name]
            usage = snapshot["usage"]
            expected_usage = {
                "frame_spill_write_count": writes,
                "frame_spill_read_count": reads,
                "frame_spill_promotions": promotions,
                "frame_spill_write_bytes": writes * physical_bytes,
                "frame_spill_read_bytes": reads * physical_bytes,
                "frame_spill_bytes": retained,
            }
            if any(usage[key] != value for key, value in expected_usage.items()):
                raise AssertionError(f"incorrect cache/spill counters at {name}")
            if snapshot["physical_spill_bytes"] != retained or snapshot[
                "physical_spill_files"
            ] != int(retained > 0):
                raise AssertionError(f"physical IPC files disagree with counters at {name}")
            if (
                usage["retained_cache_bytes"] > effective_cache
                or usage["peak_frame_spill_bytes"] > spill_bytes // 2
            ):
                raise AssertionError("cache or frame spill retention exceeded its allocation")
        if (
            result["metadata"] != result["final_metadata"]
            or result["metadata"] != result["closed_metadata"]
        ):
            raise AssertionError("cache eviction, promotion or close changed dataset metadata")
        if len(result["metadata"]) != 2 or any(info["rows"] != rows for info in result["metadata"]):
            raise AssertionError("cache metadata lacks exact source row counts")
        if result["physical_files_after_close"] or result["owned_directories_after_close"]:
            raise AssertionError("context close left owned temporary data on disk")
        return {
            "correct": True,
            "dataset_rows": rows,
            "datasets": 2,
            "spill_write_count": 1,
            "spill_read_count": 3,
            "spill_promotions": 1,
            "spill_write_bytes": physical_bytes,
            "spill_read_bytes": physical_bytes * 3,
            "retained_spill_bytes_after_promotion": 0,
            "temporary_files_after_close": 0,
            "metadata_preserved": True,
        }

    return PreparedWorkload(
        run=run,
        verify=verify,
        metadata={
            "input_rows": rows * 2,
            "api": "qavf.engine.context.DataContext",
            "timed_scope": (
                "load A/B, read spilled A twice, release B, promote A, "
                "cache hit, exact columnar value summaries, metadata and cleanup"
            ),
            "cache_max_bytes": cache_bytes,
            "frame_cache_allocation_bytes": frame_allocation,
            "spill_max_bytes": spill_bytes,
            "memory_scope": "tracked retained cache budget; process RSS is measured separately",
        },
    )
