"""Measure exact owned observations across producer fragmentation, without data files.

Run each source/repetition in a fresh process, e.g.:
python scripts/benchmark_batch_spool.py --source . --rows 100000 --batch-rows 1
Logical bytes and filesystem allocation are separate; peak RSS includes imports.
Use --trace for Python allocation evidence, separately from timing comparisons.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path


def runtime_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((source / "qavf").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--batch-rows", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.batch_rows < 1:
        parser.error("row counts must be positive")
    before = runtime_digest(args.source)
    sys.path.insert(0, str(args.source.resolve()))
    import polars as pl

    from qavf import BatchSource, ResourceLimits, __version__
    from qavf.engine.context import DataContext
    from qavf.inputs.normalize import prepare

    @contextmanager
    def batches():
        yield (
            pl.DataFrame({"x": range(start, min(args.rows, start + args.batch_rows))})
            for start in range(0, args.rows, args.batch_rows)
        )

    with tempfile.TemporaryDirectory() as directory:
        settings, frames = prepare(
            BatchSource(batches, schema={"x": pl.Int64}),
            schema={},
            config=None,
            resources=ResourceLimits(
                cache_memory=0, target_memory="16MiB", max_spill="256MiB", temp_dir=directory
            ),
        )
        gc.collect()
        if args.trace:
            tracemalloc.start()
        started = time.perf_counter()
        with DataContext.from_settings(settings, frames) as ctx, ctx.observe("data") as observation:
            paths = [path for path in Path(directory).rglob("*") if path.is_file()]
            measured = {
                "files": len(paths),
                "logical_bytes": sum(path.stat().st_size for path in paths),
                "allocated_bytes": sum(
                    getattr(path.stat(), "st_blocks", 0) * 512 for path in paths
                ),
                "resources": ctx.resource_usage(),
                "python_allocations": tracemalloc.get_traced_memory() if args.trace else None,
            }
            result = observation.reduce(
                lambda frame: frame.select(pl.len(), pl.col("x").sum())
            ).row(0)
            expected = (args.rows, args.rows * (args.rows - 1) // 2)
            if result != expected:
                raise RuntimeError(f"observation mismatch: {result} != {expected}")
        measured["seconds"] = time.perf_counter() - started
        measured["cleanup_empty"] = not list(Path(directory).iterdir())
        if not measured["cleanup_empty"]:
            raise RuntimeError("owned state was not cleaned up")
        if args.trace:
            tracemalloc.stop()
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        peak_rss = None
    if runtime_digest(args.source) != before:
        raise RuntimeError("source changed during benchmark")
    print(
        json.dumps(
            {
                "version": __version__,
                "runtime_source_sha256": before,
                "platform": platform.platform(),
                "rows": args.rows,
                "batch_rows": args.batch_rows,
                "trace": args.trace,
                "process_peak_rss_native_units": peak_rss,
                "rss_units": "KiB on Linux; bytes on macOS; unavailable on Windows",
                "outcome": list(result),
                **measured,
            }
        )
    )


if __name__ == "__main__":
    main()
