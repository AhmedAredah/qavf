"""Small-file, oversized-cell and resampling pressure with independent truth."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from qavf import Dataset, File, Status, compare_datasets, validate
from qavf.inference import paired_bootstrap

from .models import PreparedWorkload, WorkloadOptions
from .policy import resources
from .storage import read_json, write_json, write_parquet_chunks


def prepare_fixture(
    directory: Path,
    *,
    rows: int,
    seed: int,
    partitions: int = 4,
    chunk_rows: int = 65536,
    **_options: Any,
) -> dict[str, Any]:
    small = directory / "small"
    small.mkdir(parents=True, exist_ok=True)
    count = min(rows, max(64, partitions))
    for partition in range(count):

        def frames(partition=partition):
            for start in range(partition, rows, count * chunk_rows):
                key = pl.Series(
                    "id", range(start, min(rows, start + count * chunk_rows), count), dtype=pl.Int64
                )
                yield pl.DataFrame(key).with_columns(
                    pl.when(pl.col("id") % 101 == 0).then(-1).otherwise(1).alias("value")
                )

        write_parquet_chunks(small / f"part-{partition:06d}.parquet", frames())
    # Bound fixture-generation memory while retaining a single oversized cell.
    cell_bytes = max(1024, min(rows * 32, 8 * 1024 * 1024))
    text = "x" * cell_bytes
    for side in ("baseline", "candidate"):
        pl.DataFrame({"id": [0, 1], "payload": [text + side, "unchanged"]}).write_parquet(
            directory / f"{side}.parquet"
        )
    manifest = {
        "rows": rows,
        "seed": seed,
        "fixture_version": 1,
        "files": count,
        "cell_bytes": cell_bytes,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def _small_files(
    fixture: Path, work_dir: Path, options: WorkloadOptions, manifest: dict
) -> PreparedWorkload:
    rows = manifest["rows"]
    source = Dataset(
        File(str(fixture / "small/*.parquet")), schema={"columns": {"value": {"ge": 0}}}
    )
    config = {
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "run_unconfigured_rules": False},
        "rules": {"INP-003": {}},
    }
    policy = resources(work_dir / "temporary", options)

    def verify(report):
        failed = [result for result in report.results if result.status is Status.FAIL]
        if (
            report.incomplete
            or len(failed) != 1
            or failed[0].affected_count != (rows + 100) // 101
            or failed[0].total_records != rows
        ):
            raise AssertionError("small-file coverage/count oracle mismatch")
        return {
            "verified": True,
            "rows": rows,
            "files": manifest["files"],
            "failed_rows": (rows + 100) // 101,
        }

    return PreparedWorkload(
        lambda: validate({"parts": source}, config=config, resources=policy),
        verify,
        {
            "api": "qavf.validate",
            "input_rows": rows,
            "files": manifest["files"],
            "timed_scope": "glob expansion, metadata, reads and bounds validation",
        },
    )


def _huge_cells(
    fixture: Path, work_dir: Path, options: WorkloadOptions, manifest: dict
) -> PreparedWorkload:
    policy = resources(work_dir / "temporary", options)

    def run():
        return compare_datasets(
            fixture / "baseline.parquet",
            fixture / "candidate.parquet",
            keys="id",
            levels=["records"],
            resources=policy,
            entry_points=False,
        ).model_dump(mode="json")

    def verify(result):
        records = result["levels"][0]
        if (
            records["matched"] != 2
            or records["rows_changed"] != 1
            or records["values_changed"] != 1
            or records["rows_unchanged"] != 1
        ):
            raise AssertionError("huge-cell exact change oracle mismatch")
        return {"verified": True, "changed": 1, "cell_bytes": manifest["cell_bytes"]}

    return PreparedWorkload(
        run,
        verify,
        {
            "api": "qavf.compare_datasets",
            "input_rows": 4,
            "cell_bytes": manifest["cell_bytes"],
            "timed_scope": "oversized scalar comparison and bounded evidence",
        },
    )


def _resampling(manifest: dict) -> PreparedWorkload:
    rows, seed, resamples = manifest["rows"], manifest["seed"] % (2**64), 99
    baseline = np.zeros(rows, dtype=np.float64)
    candidate = np.arange(rows, dtype=np.int64) % 2 + 1.0

    def run():
        return paired_bootstrap(
            baseline, candidate, seed=seed, resamples=resamples, max_work=rows * resamples
        ).model_dump(mode="json")

    def verify(result):
        # Separate arithmetic oracle: sampled odd-index count determines each
        # mean; percentile interpolation is written here without QAVF helpers.
        rng = np.random.Generator(np.random.PCG64(seed))
        means = sorted(
            1 + int(np.count_nonzero(rng.integers(0, rows, size=rows) % 2)) / rows
            for _ in range(resamples)
        )

        def quantile(probability):
            position = (resamples - 1) * probability
            index = math.floor(position)
            weight = position - index
            return means[index] * (1 - weight) + means[min(index + 1, resamples - 1)] * weight

        expected = {
            "effect": 1 + (rows // 2) / rows,
            "p_value": 2 / (resamples + 1),
            "ci_low": quantile(0.025),
            "ci_high": quantile(0.975),
        }
        if (
            not result["assessed"]
            or result["n"] != {"pairs": rows}
            or result["resamples"] != resamples
            or result["seed"] != seed
            or any(
                not math.isclose(result[key], value, rel_tol=1e-13, abs_tol=1e-14)
                for key, value in expected.items()
            )
        ):
            raise AssertionError("resampling independent-count/percentile oracle mismatch")
        return {"verified": True, "rows": rows, "resamples": resamples, **expected}

    return PreparedWorkload(
        run,
        verify,
        {
            "api": "qavf.inference.paired_bootstrap",
            "input_rows": rows * 2,
            "resamples": resamples,
            "timed_scope": "public bootstrap; input arrays prepared in setup",
            "oracle": "separate PCG64 integer draws, parity counts and linear quantiles",
        },
    )


def build_workload(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions
) -> PreparedWorkload:
    manifest = read_json(fixture / "manifest.json")
    if case == "many_small_files":
        return _small_files(fixture, work_dir, options, manifest)
    if case == "huge_cells":
        return _huge_cells(fixture, work_dir, options, manifest)
    if case == "resampling_inference":
        return _resampling(manifest)
    raise ValueError(f"Unknown stress case: {case}")
