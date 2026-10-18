"""Keyed output comparison with sparse scalar and nested changes.

Fixture construction uses bounded batches. The timed operation includes public
FileSource reads with an explicit native thread policy and public compare_frames
execution. Comparison materializes both tables; it is not a streaming backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from qavf.regression import compare_frames
from qavf.sources import FileParams, FileSource, SourceEnvironment

from .models import PreparedWorkload, WorkloadOptions
from .storage import read_json, write_json, write_parquet_chunks

_PERIOD = 101


def _residue_count(rows: int, seed: int, residue: int) -> int:
    first = (residue - seed) % _PERIOD
    return 0 if first >= rows else 1 + (rows - 1 - first) // _PERIOD


def _expected(rows: int, seed: int) -> dict[str, int]:
    counts = [_residue_count(rows, seed, index) for index in range(6)]
    return {
        "baseline_rows": rows,
        "candidate_rows": rows,
        "rows_added": counts[0],
        "rows_removed": counts[0],
        "rows_changed": sum(counts[1:]),
        "values_changed": sum(counts[1:]) + counts[5],
        "nonfinite_changes": 0,
    }


def _verify_evidence(evidence: list[dict[str, Any]], rows: int, seed: int) -> None:
    """Check bounded findings using scalar equations, without reading either table."""
    seen: set[tuple[str, int, str | None]] = set()
    for difference in evidence:
        if not isinstance(difference, dict):
            raise AssertionError("regression evidence must contain difference objects")
        key = difference.get("key")
        if not isinstance(key, dict) or set(key) != {"id"} or type(key["id"]) is not int:
            raise AssertionError("regression evidence has an invalid row key")
        kind, column, identifier = (
            difference.get("kind"),
            difference.get("column"),
            key["id"],
        )
        original = identifier - rows if kind == "row_added" else identifier
        event = (original + seed) % _PERIOD
        if not 0 <= original < rows:
            raise AssertionError("regression evidence identifies a nonexistent fixture row")
        expected: dict[str, Any] = {
            "kind": kind,
            "key": {"id": identifier},
            "column": None,
            "path": [],
            "baseline": None,
            "candidate": None,
            "baseline_previewed": False,
            "candidate_previewed": False,
            "baseline_length": None,
            "candidate_length": None,
            "leaf": {},
        }
        if kind in {"row_added", "row_removed"}:
            if event != 0:
                raise AssertionError("regression evidence invents an added or removed row")
        elif kind == "value_changed":
            if column == "amount" and event in {1, 5}:
                before: Any = float(original % 10000 + 1000)
                after: Any = before + 100
            elif column == "status" and event in {2, 5}:
                before, after = f"status-{original % 7}", "review"
            elif column == "readings" and event == 3:
                before = [original % modulus for modulus in (13, 17, 19, 23)]
                after = [*before[:3], before[3] + 3]
                expected["path"] = [3]
            elif column == "details" and event == 4:
                before = {"code": original % 29, "score": original % 31}
                after = {**before, "score": before["score"] + 1}
                expected["path"] = ["score"]
            else:
                raise AssertionError("regression evidence invents a changed cell")
            expected.update(column=column, baseline=before, candidate=after)
        else:
            raise AssertionError("regression evidence has an unexpected difference kind")
        # Canonical JSON also distinguishes Boolean/integer/float payloads that
        # Python's ordinary equality would treat as the same value.
        observed = {field: difference[field] for field in expected if field in difference}
        if json.dumps(observed, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise AssertionError("regression evidence values or paths differ from fixture truth")
        identity = kind, identifier, column
        if identity in seen:
            raise AssertionError("regression evidence repeats a difference")
        seen.add(identity)


def _batch(start: int, stop: int, rows: int, seed: int, *, candidate: bool) -> pl.DataFrame:
    frame = pl.DataFrame({"id": pl.Series(range(start, stop), dtype=pl.Int64)})
    key = pl.col("id")
    event = (key + seed % _PERIOD) % _PERIOD
    changed = pl.lit(candidate)
    amount = (key % 10000).cast(pl.Float64) + 1000
    return frame.select(
        pl.when(changed & (event == 0)).then(key + rows).otherwise(key).alias("id"),
        (amount + pl.when(changed & event.is_in([1, 5])).then(100).otherwise(0)).alias("amount"),
        pl.when(changed & event.is_in([2, 5]))
        .then(pl.lit("review"))
        .otherwise(pl.concat_str(pl.lit("status-"), (key % 7).cast(pl.String)))
        .alias("status"),
        pl.concat_list(
            key % 13,
            key % 17,
            key % 19,
            key % 23 + pl.when(changed & (event == 3)).then(3).otherwise(0),
        ).alias("readings"),
        pl.struct(
            (key % 29).alias("code"),
            (key % 31 + pl.when(changed & (event == 4)).then(1).otherwise(0)).alias("score"),
        ).alias("details"),
        (amount + pl.when(changed & (event == 6)).then(1e-8).otherwise(0)).alias("measurement"),
        *[
            ((key % (71 + index)).cast(pl.Float64) / 8).alias(f"metric_{index}")
            for index in range(6)
        ],
    )


def prepare_fixture(
    directory: Path,
    *,
    rows: int,
    seed: int,
    partitions: int = 4,
    chunk_rows: int = 65536,
) -> dict[str, Any]:
    """Write deterministic before/after outputs without building whole fixtures in RAM."""
    if rows < 1 or chunk_rows < 1 or partitions < 1:
        raise ValueError("rows, chunk_rows and partitions must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"baseline": "baseline.parquet", "candidate": "candidate.parquet"}
    for name, relative in paths.items():
        candidate = name == "candidate"

        def frames(candidate: bool = candidate):
            for start in range(0, rows, chunk_rows):
                frame = _batch(
                    start, min(rows, start + chunk_rows), rows, seed, candidate=candidate
                )
                # Reordering within every batch makes row-position comparisons wrong.
                yield frame.reverse() if candidate else frame

        write_parquet_chunks(directory / relative, frames())
    manifest = {
        "case": "regression",
        "fixture_version": 1,
        "rows": rows,
        "seed": seed,
        "chunk_rows": chunk_rows,
        "files": paths,
        "expected": _expected(rows, seed),
        "features": [
            "keyed_reordering",
            "added_removed_rows",
            "numeric",
            "categorical",
            "list",
            "struct",
            "within_tolerance_noise",
        ],
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def build_workload(fixture: Path, work_dir: Path, options: WorkloadOptions) -> PreparedWorkload:
    """Build a public file-comparison operation and independent count/evidence checks."""
    if options.backend != "file" or options.input_format != "parquet":
        raise ValueError("regression supports only backend='file', input_format='parquet'")
    manifest = read_json(fixture / "manifest.json")
    expected = _expected(manifest["rows"], manifest["seed"])
    if manifest["expected"] != expected:
        raise ValueError("regression fixture manifest has an inconsistent oracle")
    paths = {name: fixture / value for name, value in manifest["files"].items()}

    def run() -> dict[str, Any]:
        reader_directory = work_dir / "readers"
        reader_directory.mkdir(parents=True, exist_ok=True)
        environment = SourceEnvironment(
            threads=options.threads,
            temp_directory=reader_directory,
            memory_limit=f"{options.memory_budget_bytes}B"
            if options.memory_budget_bytes is not None
            else None,
            spill_limit_bytes=options.spill_max_bytes,
        )
        frames = {
            name: FileSource(FileParams(path=str(path.resolve())), environment).load()
            for name, path in paths.items()
        }
        result = compare_frames(
            frames["baseline"],
            frames["candidate"],
            keys=["id"],
            rtol=0,
            atol=1e-6,
            max_differences=options.evidence_rows,
            max_evidence_bytes=options.evidence_bytes,
            schema="strict",
            name="keyed_output",
        )
        return result.model_dump(mode="json")

    def verify(result: dict[str, Any]) -> dict[str, Any]:
        for key, count in expected.items():
            if result[key] != count:
                raise AssertionError(f"regression {key}: expected {count}, observed {result[key]}")
        if any(result[key] for key in ("columns_added", "columns_removed", "dtypes_changed")):
            raise AssertionError("regression schema changed unexpectedly")
        evidence = result["differences"]
        encoded_bytes = len(
            json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if len(evidence) > options.evidence_rows or encoded_bytes > max(2, options.evidence_bytes):
            raise AssertionError("regression evidence exceeded a configured bound")
        total = expected["values_changed"] + expected["rows_added"] + expected["rows_removed"]
        if result["truncated"] != (len(evidence) < total):
            raise AssertionError("regression truncation disagrees with complete counts")
        if (options.evidence_rows == 0 or options.evidence_bytes < 3) and evidence:
            raise AssertionError("disabled regression evidence was retained")
        _verify_evidence(evidence, manifest["rows"], manifest["seed"])
        return {
            "correct": True,
            **expected,
            "evidence_rows": len(evidence),
            "evidence_bytes": encoded_bytes,
            "truncated": result["truncated"],
        }

    return PreparedWorkload(
        run=run,
        verify=verify,
        metadata={
            "input_rows": 2 * manifest["rows"],
            "api": "qavf.sources.FileSource.load + qavf.regression.compare_frames",
            "timed_scope": (
                "load both Parquet files with controlled reader threads, "
                "compare keyed values, serialize bounded result"
            ),
            "native_reader_threads": options.threads,
            "memory_scope": (
                "in-memory comparison; both tables are materialized; optional memory/spill "
                "limits apply to native readers, not whole-process RSS or retained tables"
            ),
            "expected": expected,
        },
    )
