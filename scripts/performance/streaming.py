"""Distinct-file public regression with scalar data and independent arithmetic truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from qavf.regression import compare_outputs

from .models import PreparedWorkload, WorkloadOptions
from .policy import resources
from .storage import read_json, write_json, write_parquet_chunks


def count(rows: int, residue: int) -> int:
    return max(0, (rows + 100 - residue) // 101)


def prepare_fixture(
    directory: Path, *, rows: int, seed: int, chunk_rows: int = 65536, **_options: Any
) -> dict[str, Any]:
    for side in ("baseline", "candidate"):

        def frames(candidate=side == "candidate"):
            for start in range(0, rows, chunk_rows):
                stop = min(rows, start + chunk_rows)
                frame = pl.DataFrame({"id": pl.Series(range(start, stop), dtype=pl.Int64)})
                key = pl.col("id")
                value = frame.select(
                    pl.when(pl.lit(candidate) & (key % 101 == 0))
                    .then(key + rows)
                    .otherwise(key)
                    .alias("id"),
                    (key * 2 + int(candidate) * (key % 101 == 1).cast(pl.Int64)).alias("value"),
                    pl.when(pl.lit(candidate) & (key % 101 == 2))
                    .then(pl.lit("changed"))
                    .otherwise(pl.concat_str(pl.lit("label-"), key % 7))
                    .alias("label"),
                )
                yield value.reverse() if candidate else value

        write_parquet_chunks(directory / f"{side}.parquet", frames())
    manifest = {
        "rows": rows,
        "seed": seed,
        "fixture_version": 1,
        "rows_added": count(rows, 0),
        "rows_removed": count(rows, 0),
        "rows_changed": count(rows, 1) + count(rows, 2),
        "values_changed": count(rows, 1) + count(rows, 2),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def build_workload(fixture: Path, work_dir: Path, options: WorkloadOptions) -> PreparedWorkload:
    manifest = read_json(fixture / "manifest.json")
    rows = manifest["rows"]
    expected = {
        "rows_added": count(rows, 0),
        "rows_removed": count(rows, 0),
        "rows_changed": count(rows, 1) + count(rows, 2),
        "values_changed": count(rows, 1) + count(rows, 2),
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("streaming fixture oracle changed")
    policy = resources(work_dir / "temporary", options)

    def run():
        return compare_outputs(
            fixture / "baseline.parquet",
            fixture / "candidate.parquet",
            keys="id",
            rtol=0,
            atol=0,
            resources=policy,
            max_differences=options.evidence_rows,
            max_evidence_bytes=options.evidence_bytes,
        ).model_dump(mode="json")

    def verify(result):
        for key, value in {**expected, "baseline_rows": rows, "candidate_rows": rows}.items():
            if result[key] != value:
                raise AssertionError(f"streaming {key}: {result[key]} != {value}")
        if result["nonfinite_changes"] or any(
            result[key] for key in ("columns_added", "columns_removed", "dtypes_changed")
        ):
            raise AssertionError("streaming comparison schema/verdict changed")
        if result["execution"] != "native_stream":
            raise AssertionError(f"expected streaming comparison, got {result['execution']}")
        evidence = result["differences"]
        if len(evidence) > options.evidence_rows or len(
            json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode()
        ) > max(2, options.evidence_bytes):
            raise AssertionError("streaming evidence bounds exceeded")
        for item in evidence:
            identifier = item["key"]["id"]
            kind = item["kind"]
            if kind == "row_added":
                valid = rows <= identifier < 2 * rows and (identifier - rows) % 101 == 0
            elif kind == "row_removed":
                valid = 0 <= identifier < rows and identifier % 101 == 0
            else:
                before = identifier * 2 if item["column"] == "value" else f"label-{identifier % 7}"
                after = before + 1 if item["column"] == "value" else "changed"
                residue = 1 if item["column"] == "value" else 2
                valid = (
                    kind == "value_changed"
                    and item["column"] in {"value", "label"}
                    and 0 <= identifier < rows
                    and identifier % 101 == residue
                    and item["baseline"] == before
                    and item["candidate"] == after
                )
            if not valid:
                raise AssertionError("streaming evidence contradicts fixture truth")
        return {"verified": True, **expected, "execution": result["execution"]}

    return PreparedWorkload(
        run,
        verify,
        {
            "api": "qavf.regression.compare_outputs",
            "input_rows": rows * 2,
            "distinct_files": True,
            "execution_mode": options.execution_mode,
            "expected": expected,
            "timed_scope": "public call including startup and cleanup",
        },
    )
