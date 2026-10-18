"""High-cardinality exact groups and scalar outliers with independent fixture truth."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import polars as pl

from qavf import Status, validate

from .models import PreparedWorkload, WorkloadOptions
from .policy import resources
from .storage import read_json, write_json, write_parquet_chunks


def prepare_fixture(
    directory: Path, *, rows: int, seed: int, chunk_rows: int = 65536, **_options: Any
) -> dict[str, Any]:
    groups = max(1, rows // 16)

    def observations():
        for start in range(0, rows, chunk_rows):
            frame = pl.DataFrame(
                {"id": pl.Series(range(start, min(rows, start + chunk_rows)), dtype=pl.Int64)}
            )
            yield frame.with_columns(
                (pl.col("id") % groups).alias("group_id"),
                pl.lit(0.5).alias("amount"),
                pl.when(pl.col("id") % 101 == 0).then(100.0).otherwise(0.0).alias("value"),
            )

    def ledger():
        for start in range(0, groups, chunk_rows):
            frame = pl.DataFrame(
                {
                    "group_id": pl.Series(
                        range(start, min(groups, start + chunk_rows)), dtype=pl.Int64
                    )
                }
            )
            group = pl.col("group_id")
            yield frame.with_columns(
                (
                    ((rows - 1 - group) // groups + 1) * 0.5 + (group % 97 == 0).cast(pl.Float64)
                ).alias("total")
            )

    write_parquet_chunks(directory / "observations.parquet", observations())
    write_parquet_chunks(directory / "ledger.parquet", ledger())
    manifest = {
        "rows": rows,
        "groups": groups,
        "seed": seed,
        "fixture_version": 1,
        "changed_groups": (groups + 96) // 97,
        "outliers": (rows + 100) // 101,
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def build_workload(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions
) -> PreparedWorkload:
    manifest = read_json(fixture / "manifest.json")
    rows, groups = manifest["rows"], manifest["groups"]
    grouped = case == "grouped_statistics"
    if case not in {"grouped_statistics", "scalar_statistics"}:
        raise ValueError(f"Unknown capacity case: {case}")
    rule = "REC-001" if grouped else "STA-001"
    inputs = {"observations": fixture / "observations.parquet"}
    params: dict[str, Any]
    if grouped:
        inputs["ledger"] = fixture / "ledger.parquet"
        params = {
            "reconciliations": [
                {
                    "name": "exact grouped float sums",
                    "on": ["group_id"],
                    "left": {"dataset": "observations", "value": "amount"},
                    "right": {"dataset": "ledger", "value": "total"},
                    "rtol": 0,
                    "atol": 0,
                }
            ]
        }
    else:
        params = {
            "targets": [{"dataset": "observations", "value": "value"}],
            "method": "zscore",
            "threshold": 3,
            "min_group_size": 8,
        }
    config = {
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "run_unconfigured_rules": False},
        "rules": {rule: {"params": params}},
        "datasets": {name: {} for name in inputs},
    }
    policy = resources(work_dir / "temporary", options)

    def verify(report):
        if report.incomplete or len(report.results) != 1:
            raise AssertionError("capacity check did not assess exactly one complete contract")
        result = report.results[0]
        expected = (groups + 96) // 97 if grouped else (rows + 100) // 101
        total = groups if grouped else rows
        if (
            result.rule_id != rule
            or result.status is not Status.FAIL
            or result.affected_count != expected
            or result.total_records != total
        ):
            raise AssertionError(f"capacity oracle mismatch: {result.model_dump()}")
        if len(result.affected_records) > options.evidence_rows:
            raise AssertionError("capacity evidence row bound exceeded")
        for row in result.affected_records:
            if grouped:
                group = row["group_id"]
                original = ((rows - 1 - group) // groups + 1) * 0.5
                valid = (
                    0 <= group < groups
                    and group % 97 == 0
                    and Fraction(row["observations_total"]) == Fraction(original)
                    and Fraction(row["ledger_total"]) == Fraction(original + 1)
                )
            else:
                valid = row["value"] == 100.0
            if not valid:
                raise AssertionError("capacity evidence contradicts fixture truth")
        return {
            "verified": True,
            "rule": rule,
            "affected_count": expected,
            "total_records": total,
            "groups": groups if grouped else 1,
        }

    return PreparedWorkload(
        lambda: validate(inputs, config=config, resources=policy),
        verify,
        {
            "api": "qavf.validate",
            "input_rows": rows,
            "groups": groups if grouped else 1,
            "execution_mode": options.execution_mode,
            "oracle": "integer formulas; exact binary-half sums or two-point z scores",
            "timed_scope": "public call including startup, imports in isolated worker, and cleanup",
        },
    )
