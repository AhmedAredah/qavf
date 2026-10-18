"""Compare bounded scan work in isolated processes; wall times are not release gates.

Example::

    python scripts/benchmark_scan_work.py --baseline /path/to/old/QAVF \
        --candidate /path/to/new/QAVF --rows 50000 --repeat 3 --output results.json

Every measurement imports the requested source explicitly through PYTHONPATH.
Input construction, Parquet writing, imports, and schema warmup are excluded from
wall time. Peak RSS includes all process work, including that setup. SQL and lazy
collection executions are observed at DataContext's public boundaries, not counted
as physical disk reads; backend buffers and filesystem caches remain uncontrolled.
Finding contents must match exactly after normalizing sample order. Each failure
has at most one distinct offending key, so sample membership is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from unittest.mock import patch

from peak_rss import peak_rss_kib

CASES = ("primary_key", "batch_identity", "mixed_columns", "foreign_key_full")
BACKENDS = ("native", "lazy")


def source_digest(source: Path) -> str:
    """Identify production Python sources, independently of benchmark/report edits."""
    digest = hashlib.sha256()
    for path in sorted((source / "qavf").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def case_inputs(case: str, rows: int) -> tuple[dict[str, Any], dict[str, Any], Any]:
    import polars as pl

    from qavf.rules.input_checks import (
        ColumnCheckRule,
        DuplicateKeyRule,
        ReferentialIntegrityRule,
    )
    from qavf.rules.reconciliation_checks import ConsistentBatchRule

    if case == "primary_key":
        frames = {"d": pl.DataFrame({"id": [*range(rows), 0, None]})}
        return frames, {"d": {"schema": {"primary_key": ["id"]}}}, DuplicateKeyRule()
    if case == "batch_identity":
        frames = {
            "left": pl.DataFrame({"batch": ["2026-10-16"] * rows}),
            "right": pl.DataFrame({"batch": ["2026-10-16"] * rows}),
        }
        rule = ConsistentBatchRule(
            {"batches": [{"datasets": ["left", "right"], "column": "batch"}]}
        )
        return frames, {"left": {}, "right": {}}, rule
    if case == "mixed_columns":
        frames = {"d": pl.DataFrame({"x": range(rows), "s": ["A"] * rows, "scope": [1] * rows})}
        columns = {
            "x": {
                "missing_values": [-999],
                "checks": [
                    {"name": "range", "ge": 0, "le": rows},
                    {"name": "membership", "notin": [-1, -2]},
                    {"name": "unique", "unique": True},
                    {"name": "selected", "where": "scope = 1", "ge": 0},
                    {
                        "name": "constant",
                        "where": "TRUE",
                        "ge": 0,
                        "min_applicable_rows": rows,
                    },
                ],
            },
            "s": {
                "regex": "[A-Z]",
                "min_length": 1,
                "max_length": 1,
                "blank_as_missing": True,
            },
        }
        return frames, {"d": {"schema": {"columns": columns}}}, ColumnCheckRule()
    if case == "foreign_key_full":
        frames = {
            "parent": pl.DataFrame({"a": [*range(rows), 0], "b": [*range(rows), 0]}),
            "child": pl.DataFrame(
                {"a": [*range(rows + 1), 0, None, None], "b": [*range(rows + 1), 0, None, 1]}
            ),
        }
        datasets = {
            "parent": {},
            "child": {
                "foreign_keys": [
                    {
                        "columns": ["a", "b"],
                        "ref_dataset": "parent",
                        "match": "full",
                        "cardinality": "one_to_one",
                    }
                ]
            },
        }
        return frames, datasets, ReferentialIntegrityRule()
    raise ValueError(case)


def normalize(value: Any) -> Any:
    """Serialize findings while retaining all outcome and evidence contents."""
    import polars as pl

    if isinstance(value, pl.DataFrame):
        return {
            "schema": {name: str(dtype) for name, dtype in value.schema.items()},
            "rows": sorted(value.to_dicts(), key=lambda row: json.dumps(row, sort_keys=True)),
        }
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [normalize(item) for item in value]
    return value


def worker(args: argparse.Namespace) -> dict[str, Any]:
    import duckdb
    import polars as pl

    import qavf
    from qavf.engine.context import DataContext
    from qavf.settings import QAVFSettings

    imported = Path(qavf.__file__).resolve()
    if imported.parent.parent != args.source.resolve():
        raise RuntimeError(f"Requested {args.source}, imported {imported}")
    frames, datasets, rule = case_inputs(args.case, args.rows)
    calls: Counter[str] = Counter()
    executions: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="qavf-scan-benchmark-") as directory:
        inputs = {name: frame.lazy() for name, frame in frames.items()}
        if args.backend == "native":
            for name, frame in frames.items():
                path = Path(directory) / f"{name}.parquet"
                frame.write_parquet(path)
                datasets[name]["source"] = {"type": "file", "path": str(path)}
            inputs = {}
        settings = QAVFSettings.model_validate(
            {"datasets": datasets, "engine": {"max_affected_records": 1}}
        )
        with DataContext.from_settings(settings, frames=inputs) as ctx:
            for name in datasets:
                ctx.schema(name)

            def observe(method: str) -> Any:
                original = getattr(ctx, method)

                def measured(*positional: Any, **keywords: Any) -> Any:
                    calls[method] += 1
                    result = original(*positional, **keywords)
                    # collect() delegates to collect_many(): count one execution.
                    if method != "collect":
                        executions.append(
                            {
                                "method": method,
                                "rows": result.height,
                                "columns": result.width,
                                "estimated_bytes": result.estimated_size(),
                            }
                        )
                    return result

                return measured

            with (
                patch.object(ctx, "sql", observe("sql")),
                patch.object(ctx, "collect", observe("collect")),
                patch.object(ctx, "collect_many", observe("collect_many")),
            ):
                started = perf_counter()
                findings = rule.evaluate(ctx)
                elapsed = perf_counter() - started
            if not all(finding.assessed and not finding.skipped for finding in findings):
                raise AssertionError([finding.message for finding in findings])
            outcomes = [
                {item.name: normalize(getattr(finding, item.name)) for item in fields(finding)}
                for finding in findings
            ]
            evidence_rows = [
                finding.affected.height
                if isinstance(finding.affected, pl.DataFrame)
                else len(finding.affected or [])
                for finding in findings
            ]
            if any(size > 1 for size in evidence_rows):
                raise AssertionError(f"Evidence limit exceeded: {evidence_rows}")
            if any(ctx.is_materialized(name) for name in datasets):
                raise AssertionError("Rule materialized a complete dataset")
    return {
        "case": args.case,
        "backend": args.backend,
        "input_rows": {name: frame.height for name, frame in frames.items()},
        "evidence_limit": 1,
        "finding_sample_rows": evidence_rows,
        "method_calls": dict(calls),
        "execution_count": len(executions),
        "executions": executions,
        "seconds": elapsed,
        "process_peak_rss_kib": peak_rss_kib(),
        "outcomes": outcomes,
        "versions": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "duckdb": duckdb.__version__,
        },
        "imported_qavf": str(imported),
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    sources = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    before = {label: source_digest(source) for label, source in sources.items()}
    measurements = []
    for case in CASES:
        for backend in BACKENDS:
            for repetition in range(1, args.repeat + 1):
                for label, source in sources.items():
                    env = dict(
                        os.environ,
                        PYTHONPATH=str(source),
                        POLARS_MAX_THREADS="4",
                        OPENBLAS_NUM_THREADS="1",
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--source",
                        str(source),
                        "--case",
                        case,
                        "--backend",
                        backend,
                        "--rows",
                        str(args.rows),
                    ]
                    result = subprocess.run(
                        command,
                        cwd=source,
                        env=env,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=120,
                    )
                    measurements.append(
                        {"source": label, "repetition": repetition, **json.loads(result.stdout)}
                    )
    after = {label: source_digest(source) for label, source in sources.items()}
    if before != after:
        raise RuntimeError(
            "Production sources changed during measurements; rerun after edits settle"
        )
    comparisons = []
    for case in CASES:
        for backend in BACKENDS:
            grouped = {
                label: [
                    row
                    for row in measurements
                    if row["source"] == label and row["case"] == case and row["backend"] == backend
                ]
                for label in sources
            }
            expected = grouped["baseline"][0]["outcomes"]
            identical = all(
                row["outcomes"] == expected for rows in grouped.values() for row in rows
            )
            if not identical:
                raise AssertionError(f"Finding contents differ for {case}/{backend}: {grouped}")
            comparisons.append(
                {
                    "case": case,
                    "backend": backend,
                    "outcomes_identical": identical,
                    **{
                        label: {
                            "execution_counts": sorted({r["execution_count"] for r in rows}),
                            "median_seconds": statistics.median(r["seconds"] for r in rows),
                            "median_process_peak_rss_kib": statistics.median(
                                r["process_peak_rss_kib"] for r in rows
                            ),
                        }
                        for label, rows in grouped.items()
                    },
                }
            )
    return {
        "format_version": 1,
        "source_paths": {label: str(path) for label, path in sources.items()},
        "production_source_sha256": before,
        "requested_rows": args.rows,
        "repetitions": args.repeat,
        "platform": platform.platform(),
        "notes": [
            "Execution counts are DataContext SQL/collection boundaries, not physical disk reads.",
            "Nested collect -> collect_many is one execution; both call counts are retained.",
            "Wall time excludes imports, inputs, Parquet writes, and schema warmup.",
            "Peak RSS includes all process work; shared-host timing/cache noise is uncontrolled.",
            "Batch identity has two datasets: per-dataset counts are half the reported total.",
            "No complete dataset was materialized; finding evidence is bounded to one row.",
        ],
        "comparisons": comparisons,
        "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.rows <= 99_996 or args.repeat < 1:
        parser.error("rows must be 1..99996 and repeat must be positive")
    if args.worker:
        result = worker(args)
    else:
        if args.baseline is None or args.candidate is None:
            parser.error("--baseline and --candidate are required")
        result = compare(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
