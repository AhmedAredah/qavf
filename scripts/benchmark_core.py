"""Fresh-process regression benchmark for bounded native validation.

Example: python scripts/benchmark_core.py --before ../qavf-previous --repeats 3
Each process verifies its exact key/violation population before reporting time
and whole-process peak RSS. Resource modes instrument load-time native memory;
startup measures uninstrumented cold and warm calls. Inputs are synthetic SQL,
so no private data, Polars, pandas, or generated CSV files are required.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

from peak_rss import peak_rss_kib

MODES = ("startup", "narrow", "wide", "row_budget", "retention", "evidence")


class _Patched:
    """A callable class attribute that binds like the method it replaces."""

    def __get__(self, instance, owner=None):
        return self if instance is None else types.MethodType(self, instance)


class _ObservedLoad(_Patched):
    """Time each input load and record native memory and relations afterwards."""

    def __init__(self, original, observations):
        self.original, self.observations = original, observations

    def __call__(self, reader, *values, **options):
        load_started = time.perf_counter()
        try:
            return self.original(reader, *values, **options)
        finally:
            memory = reader.connection.execute(
                "SELECT sum(memory_usage_bytes) FROM duckdb_memory()"
            ).fetchone()[0]
            self.observations.append(
                {
                    "seconds": time.perf_counter() - load_started,
                    "native_memory_bytes": memory,
                    "relations": reader.connection.execute("SHOW TABLES").fetchall(),
                }
            )


class _ObservedRecords(_Patched):
    """Record how many evidence records and text bytes reach the collector."""

    def __init__(self, original, captured):
        self.original, self.captured = original, captured

    def __call__(self, collector, query):
        records = self.original(collector, query)
        self.captured.append(
            {
                "records": len(records),
                "text_bytes_before_collector_budget": sum(
                    len(value.encode("utf-8"))
                    for record in records
                    for value in record.values()
                    if isinstance(value, str)
                ),
            }
        )
        return records


def _run(args, validate, value, schema, config, rows):
    """Validate once, assert the mode's expected outcome and return its timing."""
    tick = time.perf_counter()
    report = validate(value, schema=schema, backend="duckdb", config=config)
    elapsed = time.perf_counter() - tick
    errors = [result for result in report.results if result.status == "ERROR"]
    if args.mode == "row_budget":
        assert errors and report.verdict() == "incomplete"
    elif args.mode == "evidence":
        failed = [result for result in report.results if result.status == "FAIL"]
        assert not errors and len(failed) == 1
        assert failed[0].affected_count == 50 and failed[0].affected_count_exact
        assert failed[0].total_records == 50 and failed[0].evidence_truncated
        assert failed[0].affected_records == []
    else:
        assert report.exit_code() == 0
        keys = [result for result in report.results if result.rule_id == "INP-004"]
        assert len(keys) == (2 if args.mode == "retention" else 1)
        assert all(result.affected_count == 0 and result.total_records == rows for result in keys)
    return report, elapsed


def worker(args):
    sys.path.insert(0, str(args.source.resolve()))
    started = time.perf_counter()
    from qavf import Dataset, Query, validate

    imported = time.perf_counter() - started
    config = {"engine": {"native_threads": 2, "parallel": False, "max_evidence_bytes": 1024}}
    observations, captured = [], []
    if args.mode != "startup":
        from qavf.core.results import ResultCollector
        from qavf.core.sources import InputReader

        original_load, original_records = InputReader.load, ResultCollector.records
        InputReader.load = _ObservedLoad(original_load, observations)
        ResultCollector.records = _ObservedRecords(original_records, captured)
    rows = 2400 if args.mode == "startup" else args.rows
    columns = ", ".join(f"CAST(i+{n} AS BIGINT) AS c{n}" for n in range(args.width))
    value = Query(f"SELECT {columns} FROM range({rows}) t(i)")
    schema = {"primary_key": ["c0"]}
    if args.mode in {"startup", "narrow"}:
        value = Query(f"SELECT i AS c0 FROM range({rows}) t(i)")
    elif args.mode == "row_budget":
        value = Dataset(value, load_policy={"row_budget": 2})
    elif args.mode == "retention":
        value = {name: Dataset(value, schema=schema) for name in ("a", "b")}
        schema = None
    elif args.mode == "evidence":
        value = Query("SELECT i AS id, repeat('x',262144) AS payload FROM range(50) t(i)")
        schema = {"columns": {"id": {"le": -1}}, "evidence_columns": "all"}

    run = functools.partial(_run, args, validate, value, schema, config, rows)
    report, elapsed = run()
    end_to_end = time.perf_counter() - started
    warm = [run()[1] for _ in range(3)] if args.mode == "startup" else []
    peak = peak_rss_kib()
    return {
        "mode": args.mode,
        "import_seconds": imported,
        "validation_seconds": elapsed,
        "end_to_end_seconds": end_to_end,
        "warm_call_seconds": warm,
        "peak_rss_mib": None if peak is None else peak / 1024,
        "verdict": report.verdict(),
        "correctness_verified": True,
        "loads": observations,
        "evidence_materialization": captured,
        "output_records": sum(len(result.affected_records) for result in report.results),
        "report_bytes": report.json_size_bytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--before", type=Path, default=root.parent / "qavf-previous")
    parser.add_argument("--after", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root.parent / "benchmark-core.json")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(worker(args)))
        return
    results = []
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "POLARS_MAX_THREADS": "2"}
    for label, source in (("before", args.before), ("after", args.after)):
        for mode in MODES:
            runs = []
            for _ in range(args.repeats):
                process = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--source",
                        str(source.resolve()),
                        "--mode",
                        mode,
                        "--rows",
                        str(args.rows),
                        "--width",
                        str(args.width),
                    ],
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                runs.append(json.loads(process.stdout))
            summary = {
                "version": label,
                "source": str(source.resolve()),
                "mode": mode,
                "runs": runs,
            }
            for field in (
                "import_seconds",
                "validation_seconds",
                "end_to_end_seconds",
                "peak_rss_mib",
            ):
                values = [run[field] for run in runs if run[field] is not None]
                summary[f"median_{field}"] = statistics.median(values) if values else None
            if mode == "startup":
                summary["median_warm_call_seconds"] = statistics.median(
                    value for run in runs for value in run["warm_call_seconds"]
                )
            results.append(summary)
            print(
                label,
                mode,
                summary["median_validation_seconds"],
                summary["median_peak_rss_mib"],
                flush=True,
            )
    output = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("duckdb", "pyarrow")},
        "rows": args.rows,
        "width": args.width,
        "repeats": args.repeats,
        "notes": "Synthetic workloads; medians of fresh processes, two native threads. "
        "Load instrumentation applies only to resource modes. No speedup threshold asserted.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
