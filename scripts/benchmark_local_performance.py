"""Fresh-process benchmark of local column projection and SQL assertion evidence.

Example: python scripts/benchmark_local_performance.py --rows 200000 --repeats 3

``projection`` builds many unrelated wide Parquet datasets, puts one local row
expression on the first, and reports the native backend's dependency components,
retained columns, the Parquet column-chunk bytes those columns occupy, wall time
and peak memory. ``assertion`` runs one SQL assertion that every row fails, with
bounded evidence, on both backends and reports its exact count, the evidence
size, wall time and peak memory. Each measurement runs in its own process, so
peak resident memory belongs to that run alone. The data is synthetic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

from peak_rss import peak_rss_kib

MODES = ("projection", "assertion")
BACKENDS = ("duckdb", "polars")
PAYLOAD_COLUMNS = 8


class SyntheticData:
    """Write the benchmark's Parquet inputs once into a directory."""

    def __init__(self, directory: Path, rows: int, datasets: int) -> None:
        self.directory = directory
        self.rows = rows
        self.datasets = datasets

    def names(self) -> list[str]:
        return [f"d{index}" for index in range(self.datasets)]

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.parquet"

    def write(self) -> None:
        import duckdb

        payload = ", ".join(
            f"repeat(md5((range * {index + 7})::VARCHAR), 2) AS payload_{index}"
            for index in range(PAYLOAD_COLUMNS)
        )
        with duckdb.connect() as con:
            for name in self.names():
                target = str(self.path(name)).replace("'", "''")
                con.execute(
                    f"COPY (SELECT range AS id, {payload} FROM range({self.rows})) "
                    f"TO '{target}' (FORMAT parquet)"
                )

    def column_bytes(self, name: str, columns: list[str]) -> int:
        """Compressed Parquet bytes of ``columns``: what a projected scan reads."""
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(self.path(name)).metadata
        wanted = {column.casefold() for column in columns}
        total = 0
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            for index in range(row_group.num_columns):
                chunk = row_group.column(index)
                if chunk.path_in_schema.casefold() in wanted:
                    total += int(chunk.total_compressed_size)
        return total


class Workload:
    """One measured validation, configured for a mode and a backend."""

    def __init__(self, data: SyntheticData, mode: str, backend: str, evidence: int) -> None:
        self.data = data
        self.mode = mode
        self.backend = backend
        self.evidence = evidence

    def config(self) -> dict[str, Any]:
        datasets: dict[str, Any] = {
            name: {
                "path": str(self.data.path(name)),
                "schema": {"columns": {"id": {"dtype": "int"}}},
            }
            for name in self.data.names()
        }
        config: dict[str, Any] = {
            "datasets": datasets,
            "engine": {"max_affected_records": self.evidence},
        }
        first = self.data.names()[0]
        if self.mode == "projection":
            datasets[first]["checks"] = [
                {"check": "expression", "name": "non-negative", "expr": "id >= 0"}
            ]
        else:
            config["checks"] = [
                {"check": "sql", "name": "every row", "sql": f"SELECT * FROM {first}"}
            ]
        return config

    def measure(self) -> dict[str, Any]:
        import qavf

        config = self.config()
        tracemalloc.start()
        started = time.perf_counter()
        report = qavf.validate(config=config, backend=self.backend)
        elapsed = time.perf_counter() - started
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result: dict[str, Any] = {
            "mode": self.mode,
            "backend": self.backend,
            "wall_s": round(elapsed, 3),
            "peak_rss_kib": peak_rss_kib(),
            "python_peak_kib": traced_peak // 1024,
        }
        if self.mode == "projection":
            # The native backend records its projection; Polars scans lazily.
            retained = {
                info.name: list(info.notes["retained_columns"])
                for info in report.metadata.datasets
                if "retained_columns" in info.notes
            }
            result["components"] = report.metadata.resource_usage.get("dependency_components")
            result["retained_columns"] = (
                sum(len(columns) for columns in retained.values()) if retained else None
            )
            result["parquet_bytes_read"] = (
                sum(self.data.column_bytes(name, columns) for name, columns in retained.items())
                if retained
                else None
            )
            result["failed"] = sum(r.status.value == "FAIL" for r in report.results)
        else:
            failed = next(r for r in report.results if r.check_id == "sql.assertion")
            result["status"] = failed.status.value
            result["count"] = failed.affected_count
            result["evidence_rows"] = len(failed.affected_records or [])
        return result


def _worker(arguments: argparse.Namespace) -> None:
    data = SyntheticData(Path(arguments.directory), arguments.rows, arguments.datasets)
    workload = Workload(data, arguments.mode, arguments.backend, arguments.evidence)
    print(json.dumps(workload.measure()))


def _measure(arguments: argparse.Namespace, directory: Path, mode: str, backend: str) -> Any:
    command = [
        sys.executable,
        str(Path(__file__)),
        "--worker",
        "--directory",
        str(directory),
        "--mode",
        mode,
        "--backend",
        backend,
        "--rows",
        str(arguments.rows if mode == "projection" else arguments.assertion_rows),
        "--datasets",
        str(arguments.datasets if mode == "projection" else 1),
        "--evidence",
        str(arguments.evidence),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"{mode}/{backend} worker failed:\n{completed.stderr[-3000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = dict(runs[0])
    summary["wall_s"] = round(statistics.median(run["wall_s"] for run in runs), 3)
    peaks = [run["peak_rss_kib"] for run in runs if run["peak_rss_kib"] is not None]
    summary["peak_rss_kib"] = int(statistics.median(peaks)) if peaks else None
    summary["python_peak_kib"] = int(statistics.median(run["python_peak_kib"] for run in runs))
    summary["repeats"] = len(runs)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--assertion-rows", type=int, default=2_000_000)
    parser.add_argument("--datasets", type=int, default=20)
    parser.add_argument("--evidence", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--directory", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(arguments)
        return
    modes = MODES if arguments.mode == "all" else (arguments.mode,)
    backends = BACKENDS if arguments.backend == "all" else (arguments.backend,)
    with tempfile.TemporaryDirectory(prefix="qavf-bench-") as temporary:
        for mode in modes:
            directory = Path(temporary) / mode
            directory.mkdir()
            count = arguments.datasets if mode == "projection" else 1
            rows = arguments.rows if mode == "projection" else arguments.assertion_rows
            SyntheticData(directory, rows, count).write()
            for backend in backends:
                runs = [
                    _measure(arguments, directory, mode, backend) for _ in range(arguments.repeats)
                ]
                print(json.dumps(_summary(runs)))


if __name__ == "__main__":
    main()
