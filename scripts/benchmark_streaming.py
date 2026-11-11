"""Fresh-process benchmark of streamed snapshot staging and borrowed-query copies.

Example: python scripts/benchmark_streaming.py --rows 1000000 --borrowed-rows 3000000
         --checkout after=. --checkout before=../qavf-before

``snapshot`` validates a coherent snapshot group (``snapshot: true``) of three
DuckDB-file tables on the Polars backend. ``borrowed`` validates a ``Query`` on a
borrowed DuckDB connection, over a DuckDB file, on the native backend. Each run
reports wall time, peak resident memory and the run's spill bytes, in its own
process, so peak memory belongs to that run alone. ``--checkout LABEL=DIR`` runs
the workers against the QAVF source tree in ``DIR`` (by default the installed
one), to compare two revisions on the same data. The data is synthetic.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from peak_rss import peak_rss_kib

MODES = ("snapshot", "borrowed")
MEMBERS = ("t0", "t1", "t2")
PAYLOAD_COLUMNS = 4


class SyntheticDatabase:
    """Write one DuckDB file of equally shaped tables with an id and text payload."""

    def __init__(self, path: Path, rows: int, tables: tuple[str, ...]) -> None:
        self.path = path
        self.rows = rows
        self.tables = tables

    @staticmethod
    def columns() -> list[str]:
        return ["id", "amount", *(f"payload_{index}" for index in range(PAYLOAD_COLUMNS))]

    def write(self) -> None:
        import duckdb

        payload = ", ".join(
            f"md5((range * {index + 3})::VARCHAR) AS payload_{index}"
            for index in range(PAYLOAD_COLUMNS)
        )
        with duckdb.connect(str(self.path)) as con:
            for table in self.tables:
                con.execute(
                    f"CREATE TABLE {table} AS SELECT range AS id, range / 3.0 AS amount, "
                    f"{payload} FROM range({self.rows})"
                )


class Workload:
    """One measured validation of a mode over a written database."""

    def __init__(self, database: SyntheticDatabase, mode: str) -> None:
        self.database = database
        self.mode = mode

    @staticmethod
    def dataset() -> dict[str, Any]:
        """A check that reads every column, so no projection narrows the copy."""
        reads = " AND ".join(f"{column} IS NOT NULL" for column in SyntheticDatabase.columns())
        return {
            "schema": {"columns": {"id": {"dtype": "int", "ge": 0}}},
            "checks": [{"check": "expression", "name": "complete", "expr": reads}],
        }

    def run(self) -> Any:
        import qavf

        if self.mode == "snapshot":
            config = {
                "connections": {
                    "live": {"type": "duckdb", "path": str(self.database.path), "snapshot": True}
                },
                "datasets": {
                    table: {
                        "source": {"type": "database", "connection": "live", "table": table},
                        **self.dataset(),
                    }
                    for table in self.database.tables
                },
            }
            return qavf.validate(config=config, backend="polars")
        import duckdb

        with duckdb.connect(str(self.database.path), read_only=True) as borrowed:
            query = qavf.Query(f"SELECT * FROM {self.database.tables[0]}", connection=borrowed)
            config = {"datasets": {"data": self.dataset()}}
            return qavf.validate({"data": query}, config=config, backend="duckdb")

    def measure(self) -> dict[str, Any]:
        import qavf

        started = time.perf_counter()
        report = self.run()
        elapsed = time.perf_counter() - started
        usage = report.metadata.resource_usage
        return {
            "mode": self.mode,
            "qavf": str(Path(qavf.__file__).resolve().parent.parent),
            "rows": self.database.rows,
            "verdict": report.verdict(),
            "unassessed": sorted(
                r.rule_id for r in report.results if r.status.value in {"ERROR", "NOT_ASSESSED"}
            ),
            "wall_s": round(elapsed, 3),
            "peak_rss_kib": peak_rss_kib(),
            "peak_frame_spill_bytes": usage.get("peak_frame_spill_bytes"),
            "frame_spill_write_bytes": usage.get("frame_spill_write_bytes"),
            "peak_owned_spill_bytes": usage.get("peak_owned_spill_bytes"),
        }


class Benchmark:
    """Run fresh worker processes per checkout and summarize their medians."""

    def __init__(self, arguments: argparse.Namespace, directory: Path) -> None:
        self.arguments = arguments
        self.directory = directory

    def database(self, mode: str) -> SyntheticDatabase:
        if mode == "snapshot":
            return SyntheticDatabase(
                self.directory / "snapshot.duckdb", self.arguments.rows, MEMBERS
            )
        rows = self.arguments.borrowed_rows
        return SyntheticDatabase(self.directory / "borrowed.duckdb", rows, ("source",))

    def measure(self, mode: str, checkout: str | None) -> dict[str, Any]:
        database = self.database(mode)
        command = [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            "--mode",
            mode,
            "--database",
            str(database.path),
            "--rows",
            str(database.rows),
            "--tables",
            ",".join(database.tables),
        ]
        environment = dict(os.environ)
        if checkout is not None:
            # The checkout's sources come before the installed package.
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, [str(Path(checkout).resolve()), environment.get("PYTHONPATH")])
            )
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, env=environment
        )
        if completed.returncode:
            raise RuntimeError(f"{mode} worker failed:\n{completed.stderr[-3000:]}")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    @staticmethod
    def checkout(item: str) -> tuple[str, str]:
        label, separator, path = item.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"--checkout expects LABEL=DIR, not {item!r}")
        return label, path

    @staticmethod
    def summary(label: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {"checkout": label, **runs[0]}
        summary["wall_s"] = round(statistics.median(run["wall_s"] for run in runs), 3)
        peaks = [run["peak_rss_kib"] for run in runs if run["peak_rss_kib"] is not None]
        summary["peak_rss_kib"] = int(statistics.median(peaks)) if peaks else None
        summary["repeats"] = len(runs)
        return summary

    def run(self) -> None:
        modes = MODES if self.arguments.mode == "all" else (self.arguments.mode,)
        checkouts: list[tuple[str, str | None]] = [
            (label, path) for label, path in map(self.checkout, self.arguments.checkout)
        ] or [("installed", None)]
        for mode in modes:
            self.database(mode).write()
            for label, checkout in checkouts:
                runs = [self.measure(mode, checkout) for _ in range(self.arguments.repeats)]
                print(json.dumps(self.summary(label, runs)), flush=True)


def _worker(arguments: argparse.Namespace) -> None:
    tables = tuple(arguments.tables.split(","))
    database = SyntheticDatabase(Path(arguments.database), arguments.rows, tables)
    print(json.dumps(Workload(database, arguments.mode).measure()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--rows", type=int, default=1_000_000, help="rows per snapshot member")
    parser.add_argument("--borrowed-rows", type=int, default=3_000_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--checkout",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="a QAVF source tree to measure (repeatable); default: the installed package",
    )
    parser.add_argument("--database", help=argparse.SUPPRESS)
    parser.add_argument("--tables", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        _worker(arguments)
        return
    with tempfile.TemporaryDirectory(prefix="qavf-stream-bench-") as temporary:
        Benchmark(arguments, Path(temporary)).run()


if __name__ == "__main__":
    main()
