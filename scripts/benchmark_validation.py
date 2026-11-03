"""Reproducible QAVF/SQL validation benchmark with correctness assertions.

Run in a pinned environment: python scripts/benchmark_validation.py --source /path/to/qavf
Add --rows 1890000 --repeats 3 for the large synthetic workload.
The serial mode is an existing configuration, not a patched 'after' version.
"""

from __future__ import annotations

import argparse
import cProfile
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

COLS = [f"c{i}" for i in range(14)]
KEYS = COLS[:7]


def package_versions():
    versions: dict[str, str | None] = {}
    for package in ("duckdb", "pyarrow", "pandas", "numpy", "polars"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def peak_rss_mib():
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024**2
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def worker(args):
    sys.path.insert(0, str(args.source.resolve()))
    t = time.perf_counter()
    if args.mode.startswith("qavf"):
        from qavf import validate
    else:
        import duckdb
    import_s = time.perf_counter() - t
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    t = time.perf_counter()
    result: dict[str, Any]
    config: dict[str, Any]
    if args.mode.startswith("qavf"):
        config = {}
        if args.mode == "qavf_serial":
            config = {"engine": {"parallel": False, "native_threads": 2}}
        if args.memory:
            config.setdefault("engine", {}).update(duckdb_memory_limit=args.memory)
        if args.spill is not None:
            config.setdefault("engine", {})["spill_max_bytes"] = int(args.spill)
        backend_options = {"backend": args.backend} if args.backend != "auto" else {}
        report = validate(
            str(args.csv),
            schema={"columns": {c: {"dtype": "int", "ge": 0} for c in COLS}, "primary_key": KEYS},
            config=config,
            **backend_options,
        )
        duplicates = [
            r for r in report.results if r.rule_id == "INP-004" and r.metrics.get("duplicated_keys")
        ]
        errors = [r.message for r in report.results if r.status.value == "ERROR"]
        bad = [
            r.message for r in report.results if r.status.value == "FAIL" and r.rule_id != "INP-004"
        ]
        duplicate = duplicates[0] if duplicates else None
        result = {
            "duplicates": duplicate.affected_count if duplicate else None,
            "involved": duplicate.affected_rows if duplicate else None,
            "samples": duplicate.affected_records if duplicate else [],
            "errors": errors + bad,
            "resource_usage": report.metadata.resource_usage,
            "result_count": len(report.results),
        }
    else:
        config = {"threads": 2}
        if args.memory:
            config["memory_limit"] = args.memory
        with duckdb.connect(config=config) as con:
            if args.spill is not None:
                con.execute(f"SET max_temp_directory_size='{int(args.spill)}B'")
            # A strict typed projection matching QAVF's integer grammar.
            converted = [
                f"CASE WHEN regexp_full_match(trim({c}), '[+-]?[0-9]+') "
                f"THEN try_cast(trim({c}) AS BIGINT) END AS {c}"
                for c in COLS
            ]
            invalid = [
                f"({c} IS NOT NULL AND (CASE WHEN regexp_full_match(trim({c}), '[+-]?[0-9]+') "
                f"THEN try_cast(trim({c}) AS BIGINT) END) IS NULL) AS bad_{c}"
                for c in COLS
            ]
            con.execute(
                "CREATE TEMP TABLE data AS SELECT row_number() OVER () - 1 AS rowid, "
                + ",".join(converted + invalid)
                # Path literals avoid DuckDB's Python parameter adapter importing
                # optional dataframe packages just to bind one string. QAVF uses
                # the same safe literal escaping, so the baseline measures SQL.
                + " FROM read_csv('"
                + str(args.csv).replace("'", "''")
                + "', header=true, all_varchar=true, strict_mode=true)"
            )
            reductions = ["count(*)"] + [
                f"count(*) FILTER (WHERE bad_{c} OR {c} IS NULL OR {c}<0)" for c in COLS
            ]
            counts = con.execute("SELECT " + ",".join(reductions) + " FROM data").fetchone()
            assert counts is not None
            assert counts[0] == args.rows and not any(counts[1:]), counts
            keys = ",".join(KEYS)
            con.execute(
                f"CREATE TEMP TABLE groups AS SELECT {keys},count(*) AS occurrences "
                f"FROM data GROUP BY {keys} HAVING count(*)>1"
            )
            totals = con.execute("SELECT count(*),sum(occurrences) FROM groups").fetchone()
            assert totals is not None
            groups, involved = totals
            # Only sampled key groups need row-position evidence.
            con.execute(
                f"CREATE TEMP TABLE wanted AS SELECT * FROM groups ORDER BY {keys} LIMIT 50"
            )
            sql = (
                f"SELECT {keys},max(w.occurrences) AS occurrences,min(data.rowid,10) AS rows "
                f"FROM data JOIN wanted w USING ({keys}) GROUP BY {keys} ORDER BY {keys}"
            )
            samples = [
                dict(zip([*KEYS, "occurrences", "rows"], r, strict=True))
                for r in con.execute(sql).fetchall()
            ]
            result = {"duplicates": groups, "involved": involved, "samples": samples, "errors": []}
    result.update(
        mode=args.mode,
        rows=args.rows,
        import_s=import_s,
        call_s=time.perf_counter() - t,
        peak_rss_mib=peak_rss_mib(),
        duckdb=importlib.metadata.version("duckdb"),
        source=str(args.source.resolve()),
        qavf=(__import__("qavf").__version__ if args.mode.startswith("qavf") else None),
    )
    if profiler:
        profiler.disable()
        profiler.dump_stats(str(args.profile))
    # Correctness assertions prevent timing an incomplete or weaker result.
    if result["errors"]:
        raise RuntimeError(f"Validation incomplete: {result['errors']}")
    if not result["errors"]:
        assert result["duplicates"] == args.rows // 2, result
        assert result["involved"] == args.rows, result
        assert len(result["samples"]) == min(50, args.rows // 2), result
        for i, sample in enumerate(result["samples"]):
            assert [sample[c] for c in KEYS] == [i * (2 * j + 1) for j in range(7)], sample
            assert sample["occurrences"] == 2 and sample["rows"] == [2 * i, 2 * i + 1], sample
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--rows", type=int, default=2400)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mode", choices=["qavf_default", "qavf_serial", "sql"])
    p.add_argument("--csv", type=Path)
    p.add_argument("--output", type=Path, default=Path("benchmark.json"))
    p.add_argument("--backend", choices=["auto", "duckdb", "polars"], default="auto")
    p.add_argument("--memory")
    p.add_argument("--spill", type=int)
    p.add_argument("--profile", type=Path)
    args = p.parse_args()
    if args.rows < 2 or args.rows % 2:
        p.error("--rows must be positive and even")
    if args.mode:
        worker(args)
        return
    import duckdb

    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.csv or args.output.parent / f"synthetic_{args.rows}.csv"
    if not path.exists():
        expr = [
            f"(i//2)*{2 * j + 1} AS {c}" if j < 7 else f"i%1000+{j} AS {c}"
            for j, c in enumerate(COLS)
        ]
        with duckdb.connect(config={"threads": 2}) as con:
            con.execute(
                "COPY (SELECT "
                + ",".join(expr)
                + f" FROM range({args.rows}) t(i)) TO ? (HEADER, DELIMITER ',')",
                [str(path)],
            )
    runs = []
    for repeat in range(args.repeats):
        for mode in ["sql", "qavf_default", "qavf_serial"]:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(args.source),
                "--rows",
                str(args.rows),
                "--mode",
                mode,
                "--csv",
                str(path),
            ]
            if args.backend != "auto":
                command += ["--backend", args.backend]
            if args.memory:
                command += ["--memory", args.memory]
            if args.spill is not None:
                command += ["--spill", str(args.spill)]
            start = time.perf_counter()
            child = subprocess.run(command, capture_output=True, text=True, timeout=240)
            if child.returncode:
                raise RuntimeError(f"{mode}: {child.stdout}\n{child.stderr}")
            run = json.loads(child.stdout)
            run["process_wall_s"] = time.perf_counter() - start
            run["repeat"] = repeat
            run["stderr"] = child.stderr
            runs.append(run)
            print(
                mode,
                repeat,
                round(run["call_s"], 3),
                round(run["peak_rss_mib"], 1),
                "errors",
                len(run["errors"]),
                flush=True,
            )
            args.output.write_text(json.dumps({"runs": runs}, indent=2))
    summary = {
        mode: {
            key: statistics.median(r[key] for r in runs if r["mode"] == mode)
            for key in ["call_s", "import_s", "process_wall_s", "peak_rss_mib"]
        }
        for mode in ["sql", "qavf_default", "qavf_serial"]
    }
    args.output.write_text(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "input_bytes": path.stat().st_size,
                    "memory": args.memory,
                    "spill": args.spill,
                    "backend": args.backend,
                    "baseline_reader": "escaped SQL path literal",
                    "thread_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "POLARS_MAX_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "OMP_NUM_THREADS",
                        )
                    },
                    "packages": package_versions(),
                },
                "summary": summary,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
