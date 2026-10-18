"""Sequential process orchestration with durable evidence and temporary ownership."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .planning import schedule
from .provenance import (
    fixture_identity,
    harness_digest,
    host_environment,
    production_digest,
    timestamp,
)
from .runner import execute_worker
from .storage import read_json, write_json


def worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        environment.pop(name, None)
    environment.update(
        POLARS_MAX_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        PYTHONHASHSEED="0",
    )
    return environment


def persist(suite: dict[str, Any], output: Path) -> None:
    from .reporting import render_html, summarize_suite

    suite["summary"] = summarize_suite(suite)
    write_json(output, suite)
    # The complete JSON is authoritative if HTML rendering is interrupted.
    output.with_suffix(".html").write_text(render_html(suite), encoding="utf-8")


def _prepare(
    entry: Path,
    directory: Path,
    family: str,
    rows: int,
    protocol: dict[str, Any],
    environment: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    request = directory.parent / f"{directory.name}-prepare.json"
    result = request.with_name(f"{directory.name}-prepared.json")
    write_json(
        request,
        {
            "directory": str(directory),
            "family": family,
            "rows": rows,
            "seed": protocol["seed"],
            "partitions": protocol["partitions"],
            "chunk_rows": protocol["chunk_rows"],
        },
    )
    command = [sys.executable, str(entry), "--prepare", str(request), "--result", str(result)]
    log = request.with_suffix(".log")
    started = time.perf_counter()
    with log.open("wb") as stream:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    if completed.returncode != 0 or not result.is_file():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            tail = stream.read().decode(errors="replace")
        raise RuntimeError(f"Fixture preparation failed for {family}/{rows}: {tail}")
    record = read_json(result)
    if record.get("status") != "passed":
        raise RuntimeError(f"Fixture preparation failed for {family}/{rows}: {record}")
    return {
        "directory": str(directory),
        "manifest": record["manifest"],
        "preparation_process_seconds": time.perf_counter() - started,
        **fixture_identity(directory),
    }


def run_suite(
    *,
    root: Path,
    sources: dict[str, Path],
    plan: list[dict[str, Any]],
    protocol: dict[str, Any],
    output: Path,
    timeout_s: float,
    work_parent: Path | None = None,
    keep_work: bool = False,
) -> dict[str, Any]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if work_parent is not None:
        work_parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="qavf-bench-", dir=work_parent))
    entry = root / "scripts/benchmark_performance.py"
    suite: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started": timestamp(),
        "protocol": protocol,
        "host": {},
        "sources": {},
        "benchmark_sha256": None,
        "fixtures": {},
        "records": [],
        "work_directory": str(directory),
        "work_retained": keep_work,
        "planned_variants": plan,
    }
    try:
        environment = worker_environment(protocol["threads"])
        suite["host"] = host_environment()
        jobs = schedule(plan, list(sources), protocol["repetitions"], protocol["warmups"])
        suite["sources"] = {
            label: {"root": str(path), "production_sha256": production_digest(path)}
            for label, path in sources.items()
        }
        suite["benchmark_sha256"] = harness_digest(root)
        persist(suite, output)
        for variant in plan:
            key = f"{variant['family']}-{variant['rows']}"
            if key not in suite["fixtures"]:
                fixture = directory / "fixtures" / key
                suite["fixtures"][key] = _prepare(
                    entry,
                    fixture,
                    variant["family"],
                    variant["rows"],
                    protocol,
                    environment,
                    timeout_s,
                )
                persist(suite, output)
        for index, job in enumerate(jobs, 1):
            fixture = suite["fixtures"][f"{job['family']}-{job['rows']}"]
            run_dir = directory / "runs" / job["id"]
            run_dir.mkdir(parents=True)
            request = {
                **job,
                "source_root": str(sources[job["source_label"]]),
                "fixture_dir": fixture["directory"],
                "operation_dir": str(run_dir / "operation"),
                "interval_s": protocol["sample_interval_seconds"],
                "report_format": protocol["report_format"],
                "expected_fixture_sha256": fixture["sha256"],
                "expected_source_sha256": suite["sources"][job["source_label"]][
                    "production_sha256"
                ],
            }
            record = execute_worker(
                [sys.executable, str(entry)],
                request,
                run_dir,
                timeout_s=timeout_s,
                environment=environment,
            )
            suite["records"].append(record)
            persist(suite, output)
            print(
                f"[{index}/{len(jobs)}] {job['id']} {job['case']} {job['rows']:,} "
                f"{job['options']['backend']}/{job['options']['input_format']}: "
                f"{record['status']}",
                flush=True,
            )
        for fixture in suite["fixtures"].values():
            if fixture_identity(Path(fixture["directory"]))["sha256"] != fixture["sha256"]:
                raise RuntimeError("Shared fixture bytes changed during the suite")
        for label, source in sources.items():
            if production_digest(source) != suite["sources"][label]["production_sha256"]:
                raise RuntimeError(f"Product source changed during the suite: {label}")
        if harness_digest(root) != suite["benchmark_sha256"]:
            raise RuntimeError("Benchmark implementation changed during the suite")
        suite["identity_unchanged"] = True
        suite["status"] = (
            "passed" if all(r["status"] == "passed" for r in suite["records"]) else "failed"
        )
    except Exception as error:
        suite.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    except BaseException:
        suite["status"] = "interrupted"
        raise
    finally:
        suite["finished"] = timestamp()
        if not keep_work:
            try:
                shutil.rmtree(directory)
            except OSError as error:
                suite["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
                if suite["status"] != "interrupted":
                    suite["status"] = "failed"
        suite["work_directory_removed"] = not directory.exists()
        persist(suite, output)
    return suite
