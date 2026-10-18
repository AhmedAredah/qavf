"""One isolated, correctness-checked benchmark invocation."""

from __future__ import annotations

import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

from .metrics import OperationMetrics
from .models import WorkloadOptions
from .provenance import production_digest, timestamp, worker_environment
from .storage import read_json, write_json


def _write_report(value: Any, directory: Path, mode: str) -> list[dict[str, Any]]:
    from qavf.reports.html import write_html
    from qavf.reports.models import QAReport

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    if isinstance(value, QAReport):
        if mode in {"json", "both"}:
            paths.append(value.write_json(directory / "qa_report.json"))
        if mode in {"html", "both"}:
            paths.append(write_html(value, directory / "qa_report.html"))
    elif mode != "none":
        path = directory / "result.json"
        write_json(path, value)
        paths.append(path)
    return [{"name": Path(p).name, "size_bytes": Path(p).stat().st_size} for p in paths]


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Import the requested product before workloads; preserve completed phases on error."""
    source = Path(request["source_root"]).resolve()
    sys.path.insert(0, str(source))
    record: dict[str, Any] = {
        "id": request["id"],
        "status": "failed",
        "started": timestamp(),
        "phases": {},
    }
    directory = Path(request["operation_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    options = WorkloadOptions(**request["options"])
    meter = None
    phase = "identity"
    try:
        if production_digest(source) != request["expected_source_sha256"]:
            raise RuntimeError("Product source changed before the benchmark")
        phase = "setup"
        with OperationMetrics(directory, request["interval_s"]) as meter:
            from .catalog import build_workload

            environment = worker_environment(options.threads)
            if Path(environment["qavf_import_path"]) != source / "qavf/__init__.py":
                raise RuntimeError("Worker imported QAVF from a different source")
            if environment["polars_threads"] != options.threads:
                raise RuntimeError("Polars pool differs from requested thread count")
            record["environment"] = environment
            prepared = build_workload(
                request["case"], Path(request["fixture_dir"]), directory, options
            )
        record["phases"][phase] = meter.result()
        record["workload"] = prepared.metadata
        phase = "operation"
        with OperationMetrics(directory, request["interval_s"]) as meter:
            value = prepared.run()
        record["phases"][phase] = meter.result()
        phase = "oracle"
        start = time.perf_counter()
        record["outcome"] = prepared.verify(value)
        record["oracle_seconds"] = time.perf_counter() - start
        from qavf.reports.models import QAReport

        if isinstance(value, QAReport):
            record["qavf_telemetry"] = {
                "resource_usage": value.metadata.resource_usage,
                "query_plans": value.metadata.query_plans,
                "datasets": [
                    {"name": data.name, "notes": data.notes} for data in value.metadata.datasets
                ],
            }
        else:
            record["qavf_telemetry"] = {}
        phase = "report"
        with OperationMetrics(directory, request["interval_s"]) as meter:
            files = _write_report(value, directory / "reports", request["report_format"])
        record["phases"][phase] = meter.result()
        record["report_files"] = files
        record["report_bytes"] = sum(item["size_bytes"] for item in files)
        for name, metrics in record["phases"].items():
            if metrics["sampling"]["monitor_error"] or metrics["sampling"]["sample_failures"]:
                raise RuntimeError(f"Resource observer failed during {name}")
        if production_digest(source) != request["expected_source_sha256"]:
            raise RuntimeError("Product source changed during the benchmark")
        record["status"] = "passed"
    except Exception as exc:
        if meter is not None and phase in {"setup", "operation", "report"}:
            with suppress(RuntimeError):
                record["phases"][phase] = meter.result()
        record["error"] = {
            "phase": phase,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc()[-8192:],
        }
    record["finished"] = timestamp()
    return record


def main(request_path: Path, result_path: Path) -> int:
    record = run(read_json(request_path))
    write_json(result_path, record)
    return 0 if record["status"] == "passed" else 1
