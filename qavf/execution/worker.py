"""Child-side execution after the standard-library bootstrap has applied OS limits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qavf.engine.context import DataContext
from qavf.engine.runner import QARunner, config_digest
from qavf.execution.protocol import read_request, write_result
from qavf.inputs.normalize import _adapt
from qavf.redaction import exception_details, redact
from qavf.reports.models import _iter_snapshot_json
from qavf.settings import QAVFSettings

_MEMORY_SCOPE = {
    "linux": "OS virtual address space per worker process; not aggregate RSS",
    "darwin": "not available on macOS",
    "win32": "Windows job committed memory per worker process; not aggregate RSS",
}
_HARD_FIELDS = (
    "run_timeout_s", "process_address_space_bytes", "process_cpu_time_s", "process_file_bytes"
)


def _local_settings(settings: QAVFSettings, directory: Path) -> QAVFSettings:
    raw = settings.model_dump(mode="python", by_alias=True)
    raw["engine"].update({name: None for name in _HARD_FIELDS})
    # Startup pool size is process-global. Preserve an omitted per-connection
    # setting so the runner can still divide CPUs between concurrent rules.
    raw["engine"].update(execution_mode="in_process", temp_directory=directory)
    local = QAVFSettings.model_validate(raw)
    local.source_path = settings.source_path
    return local


def _usage(control: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_mode": "isolated",
        "effective_process_limits": control["effective_limits"],
        "native_threads_at_startup": control["effective_native_threads"],
        "run_timeout_s": control.get("run_timeout_s"),
        "deadline_scope": "worker startup through execution, finalization and serialization",
        "process_memory_limit_scope": _MEMORY_SCOPE.get(sys.platform, _MEMORY_SCOPE["linux"]),
        "file_limit_scope": (
            "not available on Windows"
            if sys.platform == "win32"
            else "each individual file; run-owned spill has its separate budget"
        ),
    }


def run(request_path: Path | str, result_path: Path | str, control: dict[str, Any]) -> None:
    """Publish one complete result or a small credential-safe error then fail."""
    request_path, result_path = Path(request_path), Path(result_path)
    try:
        settings, queries, options = read_request(request_path)
        local = _local_settings(settings, request_path.parent / "work")
        frames = _adapt(queries, local)
        usage = _usage(control)
        if control["operation"] == "validate":
            runner = QARunner(local)
            report = runner.run(frames=frames)
            config = redact(settings.model_dump(mode="python", by_alias=True))
            metadata = report.metadata.model_copy(update={
                "config": config,
                "config_digest": config_digest(config),
                "resource_usage": {**report.metadata.resource_usage, **usage},
            })
            report = runner._fit_report_budget(report.model_copy(update={"metadata": metadata}))
            chunks = report.iter_json_chunks()
        elif control["operation"] == "compare":
            from qavf.regression._inputs import compare_context

            keys = options.pop("keys")
            with DataContext.from_settings(
                local, frames, threads=local.engine.native_threads, execution_workers=1
            ) as context:
                comparison, _ = compare_context(
                    context, "candidate", keys, baseline="baseline", **options
                )
                if failures := context.verify_inputs():
                    raise ValueError(f"comparison source identity changed: {failures}")
            comparison = comparison.model_copy(update={"execution_limits": usage})
            chunks = _iter_snapshot_json(comparison)
        elif control["operation"] == "compare_datasets":
            from qavf.compare.engine import compare_in_context
            from qavf.compare.spec import ComparisonSpec

            spec = ComparisonSpec.model_validate(options["comparison"])
            with DataContext.from_settings(
                local, frames, threads=local.engine.native_threads, execution_workers=1
            ) as context:
                outcome = compare_in_context(context, spec)
                if failures := context.verify_inputs():
                    raise ValueError(f"comparison source identity changed: {failures}")
            outcome = outcome.model_copy(update={"execution_limits": usage})
            chunks = _iter_snapshot_json(outcome)
        else:
            raise ValueError("unknown isolated operation")
        write_result(result_path, chunks, int(control["max_report_bytes"]))
    except Exception as exc:
        message = exception_details(exc)[:1500]
        error = json.dumps({
            "phase": "execution", "error_type": type(exc).__name__, "message": message
        }, ensure_ascii=True).encode()
        # Failure to write (e.g. RLIMIT_FSIZE/ENOMEM) is still a nonzero exit;
        # the parent will reject the missing result without guessing the cause.
        Path(control["error_path"]).write_bytes(error[:8192])
        raise SystemExit(1) from None
