"""One owned scratch directory and resource policy for public API workloads."""

from pathlib import Path

from qavf import ResourceLimits

from .models import WorkloadOptions


def resources(directory: Path, options: WorkloadOptions) -> ResourceLimits:
    directory.mkdir(parents=True, exist_ok=True)
    return ResourceLimits(
        isolate=options.execution_mode == "isolated",
        temp_dir=directory,
        native_threads=options.threads,
        max_workers=1,
        target_memory=options.memory_budget_bytes,
        cache_memory=options.cache_max_bytes,
        max_spill=options.spill_max_bytes,
        evidence_rows=options.evidence_rows,
        evidence_bytes=options.evidence_bytes,
    )
