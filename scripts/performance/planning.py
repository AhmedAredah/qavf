"""Validate benchmark selection and deterministically alternate revision order."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from .catalog import selected_cases
from .models import WorkloadOptions

PROFILES = {
    "smoke": {"rows": 10_000, "repetitions": 1, "warmups": 0},
    "standard": {"rows": 250_000, "repetitions": 3, "warmups": 1},
    "capacity": {"rows": 5_000_000, "repetitions": 3, "warmups": 1},
}


def variants(
    names: list[str] | None,
    rows: list[int],
    options: WorkloadOptions,
    backends: list[str] | None = None,
    formats: list[str] | None = None,
    execution_modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rows or any(value < 100 for value in rows) or len(rows) != len(set(rows)):
        raise ValueError("Row scales must be distinct integers of at least 100")
    known_backends = {"native", "lazy", "eager", "file", "context"}
    if backends is not None and (not backends or set(backends) - known_backends):
        raise ValueError("Unknown or empty backend selection")
    if formats is not None and (not formats or set(formats) - {"csv", "parquet"}):
        raise ValueError("Unknown or empty format selection")
    result = []
    modes = execution_modes or [options.execution_mode]
    if any(mode not in {"in_process", "isolated"} for mode in modes):
        raise ValueError("Unknown execution mode")
    for size in rows:
        for case in selected_cases(names):
            for backend, file_format in case.adapters:
                if backends is not None and backend not in backends:
                    continue
                if formats is not None and file_format not in formats:
                    continue
                for mode in dict.fromkeys(modes):
                    if mode == "isolated" and not case.supports_isolation:
                        continue
                    selected = replace(
                        options,
                        backend=backend,
                        input_format=file_format,
                        hash_inputs=case.hash_inputs,
                        execution_mode=mode,
                    )
                    result.append(
                        {
                            "case": case.name,
                            "family": case.family,
                            "rows": size,
                            "options": asdict(selected),
                        }
                    )
    if not result:
        raise ValueError("No supported workload adapters match the selection")
    return result


def schedule(
    plan: list[dict[str, Any]], labels: list[str], repetitions: int, warmups: int
) -> list[dict[str, Any]]:
    if not plan:
        raise ValueError("Benchmark plan must contain at least one workload")
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups nonnegative")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Source labels must be nonempty and distinct")
    result = []
    for variant_index, variant in enumerate(plan):
        for warmup, count in ((True, warmups), (False, repetitions)):
            for repetition in range(1, count + 1):
                order = labels if (variant_index + repetition) % 2 else list(reversed(labels))
                for label in order:
                    result.append(
                        {
                            **variant,
                            "source_label": label,
                            "warmup": warmup,
                            "repetition": repetition,
                            "id": f"v{variant_index:03d}-{label}-"
                            f"{'warmup' if warmup else 'measured'}-{repetition}",
                        }
                    )
    return result
