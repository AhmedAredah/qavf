"""Explicit benchmark inventory and lazy workload dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .models import PreparedWorkload, WorkloadOptions


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    adapters: tuple[tuple[str, str], ...]
    hash_inputs: bool = False
    supports_isolation: bool = False


TABULAR = (
    ("native", "parquet"),
    ("native", "csv"),
    ("lazy", "parquet"),
    ("eager", "parquet"),
    ("lazy", "csv"),
    ("eager", "csv"),
)
CASES = (
    Case("wide_validation", "validation", TABULAR),
    Case("relational_validation", "validation", TABULAR),
    Case("grouped_reconciliation", "validation", TABULAR),
    Case("hashed_validation", "validation", (("native", "parquet"),), True),
    Case("regression", "regression", (("file", "parquet"),)),
    Case("cache_spill", "cache", (("context", "parquet"),)),
    Case("streaming_regression", "streaming", (("file", "parquet"),), supports_isolation=True),
    Case("grouped_statistics", "capacity", (("native", "parquet"),), supports_isolation=True),
    Case("scalar_statistics", "capacity", (("native", "parquet"),), supports_isolation=True),
)


def selected_cases(names: list[str] | None) -> list[Case]:
    chosen = {case.name for case in CASES} if names is None else set(names)
    unknown = chosen - {case.name for case in CASES}
    if unknown:
        raise ValueError(f"Unknown benchmark cases: {sorted(unknown)}")
    if not chosen:
        raise ValueError("Select at least one benchmark case")
    return [case for case in CASES if case.name in chosen]


def prepare_fixture(family: str, directory: Path, **options: Any) -> dict[str, Any]:
    name = "fixtures" if family == "validation" else family
    if family not in {"validation", "regression", "cache", "streaming", "capacity"}:
        raise ValueError(f"Unknown fixture family: {family}")
    module = import_module(f"{__package__}.{name}")
    return module.prepare_fixture(directory, **options)


def build_workload(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions
) -> PreparedWorkload:
    selected = selected_cases([case])[0]
    if (options.backend, options.input_format) not in selected.adapters:
        raise ValueError(
            f"Unsupported adapter for {case}: {options.backend}/{options.input_format}"
        )
    if options.execution_mode == "isolated" and not selected.supports_isolation:
        raise ValueError(f"{case} does not support isolated execution")
    if selected.family == "validation":
        module = import_module(f"{__package__}.workloads")
        actual = "wide_validation" if case == "hashed_validation" else case
        return module.build_workload(actual, fixture, work_dir, options)
    module = import_module(f"{__package__}.{selected.family}")
    if selected.family == "capacity":
        return module.build_workload(case, fixture, work_dir, options)
    return module.build_workload(fixture, work_dir, options)
