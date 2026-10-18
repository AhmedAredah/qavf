"""Small shared contracts for fixture adapters and isolated benchmark workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadOptions:
    """Execution choices; preparation is measured separately from the operation."""

    backend: str = "native"
    input_format: str = "parquet"
    evidence_rows: int = 20
    evidence_bytes: int = 65_536
    threads: int = 2
    hash_inputs: bool = False
    memory_budget_bytes: int | None = None
    cache_max_bytes: int | None = None
    spill_max_bytes: int | None = None
    execution_mode: str = "in_process"

    def __post_init__(self) -> None:
        if self.execution_mode not in {"in_process", "isolated"}:
            raise ValueError("unknown execution mode")
        if self.threads < 1:
            raise ValueError("threads must be positive")
        if self.evidence_rows < 0 or self.evidence_bytes < 0:
            raise ValueError("evidence limits must be nonnegative")
        for name in ("memory_budget_bytes", "cache_max_bytes", "spill_max_bytes"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class PreparedWorkload:
    """A prepared input adapter, the real operation, and an independent oracle."""

    run: Callable[[], Any]
    verify: Callable[[Any], dict[str, Any]]
    metadata: dict[str, Any]
