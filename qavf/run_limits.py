"""The resource limits a run's execution reads, taken once from ``settings.engine``.

Both backends read the same few limits to size a run: worker counts, memory and
spill targets, the temporary directory, native threads and evidence bounds.
:class:`RunLimits` is that one immutable reading, so the Polars data context and
the native DuckDB runner resolve them (the temporary directory against the
configuration's directory, the default worker count from the host's CPUs) in one
place. Where each limit was written is :mod:`qavf.resource_section`'s concern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qavf.settings import QAVFSettings

_DEFAULT_WORKERS = 8
"""Workers assumed when ``max_workers`` is not set (and the CPU-count cap)."""


@dataclass(frozen=True, slots=True)
class RunLimits:
    """One run's execution limits (``engine.*``), read once and immutable."""

    execution_mode: str
    parallel: bool
    max_workers: int | None
    native_threads: int | None
    memory_budget_bytes: int | None
    duckdb_memory_limit: str | None
    cache_max_bytes: int
    spill_max_bytes: int
    spill_given: bool
    """Whether ``spill_max_bytes`` was given (in ``engine``, ``resources:`` or
    ``resources=``, which the resource fold merges into ``engine``); when it was
    not, the resource plan sizes spill space from the inputs."""
    max_spill_files: int
    temp_directory: Path | None
    """``engine.temp_directory`` resolved against the configuration's directory."""
    evidence_limit: int
    """``engine.max_affected_records``."""
    evidence_bytes: int
    """``engine.max_evidence_bytes``."""
    concurrent_loads: bool
    canonical_columns: tuple[str, ...]

    @classmethod
    def of(cls, settings: QAVFSettings) -> RunLimits:
        """The limits of ``settings``."""
        engine = settings.engine
        return cls(
            execution_mode=engine.execution_mode,
            parallel=engine.parallel,
            max_workers=engine.max_workers,
            native_threads=engine.native_threads,
            memory_budget_bytes=engine.memory_budget_bytes,
            duckdb_memory_limit=engine.duckdb_memory_limit,
            cache_max_bytes=engine.cache_max_bytes,
            spill_max_bytes=engine.spill_max_bytes,
            spill_given="spill_max_bytes" in engine.model_fields_set,
            max_spill_files=engine.max_spill_files,
            temp_directory=(
                settings.resolve(engine.temp_directory)
                if engine.temp_directory is not None
                else None
            ),
            evidence_limit=engine.max_affected_records,
            evidence_bytes=engine.max_evidence_bytes,
            concurrent_loads=engine.concurrent_loads,
            canonical_columns=tuple(engine.resolved_canonical_columns()),
        )

    def requested_workers(self) -> int:
        """Workers a parallel run asks for before memory and the resource plan cap
        them: ``max_workers``, else the host's CPUs up to eight; one when serial."""
        if not self.parallel:
            return 1
        return max(1, self.max_workers or min(_DEFAULT_WORKERS, os.cpu_count() or 1))

    def connection_spill_bytes(self) -> int:
        """The spill share of one native connection: half of ``spill_max_bytes``
        divided among the configured (or default) workers."""
        return self.spill_max_bytes // (2 * max(1, self.max_workers or _DEFAULT_WORKERS))


__all__ = ["RunLimits"]
