"""Failure boundaries for independently configured rule contracts.

The caller chooses its contract label and diagnostic detail. Shared formatting
keeps an unavailable check unassessed, without changing the surrounding rule's
continuation, severity, or evidence policy.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from qavf.engine.registry import Finding
from qavf.redaction import exception_details


def unassessed_contract(
    label: str,
    reason: str,
    datasets: tuple[str, ...] = (),
    *,
    check: str | None = None,
    affected: Sequence[Mapping[str, Any]] | None = None,
) -> Finding:
    """Describe one unavailable contract, retaining its own check and evidence."""
    return Finding.unassessed(
        f"Not assessed: {label}: {reason}.",
        check=label if check is None else check,
        datasets=datasets,
        affected=affected,
    )


def contract_error(label: str, exc: Exception, datasets: tuple[str, ...] = ()) -> Finding:
    """Keep the concise exception diagnostic used by reconciliation contracts."""
    return unassessed_contract(label, f"{type(exc).__name__}: {exc}", datasets)


@contextmanager
def isolate_contract(
    label: str,
    datasets: tuple[str, ...],
    blocked: list[Finding],
    *,
    affected: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[None]:
    """Record a detailed error for one contract so independent checks can run.

    Process-control exceptions such as KeyboardInterrupt and SystemExit must
    propagate. No operation is retried or replayed inside this boundary.
    """
    try:
        yield
    except Exception as exc:
        blocked.append(
            unassessed_contract(label, exception_details(exc), datasets, affected=affected)
        )
