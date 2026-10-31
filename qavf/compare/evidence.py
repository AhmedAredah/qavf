"""Admission of optional comparison payloads without changing assessed outcomes.

Counts and statistical decisions are mandatory outcome state. Rich records,
per-member details and listed tests are optional, bounded before snapshots are
published. Built-in producers use the same collectors before constructing their
models; the final level boundary also protects results supplied by plugins.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from qavf.audit_codec import AuditSizeError, iter_audit_json
from qavf.compare.results import CheckResult, LevelResult
from qavf.inference.results import Significance
from qavf.rules.helpers import BoundedEvidence
from qavf.snapshots import SnapshotModel


def size_within(value: Any, limit: int) -> int | None:
    """Size a canonical payload with an early byte stop and bounded string chunks."""
    try:
        return sum(len(chunk) for chunk in iter_audit_json(value, max_bytes=max(0, limit)))
    except AuditSizeError:
        return None


def keep_models[T: SnapshotModel](values: Iterable[T], rows: int, byte_limit: int) -> list[T]:
    """Admit an ordered prefix of already computed optional models."""
    if rows <= 0 or byte_limit <= 0:
        return []
    collector = BoundedEvidence(rows, byte_limit=byte_limit)
    kept: list[T] = []
    for value in values:
        if len(kept) >= collector.limit or byte_limit == 0:
            break
        collector.append(value.model_dump(mode="python"))
        if len(collector.records) == len(kept):
            break
        kept.append(value)
    return kept


def model_records[T: SnapshotModel](
    model: type[T], records: Iterable[Mapping[str, Any]], rows: int, byte_limit: int
) -> list[T]:
    """Admit raw fields before immutable model construction and nested copying."""
    collector = BoundedEvidence(rows, byte_limit=byte_limit)
    if rows and byte_limit:
        for record in records:
            collector.append(record)
            if len(collector.records) >= collector.limit:
                break
    return [model.model_validate(record) for record in collector.records]


def append_model[T: SnapshotModel](
    kept: list[T], collector: BoundedEvidence, build: Callable[[], T]
) -> None:
    """Build optional rich detail only while its prefix budget has room."""
    if len(kept) >= collector.limit or collector.byte_limit == 0:
        return
    value = build()
    collector.append(value.model_dump(mode="python"))
    if len(collector.records) > len(kept):
        kept.append(value)


def bound_significance(
    value: Significance | None, rows: int, byte_limit: int
) -> Significance | None:
    if value is None:
        return None
    tests = keep_models(value.tests, rows, byte_limit)
    return value if len(tests) == len(value.tests) else value.model_copy(update={"tests": tests})


def _field_overhead(name: str, first: bool) -> int:
    """JSON framing of one field: its quoted name, colon and a brace or comma."""
    return len(name.encode("utf-8")) + (5 if first else 4)


def bound_check(check: CheckResult, rows: int, byte_limit: int) -> CheckResult:
    """Cap the optional object containing metrics, records and listed tests.

    The canonical budget scope is the nonempty subset of these three fields;
    required verdicts and counts belong to the report budget. Metrics hold the
    family summary and exact values, so they are admitted first and sample
    records take what remains. Braces, field names, colons, commas and each
    nested payload are charged.
    """
    used = 0
    metrics = check.metrics
    if metrics:
        framing = _field_overhead("metrics", True)
        metrics_size = size_within(metrics, byte_limit - framing)
        if metrics_size is None or framing + metrics_size > byte_limit:
            metrics = {}
        else:
            used += framing + metrics_size
    framing = _field_overhead("affected_records", used == 0)
    evidence = BoundedEvidence(rows, byte_limit=max(0, byte_limit - used - framing))
    evidence.extend(check.affected_records)
    if evidence.records:
        used += framing + evidence.bytes_used
    significance = check.significance
    if significance is not None:
        framing = _field_overhead("tests", used == 0)
        tests = keep_models(significance.tests, rows, max(0, byte_limit - used - framing))
        significance = significance.model_copy(update={"tests": tests})
    omitted = len(evidence.records) != len(check.affected_records) or bool(
        check.metrics and not metrics
    )
    if significance is not None and check.significance is not None:
        omitted |= len(significance.tests) != len(check.significance.tests)
    return check.model_copy(
        update={
            "affected_records": evidence.records,
            "metrics": metrics,
            "significance": significance,
            "truncated": check.truncated or omitted,
        }
    )


def bound_level(
    level: LevelResult, rows: int, byte_limit: int, *, detail_rows: int | None = None
) -> LevelResult:
    """Defensive publication boundary, including extension results and nested details."""
    any_dropped = False
    details = rows if detail_rows is None else detail_rows
    detail_remaining = byte_limit

    def trim(value: Any, field: str = "") -> Any:
        nonlocal any_dropped, detail_remaining
        if isinstance(value, Significance):
            bounded = bound_significance(value, rows, max(0, detail_remaining - 10))
            assert bounded is not None
            if bounded.tests:
                size = size_within(
                    [test.model_dump(mode="python") for test in bounded.tests],
                    max(0, detail_remaining - 10),
                )
                assert size is not None
                detail_remaining -= size + 10  # {"tests": [...]} framing
            any_dropped |= len(bounded.tests) != len(value.tests)
            return bounded
        if isinstance(value, CheckResult):
            return bound_check(value, rows, byte_limit)
        if isinstance(value, SnapshotModel):
            updates: dict[str, Any] = {}
            optional = set(getattr(value, "detail_fields", ()))
            # These models keep mandatory aggregate counts alongside optional detail.
            optional.update(
                name
                for name in ("contributions", "strata", "transitions")
                if name in type(value).model_fields
            )
            fields_kept = 0
            dropped = False
            for name in type(value).model_fields:
                original = getattr(value, name)
                if name in optional:
                    if not original:
                        continue
                    candidates = original if isinstance(original, (list, tuple)) else [original]
                    kept: list[Any] = []
                    used = 0
                    remaining = detail_remaining
                    overhead = len(name.encode("utf-8")) + (5 if fields_kept == 0 else 4)
                    available = max(0, remaining - overhead)
                    sequence = isinstance(original, (list, tuple))
                    for item in candidates:
                        if len(kept) >= details or available == 0:
                            break
                        raw = (
                            item.model_dump(mode="python")
                            if isinstance(item, SnapshotModel)
                            else item
                        )
                        framing = (2 if not kept else 1) if sequence else 0
                        size = size_within(raw, available - used - framing)
                        if size is None or used + size + framing > available:
                            break
                        kept.append(trim(item, name))
                        used += size + framing
                    if kept:
                        detail_remaining = max(0, detail_remaining - used - overhead)
                        fields_kept += 1
                    dropped |= len(kept) != len(candidates)
                    updates[name] = (
                        kept if isinstance(original, (list, tuple)) else (kept[0] if kept else None)
                    )
                else:
                    updates[name] = trim(original, name)
            any_dropped |= dropped
            if dropped and "truncated" in type(value).model_fields:
                updates["truncated"] = True
            return value.model_copy(update=updates)
        if isinstance(value, (list, tuple)):
            return [trim(item, field) for item in value]
        return value

    result: LevelResult = trim(level)
    return result.model_copy(update={"details_truncated": level.details_truncated or any_dropped})
