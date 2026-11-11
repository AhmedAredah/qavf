"""Bounded, versioned JSON transport; never pickle or evaluate caller objects."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from qavf.audit_codec import AuditSizeError, decode_audit, iter_audit_json
from qavf.execution.errors import ExecutionLimitError
from qavf.file_replace import replace_file
from qavf.inputs.descriptors import Query
from qavf.settings import QAVFSettings

REQUEST_LIMIT = 16 * 1024 * 1024


def bounded_chunks(chunks: Iterable[bytes], limit: int) -> Iterable[bytes]:
    written = 0
    for chunk in chunks:
        written += len(chunk)
        if written > limit:
            raise ExecutionLimitError(f"isolated execution transport exceeds {limit} bytes")
        yield chunk


def bounded_bytes(chunks: Iterable[bytes], limit: int) -> bytes:
    """Accumulate encoded bytes without retaining one object per JSON fragment."""
    with BytesIO() as output:
        for chunk in bounded_chunks(chunks, limit):
            output.write(chunk)
        return output.getvalue()


def selection_payload(settings: QAVFSettings) -> dict[str, Any] | None:
    """The run-time selection of ``settings`` in transportable form."""
    applied = settings.selection
    if applied is None:
        return None
    return {**applied.selection.as_dict(), "not_selected": sorted(applied.not_selected)}


def restore_selection(settings: QAVFSettings, payload: Any) -> None:
    """Reattach a selection sent by :func:`selection_payload`."""
    if not isinstance(payload, dict):
        return
    from qavf.checks.selection import AppliedSelection, Selection

    values = dict(payload)
    not_selected = frozenset(values.pop("not_selected", ()))
    settings.apply_selection(AppliedSelection(Selection.of(**values), not_selected))


def request_bytes(
    settings: QAVFSettings, inputs: Mapping[str, object], options: Mapping[str, Any] | None = None
) -> bytes:
    queries = {}
    for name, value in inputs.items():
        if not isinstance(value, Query):
            raise ValueError("isolated runner inputs must be transportable Query descriptors")
        queries[name] = {
            "sql": value.sql,
            "connection": value.connection,
            "parameters": dict(value.parameters),
        }
    payload = {
        "version": 1,
        "settings": settings.model_dump(mode="python", by_alias=True),
        "source_path": settings.source_path,
        "engine_fields_set": sorted(settings.engine.model_fields_set),
        "queries": queries,
        "options": dict(options or {}),
        "selection": selection_payload(settings),
    }
    try:
        return bounded_bytes(iter_audit_json(payload, max_bytes=REQUEST_LIMIT), REQUEST_LIMIT)
    except AuditSizeError as exc:
        raise ExecutionLimitError(str(exc)) from None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"isolated settings must contain supported audit values: {exc}") from None


def read_request(path: Path) -> tuple[QAVFSettings, dict[str, object], dict[str, Any]]:
    with path.open("rb") as handle:
        data = handle.read(REQUEST_LIMIT + 1)
    if len(data) > REQUEST_LIMIT:
        raise ExecutionLimitError("isolated request exceeds transport budget")
    payload = decode_audit(json.loads(data))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ExecutionLimitError("invalid isolated request version")
    settings = QAVFSettings.model_validate(payload["settings"])
    settings.source_path = payload["source_path"]
    if "engine_fields_set" in payload:
        settings.engine.__pydantic_fields_set__ = set(payload["engine_fields_set"])
    restore_selection(settings, payload.get("selection"))
    queries: dict[str, object] = {
        name: Query(**value) for name, value in payload["queries"].items()
    }
    return settings, queries, payload["options"]


def write_result(path: Path, chunks: Iterable[bytes], limit: int) -> None:
    """Publish only complete output, never leave a partially valid result name."""
    partial = path.with_suffix(".partial")
    try:
        with partial.open("xb") as handle:
            for chunk in bounded_chunks(chunks, limit):
                handle.write(chunk)
        replace_file(partial, path)
    finally:
        partial.unlink(missing_ok=True)
