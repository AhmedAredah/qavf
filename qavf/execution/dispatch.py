"""Typed entry points over the common isolated-process supervisor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from qavf.execution.errors import ExecutionLimitError
from qavf.execution.inputs import isolated_inputs
from qavf.execution.protocol import request_bytes
from qavf.reports.models import QAReport
from qavf.settings import QAVFSettings

if TYPE_CHECKING:
    from qavf.regression.models import ComparisonResult


def validate_isolated(settings: QAVFSettings, frames: Mapping[str, Any] | None) -> QAReport:
    from qavf.execution.supervisor import execute

    inputs = isolated_inputs(frames or {}, settings)
    if set(inputs) != set(frames or {}):
        raise ValueError("pass file inputs to validate() or configure them under datasets")
    if not set(inputs) <= settings.datasets.keys():
        raise ValueError("isolated Query input names must be configured datasets")
    data, _ = execute(settings, request_bytes(settings, inputs))
    try:
        return QAReport.model_validate_json(data)
    except (ValueError, TypeError) as exc:
        raise ExecutionLimitError("isolated worker returned an invalid QA report") from exc


def compare_isolated(
    settings: QAVFSettings, frames: Mapping[str, Any], keys: list[str], options: dict[str, Any]
) -> ComparisonResult:
    from qavf.execution.supervisor import execute
    from qavf.regression.models import ComparisonResult

    inputs = isolated_inputs(frames, settings)
    data, _ = execute(
        settings, request_bytes(settings, inputs, {"keys": keys, **options}), operation="compare"
    )
    try:
        return ComparisonResult.model_validate_json(data)
    except (ValueError, TypeError) as exc:
        raise ExecutionLimitError("isolated worker returned an invalid comparison result") from exc
