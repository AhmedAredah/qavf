"""Public validation orchestration over explicit input descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qavf.engine.runner import QARunner
from qavf.inputs.normalize import Configuration, prepare
from qavf.inputs.resources import ResourceLimits
from qavf.reports.models import QAReport
from qavf.settings import SchemaSpec


def validate(
    data: object | Mapping[str, object] | None = None,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = None,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
) -> QAReport:
    """Validate explicit contracts and return a report without writing files.

    Pass a path, File, frame, Query, Loader, or named mapping of those inputs.
    Dataset wraps an input with its own schema, foreign keys, expressions and
    loading safeguards. Plain mappings always name datasets, never columns.
    All input names and contracts are checked before adapting or executing data.
    Lazy frames, deferred loaders and caller-owned handles keep their ownership.
    """
    settings, frames = prepare(data, schema=schema, config=config, resources=resources)
    return QARunner(settings).run(frames=frames)


__all__ = ["validate"]
