"""Public validation orchestration over explicit input descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from qavf.engine.runner import QARunner
from qavf.inputs.normalize import Configuration, prepare
from qavf.inputs.resources import ResourceLimits
from qavf.profiles import ProfileName, apply_profile
from qavf.reports.models import QAReport
from qavf.settings import SchemaSpec


def validate(
    data: object | Mapping[str, object] | None = None,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = None,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
    profile: ProfileName | None = None,
) -> QAReport:
    """Validate explicit contracts and return a report without writing files.

    Pass a path, File, frame, Query, Loader, or named mapping of those inputs.
    Dataset wraps an input with its own schema, foreign keys, expressions and
    loading safeguards. Plain mappings always name datasets, never columns.
    All input names and contracts are checked before adapting or executing data.
    Lazy frames, deferred loaders and caller-owned handles keep their ownership.
    ``profile`` explicitly combines declared validity checks with configured
    regression or change analysis; it never infers acceptance limits.
    """
    if profile is None:
        settings, frames = prepare(data, schema=schema, config=config, resources=resources)
    else:
        from qavf.inputs.normalize import _adapt, prepare_configuration

        settings, inputs = prepare_configuration(
            data, schema=schema, config=config, resources=resources
        )
        settings = apply_profile(settings, profile)
        frames = _adapt(inputs, settings)
    return QARunner(settings).run(frames=frames)


__all__ = ["validate"]
