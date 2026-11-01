"""Public validation orchestration over explicit input descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qavf.inputs.normalize import Configuration
    from qavf.inputs.resources import ResourceLimits
    from qavf.profiles import ProfileName
    from qavf.reports.models import QAReport
    from qavf.settings import SchemaSpec


def validate(
    data: object | Mapping[str, object] | None = None,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = None,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
    profile: ProfileName | None = None,
    backend: str = "auto",
) -> QAReport:
    """Validate explicit contracts and return a report without writing files.

    Pass a path, File, frame, Query, Loader, or named mapping of those inputs.
    Dataset wraps an input with its own schema, foreign keys, expressions and
    loading safeguards. Plain mappings always name datasets, never columns.
    All input names and contracts are checked before adapting or executing data.
    Lazy frames, deferred loaders and caller-owned handles keep their ownership.
    ``profile`` explicitly combines declared validity checks with configured
    regression or change analysis; it never infers acceptance limits.

    ``backend="auto"`` preserves the full engine when Polars is installed and
    otherwise uses the DuckDB/Arrow core. ``"duckdb"`` explicitly selects that
    core; ``"polars"`` requires the optional extra. Unsupported configured core
    capabilities raise an actionable optional-dependency error before execution.
    """
    from qavf.optional import has_polars

    if backend not in ("auto", "duckdb", "polars"):
        raise ValueError("backend must be 'auto', 'duckdb', or 'polars'")
    if backend == "polars":
        from qavf.optional import require_polars

        require_polars("backend='polars'")
    if backend == "duckdb" or not has_polars():
        from qavf.core.runner import validate_core

        return validate_core(
            data, schema=schema, config=config, resources=resources, profile=profile
        )
    from qavf.engine.runner import QARunner
    from qavf.inputs.normalize import prepare
    from qavf.profiles import apply_profile

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
