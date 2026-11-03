"""Public validation orchestration over explicit input descriptions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qavf._input_types import ValidationInput
    from qavf.inputs.normalize import Configuration
    from qavf.inputs.resources import ResourceLimits
    from qavf.profiles import ProfileName
    from qavf.reports.models import QAReport
    from qavf.run_control import CancellationToken, RunEvent
    from qavf.settings import SchemaSpec


def validate(
    data: ValidationInput = None,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = None,
    config: Configuration = None,
    resources: ResourceLimits | None = None,
    profile: ProfileName | None = None,
    backend: str = "auto",
    on_event: Callable[[RunEvent], None] | None = None,
    cancel: CancellationToken | None = None,
) -> QAReport:
    """Validate explicit contracts and return a report without writing files.

    Pass a path, File, frame, Query, Loader, or named mapping of those inputs.
    Dataset wraps an input with its own schema, foreign keys, check items and
    loading safeguards. Plain mappings always name datasets, never columns.
    All input names and contracts are checked before adapting or executing data.
    Lazy frames, deferred loaders and caller-owned handles keep their ownership.
    ``profile`` explicitly combines declared validity checks with configured
    regression or change analysis; it never infers acceptance limits.

    ``backend="auto"`` preserves the full engine when Polars is installed and
    otherwise uses the DuckDB/Arrow core. ``"duckdb"`` explicitly selects that
    core; ``"polars"`` requires the optional extra. Unsupported configured core
    capabilities raise an actionable optional-dependency error before execution.

    ``on_event`` receives a :class:`~qavf.RunEvent` as each step starts and
    finishes; ``cancel`` is a :class:`~qavf.CancellationToken` whose ``cancel()``
    stops the run from any thread. A cancelled run still returns its report, with
    the steps it did not finish reported as not assessed. Both need
    ``engine.execution_mode: in_process``.
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
            data,
            schema=schema,
            config=config,
            resources=resources,
            profile=profile,
            on_event=on_event,
            cancel=cancel,
        )
    from qavf.engine.runner import QARunner
    from qavf.inputs.normalize import prepare
    from qavf.profiles import apply_profile
    from qavf.run_control import control_for

    if profile is None:
        settings, frames = prepare(data, schema=schema, config=config, resources=resources)
    else:
        from qavf.inputs.normalize import _adapt, prepare_configuration

        settings, inputs = prepare_configuration(
            data, schema=schema, config=config, resources=resources
        )
        settings = apply_profile(settings, profile)
        frames = _adapt(inputs, settings)
    control = control_for(on_event, cancel, settings.engine.execution_mode)
    if control is None:
        # Runners overridden without the hooks keep working when none is requested.
        return QARunner(settings).run(frames=frames)
    return QARunner(settings).run(frames=frames, control=control)


__all__ = ["validate"]
