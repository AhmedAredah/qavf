"""Public validation orchestration over explicit input descriptions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qavf._input_types import ValidationInput
    from qavf.backend_choice import BackendRequest
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
    backend: str | BackendRequest | None = None,
    on_event: Callable[[RunEvent], None] | None = None,
    cancel: CancellationToken | None = None,
    only: str | Iterable[str] | None = None,
    tags: str | Iterable[str] | None = None,
    skip_tags: str | Iterable[str] | None = None,
    datasets: str | Iterable[str] | None = None,
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
    Without ``backend``, the configuration's ``engine.backend`` decides (``auto``
    when it pins none); an explicit ``backend``, ``"auto"`` included, overrides it
    (:mod:`qavf.backend_choice`). The report's ``metadata.backend`` and
    ``metadata.backend_source`` record the outcome.

    ``resources`` (:class:`~qavf.ResourceLimits`) and the configuration's
    ``resources:`` section and ``engine`` limits combine: a limit given in more
    than one of them must have the same value in each, otherwise this raises an
    error naming every place (:mod:`qavf.resource_section`).

    ``on_event`` receives a :class:`~qavf.RunEvent` as each step starts and
    finishes; ``cancel`` is a :class:`~qavf.CancellationToken` whose ``cancel()``
    stops the run from any thread. A cancelled run still returns its report, with
    the steps it did not finish reported as not assessed. Both need
    ``engine.execution_mode: in_process``.

    ``only`` (check item names), ``tags``, ``skip_tags`` and ``datasets`` narrow the
    run without editing the configuration, like ``qavf run --only/--tags/--skip-tags/
    --dataset``. Built-in checks carry their family (``temporal``, ``input``, …) as
    an implicit tag; schema contracts carry ``input``. Selecting nothing is an error
    (:class:`~qavf.checks.selection.SelectionError`).
    """
    from qavf.backend_choice import BackendChoice
    from qavf.checks.selection import Selection, narrow

    selection = Selection.of(only=only, tags=tags, skip_tags=skip_tags, datasets=datasets)
    choice = BackendChoice.for_config(backend, config)
    if choice.backend == "duckdb":
        from qavf.core.runner import validate_core

        return validate_core(
            data,
            schema=schema,
            config=config,
            resources=resources,
            profile=profile,
            on_event=on_event,
            cancel=cancel,
            selection=selection,
            backend=choice,
        )
    from qavf.engine.runner import QARunner
    from qavf.inputs.normalize import _adapt, prepare_configuration
    from qavf.profiles import apply_profile
    from qavf.run_control import control_for

    # Resolve and narrow before adapting: an input the selection excludes is never
    # converted or consumed (a one-shot Arrow stream stays unread).
    settings, inputs = prepare_configuration(
        data, schema=schema, config=config, resources=resources
    )
    if profile is not None:
        settings = apply_profile(settings, profile)
    settings = narrow(settings, selection)
    settings.apply_backend(choice)
    frames = _adapt(inputs, settings)
    control = control_for(on_event, cancel, settings.engine.execution_mode)
    if control is None:
        # Runners overridden without the hooks keep working when none is requested.
        return QARunner(settings).run(frames=frames)
    return QARunner(settings).run(frames=frames, control=control)


__all__ = ["validate"]
