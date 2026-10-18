"""Policy-preserving regression input dispatch and exact projected fallback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qavf.engine.context import DataContext
from qavf.engine.sql_read import NativeReadUnavailable, SqlReadSession
from qavf.inputs.normalize import prepare
from qavf.inputs.resources import ResourceLimits
from qavf.regression._plan import comparison_plan
from qavf.regression._streaming import compare_native
from qavf.regression.models import ComparisonResult
from qavf.sources.sql import quote_identifier


def compare_context(
    ctx: DataContext,
    candidate: str,
    keys: Sequence[str],
    *,
    baseline: str | None = None,
    source: Mapping[str, Any] | None = None,
    fingerprint: bool = False,
    **options: Any,
) -> tuple[ComparisonResult, dict[str, Any] | None]:
    """Compare configured input and optional ad-hoc source within context policy."""
    if (baseline is None) == (source is None):
        raise ValueError("comparison needs exactly one baseline dataset or source")
    candidate_schema = ctx.schema(candidate)
    if source is not None:
        baseline = "__qavf_regression_baseline"
        occupied = {name.casefold() for name in ctx.declared()}
        while baseline.casefold() in occupied:
            baseline += "_"
    assert baseline is not None
    configured = [candidate] if source is not None else [baseline, candidate]
    ad_hoc = {baseline: source} if source is not None else None
    baseline_schema = ctx.schema(baseline) if source is None else None

    def factory():
        def consume(session: SqlReadSession):
            for alias in (baseline, candidate):
                if not session.repeatable_scan(alias):
                    raise NativeReadUnavailable(
                        f"source {alias!r} has no verified repeatable file reader"
                    )
            schema = (
                baseline_schema
                if baseline_schema is not None
                else session.schema(f"SELECT * FROM {quote_identifier(baseline)} LIMIT 0")
            )
            plan = comparison_plan(schema, candidate_schema, keys, **options)
            result = compare_native(session, baseline, candidate, plan)
            identity = session.fingerprint(baseline, hash_contents=True) if fingerprint else None
            return result, identity

        return consume

    try:
        return ctx.consume_sql(factory, datasets=configured, sources=ad_hoc, native_only=True)
    except NativeReadUnavailable as exc:
        fallback_reason = str(exc)

    # Import after module initialization: compare_frames shares this exact kernel.
    from qavf.regression._indexed import compare_indexed, supported
    from qavf.regression.compare_outputs import _compare_projected

    if source is not None:

        def projected(view):
            plan = comparison_plan(view.schema(), candidate_schema, keys, **options)
            if supported(plan):
                with (
                    view.observe(plan.shared) as left,
                    ctx.observe(candidate, plan.shared) as right,
                    ctx.work_store("typed regression alignment") as store,
                ):
                    result = compare_indexed(
                        store, left, right, plan, fallback_reason=fallback_reason
                    )
                    return result, view.fingerprint() if fingerprint else None
            result = _compare_projected(
                plan,
                view.project(list(plan.shared)),
                ctx.project(candidate, list(plan.shared)),
                execution="projected_fallback",
                fallback_reason=fallback_reason,
            )
            return result, view.fingerprint() if fingerprint else None

        return ctx.read_source(source, projected)
    assert baseline_schema is not None
    plan = comparison_plan(baseline_schema, candidate_schema, keys, **options)
    if supported(plan):
        with (
            ctx.projected_observations({baseline: plan.shared, candidate: plan.shared}) as inputs,
            ctx.work_store("typed regression alignment") as store,
        ):
            return compare_indexed(
                store, inputs[baseline], inputs[candidate], plan, fallback_reason=fallback_reason
            ), None
    result = _compare_projected(
        plan,
        ctx.project(baseline, list(plan.shared)),
        ctx.project(candidate, list(plan.shared)),
        execution="projected_fallback",
        fallback_reason=fallback_reason,
    )
    return result, None


def compare_inputs(
    baseline: Any,
    candidate: Any,
    keys: Sequence[str],
    *,
    resources: ResourceLimits | None = None,
    **options: Any,
) -> ComparisonResult:
    """Normalize public inputs and apply the same resource policy as validation."""
    for value in (baseline, candidate):
        if callable(getattr(value, "read_next_batch", None)):
            raise ValueError(
                "one-pass Arrow readers must be materialized explicitly before comparison"
            )
    settings, frames = prepare(
        {"baseline": baseline, "candidate": candidate},
        schema=None,
        config={"engine": {"parallel": False}, "plugins": {"entry_points": False}},
        resources=resources,
    )
    if settings.engine.execution_mode == "isolated":
        from qavf.execution.dispatch import compare_isolated
        from qavf.regression._columns import column_names

        return compare_isolated(settings, frames, column_names(keys), options)
    with DataContext.from_settings(
        settings,
        frames,
        threads=settings.engine.native_threads,
        execution_workers=1,
    ) as context:
        result, _ = compare_context(context, "candidate", keys, baseline="baseline", **options)
        return result
