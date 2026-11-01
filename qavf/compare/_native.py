"""Native exact record comparison over already aligned, owned observations."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from qavf.duckdb_results import settle
from qavf.engine.sql_read import SqlReadSession
from qavf.regression._plan import ComparisonPlan
from qavf.regression._streaming import compare_native, native_supported
from qavf.regression.models import ComparisonResult

if TYPE_CHECKING:
    from qavf.compare.levels import LevelInputs


def compare_observations(inputs: LevelInputs, plan: ComparisonPlan) -> ComparisonResult:
    """Use the existing exact accumulator without rereading either original source.

    Capability admission precedes every Arrow binding. The run's connection
    owner supplies cancellation, memory, temporary-disk, and close policies.
    Bindings remain leased until the connection and every result reader close.
    """
    native_supported(plan)
    with ExitStack() as stack:
        bindings = {
            side: stack.enter_context(observation.native_binding(plan.shared))
            for side, observation in inputs.observations.items()
        }
        connection = stack.enter_context(
            inputs.ctx.native_read_services().connections.temporary((), {}, {})
        )
        for side, binding in bindings.items():
            settle(connection)
            connection.register(side, binding)
        session = SqlReadSession(inputs.ctx.native_read_services().policy, connection, {})
        try:
            result = compare_native(session, "baseline", "candidate", plan)
            session.close(require_exhausted=True)
            return result
        finally:
            session.close()
