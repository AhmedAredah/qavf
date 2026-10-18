"""Revalidate mutable engine settings before selecting an execution boundary."""

from qavf.settings import EngineSettings


def check_policy(engine: EngineSettings) -> None:
    """Reject mutated/inconsistent limits before discovery, loading, or execution."""
    EngineSettings.model_validate(engine.model_dump())
