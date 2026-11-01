"""Explicit failures of the supervised execution contract."""

from qavf.resource_errors import ResourceLimitError


class ExecutionLimitError(ResourceLimitError):
    """A supervised run could not return a complete, validated result.

    No partial result is accepted after a deadline, worker crash, failed limit
    setup, or invalid transport response. The caller may retry with a new budget.
    """
