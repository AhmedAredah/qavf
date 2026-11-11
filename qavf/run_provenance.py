"""The backend and resource limits a run used, with where each came from, for its report.

Both runners stamp their report with :class:`RunProvenance`, and so does the parent
of an isolated worker (whose settings no longer know how the backend or a limit
was chosen). The backend choice is resolved once per run and passed to it
explicitly, also across the isolated dispatch: ``metadata.backend``, ``metadata.backend_source`` and
``metadata.resource_usage["limits"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qavf.backend_choice import BackendChoice
from qavf.resource_section import EffectiveLimits

if TYPE_CHECKING:
    from qavf.reports.models import QAReport
    from qavf.settings import QAVFSettings


class RunProvenance:
    """How the backend and each resource limit of a run were given."""

    def __init__(self, settings: QAVFSettings, choice: BackendChoice) -> None:
        self.settings = settings
        self.choice = choice

    @classmethod
    def of(cls, settings: QAVFSettings, backend: str) -> RunProvenance:
        """The provenance of a run of ``backend``: the choice recorded on ``settings``
        when it names that backend, else how ``engine.backend`` chose it."""
        return cls(settings, BackendChoice.ran(settings, backend))

    def limits(self) -> dict[str, dict[str, Any]]:
        """Each limit set anywhere (not left at its default): value, source, origin."""
        return {
            row["name"]: {
                "value": row["value"],
                "source": row["source"],
                "written": list(row["written"]),
                "origin": row["origin"],
            }
            for row in EffectiveLimits(self.settings).rows()
            if row["source"] != "default"
        }

    def stamp(self, report: QAReport) -> QAReport:
        """``report`` with the backend, its source and the limits in its metadata."""
        metadata = report.metadata.model_copy(
            update={
                "backend": self.choice.backend,
                "backend_source": self.choice.source,
                "resource_usage": {**report.metadata.resource_usage, "limits": self.limits()},
            }
        )
        return report.model_copy(update={"metadata": metadata})


__all__ = ["RunProvenance"]
