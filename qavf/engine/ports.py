"""Narrow ports of the run's data context, for collaborators that need only part of it.

:class:`~qavf.engine.context.DataContext` is the one public facade every rule,
source, comparison level and plugin receives, and it keeps every method it has.
Internally, code that needs only one capability of it types its parameter as the
narrowest port below instead, so what it depends on is explicit and a test (or
another backend) can supply just that capability:

- :class:`SourceLookup`: the configured source of a dataset;
- :class:`DatasetReads`: the catalog and eager reads (``get``, ``has``, ``declared``,
  ``describe``, ``schema``, specs and why a dataset is unavailable);
- :class:`DatasetReductions`: lazy scans, projections and single-source reductions;
- :class:`SqlQueries`: DuckDB SQL over datasets, bounded samples, native consumers;
- :class:`ObservationAccess`: owned, coherent projections (observations);
- :class:`WorkStorage`: the run's memory target, owned work stores and spill files,
  and releasing datasets after their last consumer;
- :class:`RuleSessions`: rule invocations and their cancellation;
- :class:`EvidencePolicy`: how much evidence a rule keeps per result;
- :class:`RunReporting`: what the run's report reads at the end.

Each port is a :class:`typing.Protocol` the facade satisfies structurally; mypy
checks that it does (:func:`_facade_satisfies_every_port`). Ports compose:
:class:`FindingReads` is what a rule's findings need to be completed and
:class:`LevelReads` what a group comparison level reads.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

if TYPE_CHECKING:
    from qavf.engine.context import DataContext
    from qavf.engine.observation import Observation, Observations
    from qavf.engine.observation_plan import ObservationPlan
    from qavf.engine.session import Invocation
    from qavf.engine.spill import SpillFile
    from qavf.engine.sql_read import SqlReadSession
    from qavf.engine.sqlite_store import Store
    from qavf.reports.models import DatasetInfo
    from qavf.settings import DatasetSpec
    from qavf.sources.base import DataSource


class SourceLookup(Protocol):
    """The configured source of a dataset."""

    def source(self, name: str) -> DataSource | None:
        """The configured source of ``name``; ``None`` for in-memory inputs."""
        ...


class DatasetReads(SourceLookup, Protocol):
    """The run's dataset catalog and eager reads."""

    @property
    def specs(self) -> dict[str, DatasetSpec]:
        """Dataset declarations."""
        ...

    def contract_spec(self, name: str) -> DatasetSpec | None:
        """The declaration whose contract ``name`` follows (a view's parent's)."""
        ...

    def names(self) -> list[str]:
        """Every dataset that has data or a source, loaded or not."""
        ...

    def declared(self) -> list[str]:
        """Every dataset named in the configuration or supplied in memory."""
        ...

    def has(self, name: str) -> bool:
        """Whether the dataset is accessible."""
        ...

    def unavailable_reason(self, name: str) -> str | None:
        """Why ``name`` could not be loaded, or ``None`` if it is available."""
        ...

    def get(self, name: str) -> pl.DataFrame:
        """Dataset ``name``, loaded if needed."""
        ...

    def schema(self, name: str) -> dict[str, pl.DataType]:
        """A dataset's schema, without materializing native scan sources."""
        ...

    def describe(self) -> list[DatasetInfo]:
        """The loaded datasets, described."""
        ...


class DatasetReductions(Protocol):
    """Lazy scans, projections and reductions under each source's policy."""

    def scan(self, name: str) -> pl.LazyFrame:
        """A lazy plan of ``name``."""
        ...

    def collect(self, name: str, plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect a projected plan of ``name`` under its source policy."""
        ...

    def collect_many(self, names: Iterable[str], plan: pl.LazyFrame) -> pl.DataFrame:
        """Collect one plan over several datasets under every one's policy."""
        ...

    def project(self, name: str, columns: Iterable[str]) -> pl.DataFrame:
        """Selected raw columns of ``name``."""
        ...

    def reduce(self, name: str, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        """Collect a single-source lazy reduction of ``name`` (a view included)."""
        ...


class SqlQueries(Protocol):
    """DuckDB SQL over the run's datasets."""

    def sql(
        self,
        query: str,
        datasets: Iterable[str] | None = None,
        *,
        normalize_missing: bool = False,
    ) -> pl.DataFrame:
        """The result of ``query``, as one verified, retryable operation."""
        ...

    def sql_sample(
        self, query: str, limit: int, *, ordered: bool = False, normalize_missing: bool = False
    ) -> tuple[pl.DataFrame, int]:
        """A bounded sample of ``query``'s rows and their count."""
        ...

    def consume_sql[T](
        self,
        consumer_factory: Callable[[], Callable[[SqlReadSession], T]],
        *,
        datasets: Iterable[str] = (),
        sources: Mapping[str, Mapping[str, Any]] | None = None,
        native_only: bool = True,
    ) -> T:
        """Consume native SQL batches inside one verified, resettable operation."""
        ...

    def referenced(self, query: str) -> list[str]:
        """Datasets ``query`` reads as tables."""
        ...


class ObservationAccess(Protocol):
    """Owned, coherent projections of datasets."""

    def observe(
        self, name: str, columns: Iterable[str] | None = None
    ) -> contextlib.AbstractContextManager[Observation]:
        """One coherent projection of ``name``."""
        ...

    def projected_observations(
        self, selections: Mapping[str, Iterable[str] | None]
    ) -> contextlib.AbstractContextManager[Observations]:
        """Stable projections for one multi-input operation."""
        ...

    def observation_plan(
        self, contracts: Sequence[Mapping[str, Iterable[str] | None]]
    ) -> ObservationPlan:
        """Projections reused until each input's final scheduled use."""
        ...


class WorkStorage(Protocol):
    """The run's memory target, owned work storage and dataset lifetimes."""

    @property
    def memory_budget_bytes(self) -> int | None:
        """The memory target for materialized results (``None``: none set)."""
        ...

    def check_materialization(self, name: str, frame: pl.DataFrame) -> None:
        """Admit a completed result against the memory target."""
        ...

    def check_retained_bytes(self, name: str, size: int) -> None:
        """Admit measured retained state against the memory target."""
        ...

    def work_store(self, label: str) -> Store:
        """A bounded owned index under the run's shared spill reservations.

        It may outlive the invocation that created it (an inference family's
        measured tests are judged in a later invocation)."""
        ...

    def spill_file(self, path: Path) -> SpillFile:
        """A caller-selected staging file under the run's disk quota."""
        ...

    def release(self, names: Iterable[str]) -> None:
        """Release cached data after its last planned consumer."""
        ...


class RuleSessions(Protocol):
    """Rule invocations and their cancellation."""

    def begin_rule(self, invocation: Invocation | None = None) -> Invocation:
        """Start an invocation on this thread."""
        ...

    def end_rule(self) -> None:
        """End this thread's invocation."""
        ...

    def cancel_rule(self, invocation: Invocation) -> None:
        """Cancel one invocation."""
        ...

    def cancel_active_rules(self) -> None:
        """Cancel every running invocation."""
        ...

    def has_active_rules(self) -> bool:
        """Whether any invocation is running now."""
        ...

    def wait_for_rules(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for running invocations to end; whether they did."""
        ...

    def finish_worker(self, owner: int | None = None) -> None:
        """Release the connections of a worker that has stopped."""
        ...

    def thread_load_seconds(self) -> float:
        """Seconds this thread has spent loading datasets."""
        ...


class EvidencePolicy(Protocol):
    """How much evidence a rule keeps per result."""

    evidence_limit: int
    """Offending records kept per result (``engine.max_affected_records``)."""
    evidence_bytes: int
    """Bytes of evidence kept per result (``engine.max_evidence_bytes``)."""


class FindingReads(DatasetReads, EvidencePolicy, Protocol):
    """What completing a rule's findings reads: datasets and the evidence bound."""


class LevelReads(DatasetReads, DatasetReductions, EvidencePolicy, Protocol):
    """What a group comparison level reads: datasets, projections, the evidence bound."""


class RunReporting(Protocol):
    """What the run's report reads once its rules have run."""

    def reporting_snapshot(self) -> dict[str, Any]:
        """Dataset metadata and resource usage, without joining overdue work."""
        ...

    def resource_usage(self) -> dict[str, Any]:
        """The run's resource usage."""
        ...

    def verify_inputs(self) -> dict[str, str]:
        """Final identity check of every source used."""
        ...


if TYPE_CHECKING:
    from qavf.engine.admission import AdmissionPolicy

    def _admission_is_the_evidence_policy(policy: AdmissionPolicy) -> EvidencePolicy:
        """Checked by mypy only: the facade's evidence bounds come from its policy."""
        return policy

    def _facade_satisfies_every_port(
        context: DataContext,
    ) -> tuple[
        SourceLookup,
        DatasetReads,
        DatasetReductions,
        SqlQueries,
        ObservationAccess,
        WorkStorage,
        RuleSessions,
        EvidencePolicy,
        FindingReads,
        LevelReads,
        RunReporting,
    ]:
        """Checked by mypy only: the data context provides every port."""
        return (
            context,
            context,
            context,
            context,
            context,
            context,
            context,
            context,
            context,
            context,
            context,
        )


PORTS: tuple[type, ...] = (
    SourceLookup,
    DatasetReads,
    DatasetReductions,
    SqlQueries,
    ObservationAccess,
    WorkStorage,
    RuleSessions,
    EvidencePolicy,
    FindingReads,
    LevelReads,
    RunReporting,
)
"""Every port, for tests that check the facade still provides each of them."""


__all__ = [
    "PORTS",
    "DatasetReads",
    "DatasetReductions",
    "EvidencePolicy",
    "FindingReads",
    "LevelReads",
    "ObservationAccess",
    "RuleSessions",
    "RunReporting",
    "SourceLookup",
    "SqlQueries",
    "WorkStorage",
]
