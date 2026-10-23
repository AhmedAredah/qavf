"""Coherent projected observations with exact typed batches and owned spill files.

A source is evaluated once before its observation is published. The resulting
frame or IPC files stay alive for the scope; derived plans never revisit that
source. IPC staging bounds owned disk bytes and supports Polars' minimum version
without relying on its newer batch-collection APIs.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import polars as pl

from qavf.engine.cache_ownership import compact_polars, compact_polars_with_size
from qavf.engine.owned_ipc import _SCAN_OPTIONS
from qavf.engine.projection import selected_columns
from qavf.engine.retention import RetainedLease
from qavf.engine.spill import SpillFile, SpillLease
from qavf.sources.base import DataSource, SourceError
from qavf.sources.sql import quote_identifier

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


class Observation:
    """A stable typed projection, valid only inside its owning context manager.

    Plans returned by scan() must be collected while this scope remains open.
    collect()/reduce()/batches() return detached values that may outlive it.
    """

    def __init__(
        self,
        ctx: DataContext,
        *,
        frame: pl.DataFrame | None = None,
        files: Iterable[SpillFile | SpillLease] = (),
        columns: Iterable[str] | None = None,
        retained_bytes: int = 0,
        retained_lease: RetainedLease | None = None,
    ) -> None:
        self._ctx = ctx
        self._frame = frame
        self._files = tuple(files)
        self._columns = None if columns is None else selected_columns(columns)
        self._closed = False
        self._retained_lease = retained_lease or (
            RetainedLease(ctx._resources.retained_budget, retained_bytes)
            if retained_bytes else None
        )

    def _check(self) -> None:
        self._ctx._check_cancelled()
        if self._closed:
            raise RuntimeError("projected observation has ended")

    def scan(self) -> pl.LazyFrame:
        self._check()
        if self._frame is not None:
            plan = self._frame.lazy()
            return plan if self._columns is None else plan.select(self._columns)
        plans = [pl.scan_ipc(owned.path, **_SCAN_OPTIONS) for owned in self._files]
        if not plans:
            raise RuntimeError("observation has no typed representation")
        plan = plans[0] if len(plans) == 1 else pl.concat(plans, how="vertical")
        return plan if self._columns is None else plan.select(self._columns)

    def schema(self) -> dict[str, pl.DataType]:
        return dict(self.scan().collect_schema())

    def borrow(self, columns: Iterable[str] | None = None) -> Observation:
        """Lease the same immutable values independently of the original owner."""
        self._check()
        selected = self._columns if columns is None else selected_columns(columns)
        if selected is not None:
            missing = set(selected) - self.schema().keys()
            if missing:
                raise ValueError(f"observation does not contain columns {sorted(missing)!r}")
        return Observation(
            self._ctx,
            frame=None if self._frame is None else self._frame.clone(),
            files=[owned.borrow() for owned in self._files],
            columns=selected,
            retained_lease=(
                self._retained_lease.borrow() if self._retained_lease is not None else None
            ),
        )

    def collect(self, plan: pl.LazyFrame) -> pl.DataFrame:
        self._check()
        result = plan.collect(engine="streaming")
        # New Polars may retain IPC mappings; measure the detached result.
        if self._files:
            result = cast(pl.DataFrame, compact_polars(result))
        self._ctx._check_materialization("projected observation", result)
        self._ctx._check_cancelled()
        return result

    def reduce(self, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        return self.collect(build(self.scan()))

    @contextlib.contextmanager
    def derived(self, build: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Iterator[Observation]:
        """Own one derived evaluation, including sorts reused by several consumers."""
        plan = self.scan()
        derived = build(plan)
        if derived is plan:
            with self.borrow() as view:
                yield view
            return
        with _observe_plan(self._ctx, derived) as observation:
            yield observation

    def batches(
        self,
        batch_rows: int = 65536,
        *,
        build: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yield bounded result slices, evaluating a derived plan only once.

        Upstream sort/group operators can require native working state; the row
        size bounds decoded result batches, not arbitrary native allocations.
        """
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("batch_rows must be a positive integer")
        self._check()
        if build is not None:
            with self.derived(build) as derived:
                yield from derived.batches(batch_rows)
            return
        if self._frame is not None:
            selected = self._frame if self._columns is None else self._frame.select(self._columns)
            for frame in selected.iter_slices(batch_rows):
                self._check()
                frame = cast(pl.DataFrame, compact_polars(frame))
                self._ctx._check_materialization("observation batch", frame)
                yield frame
            return
        for owned in self._files:
            plan = pl.scan_ipc(owned.path, **_SCAN_OPTIONS)
            if self._columns is not None:
                plan = plan.select(self._columns)
            if hasattr(plan, "collect_batches"):
                for batch in plan.collect_batches(
                    chunk_size=batch_rows, maintain_order=True, engine="streaming"
                ):
                    self._check()
                    frame = cast(pl.DataFrame, compact_polars(batch))
                    self._ctx._check_materialization("observation batch", frame)
                    yield frame
                continue
            rows = int(plan.select(pl.len()).collect(engine="streaming").item())
            for offset in range(0, rows, batch_rows):
                self._check()
                frame = plan.slice(offset, batch_rows).collect(engine="streaming")
                frame = cast(pl.DataFrame, compact_polars(frame))
                self._ctx._check_materialization("observation batch", frame)
                yield frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._frame = None
        if self._retained_lease is not None:
            self._retained_lease.release()
            self._retained_lease = None
        for owned in self._files:
            owned.close()
        self._files = ()

    def __enter__(self) -> Observation:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _file(ctx: DataContext) -> SpillFile:
    return SpillFile(
        ctx._resources.directory / "observations" / f"{uuid4().hex}.arrow",
        ctx._resources.spill_budget,
    )


def _sink(ctx: DataContext, plan: pl.LazyFrame) -> SpillFile:
    ctx._check_cancelled()
    owned = _file(ctx)
    try:
        with owned.writer() as writer:
            plan.sink_ipc(
                writer, compression="uncompressed", maintain_order=True, engine="streaming"
            )
        ctx._check_cancelled()
        return owned
    except BaseException:
        owned.close()
        raise


def _write_batch(ctx: DataContext, frame: pl.DataFrame) -> SpillFile:
    ctx._check_cancelled()
    owned = _file(ctx)
    try:
        with owned.writer() as writer:
            frame.write_ipc(writer, compression="uncompressed")
        return owned
    except BaseException:
        owned.close()
        raise


class _ObservationBuilder:
    """One-pass admission: retain bounded chunks, then spill without source replay."""

    def __init__(self, ctx: DataContext) -> None:
        self.ctx = ctx
        self.frames: list[pl.DataFrame] = []
        self.files: list[SpillFile] = []
        self.bytes = 0
        self.spilling = False

    def append(self, frame: pl.DataFrame) -> None:
        self.ctx._check_cancelled()
        if not frame.height:
            return
        if not self.spilling:
            owned, size = compact_polars_with_size(frame)
            frame = cast(pl.DataFrame, owned)
            size += sys.getsizeof(frame) + 8  # retained wrapper and chunk-list slot
            if self.ctx._resources.retained_budget.reserve(size):
                self.frames.append(frame)
                self.bytes += size
                return
            self.spilling = True
            # Retained chunks are already one coherent observation. Consolidate
            # them without rereading the original plan.
            if self.frames:
                self.files.append(_sink(self.ctx, pl.concat(self.frames, rechunk=False).lazy()))
                self.frames.clear()
                self.ctx._resources.retained_budget.release(self.bytes)
                self.bytes = 0
        self.files.append(_write_batch(self.ctx, frame))

    def finish(self, schema: Mapping[str, pl.DataType]) -> Observation:
        frame = pl.concat(self.frames, rechunk=False) if self.frames else None
        if frame is None and not self.files:
            frame = pl.DataFrame(schema=schema)
        result = Observation(
            self.ctx,
            frame=frame,
            files=self.files,
            retained_bytes=self.bytes,
        )
        self.frames, self.files, self.bytes = [], [], 0
        return result

    def close(self) -> None:
        self.frames.clear()
        self.ctx._resources.retained_budget.release(self.bytes)
        self.bytes = 0
        for owned in self.files:
            owned.close()
        self.files.clear()
        self.spilling = False


def _observe_plan(ctx: DataContext, plan: pl.LazyFrame) -> Observation:
    # Current Polars offers a genuinely incremental typed batch reader. Keep the
    # exact IPC fallback on older supported versions; never emulate streaming by
    # collecting the complete source or evaluating it repeatedly.
    if not hasattr(plan, "collect_batches"):
        return Observation(ctx, files=[_sink(ctx, plan)])
    builder = _ObservationBuilder(ctx)
    try:
        schema = dict(plan.collect_schema())
        for batch in plan.collect_batches(
            chunk_size=65536, maintain_order=True, engine="streaming"
        ):
            builder.append(batch)
        ctx._check_cancelled()
        return builder.finish(schema)
    finally:
        builder.close()


def _lazy(
    ctx: DataContext, plan: pl.LazyFrame, columns: Iterable[str] | None, budget: int | None
) -> Observation:
    if columns is not None:
        plan = plan.select(selected_columns(columns))
    if budget is not None:
        plan = plan.limit(budget + 1)
    observation = _observe_plan(ctx, plan)
    try:
        if budget is not None:
            rows = observation.reduce(lambda stable: stable.select(pl.len())).item()
            if rows > budget:
                raise SourceError(f"observation has more than {budget:,} rows (row_budget)")
        return observation
    except BaseException:
        observation.close()
        raise


def _native(
    ctx: DataContext, name: str, columns: Iterable[str] | None, source: DataSource | None = None
) -> Observation:
    from qavf.engine.sql_read import consume_native

    builder = _ObservationBuilder(ctx)
    schema: dict[str, pl.DataType] = {}
    selection = (
        "*"
        if columns is None
        else ", ".join(quote_identifier(column) for column in selected_columns(columns))
    )
    query = f"SELECT {selection} FROM {quote_identifier(name)}"

    def factory():
        # A retry starts a new coherent observation; no prior attempt survives.
        builder.close()

        def consume(session):
            nonlocal schema
            schema = session.schema(query)
            for batch in session.batches(query):
                builder.append(batch)

        return consume

    try:
        consume_native(
            ctx,
            factory,
            datasets=[name] if source is None else (),
            sources=None if source is None else {name: source},
        )
        return builder.finish(schema)
    finally:
        builder.close()


def _stored(
    ctx: DataContext, name: str, columns: Iterable[str] | None
) -> Observation:
    """Share the exact admitted cache allocation, including its retained charge."""
    frame, lease = ctx._frames.observe(name)
    try:
        return Observation(
            ctx,
            frame=frame,
            files=[lease] if isinstance(lease, SpillLease) else [],
            columns=columns,
            retained_lease=lease if isinstance(lease, RetainedLease) else None,
        )
    except BaseException:
        if isinstance(lease, RetainedLease):
            lease.release()
        elif lease is not None:
            lease.close()
        raise


@contextlib.contextmanager
def observe(
    ctx: DataContext,
    name: str,
    columns: Iterable[str] | None = None,
) -> Iterator[Observation]:
    """Snapshot selected columns once, enforcing complete-source result policy."""
    ctx._check_cancelled()
    ctx._require_available(name)
    observation: Observation | None = None
    attempts: list[Observation] = []
    try:
        if name in ctx._frames:
            observation = _stored(ctx, name, columns)
        elif name in ctx._budget_observations:
            observation = ctx._budget_observations[name].borrow(columns)
        elif ctx._is_lazy(name):
            source = ctx._catalog.sources.get(name)
            spec = ctx.specs.get(name)
            budget = (
                source.params.row_budget
                if source is not None
                else (spec.effective_load_policy().row_budget if spec is not None else None)
            )

            def attempt():
                for previous in attempts:
                    previous.close()
                attempts.clear()
                plan = ctx._prepare_lazy(name, ctx._lazy_snapshot(name))
                current = _lazy(ctx, plan, columns, budget)
                attempts.append(current)
                return current

            observation = ctx._source_operation([name], attempt)
            ctx._remember_lazy_dataset(name)
        elif ctx.execution_kind(name) == "native_sql":
            observation = _native(ctx, name, columns)
        else:
            ctx.get(name)
            observation = _stored(ctx, name, columns)
        yield observation
    finally:
        if observation is not None:
            observation.close()
        for previous in attempts:
            previous.close()


@contextlib.contextmanager
def observe_source(
    ctx: DataContext,
    source: DataSource,
    columns: Iterable[str] | None = None,
) -> Iterator[Observation]:
    """The same coherent reader for a bound ad-hoc source."""
    attempts: list[Observation] = []
    observation: Observation | None = None

    def attempt():
        for previous in attempts:
            previous.close()
        attempts.clear()
        source.prepare_identity(hash_contents=True)
        plan = source.scan_polars() if source.snapshot_connection() is None else None
        if plan is not None:
            current = _lazy(ctx, plan, columns, source.params.row_budget)
        elif source.snapshot_connection() is None and source.scan() is not None:
            alias = "__qavf_observation"
            while alias.lower() in {name.lower() for name in ctx.declared()}:
                alias += "_"
            current = _native(ctx, alias, columns, source)
        else:
            frame = ctx._load_with_retries("<observation>", source)
            if columns is not None:
                frame = frame.select(selected_columns(columns))
            builder = _ObservationBuilder(ctx)
            try:
                builder.append(frame)
                current = builder.finish(frame.schema)
            finally:
                builder.close()
        attempts.append(current)
        return current

    try:
        observation = ctx._operations.run([source], attempt)
        yield observation
    finally:
        if observation is not None:
            observation.close()
        for previous in attempts:
            previous.close()


class Observations(Mapping[str, Observation]):
    def __init__(self, ctx: DataContext, values: Mapping[str, Observation]) -> None:
        self._ctx, self._values = ctx, dict(values)

    def __getitem__(self, key: str) -> Observation:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def collect(self, plan: pl.LazyFrame) -> pl.DataFrame:
        for observation in self._values.values():
            observation._check()
        frame = plan.collect(engine="streaming")
        frame = cast(pl.DataFrame, compact_polars(frame))
        self._ctx._check_materialization("combined observations", frame)
        self._ctx._check_cancelled()
        return frame


@contextlib.contextmanager
def projected_observations(
    ctx: DataContext,
    selections: Mapping[str, Iterable[str] | None],
) -> Iterator[Observations]:
    with contextlib.ExitStack() as stack:
        values = {
            name: stack.enter_context(observe(ctx, name, columns))
            for name, columns in selections.items()
        }
        yield Observations(ctx, values)
