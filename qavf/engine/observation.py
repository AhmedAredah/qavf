"""Coherent projected observations with exact typed batches and owned spill files.

A source is evaluated once before its observation is published. The resulting
frame or IPC files stay alive for the scope; derived plans never revisit that
source. IPC staging bounds owned disk bytes and supports Polars' minimum version
without relying on its newer batch-collection APIs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import polars as pl

from qavf.engine.cache_ownership import compact_polars
from qavf.engine.owned_ipc import _SCAN_OPTIONS
from qavf.engine.projection import selected_columns
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
    ) -> None:
        self._ctx = ctx
        self._frame = frame
        self._files = tuple(files)
        self._columns = None if columns is None else selected_columns(columns)
        self._closed = False

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
        owned = _sink(self._ctx, build(self.scan()))
        with Observation(self._ctx, files=[owned]) as observation:
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
    ctx._check_materialization("observation batch", frame)
    owned = _file(ctx)
    try:
        with owned.writer() as writer:
            frame.write_ipc(writer, compression="uncompressed")
        return owned
    except BaseException:
        owned.close()
        raise


def _lazy(
    ctx: DataContext, plan: pl.LazyFrame, columns: Iterable[str] | None, budget: int | None
) -> Observation:
    if columns is not None:
        plan = plan.select(selected_columns(columns))
    if budget is not None:
        plan = plan.limit(budget + 1)
    owned = _sink(ctx, plan)
    observation = Observation(ctx, files=[owned])
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

    staged: list[SpillFile] = []
    empty: pl.DataFrame | None = None
    selection = (
        "*"
        if columns is None
        else ", ".join(quote_identifier(column) for column in selected_columns(columns))
    )
    query = f"SELECT {selection} FROM {quote_identifier(name)}"

    def factory():
        # A retry starts a new coherent observation; no prior attempt survives.
        for owned in staged:
            owned.close()
        staged.clear()

        def consume(session):
            nonlocal empty
            schema = session.schema(query)
            empty = pl.DataFrame(schema=schema)
            for batch in session.batches(query):
                staged.append(_write_batch(ctx, batch))

        return consume

    try:
        consume_native(
            ctx,
            factory,
            datasets=[name] if source is None else (),
            sources=None if source is None else {name: source},
        )
        result = Observation(ctx, files=staged, frame=empty if not staged else None)
        staged = []
        return result
    finally:
        for owned in staged:
            owned.close()


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
            frame, lease = ctx._frames.observe(name)
            observation = Observation(
                ctx, frame=frame, files=[] if lease is None else [lease], columns=columns
            )
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
            frame = ctx.get(name)
            observation = Observation(
                ctx, frame=frame if columns is None else frame.select(selected_columns(columns))
            )
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
            current = Observation(
                ctx, frame=frame if columns is None else frame.select(selected_columns(columns))
            )
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
