"""Reuse validated projections across contracts without extending source lifetimes."""

from __future__ import annotations

import contextlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qavf.engine.observation import Observation
from qavf.engine.projection import selected_columns
from qavf.sources.base import SourceError

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


class ObservationView(Observation):
    """A non-owning projection; closing it never closes the shared observation."""

    def __init__(self, parent: Observation, columns: Iterable[str] | None) -> None:
        schema = parent.schema()
        selected = list(schema) if columns is None else selected_columns(columns)
        missing = [column for column in selected if column not in schema]
        if missing:
            raise ValueError(f"observation does not contain columns {missing!r}")
        self._parent = parent
        super().__init__(
            parent._ctx, frame=parent._frame, files=parent._files, columns=selected
        )

    def _check(self) -> None:
        super()._check()
        self._parent._check()

    def close(self) -> None:
        self._closed = True
        self._frame = None
        self._files = ()


@dataclass(frozen=True)
class _Failure:
    """Cache messages/categories, never traceback frames or exception payloads."""

    kind: type[Exception]
    message: str
    retryable: bool

    @classmethod
    def capture(cls, error: Exception) -> _Failure:
        return cls(type(error), str(error), bool(getattr(error, "retryable", False)))

    def exception(self) -> Exception:
        try:
            error = self.kind(self.message)
        except Exception:
            error = SourceError(f"{self.kind.__name__}: {self.message}")
        if isinstance(error, SourceError):
            error.retryable = self.retryable
        return error


class ContractObservations(Mapping[str, ObservationView]):
    """Resolve each input only if its contract actually reads that input."""

    def __init__(
        self, plan: ObservationPlan, selections: Mapping[str, tuple[str, ...] | None]
    ) -> None:
        self._plan, self._selections = plan, selections
        self._views: dict[str, ObservationView] = {}
        self._closed = False

    def __getitem__(self, name: str) -> ObservationView:
        if self._closed:
            raise RuntimeError("observation contract has ended")
        columns = self._selections[name]
        if name not in self._views:
            self._views[name] = ObservationView(self._plan._observation(name), columns)
        return self._views[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._selections)

    def __len__(self) -> int:
        return len(self._selections)

    def close(self) -> None:
        self._closed = True
        for view in self._views.values():
            view.close()
        self._views.clear()


class ObservationPlan:
    """Union valid contract columns, read once, release at each input's last use.

    Callers validate contracts before planning so an invalid projection cannot
    poison other contracts' shared input. Contract indices may be consumed once
    each; unused inputs remain unopened and plan exit closes everything retained.
    This owner is for one sequential rule invocation, not concurrent consumers.
    """

    def __init__(
        self,
        ctx: DataContext,
        contracts: Sequence[Mapping[str, Iterable[str] | None]],
    ) -> None:
        self._ctx = ctx
        self._contracts = [
            {
                name: None if columns is None else tuple(selected_columns(columns))
                for name, columns in contract.items()
            }
            for contract in contracts
        ]
        self._union: dict[str, tuple[str, ...] | None] = {}
        self._remaining: Counter[str] = Counter()
        for contract in self._contracts:
            self._remaining.update(contract.keys())
            for name, columns in contract.items():
                previous = self._union.get(name, ())
                self._union[name] = (
                    None
                    if previous is None or columns is None
                    else tuple(dict.fromkeys((*previous, *columns)))
                )
        self._owners: dict[str, contextlib.ExitStack] = {}
        self._observed: dict[str, Observation] = {}
        self._failures: dict[str, _Failure] = {}
        self._started: set[int] = set()
        self._stack = contextlib.ExitStack()
        self._closed = False

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("observation plan has ended")
        self._ctx._check_cancelled()

    def _observation(self, name: str) -> Observation:
        self._check()
        if name in self._failures:
            raise self._failures[name].exception() from None
        if name not in self._observed:
            owner = contextlib.ExitStack()
            try:
                observed = owner.enter_context(self._ctx.observe(name, self._union[name]))
            except Exception as error:
                owner.close()
                self._failures[name] = _Failure.capture(error)
            except BaseException:
                owner.close()
                raise
            else:
                self._stack.callback(owner.close)
                self._owners[name], self._observed[name] = owner, observed
            if name in self._failures:
                raise self._failures[name].exception() from None
        return self._observed[name]

    @contextlib.contextmanager
    def contract(self, index: int) -> Iterator[ContractObservations]:
        self._check()
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(
            self._contracts
        ):
            raise IndexError("invalid observation contract index")
        if index in self._started:
            raise RuntimeError("observation contract was already consumed")
        self._started.add(index)
        selections = self._contracts[index]
        observations = ContractObservations(self, selections)
        self._stack.callback(observations.close)
        try:
            yield observations
        finally:
            observations.close()
            for name in selections:
                self._remaining[name] -= 1
                if self._remaining[name] == 0:
                    self._observed.pop(name, None)
                    self._failures.pop(name, None)
                    owner = self._owners.pop(name, None)
                    if owner is not None:
                        owner.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stack.close()
        finally:
            self._observed.clear()
            self._owners.clear()
            self._failures.clear()

    def __enter__(self) -> ObservationPlan:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
