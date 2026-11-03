"""A cancelled run starts no further source attempt and never waits out a retry delay (EP-02)."""

from __future__ import annotations

import threading
import time

import pytest

from qavf import CancellationToken, Loader, validate
from qavf.sources.base import SourceError

DELAY_S = 30.0
"""A retry delay far longer than any run here: waiting it out would fail the test."""

CONFIG = {
    "datasets": {
        "a": {
            "schema": {"columns": {"x": {"dtype": "int"}}},
            "load_policy": {"retries": 3, "retry_delay_s": DELAY_S},
        }
    }
}


class FlakyLoader:
    """Fails transiently on every call; cancels the run on its first call."""

    def __init__(self, token: CancellationToken, *, later: bool) -> None:
        self.token = token
        self.later = later
        """Cancel shortly after failing (during the backoff) rather than at once."""
        self.calls: list[bool] = []

    def __call__(self) -> object:
        self.calls.append(self.token.cancelled)
        if len(self.calls) == 1:
            if self.later:
                threading.Timer(0.2, self.token.cancel).start()
            else:
                self.token.cancel()
        raise SourceError("transient", retryable=True)


@pytest.mark.parametrize("later", [False, True], ids=["in_loader", "during_backoff"])
@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_cancellation_stops_source_retries(backend: str, later: bool) -> None:
    token = CancellationToken()
    loader = FlakyLoader(token, later=later)
    started = time.monotonic()
    report = validate(
        {"a": Loader(loader, replayable=True, retry_safe=True)},
        config=CONFIG,
        backend=backend,
        cancel=token,
    )
    assert loader.calls == [False]
    assert time.monotonic() - started < DELAY_S / 3
    assert report.metadata.cancelled


def test_the_token_wait_returns_on_cancellation() -> None:
    token = CancellationToken()
    threading.Timer(0.05, token.cancel).start()
    started = time.monotonic()
    assert token.wait(DELAY_S)
    assert time.monotonic() - started < DELAY_S / 3
    assert not CancellationToken().wait(0)
