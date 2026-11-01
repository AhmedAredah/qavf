"""Execution boundaries shared with the optional engine's process supervisor."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping

from qavf.inputs.descriptors import File, Query


class Deadline:
    """Cooperatively interrupt one owned native operation at its deadline."""

    def __init__(self, connection, seconds):
        self.connection = connection
        self.seconds = seconds
        self.expired = False
        self.timer = None

    def __enter__(self):
        if self.seconds is not None:
            self.timer = threading.Timer(self.seconds, self._interrupt)
            self.timer.daemon = True
            self.timer.start()
        return self

    def _interrupt(self):
        self.expired = True
        self.connection.interrupt()

    def __exit__(self, *_):
        if self.timer is not None:
            self.timer.cancel()
            self.timer.join()


def validate_isolated(settings, inputs):
    """Transport descriptors only; never consume a borrowed object to serialize it."""
    from qavf.execution.dispatch import validate_isolated as dispatch

    local = settings.model_copy(deep=True)
    queries = {}
    for name, value in inputs.items():
        spec = local.datasets[name]
        if isinstance(value, File):
            source = value.source_config()
        elif isinstance(value, str | os.PathLike):
            source = dict(spec.source_config() or {})
            source.update(type="file", path=os.fspath(value))
        elif isinstance(value, Mapping):
            source = dict(value)
        elif isinstance(value, Query):
            queries[name] = value
            continue
        else:
            # The shared admission reports a precise ownership/transport error.
            queries[name] = value
            continue
        local.datasets[name] = spec.model_copy(
            update={
                "path": None,
                "format": None,
                "source": source,
            }
        )
    return dispatch(local, queries, backend="duckdb")
