"""Transportable inputs without loading data or borrowing live process state."""

from __future__ import annotations

import os
from collections.abc import Mapping

from qavf.inputs.descriptors import File, Query
from qavf.settings import QAVFSettings


def isolated_inputs(
    inputs: Mapping[str, object], settings: QAVFSettings
) -> dict[str, object]:
    """Validate every input before any conversion, callback, or reader is invoked."""
    queries: dict[str, object] = {}
    for name, value in inputs.items():
        if isinstance(value, File | str | os.PathLike):
            continue  # File definitions are already part of the settings.
        if isinstance(value, Query) and (
            value.connection is None or isinstance(value.connection, str)
        ):
            if isinstance(value.connection, str) and value.connection not in settings.connections:
                raise ValueError("Query refers to an unknown configured connection")
            queries[name] = value
            continue
        raise ValueError(
            f"isolated execution cannot transport input {name!r} ({type(value).__name__}); "
            "use a path, File, Query with a configured connection, or configured source plugin. "
            "Live frames, lazy plans, Loader callbacks, source instances and borrowed handles "
            "must be saved or described explicitly before isolated execution."
        )
    return queries
