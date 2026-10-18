"""Canonical exact keys for bounded indexed grouping and alignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import BytesIO
from typing import Any

from qavf.audit_codec import iter_audit_json


def canonical_key(values: Sequence[Any], *, max_bytes: int | None = None) -> bytes:
    """Encode normalized key values without hashes or collision assumptions.

    Apply ``typed_values.encode_temporal`` before extracting Python rows. Numeric
    dtype alignment is the caller's policy. Signed float zero compares equal;
    NaNs use the audit codec's canonical tag, useful for grouping (comparisons
    reject NaN identity keys before calling this helper).
    """

    def normalized(value: Any) -> Any:
        if isinstance(value, float) and value == 0:
            return 0.0
        if isinstance(value, Mapping):
            return {key: normalized(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(child) for child in value]
        return value

    result = BytesIO()
    for chunk in iter_audit_json(normalized(values), sort_keys=True):
        if max_bytes is not None and result.tell() + len(chunk) > max_bytes:
            raise ValueError("encoded key exceeds the bounded work allowance")
        result.write(chunk)
    return result.getvalue()
