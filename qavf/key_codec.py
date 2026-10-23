"""Canonical exact keys for bounded indexed grouping and alignment."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from typing import Any

from qavf.audit_codec import AuditSizeError, iter_audit_json


def canonical_key(values: Sequence[Any], *, max_bytes: int | None = None) -> bytes:
    """Encode normalized key values without hashes or collision assumptions.

    Apply ``typed_values.encode_temporal`` before extracting Python rows. Numeric
    dtype alignment is the caller's policy. Signed float zero compares equal;
    NaNs use the audit codec's canonical tag, useful for grouping (comparisons
    reject NaN identity keys before calling this helper).
    """

    result = BytesIO()
    try:
        for chunk in iter_audit_json(
            values, sort_keys=True, max_bytes=max_bytes, normalize_float_zero=True
        ):
            result.write(chunk)
    except AuditSizeError:
        raise ValueError("encoded key exceeds the bounded work allowance") from None
    return result.getvalue()
