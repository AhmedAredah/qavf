"""Spill-owned compact evidence payloads for decisions made after a whole family.

Statistical families retain numeric state in memory; optional presentation data
has a separate, budgeted lifetime and is decoded only after final admission.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any, Self

from qavf.audit_codec import decode_audit, encode_audit
from qavf.engine.cache_ownership import retained_size
from qavf.engine.context import DataContext
from qavf.engine.resources import ResourceLimitError
from qavf.engine.sqlite_store import Store


class DeferredEvidence:
    """Own typed optional payloads without retaining a rich record per candidate."""

    def __init__(self, ctx: DataContext) -> None:
        self._ctx = ctx
        self._scope = ExitStack()
        self._store: Store | None = None
        self._unavailable = False

    def add(self, ordinal: int, payload: tuple[Any, ...]) -> None:
        if self._unavailable or not self._ctx.evidence_limit or not self._ctx.evidence_bytes:
            return
        if retained_size(payload) > self._ctx.evidence_bytes:
            return
        encoded = json.dumps(encode_audit(payload), ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > self._ctx.evidence_bytes:
            return
        try:
            if self._store is None:
                self._store = self._scope.enter_context(self._ctx.work_store("deferred evidence"))
                self._store.execute(
                    "CREATE TABLE evidence(ordinal INTEGER PRIMARY KEY,payload TEXT)"
                )
            self._store.execute("INSERT INTO evidence VALUES(?,?)", (ordinal, encoded))
        except ResourceLimitError:
            # Optional detail cannot turn an otherwise assessed family into an
            # error. A full SQLite page may invalidate the statement's work;
            # retain its owner for cleanup, but never reuse that partial store.
            self._unavailable = True

    def get(self, ordinal: int) -> tuple[Any, ...] | None:
        if self._unavailable or self._store is None:
            return None
        row = self._store.execute(
            "SELECT payload FROM evidence WHERE ordinal=?", (ordinal,)
        ).fetchone()
        return tuple(decode_audit(json.loads(row[0]))) if row is not None else None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._scope.__exit__(*exc)
