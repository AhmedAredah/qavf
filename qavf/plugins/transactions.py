"""Atomic, context-local publication across QAVF's registry families.

Imports run without the publication lock. Each transaction sees its own staged
mappings; other threads see only committed state. Commit validates every change
before publishing any mapping, preserving unrelated concurrent registrations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

LOCK = RLock()
_INSTALLED_VISIBLE: ContextVar[bool] = ContextVar("qavf_installed_visible", default=False)


@dataclass
class _Session:
    parent: _Session | None
    installed: bool
    maps: dict[tuple[object, str], tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=dict
    )

    def mapping(self, owner: object, attribute: str) -> dict[str, Any]:
        key = owner, attribute
        if key not in self.maps:
            original = (
                self.parent.mapping(owner, attribute)
                if self.parent is not None
                else getattr(owner, attribute)
            )
            before = dict(original)
            self.maps[key] = before, dict(before)
        return self.maps[key][1]


_CURRENT: ContextVar[_Session | None] = ContextVar("qavf_registration_session", default=None)
_MISSING = object()


def mapping(owner: object, attribute: str) -> dict[str, Any]:
    """The caller's staged mapping, or committed mapping outside a transaction.

    Hold LOCK while reading or mutating the returned mapping. Never retain it
    across a transaction boundary or user/plugin code.
    """
    session = _CURRENT.get()
    return getattr(owner, attribute) if session is None else session.mapping(owner, attribute)


def installed_origin() -> bool:
    session = _CURRENT.get()
    return session is not None and session.installed


def installed_visible() -> bool:
    return _INSTALLED_VISIBLE.get()


@contextmanager
def visibility(enabled: bool) -> Iterator[None]:
    token = _INSTALLED_VISIBLE.set(enabled)
    try:
        yield
    finally:
        _INSTALLED_VISIBLE.reset(token)


def _same(left: Any, right: Any) -> bool:
    return left is right or (
        isinstance(left, type)
        and isinstance(right, type)
        and (left.__module__, left.__qualname__) == (right.__module__, right.__qualname__)
    )


@contextmanager
def transaction(*, installed: bool | None = None) -> Iterator[None]:
    """Stage all participating registries with inherited entry-point provenance."""
    parent = _CURRENT.get()
    session = _Session(parent, installed if installed is not None else installed_origin())
    token = _CURRENT.set(session)
    try:
        yield
        with LOCK:
            prepared = []
            for (owner, attribute), (before, staged) in session.maps.items():
                destination = (
                    parent.mapping(owner, attribute)
                    if parent is not None
                    else getattr(owner, attribute)
                )
                merged = dict(destination)
                for key in before.keys() | staged.keys():
                    old, new = before.get(key, _MISSING), staged.get(key, _MISSING)
                    if old is new:
                        continue
                    current = destination.get(key, _MISSING)
                    if current is not old and not _same(current, new):
                        raise ValueError(f"concurrent registry conflict for {key!r}")
                    if new is _MISSING:
                        merged.pop(key, None)
                    else:
                        merged[key] = new
                prepared.append((owner, attribute, merged))
            for owner, attribute, merged in prepared:
                if parent is None:
                    setattr(owner, attribute, merged)
                else:
                    target = parent.mapping(owner, attribute)
                    target.clear()
                    target.update(merged)
    finally:
        _CURRENT.reset(token)
