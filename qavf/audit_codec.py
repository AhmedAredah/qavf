"""Canonical, collision-safe values shared by audit snapshots, JSON and digests.

Mapping keys have one documented representation: enums use their value, and JSON
scalar keys use ``str(key)``. Collisions are rejected before freezing/serialization.
Rich immutable scalars use explicit tags, so a Decimal/date/byte string cannot
silently become an ordinary string or null. Reserved user mappings are escaped.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn
from zoneinfo import ZoneInfo

import numpy as np

from qavf._audit_temporal import normalize_temporal_scalar

FLOAT_TAG = "$float"
_VALUE_TAG = "$qavf_value"
_MAPPING_TAG = "$qavf_mapping"
_RESERVED = {FLOAT_TAG, _VALUE_TAG, _MAPPING_TAG}
_FLOATS = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
_SCALARS = (str, bool, int, float, Decimal, date, time, timedelta, Path, bytes)


class AuditSizeError(ValueError):
    """An audit payload cannot fit its requested encoded-byte allowance."""


def mapping_key(value: Any) -> str:
    """Canonical string spelling for supported mapping keys, before collision checks."""
    if isinstance(value, Enum):
        value = value.value
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("evidence mapping keys must be strings or JSON scalars")
    return str(value)


def _immutable(*args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError("QA report snapshots are immutable")


class FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = (
        __iadd__
    ) = __imul__ = _immutable

    def __deepcopy__(self, memo: dict) -> Any:
        return self


class FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo: dict) -> Any:
        return self


def _normalize(
    value: Any, *, max_depth: int, frozen: bool, preserved_types: tuple[type, ...] = ()
) -> Any:
    active: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        if depth > max_depth:
            raise ValueError(f"evidence nesting exceeds {max_depth} levels")
        if preserved_types and isinstance(item, preserved_types):
            return item
        if isinstance(item, Enum):
            return visit(item.value, depth)
        if isinstance(item, str):
            # Caller text may carry a renderer-specific __html__ trust marker
            # (e.g. Markup). Evidence is data, never trusted template content.
            return str.__str__(item)
        if isinstance(item, datetime | timedelta) and type(item) not in (datetime, timedelta):
            item = normalize_temporal_scalar(item)
        if (
            isinstance(item, datetime | time)
            and item.tzinfo is not None
            and not isinstance(item.tzinfo, timezone | ZoneInfo)
        ):
            # Pydantic and adapters may provide other tzinfo implementations.
            # Snapshot their actual offset/name instead of borrowing mutable
            # timezone state. Named ZoneInfo rules remain explicitly tagged.
            offset = item.utcoffset()
            if offset is None:
                raise ValueError("audit temporal value has no determinate timezone offset")
            item = item.replace(tzinfo=timezone(offset, item.tzname() or str(offset)))
        if item is None or isinstance(item, _SCALARS):
            return item
        if isinstance(item, np.datetime64 | np.timedelta64):
            return item
        if isinstance(item, np.generic):
            scalar = item.item()
            if isinstance(scalar, np.generic):
                raise ValueError(f"unsupported evidence NumPy dtype: {item.dtype}")
            return visit(scalar, depth)
        identity = id(item)
        if identity in active:
            raise ValueError("evidence must not contain reference cycles")
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                result = {}
                names: set[str] = set()
                for key, child in item.items():
                    name = mapping_key(key)
                    if name in names:
                        raise ValueError(f"evidence mapping has colliding key {name!r}")
                    names.add(name)
                    # Typed model dictionaries (e.g. Severity counts) retain
                    # their enum annotation until JSON encoding. Collision
                    # validation still uses the identical canonical spelling.
                    typed_key = (
                        key
                        if frozen and isinstance(key, Enum) and isinstance(key, preserved_types)
                        else name
                    )
                    result[typed_key] = visit(child, depth + 1)
                return FrozenDict(result) if frozen else result
            if isinstance(item, np.ndarray):
                if item.dtype.kind in "Mm":
                    return visit(item[()] if item.ndim == 0 else list(item), depth + 1)
                return visit(item.tolist(), depth + 1)
            if isinstance(item, (list, tuple, set, frozenset)):
                values = [visit(child, depth + 1) for child in item]
                if isinstance(item, (set, frozenset)):
                    values.sort(key=lambda child: _json(_encode(child)))
                if frozen:
                    return tuple(values) if isinstance(item, tuple) else FrozenList(values)
                return values
            raise ValueError(f"unsupported evidence type: {type(item).__name__}")
        finally:
            active.remove(identity)

    return visit(value, 0)


def normalize_evidence(value: Any, *, max_depth: int = 64) -> Any:
    """Detach supported containers; preserve rich scalars and reject opaque/cyclic data.

    This is an evidence representation, not a process-memory limit. Apply raw-size
    admission before converting caller-owned large arrays or containers.
    """
    return _normalize(value, max_depth=max_depth, frozen=False)


def freeze_evidence(value: Any, *, preserved_types: tuple[type, ...] = ()) -> Any:
    """Detach and freeze the same algebra used by the encoder and byte accounting."""
    return _normalize(value, max_depth=64, frozen=True, preserved_types=preserved_types)


def _timezone_tag(value: datetime | time) -> Any:
    zone = value.tzinfo
    if zone is None:
        return None
    if isinstance(zone, ZoneInfo):
        return ["zone", zone.key]
    if not isinstance(zone, timezone):
        raise ValueError("audit temporal values require a fixed offset or ZoneInfo timezone")
    offset = value.utcoffset()
    assert offset is not None
    return ["offset", offset.days, offset.seconds, offset.microseconds, value.tzname()]


def _timezone_value(value: Any) -> tzinfo | None:
    match value:
        case None:
            return None
        case ["zone", str(key)]:
            return ZoneInfo(key)
        case ["offset", int(days), int(seconds), int(microseconds), str(name)]:
            return timezone(timedelta(days=days, seconds=seconds, microseconds=microseconds), name)
        case _:
            raise ValueError("invalid audit timezone")


def _encode(value: Any) -> Any:
    if isinstance(value, np.datetime64 | np.timedelta64):
        return {_VALUE_TAG: ["numpy_temporal", value.dtype.str, int(value.astype("int64"))]}
    if isinstance(value, float) and not math.isfinite(value):
        return {
            FLOAT_TAG: "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        }
    if isinstance(value, Decimal):
        return {_VALUE_TAG: ["decimal", str(value)]}
    if isinstance(value, datetime):
        return {_VALUE_TAG: ["datetime", value.isoformat(), value.fold, _timezone_tag(value)]}
    if isinstance(value, date):
        return {_VALUE_TAG: ["date", value.isoformat()]}
    if isinstance(value, time):
        return {_VALUE_TAG: ["time", value.isoformat(), value.fold, _timezone_tag(value)]}
    if isinstance(value, timedelta):
        return {_VALUE_TAG: ["timedelta", value.days, value.seconds, value.microseconds]}
    if isinstance(value, Path):
        return {_VALUE_TAG: ["path", str(value)]}
    if isinstance(value, bytes):
        return {_VALUE_TAG: ["bytes", base64.b64encode(value).decode("ascii")]}
    if isinstance(value, Mapping):
        result = {key: _encode(child) for key, child in value.items()}
        if len(result) == 1 and next(iter(result)) in _RESERVED:
            return {_MAPPING_TAG: list(result.items())}
        return result
    if isinstance(value, (list, tuple)):
        return [_encode(child) for child in value]
    return value


def encode_audit(value: Any) -> Any:
    """Encode a supported Python payload as collision-safe standard JSON values."""
    return _encode(normalize_evidence(value))


def decode_audit(value: Any) -> Any:
    """Decode a payload emitted by :func:`encode_audit`; never evaluate user code."""
    if isinstance(value, dict):
        if len(value) == 1:
            floating = value.get(FLOAT_TAG)
            if isinstance(floating, str) and floating in _FLOATS:
                return _FLOATS[floating]
            if _MAPPING_TAG in value:
                pairs = value[_MAPPING_TAG]
                if not isinstance(pairs, list) or not all(
                    isinstance(pair, (list, tuple)) and len(pair) == 2 and isinstance(pair[0], str)
                    for pair in pairs
                ):
                    raise ValueError("invalid escaped audit mapping")
                result = {}
                for key, child in pairs:
                    if key in result:
                        raise ValueError(f"duplicate escaped audit key {key!r}")
                    result[key] = decode_audit(child)
                return result
            if _VALUE_TAG in value:
                parts = value[_VALUE_TAG]
                if not isinstance(parts, list) or not parts:
                    raise ValueError("invalid rich audit value")
                match parts:
                    case ["numpy_temporal", str(dtype), int(physical)]:
                        native_type = np.dtype(dtype)
                        if native_type.kind not in "Mm":
                            raise ValueError("invalid NumPy temporal audit dtype")
                        return np.array(physical, dtype=native_type)[()]
                    case ["decimal", str(text)]:
                        return Decimal(text)
                    case ["datetime", str(text), int(fold), zone] if fold in (0, 1):
                        return datetime.fromisoformat(text).replace(
                            fold=fold, tzinfo=_timezone_value(zone)
                        )
                    case ["date", str(text)]:
                        return date.fromisoformat(text)
                    case ["time", str(text), int(fold), zone] if fold in (0, 1):
                        return time.fromisoformat(text).replace(
                            fold=fold, tzinfo=_timezone_value(zone)
                        )
                    case ["timedelta", int(days), int(seconds), int(microseconds)]:
                        return timedelta(days=days, seconds=seconds, microseconds=microseconds)
                    case ["path", str(text)]:
                        return Path(text)
                    case ["bytes", str(text)]:
                        return base64.b64decode(text, validate=True)
                    case _:
                        raise ValueError("unknown or invalid rich audit value")
        return {key: decode_audit(child) for key, child in value.items()}
    if isinstance(value, list):
        return [decode_audit(child) for child in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def iter_json_string(value: str) -> Iterator[bytes]:
    """Encode a string in bounded pieces, including JSON escaping."""
    if len(value) <= 16_384:
        yield json.dumps(value, ensure_ascii=False).encode("utf-8")
        return
    yield b'"'
    for start in range(0, len(value), 16_384):
        escaped = json.dumps(value[start : start + 16_384], ensure_ascii=False)[1:-1]
        yield escaped.encode("utf-8")
    yield b'"'


def iter_audit_json(
    value: Any, *, sort_keys: bool = False, max_bytes: int | None = None,
    normalize_float_zero: bool = False,
) -> Iterator[bytes]:
    """Stream canonical tagged values without constructing a second evidence graph.

    State is bounded by nesting and mapping keys, apart from unordered sets whose
    canonical ordering necessarily requires sorting. Frozen report evidence already
    contains ordered lists instead of sets. Strings and binary values are emitted
    in bounded chunks. A byte ceiling rejects a known oversized binary value
    before expanding it into base64.
    ``normalize_float_zero`` gives indexed grouping equal signed-zero keys
    without constructing a second container graph; audit output preserves signs.
    """
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a nonnegative integer or None")
    active: set[int] = set()
    written = 0

    def visit(item: Any, depth: int) -> Iterator[bytes]:
        if depth > 64:
            raise ValueError("evidence nesting exceeds 64 levels")
        if isinstance(item, Enum):
            yield from visit(item.value, depth)
            return
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError("evidence must not contain reference cycles")
            active.add(identity)
            try:
                names: dict[str, Any] = {}
                for key in item:
                    name = mapping_key(key)
                    if name in names:
                        raise ValueError(f"evidence mapping has colliding key {name!r}")
                    names[name] = key
                if len(names) == 1 and next(iter(names)) in _RESERVED:
                    name = next(iter(names))
                    yield b'{"$qavf_mapping":[['
                    yield from iter_json_string(name)
                    yield b","
                    yield from visit(item[names[name]], depth + 1)
                    yield b"]]}"
                    return
                yield b"{"
                for index, name in enumerate(sorted(names) if sort_keys else names):
                    if index:
                        yield b","
                    yield from iter_json_string(name)
                    yield b":"
                    yield from visit(item[names[name]], depth + 1)
                yield b"}"
            finally:
                active.remove(identity)
            return
        if isinstance(item, (list, tuple, np.ndarray, set, frozenset)):
            identity = id(item)
            if identity in active:
                raise ValueError("evidence must not contain reference cycles")
            active.add(identity)
            try:
                if isinstance(item, np.ndarray) and item.ndim == 0:
                    yield from visit(item[()], depth + 1)
                    return
                values = item
                if isinstance(item, (set, frozenset)):
                    values = sorted(
                        item, key=lambda child: b"".join(iter_audit_json(child, sort_keys=True))
                    )
                yield b"["
                for index, child in enumerate(values):
                    if index:
                        yield b","
                    yield from visit(child, depth + 1)
                yield b"]"
            finally:
                active.remove(identity)
            return
        if item is None:
            yield b"null"
            return
        if isinstance(item, bool):
            yield b"true" if item else b"false"
            return
        if isinstance(item, int):
            yield str(item).encode("ascii")
            return
        if isinstance(item, str):
            yield from iter_json_string(item)
            return
        if isinstance(item, bytes):
            prefix = b'{"$qavf_value":["bytes","'
            suffix = b'"]}'
            size = len(prefix) + 4 * ((len(item) + 2) // 3) + len(suffix)
            if max_bytes is not None and size > max_bytes - written:
                raise AuditSizeError(f"audit payload exceeds {max_bytes} encoded bytes")
            yield prefix
            # A multiple of three prevents padding inside the base64 stream.
            view = memoryview(item)
            for start in range(0, len(view), 12_288):
                yield base64.b64encode(view[start : start + 12_288])
            yield suffix
            return
        if isinstance(item, float) and math.isfinite(item):
            if normalize_float_zero and item == 0:
                item = 0.0
            yield json.dumps(item, allow_nan=False).encode("ascii")
            return
        normalized = normalize_evidence(item)
        encoded = _encode(normalized)
        # Scalar tags are small JSON containers. They are already encoded, and
        # must not be escaped a second time as ordinary reserved user mappings.
        if isinstance(encoded, str):
            yield from iter_json_string(encoded)
        else:
            encoder = json.JSONEncoder(
                sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            for chunk in encoder.iterencode(encoded):
                yield chunk.encode("utf-8")

    for chunk in visit(value, 0):
        written += len(chunk)
        if max_bytes is not None and written > max_bytes:
            raise AuditSizeError(f"audit payload exceeds {max_bytes} encoded bytes")
        yield chunk


def evidence_json_size(value: Any) -> int:
    return sum(len(chunk) for chunk in iter_audit_json(value))


def canonical_digest(value: Any) -> str:
    """SHA-256 over the same canonical tagged values used for audit serialization."""
    digest = hashlib.sha256()
    for chunk in iter_audit_json(value, sort_keys=True):
        digest.update(chunk)
    return digest.hexdigest()
