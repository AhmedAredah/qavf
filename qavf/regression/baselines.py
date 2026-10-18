"""Accepted baselines with an identity and acceptance record (a manifest).

A baseline match proves stability, not correctness, so it matters which baseline was
matched and why it was accepted. :func:`accept_baseline` writes the data together
with ``<file>.manifest.json``: its keys, column types, row count, a SHA-256 of the
bytes written, when and why it was accepted, and the QAVF version. ``REG-001`` checks
the manifest before comparing: a baseline whose bytes or keys no longer match it is
not used.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from pydantic import BaseModel, Field, ValidationInfo, field_serializer, model_validator

from qavf._version import __version__
from qavf.audit_codec import decode_audit, encode_audit
from qavf.inputs.normalize import Configuration
from qavf.inputs.resources import ResourceLimits
from qavf.io import atomic_write
from qavf.regression._baseline_writer import stage_baseline
from qavf.regression._columns import column_names

MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_FORMAT_VERSION = 2


class BaselineManifest(BaseModel):
    """What was accepted as a baseline, when and why."""

    format_version: int = MANIFEST_FORMAT_VERSION
    keys: list[str]
    columns: dict[str, str]
    """Column types of the accepted input (text formats may infer differently on read)."""
    rows: int
    sha256: str
    """Digest of the baseline file's bytes."""
    accepted_at: datetime
    reason: str | None = None
    """Why this output was accepted as correct (review notes, ticket, run id)."""
    source: str | None = None
    """Where the accepted data came from."""
    qavf_version: str
    comparison: dict[str, Any] = Field(default_factory=dict)
    """Comparison policy intended for it (tolerances, per-column settings), if given."""

    @field_serializer("comparison", when_used="json")
    def _encode_comparison(self, value: Any) -> Any:
        return encode_audit(value)

    @model_validator(mode="before")
    @classmethod
    def _decode_comparison(cls, data: Any, info: ValidationInfo) -> Any:
        if info.mode == "json" and isinstance(data, dict) and "comparison" in data:
            data = {**data, "comparison": decode_audit(data["comparison"])}
        return data


def manifest_path(path: str | Path) -> Path:
    """The manifest file that belongs to the baseline at ``path``."""
    target = Path(path)
    return target.with_name(target.name + MANIFEST_SUFFIX)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_baseline_keys(frame: pl.DataFrame, keys: Sequence[str]) -> None:
    """Keys must exist, be non-null (and not NaN) and identify rows uniquely.

    Raises:
        ValueError: describing the first violation.
    """
    keys = column_names(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("baseline key columns must be unique")
    if not keys:
        raise ValueError("a baseline needs key columns that identify its rows")
    absent = [key for key in keys if key not in frame.columns]
    if absent:
        raise ValueError(f"baseline lacks key column(s) {', '.join(absent)}")
    for key in keys:
        column = frame[key]
        missing = column.null_count() + (
            int(column.is_nan().sum()) if column.dtype.is_float() else 0
        )
        if missing:
            raise ValueError(f"baseline key column {key!r} has {missing} missing value(s)")
    duplicates = int(frame.select(keys).is_duplicated().sum())
    if duplicates:
        raise ValueError(
            f"baseline has {duplicates} row(s) with a duplicated key ({', '.join(keys)})"
        )


def accept_baseline(
    frame: object,
    path: str | Path,
    keys: Sequence[str],
    *,
    reason: str | None = None,
    source: str | None = None,
    replace: bool = False,
    comparison: dict[str, Any] | None = None,
    resources: ResourceLimits | None = None,
    config: Configuration = None,
) -> BaselineManifest:
    """Accept one supported input at ``path`` after bounded exact round-trip validation.

    Rows are sorted by ``keys`` for stable diffs in review.

    Raises:
        ValueError: if the keys are missing, null or not unique (nothing is written).
        FileExistsError: if a baseline exists at ``path`` and ``replace`` is false.
    """
    target = Path(path)
    keys = column_names(keys)
    if not keys:
        raise ValueError("a baseline needs key columns that identify its rows")
    if len(set(keys)) != len(keys):
        raise ValueError("baseline key columns must be unique")
    with _baseline_lock(target, create=True):
        _recover(target)
        if target.exists() and not replace:
            raise FileExistsError(
                f"{target} already exists; replacing an accepted baseline must be explicit "
                "(replace=True or --force)"
            )
        transaction = _transaction_path(target)
        transaction.mkdir()
        try:
            with ExitStack() as staging:
                payload = transaction / "payload"
                payload.mkdir()
                staged = payload / target.name
                accepted = staging.enter_context(
                    stage_baseline(frame, staged, keys, resources=resources, config=config)
                )
                manifest = BaselineManifest(
                    keys=list(keys),
                    columns=accepted.columns,
                    rows=accepted.rows,
                    sha256=_sha256(staged),
                    accepted_at=datetime.now(UTC),
                    reason=reason,
                    source=source,
                    qavf_version=__version__,
                    comparison=dict(comparison or {}),
                )
                staged_manifest = transaction / "manifest.after"
                atomic_write(staged_manifest, manifest.model_dump_json(indent=2))
                locations = (target, manifest_path(target))
                old = [location.exists() for location in locations]
                for location, existed, backup in zip(
                    locations, old, ("data.before", "manifest.before"), strict=True
                ):
                    if existed:
                        try:
                            os.link(location, transaction / backup)
                        except OSError:
                            shutil.copy2(location, transaction / backup)
                # The durable journal precedes both visible replacements. Readers hold
                # this same lock and recover an interrupted transaction first.
                atomic_write(
                    transaction / "state.json", json.dumps({"old": old, "committed": False})
                )
                os.replace(staged, target)
                os.replace(staged_manifest, manifest_path(target))
                atomic_write(
                    transaction / "state.json", json.dumps({"old": old, "committed": True})
                )
        except BaseException:
            _recover(target)
            raise
        shutil.rmtree(transaction)
        return manifest


def _transaction_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.transaction")


@contextmanager
def _baseline_lock(path: Path, *, create: bool = False) -> Iterator[None]:
    """Cross-process advisory lock for acceptance and verified reads.

    Only acceptance creates the baseline directory. A reader of a baseline whose
    directory is absent has nothing to read or recover, so it never creates one.
    """
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir():
        yield
        return
    lock = path.with_name(f".{path.name}.lock")
    with lock.open("a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            # Windows may lock bytes beyond end of file, so never read the byte a
            # holder has locked. LK_LOCK gives up after ten seconds; wait like flock.
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except PermissionError:
                    time.sleep(0.01)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _recover(path: Path) -> None:
    transaction = _transaction_path(path)
    if not transaction.exists():
        return
    journal = transaction / "state.json"
    if journal.exists():
        state = json.loads(journal.read_text(encoding="utf-8"))
        if not state["committed"]:
            for destination, existed, backup in zip(
                (path, manifest_path(path)),
                state["old"],
                ("data.before", "manifest.before"),
                strict=True,
            ):
                if existed:
                    # Copy, rather than consuming the backup, makes recovery itself
                    # restartable if interrupted between its two replacements.
                    temporary = transaction / f"restore-{uuid4().hex}"
                    shutil.copy2(transaction / backup, temporary)
                    os.replace(temporary, destination)
                else:
                    destination.unlink(missing_ok=True)
    shutil.rmtree(transaction)


def _read_manifest(path: str | Path) -> BaselineManifest | None:
    location = manifest_path(path)
    if not location.exists():
        return None
    try:
        manifest = BaselineManifest.model_validate_json(location.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"unreadable baseline manifest {location}: {exc}") from None
    if manifest.format_version != MANIFEST_FORMAT_VERSION:
        raise ValueError(f"unsupported baseline manifest version {manifest.format_version}")
    return manifest


def read_manifest(path: str | Path) -> BaselineManifest | None:
    """The manifest of the baseline at ``path``, or ``None`` if it has none.

    Raises:
        ValueError: if the manifest exists but cannot be read.
    """
    target = Path(path)
    with _baseline_lock(target):
        _recover(target)
        return _read_manifest(target)


def verify_baseline(path: str | Path, keys: Sequence[str] | None = None) -> list[str]:
    """Problems with the baseline at ``path`` against its manifest (empty when it matches).

    Checks that the file's bytes are the ones accepted and, when ``keys`` is given,
    that it is compared on the keys it was accepted with.
    """
    target = Path(path)
    with _baseline_lock(target):
        _recover(target)
        manifest = _read_manifest(target)
        if manifest is None:
            return ["it has no manifest"]
        return _identity_problems(target, manifest, keys)


def _identity_problems(
    path: Path, manifest: BaselineManifest, keys: Sequence[str] | None, digest: str | None = None
) -> list[str]:
    problems = []
    if (digest or _sha256(path)) != manifest.sha256:
        problems.append(
            "its contents changed since it was accepted (SHA-256 differs from the manifest)"
        )
    keys = column_names(keys) if keys is not None else None
    if keys is not None and keys != manifest.keys:
        problems.append(
            f"it was accepted with keys {manifest.keys}, but is compared on {list(keys)}"
        )
    return problems


def read_verified_baseline[T](
    path: str | Path,
    keys: Sequence[str],
    required: bool,
    loader: Callable[[], T],
) -> tuple[T, BaselineManifest | None, dict[str, Any]]:
    """Load one stable baseline snapshot while verifying the actual bytes read.

    The callback must complete all reads before returning. The lock covers journal
    recovery, acceptance verification and the entire callback. A
    second streamed digest detects edits by writers that do not use this lock.
    """
    target = Path(path)
    with _baseline_lock(target):
        _recover(target)
        manifest = _read_manifest(target)
        if required and manifest is None:
            raise ValueError("it has no manifest (accept it with 'qavf baseline')")
        before = _sha256(target)
        if manifest is not None:
            problems = _identity_problems(target, manifest, keys, before)
            if problems:
                raise ValueError("; ".join(problems))
        frame = loader()
        if before != _sha256(target):
            raise ValueError("baseline contents changed while the snapshot was being loaded")
        identity: dict[str, Any] = {
            "identity": "content",
            "path": str(target),
            "sha256": before,
            "size_bytes": target.stat().st_size,
        }
        if manifest is not None:
            identity["manifest"] = manifest.model_dump(mode="json")
        return frame, manifest, identity
