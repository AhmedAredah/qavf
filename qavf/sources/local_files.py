"""Canonical local input discovery shared by readers and output admission."""

from __future__ import annotations

import glob
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def discover_local_files(location: str | Path) -> tuple[Path, ...] | None:
    """Resolve literal files before patterns; return None for remote locations.

    Results are deterministic, include hidden files, and exclude directories.
    The caller owns the resulting immutable manifest and its missing-file policy.
    """
    text = str(location)
    if "://" in text:
        return None
    path = Path(text)
    if path.is_file():
        return (path,)
    if any(mark in text for mark in "*?["):
        return tuple(
            sorted(
                {
                    Path(p)
                    for p in glob.iglob(text, recursive=True, include_hidden=True)
                    if Path(p).is_file()
                }
            )
        )
    return ()


@dataclass(frozen=True, slots=True)
class LocalManifest:
    """A frozen discovery result and its initial metadata identity."""

    paths: tuple[Path, ...]
    stats: tuple[tuple[int, int, int, int, int], ...]

    @classmethod
    def capture(cls, paths, stat: Callable[[Path], tuple[int, int, int, int, int]]):
        return cls(paths, tuple(stat(path) for path in paths))

    def verify(self, stat: Callable[[Path], tuple[int, int, int, int, int]]) -> None:
        from qavf.redaction import redact_text
        from qavf.sources.base import SourceIntegrityError

        for path, original in zip(self.paths, self.stats, strict=True):
            try:
                current = stat(path)
            except OSError:
                raise SourceIntegrityError(
                    f"input file disappeared during validation: {redact_text(str(path))}"
                ) from None
            if current != original:
                raise SourceIntegrityError(
                    f"input file changed during validation: {redact_text(str(path))}"
                )
