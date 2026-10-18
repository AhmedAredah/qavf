"""Linux live-descendant RSS observations without adding a runtime dependency.

Read procfs-visible IDs rather than os.getpid(): /proc can belong to a parent PID
namespace. Sums include shared pages in every process (RSS, not PSS), and samples
can miss short-lived descendants. This is observation, never quota enforcement.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TreeSample:
    rss_bytes: int | None
    processes: int | None
    complete: bool


def _stat(text: str) -> tuple[int, int, int, int]:
    before, _, after = text.rpartition(")")
    fields = after.split()
    return int(before.split("(", 1)[0]), int(fields[1]), int(fields[19]), int(fields[21])


def _parent_index(proc: Path) -> tuple[dict[int, list[int]], bool]:
    """Fallback for kernels that do not expose task/children (CONFIG_CHECKPOINT_RESTORE)."""
    children: dict[int, list[int]] = {}
    complete = True
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            pid, parent, _, _ = _stat((entry / "stat").read_text())
            children.setdefault(parent, []).append(pid)
        except (OSError, ValueError, IndexError):
            complete = False
    return children, complete


def sample_tree(proc: Path = Path("/proc")) -> TreeSample:
    """Count the current process and children reachable through every live thread."""
    if sys.platform != "linux":
        return TreeSample(None, None, False)
    try:
        pid, _, started, _pages = _stat((proc / "self/stat").read_text())
        page_bytes = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return TreeSample(None, None, False)
    pending: list[tuple[int, int | None]] = [(pid, None)]
    seen: set[tuple[int, int]] = set()
    total = 0
    complete = True
    parent_index: dict[int, list[int]] | None = None
    while pending:
        current, parent = pending.pop()
        try:
            actual, ppid, birth, resident = _stat((proc / str(current) / "stat").read_text())
            if parent is not None and ppid != parent:
                complete = False
                continue
            if current == pid and (actual, birth) != (pid, started):
                return TreeSample(None, None, False)
            key = actual, birth
            if key in seen:
                continue
            seen.add(key)
            total += max(0, resident) * page_bytes
            try:
                if parent_index is None:
                    for task in (proc / str(current) / "task").iterdir():
                        pending.extend(
                            (int(child), actual)
                            for child in (task / "children").read_text().split()
                        )
            except FileNotFoundError:
                # Some kernels omit children entirely. PPID uses the same
                # procfs PID namespace as stat, including containers.
                parent_index, index_complete = _parent_index(proc)
                complete = complete and index_complete
            if parent_index is not None:
                pending.extend((child, actual) for child in parent_index.get(actual, []))
        except (OSError, ValueError, IndexError):
            # Exits during enumeration make a partial, explicitly disclosed sample.
            complete = False
    return TreeSample(total if seen else None, len(seen) if seen else None, complete)
