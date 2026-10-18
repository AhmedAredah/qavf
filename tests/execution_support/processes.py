"""Independent observation of process liveness, without trusting QAVF internals."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
from contextlib import suppress

_STILL_ACTIVE = 259
_QUERY_LIMITED = 0x1000
_SYNCHRONIZE = 0x00100000


def _windows_running(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    handle = kernel32.OpenProcess(_QUERY_LIMITED | _SYNCHRONIZE, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def running(pid: int) -> bool:
    """Whether ``pid`` is executing; exited processes awaiting a reap are not."""
    if sys.platform == "win32":
        return _windows_running(pid)
    if sys.platform == "linux":
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return False
        try:
            # A pidfd respects the current PID namespace and becomes readable at
            # termination, including an exited descendant awaiting another
            # parent's reap. A shared /proc mount can expose unrelated numbers.
            return not select.select([descriptor], [], [], 0)[0]
        finally:
            os.close(descriptor)
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def kill(pid: int) -> None:
    """Best-effort test safety net; never a substitute for ownership assertions."""
    with suppress(OSError):
        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
