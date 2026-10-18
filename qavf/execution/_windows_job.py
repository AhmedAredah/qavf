"""Standard-library Windows Job Object access for isolated execution.

Imported by the bootstrap before QAVF or native dependencies, so it must only use
the standard library. A job supplies the per-process committed-memory and
user-mode CPU ceilings, whole-tree termination and live-member accounting that
POSIX rlimits, process groups and procfs provide on Linux. See
https://learn.microsoft.com/windows/win32/procthread/job-objects.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any

_JOB_OBJECT_ASSIGN_PROCESS = 0x0001
_JOB_OBJECT_QUERY = 0x0004
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_MORE_DATA = 234
_BASIC_ACCOUNTING = 1
_PROCESS_ID_LIST = 3
_EXTENDED_LIMITS = 9
_LIMIT_PROCESS_TIME = 0x0002
_LIMIT_PROCESS_MEMORY = 0x0100
_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x0400
_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_SEM_NOOPENFILEERRORBOX = 0x8000
_TICKS_PER_SECOND = 10_000_000
UNLIMITED = -1
# The system terminates a process that exceeds its job user-time allowance with
# this NTSTATUS; QAVF terminates the job itself with KILLED_EXIT_CODE.
CPU_LIMIT_EXIT_CODE = 0xC0000044  # STATUS_QUOTA_EXCEEDED
KILLED_EXIT_CODE = 137  # the conventional 128 + SIGKILL status


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


def _kernel32() -> Any:
    if sys.platform != "win32":
        raise RuntimeError("Windows job objects require Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle, boolean = wintypes.HANDLE, wintypes.BOOL
    signatures: dict[str, tuple[Any, list[Any]]] = {
        "CreateJobObjectW": (handle, [ctypes.c_void_p, wintypes.LPCWSTR]),
        "OpenJobObjectW": (handle, [wintypes.DWORD, boolean, wintypes.LPCWSTR]),
        "SetInformationJobObject": (boolean, [handle, ctypes.c_int, ctypes.c_void_p,
                                              wintypes.DWORD]),
        "QueryInformationJobObject": (boolean, [handle, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.c_void_p]),
        "AssignProcessToJobObject": (boolean, [handle, handle]),
        "IsProcessInJob": (boolean, [handle, handle, ctypes.POINTER(boolean)]),
        "TerminateJobObject": (boolean, [handle, wintypes.UINT]),
        "CloseHandle": (boolean, [handle]),
        "GetCurrentProcess": (handle, []),
        "OpenProcess": (handle, [wintypes.DWORD, boolean, wintypes.DWORD]),
        "GetProcessTimes": (boolean, [handle] + [ctypes.POINTER(ctypes.c_int64)] * 4),
        "TerminateProcess": (boolean, [handle, wintypes.UINT]),
        "GetErrorMode": (wintypes.UINT, []),
        "SetErrorMode": (wintypes.UINT, [wintypes.UINT]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(kernel32, name)
        function.restype, function.argtypes = restype, argtypes
    return kernel32


def _check(result: object, action: str) -> None:
    if not result:
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        raise OSError(f"could not {action} (Windows error {error})")


def _limits_of(kernel32: Any, job: Any) -> _ExtendedLimits:
    info = _ExtendedLimits()
    _check(
        kernel32.QueryInformationJobObject(
            job, _EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info), None
        ),
        "query job limits",
    )
    return info


class Job:
    """A named, parent-owned job whose members die when its last handle closes."""

    def __init__(self, name: str, *, memory_bytes: int | None, cpu_seconds: int | None):
        self._kernel32 = _kernel32()
        self.name = name
        self._cpu_ticks = None if cpu_seconds is None else cpu_seconds * _TICKS_PER_SECOND
        self.limits_cpu = cpu_seconds is not None
        self._handle = self._kernel32.CreateJobObjectW(None, name)
        _check(self._handle, "create a job object")
        try:
            info = _ExtendedLimits()
            flags = _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            if memory_bytes is not None:
                flags |= _LIMIT_PROCESS_MEMORY
                info.ProcessMemoryLimit = memory_bytes
            if cpu_seconds is not None:
                flags |= _LIMIT_PROCESS_TIME
                info.BasicLimitInformation.PerProcessUserTimeLimit = (
                    cpu_seconds * _TICKS_PER_SECOND
                )
            info.BasicLimitInformation.LimitFlags = flags
            _check(
                self._kernel32.SetInformationJobObject(
                    self._handle, _EXTENDED_LIMITS, ctypes.byref(info), ctypes.sizeof(info)
                ),
                "set job limits",
            )
        except BaseException:
            self.close()
            raise

    def active_processes(self) -> int:
        info = _Accounting()
        _check(
            self._kernel32.QueryInformationJobObject(
                self._handle, _BASIC_ACCOUNTING, ctypes.byref(info), ctypes.sizeof(info), None
            ),
            "query job membership",
        )
        return int(info.ActiveProcesses)

    def _member_ids(self) -> list[int]:
        capacity = 64
        while True:
            class _IdList(ctypes.Structure):
                _fields_ = [
                    ("NumberOfAssignedProcesses", wintypes.DWORD),
                    ("NumberOfProcessIdsInList", wintypes.DWORD),
                    ("ProcessIdList", ctypes.c_size_t * capacity),
                ]

            info = _IdList()
            if self._kernel32.QueryInformationJobObject(
                self._handle, _PROCESS_ID_LIST, ctypes.byref(info), ctypes.sizeof(info), None
            ):
                return [int(pid) for pid in info.ProcessIdList[: info.NumberOfProcessIdsInList]]
            if ctypes.get_last_error() != _ERROR_MORE_DATA:  # type: ignore[attr-defined]
                _check(False, "list job members")
            capacity *= 4

    def enforce_cpu(self) -> None:
        """Terminate members whose user-mode CPU time reached the per-process limit.

        Windows itself checks job time limits only every few seconds; the parent
        calls this while it waits so a CPU allowance is enforced promptly. The
        job limit remains as a backstop.
        """
        if self._cpu_ticks is None:
            return
        kernel32 = self._kernel32
        for pid in self._member_ids():
            process = kernel32.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE, False, pid
            )
            if not process:
                continue  # Already exited.
            try:
                times = [ctypes.c_int64() for _ in range(4)]
                member = wintypes.BOOL()
                if (
                    kernel32.GetProcessTimes(process, *map(ctypes.byref, times))
                    and times[3].value >= self._cpu_ticks
                    # The ID may have been reused after the listing.
                    and kernel32.IsProcessInJob(process, self._handle, ctypes.byref(member))
                    and member.value
                ):
                    kernel32.TerminateProcess(process, CPU_LIMIT_EXIT_CODE)
            finally:
                kernel32.CloseHandle(process)

    def terminate(self) -> None:
        _check(
            self._kernel32.TerminateJobObject(self._handle, KILLED_EXIT_CODE),
            "terminate the job",
        )

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def join(name: str) -> dict[str, dict[str, int]]:
    """Enter the parent's job from inside the worker and report its effective limits.

    Self-assignment happens before any work, so no descendant can start outside the
    job. The worker closes its own job handle so that parent exit still closes the
    job and kills every member.
    """
    kernel32 = _kernel32()
    job = kernel32.OpenJobObjectW(_JOB_OBJECT_ASSIGN_PROCESS | _JOB_OBJECT_QUERY, False, name)
    _check(job, "open the isolation job")
    try:
        current = kernel32.GetCurrentProcess()
        member = wintypes.BOOL()
        _check(kernel32.IsProcessInJob(current, job, ctypes.byref(member)), "inspect the job")
        if not member.value:
            _check(kernel32.AssignProcessToJobObject(job, current), "enter the isolation job")
            _check(kernel32.IsProcessInJob(current, job, ctypes.byref(member)), "inspect the job")
            if not member.value:
                raise OSError("the worker is not a member of its isolation job")
        # Suppress crash and critical-error dialogs that would otherwise stall
        # the worker until the deadline; descendants inherit this error mode.
        kernel32.SetErrorMode(
            kernel32.GetErrorMode()
            | _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX | _SEM_NOOPENFILEERRORBOX
        )
        info = _limits_of(kernel32, job)
    finally:
        kernel32.CloseHandle(job)
    basic = info.BasicLimitInformation
    memory = (
        int(info.ProcessMemoryLimit)
        if basic.LimitFlags & _LIMIT_PROCESS_MEMORY else UNLIMITED
    )
    cpu = (
        int(basic.PerProcessUserTimeLimit) // _TICKS_PER_SECOND
        if basic.LimitFlags & _LIMIT_PROCESS_TIME else UNLIMITED
    )
    crash = 0 if basic.LimitFlags & _LIMIT_DIE_ON_UNHANDLED_EXCEPTION else UNLIMITED
    return {
        "address_space_bytes": {"soft": memory, "hard": memory},
        "cpu_time_s": {"soft": cpu, "hard": cpu},
        "core_bytes": {"soft": crash, "hard": crash},
    }
