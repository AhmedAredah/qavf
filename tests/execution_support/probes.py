"""Real configured plugin workloads; none depends on implementation internals."""

from __future__ import annotations

import json
import mmap
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from qavf.engine.registry import Finding, Rule, RuleParams


class ProbeParams(RuleParams):
    mode: str = "pass"
    directory: str


class ExecutionProbe(Rule[ProbeParams]):
    rule_id = "ISOLATION-PROBE"
    name = "Independent execution probe"
    description = "Exercise an execution boundary with an observable real workload."
    category = "test"
    Params = ProbeParams

    def evaluate(self, ctx):
        directory = Path(self.params.directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "worker.json").write_text(
            json.dumps({"pid": os.getpid(), "parent": os.getppid()}), encoding="utf-8"
        )
        mode = self.params.mode
        if mode == "sleep":
            time.sleep(90)
        elif mode == "cpu":
            while True:
                pass
        elif mode == "memory" and sys.platform == "win32":
            # Windows limits committed private memory; an anonymous mmap is a
            # pagefile-backed section outside that charge. Commit without touching.
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32
            ]
            mem_commit_reserve, page_readwrite = 0x3000, 0x04
            if not kernel32.VirtualAlloc(None, 8 << 30, mem_commit_reserve, page_readwrite):
                raise MemoryError("committed allocation was refused")
        elif mode == "memory":
            # Reserve virtual space without committing physical RAM. This request
            # must be refused under the configured address-space allowance.
            with mmap.mmap(-1, 8 << 30) as allocation:
                allocation[0] = 1
        elif mode == "file":
            with (directory / "oversized.bin").open("wb") as handle:
                handle.write(b"x" * (128 << 10))
        elif mode == "exit":
            os._exit(23)
        elif mode == "thread_shutdown":
            # The rule returns a complete report, but Python cannot finish its
            # process shutdown while this non-daemon thread is still running.
            threading.Thread(target=time.sleep, args=(90,), daemon=False).start()
        elif mode in {"descendant_timeout", "descendant_success"}:
            program = (
                "import json, os, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "open(sys.argv[1], 'w').write(json.dumps({'pid': os.getpid()})); "
                "time.sleep(90)"
            )
            subprocess.Popen(
                [sys.executable, "-c", program, str(directory / "descendant.json")],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 10
            while not (directory / "descendant.json").exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("descendant did not start")
                time.sleep(0.01)
            if mode == "descendant_timeout":
                time.sleep(90)
        elif mode != "pass":
            raise ValueError(f"Unknown execution probe mode: {mode}")
        return Finding.ok(
            "Complete probe observation",
            total_records=3,
            affected_count=0,
            metrics={
                "worker_pid": os.getpid(),
                "worker_optimization": sys.flags.optimize,
                "decimal": Decimal("123456789.123456789123456789"),
                "bytes": b"\x00\xff\x80",
                "date": date(2024, 2, 29),
                "timestamp": datetime(2024, 2, 29, 12, 3, 4, 567890, tzinfo=UTC),
                "duration": timedelta(days=-2, microseconds=123456),
                "integer": 2**100 + 1,
                "nested": {"text": "\u03b1\x00\U0001f69a", "values": [True, None, 3]},
            },
        )
