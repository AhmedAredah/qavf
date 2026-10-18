"""Fresh-interpreter entry point: apply OS limits before importing QAVF.

Executed by file path, deliberately without importing the qavf package first.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def main(control_path: str) -> int:
    control: dict[str, Any] = {}
    phase = "bootstrap"
    try:
        with Path(control_path).open(encoding="utf-8") as handle:
            control = json.load(handle)
        # This module is adjacent to the script, not imported through qavf.
        from os_limits import apply_limits

        control["effective_limits"] = apply_limits(control["limits"], control.get("job"))
        for name, value in control["native_environment"].items():
            os.environ[name] = value
        sys.path[:] = control["sys_path"]
        # -S deliberately bypasses site startup. The release harness opts in to
        # coverage explicitly; normal execution has no coverage dependency.
        # Start only after OS limits have been applied, before QAVF imports.
        if os.environ.get("COVERAGE_PROCESS_START"):
            import coverage

            coverage.process_startup()
        phase = "worker"
        from qavf.execution.worker import run

        run(control["request_path"], control["result_path"], control)
        return 0
    except BaseException as exc:
        # A worker may already have emitted a safe, more useful diagnostic.
        # Bootstrap failure descriptions never echo input/configuration values.
        error_path = control.get("error_path")
        if error_path:
            try:
                with Path(error_path).open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "phase": phase,
                            "error_type": type(exc).__name__,
                            "message": f"Isolated {phase} failed ({type(exc).__name__}).",
                        },
                        handle,
                    )
            except OSError:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
