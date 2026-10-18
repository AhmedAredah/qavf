"""Observe a complete public call from outside its isolated execution worker.

The pytest parent places a separate watchdog around this driver. Resource failures
cannot kill or hang pytest, and observations do not trust worker telemetry.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _parent_limits():
    if sys.platform not in {"linux", "darwin"}:
        return {}
    import resource

    return {
        name: resource.getrlimit(getattr(resource, name))
        for name in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE")
    }


def main() -> None:
    from qavf import ResourceLimits, validate

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    started = time.monotonic()
    record = {
        "driver_pid": os.getpid(),
        "driver_optimization": sys.flags.optimize,
        "limits_before": _parent_limits(),
    }
    try:
        report = validate(
            config=request["config"], resources=ResourceLimits(**request["resources"])
        )
        report.write_json(output.with_suffix(".report.json"))
        record.update(outcome="report", verdict=report.verdict(), exit_code=report.exit_code())
    except Exception as exc:
        record.update(outcome="error", error_type=type(exc).__name__, message=str(exc))
    record.update(elapsed=time.monotonic() - started, limits_after=_parent_limits())
    output.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
