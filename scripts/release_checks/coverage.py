"""Branch coverage with explicit configuration shared by ordinary and -S children."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def prepare(root: Path, output: Path) -> Path:
    """Use run-owned data paths, preventing stale or concurrent runs from combining."""
    config = output.with_suffix(".coverage.toml")
    data = output.with_suffix(".coverage-data")
    for stale in data.parent.glob(data.name + "*"):
        if stale.is_file():
            stale.unlink()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "[tool.coverage.run]\nbranch = true\nparallel = true\n"
        'patch = ["subprocess", "_exit"]\n'
        f"source = [{json.dumps(str(root / 'qavf'))}]\n"
        f"data_file = {json.dumps(str(data))}\n",
        encoding="utf-8",
    )
    return config


def finish(
    python: Path, config: Path, output: Path, workdir: Path, environment: dict[str, str]
) -> dict[str, Any]:
    """Persist missing coverage as failure; expose actual isolated worker lines."""
    target = output.with_suffix(".coverage.json")
    env = {key: value for key, value in environment.items() if not key.startswith("COVERAGE_")}
    for operation in (
        ["combine", f"--rcfile={config}"],
        ["json", f"--rcfile={config}", "-o", str(target)],
    ):
        subprocess.run(
            [str(python), "-m", "coverage", *operation], cwd=workdir, env=env, check=True
        )
    value = json.loads(target.read_text(encoding="utf-8"))
    worker_lines = sorted(
        {
            line
            for name, details in value["files"].items()
            if name.replace("\\", "/").endswith("qavf/execution/worker.py")
            for line in details.get("executed_lines", [])
        }
    )
    return {
        "status": "passed" if value["meta"].get("branch_coverage") else "failed",
        "path": str(target),
        "totals": value["totals"],
        "isolated_worker_executed_lines": worker_lines,
        "scope": "Ordinary children and successful -S workers; SIGKILL cannot flush coverage.",
    }
