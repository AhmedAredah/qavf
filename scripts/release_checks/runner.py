"""Run verification processes and persist success, failure, or missing evidence."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import coverage as coverage_checks
from .provenance import repository_identity, timestamp, write_json


def clean_environment() -> dict[str, str]:
    values = os.environ.copy()
    # Do not let external pytest flags/plugin autoload or Python paths weaken a gate.
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        values.pop(key, None)
    values["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    values["QAVF_HYPOTHESIS_PROFILE"] = "ci"
    return values


def run_tests(
    *,
    root: Path,
    workdir: Path,
    python: Path,
    output: Path,
    check_id: str,
    optimized: bool = False,
    coverage: bool = False,
    selection: str = "not integration",
    paths: tuple[str, ...] = ("tests",),
    metadata: dict[str, Any] | None = None,
) -> int:
    output = output.resolve()
    results = output.with_suffix(".tests.json")
    results.unlink(missing_ok=True)
    identity = repository_identity(root)
    command = [
        str(python),
        *(["-O"] if optimized else []),
        "-m",
        "pytest",
        "-p",
        "scripts.release_checks.pytest_plugin",
        "-p",
        "_hypothesis_pytestplugin",
        "--release-results",
        str(results),
        "-m",
        selection,
        *paths,
    ]
    environment = clean_environment()
    coverage_config = coverage_checks.prepare(root, output) if coverage else None
    if coverage_config is not None:
        environment["COVERAGE_PROCESS_START"] = str(coverage_config)
        module_index = command.index("-m")
        command[module_index:module_index] = [
            "-m",
            "coverage",
            "run",
            f"--rcfile={coverage_config}",
        ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "id": check_id,
        "kind": "pytest",
        "status": "running",
        "started": timestamp(),
        "identity": identity,
        "command": command,
        "metadata": {
            "test_paths": list(paths),
            "selection": selection,
            "hypothesis_profile": "ci",
            "branch_coverage": coverage,
            **(metadata or {}),
        },
    }
    write_json(output, record)
    try:
        # A fresh owned base avoids pytest's shared pytest-current symlink cleanup
        # on Windows. Never give --basetemp a caller-owned directory: pytest clears it.
        with tempfile.TemporaryDirectory(prefix="qavf-pytest-") as directory:
            command.extend(["--basetemp", str(Path(directory) / "tests")])
            write_json(output, record)
            process = subprocess.run(command, cwd=workdir, env=environment, check=False)
        record["process_exit_code"] = process.returncode
        if results.is_file():
            record["tests"] = json.loads(results.read_text(encoding="utf-8"))
        else:
            record["error"] = "pytest did not produce completion evidence"
        unchanged = identity == repository_identity(root)
        record["source_unchanged"] = unchanged
        record["status"] = (
            "passed"
            if process.returncode == 0
            and unchanged
            and record.get("tests", {}).get("status") == "passed"
            else "failed"
        )
        if coverage_config is not None:
            record["coverage"] = coverage_checks.finish(
                python, coverage_config, output, workdir, environment
            )
            if record["coverage"]["status"] != "passed":
                record["status"] = "failed"
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        record["finished"] = timestamp()
        write_json(output, record)
    return 0 if record["status"] == "passed" else 1
