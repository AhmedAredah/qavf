"""Run the full release checks on one real host, retaining every failure."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .gate import evidence_errors
from .provenance import SOURCE_INPUTS, repository_identity, write_json


def host_label() -> str:
    system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(platform.system())
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    allowed = {"3.12", "3.13", "3.14"} if system == "linux" else {"3.12"}
    if system is None or version not in allowed:
        raise ValueError("Use Linux Python 3.12-3.14, or Windows/macOS Python 3.12")
    return f"{system}-{version}"


def command_plan(root: Path, output: Path, label: str) -> list[tuple[str, list[str]]]:
    python = sys.executable
    release = str(root / "scripts" / "verify_release.py")
    normal = [python, release, "test", "--id", f"{label}-source",
              "--output", str(output / f"{label}-source.json")]
    if label == "linux-3.12":
        normal.append("--coverage")
    return [
        ("build", ["uv", "build", "--clear"]),
        ("payload", [python, str(root / "scripts" / "verify_dist.py")]),
        ("extensions", [python, release, "extensions"]),
        ("source", normal),
        ("source-optimized", [python, release, "test", "--optimized", "--id",
                              f"{label}-source-optimized", "--output",
                              str(output / f"{label}-source-optimized.json")]),
        ("wheel", [python, release, "installed", "--mode", "locked", "--id",
                   f"{label}-wheel", "--output-dir", str(output)]),
    ]


def verify_platform(root: Path, output: Path) -> int:
    root, output = root.resolve(), output.resolve()
    label = host_label()
    if shutil.which("uv") is None:
        raise ValueError("Install uv before running platform verification")
    protected = [root / name for name in (*SOURCE_INPUTS, "dist")]
    if output == root or any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("Evidence must be outside maintained source and dist directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new empty evidence directory for each attempt")
    output.mkdir(parents=True, exist_ok=True)
    identity = repository_identity(root)
    environment = os.environ.copy()
    environment.update(POLARS_MAX_THREADS="2", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", PYTHONUTF8="1")
    steps: list[dict[str, Any]] = []
    build_ok = False
    interrupted = False
    try:
        for name, argv in command_plan(root, output, label):
            if name in {"payload", "wheel"} and not build_ok:
                steps.append({"name": name, "status": "blocked", "reason": "build failed"})
                continue
            print(f"Running {name}; log: {str(output / (name + '.log'))!a}", flush=True)
            try:
                with (output / f"{name}.log").open("wb") as log:
                    code = subprocess.run(argv, cwd=root, env=environment, stdout=log,
                                          stderr=subprocess.STDOUT, check=False).returncode
                steps.append({"name": name, "argv": argv, "exit_code": code,
                              "status": "passed" if code == 0 else "failed"})
                if name == "build":
                    build_ok = code == 0
            except OSError as error:
                steps.append({"name": name, "argv": argv, "status": "failed", "error": str(error)})
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        records = []
        for path in output.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict):
                    records.append(record)
            except (OSError, ValueError):
                continue
        expected = {f"{label}-{suffix}" for suffix in (
            "source", "source-optimized", "wheel", "wheel-optimized", "wheel-setup")}
        errors = evidence_errors(records, identity, expected)
        if repository_identity(root) != identity:
            errors.append("source changed during verification")
        passed = not interrupted and not errors and all(s["status"] == "passed" for s in steps)
        write_json(output / "platform-summary.json", {
            "platform": label, "identity": identity, "steps": steps, "errors": errors,
            "status": "passed" if passed else "blocked", "interrupted": interrupted,
            "release_ready": False,
        })
    return 0 if passed else 1
