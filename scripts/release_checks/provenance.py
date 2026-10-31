"""Stable source identity and explicit, non-secret execution provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_INPUTS = (
    "qavf",
    "tests",
    "scripts",
    "config",
    "docs",
    ".github",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    ".gitattributes",
    ".gitignore",
)
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".lock",
    ".yml",
    ".yaml",
    ".md",
    ".txt",
    ".json",
    ".j2",
    ".csv",
    ".html",
    ".sql",
    ".gitattributes",
}


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root: Path) -> str:
    """Hash release inputs, with checkout line endings normalized for text files."""
    digest = hashlib.sha256()
    for name in SOURCE_INPUTS:
        base = root / name
        paths = sorted(base.rglob("*")) if base.is_dir() else [base]
        for path in paths:
            if not path.is_file() or any(
                part.startswith(".") for part in path.relative_to(base).parts
            ):
                continue
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            content = path.read_bytes()
            if path.suffix in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitattributes"}:
                content = content.replace(b"\r\n", b"\n")
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def repository_identity(root: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        revision = None
    lock = root / "uv.lock"
    return {
        "source_sha256": source_digest(root),
        "lock_sha256": hashlib.sha256(lock.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if lock.is_file()
        else None,
        "git_revision": revision,
        "ci_run_id": os.environ.get("GITHUB_RUN_ID"),
        "ci_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }


def environment() -> dict[str, Any]:
    import numpy as np

    import qavf

    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "optimization": sys.flags.optimize,
        "hypothesis_profile": os.environ.get("QAVF_HYPOTHESIS_PROFILE"),
        "qavf_import_path": str(Path(qavf.__file__).resolve()),
        "capabilities": {
            "scipy": importlib.util.find_spec("scipy") is not None,
            "numpy_item_nonprogress": {
                name: isinstance(kind("1.5").item(), np.generic)
                for name, kind in (("longdouble", np.longdouble), ("clongdouble", np.clongdouble))
            },
        },
        "packages": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
