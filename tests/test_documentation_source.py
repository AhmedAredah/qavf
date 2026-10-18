"""Documentation generation must describe the checkout that owns the command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import qavf


def test_documentation_command_uses_its_package_before_unrelated_installation(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "docs").mkdir()
    shutil.copytree(
        Path(qavf.__file__).parent,
        checkout / "qavf",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    command = checkout / "scripts/generate_docs.py"
    shutil.copy2(Path(__file__).parents[1] / "scripts/generate_docs.py", command)
    unrelated = tmp_path / "unrelated"
    (unrelated / "qavf").mkdir(parents=True)
    (unrelated / "qavf/__init__.py").write_text(
        "raise RuntimeError('unrelated installed package imported')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(unrelated)}
    result = subprocess.run(
        [sys.executable, str(command)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unrelated installed package imported" not in result.stderr
    sources = (checkout / "docs/sources.md").read_text()
    assert "Leading and trailing whitespace" in sources
    assert (checkout / "docs/rules.md").is_file()
