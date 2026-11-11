"""The documented way to run one comparison of a project works as written (UX-04):
``qavf run -c project.yaml --only "<compare item name>"`` and ``--tags``."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

import pytest

from qavf.cli import main

ROOT = Path(__file__).resolve().parents[1]
SECTION = "### Run one comparison of a project"
_BLOCK = re.compile(r"```(yaml|bash)\n(.*?)```", re.DOTALL)
BEFORE = "site,aadt\n1,100\n2,200\n3,300\n"
AFTER = "site,aadt\n1,101\n2,199\n3,300\n"


def _section() -> tuple[str, list[str]]:
    """The section's project file and its commands."""
    text = (ROOT / "docs" / "comparison.md").read_text(encoding="utf-8")
    body = text.split(SECTION, 1)[1].split("\n## ", 1)[0].split("\n### ", 1)[0]
    blocks = _BLOCK.findall(body)
    project = next(code for kind, code in blocks if kind == "yaml")
    commands = [
        line.strip()
        for kind, code in blocks
        if kind == "bash"
        for line in code.splitlines()
        if line.strip()
    ]
    return project, commands


PROJECT, COMMANDS = _section()


def test_the_section_documents_a_project_and_both_selectors():
    assert PROJECT.startswith("# project.yaml")
    assert any("--only" in command for command in COMMANDS)
    assert any("--tags" in command for command in COMMANDS)


@pytest.mark.parametrize("command", COMMANDS)
def test_the_documented_command_runs_the_comparison_alone(command, tmp_path, monkeypatch):
    for year, data in (("2024", BEFORE), ("2025", AFTER)):
        folder = tmp_path / "releases" / year
        folder.mkdir(parents=True)
        (folder / "counts.csv").write_text(data, encoding="utf-8")
    (tmp_path / "project.yaml").write_text(PROJECT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    arguments = shlex.split(command)
    assert arguments[:2] == ["qavf", "run"]
    assert main(arguments[1:]) == 0
    report = json.loads((tmp_path / "qa_output" / "qa_report.json").read_text(encoding="utf-8"))
    names = {result["item_name"] for result in report["results"]}
    assert "counts 2024 vs 2025" in names
    assert "counts are positive" not in names
    assert {result["rule_id"] for result in report["results"]} == {"CMP-001"}
    assert report["metadata"]["resource_usage"]["limits"]["target_memory"]["source"] == (
        "resources"
    )
