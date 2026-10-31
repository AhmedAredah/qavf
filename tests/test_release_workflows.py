"""The checked-in workflow must orchestrate the same strict gate it claims."""

from pathlib import Path

import yaml

from scripts.release_checks.gate import REQUIRED_JOBS

ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_covers_gate_matrix_and_cannot_publish():
    config = yaml.safe_load((ROOT / ".github/workflows/verify.yml").read_text())
    jobs = config["jobs"]
    assert jobs.keys() >= REQUIRED_JOBS
    assert set(jobs["aggregate"]["needs"]) == REQUIRED_JOBS
    assert jobs["aggregate"]["if"] == "always()"
    assert jobs["test"]["strategy"]["matrix"]["python"] == ["3.12", "3.13", "3.14"]
    assert {
        item["label"] for item in jobs["platform-contracts"]["strategy"]["matrix"]["include"]
    } == {
        "windows-3.12",
        "macos-3.12",
    }
    assert config["permissions"] == {"contents": "read"}
    for job in jobs.values():
        assert not job.get("permissions")
        for step in job["steps"]:
            command = step.get("run", "")
            assert "uv publish" not in command and "gh release" not in command
            assert "pypa/gh-action-pypi-publish" not in step.get("uses", "")
    aggregation = next(
        step for step in jobs["aggregate"]["steps"] if "aggregate --" in step.get("run", "")
    )
    assert aggregation["env"]["QAVF_CI_JOBS"] == "${{ toJSON(needs) }}"


def test_routine_ci_checks_public_input_type_fixture():
    config = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert any(
        "--warn-unused-ignores tests/typing/inputs.py" in step.get("run", "")
        for step in config["jobs"]["quality"]["steps"]
    )
