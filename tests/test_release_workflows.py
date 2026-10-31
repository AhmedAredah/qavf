"""The checked-in workflow must orchestrate the same strict gate it claims."""

from pathlib import Path

import yaml

from scripts.release_checks.gate import REQUIRED_JOBS

ROOT = Path(__file__).resolve().parents[1]


PUBLISHERS = {"publish-pypi", "github-release", "container"}
"""The only jobs that may publish, each behind the strict aggregate gate."""


def test_release_workflow_covers_gate_matrix_and_publishes_only_after_it():
    config = yaml.safe_load((ROOT / ".github/workflows/verify.yml").read_text())
    jobs = config["jobs"]
    assert jobs.keys() >= REQUIRED_JOBS | PUBLISHERS
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
    # PyPI publishes only for a v* tag and only after the aggregate passed; the
    # release and image follow PyPI. No other job publishes or gains write access.
    assert jobs["publish-pypi"]["needs"] == "aggregate"
    assert jobs["publish-pypi"]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert jobs["github-release"]["needs"] == jobs["container"]["needs"] == "publish-pypi"
    for name, job in jobs.items():
        if name in PUBLISHERS:
            continue
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
