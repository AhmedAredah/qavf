"""Release gates must reject incomplete, stale and misleading success evidence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release_checks.gate import evidence_errors, required_checks
from scripts.release_checks.policy import accepted_skip
from scripts.release_checks.provenance import source_digest
from scripts.release_checks.runner import clean_environment, run_tests
from scripts.release_checks.validation import outcome_errors

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "source_sha256": "a" * 64,
    "lock_sha256": "b" * 64,
    "git_revision": None,
    "ci_run_id": None,
    "ci_run_attempt": None,
}


def evidence() -> dict:
    return {
        "schema_version": 1,
        "id": "linux-3.12-source",
        "kind": "pytest",
        "status": "passed",
        "identity": dict(IDENTITY),
        "process_exit_code": 0,
        "source_unchanged": True,
        "metadata": {"test_paths": ["tests"], "selection": "not integration"},
        "coverage": {"status": "passed", "isolated_worker_executed_lines": [1, 2]},
        "tests": {
            "status": "passed",
            "exit_code": 0,
            "passed": 1,
            "collection_errors": [],
            "unexpected_outcomes": [],
            "selected": ["tests/test_example.py::test_value"],
            "outcomes": [
                {
                    "nodeid": "tests/test_example.py::test_value",
                    "phase": "call",
                    "outcome": "passed",
                    "reason": "",
                    "accepted": False,
                }
            ],
            "environment": {
                "system": "Linux",
                "python_version": "3.12.14",
                "optimization": 0,
                "qavf_import_path": "/checkout/qavf/__init__.py",
                "packages": {"qavf": "0.4.0.dev0"},
                "capabilities": {"scipy": True},
            },
        },
    }


def errors(record: dict) -> list[str]:
    return evidence_errors([record], IDENTITY, {"linux-3.12-source"})


def test_complete_matching_evidence_passes():
    assert errors(evidence()) == []


@pytest.mark.parametrize("value", [None, {}, {"status": "passed"}])
def test_missing_subprocess_coverage_blocks_release(value):
    record = evidence()
    record["coverage"] = value
    assert errors(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "installation"),
        ("status", "running"),
        ("status", "failed"),
        ("schema_version", 999),
        ("process_exit_code", -9),
        ("source_unchanged", False),
    ],
)
def test_incomplete_or_wrong_phase_rejected(field, value):
    record = evidence()
    record[field] = value
    assert errors(record)


@pytest.mark.parametrize("field", list(IDENTITY))
def test_stale_or_foreign_identity_rejected(field):
    record = evidence()
    record["identity"][field] = "different"
    assert errors(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("system", "Windows"),
        ("python_version", "3.14.0"),
        ("optimization", 1),
        ("qavf_import_path", None),
        ("packages", {}),
    ],
)
def test_environment_label_cannot_substitute_for_actual_runtime(field, value):
    record = evidence()
    record["tests"]["environment"][field] = value
    assert errors(record)


def test_missing_duplicate_and_empty_evidence_rejected():
    expected = {"linux-3.12-source"}
    assert evidence_errors([], IDENTITY, expected)
    assert evidence_errors([evidence(), evidence()], IDENTITY, expected)
    record = evidence()
    record["tests"]["passed"] = 0
    record["tests"]["selected"] = []
    record["tests"]["outcomes"] = []
    assert errors(record)


@pytest.mark.parametrize("outcome", ["skipped", "xfailed", "xpassed", "failed", "error", "unknown"])
def test_success_summary_cannot_hide_bad_outcome(outcome):
    record = evidence()
    record["tests"]["outcomes"][0].update(outcome=outcome, accepted=True)
    assert errors(record)


def test_unfinished_collected_test_is_not_passing():
    record = evidence()
    record["tests"]["selected"].append("tests/test_example.py::test_never_ran")
    assert errors(record)


def test_partial_suite_cannot_satisfy_full_suite_gate():
    record = evidence()
    record["metadata"]["test_paths"] = ["tests/test_example.py"]
    assert errors(record)


def test_release_manifest_includes_all_required_environments():
    required = required_checks()
    assert {"integrations", "core-setup", "floors", "floors-optimized"} <= required
    for platform in ("linux-3.12", "linux-3.13", "linux-3.14", "windows-3.12", "macos-3.12"):
        assert {
            f"{platform}-source",
            f"{platform}-wheel-optimized",
            f"{platform}-wheel-setup",
        } <= required


def test_skip_policy_requires_actual_environment_and_exact_reason():
    node = "tests/test_geo_and_drift.py::test_ks_without_scipy_is_unassessed"
    reason = "checks the behaviour without the stats extra"
    assert accepted_skip(node, reason, {"capabilities": {"scipy": True}})
    assert not accepted_skip(node, reason, {"capabilities": {"scipy": False}})
    assert not accepted_skip(node, "network unavailable", {"capabilities": {"scipy": True}})
    assert not accepted_skip("tests/test_sources.py::test_excel", "extension unavailable", {})


def test_accepted_platform_skip_revalidated_from_evidence():
    record = evidence()["tests"]
    node = (
        "tests/test_report_encoding.py::"
        "test_numpy_nonprogress_scalar_is_explicitly_rejected[longdouble]"
    )
    skip = {
        "nodeid": node,
        "phase": "setup",
        "outcome": "skipped",
        "accepted": True,
        "reason": "this platform converts the dtype to a Python scalar",
    }
    record["selected"].append(node)
    record["outcomes"].append(skip)
    record["environment"]["capabilities"]["numpy_item_nonprogress"] = {"longdouble": False}
    assert outcome_errors(record) == []
    record["environment"]["capabilities"]["numpy_item_nonprogress"]["longdouble"] = True
    assert outcome_errors(record)


def test_external_pytest_options_and_plugins_cannot_weaken_gate(monkeypatch):
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        monkeypatch.setenv(key, "external-value")
    env = clean_environment()
    assert all(
        key not in env for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    )
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


@pytest.mark.parametrize(
    "body,passed",
    [
        ("def test_ok(): assert True", True),
        ("import pytest\ndef test_skip(): pytest.skip('unexpected unavailable dependency')", False),
        (
            "import pytest\n@pytest.mark.xfail(reason='known defect')\n"
            "def test_xfail(): assert False",
            False,
        ),
        (
            "import pytest\n@pytest.mark.xfail(reason='stale expected failure')\n"
            "def test_xpass(): assert True",
            False,
        ),
        ("raise RuntimeError('collection failed')", False),
        ("# No tests were collected", False),
    ],
)
def test_real_pytest_exit_and_record_are_fail_closed(tmp_path, body, passed):
    test_file = tmp_path / "test_gate_probe.py"
    test_file.write_text(body + "\n")
    output = tmp_path / "results.json"
    # Pin the probe's root and configuration to its own directory. Otherwise pytest
    # roots it at the common ancestor of the project and the temporary directory
    # (/tmp), and collection walks unrelated, possibly unreadable directories.
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(config),
        "--rootdir",
        str(tmp_path),
        "-p",
        "scripts.release_checks.pytest_plugin",
        "--release-results",
        str(output),
        "-q",
        str(test_file),
    ]
    process = subprocess.run(
        command, cwd=ROOT, env=clean_environment(), capture_output=True, text=True
    )
    assert (process.returncode == 0) is passed, process.stdout + process.stderr
    result = json.loads(output.read_text())
    assert (result["status"] == "passed") is passed


def test_missing_child_completion_record_rejected(tmp_path, monkeypatch):
    from scripts.release_checks import runner

    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0)
    )
    monkeypatch.setattr(runner, "repository_identity", lambda root: copy.deepcopy(IDENTITY))
    output = tmp_path / "evidence.json"
    assert (
        run_tests(
            root=ROOT, workdir=ROOT, python=Path(sys.executable), output=output, check_id="probe"
        )
        == 1
    )
    assert json.loads(output.read_text())["status"] == "failed"


def test_source_identity_tracks_changes_but_ignores_interpreter_caches(tmp_path):
    package = tmp_path / "qavf"
    package.mkdir()
    source = package / "__init__.py"
    source.write_bytes(b"value = 1\n")
    original = source_digest(tmp_path)
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "cached.pyc").write_bytes(b"arbitrary")
    assert source_digest(tmp_path) == original
    source.write_bytes(b"value = 1\r\n")
    assert source_digest(tmp_path) == original
    source.write_bytes(b"value = 2\n")
    assert source_digest(tmp_path) != original


def test_duplicate_terminal_outcomes_cannot_inflate_success_count():
    record = evidence()
    record["tests"]["outcomes"].append(dict(record["tests"]["outcomes"][0]))
    record["tests"]["passed"] = 2
    assert errors(record)


def test_malformed_skip_capabilities_fail_closed():
    node = "tests/test_geo_and_drift.py::test_ks_without_scipy_is_unassessed"
    assert not accepted_skip(
        node, "checks the behaviour without the stats extra", {"capabilities": None}
    )


def test_invalid_core_environment_blocks_without_crashing():
    record = {
        "schema_version": 1,
        "id": "core-setup",
        "kind": "installation",
        "status": "passed",
        "identity": IDENTITY,
        "mode": "core",
        "wheel_sha256": "a" * 64,
        "qavf_import_path": "/venv/qavf/__init__.py",
        "verification_modes": [0, 1],
        "environment": None,
    }
    assert evidence_errors([record], IDENTITY, {"core-setup"})


def test_extension_bootstrap_uses_matching_https_signed_local_install(tmp_path, monkeypatch):
    import gzip
    import io

    from scripts.release_checks import extensions

    payload = b"synthetic extension payload for transport contract"
    statements, urls, installed_paths = [], [], []

    class Response(io.BytesIO):
        def geturl(self):
            return "https://extensions.duckdb.org/binary"

    class Connection:
        def execute(self, statement):
            statements.append(statement)
            if statement.startswith("INSTALL '"):
                path = Path(statement[9:-1].replace("''", "'"))
                assert path.read_bytes() == payload
                installed_paths.append(path)
            return self

        def fetchone(self):
            return ("v1.2.0" if statements[-1] == "SELECT version()" else "linux_amd64_gcc4",)

    def download(url, *, timeout):
        urls.append(url)
        assert timeout == 30
        return Response(gzip.compress(payload))

    monkeypatch.setattr(extensions.urllib.request, "urlopen", download)
    extensions.bootstrap_httpfs(Connection())
    assert urls == [
        "https://extensions.duckdb.org/v1.2.0/linux_amd64_gcc4/httpfs.duckdb_extension.gz"
    ]
    assert statements[-1] == "LOAD httpfs"
    assert not any("unsigned" in statement.lower() for statement in statements)
    assert all(not path.exists() for path in installed_paths)


def test_extension_bootstrap_preserves_download_failure(monkeypatch):
    import urllib.error

    from scripts.release_checks import extensions

    class Connection:
        def execute(self, statement):
            self.statement = statement
            return self

        def fetchone(self):
            return ("v1.2.0" if self.statement == "SELECT version()" else "linux_amd64_gcc4",)

    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError("https://extensions.duckdb.org", 403, "Forbidden", {}, None)

    monkeypatch.setattr(extensions.urllib.request, "urlopen", unavailable)
    with pytest.raises(urllib.error.HTTPError, match="403"):
        extensions.bootstrap_httpfs(Connection())


def test_release_environment_forces_reproducible_hypothesis_profile(monkeypatch):
    monkeypatch.setenv("QAVF_HYPOTHESIS_PROFILE", "external-weak-profile")
    assert clean_environment()["QAVF_HYPOTHESIS_PROFILE"] == "ci"
