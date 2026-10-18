"""Validate a complete, matching set of release evidence; missing is never passing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import repository_identity, timestamp, write_json
from .validation import environment_errors, outcome_errors

REQUIRED_JOBS = {
    "test",
    "dependency-floors",
    "platform-contracts",
    "core-installation",
    "integrations",
}


def required_checks() -> set[str]:
    required = {"core-setup", "integrations"}
    for label in ("linux-3.12", "linux-3.13", "linux-3.14", "windows-3.12", "macos-3.12"):
        for mode in ("source", "wheel"):
            required.update({f"{label}-{mode}", f"{label}-{mode}-optimized"})
        required.add(f"{label}-wheel-setup")
    required.update({"floors", "floors-optimized", "floors-setup"})
    return required


def evidence_errors(
    records: list[dict[str, Any]],
    identity: dict[str, Any],
    expected: set[str],
) -> list[str]:
    errors: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            grouped.setdefault(record["id"], []).append(record)
    for check_id in sorted(expected):
        matches = grouped.get(check_id, [])
        if len(matches) != 1:
            errors.append(f"{check_id}: expected exactly one evidence record; found {len(matches)}")
            continue
        record = matches[0]
        if record.get("schema_version") != 1 or record.get("status") != "passed":
            errors.append(f"{check_id}: evidence is not a passing supported record")
        recorded_identity = record.get("identity", {})
        if not isinstance(recorded_identity, dict):
            recorded_identity = {}
        for field in (
            "source_sha256",
            "lock_sha256",
            "git_revision",
            "ci_run_id",
            "ci_run_attempt",
        ):
            if recorded_identity.get(field) != identity.get(field):
                errors.append(f"{check_id}: mismatched {field}")
        expected_kind = "installation" if check_id.endswith("-setup") else "pytest"
        if record.get("kind") != expected_kind:
            errors.append(f"{check_id}: wrong evidence kind; expected {expected_kind}")
            continue
        if expected_kind == "pytest":
            tests = record.get("tests", {})
            if not isinstance(tests, dict):
                tests = {}
            if (
                record.get("process_exit_code") != 0
                or record.get("source_unchanged") is not True
                or tests.get("status") != "passed"
                or tests.get("exit_code") != 0
                or not isinstance(tests.get("passed"), int)
                or tests.get("passed", 0) <= 0
                or tests.get("collection_errors")
                or tests.get("unexpected_outcomes")
            ):
                errors.append(f"{check_id}: incomplete or unsuccessful test execution")
            errors.extend(f"{check_id}: {error}" for error in outcome_errors(tests))
            if check_id == "linux-3.12-source":
                coverage = record.get("coverage", {})
                if (
                    not isinstance(coverage, dict)
                    or coverage.get("status") != "passed"
                    or not coverage.get("isolated_worker_executed_lines")
                ):
                    errors.append(f"{check_id}: missing branch/subprocess coverage evidence")
            env = tests.get("environment", {})
            if not isinstance(env, dict):
                env = {}
            errors.extend(f"{check_id}: {error}" for error in environment_errors(check_id, env))
            metadata = record.get("metadata", {})
            expected_paths = ["tests/integration"] if check_id == "integrations" else ["tests"]
            expected_selection = "integration" if check_id == "integrations" else "not integration"
            if (
                not isinstance(metadata, dict)
                or metadata.get("test_paths") != expected_paths
                or metadata.get("selection") != expected_selection
            ):
                errors.append(f"{check_id}: wrong test selection")
        else:
            mode = (
                "core"
                if check_id == "core-setup"
                else "floors"
                if check_id == "floors-setup"
                else "locked"
            )
            if record.get("mode") != mode or not record.get("wheel_sha256"):
                errors.append(f"{check_id}: missing wheel identity or wrong installation mode")
            if not record.get("qavf_import_path"):
                errors.append(f"{check_id}: missing installed-wheel import provenance")
            if check_id == "core-setup":
                if record.get("verification_modes") != [0, 1]:
                    errors.append(f"{check_id}: missing normal/optimized core verification")
                env = record.get("environment", {})
                errors.extend(f"{check_id}: {error}" for error in environment_errors(check_id, env))
    for check_id in sorted(expected):
        if "-wheel" not in check_id and not check_id.startswith("floors"):
            continue
        if check_id.endswith("-setup"):
            continue
        base = check_id.removesuffix("-optimized")
        tests = grouped.get(check_id, [])
        setups = grouped.get(base + "-setup", [])
        if len(tests) == len(setups) == 1:
            metadata = tests[0].get("metadata", {})
            if (
                not isinstance(metadata, dict)
                or metadata.get("wheel_sha256") != setups[0].get("wheel_sha256")
                or metadata.get("installation") != setups[0].get("mode")
            ):
                errors.append(f"{check_id}: tested wheel differs from installation evidence")
    return errors


def aggregate(root: Path, directory: Path, output: Path, jobs: dict[str, Any] | None) -> int:
    identity = repository_identity(root)
    errors: list[str] = []
    records = []
    for path in sorted(directory.rglob("*.json")):
        if path.resolve() == output.resolve() or path.name.endswith(".tests.json"):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "id" in value:
                records.append(value)
        except (OSError, ValueError) as exc:
            errors.append(f"Unreadable evidence {path.name}: {exc}")
    errors.extend(evidence_errors(records, identity, required_checks()))
    # CI needs supplies job conclusions independently of artifact contents.
    if jobs is not None:
        for job in sorted(REQUIRED_JOBS):
            value = jobs.get(job, {})
            if not isinstance(value, dict) or value.get("result") != "success":
                errors.append(f"Required CI job {job} did not succeed")
    write_json(
        output,
        {
            "schema_version": 1,
            "status": "passed" if not errors else "blocked",
            "release_ready": not errors,
            "identity": identity,
            "finished": timestamp(),
            "required_checks": sorted(required_checks()),
            "errors": errors,
            "ci_jobs_checked": jobs is not None,
        },
    )
    for error in errors:
        print(error)
    return 1 if errors else 0
