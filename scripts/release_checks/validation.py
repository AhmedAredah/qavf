"""Pure validation of recorded outcomes, independent of pytest's summary counters."""

from __future__ import annotations

from typing import Any

from .policy import accepted_skip


def outcome_errors(tests: dict[str, Any]) -> list[str]:
    errors = []
    selected, outcomes = tests.get("selected"), tests.get("outcomes")
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(node, str) for node in selected)
        or not isinstance(outcomes, list)
        or not outcomes
        or not all(isinstance(item, dict) for item in outcomes)
    ):
        return ["missing or invalid selected tests/outcomes"]
    if len(selected) != len(set(selected)):
        errors.append("duplicate selected tests")
    observed, passed = set(), 0
    env = tests.get("environment", {})
    if not isinstance(env, dict):
        return ["missing environment"]
    for item in outcomes:
        node, outcome = item.get("nodeid"), item.get("outcome")
        if not isinstance(node, str):
            errors.append("invalid test node identifier")
            continue
        if node in observed:
            errors.append(f"duplicate terminal outcome: {node}")
        observed.add(node)
        if outcome == "passed" and item.get("phase") == "call":
            passed += 1
        elif (
            outcome == "skipped"
            and isinstance(item.get("reason"), str)
            and accepted_skip(node, item["reason"], env)
        ):
            continue
        else:
            errors.append(f"unexpected outcome: {node}: {outcome}")
    if observed != set(selected):
        errors.append("selected tests do not match completed outcomes")
    if not passed or passed != tests.get("passed"):
        errors.append("invalid passed-test count")
    return errors


def environment_errors(check_id: str, env: dict[str, Any]) -> list[str]:
    if not isinstance(env, dict):
        return ["missing or invalid environment"]
    parts = check_id.split("-")
    system = {"linux": "Linux", "windows": "Windows", "macos": "Darwin"}.get(parts[0], "Linux")
    version = parts[1] if parts[0] in {"linux", "windows", "macos"} else "3.12"
    errors = []
    if env.get("system") != system or not str(env.get("python_version", "")).startswith(
        version + "."
    ):
        errors.append("wrong operating system or Python version")
    expected_opt = 1 if check_id.endswith("-optimized") else 0
    if env.get("optimization") != expected_opt:
        errors.append("wrong interpreter optimization mode")
    if not env.get("qavf_import_path") or not env.get("packages"):
        errors.append("missing imported package provenance")
    return errors
