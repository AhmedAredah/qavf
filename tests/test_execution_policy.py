"""Execution selection must never silently weaken a requested hard policy."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from qavf import DataContext, QARunner, QAVFSettings, ResourceLimits
from qavf.engine.registry import RuleRegistry
from qavf.execution.errors import ExecutionLimitError
from qavf.execution.protocol import bounded_chunks, read_request, request_bytes, write_result
from qavf.inputs.descriptors import Query


@pytest.mark.parametrize("field,value,expected", [
    ("wall_time", 0.5, "run_timeout_s"),
    ("address_space", "4GiB", "process_address_space_bytes"),
    ("cpu_time", 2, "process_cpu_time_s"),
    ("file_size", 0, "process_file_bytes"),
])
def test_hard_controls_select_isolation(field, value, expected):
    overrides = ResourceLimits(**{field: value}).engine_overrides()
    assert overrides["execution_mode"] == "isolated"
    assert expected in overrides
    with pytest.raises(ValueError, match="require"):
        ResourceLimits(isolate=False, **{field: value})


@pytest.mark.parametrize("values", [
    {"wall_time": True}, {"wall_time": 0}, {"wall_time": float("nan")},
    {"wall_time": float("inf")}, {"wall_time": "3"}, {"cpu_time": 1.5},
    {"cpu_time": True}, {"cpu_time": 0}, {"address_space": 0},
    {"address_space": True}, {"file_size": -1}, {"isolate": "yes"},
])
def test_invalid_public_limits_rejected(values):
    with pytest.raises(ValueError):
        ResourceLimits(**values)


@pytest.mark.parametrize("where", ["constructor", "run", "context", "direct_context"])
def test_mutated_hard_settings_cannot_run_without_supervision(where, monkeypatch):
    import qavf.engine.runner as runner_module

    settings = QAVFSettings.model_validate({"plugins": {"entry_points": False}})
    runner = QARunner(settings)
    settings.engine.run_timeout_s = 1

    def forbidden(*args, **kwargs):
        pytest.fail("plugin discovery occurred before invalid policy rejection")

    monkeypatch.setattr(runner_module, "build_registry", forbidden)
    with pytest.raises(ValueError, match="require"):
        if where == "constructor":
            QARunner(settings)
        elif where == "run":
            runner.run()
        elif where == "context":
            runner.context()
        else:
            DataContext.from_settings(settings)


def test_mutated_mode_cannot_transport_a_supplied_registry():
    settings = QAVFSettings()
    runner = QARunner(settings, registry=RuleRegistry())
    settings.engine.execution_mode = "isolated"
    with pytest.raises(ValueError, match="configured plugins"):
        runner.run()


def test_direct_runner_path_override_is_rejected_before_execution(tmp_path):
    settings = QAVFSettings.model_validate({
        "engine": {"execution_mode": "isolated"},
        "datasets": {"d": {"path": "original.parquet"}},
    })
    with pytest.raises(ValueError, match="validate"):
        QARunner(settings).run(frames={"d": tmp_path / "replacement.parquet"})


def test_request_preserves_tagged_values_and_policy_omission(tmp_path):
    settings = QAVFSettings.model_validate({
        "engine": {"run_timeout_s": 3},
        "datasets": {"d": {}},
        "rules": {"CUSTOM-001": {"enabled": False}},
    })
    settings.source_path = tmp_path / "config.json"
    request = request_bytes(settings, {"d": Query(
        "select $v as x", parameters={"v": Decimal("1.234567890123456789")}
    )})
    path = tmp_path / "request.json"
    path.write_bytes(request)
    restored, inputs, options = read_request(path)
    assert restored.model_dump() == settings.model_dump()
    assert "params" not in restored.rules["CUSTOM-001"].model_fields_set
    assert restored.source_path == settings.source_path
    assert inputs["d"].parameters["v"] == Decimal("1.234567890123456789")
    assert options == {}


def test_exact_transport_cap_and_atomic_failure(tmp_path):
    assert b"".join(bounded_chunks([b"12", b"3"], 3)) == b"123"
    target = tmp_path / "result.json"
    with pytest.raises(ExecutionLimitError, match="exceeds"):
        write_result(target, [b"12", b"34"], 3)
    assert not target.exists()
    assert not target.with_suffix(".partial").exists()


def test_unknown_protocol_version_cannot_load_settings(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ExecutionLimitError, match="version"):
        read_request(path)


@pytest.mark.parametrize("system,accepted", [
    ("Linux", False), ("Windows", True), ("Darwin", True), (None, False),
])
def test_linux_only_release_exception_is_exact(system, accepted):
    from scripts.release_checks.policy import accepted_skip

    node = "tests/test_isolated_supervisor.py::test_worker_timeout"
    assert accepted_skip(node, "Linux OS enforcement", {"system": system}) is accepted
    assert not accepted_skip(node, "extension unavailable", {"system": system})
    assert not accepted_skip("tests/test_sources.py::test_file", "Linux OS enforcement", {
        "system": system
    })


@pytest.mark.parametrize(("reason", "system", "accepted"), [
    ("POSIX OS enforcement", "Windows", True),
    ("POSIX OS enforcement", "Darwin", False),
    ("POSIX OS enforcement", "Linux", False),
    ("macOS does not enforce address space", "Darwin", True),
    ("macOS does not enforce address space", "Windows", False),
    ("macOS does not enforce address space", "Linux", False),
])
def test_platform_isolation_release_exceptions_are_exact(reason, system, accepted):
    _check_exact_exception(
        "tests/test_isolated_supervisor.py::test_os_file_size_rejects_oversized_worker_output",
        reason, system, accepted,
    )


@pytest.mark.parametrize(
    "system,accepted", [("Windows", True), ("Darwin", False), ("Linux", False)]
)
@pytest.mark.parametrize(("node", "reason"), [
    (
        "tests/test_loading_native_binding.py::"
        "test_generated_reader_literals_preserve_hostile_and_glob_filenames[x-csv-context]",
        "Windows forbids '?', '*' and newlines in file names",
    ),
    (
        "tests/test_input_path_identity.py::"
        "test_facade_reads_exact_padded_path_instead_of_valid_trimmed_neighbor[x-raw]",
        "Filesystem does not distinguish this whitespace-padded filename",
    ),
])
def test_windows_filename_release_exceptions_are_exact(node, reason, system, accepted):
    _check_exact_exception(node, reason, system, accepted)


def _check_exact_exception(node, reason, system, accepted):
    from scripts.release_checks.policy import accepted_skip

    assert accepted_skip(node, reason, {"system": system}) is accepted
    assert accepted_skip(node, "Skipped: " + reason, {"system": system}) is accepted
    assert not accepted_skip("tests/test_sources.py::test_file", reason, {"system": system})
