"""Narrow, auditable exceptions for inapplicable test branches."""

from __future__ import annotations

import importlib.util
import platform
from typing import Any

# Tests of one host's own memory API: skipped, as intended, everywhere else.
# Maps (node, skip reason) to the hosts on which the skip is not accepted.
_HOST_ONLY_TESTS: dict[tuple[str, str], set[str]] = {
    (
        "tests/test_resource_planning.py::test_windows_reports_available_memory",
        "Windows memory API",
    ): {"Windows"},
    (
        "tests/test_resource_planning.py::"
        "test_cgroup_reclaimable_cache_is_not_treated_as_live_allocations",
        "Linux /proc and cgroup files",
    ): {"Linux"},
    (
        "tests/test_resource_planning.py::test_macos_reports_physical_memory",
        "macOS physical memory",
    ): {"Darwin"},
}


def accepted_skip(nodeid: str, reason: str, env: dict[str, Any] | None = None) -> bool:
    """Extensions and live services are deliberately absent from this allowlist."""
    system = env.get("system") if env is not None else platform.system()
    capabilities = env.get("capabilities", {}) if env is not None else {}
    if not isinstance(capabilities, dict):
        return False
    reason = reason.removeprefix("Skipped: ")
    if (
        nodeid
        == (
            "tests/test_performance_runner.py::"
            "test_real_windows_timeout_reaps_grandchild_and_releases_files"
        )
        and reason == "real Windows Job Object contract"
        and system in {"Linux", "Darwin"}
    ):
        return True
    if (nodeid, reason) in _HOST_ONLY_TESTS:
        return system not in _HOST_ONLY_TESTS[nodeid, reason]
    if nodeid.startswith("tests/test_isolated_supervisor.py::test_") and (
        reason,
        system,
    ) in {
        # procfs group probes are Linux internals; other hosts use their own probes.
        ("Linux OS enforcement", "Windows"),
        ("Linux OS enforcement", "Darwin"),
        # Symlinks, FIFOs and rlimits are absent; Windows rejects file_size first.
        ("POSIX OS enforcement", "Windows"),
        # macOS rejects address_space before launch, which a public test exercises.
        ("macOS does not enforce address space", "Darwin"),
    }:
        return True
    if (
        nodeid == "tests/test_geo_and_drift.py::test_ks_without_scipy_is_unassessed"
        and reason == "checks the behaviour without the stats extra"
    ):
        return (
            capabilities.get("scipy") is True
            if env is not None
            else importlib.util.find_spec("scipy") is not None
        )
    if system == "Windows":
        if (
            nodeid.startswith(
                "tests/test_review_boundaries.py::test_cli_never_replaces_selected_input["
            )
            and reason == "Windows filenames forbid these characters"
        ):
            return True
        if (
            nodeid
            == ("tests/test_review_boundaries.py::test_protection_preserves_file_aliases[symlink]")
            and reason == "Windows symlink privileges unavailable"
        ):
            return True
        if (
            nodeid.startswith(
                "tests/test_source_expressions.py::test_literal_forms_resolve_to_configuration_root["
            )
            and reason == "Windows forbids control characters in literal file names"
        ):
            return True
        if (
            nodeid.startswith(
                "tests/test_loading_native_binding.py::"
                "test_generated_reader_literals_preserve_hostile_and_glob_filenames["
            )
            and reason == "Windows forbids '?', '*' and newlines in file names"
        ):
            return True
        if (
            nodeid.startswith(
                "tests/test_input_path_identity.py::"
                "test_facade_reads_exact_padded_path_instead_of_valid_trimmed_neighbor["
            )
            and reason == "Filesystem does not distinguish this whitespace-padded filename"
        ):
            # Win32 path normalization strips trailing spaces from file names.
            return True
        if nodeid.startswith(
            "tests/test_input_identity.py::test_expanded_manifest_names_are_literal[a"
        ) and reason in {
            "Windows forbids '*' in literal file names",
            "Windows forbids '?' in literal file names",
        }:
            return True
    if (
        nodeid.startswith(
            "tests/test_report_encoding.py::test_numpy_nonprogress_scalar_is_explicitly_rejected["
        )
        and reason == "this platform converts the dtype to a Python scalar"
    ):
        name = nodeid.rsplit("[", 1)[-1].removesuffix("]")
        if env is not None:
            dtypes = capabilities.get("numpy_item_nonprogress", {})
            return isinstance(dtypes, dict) and dtypes.get(name) is False
        import numpy as np

        kinds = {"longdouble": np.longdouble, "clongdouble": np.clongdouble}
        kind = kinds.get(name)
        return kind is not None and not isinstance(kind("1.5").item(), np.generic)
    return False


def unexpected_outcomes(outcomes: list[dict]) -> list[dict]:
    return [
        item
        for item in outcomes
        if item["outcome"] in {"failed", "error", "xfailed", "xpassed"}
        or (item["outcome"] == "skipped" and not item.get("accepted", False))
    ]
