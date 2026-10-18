"""Benchmark conclusions must not hide failed runs or incompatible measurements."""

from copy import deepcopy

import pytest

from scripts.performance.reporting import render_html, summarize_suite


def suite(times=(1.0, 2.0, 3.0), *, baseline=None, warmups=0):
    result = {
        "schema_version": 1,
        "status": "passed",
        "identity_unchanged": True,
        "protocol": {"repetitions": len(times), "warmups": warmups},
        "sources": {"candidate": {}},
        "records": [],
    }
    sources = {"candidate": times}
    if baseline is not None:
        sources["baseline"] = baseline
        result["sources"]["baseline"] = {}
    for label, values in sources.items():
        for is_warmup, durations in ((True, [99.0] * warmups), (False, values)):
            for repetition, seconds in enumerate(durations, 1):
                result["records"].append(
                    {
                        "id": f"{label}-{is_warmup}-{repetition}",
                        "source_label": label,
                        "case": "columns",
                        "rows": 1000,
                        "options": {"threads": 2, "backend": "native"},
                        "repetition": repetition,
                        "warmup": is_warmup,
                        "status": "passed",
                        "environment": {
                            "python_version": "3.12",
                            "python_implementation": "CPython",
                            "dependencies": {"polars": "1.32"},
                            "polars_threads": 2,
                            "native_threads": 2,
                            "qavf_import_path": f"/{label}/qavf/__init__.py",
                        },
                        "phases": {
                            "operation": {
                                "wall_seconds": seconds,
                                "cpu_seconds": seconds,
                                "rss": {"sampled_peak_bytes": 1024},
                                "disk": {"sampled_peak_logical_bytes": 0},
                                "io": {"read_bytes": None},
                            }
                        },
                        "outcome": {"violations": 10},
                        "report_bytes": 200,
                        "process_elapsed_seconds": seconds + 1,
                    }
                )
    return result


def test_medians_dispersion_and_warmups_are_separate():
    group = summarize_suite(suite((1.0, 2.0, 8.0), warmups=1))["groups"][0]
    assert group["status"] == "complete"
    assert group["warmup_count"] == 1
    assert group["metrics"]["operation.wall_seconds"] == {
        "values": [1.0, 2.0, 8.0],
        "available_count": 3,
        "median": 2.0,
        "min": 1.0,
        "max": 8.0,
        "mad": 1.0,
    }
    assert group["metrics"]["report_bytes"]["median"] == 200


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "failed", "warmup_failed", "warmup_missing"]
)
def test_incomplete_records_cannot_be_comparable(change):
    data = suite(baseline=(1, 2, 3), warmups=1)
    if change == "missing":
        data["records"].pop(1)
    elif change == "duplicate":
        data["records"][1]["repetition"] = 2
    elif change == "failed":
        data["records"][1]["status"] = "timed_out"
    elif change == "warmup_failed":
        data["records"][0]["status"] = "failed"
    else:
        data["records"].pop(0)
    summary = summarize_suite(data)
    candidate = next(group for group in summary["groups"] if group["source_label"] == "candidate")
    assert candidate["status"] == "incomplete"
    assert summary["comparisons"][0]["status"] == "inconclusive"
    assert summary["comparisons"][0]["operation_wall_ratio"] is None


@pytest.mark.parametrize(
    "field", ["python_version", "dependencies", "polars_threads", "native_threads"]
)
def test_contradictory_environments_prohibit_comparison(field):
    data = suite(baseline=(1, 2, 3))
    for record in data["records"]:
        if record["source_label"] == "candidate":
            record["environment"][field] = "different"
    comparison = summarize_suite(data)["comparisons"][0]
    assert comparison["status"] == "inconclusive"
    assert comparison["operation_wall_ratio"] is None


def test_source_import_path_can_differ_but_outcome_must_match():
    data = suite((6, 7, 8), baseline=(1, 2, 3))
    comparison = summarize_suite(data)["comparisons"][0]
    assert comparison["status"] == "comparable"
    assert comparison["operation_wall_ratio"] == 3.5
    assert comparison["wall_range_relation"] == "slower"
    for record in data["records"]:
        if record["source_label"] == "candidate":
            record["outcome"] = {"violations": 11}
    assert summarize_suite(data)["comparisons"][0]["status"] == "inconclusive"


def test_noise_and_insufficient_repetitions_remain_inconclusive():
    comparison = summarize_suite(suite((1, 2, 5), baseline=(1, 2, 3)))["comparisons"][0]
    assert comparison["operation_wall_ratio"] == 1
    assert comparison["wall_range_relation"] == "overlap"
    assert "inconclusive" in comparison["advisory"]
    comparison = summarize_suite(suite((8, 9), baseline=(1, 2)))["comparisons"][0]
    assert comparison["status"] == "inconclusive"
    assert comparison["operation_wall_ratio"] is None


def test_unsupported_metrics_remain_null_not_zero():
    group = summarize_suite(suite())["groups"][0]
    metric = group["metrics"]["operation.io.read_bytes"]
    assert metric == {
        "values": [None] * 3,
        "available_count": 0,
        "median": None,
        "min": None,
        "max": None,
        "mad": None,
    }
    assert group["metrics"]["operation.disk.sampled_peak_logical_bytes"]["median"] == 0


def test_partial_metrics_do_not_produce_misleading_ratios():
    data = suite(baseline=(1, 2, 3))
    data["records"][0]["phases"]["operation"]["rss"]["sampled_peak_bytes"] = None
    comparison = summarize_suite(data)["comparisons"][0]
    assert comparison["operation_wall_ratio"] == 1
    assert comparison["rss_peak_ratio"] is None


@pytest.mark.parametrize("change", ["failed", "changed", "unverified"])
def test_failed_final_verification_blocks_comparison_despite_passing_records(change):
    data = suite(baseline=(1, 2, 3))
    if change == "failed":
        data["status"] = "failed"
    elif change == "changed":
        data["identity_unchanged"] = False
    else:
        data.pop("identity_unchanged")
    summary = summarize_suite(data)
    assert all(group["status"] == "complete" for group in summary["groups"])
    assert summary["comparisons"][0]["status"] == "inconclusive"
    assert summary["comparisons"][0]["operation_wall_ratio"] is None


def test_missing_source_records_and_changing_oracle_are_reported():
    data = suite()
    data["sources"]["baseline"] = {}
    assert summarize_suite(data)["comparisons"][0]["status"] == "inconclusive"
    data["records"][0]["outcome"]["violations"] = 9
    assert summarize_suite(data)["groups"][0]["status"] == "incomplete"


def test_html_is_escaped_self_contained_and_retains_failures():
    data = suite()
    data["records"][0].update(status="crashed", error="<script>alert('x')</script>")
    before = deepcopy(data)
    report = render_html(data)
    assert data == before
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert "crashed" in report and "incomplete" in report
    assert "unavailable" in report and "MAD" in report
    assert "1,048,576" in report and "not cold-disk" in report
    assert "https://www.kernel.org/doc/html/latest/filesystems/proc.html" in report


def test_options_group_independently_and_missing_warmups_are_visible():
    data = suite(warmups=1)
    data["records"][1]["options"]["threads"] = 4
    groups = summarize_suite(data)["groups"]
    assert len(groups) == 2
    assert all(group["status"] == "incomplete" for group in groups)


def test_suite_and_cleanup_errors_are_visible_even_when_all_workers_pass():
    data = suite()
    data.update(
        status="failed",
        error={"type": "RuntimeError", "message": "Fixture changed <script>bad</script>"},
        cleanup_error="Could not remove <img src=x onerror=bad> & work directory",
    )
    before = deepcopy(data)
    report = render_html(data)
    assert data == before
    assert "Suite error" in report and "Cleanup error" in report
    assert "Fixture changed &lt;script&gt;bad&lt;/script&gt;" in report
    assert "&lt;img src=x onerror=bad&gt; &amp; work directory" in report
    assert "<script>" not in report and "<img" not in report
    assert "0 failed worker records" in report and "Suite status: failed" in report
    assert "Suite error" not in render_html(suite())


def test_nonzero_exit_keeps_original_worker_oracle_error_in_summary_and_html():
    data = suite()
    original = {"phase": "oracle", "type": "AssertionError", "message": "Expected 9 < observed 10"}
    data["records"][0].update(
        status="crashed", error="Worker exited with code 1", worker_error=original
    )
    failure = summarize_suite(data)["failures"][0]
    assert failure["error"] == "Worker exited with code 1"
    assert failure["worker_error"] == original
    report = render_html(data)
    assert "Worker exited with code 1" in report
    assert "oracle" in report and "AssertionError" in report
    assert "Expected 9 &lt; observed 10" in report


@pytest.mark.parametrize(
    ("baseline_hash", "candidate_hash", "control"),
    [("a" * 64, "a" * 64, True), ("a" * 64, "b" * 64, False), (None, None, False)],
)
def test_matching_production_hashes_label_repeatability_not_code_improvement(
    baseline_hash, candidate_hash, control
):
    data = suite((1, 2, 3), baseline=(6, 7, 8))
    data["sources"]["baseline"]["production_sha256"] = baseline_hash
    data["sources"]["candidate"]["production_sha256"] = candidate_hash
    comparison = summarize_suite(data)["comparisons"][0]
    assert comparison["status"] == "comparable"
    assert comparison["wall_range_relation"] == "faster"
    assert comparison["operation_wall_ratio"] == 2 / 7
    assert (comparison["comparison_kind"] == "repeatability_control") is control
    if control:
        assert "not evidence of a code improvement" in comparison["advisory"]
        assert "repeatability/control" in render_html(data)
    else:
        assert comparison["comparison_kind"] == "source_comparison"
        assert "production digests are identical" not in comparison["advisory"]
