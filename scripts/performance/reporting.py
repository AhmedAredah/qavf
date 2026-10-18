"""Pure summaries and a portable HTML view of independently verified measurements."""

from __future__ import annotations

import html
import json
import math
import statistics
from collections import defaultdict
from typing import Any

PHASE_FIELDS = [
    "wall_seconds",
    "cpu_seconds",
    "waited_tree_cpu_seconds",
    "process_tree.sampled_peak_rss_bytes",
    "process_tree.sampled_peak_processes",
    "process_tree.partial_samples",
    "rss.sampled_start_bytes",
    "rss.sampled_peak_bytes",
    "rss.sampled_end_bytes",
    "rss.lifetime_high_water_start_bytes",
    "rss.lifetime_high_water_end_bytes",
    "disk.start_logical_bytes",
    "disk.start_allocated_bytes",
    "disk.sampled_peak_logical_bytes",
    "disk.sampled_peak_allocated_bytes",
    "disk.final_logical_bytes",
    "disk.final_allocated_bytes",
    "io.read_bytes",
    "io.write_bytes",
    "io.cancelled_write_bytes",
    "io.rchar",
    "io.wchar",
    "sampling.sample_count",
    "sampling.overhead_wall_seconds",
    "sampling.overhead_cpu_seconds",
    "sampling.sample_failures",
    "sampling.partial_disk_samples",
]
METRICS = [
    f"{phase}.{field}" for phase in ("setup", "operation", "report") for field in PHASE_FIELDS
]
METRICS.extend(("process_elapsed_seconds", "report_bytes"))
REFERENCES = (
    ("Python clocks", "https://docs.python.org/3/library/time.html#time.perf_counter"),
    ("Python resource usage", "https://docs.python.org/3/library/resource.html#resource.getrusage"),
    ("Linux process accounting", "https://www.kernel.org/doc/html/latest/filesystems/proc.html"),
    (
        "Polars thread pool",
        "https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html",
    ),
    ("Polars lazy execution", "https://docs.pola.rs/user-guide/concepts/lazy-api/"),
    (
        "DuckDB workload tuning",
        "https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads",
    ),
    ("DuckDB benchmark correctness", "https://duckdb.org/docs/current/dev/benchmark"),
    (
        "pyperf measurement methodology",
        "https://pyperf.readthedocs.io/en/latest/run_benchmark.html",
    ),
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(record: dict[str, Any]) -> tuple[str, int, str]:
    return record["case"], record["rows"], _canonical(record["options"])


def _environment(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.get("environment", {}).items()
        if key != "qavf_import_path"
    }


def _metric(record: dict[str, Any], path: str) -> float | int | None:
    value: Any = (
        record if path in {"process_elapsed_seconds", "report_bytes"} else record.get("phases", {})
    )
    for key in path.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return (
        value
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def _statistics(values: list[float | int | None]) -> dict[str, Any]:
    available = [value for value in values if value is not None]
    center = statistics.median(available) if available else None
    return {
        "values": values,
        "available_count": len(available),
        "median": center,
        "min": min(available) if available else None,
        "max": max(available) if available else None,
        "mad": statistics.median(abs(value - center) for value in available)
        if center is not None
        else None,
    }


def _group(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    first = records[0]
    measured = [record for record in records if not record["warmup"]]
    warmups = [record for record in records if record["warmup"]]
    passed = [record for record in measured if record["status"] == "passed"]
    all_passed = [record for record in records if record["status"] == "passed"]
    failures = [record["id"] for record in records if record["status"] != "passed"]
    issues = []
    for label, rows, expected in (
        ("Measured", measured, protocol["repetitions"]),
        ("Warmup", warmups, protocol.get("warmups", 0)),
    ):
        if sorted(record["repetition"] for record in rows) != list(range(1, expected + 1)):
            issues.append(
                f"{label} repetitions are missing, duplicated, or unexpected (expected {expected})."
            )
    if failures:
        issues.append(f"{len(failures)} failed, timed out, or crashed records, including warmups.")
    outcomes = {_canonical(record.get("outcome")) for record in all_passed}
    environments = {_canonical(_environment(record)) for record in all_passed}
    if len(outcomes) != 1 or any(
        not isinstance(record.get("outcome"), dict) for record in all_passed
    ):
        issues.append("Independent oracle outcomes are missing or inconsistent.")
    if len(environments) != 1:
        issues.append("Worker environments are inconsistent or missing.")
    environment = _environment(all_passed[0]) if all_passed else {}
    required = set(
        [
            "python_version",
            "python_implementation",
            "dependencies",
            "polars_threads",
            "native_threads",
        ]
    )
    if not required.issubset(environment):
        issues.append("Worker environment identity is incomplete.")
    return {
        "case": first["case"],
        "rows": first["rows"],
        "options": first["options"],
        "source_label": first["source_label"],
        "status": "incomplete" if issues else "complete",
        "expected_repetitions": protocol["repetitions"],
        "measured_count": len(passed),
        "warmup_count": len(warmups),
        "warmup_failures": sum(row["status"] != "passed" for row in warmups),
        "failed_records": failures,
        "issues": issues,
        "metrics": {
            name: _statistics([_metric(record, name) for record in passed]) for name in METRICS
        },
        "outcomes_consistent": len(outcomes) == 1,
        "environment_consistent": len(environments) == 1,
        "outcome": all_passed[0].get("outcome") if len(outcomes) == 1 else None,
        "environment": environment,
    }


def _compare(groups: dict[str, dict[str, Any]], suite_issues: list[str]) -> dict[str, Any]:
    first = next(iter(groups.values()))
    result: dict[str, Any] = {key: first[key] for key in ("case", "rows", "options")}
    reasons = list(suite_issues)
    baseline, candidate = groups.get("baseline"), groups.get("candidate")
    if baseline is None or candidate is None:
        reasons.append("Baseline or candidate measurements are missing.")
    else:
        for name, group in (("baseline", baseline), ("candidate", candidate)):
            if group["status"] != "complete":
                reasons.append(f"{name.capitalize()} group is incomplete.")
            if group["measured_count"] < 3:
                reasons.append(f"{name.capitalize()} has fewer than three measured repetitions.")
        if baseline["outcome"] != candidate["outcome"]:
            reasons.append("Baseline and candidate independent oracle outcomes differ.")
        if baseline["environment"] != candidate["environment"]:
            reasons.append("Python, dependency, platform, or thread environments differ.")
    result.update(status="inconclusive" if reasons else "comparable", reasons=reasons)
    for label, metric in (
        ("operation_wall", "operation.wall_seconds"),
        ("operation_cpu", "operation.cpu_seconds"),
        ("rss_peak", "operation.rss.sampled_peak_bytes"),
    ):
        left = baseline["metrics"][metric] if baseline else {}
        right = candidate["metrics"][metric] if candidate else {}
        complete = (
            baseline
            and candidate
            and left.get("available_count") == baseline["measured_count"]
            and right.get("available_count") == candidate["measured_count"]
        )
        ratio = (
            right["median"] / left["median"]
            if not reasons and complete and left.get("median", 0) > 0
            else None
        )
        result[f"{label}_ratio"] = ratio
    result["wall_range_relation"] = "unavailable"
    if (
        result["operation_wall_ratio"] is not None
        and baseline is not None
        and candidate is not None
    ):
        left, right = (
            baseline["metrics"]["operation.wall_seconds"],
            candidate["metrics"]["operation.wall_seconds"],
        )
        relation = (
            "slower"
            if right["min"] > left["max"]
            else "faster"
            if right["max"] < left["min"]
            else "overlap"
        )
        result["wall_range_relation"] = relation
        result["advisory"] = (
            "Observed ranges overlap; timing direction is inconclusive."
            if relation == "overlap"
            else f"Candidate observed range is {relation}; repeat on a controlled host "
            "before a performance decision."
        )
    else:
        result["advisory"] = (
            "Inconclusive: incomplete, incompatible, insufficient, or unavailable measurements."
        )
    return result


def summarize_suite(suite: dict[str, Any]) -> dict[str, Any]:
    """Summarize passed measurements without hiding failed or missing repetitions."""
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in suite.get("records", []):
        grouped[(*_signature(record), record["source_label"])].append(record)
    groups = [_group(rows, suite["protocol"]) for _, rows in sorted(grouped.items())]
    paired: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for group in groups:
        paired[_signature(group)][group["source_label"]] = group
    suite_issues = []
    if suite.get("status") != "passed":
        suite_issues.append("Suite did not pass its final verification.")
    if suite.get("identity_unchanged") is not True:
        suite_issues.append("Unchanged fixture, source, and benchmark identities are not verified.")
    comparisons = (
        [_compare(pair, suite_issues) for pair in paired.values()]
        if "baseline" in suite.get("sources", {})
        else []
    )
    sources = suite.get("sources", {})
    baseline_hash = sources.get("baseline", {}).get("production_sha256")
    candidate_hash = sources.get("candidate", {}).get("production_sha256")
    same_production = bool(baseline_hash) and baseline_hash == candidate_hash
    for comparison in comparisons:
        comparison["comparison_kind"] = (
            "repeatability_control" if same_production else "source_comparison"
        )
        if same_production:
            comparison["advisory"] = (
                "Recorded production digests are identical: this is a repeatability/control "
                "measurement, not evidence of a code improvement. " + comparison["advisory"]
            )
    failure_fields = [
        "id",
        "source_label",
        "case",
        "rows",
        "repetition",
        "warmup",
        "status",
        "error",
        "worker_error",
    ]
    failures = [
        {key: record.get(key) for key in failure_fields}
        for record in suite.get("records", [])
        if record["status"] != "passed"
    ]
    return {"groups": groups, "comparisons": comparisons, "failures": failures}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _display(group: dict[str, Any], name: str, *, mib: bool = False) -> str:
    summary = group["metrics"][name]
    median = summary["median"]
    if median is None:
        return "unavailable"
    scale = 1024**2 if mib else 1
    count = summary["available_count"]
    suffix = (
        f" ({count}/{group['measured_count']} available)"
        if count != group["measured_count"]
        else ""
    )
    return f"{median / scale:,.3f}{suffix}"


def _table(headers: tuple[str, ...], rows: list[list[str]]) -> str:
    return (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{_escape(name)}</th>" for name in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def render_html(suite: dict[str, Any]) -> str:
    """Render a self-contained, script-free report; raw records remain in JSON."""
    summary = summarize_suite(suite)
    groups = summary["groups"]
    complete = sum(group["status"] == "complete" for group in groups)
    issues = [
        f"{group['case']} / {group['source_label']}: {issue}"
        for group in groups
        for issue in group["issues"]
    ]
    comparisons = [
        [
            f"{row['case']} · {row['rows']:,} rows",
            _canonical(row["options"]),
            row["comparison_kind"],
            row["status"],
            f"{row['operation_wall_ratio']:.3f}x"
            if row["operation_wall_ratio"] is not None
            else "unavailable",
            row["advisory"] + " " + " ".join(row["reasons"]),
        ]
        for row in summary["comparisons"]
    ]
    failures = [
        [
            str(row[key])
            for key in ("id", "source_label", "case", "warmup", "status", "error", "worker_error")
        ]
        for row in summary["failures"]
    ]
    fields = [
        "schema_version",
        "status",
        "protocol",
        "sources",
        "benchmark_sha256",
        "host",
        "fixtures",
    ]
    details = {key: suite.get(key) for key in fields}
    status = (
        f"{complete}/{len(groups)} complete groups; {len(failures)} failed worker records. "
        f"Suite status: {suite.get('status', 'unknown')}."
    )
    sections = [
        "<h2>Validation and measurement status</h2><p>" + _escape(status) + "</p>",
    ]
    for label, key in (("Suite error", "error"), ("Cleanup error", "cleanup_error")):
        if key in suite and suite[key] is not None:
            message = _escape(json.dumps(suite[key], indent=2, sort_keys=True, allow_nan=False))
            sections.append(f"<h3>{label}</h3><pre>{message}</pre>")
    sections.extend(
        (
            "<ul>" + "".join(f"<li>{_escape(issue)}</li>" for issue in issues) + "</ul>",
            _table(
                (
                    "Record",
                    "Source",
                    "Case",
                    "Warmup",
                    "Status",
                    "Controller error",
                    "Worker error",
                ),
                failures,
            ),
        )
    )
    sections.append("""<h2>Measurements</h2><p>Passed measured repetitions only. Values are medians,
with dispersion recorded as MAD (median absolute deviation) in JSON summaries. Times are seconds;
MiB = 1,048,576 bytes. Incomplete groups are not passing evidence. Setup, operation and report
phases remain separate. Process elapsed includes worker setup and verification.</p>""")
    phase_columns = (
        ("Setup s", "setup.wall_seconds", False),
        ("Operation wall s", "operation.wall_seconds", False),
        ("Operation CPU s", "operation.cpu_seconds", False),
        ("Report s", "report.wall_seconds", False),
        ("Process elapsed s", "process_elapsed_seconds", False),
    )
    resource_columns = (
        ("Tree RSS MiB", "operation.process_tree.sampled_peak_rss_bytes", True),
        ("Operation RSS MiB", "operation.rss.sampled_peak_bytes", True),
        ("Lifetime RSS after report MiB", "report.rss.lifetime_high_water_end_bytes", True),
        ("Disk logical MiB", "operation.disk.sampled_peak_logical_bytes", True),
        ("Disk allocated MiB", "operation.disk.sampled_peak_allocated_bytes", True),
        ("Storage read MiB", "operation.io.read_bytes", True),
        ("Storage write MiB", "operation.io.write_bytes", True),
        ("Report bytes", "report_bytes", False),
    )
    for columns in (phase_columns, resource_columns):
        rows = []
        for group in groups:
            identity = f"{group['case']} / {group['source_label']} / {group['rows']:,} rows"
            state = f"{group['status']}: {group['measured_count']}/{group['expected_repetitions']}"
            values = [_display(group, metric, mib=mib) for _, metric, mib in columns]
            rows.append([identity, _canonical(group["options"]), state, *values])
        headers = ("Workload", "Options", "Passed / expected", *(item[0] for item in columns))
        sections.append(_table(headers, rows))
    sections.append("""<p>Operation disk peaks cover owned scratch files; zero storage reads may
reflect OS caching. Report size is encoded output bytes. Lifetime RSS includes imports and all
earlier work, not phase allocations. Raw minimum, maximum and MAD remain in JSON summaries.</p>
<h2>Source comparison</h2><p>Ratios are candidate / baseline. Eligibility requires complete
matching workloads, identical oracle outcomes, compatible environments, and at least three
measurements each. Range separation is advisory, not statistical significance. An empty table
means no baseline comparison was requested.</p>""")
    sections.append(
        _table(
            ("Workload", "Options", "Purpose", "Eligibility", "Wall ratio", "Interpretation"),
            comparisons,
        )
    )
    sections.append("""<h2>Protocol and limits</h2><ul>
<li>Fresh workers isolate process state; warmups are excluded from summaries but their failures
invalidate a group. OS caches remain uncontrolled: these are not cold-disk measurements.</li>
<li>Wall time uses perf_counter; CPU includes all process threads and the monitor. Sampling adds
interference; its overlapping overhead must not be subtracted.</li>
<li>Sampled RSS and disk peaks are lower bounds and may miss transient peaks. Linux RSS is
approximate; unsupported observations stay null, displayed as unavailable.</li>
<li>Disk logical size differs from allocated blocks. Only distinct regular files under owned
folders are counted, excluding input fixtures, symlinks and directory metadata. Partial snapshots
during deletion are disclosed in raw metrics.</li>
<li>Linux rchar/wchar include cached reads and monitor activity. read_bytes/write_bytes are
kernel-accounted storage activity; writes are accounted before final writeback. These counters
are not device throughput. Cancelled writes are not subtracted.</li>
<li>DuckDB memory limits constrain its buffer manager, not total RSS. Synthetic cases exercise
specified behaviors; conclusions apply to the recorded fixture, host and dependencies.</li>
<li>The companion JSON retains raw repetitions, null capabilities, sampler diagnostics and
QAVF cache/hash telemetry.</li></ul>""")
    sections.extend(
        (
            "<details><summary>Exact protocol, source identities, host and fixture "
            "footprints</summary><pre>"
            + _escape(json.dumps(details, indent=2, sort_keys=True, allow_nan=False))
            + "</pre></details>",
            "<h2>Primary references</h2><ul>",
            "".join(
                f"<li><a href='{_escape(url)}'>{_escape(label)}</a></li>"
                for label, url in REFERENCES
            ),
            "</ul>",
        )
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QAVF realistic performance benchmarks</title><style>
body{margin:0;background:#f2f5f8;color:#172b3a;font:15px/1.6 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:40px 28px}
header{border-top:7px solid #087f8c;padding:28px;background:#fff;border-radius:8px}
h1{margin:0;font-size:32px}h2{margin:34px 0 12px;font-size:21px}p{max-width:1050px}
a{color:#096b84}.scroll{overflow:auto;border:1px solid #d4dee7;border-radius:8px;background:white}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:12px 14px;text-align:left;border-bottom:1px solid #e4eaf0;vertical-align:top}
th{background:#e8f0f4;color:#17374b}tr:nth-child(even){background:#f8fafc}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#fff;padding:18px;font-size:12px}
details{margin-top:25px}summary{cursor:pointer;font-weight:600}li{margin:7px 0}
@media print{body{background:white}main{padding:0}.scroll{overflow:visible}table{font-size:10px}}
</style><main><header><h1>QAVF realistic performance benchmarks</h1>
<p>Workload measurements with independent correctness checks, explicit resource scope,
and durable failure evidence.</p></header>"""
        + "".join(sections)
        + "</main></html>"
    )
