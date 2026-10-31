"""Public-path temporal, spatial, dirty-file and report workloads with exact truth.

Each fixture is generated in bounded chunks. Oracles use fixture arithmetic,
not a second invocation of a QAVF kernel. These cases complement the capacity
and streaming families; they do not claim exhaustive workload coverage.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import polars as pl

from qavf import Dataset, File, Status, compare_datasets, validate
from qavf.reports.html import write_html
from qavf.reports.models import QAReport, RuleResult, RunMetadata
from qavf.reports.presentation import HtmlLimits

from .models import PreparedWorkload, WorkloadOptions
from .policy import resources
from .storage import read_json, write_json, write_parquet_chunks


def _fixture_frames(rows: int, chunk_rows: int, candidate: bool) -> Iterator[pl.DataFrame]:
    """One side's chunks; the candidate shifts every 101st row's value and location."""
    for start in range(0, rows, chunk_rows):
        frame = pl.DataFrame({"id": range(start, min(rows, start + chunk_rows))})
        key = pl.col("id")
        changed = (key % 101 == 0).cast(pl.Int64) * int(candidate)
        yield frame.with_columns(
            (key * 60_000_000).cast(pl.Datetime("us")).alias("time"),
            (key * 2 + changed).alias("value"),
            ((key % 10) + changed * 3).cast(pl.Float64).alias("x"),
            ((key // 10) % 10 + changed * 4).cast(pl.Float64).alias("y"),
        )


def prepare_fixture(
    directory: Path, *, rows: int, seed: int, chunk_rows: int = 65536, **_options: Any
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    for candidate in (False, True):
        write_parquet_chunks(
            directory / f"{'candidate' if candidate else 'baseline'}.parquet",
            _fixture_frames(rows, chunk_rows, candidate),
        )
    with (directory / "dirty.csv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("id,value\n")
        for start in range(0, rows, chunk_rows):
            stream.writelines(
                f"{index},{index * 2}" + (",extra\n" if index % 101 == 0 else "\n")
                for index in range(start, min(rows, start + chunk_rows))
            )
    manifest = {"rows": rows, "seed": seed, "fixture_version": 1}
    write_json(directory / "manifest.json", manifest)
    return manifest


def _run_comparison(fixture: Path, level: str | dict[str, Any], policy: Any) -> Any:
    """The public comparison of the fixture's two sides at one level, as JSON."""
    return compare_datasets(
        fixture / "baseline.parquet",
        fixture / "candidate.parquet",
        keys="id",
        levels=[level],
        rtol=0,
        atol=0,
        resources=policy,
        entry_points=False,
    ).model_dump(mode="json")


def _verify_comparison(
    case: str, name: str, expected: dict[str, int], changed: int, rows: int, result: Any
) -> dict[str, Any]:
    """Check one comparison level against the fixture's arithmetic oracle."""
    observed = next(item for item in result["levels"] if item["level"] == name)
    for key, value in expected.items():
        if observed[key] != value:
            raise AssertionError(f"{case} oracle mismatch: {key}")
    if any(not check["assessed"] for check in observed["checks"]):
        raise AssertionError(f"{case} was not fully assessed")
    if case == "temporal_buckets":
        measure = observed["measures"][0]
        if measure["material"] != changed or measure["compared"] != expected["baseline_buckets"]:
            raise AssertionError("temporal material-bucket oracle mismatch")
    if case == "spatial_locations":
        moved = next(check for check in observed["checks"] if check["check"] == "moved")
        if moved["affected_count"] != changed or moved["passed"]:
            raise AssertionError("spatial move-count oracle mismatch")
        if any(row["id"] % 101 or row["distance"] != 5.0 for row in moved["affected_records"]):
            raise AssertionError("spatial 3-4-5 distance oracle mismatch")
    return {"verified": True, "case": case, "rows": rows, "changed": changed, **expected}


def _comparison(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions, rows: int
) -> PreparedWorkload:
    changed = (rows + 100) // 101
    level: str | dict[str, Any]
    if case == "dataset_records":
        level = "records"
        expected = {
            "matched": rows,
            "rows_added": 0,
            "rows_removed": 0,
            "rows_changed": changed,
            "rows_unchanged": rows - changed,
            "values_changed": changed * 3,
        }
    elif case == "temporal_buckets":
        level = {
            "level": "temporal",
            "time_column": "time",
            "bucket": "hour",
            "measures": ["sum(value)"],
            "decision": "material",
        }
        expected = {
            "baseline_buckets": (rows + 59) // 60,
            "candidate_buckets": (rows + 59) // 60,
            "only_in_baseline": 0,
            "only_in_candidate": 0,
        }
    else:
        level = {"level": "locations", "x": "x", "y": "y", "crs": "EPSG:32614", "max_distance": 3.0}
        expected = {}
    name = level if isinstance(level, str) else level["level"]
    policy = resources(work_dir / "temporary", options)
    return PreparedWorkload(
        partial(_run_comparison, fixture, level, policy),
        partial(_verify_comparison, case, name, expected, changed, rows),
        {
            "api": "qavf.compare_datasets",
            "input_rows": rows * 2,
            "oracle": "integer change counts, 60-minute bins and exact 3-4-5 distances",
            "timed_scope": "public comparison including startup and cleanup",
        },
    )


def _validate_named(name: str, data: Dataset, config: dict[str, Any], policy: Any) -> Any:
    """Validate one named dataset through the public entry point."""
    return validate({name: data}, config=config, resources=policy)


def _verify_dirty(changed: int, rows: int, report: Any) -> dict[str, Any]:
    """Check the dirty CSV's rejected rows against the fixture's oracle."""
    found = [r for r in report.results if r.check and "unreadable" in r.check]
    if len(found) != 1 or found[0].status is not Status.FAIL or found[0].affected_count != changed:
        raise AssertionError("dirty CSV reject-count oracle mismatch")
    for item in found[0].affected_records:
        if (item["line"] - 2) % 101 or item["error"] != "TOO MANY COLUMNS":
            raise AssertionError("dirty CSV line oracle mismatch")
    return {"verified": True, "rows": rows, "rejected": changed}


def _dirty(fixture: Path, work_dir: Path, options: WorkloadOptions, rows: int) -> PreparedWorkload:
    changed = (rows + 100) // 101
    source = File(
        fixture / "dirty.csv",
        options={"store_rejects": True},
        acknowledge_lossy=True,
        rejects_budget=changed + 1,
    )
    data = Dataset(source, schema={"columns": {"id": {"dtype": "int"}, "value": {"dtype": "int"}}})
    config = {
        "plugins": {"entry_points": False},
        "engine": {"parallel": False, "run_unconfigured_rules": False},
        "rules": {"INP-002": {}},
    }
    policy = resources(work_dir / "temporary", options)
    return PreparedWorkload(
        partial(_validate_named, "dirty", data, config, policy),
        partial(_verify_dirty, changed, rows),
        {
            "api": "qavf.validate",
            "input_rows": rows,
            "oracle": "one structurally invalid row at each index divisible by 101",
            "timed_scope": "public ingestion, diagnostics and validation",
        },
    )


class _Navigation(HTMLParser):
    """Verify report navigation without retaining the rendered document."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: set[str] = set()
        self.duplicates = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.duplicates += int(value in self.ids)
                self.ids.add(value)
            if name == "href" and value and value.startswith("#"):
                self.links.add(value[1:])


def _run_report(
    report: QAReport, html_path: Path, json_path: Path, limits: HtmlLimits, count: int
) -> dict[str, int]:
    """Write the report as HTML and JSON and return the output sizes."""
    write_html(report, html_path, limits=limits)
    report.write_json(json_path)
    return {
        "results": count,
        "html_bytes": html_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
    }


def _verify_report(json_path: Path, html_path: Path, count: int, result: Any) -> dict[str, Any]:
    """Check the written report's occurrences, counts and in-page navigation."""
    with json_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    navigation = _Navigation()
    with html_path.open(encoding="utf-8") as stream:
        while chunk := stream.read(65536):
            navigation.feed(chunk)
    navigation.close()
    if (
        result["results"] != count
        or len(payload["results"]) != count
        or payload["summary"]["failed"] != count
        or navigation.duplicates
        or not navigation.links <= navigation.ids
    ):
        raise AssertionError("report occurrence/count/navigation oracle mismatch")
    return {"verified": True, "results": count}


def _report(work_dir: Path, rows: int) -> PreparedWorkload:
    count = max(1, rows // 100)
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    result = RuleResult(
        rule_id="PERF-001",
        name="Synthetic finding",
        description="Report stress",
        category="performance",
        status=Status.FAIL,
        severity="WARNING",
        message="Exact fixture finding",
        affected_count=1,
        affected_records=[{"id": 0}],
        recommendation="Review fixture",
    )
    report = QAReport(
        metadata=RunMetadata(
            run_id="benchmark",
            project="performance",
            qavf_version="benchmark",
            started_at=stamp,
            finished_at=stamp,
        ),
        results=[result] * count,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = work_dir / "large.html", work_dir / "large.json"

    limits = HtmlLimits(max_output_bytes=max(16_777_216, count * 8192))
    return PreparedWorkload(
        partial(_run_report, report, html_path, json_path, limits, count),
        partial(_verify_report, json_path, html_path, count),
        {
            "api": "QAReport.write_json and write_html",
            "input_rows": rows,
            "result_occurrences": count,
            "html_output_limit_bytes": limits.max_output_bytes,
            "timed_scope": "streamed HTML and canonical JSON serialization plus file writes",
        },
    )


def build_workload(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions
) -> PreparedWorkload:
    rows = read_json(fixture / "manifest.json")["rows"]
    if case in {"dataset_records", "temporal_buckets", "spatial_locations"}:
        return _comparison(case, fixture, work_dir, options, rows)
    if case == "dirty_csv":
        return _dirty(fixture, work_dir, options, rows)
    if case == "report_rendering":
        return _report(work_dir, rows)
    raise ValueError(f"Unknown breadth case: {case}")
