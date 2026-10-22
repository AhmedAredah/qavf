"""The comparison section of HTML and JSON reports, standalone comparison reports and
the ``qavf compare-datasets`` command."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from qavf import File, ResourceLimits, compare_datasets, validate
from qavf.cli import EXIT_OK, EXIT_QA_FAILED, EXIT_USAGE, main
from qavf.compare import DatasetComparison
from qavf.engine.resources import ResourceLimitError
from qavf.execution.os_limits import SUPPORTED_PLATFORMS
from qavf.reports.html import render_html
from qavf.reports.models import QAReport
from qavf.reports.presentation import HtmlLimits
from qavf.spatial import extension
from qavf.spatial.extension import SpatialUnavailable

pytest.importorskip("scipy")


def _support():
    name = "qavf_spatial_support"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("spatial_support.py")
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _frames(seed: int = 3, n: int = 400):
    rng = np.random.default_rng(seed)
    base = pl.DataFrame(
        {
            "id": np.arange(n),
            "zone": rng.choice(["a", "b", "c"], n),
            "day": [date(2024, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 90, n)],
            "x": rng.normal(10, 2, n),
            "y": rng.normal(5, 1, n),
            "fatal": rng.poisson(2, n),
            "crashes": rng.poisson(20, n) + 1,
            "lon": rng.uniform(0, 3, n),
            "lat": rng.uniform(0, 2, n),
        }
    )
    shifted = pl.when(pl.col("zone") == "a").then(pl.col("x") + 3).otherwise(pl.col("x"))
    return base, base.with_columns(x=shifted)


STEPS = [{"step": "assign_zones", "x": "lon", "y": "lat", "column": "cell",
          "grid": {"size": 1.0, "extent": [0, 0, 3, 2]}}]  # fmt: skip
LEVELS = [
    "structure",
    {"level": "temporal", "time_column": "day", "bucket": "month", "group_by": ["zone"],
     "measures": ["sum(x)"]},
    {"level": "distribution", "columns": ["x"], "group_by": ["zone"]},
    {"level": "relationships", "correlation": {"columns": ["x", "y"]},
     "agreement": {"columns": ["x"]}},
    {"level": "contribution", "measure": "sum(x)",
     "dimensions": ["zone", {"columns": ["zone"],
                             "time": {"time_column": "day", "bucket": "month"}}],
     "rates": ["fatal / crashes"]},
    {"level": "spatial", "zone_column": "cell", "measures": ["sum(x)"],
     "weights": {"method": "rook"}},
]  # fmt: skip


def _compare(**options):
    base, cand = _frames()
    return compare_datasets(base, cand, keys=["id"], steps=STEPS, levels=LEVELS, **options)


class _Structure(HTMLParser):
    """Element ids, chart and section classes in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.marks: list[str] = []

    def handle_starttag(self, tag, attrs):
        found = dict(attrs)
        if found.get("id"):
            self.marks.append(f"{tag}#{found['id']}")
        for css in ("comparison", "levels", "level-detail", "figure-grid", "checks", "tests"):
            if css in (found.get("class") or "").split():
                self.marks.append(f"{tag}.{css}")
        if tag == "svg" and "cmp" in (found.get("class") or ""):
            self.marks.append("svg.chart")


def _structure(html: str) -> list[str]:
    parser = _Structure()
    parser.feed(html)
    return parser.marks


# ------------------------------------------------------------- standalone
def test_a_standalone_report_has_the_summary_drill_down_and_charts(tmp_path):
    result = _compare()
    html = result.write_html(tmp_path / "comparison.html").read_text(encoding="utf-8")
    assert "<title>Dataset comparison: comparison</title>" in html
    assert html.count("<svg") >= 8 and "Chart omitted" not in html
    for title in (
        "Contributions to the change of sum(x) by zone",
        "Relative change by zone and bucket",
        "fatal / crashes: mix and rate effects over zone",
        "x (a): share per bin",
        "x (a): quantile deltas",
        "sum(x) per bucket (a)",
        "sum(x): relative change by zone and bucket",
        "sum(x): relative change per zone",
    ):
        assert title in html, title
    assert 'class="zone ' in html  # the choropleth's zone paths
    assert "Adjusted p" in html and "Evidence and significance of x" in html
    assert re.search(r"https?://", html) is None  # no external assets
    marks = _structure(html)
    assert marks[:2] == ["article#cmp-comparison", "article.comparison"]
    details = [m for m in marks if m.startswith("details#cmp-comparison-")]
    assert details == [f"details#cmp-comparison-{i}" for i in range(1, len(LEVELS) + 1)]
    # Within each level: charts first, then the checks table.
    first, second = marks.index("details#cmp-comparison-2"), marks.index("details#cmp-comparison-3")
    inside = marks[first:second]
    assert inside.index("div.figure-grid") < inside.index("table.checks")
    assert marks.index("div.levels") < marks.index("details#cmp-comparison-1")


def test_charts_beyond_the_limits_are_omitted_with_a_note():
    result = _compare()
    few = result.render_html(limits=HtmlLimits(max_chart_points=3))
    assert "Chart omitted: too large" in few and "the limit is 3" in few
    tiny = result.render_html(limits=HtmlLimits(max_chart_bytes=2_000))
    assert "chart budget is used" in tiny and tiny.count("<svg") < 3


def test_the_map_is_omitted_without_the_spatial_extension(tmp_path, monkeypatch):
    S = _support()
    layer = S.geoparquet(tmp_path / "zones.parquet", S.grid_zones(2))
    frame = pl.DataFrame({"zone": ["z0_0", "z0_1", "z1_0", "z1_1"] * 3, "x": np.arange(12.0)})
    level = {"level": "spatial", "measures": ["sum(x)"], "layer": {"path": str(layer)},
             "moran": None, "lisa": None, "shares": None}  # fmt: skip
    drawn = compare_datasets(frame, frame, levels=[level])
    if drawn.level("spatial").outlines.note is None:
        assert len(drawn.level("spatial").outlines.paths) == 4
        assert 'class="zone ' in drawn.render_html()

    def unavailable(con):
        raise SpatialUnavailable("DuckDB's spatial extension could not be loaded (simulated)")

    monkeypatch.setattr(extension, "load_spatial", unavailable)
    result = compare_datasets(frame, frame, levels=[level])
    outlines = result.level("spatial").outlines
    assert outlines.paths == {} and "simulated" in outlines.note
    html = result.render_html()
    assert "Map omitted: no zone outlines: DuckDB&#39;s spatial extension" in html
    assert result.passed  # the map is presentation only; the checks still ran


def test_json_round_trips_and_discloses_truncation(tmp_path):
    result = _compare()
    path = result.write_json(tmp_path / "comparison.json")
    again = DatasetComparison.model_validate_json(path.read_text(encoding="utf-8"))
    assert again == result and again.omitted == []
    assert type(again.level("contribution")).__name__ == "ContributionResult"
    full = result.json_size_bytes()
    lean = result.bounded(0).json_size_bytes()  # everything optional left out
    budget = (lean + full) // 2
    small = result.bounded(budget)
    assert small.omitted and small.json_size_bytes() < full
    assert small.verdict == result.verdict
    assert [len(level.checks) for level in small.levels] == [
        len(level.checks) for level in result.levels
    ]
    assert small.json_size_bytes() <= budget
    written = result.write_json(tmp_path / "small.json", max_bytes=budget)
    assert json.loads(written.read_text(encoding="utf-8"))["omitted"] == small.omitted
    with pytest.raises(ResourceLimitError):
        result.write_json(tmp_path / "tiny.json", max_bytes=100)


# --------------------------------------------------------------- CMP-001
def _config(levels=LEVELS, **engine):
    return {
        "rules": {"CMP-001": {"params": {"comparisons": [
            {"name": "v1 vs v2", "baseline": "v1", "candidate": "v2", "keys": ["id"],
             "steps": STEPS, "levels": levels},
        ]}}},
        "engine": {"run_unconfigured_rules": False, **engine},
        "plugins": {"entry_points": False},
    }  # fmt: skip


def test_cmp001_reports_carry_the_typed_comparison_and_render_its_section():
    base, cand = _frames()
    report = validate({"v1": base, "v2": cand}, config=_config())
    [comparison] = report.comparisons()
    assert comparison.name == "v1 vs v2" and comparison.verdict == "failed"
    direct = compare_datasets(base, cand, keys=["id"], steps=STEPS, levels=LEVELS)
    names = {"name", "baseline", "candidate"}
    assert comparison.model_dump(exclude=names) == direct.model_dump(exclude=names)
    again = QAReport.model_validate_json(b"".join(report.iter_json_chunks()))
    assert again.comparisons() == [comparison]
    html = render_html(report)
    marks = _structure(html)
    assert marks.index("h2#comparisons") < marks.index("article#cmp-v1-vs-v2")
    assert "svg.chart" in marks and "Dataset comparisons (1)" in html
    assert html.index("Dataset comparisons (1)") < html.index("<h2>All results</h2>")
    # The comparison is not repeated in the metrics dump.
    assert html.count('"comparison":') == 0


def test_a_report_too_small_for_the_detail_keeps_the_findings_section():
    base, cand = _frames()
    report = validate({"v1": base, "v2": cand}, config=_config(max_evidence_bytes=20_000))
    assert report.comparisons() == []
    html = render_html(report)
    assert "Dataset comparisons (1)" in html and "did not fit the evidence limits" in html
    assert 'id="cmp-v1-vs-v2"' in html


def test_a_trimmed_report_budget_discloses_the_omission():
    base, cand = _frames()
    report = validate({"v1": base, "v2": cand}, config=_config())
    budget = report.json_size_bytes() // 2
    trimmed = validate({"v1": base, "v2": cand}, config=_config(max_report_bytes=budget))
    assert trimmed.comparisons() == []
    assert "noncoverage_metrics" in trimmed.metadata.resource_usage["report_details_omitted"]


def test_the_new_levels_agree_in_an_isolated_run(tmp_path):
    if sys.platform not in SUPPORTED_PLATFORMS:
        pytest.skip("isolated execution is not supported here")
    base, cand = _frames()
    base.write_parquet(tmp_path / "v1.parquet")
    cand.write_parquet(tmp_path / "v2.parquet")
    levels = LEVELS[2:5]
    local = compare_datasets(tmp_path / "v1.parquet", tmp_path / "v2.parquet", keys=["id"],
                             levels=levels)  # fmt: skip
    isolated = compare_datasets(
        File(tmp_path / "v1.parquet"),
        File(tmp_path / "v2.parquet"),
        keys=["id"],
        levels=levels,
        resources=ResourceLimits(isolate=True, wall_time=180),
    )
    assert isolated.model_dump(exclude={"execution_limits"}) == local.model_dump(
        exclude={"execution_limits"}
    )


def test_the_new_levels_are_built_in_whatever_the_plugin_choice():
    base, _ = _frames()
    for enabled in (False, True):
        result = compare_datasets(base, base, keys=["id"], levels=LEVELS[2:5],
                                  entry_points=enabled)  # fmt: skip
        assert result.passed, result.summary()


# -------------------------------------------------------------------- CLI
def test_compare_datasets_command_writes_reports_and_returns_the_verdict(tmp_path, capsys):
    base, cand = _frames()
    base.write_parquet(tmp_path / "v1.parquet")
    cand.write_parquet(tmp_path / "v2.parquet")
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "name: counts\nkeys: [id]\nlevels:\n  - structure\n"
        "  - {level: distribution, columns: [x], group_by: [zone]}\n",
        encoding="utf-8",
    )
    html, out = tmp_path / "out" / "comparison.html", tmp_path / "out" / "comparison.json"
    args = ["compare-datasets", str(tmp_path / "v1.parquet"), str(tmp_path / "v2.parquet"),
            "-c", str(spec), "--html", str(html), "--json", str(out)]  # fmt: skip
    assert main(args) == EXIT_QA_FAILED
    printed = capsys.readouterr().out
    assert "COMPARISON FAILED" in printed and "[FAILED] distribution: x" in printed
    assert html.exists() and "Dataset comparison: counts" in html.read_text(encoding="utf-8")
    assert DatasetComparison.model_validate_json(out.read_text(encoding="utf-8")).name == "counts"
    same = ["compare-datasets", str(tmp_path / "v1.parquet"), str(tmp_path / "v1.parquet"),
            "-c", str(spec)]  # fmt: skip
    assert main(same) == EXIT_OK and "COMPARISON PASSED" in capsys.readouterr().out
    spec.write_text("levels: [{level: distribution, columns: [nowhere]}]\n", encoding="utf-8")
    assert main(same) == EXIT_QA_FAILED  # not assessed is never a pass
    assert "COMPARISON INCOMPLETE" in capsys.readouterr().out
    spec.write_text("levels: [{level: nonsense}]\n", encoding="utf-8")
    assert main(same) == EXIT_USAGE
    spec.write_text("baseline: elsewhere\n", encoding="utf-8")
    assert main(same) == EXIT_USAGE and "the command's arguments" in capsys.readouterr().err
    assert main(["compare-datasets", str(tmp_path / "v1.parquet"), str(tmp_path / "absent.csv")]
                ) == EXIT_USAGE  # fmt: skip
