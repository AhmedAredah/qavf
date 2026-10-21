"""The shipped example configurations run end to end.

Planning only proves a configuration is well formed. These tests run each example
against clean synthetic data matching what it describes and require every
configured check to be assessed and to pass, so an example that refers to a column
its data does not have (or a time column of the wrong type) fails here.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.settings import load_settings
from qavf.testing import discovered_rules

ROOT = Path(__file__).resolve().parents[1]
STATIONS = [f"ABC{i:02d}" for i in range(1, 7)]


def _run(config: Path, frames: dict[str, pl.DataFrame] | None = None):
    settings = load_settings(config)
    settings.plugins.entry_points = False
    return QARunner(settings, discovered_rules(entry_points=False)).run(frames=frames)


def _assert_clean(report) -> None:
    unassessed = [f"{r.label}: {r.message}" for r in report.results if r.status is Status.ERROR]
    failed = [f"{r.label}: {r.message}" for r in report.results if r.status is Status.FAIL]
    assert not unassessed, unassessed
    assert not failed, failed
    assert report.verdict() == "passed"


def test_full_example_runs_clean(tmp_path) -> None:
    config = tmp_path / "qavf.yaml"
    shutil.copy(ROOT / "config" / "qavf.full.yaml", config)
    (tmp_path / "data" / "readings").mkdir(parents=True)
    start = datetime(2024, 1, 1)
    rows = [
        {
            "Station_ID": station,
            "timestamp": start + timedelta(hours=hour),
            "value": 20.0 + 0.5 * (hour % 4) + 0.01 * index,
            "status": "valid",
            "serial": index * 10_000 + hour,
        }
        for index, station in enumerate(STATIONS)
        for hour in range(24 * 10)
    ]
    pl.DataFrame(rows).write_parquet(tmp_path / "data" / "readings" / "part-1.parquet")
    frames = {
        # Databases and Python loaders of the example are supplied in memory.
        "stations": pl.DataFrame({"station": STATIONS}),
        "forecasts": pl.DataFrame({"station": STATIONS, "horizon": [7] * len(STATIONS)}),
    }
    report = _run(config, frames)
    _assert_clean(report)
    # The configured temporal, spatial and statistical checks were all assessed.
    for rule_id in ("TMP-001", "SPA-001", "SPA-003", "STA-001"):
        assert report.get(rule_id).status is Status.PASS, rule_id
    assert report.get("TMP-001").total_records == len(STATIONS) * 9
    # The example comparison ran every configured level and every check passed.
    compared = [r for r in report.results if r.rule_id == "CMP-001"]
    assert {r.check.split(": ")[1] for r in compared} == {
        "structure", "coverage", "records", "groups"
    }
    assert all(r.status is Status.PASS for r in compared)


def test_minimal_example_runs_clean(tmp_path) -> None:
    config = tmp_path / "qavf.yaml"
    shutil.copy(ROOT / "config" / "qavf.minimal.yaml", config)
    (tmp_path / "data").mkdir()
    pl.DataFrame(
        {
            "station": [s for s in STATIONS for _ in range(3)],
            "date": [date(2024, 1, 1 + d) for _ in STATIONS for d in range(3)],
            "pm25": [12.5] * (len(STATIONS) * 3),
        }
    ).write_csv(tmp_path / "data" / "measurements.csv")
    _assert_clean(_run(config))


@pytest.mark.parametrize("name", ["qavf.full.yaml", "qavf.minimal.yaml"])
def test_example_configurations_detect_a_planted_defect(tmp_path, name) -> None:
    """The examples are not vacuous: a negative measurement is caught."""
    config = tmp_path / "qavf.yaml"
    shutil.copy(ROOT / "config" / name, config)
    (tmp_path / "data" / "readings").mkdir(parents=True)
    if name == "qavf.minimal.yaml":
        pl.DataFrame({"station": ["ABC01"], "date": [date(2024, 1, 1)], "pm25": [-1.0]}).write_csv(
            tmp_path / "data" / "measurements.csv"
        )
        report = _run(config)
    else:
        pl.DataFrame(
            {
                "Station_ID": ["ABC01"],
                "timestamp": [datetime(2024, 1, 1)],
                "value": [-1.0],
                "status": ["valid"],
                "serial": [1],
            }
        ).write_parquet(tmp_path / "data" / "readings" / "part-1.parquet")
        report = _run(
            config,
            {
                "stations": pl.DataFrame({"station": ["ABC01"]}),
                "forecasts": pl.DataFrame({"x": [1]}),
            },
        )
    assert any(r.rule_id == "INP-003" and r.status is Status.FAIL for r in report.results)
