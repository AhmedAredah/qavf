"""Exercise installed core capabilities with optional dependencies truly absent.

Run with ``python -I scripts/verify_core.py`` from a core-only wheel environment.
Checks deliberately remain active under ``python -O``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from qavf import Status, validate
from qavf.io import read_table


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_dependencies() -> None:
    for name in ("pandas", "scipy", "pyproj"):
        require(importlib.util.find_spec(name) is None, f"core environment includes {name}")


def check_input_verdicts() -> None:
    schema = {"columns": {"v": {"dtype": "float", "ge": 0}}}
    for values, expected in (([1.0, 2.0], "passed"), ([1.0, -2.0], "failed")):
        frame = pl.DataFrame({"v": values})
        for data in (frame, frame.lazy(), frame.to_arrow()):
            report = validate(data, schema=schema)
            require(report.verdict() == expected, f"incorrect core verdict: {report.verdict()}")


def check_temporal_roundtrip() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "precise.parquet"
        original = pl.DataFrame({"id": [1], "t": pl.Series([1], dtype=pl.Datetime("ns", "UTC"))})
        original.write_parquet(path)
        restored = read_table(path)
        require(restored.schema == original.schema, "Parquet changed the logical schema")
        require(restored["t"].cast(pl.Int64).to_list() == [1], "Parquet changed temporal ticks")


def check_unavailable_statistics() -> None:
    frame = pl.DataFrame({"x": [float(i) for i in range(200)]})
    report = validate(
        {"reference": frame, "current": frame},
        config={
            "plugins": {"entry_points": False},
            "engine": {"parallel": False, "run_unconfigured_rules": False},
            "datasets": {"reference": {}, "current": {}},
            "rules": {"STA-002": {"params": {"comparisons": [{
                "dataset": "current", "reference": "reference", "column": "x", "method": "ks",
            }]}}},
        },
    )
    results = report.results_for("STA-002")
    require(len(results) == 1, "missing-stats check did not execute exactly once")
    require(results[0].status is Status.ERROR, "missing SciPy produced an assessed verdict")
    require("stats extra" in results[0].message, "missing SciPy error lacks actionable guidance")
    require(report.exit_code() != 0, "missing SciPy left the run acceptable")


def check_core_coordinate_systems() -> None:
    from qavf.spatial.crs import is_geographic

    for crs in ("EPSG:4326", "OGC:CRS84"):
        require(is_geographic(crs), f"core CRS classification failed for {crs}")
    for crs in ("EPSG:3857", "EPSG:32614", "EPSG:32760"):
        require(not is_geographic(crs), f"core CRS classification failed for {crs}")
    try:
        is_geographic("EPSG:26915")
    except ValueError as exc:
        require("qavf[spatial]" in str(exc), "missing pyproj error lacks actionable guidance")
    else:
        raise RuntimeError("unsupported core CRS silently received a coordinate type")


def main() -> None:
    check_dependencies()
    check_input_verdicts()
    check_temporal_roundtrip()
    check_unavailable_statistics()
    check_core_coordinate_systems()
    require("pandas" not in sys.modules, "ordinary core validation initialized pandas")
    print("Core-only installed-wheel checks passed: valid/invalid inputs, temporal ticks, "
          "core CRS resolution, and explicit unavailable-dependency errors.")


if __name__ == "__main__":
    main()
