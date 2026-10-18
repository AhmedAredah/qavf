"""A self-contained demo project: synthetic traffic data, a configuration, a baseline.

``qavf demo DIR`` writes a project that exercises every source type and rule
family::

    DIR/
      qavf.yaml              annotated configuration
      data/counts.csv        traffic counts           (file source)
      data/crashes.parquet   crashes                  (file source)
      data/sites.duckdb      site inventory           (database source)
      baseline/counts.csv    accepted counts          (regression baseline)

The data is synthetic and has no meaning beyond demonstrating QA. Named defects
can be injected so the report has something to find.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Iterable
from datetime import date
from importlib.resources import files
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from qavf.io import write_table

logger = logging.getLogger(__name__)

Datasets = dict[str, pl.DataFrame]

YEARS: tuple[int, ...] = tuple(range(2016, 2025))
_CLASSES = {
    "Interstate": (60_000, 4, 0.18),
    "Arterial": (22_000, 2, 0.08),
    "Collector": (6_000, 1, 0.05),
}
_CRASH_RATE = 1.1e-5  # crashes per vehicle-day, purely illustrative
_GROWTH = 0.02


def make_demo_data(seed: int = 0) -> Datasets:
    """Clean, mutually consistent ``sites``, ``counts`` and ``crashes`` tables."""
    rng = np.random.default_rng(seed)
    site_rows, count_rows, crash_rows = [], [], []
    classes = list(_CLASSES)
    for number in range(1, 13):
        site_id = f"S{number:02d}"
        region = "North" if number <= 6 else "South"
        fclass = classes[(number - 1) % 3]
        base, lane_pairs, trucks = _CLASSES[fclass]
        size = rng.uniform(0.8, 1.25)
        site_rows.append(
            (
                site_id,
                region,
                fclass,
                lane_pairs * 2,
                round(float(rng.uniform(29.5, 33.5)), 5),
                round(float(rng.uniform(-98.5, -95.0)), 5),
            )
        )
        for year in YEARS:
            aadt = int(base * size * (1 + _GROWTH) ** (year - YEARS[0]) * rng.uniform(0.99, 1.01))
            share = round(trucks * rng.uniform(0.95, 1.05), 4)
            count_rows.append(
                (
                    site_id,
                    year,
                    aadt,
                    share,
                    date(year, 6, 15),
                    "permanent" if number % 4 else "portable",
                )
            )
            total = max(1, round(aadt * 365 * _CRASH_RATE * rng.uniform(0.97, 1.03)))
            crash_rows.append((site_id, year, total, round(total * 0.012)))
    sites = pl.DataFrame(
        site_rows,
        schema=["site_id", "region", "functional_class", "lanes", "latitude", "longitude"],
        orient="row",
    )
    counts = pl.DataFrame(
        count_rows,
        schema=["site_id", "year", "aadt", "truck_share", "count_date", "count_type"],
        orient="row",
    )
    crashes = pl.DataFrame(crash_rows, schema=["site_id", "year", "crashes", "fatal"], orient="row")
    return {"sites": sites, "counts": counts, "crashes": crashes}


# --------------------------------------------------------------------- defects
def _where(frame: pl.DataFrame, column: str, when: pl.Expr, then: pl.Expr) -> pl.DataFrame:
    return frame.with_columns(pl.when(when).then(then).otherwise(pl.col(column)).alias(column))


def _site_year(site: str, year: int) -> pl.Expr:
    return (pl.col("site_id") == site) & (pl.col("year") == year)


def _negative_aadt(d: Datasets) -> None:
    d["counts"] = _where(d["counts"], "aadt", _site_year("S02", 2019), -pl.col("aadt"))


def _truck_share_percent(d: Datasets) -> None:
    d["counts"] = _where(
        d["counts"], "truck_share", _site_year("S05", 2020), pl.col("truck_share") * 100
    )


def _missing_count(d: Datasets) -> None:
    d["counts"] = d["counts"].filter(~_site_year("S11", 2022))


def _aadt_spike(d: Datasets) -> None:
    d["counts"] = _where(d["counts"], "aadt", _site_year("S03", 2021), pl.col("aadt") * 3)


def _fatal_exceeds(d: Datasets) -> None:
    d["crashes"] = _where(d["crashes"], "fatal", _site_year("S07", 2018), pl.col("crashes") + 5)


def _unknown_site(d: Datasets) -> None:
    extra = (
        d["crashes"].filter(pl.col("site_id") == "S01").with_columns(pl.lit("S99").alias("site_id"))
    )
    d["crashes"] = pl.concat([d["crashes"], extra])


def _duplicate_count(d: Datasets) -> None:
    d["counts"] = pl.concat([d["counts"], d["counts"].filter(_site_year("S09", 2017))])


def _bad_site_id(d: Datasets) -> None:
    d["counts"] = _where(d["counts"], "site_id", _site_year("S10", 2024), pl.lit("s10 "))


def _future_date(d: Datasets) -> None:
    d["counts"] = _where(
        d["counts"], "count_date", _site_year("S04", 2023), pl.lit(date(2031, 6, 15))
    )


def _declining_site(d: Datasets) -> None:
    factor = (1 + _GROWTH) ** (-4 * (pl.col("year") - YEARS[0]))
    d["counts"] = _where(
        d["counts"], "aadt", pl.col("site_id") == "S08", (pl.col("aadt") * factor).cast(pl.Int64)
    )


DEFECTS: dict[str, Callable[[Datasets], None]] = {
    "negative_aadt": _negative_aadt,
    "truck_share_percent": _truck_share_percent,
    "missing_count": _missing_count,
    "aadt_spike": _aadt_spike,
    "fatal_exceeds": _fatal_exceeds,
    "unknown_site": _unknown_site,
    "duplicate_count": _duplicate_count,
    "bad_site_id": _bad_site_id,
    "future_date": _future_date,
    "declining_site": _declining_site,
}
"""Named defects accepted by :func:`inject_defects`."""


def inject_defects(data: Datasets, defects: Iterable[str] | None = None) -> Datasets:
    """Return a copy of ``data`` with the named defects applied (all by default).

    Raises:
        KeyError: for an unknown defect name.
    """
    broken = dict(data)
    for name in DEFECTS if defects is None else defects:
        if name not in DEFECTS:
            raise KeyError(f"unknown defect {name!r}; choose from {', '.join(DEFECTS)}")
        DEFECTS[name](broken)
    return broken


def stats_available() -> bool:
    """Whether the optional ``stats`` extra (SciPy, StatsModels) is installed."""
    return importlib.util.find_spec("scipy") is not None


def demo_config(with_stats: bool | None = None) -> str:
    """The demo project's annotated configuration (YAML text).

    Without the ``stats`` extra the two trend rules that need it are disabled,
    with a note, so that a core install runs the demo to completion rather than
    reporting those configured checks as not assessed.
    """
    text = files("qavf").joinpath("demo_config.yaml").read_text(encoding="utf-8")
    if with_stats if with_stats is not None else stats_available():
        return text
    note = "    enabled: false                # needs the stats extra: pip install 'qavf[stats]'\n"
    for rule in ("TMP-002", "TMP-003"):
        start = text.index(f"  {rule}:")
        line_end = text.index("\n", start) + 1
        text = text[:line_end] + note + text[line_end:]
    return text


def write_demo_project(
    directory: str | Path, defects: Iterable[str] | None = None, *, clean: bool = False
) -> Path:
    """Write the demo project to ``directory`` and return the configuration path.

    Args:
        directory: Target directory (created if needed).
        defects: Defects to inject; all of them when ``None``.
        clean: Inject no defects at all.
    """
    root = Path(directory)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    accepted = make_demo_data()
    data = accepted if clean else inject_defects(accepted, defects)

    write_table(data["counts"], data_dir / "counts.csv")
    write_table(data["crashes"], data_dir / "crashes.parquet")
    database = data_dir / "sites.duckdb"
    database.unlink(missing_ok=True)
    with duckdb.connect(str(database)) as con:
        sites = data["sites"]
        con.register("sites_frame", sites)
        con.execute("CREATE TABLE sites AS SELECT * FROM sites_frame")
    write_table(accepted["counts"].sort("site_id", "year"), root / "baseline" / "counts.csv")

    config = root / "qavf.yaml"
    config.write_text(demo_config(), encoding="utf-8")
    logger.info("Demo project written to %s", root)
    return config
