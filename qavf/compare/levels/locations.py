"""Locations: did features matched by key move, and did polygon features change shape?

Features are matched by the comparison's ``keys`` (a key repeated on a side
leaves the level not assessed). Their location is either a point (``x``/``y``
columns) or a WKB ``geometry`` column (points, lines or polygons), in ``crs``.

* ``located``: features with a location on one side only (both missing is
  counted, not a change);
* ``moved``: features whose point, or geometry centroid, moved more than
  ``max_distance``: haversine metres for a geographic CRS, Euclidean CRS units
  otherwise (:mod:`qavf.spatial.distance`), computed natively;
* for geometries (DuckDB's spatial extension; not assessed without it):
  ``area`` (a material change of area, ``rtol``/``atol``; square metres on the
  WGS 84 spheroid for a geographic CRS, else square CRS units) and ``shape``
  (geometries that are not equal and whose symmetric difference,
  ``area(a) + area(b) - 2 area(a n b)``, exceeds ``max_difference``).

Evidence lists the largest changes first, bounded by the run's limits; counts
are complete.
"""

from __future__ import annotations

from typing import Any, ClassVar

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import Materiality
from qavf.compare.levels import SIDES, ComparisonLevel, LevelInputs, graded, register_level
from qavf.compare.results import CheckResult, LevelResult, LocationsResult
from qavf.engine.registry import RuleParams
from qavf.keys import align_keys
from qavf.rules.helpers import fresh_internal, plural
from qavf.spatial.crs import DEFAULT_CRS, Crs, distance_unit, is_geographic
from qavf.spatial.distance import distance
from qavf.spatial.extension import fetch, register, spatial_connection


class AreaCheck(RuleParams):
    """A geometry's area may change within ``max(atol, rtol * max(|a|, |b|))``."""

    rtol: float = Field(default=0.0, ge=0)
    """Relative tolerance of the areas."""
    atol: float = Field(default=0.0, ge=0)
    """Absolute tolerance: square metres for a geographic CRS, else square CRS units."""


class ShapeCheck(RuleParams):
    """Geometries are equal, or differ by at most ``max_difference`` of area."""

    max_difference: float = Field(default=0.0, ge=0)
    """Largest symmetric-difference area (the units of ``area``)."""


class LocationsParams(RuleParams):
    """Parameters of the ``locations`` level."""

    x: str | None = None
    """Longitude or easting column."""
    y: str | None = None
    """Latitude or northing column."""
    geometry: str | None = None
    """A WKB geometry column instead of ``x``/``y`` (points, lines or polygons)."""
    crs: Crs = DEFAULT_CRS
    """CRS of both sides' locations."""
    max_distance: float = Field(default=0.0, ge=0)
    """Largest move (point, or centroid) that is no change: metres for a geographic CRS."""
    area: AreaCheck | None = Field(default_factory=AreaCheck)
    """Area changes of geometries; ``null`` leaves the check out."""
    shape: ShapeCheck | None = Field(default_factory=ShapeCheck)
    """Shape changes of geometries; ``null`` leaves the check out."""

    @model_validator(mode="after")
    def _columns(self) -> LocationsParams:
        if (self.geometry is None) == (self.x is None or self.y is None):
            raise ValueError("locations are 'x' and 'y' columns, or one WKB 'geometry' column")
        return self


def _sql_area(geometry: str, geographic: bool) -> str:
    """Area SQL: square metres on the WGS 84 spheroid (x-first data flipped), else planar."""
    return (
        f"ST_Area_Spheroid(ST_FlipCoordinates({geometry}))"
        if geographic
        else f"ST_Area({geometry})"
    )


@register_level
class LocationsLevel(ComparisonLevel):
    """Distance moved per keyed feature; area and shape changes of geometries."""

    name: ClassVar[str] = "locations"
    description: ClassVar[str] = (
        "Features matched by key: locations on one side only, distance moved (haversine "
        "metres for longitude/latitude, else CRS units) beyond max_distance, and for "
        "geometries the change of area and the symmetric-difference area."
    )
    Params: ClassVar[type[RuleParams]] = LocationsParams
    Result: ClassVar[type[LevelResult]] = LocationsResult
    requires_keys: ClassVar[bool] = True

    def evaluate(self, inputs: LevelInputs, name: str) -> LocationsResult:
        params: LocationsParams = self.params  # type: ignore[assignment]
        keys = list(inputs.spec.keys)
        columns = [c for c in (params.x, params.y, params.geometry) if c is not None]
        frames: dict[str, pl.DataFrame] = {}
        for side in SIDES:
            schema = inputs.schema(side)
            if absent := [c for c in (*keys, *columns) if c not in schema]:
                raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
            frame = inputs.collect(inputs.scan(side).select(*keys, *columns))
            if repeated := frame.select(pl.struct(keys).is_duplicated().sum()).item():
                raise ValueError(f"the {side} repeats keys in {plural(repeated, 'row')}")
            frames[side] = frame
        base, cand = align_keys(frames["baseline"], frames["candidate"], keys)
        taken = [*keys, *columns]
        names = {(side, c): fresh_internal(f"{side}_{c}", taken) for side in SIDES for c in columns}
        joined = base.rename({c: names["baseline", c] for c in columns}).join(
            cand.rename({c: names["candidate", c] for c in columns}),
            on=keys,
            how="inner",
            nulls_equal=True,
        )
        geographic = is_geographic(params.crs)
        unit = distance_unit(params.crs)
        if params.geometry is None:
            assert params.x is not None and params.y is not None
            located = {
                side: pl.all_horizontal(
                    pl.col(names[side, c]).cast(pl.Float64).is_finite()
                    for c in (params.x, params.y)
                ).fill_null(False)
                for side in SIDES
            }
            moves = joined.select(
                *keys,
                *(pl.col(names[side, c]).alias(f"{side}_{c}") for side in SIDES for c in columns),
                located["baseline"].alias("_b"),
                located["candidate"].alias("_c"),
                distance(
                    *(pl.col(names[side, c]).cast(pl.Float64) for side in SIDES
                      for c in (params.x, params.y)),
                    geographic=geographic,
                ).alias("distance"),
            )  # fmt: skip
            kind = "points"
        else:
            moves = self._geometries(inputs, params, joined, keys, names, geographic)
            kind = "geometries"
        checks, counts = self._checks(inputs, params, moves, keys, unit)
        return LocationsResult(
            level=self.name,
            name=name,
            checks=checks,
            keys=keys,
            kind=kind,
            crs=params.crs,
            unit=unit,
            matched=joined.height,
            **counts,
        )

    @staticmethod
    def _geometries(
        inputs: LevelInputs,
        params: LocationsParams,
        joined: pl.DataFrame,
        keys: list[str],
        names: dict[tuple[Any, str], str],
        geographic: bool,
    ) -> pl.DataFrame:
        """Per matched feature: presence, centroid move, both areas and the shape change."""
        assert params.geometry is not None
        column = params.geometry
        for side in SIDES:
            if joined.schema[names[side, column]] != pl.Binary:
                raise ValueError(f"{side} geometry column {column!r} must hold WKB (binary)")
        pairs = joined.select(
            pl.int_range(0, pl.len(), dtype=pl.Int64).alias("pid"),
            pl.col(names["baseline", column]).alias("gb"),
            pl.col(names["candidate", column]).alias("gc"),
        )
        with spatial_connection(inputs.ctx) as con:
            register(con, "qavf_pairs", pairs)
            a, b = "ST_GeomFromWKB(gb)", "ST_GeomFromWKB(gc)"
            try:
                measured = fetch(
                    con,
                    "SELECT pid, gb IS NOT NULL AS _b, gc IS NOT NULL AS _c, "
                    f"ST_X(ST_Centroid({a})) AS bx, ST_Y(ST_Centroid({a})) AS by, "
                    f"ST_X(ST_Centroid({b})) AS cx, ST_Y(ST_Centroid({b})) AS cy, "
                    f"{_sql_area(a, geographic)} AS baseline_area, "
                    f"{_sql_area(b, geographic)} AS candidate_area, "
                    f"{_sql_area(f'ST_Intersection({a}, {b})', geographic)} AS shared_area, "
                    f"ST_Equals({a}, {b}) AS equal FROM qavf_pairs ORDER BY pid",
                )
            finally:
                con.unregister("qavf_pairs")
        measured = measured.with_columns(
            distance(
                pl.col("bx"), pl.col("by"), pl.col("cx"), pl.col("cy"), geographic=geographic
            ).alias("distance"),
            (pl.col("baseline_area") + pl.col("candidate_area") - 2 * pl.col("shared_area"))
            .clip(lower_bound=0)
            .alias("difference"),
        )
        return pl.concat([joined.select(keys), measured.drop("pid")], how="horizontal")

    @staticmethod
    def _checks(
        inputs: LevelInputs,
        params: LocationsParams,
        moves: pl.DataFrame,
        keys: list[str],
        unit: str,
    ) -> tuple[list[CheckResult], dict[str, Any]]:
        limit = inputs.ctx.evidence_limit
        both = pl.col("_b") & pl.col("_c")
        one = pl.col("_b") != pl.col("_c")
        lone = moves.filter(one)
        evidence = inputs.evidence()
        for record in lone.sort(keys, nulls_last=True).head(limit).iter_rows(named=True):
            side = "baseline" if record["_b"] else "candidate"
            evidence.append({**{k: record[k] for k in keys}, "located": f"only in {side}"})
        evidence.count = lone.height
        unlocated = moves.filter(~both).height
        checks = [
            graded(
                "located",
                lone.height,
                f"Every matched feature is located on both sides or on neither "
                f"({plural(moves.height, 'feature')}).",
                f"{plural(lone.height, 'feature')} have a location on one side only.",
                evidence=evidence,
                total=moves.height,
                unit="features",
                metrics={"unlocated_on_both": unlocated - lone.height},
            )
        ]
        compared = moves.filter(both)
        largest: Any = compared.select(pl.col("distance").max()).item()
        moved = compared.filter(pl.col("distance") > params.max_distance)
        shown = moved.sort(["distance", *keys], descending=[True, *[False] * len(keys)])
        evidence = inputs.evidence()
        drop = ["_b", "_c"]
        for record in shown.head(limit).drop(drop).iter_rows(named=True):
            evidence.append(record)
        evidence.count = moved.height
        what = "centroid" if params.geometry is not None else "point"
        checks.append(
            graded(
                "moved",
                moved.height,
                f"No feature's {what} moved more than {params.max_distance:g} {unit} "
                f"({plural(compared.height, 'feature')}).",
                f"{plural(moved.height, 'feature')} moved more than {params.max_distance:g} "
                f"{unit} (largest {float(largest or 0):.6g}).",
                evidence=evidence,
                total=compared.height,
                unit="features",
                metrics={"max_distance": params.max_distance, "largest": largest},
            )
        )
        counts: dict[str, Any] = {
            "unlocated": unlocated,
            "moved": moved.height,
            "max_moved": None if largest is None else float(largest),  # type: ignore[arg-type]
        }
        if params.geometry is None:
            return checks, counts
        area_unit = "square metres" if unit == "metres" else f"square {unit}"
        if params.area is not None:
            judged = Materiality(params.area.rtol, params.area.atol)
            flags = [
                judged.material(b, c)
                for b, c in compared.select("baseline_area", "candidate_area").iter_rows()
            ]
            changed = compared.filter(pl.Series(flags, dtype=pl.Boolean))
            changed = changed.with_columns(
                (pl.col("candidate_area") - pl.col("baseline_area")).alias("area_change")
            ).sort(
                [pl.col("area_change").abs(), *keys],
                descending=[True, *[False] * len(keys)],
            )
            evidence = inputs.evidence()
            fields = [*keys, "baseline_area", "candidate_area", "area_change"]
            evidence.extend(changed.head(limit).select(fields).iter_rows(named=True))
            evidence.count = changed.height
            counts["area_changed"] = changed.height
            checks.append(
                graded(
                    "area",
                    changed.height,
                    f"No geometry's area changed materially (rtol={params.area.rtol:g}, "
                    f"atol={params.area.atol:g} {area_unit}).",
                    f"{plural(changed.height, 'feature')} changed area materially.",
                    evidence=evidence,
                    total=compared.height,
                    unit="features",
                )
            )
        if params.shape is not None:
            threshold = params.shape.max_difference
            reshaped = compared.filter(
                ~pl.col("equal").fill_null(False) & (pl.col("difference") > threshold)
            ).sort(["difference", *keys], descending=[True, *[False] * len(keys)])
            evidence = inputs.evidence()
            fields = [*keys, "baseline_area", "candidate_area", "difference"]
            evidence.extend(reshaped.head(limit).select(fields).iter_rows(named=True))
            evidence.count = reshaped.height
            counts["shape_changed"] = reshaped.height
            checks.append(
                graded(
                    "shape",
                    reshaped.height,
                    f"Every geometry is unchanged within {threshold:g} {area_unit} of "
                    "symmetric difference.",
                    f"{plural(reshaped.height, 'feature')} changed shape by more than "
                    f"{threshold:g} {area_unit}.",
                    evidence=evidence,
                    total=compared.height,
                    unit="features",
                )
            )
        return checks, counts
