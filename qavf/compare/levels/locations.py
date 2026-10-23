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

from threading import Lock
from typing import Any, ClassVar

import polars as pl
from pydantic import Field, model_validator

from qavf.compare.comparators import Materiality
from qavf.compare.levels import (
    SIDES,
    ComparisonLevel,
    LevelInputs,
    graded,
    not_assessed,
    register_level,
)
from qavf.compare.results import CheckResult, LevelResult, LocationsResult
from qavf.engine.registry import RuleParams
from qavf.keys import FLOAT_EXACT_LIMIT, key_casts
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

    def required_columns(self) -> frozenset[str]:
        params: LocationsParams = self.params  # type: ignore[assignment]
        return frozenset(c for c in (params.x, params.y, params.geometry) if c is not None)

    def evaluate(self, inputs: LevelInputs, name: str) -> LocationsResult:
        params: LocationsParams = self.params  # type: ignore[assignment]
        keys = list(inputs.spec.keys)
        columns = [c for c in (params.x, params.y, params.geometry) if c is not None]
        frames: dict[str, pl.LazyFrame] = {}
        for side in SIDES:
            schema = inputs.schema(side)
            if absent := [c for c in (*keys, *columns) if c not in schema]:
                raise ValueError(f"{side} lacks column(s) {', '.join(absent)}")
            frame = inputs.scan(side).select(*keys, *columns)
            if repeated := inputs.collect(
                frame.select(pl.struct(keys).is_duplicated().sum())
            ).item():
                raise ValueError(f"the {side} repeats keys in {plural(repeated, 'row')}")
            frames[side] = frame
        schemas = {side: inputs.schema(side) for side in SIDES}
        for key in keys:

            def exact_float(key: str = key) -> bool:
                for side in SIDES:
                    value = pl.col(key)
                    valid = (value >= -FLOAT_EXACT_LIMIT) & (value <= FLOAT_EXACT_LIMIT)
                    if schemas[side][key].is_float():
                        valid = valid & value.is_finite() & (value == value.round(0))
                    invalid = value.is_not_null() & ~valid
                    if inputs.collect(frames[side].select(invalid.any())).item():
                        return False
                return True

            casts = key_casts(
                key, schemas["baseline"][key], schemas["candidate"][key], exact_float=exact_float
            )
            for side, dtype in zip(SIDES, casts, strict=True):
                if dtype is not None:
                    frames[side] = frames[side].with_columns(pl.col(key).cast(dtype))
        base, cand = frames["baseline"], frames["candidate"]
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
            # Polars 1.32 leaves NumPy-ufunc expressions with an Unknown dtype.
            # Declare the distance storage before streaming the owned IPC snapshot.
            moves = joined.select(
                *keys,
                *(pl.col(names[side, c]).alias(f"{side}_{c}") for side in SIDES for c in columns),
                located["baseline"].alias("_b"),
                located["candidate"].alias("_c"),
                distance(
                    *(pl.col(names[side, c]).cast(pl.Float64) for side in SIDES
                      for c in (params.x, params.y)),
                    geographic=geographic,
                ).cast(pl.Float64).alias("distance"),
            )  # fmt: skip
            kind = "points"
        else:
            output_schema = {key: joined.collect_schema()[key] for key in keys}
            output_schema.update({"_b": pl.Boolean(), "_c": pl.Boolean(), "equal": pl.Boolean()})
            output_schema.update(
                dict.fromkeys(
                    (
                        "bx",
                        "by",
                        "cx",
                        "cy",
                        "baseline_area",
                        "candidate_area",
                        "shared_area",
                        "distance",
                        "difference",
                    ),
                    pl.Float64(),
                )
            )
            # Polars may dispatch batches concurrently; serialize the optional native
            # extension so one comparison holds at most one connection at a time.
            geometry_lock = Lock()

            def measure_geometry(batch: pl.DataFrame) -> pl.DataFrame:
                with geometry_lock:
                    return self._geometries(inputs, params, batch, keys, names, geographic)

            moves = joined.map_batches(
                measure_geometry,
                schema=output_schema,
                streamable=True,
                predicate_pushdown=False,
                projection_pushdown=False,
                slice_pushdown=False,
            )
            kind = "geometries"
        with inputs.observations["baseline"].derived(lambda _: moves) as measured:
            checks, counts = self._checks(inputs, params, measured.scan(), keys, unit)
        matched = counts.pop("matched")
        return LocationsResult(
            level=self.name,
            name=name,
            checks=checks,
            keys=keys,
            kind=kind,
            crs=params.crs,
            unit=unit,
            matched=matched,
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
            # A non-null WKB can still be empty or have no finite centroid.
            # Match point-input presence semantics before checking movement.
            *(
                pl.all_horizontal(pl.col(f"{side}x", f"{side}y").is_finite())
                .fill_null(False)
                .alias(f"_{side}")
                for side in ("b", "c")
            ),
            distance(pl.col("bx"), pl.col("by"), pl.col("cx"), pl.col("cy"), geographic=geographic)
            .cast(pl.Float64)
            .alias("distance"),
            (pl.col("baseline_area") + pl.col("candidate_area") - 2 * pl.col("shared_area"))
            .clip(lower_bound=0)
            .alias("difference"),
        )
        return pl.concat([joined.select(keys), measured.drop("pid")], how="horizontal").select(
            *keys,
            "_b",
            "_c",
            "equal",
            "bx",
            "by",
            "cx",
            "cy",
            "baseline_area",
            "candidate_area",
            "shared_area",
            "distance",
            "difference",
        )

    @staticmethod
    def _checks(
        inputs: LevelInputs,
        params: LocationsParams,
        moves: pl.LazyFrame,
        keys: list[str],
        unit: str,
    ) -> tuple[list[CheckResult], dict[str, Any]]:
        limit = inputs.ctx.evidence_limit
        both = pl.col("_b") & pl.col("_c")
        one = pl.col("_b") != pl.col("_c")
        moved_flag = both & (pl.col("distance") > params.max_distance)
        aggregations = [
            pl.len().alias("matched"),
            one.sum().alias("lone"),
            (~both).sum().alias("unlocated"),
            both.sum().alias("compared"),
            moved_flag.sum().alias("moved"),
            pl.col("distance").filter(both).max().alias("largest"),
        ]
        area_flag = None
        if params.geometry is not None and params.area is not None:
            judged = Materiality(params.area.rtol, params.area.atol)
            area_flag = pl.struct("baseline_area", "candidate_area").map_elements(
                lambda row: judged.material(row["baseline_area"], row["candidate_area"]),
                return_dtype=pl.Boolean,
            )
            aggregations.append((both & area_flag).sum().alias("area"))
        shape_flag = None
        if params.geometry is not None and params.shape is not None:
            shape_flag = ~pl.col("equal").fill_null(False) & (
                pl.col("difference") > params.shape.max_difference
            )
            aggregations.append((both & shape_flag).sum().alias("shape"))
        totals = inputs.collect(moves.select(aggregations)).row(0, named=True)
        matched, compared_count = int(totals["matched"]), int(totals["compared"])
        lone_count, moved_count = int(totals["lone"] or 0), int(totals["moved"] or 0)
        largest = totals["largest"]
        lone = moves.filter(one)
        evidence = inputs.evidence()
        for record in inputs.collect(lone.sort(keys, nulls_last=True).head(limit)).iter_rows(
            named=True
        ):
            side = "baseline" if record["_b"] else "candidate"
            evidence.append({**{k: record[k] for k in keys}, "located": f"only in {side}"})
        evidence.count = lone_count
        unlocated = int(totals["unlocated"] or 0)
        checks = [
            graded(
                "located",
                lone_count,
                f"Every matched feature is located on both sides or on neither "
                f"({plural(matched, 'feature')}).",
                f"{plural(lone_count, 'feature')} have a location on one side only.",
                evidence=evidence,
                total=matched,
                unit="features",
                metrics={"unlocated_on_both": unlocated - lone_count},
            )
        ]
        compared = moves.filter(both)
        moved = compared.filter(pl.col("distance") > params.max_distance)
        shown = moved.sort(["distance", *keys], descending=[True, *[False] * len(keys)])
        evidence = inputs.evidence()
        drop = ["_b", "_c"]
        for record in inputs.collect(shown.head(limit).drop(drop)).iter_rows(named=True):
            evidence.append(record)
        evidence.count = moved_count
        what = "centroid" if params.geometry is not None else "point"
        checks.append(
            graded(
                "moved",
                moved_count,
                f"No feature's {what} moved more than {params.max_distance:g} {unit} "
                f"({plural(compared_count, 'feature')}).",
                f"{plural(moved_count, 'feature')} moved more than {params.max_distance:g} "
                f"{unit} (largest {float(largest or 0):.6g}).",
                evidence=evidence,
                total=compared_count,
                unit="features",
                metrics={"max_distance": params.max_distance, "largest": largest},
            )
        )
        counts: dict[str, Any] = {
            "matched": matched,
            "unlocated": unlocated,
            "moved": moved_count,
            "max_moved": None if largest is None else float(largest),  # type: ignore[arg-type]
        }
        if not matched:
            checks = [not_assessed(check.check, "no matched features") for check in checks]
        elif not compared_count:
            checks[1] = not_assessed("moved", "no matched features located on both sides")
        if params.geometry is None:
            return checks, counts
        area_unit = "square metres" if unit == "metres" else f"square {unit}"
        if params.area is not None:
            assert area_flag is not None
            changed = compared.filter(area_flag)
            changed_count = int(totals["area"] or 0)
            changed = changed.with_columns(
                (pl.col("candidate_area") - pl.col("baseline_area")).alias("area_change")
            ).sort(
                [pl.col("area_change").abs(), *keys],
                descending=[True, *[False] * len(keys)],
            )
            evidence = inputs.evidence()
            fields = [*keys, "baseline_area", "candidate_area", "area_change"]
            evidence.extend(
                inputs.collect(changed.head(limit).select(fields)).iter_rows(named=True)
            )
            evidence.count = changed_count
            counts["area_changed"] = changed_count
            checks.append(
                graded(
                    "area",
                    changed_count,
                    f"No geometry's area changed materially (rtol={params.area.rtol:g}, "
                    f"atol={params.area.atol:g} {area_unit}).",
                    f"{plural(changed_count, 'feature')} changed area materially.",
                    evidence=evidence,
                    total=compared_count,
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
            evidence.extend(
                inputs.collect(reshaped.head(limit).select(fields)).iter_rows(named=True)
            )
            evidence.count = int(totals["shape"] or 0)
            counts["shape_changed"] = int(totals["shape"] or 0)
            checks.append(
                graded(
                    "shape",
                    int(totals["shape"] or 0),
                    f"Every geometry is unchanged within {threshold:g} {area_unit} of "
                    "symmetric difference.",
                    f"{plural(int(totals['shape'] or 0), 'feature')} changed shape by more than "
                    f"{threshold:g} {area_unit}.",
                    evidence=evidence,
                    total=compared_count,
                    unit="features",
                )
            )
        if not compared_count:
            checks = [
                check
                if check.check == "located" and matched
                else not_assessed(check.check, "no usable matched locations")
                for check in checks
            ]
        return checks, counts
