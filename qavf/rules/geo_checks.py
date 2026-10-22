"""Coordinate QA: latitude/longitude values that can be real places.

The "spatial" rules elsewhere compare entities; this checks point coordinates
themselves: present, finite, in range, not (0, 0), inside a bounding box and,
with ``within``, inside the polygons of a zone layer or study area. ``within``
uses the shared point-in-polygon path (:func:`qavf.spatial.assign.assign`) and
needs DuckDB's spatial extension; without it that check alone is not assessed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import polars as pl
from pydantic import Field, model_validator

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.redaction import exception_details
from qavf.rules._coordinate_sql import PROBLEMS, coordinate_problem_sql
from qavf.rules._execution import normalized_plan
from qavf.rules._predicates import row_tests
from qavf.rules._row_summary import row_aliases, summarize_rows
from qavf.rules.helpers import fresh_internal, missing_expr, outcome, plural, present, usable
from qavf.settings import NumericBound, QAVFSettings
from qavf.sources.sql import quote_identifier
from qavf.spatial.assign import AssignmentPolicy, assign
from qavf.spatial.crs import DEFAULT_CRS
from qavf.spatial.extension import spatial_connection
from qavf.spatial.layers import ZoneLayer, load_zones

CATEGORY = "spatial"


class BoundingBox(RuleParams):
    """The area all points must fall in (inclusive), in degrees."""

    min_lat: NumericBound = Field(ge=-90, le=90)
    max_lat: NumericBound = Field(ge=-90, le=90)
    min_lon: NumericBound = Field(ge=-180, le=180)
    max_lon: NumericBound = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.min_lat > self.max_lat or self.min_lon > self.max_lon:
            raise ValueError("a bounding box needs min_lat <= max_lat and min_lon <= max_lon")
        return self


class PointSet(RuleParams):
    """Latitude/longitude columns of a dataset (WGS 84 degrees)."""

    name: str | None = None
    dataset: str
    latitude: str = "latitude"
    longitude: str = "longitude"
    bbox: BoundingBox | None = None
    """Points outside this box are violations (e.g. the study area)."""
    allow_missing: bool = False
    """Rows without coordinates are allowed (by default they are violations)."""
    null_island: bool = True
    """Flag (0, 0), the usual sign of a missing value written as zero."""
    within: ZoneLayer | None = None
    """Points must fall inside these polygons (a zone layer or study area, in EPSG:4326
    or transformed from its ``crs``); valid points outside are violations of a separate
    ``within`` check. Needs DuckDB's spatial extension."""
    boundary: Literal["intersects", "contains"] = "intersects"
    """Whether a point on the boundary of ``within`` is inside (``intersects``)."""

    def label(self) -> str:
        return self.name or f"{self.dataset}({self.latitude}, {self.longitude})"


class CoordinateParams(RuleParams):
    """Parameters for :class:`CoordinateRule`."""

    points: tuple[PointSet, ...] = ()


class CoordinateRule(Rule):
    """Point coordinates are valid, plausible and inside the expected area."""

    rule_id: ClassVar[str] = "GEO-001"
    name: ClassVar[str] = "Coordinate validity"
    description: ClassVar[str] = (
        "Latitudes lie in [-90, 90] and longitudes in [-180, 180], coordinates are present "
        "and finite, not (0, 0), inside the configured bounding box and, with 'within', inside "
        "the polygons of a zone layer or study area."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Out-of-range latitudes usually mean latitude and longitude were swapped, or a "
        "projected coordinate (metres) was stored as degrees; (0, 0) is almost always a "
        "missing location written as zero."
    )
    Params: ClassVar[type[RuleParams]] = CoordinateParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements:
        layers = (point.within.datasets() for point in self.params.points if point.within)
        return DataRequirements(
            frozenset(point.dataset for point in self.params.points).union(*layers)
        )

    def evaluate(self, ctx: DataContext) -> list[Finding]:
        failures: list[Finding] = []
        passed: list[str] = []
        for points in self.params.points:
            checks = [(points.label(), self._check)]
            if points.within is not None:
                checks.append((f"{points.label()} within {points.within.label()}", self._within))
            for label, check in checks:
                try:
                    finding = check(ctx, points)
                except Exception as exc:
                    finding = Finding.unassessed(
                        f"Not assessed: {label}: {exception_details(exc)}.",
                        check=label,
                        datasets=(points.dataset,),
                    )
                if finding is None:
                    passed.append(label)
                else:
                    failures.append(finding)
        noun = "check" if any(p.within for p in self.params.points) else "point set"
        return outcome(
            failures,
            passed,
            f"{plural(len(passed), noun)} valid.",
            "No coordinate columns are configured.",
        )

    def _within(self, ctx: DataContext, points: PointSet) -> Finding | None:
        """Valid points outside the ``within`` polygons (one native point-in-polygon pass)."""
        assert points.within is not None
        label = f"{points.label()} within {points.within.label()}"
        columns = (points.latitude, points.longitude)
        if reason := usable(ctx, points.dataset, columns):
            return Finding.unassessed(
                f"Not assessed: {label}: {reason}.", check=label, datasets=(points.dataset,)
            )
        with ctx.observe(points.dataset, list(dict.fromkeys(columns))) as observation:
            plan = normalized_plan(ctx, points.dataset, observation.scan())
            lat, lon = (pl.col(c).cast(pl.Float64) for c in columns)
            # Missing, non-finite and out-of-range coordinates are the coordinate check's.
            valid = lat.is_finite() & lon.is_finite() & lat.is_between(-90, 90)
            plan = plan.filter((valid & lon.is_between(-180, 180)).fill_null(False))
            zone = fresh_internal("zone", columns)
            policy = AssignmentPolicy(predicate=points.boundary, overlap="first")
            with spatial_connection(ctx) as con:
                layer = load_zones(con, ctx, points.within)
                assigned = assign(
                    plan,
                    con,
                    layer,
                    policy,
                    column=zone,
                    crs=DEFAULT_CRS,
                    collect=observation.collect,
                    x=points.longitude,
                    y=points.latitude,
                )
            if not assigned.unassigned:
                return None
            outside = assigned.plan.filter(pl.col(zone).is_null()).drop(zone)
            sample = observation.collect(
                outside.sort(list(dict.fromkeys(columns))).head(ctx.evidence_limit)
            )
        return Finding.fail(
            f"{label}: {plural(assigned.unassigned, 'row')} of {assigned.rows:,} with valid "
            f"coordinates lie outside the zones.",
            check=label,
            affected=sample,
            affected_count=assigned.unassigned,
            total_records=assigned.rows,
            unit="rows",
            metrics={"notes": list(layer.notes)} if layer.notes else {},
            datasets=(points.dataset, *points.within.datasets()),
        )

    def _check(self, ctx: DataContext, points: PointSet) -> Finding | None:
        label = points.label()
        columns = (points.latitude, points.longitude)
        if reason := usable(ctx, points.dataset, columns):
            return Finding.unassessed(
                f"Not assessed: {label}: {reason}.", check=label, datasets=(points.dataset,)
            )
        schema = ctx.schema(points.dataset)
        non_numeric = [column for column in columns if not schema[column].is_numeric()]
        if non_numeric:
            return Finding.unassessed(
                f"Not assessed: {label}: coordinate column(s) {', '.join(non_numeric)} "
                "are not numeric.",
                check=label,
                datasets=(points.dataset,),
            )
        row_name, problem_name = row_aliases(list(schema))
        selected = [*schema, row_name, problem_name]
        if ctx.execution_kind(points.dataset) == "native_sql":
            q = quote_identifier
            projection = "*, " if ctx.evidence_limit else ""
            row = f"row_number() OVER () - 1 AS {q(row_name)}, " if ctx.evidence_limit else ""
            summary = summarize_rows(
                ctx,
                points.dataset,
                problem=problem_name,
                labels=PROBLEMS,
                sample_columns=selected,
                sample_order=row_name,
                sql=f"SELECT {projection}{row}{coordinate_problem_sql(ctx, points, schema)} "
                f"AS {q(problem_name)} FROM {q(points.dataset)}",
            )
        else:
            used = {name.casefold() for name in selected}
            saved = {}
            for column in dict.fromkeys(columns):
                alias = fresh_internal("original_" + column.casefold(), used)
                used.add(alias)
                saved[column] = alias
            spec = ctx.specs.get(points.dataset)
            specs = spec.schema_spec.columns if spec is not None and spec.schema_spec else {}
            typed = pl.DataFrame(schema=schema)

            def classify(plan: pl.LazyFrame) -> pl.LazyFrame:
                if ctx.evidence_limit:
                    plan = plan.with_row_index(row_name).with_columns(
                        pl.col(column).alias(alias) for column, alias in saved.items()
                    )
                else:
                    plan = plan.select(list(saved))
                plan = plan.with_columns(
                    pl.when(missing_expr(typed, column, specs.get(column)))
                    .then(None)
                    .otherwise(pl.col(column))
                    .alias(column)
                    for column in saved
                ).with_columns(self._problem(points, schema).alias(problem_name))
                if ctx.evidence_limit:
                    plan = plan.select(
                        *(pl.col(saved.get(column, column)).alias(column) for column in schema),
                        row_name,
                        problem_name,
                    )
                return plan

            summary = summarize_rows(
                ctx,
                points.dataset,
                problem=problem_name,
                labels=PROBLEMS,
                sample_columns=selected,
                build=classify,
            )
        count = summary.invalid
        if not count:
            return None
        return Finding.fail(
            f"{label}: {plural(count, 'row')} of {summary.rows:,} have invalid coordinates.",
            check=label,
            affected=present(summary.sample, {row_name: "_row", problem_name: "problem"}),
            affected_count=count,
            total_records=summary.rows,
            unit="rows",
            metrics={
                "problems": [
                    {"problem": problem, "count": count}
                    for problem, count in sorted(summary.counts.items(), key=lambda item: -item[1])
                    if count
                ]
            },
            datasets=(points.dataset,),
        )

    @staticmethod
    def _problem(points: PointSet, schema: dict[str, pl.DataType]) -> pl.Expr:
        def within(column: str, lower: Any, upper: Any) -> pl.Expr:
            tests = row_tests({"ge": lower, "le": upper}, schema[column])
            return tests["ge"][1](column) & tests["le"][1](column)

        lat, lon = pl.col(points.latitude), pl.col(points.longitude)
        missing = lat.is_null() | lon.is_null()
        finite = pl.lit(True)
        for column in (points.latitude, points.longitude):
            if schema[column].is_float():
                missing |= pl.col(column).is_nan()
                finite &= pl.col(column).is_finite()
        lat_range = within(points.latitude, -90, 90)
        lon_range = within(points.longitude, -180, 180)
        expr: Any = pl.when(missing).then(
            pl.lit(None, dtype=pl.String) if points.allow_missing else pl.lit("missing coordinate")
        )
        expr = (
            expr.when(~finite)
            .then(pl.lit("non-finite coordinate"))
            .when(
                ~lat_range & within(points.longitude, -90, 90) & within(points.latitude, -180, 180)
            )
            .then(pl.lit("latitude out of range (latitude and longitude swapped?)"))
            .when(~lat_range)
            .then(pl.lit("latitude out of range"))
            .when(~lon_range)
            .then(pl.lit("longitude out of range"))
        )
        if points.null_island:
            expr = expr.when((lat == 0) & (lon == 0)).then(pl.lit("(0, 0): a missing location?"))
        if points.bbox is not None:
            box = points.bbox
            outside = ~within(points.latitude, box.min_lat, box.max_lat) | ~within(
                points.longitude, box.min_lon, box.max_lon
            )
            expr = expr.when(outside).then(pl.lit("outside the bounding box"))
        return expr.otherwise(pl.lit(None, dtype=pl.String))
