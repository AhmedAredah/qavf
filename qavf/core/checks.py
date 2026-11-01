"""Composable SQL implementations of declared input contracts."""

from __future__ import annotations

import functools
import math
import operator
from fractions import Fraction

from qavf.applicability import AllOrNoneApplicability
from qavf.core.aggregates import ExactAggregates, observed_number
from qavf.core.contracts import canonical_key, condition_dependencies
from qavf.core.key_policy import FLOATS, key_casts
from qavf.core.policy import (
    ColumnValue,
    DecimalColumnScale,
    NativeValueDomain,
    decimal_places_condition,
    native_bound_condition,
    native_literal,
    normalized,
)
from qavf.engine.severity import Status
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit


def _unclaimed_name(role, columns):
    """Prefix a generated column name until it collides with no data column."""
    while role.casefold() in {column.casefold() for column in columns}:
        role = "qavf_" + role
    return role


def _sample_records(out, evidence_columns, summary_table, field):
    """Fetch the sampled key records stored in one summary-table list field."""
    return out.records(
        f"SELECT {evidence_columns} FROM "
        f"(SELECT unnest({qi(field)}) AS key FROM {qi(summary_table)}) sample_keys"
    )


class DatasetChecks:
    def __init__(self, collector, loaded, spec):
        self.out = collector
        self.con = collector.connection
        self.data = loaded
        self.spec = spec
        self.schema = spec.schema_spec
        self.values = {}
        if self.schema:
            scales = DecimalColumnScale.resolve_many(
                self.con, loaded.table, loaded.types, self.schema.columns
            )
            self.values = {
                name: ColumnValue.compile(
                    name, loaded.types[name], column, decimal_scale=scales.get(name)
                )
                for name, column in self.schema.columns.items()
                if name in loaded.columns
            }
        self.typed = loaded.table + "_typed"
        columns = [qi(loaded.row_column)]
        columns += [f"{self.canonical(name)} AS {qi(name)}" for name in loaded.columns]
        self.con.execute(
            f"CREATE TEMP VIEW {qi(self.typed)} AS SELECT "
            f"{', '.join(columns)} FROM {qi(loaded.table)}"
        )
        # SQL retains explicitly N/A tokens for caller-authored applicability tests.
        # Other values use the same missing/conversion contract as value checks.
        self.expression_table = loaded.table + "_expressions"
        expressions = [qi(loaded.row_column)]
        for column in loaded.columns:
            if column not in self.values:
                expression = qi(column)
            else:
                policy = self.values[column]
                if self.schema.columns[column].na_values and loaded.types[column].startswith(
                    "VARCHAR"
                ):
                    expression = (
                        f"CASE WHEN {policy.na} THEN {qi(column)} ELSE "
                        f"CAST({policy.value} AS VARCHAR) END"
                    )
                else:
                    expression = policy.value
            expressions.append(f"{expression} AS {qi(column)}")
        self.con.execute(
            f"CREATE TEMP VIEW {qi(self.expression_table)} AS SELECT "
            f"{', '.join(expressions)} FROM {qi(loaded.table)}"
        )
        self.con.execute(
            f"CREATE TEMP VIEW {qi(loaded.name)} AS SELECT "
            f"{', '.join(qi(c) for c in loaded.columns) or qi(loaded.row_column)} FROM "
            f"{qi(self.expression_table)}"
        )

    def run(self):
        if self.schema is None:
            return
        if self.out.enabled("INP-002", self.data.name):
            self.shape()
        self.columns()
        if self.out.enabled("INP-004", self.data.name):
            self.keys()
        if self.out.enabled("INP-009", self.data.name):
            self.applicability()

    def canonical(self, column):
        if column in self.values:
            return self.values[column].value
        return canonical_key(qi(column), self.data.types[column])

    def shape(self):
        data, schema = self.data, self.schema
        columns = getattr(data, "original_columns", ()) or data.columns
        problems = []
        for name, spec in schema.columns.items():
            if spec.required and name not in columns:
                problems.append({"column": name, "problem": "missing required column"})
        if schema.strict:
            problems.extend(
                {"column": c, "problem": "unexpected column"}
                for c in columns
                if c not in schema.columns
            )
        if data.rows < schema.min_rows or (
            schema.max_rows is not None and data.rows > schema.max_rows
        ):
            problems.append({"problem": "row count outside declared bounds", "rows": data.rows})
        self.out.add(
            "INP-002",
            "schema",
            data.name,
            count=len(problems),
            records=problems,
            parameters={
                "min_rows": schema.min_rows,
                "max_rows": schema.max_rows,
                "strict": schema.strict,
            },
            unit="problems",
        )

    def columns(self):
        data = self.data
        if not any(self.out.enabled(rule, data.name) for rule in ("INP-002", "INP-003", "INP-009")):
            return
        self.pending_rows = []
        self.scope_totals = {}
        population = dict.fromkeys(self.values, 0)
        names = [c for c in self.values if self.schema.columns[c].na_values]
        if names:
            aggregates = ", ".join(f"count(*) FILTER (WHERE {self.values[c].na})" for c in names)
            counts = self.con.execute(f"SELECT {aggregates} FROM {qi(data.table)}").fetchone()
            population.update(zip(names, counts, strict=True))
        self.population = population
        for column, spec in self.schema.columns.items():
            if column not in self.values:
                continue
            value = self.values[column]
            count_na = population[column]
            eligible = data.rows - count_na
            metrics = {"na_count": count_na, "applicable_count": eligible}
            for name, condition, parameters in (
                (
                    "type",
                    value.invalid,
                    {
                        "dtype": spec.dtype,
                        "number_grammar": spec.number_grammar.model_dump(),
                        "trim_whitespace": spec.trim_whitespace,
                    },
                ),
                ("nullable", value.missing, {"nullable": False}),
                ("finite", f"isinf({value.value})", {"allow_infinite": False}),
            ):
                if name == "type" and spec.dtype is None:
                    continue
                if name == "nullable" and spec.nullable:
                    continue
                if name == "finite" and (value.dtype not in FLOATS or spec.allow_infinite):
                    continue
                rule_id = "INP-002" if name == "type" else "INP-003"
                if self.out.enabled(rule_id, data.name):
                    self.pending_rows.append(
                        (rule_id, name, column, condition, parameters, metrics, eligible, None)
                    )
            if spec.unit_column and self.out.enabled("INP-003", data.name):
                units = spec.allowed_units or ([spec.unit] if spec.unit else [])
                if spec.unit_column not in data.columns:
                    self.out.add(
                        "INP-003",
                        "unit",
                        data.name,
                        [column, spec.unit_column],
                        status=Status.ERROR,
                        message="Unit column is missing",
                    )
                elif units:
                    try:
                        self.out.row_check(
                            "INP-003",
                            "unit",
                            data,
                            f"NOT ({value.na}) AND NOT ({value.missing}) AND "
                            f"({qi(spec.unit_column)} IS NULL OR "
                            f"{qi(spec.unit_column)} NOT IN "
                            f"({', '.join(lit(u) for u in units)}))",
                            columns=[column, spec.unit_column],
                            parameters={"allowed_units": units},
                        )
                    except Exception as exc:
                        self.error(
                            "INP-003",
                            "unit",
                            [column, spec.unit_column],
                            exc,
                            {"allowed_units": units},
                        )
            if self.out.enabled("INP-003", data.name):
                for check in spec.all_checks():
                    for kind in check.constraints():
                        try:
                            self.constraint(column, spec, check, metrics, only_kind=kind)
                        except Exception as exc:
                            self.error(
                                "INP-003",
                                kind,
                                [column],
                                exc,
                                check.model_dump(exclude_none=True),
                                check.severity,
                            )
        if not self.pending_rows:
            return
        aggregates = ", ".join(f"count(*) FILTER (WHERE {item[3]})" for item in self.pending_rows)
        try:
            counts = self.con.execute(f"SELECT {aggregates} FROM {qi(data.table)}").fetchone()
        except Exception:
            # A bad contract must not suppress independent valid contracts.
            counts = []
            for rule_id, name, column, bad, params, _, _, severity in self.pending_rows:
                try:
                    counts.append(
                        self.con.execute(
                            f"SELECT count(*) FROM {qi(data.table)} WHERE {bad}"
                        ).fetchone()[0]
                    )
                except Exception as exc:
                    self.error(rule_id, name, [column], exc, params, severity)
                    counts.append(None)
        limit = self.out.settings.engine.max_affected_records
        for (rule_id, name, column, bad, params, metrics, total, severity), count in zip(
            self.pending_rows, counts, strict=True
        ):
            if count is None:
                continue
            records = []
            if count and limit:
                selected = (
                    list(data.columns)
                    if self.schema.evidence_columns == "all"
                    else list(
                        dict.fromkeys(
                            [
                                column,
                                *self.schema.primary_key,
                                *condition_dependencies(params.get("where"), data.columns),
                            ]
                        )
                    )
                )
                selected = [c for c in selected if c in data.columns]
                records = self.out.records(
                    f"SELECT {qi(data.row_column)}, "
                    f"{', '.join(qi(c) for c in selected)} FROM "
                    f"{qi(data.table)} WHERE {bad} ORDER BY "
                    f"{qi(data.row_column)} LIMIT {limit}"
                )
            self.out.add(
                rule_id,
                name,
                data.name,
                [column],
                count=count,
                total=total,
                records=records,
                parameters=params,
                metrics=metrics,
                severity=severity,
            )

    def error(self, rule, name, columns, exc, parameters=None, severity=None):
        self.out.add(
            rule,
            name,
            self.data.name,
            columns,
            status=Status.ERROR,
            message=str(exc),
            parameters=parameters,
            severity=severity,
        )

    def constraint(self, column, spec, check, metrics, only_kind=None):
        data, out = self.data, self.out
        value = self.values[column]
        expression = normalized(value.value, check.normalize)
        where = f"({check.where}) IS TRUE" if check.where else "TRUE"
        # Where clauses apply to canonical values, while N/A masks retain raw tokens.
        if check.where:
            selected = (
                f"{qi(data.row_column)} IN (SELECT {qi(data.row_column)} "
                f"FROM {qi(self.typed)} WHERE {where})"
            )
        else:
            selected = "TRUE"
        scope = f"({selected}) AND NOT ({value.na})"
        if check.where and scope not in self.scope_totals:
            self.scope_totals[scope] = self.con.execute(
                f"SELECT count(*) FROM {qi(data.table)} WHERE {scope}"
            ).fetchone()[0]
        total = self.scope_totals[scope] if check.where else metrics["applicable_count"]
        if check.where and total < check.min_applicable_rows:
            out.add(
                "INP-003",
                check.name or "applicability",
                data.name,
                [column],
                status=Status.ERROR,
                total=total,
                parameters={"where": check.where, "min_applicable_rows": check.min_applicable_rows},
                message="Fewer applicable rows than required",
            )
            return
        row_checks = []
        for kind, expected in check.constraints().items():
            if only_kind is not None and kind != only_kind:
                continue
            params = {
                kind: expected.model_dump() if hasattr(expected, "model_dump") else expected,
                "where": check.where,
                "normalize": list(check.normalize),
                "name": check.name,
                "each_allowed_once": check.each_allowed_once,
                "convert_options": check.convert_options,
            }
            name = kind
            bad = None
            if kind in ("ge", "le", "gt", "lt"):
                bad = f"NOT ({native_bound_condition(expression, value.dtype, kind, expected)})"
            elif kind in ("isin", "notin"):
                domain_dtype = value.dtype
                usable, foreign = NativeValueDomain(domain_dtype).members(expected)
                if foreign:
                    out.add(
                        "INP-003",
                        kind,
                        data.name,
                        [column],
                        status=Status.ERROR,
                        parameters=params,
                        severity=check.severity,
                        message=(
                            f"Options are incompatible with the converted column type: {foreign!r}"
                        ),
                    )
                    continue
                allowed = [normalized(native_literal(v), check.normalize) for v in usable]
                op = "NOT IN" if kind == "isin" else "IN"
                bad = (
                    f"{expression} {op} ({', '.join(allowed)})"
                    if allowed
                    else ("TRUE" if kind == "isin" else "FALSE")
                )
            elif kind == "regex":
                from qavf.regex_policy import require_native_regex

                require_native_regex(expected)
                bad = f"NOT regexp_full_match({expression}, {lit(expected)})"
            elif kind in ("min_length", "max_length"):
                bad = f"length({expression}) {'<' if kind == 'min_length' else '>'} {expected}"
            elif kind == "integral":
                bad = f"{expression} != trunc({expression})"
            elif kind == "max_decimals":
                bad = f"NOT ({decimal_places_condition(expression, value.dtype, expected)})"
            elif kind == "not_null":
                bad = value.missing
            elif kind == "unique":
                # Null/N/A are outside uniqueness, matching the existing value contract.
                bad = (
                    f"{expression} IN (SELECT {expression} FROM "
                    f"{qi(data.table)} WHERE {scope} GROUP BY {expression} "
                    f"HAVING count(*) > 1)"
                )
            if bad is not None:
                if kind != "not_null":
                    bad = f"({expression}) IS NOT NULL AND ({bad})"
                if (
                    kind == "notin"
                    and value.dtype in ("FLOAT", "REAL", "DOUBLE")
                    and any(isinstance(v, float) and math.isnan(v) for v in expected)
                ):
                    bad = f"({bad}) OR isnan(try_cast({qi(column)} AS DOUBLE))"
                row_checks.append((name, f"({scope}) AND ({bad})", params))
                continue
            if kind == "max_null_fraction":
                count = self.con.execute(
                    f"SELECT count(*) FROM {qi(data.table)} WHERE ({scope}) AND ({value.missing})"
                ).fetchone()[0]
                actual = Fraction(count, total) if total else Fraction()
                failed = actual is not None and actual > expected
            else:
                actual = ExactAggregates(
                    self.con, data.table, expression, value.dtype, scope
                ).statistic(kind)
                bounds = expected.model_dump(exclude_none=True)
                failed = actual is not None and any(
                    not getattr(operator, op)(actual, bound) for op, bound in bounds.items()
                )
            out.add(
                "INP-003",
                name,
                data.name,
                [column],
                count=int(failed),
                total=1,
                parameters=params,
                metrics={**metrics, "observed": observed_number(actual)},
                severity=check.severity,
                status=Status.ERROR if actual is None else None,
                message=(
                    f"{kind} is undefined for the applicable nonmissing population"
                    if actual is None
                    else None
                ),
                unit="columns",
            )
        for name, bad, params in row_checks:
            self.pending_rows.append(
                ("INP-003", name, column, bad, params, metrics, total, check.severity)
            )

    def keys(self):
        schema, data = self.schema, self.data
        if schema.primary_key:
            try:
                self.key(schema.primary_key, "forbid", "primary_key")
            except Exception as exc:
                self.error("INP-004", "key.primary", schema.primary_key, exc)
        for key in schema.alternate_keys:
            try:
                self.key(key.columns, key.nulls, "unique_key", key.where, key.name)
            except Exception as exc:
                self.error("INP-004", "key.unique", key.columns, exc, key.model_dump())
        if schema.unique_rows:
            try:
                self.key(list(data.columns), "equal", "unique_rows")
            except Exception as exc:
                self.error("INP-004", "key.unique_rows", data.columns, exc)
        for contract in schema.grouped_values:
            try:
                self.grouped_values(contract)
            except Exception as exc:
                self.error(
                    "INP-004",
                    "key.grouped_values",
                    [*contract.group_by, contract.column],
                    exc,
                    contract.model_dump(),
                )

    def key(self, columns, nulls, name, where=None, contract_name=None):
        """Summarize canonical keys once and stream positions for only their sample."""
        from uuid import uuid4

        from qavf.duplicate_evidence import DuplicateEvidencePolicy

        out, data = self.out, self.data
        if not out.enabled("INP-004", data.name):
            return
        primary = name == "primary_key"
        whole_rows = name == "unique_rows"
        check_id = "key.primary" if primary else "key.unique_rows" if whole_rows else "key.unique"
        label = data.name if primary else f"{data.name}: unique ({', '.join(columns)})"
        cap = self.schema.duplicate_rows_per_key
        parameters = {"nulls": nulls, "rows_per_key": cap}
        if where is not None:
            parameters["where"] = where
        if contract_name is not None:
            parameters["name"] = contract_name
        if any(c not in data.columns for c in columns):
            out.add(
                "INP-004",
                check_id,
                data.name,
                columns,
                status=Status.FAIL,
                count=sum(c not in data.columns for c in columns),
                unit="columns",
                check=label,
                message="Key column is missing",
                parameters=parameters,
            )
            return

        if whole_rows:
            row_name = _unclaimed_name("_row", columns)
            problem = _unclaimed_name("violates", columns)
            selected = ", ".join(qi(column) for column in columns)
            duplicated = _unclaimed_name("__qavf_duplicated", columns)
            rows = (
                f"SELECT {selected}, {qi(data.row_column)} AS {qi(row_name)}, "
                f"count(*) OVER (PARTITION BY {selected}) > 1 AS {qi(duplicated)} "
                f"FROM {qi(data.table)}"
            )
            count = self.con.execute(
                f"SELECT count(*) FROM ({rows}) source WHERE {qi(duplicated)}"
            ).fetchone()[0]
            limit = out.settings.engine.max_affected_records
            records = (
                out.records(
                    f"SELECT {selected}, {qi(row_name)}, 'duplicate row' AS {qi(problem)} "
                    f"FROM ({rows}) source WHERE {qi(duplicated)} ORDER BY "
                    f"{qi(row_name)} LIMIT {limit}"
                )
                if count and limit
                else []
            )
            out.add(
                "INP-004",
                "key.unique_rows",
                data.name,
                columns,
                count=count,
                total=data.rows,
                records=records,
                unit="rows",
                affected_rows=count,
                check=f"{data.name}: duplicate rows",
                parameters={},
            )
            return

        frequency = _unclaimed_name("occurrences", columns)
        positions = _unclaimed_name("rows", columns)
        row = qi(data.row_column)
        keys = ", ".join(qi(c) for c in columns)
        order = ", ".join(f"{qi(c)} NULLS FIRST" for c in columns)
        missing = " OR ".join(f"{qi(c)} IS NULL" for c in columns)
        eligible = "TRUE" if nulls == "equal" else f"NOT ({missing})"
        na = (
            " OR ".join(self.values[c].na for c in columns if c in self.values)
            if not whole_rows
            else "FALSE"
        )
        na = na or "FALSE"
        selected = self.selection(where)
        exempt_na = (
            self.con.execute(
                f"SELECT count(*) FILTER (WHERE {na}) FROM {qi(data.table)} WHERE {selected}"
            ).fetchone()[0]
            if any(self.schema.columns[c].na_values for c in columns if c in self.schema.columns)
            and not whole_rows
            else 0
        )
        projection = ", ".join(f"{self.canonical(c)} AS {qi(c)}" for c in columns)
        source = (
            f"SELECT {row}, {projection} FROM {qi(data.table)} WHERE NOT ({na}) AND ({selected})"
        )
        groups = (
            f"SELECT {keys}, count(*)::UBIGINT AS {qi(frequency)} FROM "
            f"({source}) source GROUP BY {keys}"
        )
        duplicates = f"({eligible}) AND {qi(frequency)} > 1"
        sample_limit = out.settings.engine.max_affected_records
        summary_table = "__qavf_key_summary_" + uuid4().hex
        # Retain only one scalar summary plus bounded samples in a temporary
        # table. Its nested key values keep exact DECIMAL/TIMESTAMP_NS types for
        # the lookup, without reconstructing SQL literals through Python.
        try:
            self.con.execute(
                f"CREATE TEMP TABLE {qi(summary_table)} AS WITH grouped AS MATERIALIZED ({groups}) "
                f"SELECT count(*) FILTER (WHERE {eligible})::UBIGINT AS total_keys, "
                f"count(*) FILTER (WHERE {duplicates})::UBIGINT AS duplicated_keys, "
                f"coalesce(sum({qi(frequency)}) FILTER (WHERE "
                f"{duplicates}), 0)::UBIGINT AS involved, "
                f"coalesce(sum({qi(frequency)}) FILTER (WHERE {missing}), "
                f"0)::UBIGINT AS null_rows, "
                f"coalesce(sum({qi(frequency)}), 0)::UBIGINT AS scoped_rows, "
                f"(SELECT list(s) FROM (SELECT * FROM grouped WHERE "
                f"{duplicates} ORDER BY {order} LIMIT {sample_limit}) s) "
                f"AS sample, "
                f"(SELECT list(s) FROM (SELECT * FROM grouped WHERE "
                f"{missing} ORDER BY {order} LIMIT "
                f"{sample_limit if nulls == 'forbid' else 0}) s) AS "
                f"null_sample "
                f"FROM grouped"
            )
            total, duplicate_count, involved, null_count, scoped_rows = self.con.execute(
                f"SELECT total_keys, duplicated_keys, involved, null_rows, scoped_rows "
                f"FROM {qi(summary_table)}"
            ).fetchone()
            evidence_columns = (
                ", ".join(f"key.{qi(c)} AS {qi(c)}" for c in columns)
                + f", key.{qi(frequency)} AS {qi(frequency)}"
            )
            records = (
                _sample_records(out, evidence_columns, summary_table, "sample")
                if duplicate_count
                else []
            )
            null_records = (
                _sample_records(out, evidence_columns, summary_table, "null_sample")
                if nulls == "forbid" and null_count
                else []
            )
            metrics = {
                "duplicated_keys": duplicate_count,
                "rows_involved": involved,
                "surplus_rows": involved - duplicate_count,
            }
            key_label = f"{data.name} ({', '.join(columns)})"
            if exempt_na:
                metrics["exempt_na_rows"] = {key_label: exempt_na}
            if nulls == "exempt" and null_count:
                metrics["rows_exempt_null_unique_keys"] = {key_label: null_count}
            if nulls == "forbid" and null_count:
                out.add(
                    "INP-004",
                    check_id + ".nulls",
                    data.name,
                    columns,
                    count=null_count,
                    total=scoped_rows + exempt_na if where else data.rows,
                    records=null_records,
                    parameters=parameters,
                    metrics=(
                        {"exempt_na_rows": metrics["exempt_na_rows"]}
                        if not duplicate_count and "exempt_na_rows" in metrics
                        else None
                    ),
                    check=f"{data.name}: null keys" if primary else f"{key_label}: null keys",
                    message=f"{null_count} rows have a null key value.",
                )
            from qavf.audit_codec import evidence_json_size

            position_metadata = {
                **metrics,
                "duplicate_rows_per_key": cap,
                "duplicate_positions_available": False,
                "duplicate_positions_complete": False,
                "duplicate_positions_truncated": False,
            }
            position_budget = max(
                0,
                out.settings.engine.max_evidence_bytes - evidence_json_size(position_metadata) - 64,
            )
            accumulated = DuplicateEvidencePolicy(cap, position_budget).accumulator(
                records,
                frequency=frequency,
                positions=positions,
            )
            if records:
                wanted = (
                    f"SELECT generate_subscripts(sample, 1) - 1 AS "
                    f"sample_index, unnest(sample) AS key "
                    f"FROM {qi(summary_table)}"
                )
                wanted = f"SELECT * FROM ({wanted}) sampled WHERE sample_index < {len(records)}"
                match = " AND ".join(
                    f"source.{qi(c)} IS NOT DISTINCT FROM wanted.key.{qi(c)}" for c in columns
                )
                joined = f"FROM ({source}) source JOIN ({wanted}) wanted ON {match}"
                query = (
                    f"SELECT wanted.sample_index, source.{row} AS position {joined} "
                    f"ORDER BY wanted.sample_index, position"
                    if cap == "all"
                    else (
                        f"SELECT wanted.sample_index, unnest(min(source.{row}, "
                        f"{cap})) AS position {joined} GROUP BY wanted.sample_index "
                        f"ORDER BY wanted.sample_index, position"
                    )
                )
                try:
                    cursor = self.con.execute(query)
                    keep = True
                    while keep and (batch := cursor.fetchmany(1024)):
                        for index, position in batch:
                            if not accumulated.append(index, position):
                                keep = False
                                break
                except Exception:
                    accumulated.available = False
            metrics.update(accumulated.metrics)
            # Preserve the old single null-failure result when no duplicate key
            # exists; successful datasets still have an explicit key identity.
            if duplicate_count or not (nulls == "forbid" and null_count):
                out.add(
                    "INP-004",
                    check_id,
                    data.name,
                    columns,
                    count=duplicate_count,
                    total=total,
                    records=records,
                    unit="keys",
                    affected_rows=involved,
                    check=label,
                    parameters=parameters,
                    metrics=metrics,
                )
        finally:
            self.con.execute(f"DROP TABLE IF EXISTS {qi(summary_table)}")

    def selection(self, where):
        if not where:
            return "TRUE"
        row = qi(self.data.row_column)
        return f"{row} IN (SELECT {row} FROM {qi(self.typed)} WHERE ({where}) IS TRUE)"

    def _grouped_literals(self, column, values):
        """Render a grouped-value domain as exact typed literals, or reject it."""
        from qavf.settings import ColumnCheck

        data = self.data
        dtype = self.values[column].dtype if column in self.values else data.types[column]
        if column in self.schema.columns:
            values = (
                self.schema.columns[column]
                ._resolved_check(
                    ColumnCheck(
                        isin=values,
                        convert_options=self.schema.columns[column].dtype
                        in ("int", "float", "decimal"),
                    )
                )
                .isin
            )
        admitted, foreign = NativeValueDomain(dtype).members(values)
        if foreign or len(admitted) != len(values):
            raise ValueError(f"Grouped-value domain for {column!r} is not losslessly representable")
        return [f"CAST({native_literal(v)} AS {dtype})" for v in admitted]

    def _applicable_rows(self, columns):
        """SQL predicate admitting only rows where none of the columns is N/A."""
        data = self.data
        row = qi(data.row_column)
        predicates = [
            self.values[c].na
            for c in columns
            if c in self.values and self.schema.columns[c].na_values
        ]
        if not predicates:
            return "TRUE"
        na = " OR ".join(predicates)
        return f"{row} NOT IN (SELECT {row} FROM {qi(data.table)} WHERE {na})"

    def grouped_values(self, spec):
        from qavf.grouped_values import GroupedValuePlan

        data, out = self.data, self.out
        columns = [*spec.group_by, spec.column]
        if any(c not in data.columns for c in columns):
            raise ValueError("Grouped-value column is missing")
        domain = self._grouped_literals(spec.column, spec.allowed_values)
        expected = [
            [
                "NULL" if value is None else self._grouped_literals(column, [value])[0]
                for column, value in zip(spec.group_by, group, strict=True)
            ]
            for group in spec.expected_groups or []
        ]
        plan = GroupedValuePlan.build(
            spec,
            f"SELECT * FROM {qi(self.typed)}",
            domain,
            expected,
            eligible_sql=self._applicable_rows(spec.group_by),
            value_present_sql=(
                f"{qi(spec.column)} IS NOT NULL AND ({self._applicable_rows([spec.column])})"
            ),
        )
        count = self.con.execute(
            f"SELECT count(*) FROM ({plan.violations_sql}) violations"
        ).fetchone()[0]
        total = self.con.execute(plan.total_sql).fetchone()[0]
        records = (
            out.records(
                f"SELECT * FROM ({plan.violations_sql}) violations ORDER BY "
                f"{', '.join(qi(c) + ' NULLS FIRST' for c in columns)} LIMIT "
                f"{out.settings.engine.max_affected_records}"
            )
            if count
            else []
        )
        out.add(
            "INP-004",
            "key.grouped_values",
            data.name,
            columns,
            count=count,
            total=total,
            records=records,
            parameters=spec.model_dump(),
            unit="group values",
        )

    def applicability(self):
        if not hasattr(self, "population"):
            names = [c for c in self.values if self.schema.columns[c].na_values]
            aggregates = ", ".join(f"count(*) FILTER (WHERE {self.values[c].na})" for c in names)
            counts = (
                self.con.execute(f"SELECT {aggregates} FROM {qi(self.data.table)}").fetchone()
                if names
                else ()
            )
            self.population = dict.fromkeys(self.values, 0)
            self.population.update(zip(names, counts, strict=True))
        for column, spec in self.schema.columns.items():
            if not spec.na_values or column not in self.values:
                continue
            na_count = self.population[column]
            self.out.add(
                "INP-009",
                "N/A population",
                self.data.name,
                [column],
                check=f"{self.data.name}.{column}: N/A population",
                check_id="applicability.population",
                total=self.data.rows,
                parameters={
                    "na_values": list(spec.na_values),
                    "trim_whitespace": spec.trim_whitespace,
                },
                metrics={"na_count": na_count, "applicable_count": self.data.rows - na_count},
                message=(
                    f"{self.data.name}.{column}: {na_count:,} of "
                    f"{self.data.rows:,} values are not applicable."
                ),
            )
        for group in self.schema.na_groups:
            parameters = {
                "na_values": {column: self.schema.columns[column].na_values for column in group}
            }
            if any(column not in self.data.columns for column in group):
                self.out.add(
                    "INP-009",
                    "applicability.all_or_none",
                    self.data.name,
                    group,
                    status=Status.ERROR,
                    parameters=parameters,
                    message="N/A group column is missing",
                )
                continue
            rule = AllOrNoneApplicability.from_schema(group, self.schema)
            self.out.row_check(
                "INP-009",
                "applicability.all_or_none",
                self.data,
                rule.sql(),
                columns=group,
                parameters=parameters,
            )

    def expressions(self):
        if not self.out.enabled("INP-006", self.data.name):
            return
        for check in self.spec.expressions:
            parameters = {"expr": check.expr, "null_as": check.null_as, "name": check.name}
            try:
                bad = (
                    f"({check.expr}) IS NOT TRUE"
                    if check.null_as == "fail"
                    else f"({check.expr}) IS FALSE"
                )
                self.out.row_check(
                    "INP-006",
                    "expression",
                    self.data,
                    bad,
                    table=self.expression_table,
                    columns=self.data.columns,
                    parameters=parameters,
                    severity=check.severity,
                )
            except Exception as exc:
                self.error(
                    "INP-006", "expression", self.data.columns, exc, parameters, check.severity
                )

    def foreign_keys(self, datasets):
        if not self.out.enabled("INP-005", self.data.name):
            return
        for fk in self.spec.foreign_keys:
            try:
                self.foreign_key(fk, datasets[fk.ref_dataset])
            except Exception as exc:
                self.error("INP-005", "foreign_key", fk.columns, exc, fk.model_dump(), fk.severity)

    def _relation(self, side, fields):
        """Select one side's applicable key rows and count its N/A exemptions."""
        predicates = [
            side.values[c].na
            for c in fields
            if c in side.values and side.schema.columns[c].na_values
        ]
        if not predicates:
            return f"SELECT * FROM {qi(side.typed)}", 0
        na = " OR ".join(predicates)
        row = qi(side.data.row_column)
        exempt = (
            self.con.execute(f"SELECT count(*) FROM {qi(side.data.table)} WHERE {na}").fetchone()[0]
            if na != "FALSE"
            else 0
        )
        sql = (
            f"SELECT * FROM {qi(side.typed)} WHERE {row} NOT IN "
            f"(SELECT {row} FROM {qi(side.data.table)} WHERE {na})"
        )
        return sql, exempt

    def _exact_float_keys(self, child_sql, column, left, parent_sql, target, right):
        """Prove both key columns hold only integers exactly representable as DOUBLE."""
        for sql, field, dtype in ((child_sql, column, left), (parent_sql, target, right)):
            value = qi(field)
            invalid = f"{value} < -9007199254740992 OR {value} > 9007199254740992"
            if dtype in FLOATS:
                invalid += f" OR NOT isfinite({value}) OR {value} != trunc({value})"
            if self.con.execute(f"SELECT count(*) FROM ({sql}) keys WHERE {invalid}").fetchone()[0]:
                return False
        return True

    def foreign_key(self, fk, parent):
        columns, targets = fk.columns, fk.target_columns
        if any(c not in self.data.columns for c in columns) or any(
            c not in parent.data.columns for c in targets
        ):
            raise ValueError("Foreign-key column is missing")
        child_sql, child_na = self._relation(self, columns)
        parent_sql, parent_na = self._relation(parent, targets)
        comparisons = []
        for column, target in zip(columns, targets, strict=True):
            left = self.values[column].dtype if column in self.values else self.data.types[column]
            right = (
                parent.values[target].dtype
                if target in parent.values
                else parent.data.types[target]
            )
            exact_float = functools.partial(
                self._exact_float_keys, child_sql, column, left, parent_sql, target, right
            )
            a, b = key_casts(column, left, right, exact_float)
            lhs, rhs = f"child.{qi(column)}", f"parent.{qi(target)}"
            if a:
                lhs = f"CAST({lhs} AS {a})"
            if b:
                rhs = f"CAST({rhs} AS {b})"
            comparisons.append(f"{lhs} = {rhs}")
        missing = " OR ".join(f"child.{qi(c)} IS NULL" for c in columns)
        all_missing = " AND ".join(f"child.{qi(c)} IS NULL" for c in columns)
        unknown = (
            f"NOT EXISTS (SELECT 1 FROM ({parent_sql}) parent WHERE {' AND '.join(comparisons)})"
        )
        bad = f"NOT ({missing}) AND {unknown}"
        if fk.match == "full":
            bad = f"({bad}) OR (({missing}) AND NOT ({all_missing}))"
        query = f"SELECT child.* FROM ({child_sql}) child WHERE {bad}"
        count = self.con.execute(f"SELECT count(*) FROM ({query}) violations").fetchone()[0]
        records = (
            self.out.records(
                f"SELECT * FROM ({query}) violations ORDER BY {qi(self.data.row_column)} LIMIT "
                f"{self.out.settings.engine.max_affected_records}"
            )
            if count
            else []
        )
        metrics = {}
        label = f"{self.data.name} ({', '.join(columns)})"
        if child_na:
            metrics["exempt_na_rows"] = {label: child_na}
        if parent_na:
            metrics["parent_exempt_na_rows"] = {label: parent_na}
        self.out.add(
            "INP-005",
            "foreign_key",
            self.data.name,
            columns,
            count=count,
            total=self.data.rows,
            records=records,
            parameters=fk.model_dump(),
            severity=fk.severity,
            metrics=metrics,
        )
        sides = []
        if fk.cardinality in ("many_to_one", "one_to_one"):
            sides.append((parent_sql, targets, "parent"))
        if fk.cardinality == "one_to_one":
            sides.append((child_sql, columns, "child"))
        for sql, fields, role in sides:
            selected = ", ".join(qi(c) for c in fields)
            present = " AND ".join(f"{qi(c)} IS NOT NULL" for c in fields)
            groups = (
                f"SELECT {selected}, count(*) AS occurrences FROM ({sql}) keys "
                f"WHERE {present} GROUP BY {selected} HAVING count(*) > 1"
            )
            duplicates, involved = self.con.execute(
                f"SELECT count(*), coalesce(sum(occurrences),0) FROM ({groups}) groups"
            ).fetchone()
            records = (
                self.out.records(
                    f"SELECT * FROM ({groups}) groups ORDER BY {selected} LIMIT "
                    f"{self.out.settings.engine.max_affected_records}"
                )
                if duplicates
                else []
            )
            self.out.add(
                "INP-005",
                "foreign_key." + role + "_cardinality",
                self.data.name,
                columns,
                count=duplicates,
                unit="keys",
                affected_rows=involved,
                records=records,
                parameters=fk.model_dump(),
                severity=fk.severity,
                metrics=metrics,
            )
