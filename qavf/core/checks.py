"""Composable SQL implementations of declared input contracts.

:class:`DatasetChecks` owns one loaded dataset's SQL views; each rule's checks
live in their own module and report through the shared result collector, in the
full backend's shape and wording.
"""

from __future__ import annotations

from dataclasses import dataclass

from qavf import input_rules as wording
from qavf.applicability import AllOrNoneApplicability
from qavf.core.columns import ColumnChecks
from qavf.core.contracts import canonical_key
from qavf.core.keys import KeyChecks
from qavf.core.policy import ColumnValue, DecimalColumnScale, NativeValueDomain, native_literal
from qavf.core.references import ReferenceCheck
from qavf.core.schema import SchemaConformance, evidence_name
from qavf.engine.severity import Status
from qavf.sources.sql import quote_identifier as qi
from qavf.sources.sql import sql_literal as lit


class DatasetChecks:
    """The declared contracts of one loaded dataset, over its typed SQL views."""

    def __init__(self, collector, loaded, spec):
        self.out = collector
        self.con = collector.connection
        self.data = loaded
        self.spec = spec
        self.schema = spec.schema_spec
        self.values = {}
        self._population = None
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

    # ------------------------------------------------------------------ views
    @property
    def exact_kernel(self):
        """Whether the full backend assesses this input's columns in memory.

        It does for frames supplied in memory; files and queries keep its SQL
        reductions. The two word an unassessable column check differently.
        """
        return self.data.reader is None and self.data.metadata.get("source") != "query"

    @property
    def available(self):
        """The dataset's columns as read, before any projection."""
        return list(getattr(self.data, "original_columns", ()) or self.data.columns)

    @property
    def population(self):
        """Declared N/A values per column, counted once."""
        if self._population is None:
            names = [c for c in self.values if self.schema.columns[c].na_values]
            counts = ()
            if names:
                aggregates = ", ".join(
                    f"count(*) FILTER (WHERE {self.values[c].na})" for c in names
                )
                counts = self.con.execute(
                    f"SELECT {aggregates} FROM {qi(self.data.table)}"
                ).fetchone()
            self._population = dict.fromkeys(self.values, 0)
            self._population.update(zip(names, counts, strict=True))
        return self._population

    def canonical(self, column):
        if column in self.values:
            return self.values[column].value
        return canonical_key(qi(column), self.data.types[column])

    def dtype(self, column):
        return self.values[column].dtype if column in self.values else self.data.types[column]

    def converted(self, columns):
        """Select ``columns`` converted to their declared types, missing markers kept."""
        return ", ".join(
            f"{self.values[c].converted if c in self.values else qi(c)} AS {qi(c)}" for c in columns
        )

    def as_read(self, columns):
        """Select ``columns`` as the dataset reads them.

        A CSV file's declared columns are read as their types; other sources keep
        their own values, also where a declared type converts them for checks.
        """
        if self.data.reader == "csv":
            return self.converted(columns)
        return ", ".join(qi(c) for c in columns)

    def as_read_relation(self):
        """The dataset as read, with its row positions."""
        columns = self.as_read(self.data.columns)
        return (
            f"SELECT {qi(self.data.row_column)}{', ' if columns else ''}{columns} "
            f"FROM {qi(self.data.table)}"
        )

    def shown(self, dependencies):
        """The data columns of a check's evidence, in dataset order."""
        if self.schema.evidence_columns == "all":
            return list(self.data.columns)
        wanted = {*dependencies, *self.schema.primary_key}
        return [column for column in self.data.columns if column in wanted]

    def selection(self, where):
        """Rows a ``where`` condition selects. Like row expressions, it reads
        declared values with their missing markers removed and N/A tokens kept."""
        if not where:
            return "TRUE"
        row = qi(self.data.row_column)
        return f"{row} IN (SELECT {row} FROM {qi(self.expression_table)} WHERE ({where}) IS TRUE)"

    def applicable_relation(self, fields, where=None):
        """One side's canonical key rows without N/A values, and the N/A rows exempted.

        ``where`` (a view's condition, over the rows SQL reads as the dataset)
        limits both to the rows it selects.
        """
        predicates = [
            self.values[c].na
            for c in fields
            if c in self.values and self.schema.columns[c].na_values
        ]
        row = qi(self.data.row_column)
        scope = (
            ""
            if where is None
            else f"{row} IN (SELECT {row} FROM {qi(self.expression_table)} WHERE ({where}))"
        )
        if not predicates:
            selected = f" WHERE {scope}" if scope else ""
            return f"SELECT * FROM {qi(self.typed)}{selected}", 0
        na = " OR ".join(predicates)
        within = f" AND {scope}" if scope else ""
        exempt = self.con.execute(
            f"SELECT count(*) FROM {qi(self.data.table)} WHERE ({na}){within}"
        ).fetchone()[0]
        sql = (
            f"SELECT * FROM {qi(self.typed)} WHERE {row} NOT IN "
            f"(SELECT {row} FROM {qi(self.data.table)} WHERE {na}){within}"
        )
        return sql, exempt

    def view(self, spec):
        """The rows of view ``spec`` of this dataset, as a foreign key's parent side."""
        return ViewRows(self, spec)

    # ------------------------------------------------------------------ rules
    def shape(self):
        if self.schema is not None and self.out.enabled("INP-002", self.data.name):
            SchemaConformance(self).run()

    def columns(self):
        if self.schema is not None and self.out.enabled("INP-003", self.data.name):
            ColumnChecks(self).run()

    def keys(self):
        if self.schema is not None and self.out.enabled("INP-004", self.data.name):
            KeyChecks(self).run()

    def foreign_keys(self, datasets, unavailable=None, views=None):
        """Check each foreign key against its loaded parent dataset or a view of one
        (``views``: ``<dataset>.<view>`` -> (dataset, definition))."""
        if not self.out.enabled("INP-005", self.data.name):
            return
        for fk in self.spec.foreign_keys:
            dataset, view = (views or {}).get(fk.ref_dataset, (fk.ref_dataset, None))
            parent = datasets.get(dataset)
            reason = (unavailable or {}).get(dataset)
            if parent is None and reason is None:
                reason = f"dataset {dataset!r} is not available"
            if parent is not None and view is not None:
                parent = parent.view(view)
            ReferenceCheck(self, fk, parent, reason).run()

    def expressions(self):
        name = self.data.name
        if not self.out.enabled("INP-006", name):
            return
        for check in self.spec.expressions_of(name):
            self._expression(check)

    def _expression(self, check):
        name = self.data.name
        label = check.label
        item_ids = (check.item_id,)
        identity = {
            "check_id": "row.expression",
            "datasets": (name,),
            "parameters": check.model_dump(mode="python"),
        }
        unassessed = {
            "check": f"{name}: {label}",
            "severity": check.severity,
            "item_ids": item_ids,
            **identity,
        }
        try:
            applicable = self._applicable(check.scope_condition())
            scope = (
                {} if check.where is None else wording.expression_scope(self.data.rows, applicable)
            )
            if applicable < check.min_applicable_rows:
                self.out.report(
                    "INP-006",
                    name,
                    Status.ERROR,
                    wording.expression_too_few_applicable(
                        label, name, applicable, check.where, check.min_applicable_rows
                    ),
                    metrics=scope,
                    **unassessed,
                )
                return
            count, records = self._violations(check.violation_condition())
        except Exception as exc:
            self.out.report(
                "INP-006",
                name,
                Status.ERROR,
                wording.expression_not_evaluated(label, name, str(exc)),
                **unassessed,
            )
            return
        if not count:
            self.out.passed("INP-006", name, f"{name}: {label}", identity, item_id=check.item_id)
            return
        self.out.report(
            "INP-006",
            name,
            Status.FAIL,
            wording.expression_violations(count, applicable, name, check.expr, check.where),
            check=f"{name}: {label}",
            severity=check.severity,
            records=records,
            count=count,
            total=applicable,
            unit="rows",
            metrics=scope,
            item_ids=item_ids,
            **identity,
        )

    def _applicable(self, scope):
        """The rows ``scope`` selects (every row without one)."""
        if scope is None:
            return self.data.rows
        table = qi(self.expression_table)
        return self.con.execute(f"SELECT count(*) FROM {table} WHERE {scope}").fetchone()[0]

    def _violations(self, bad):
        table, row = qi(self.expression_table), qi(self.data.row_column)
        count = self.con.execute(f"SELECT count(*) FROM {table} WHERE {bad}").fetchone()[0]
        limit = self.out.evidence_limit
        if not count or not limit:
            return count, None
        columns = ", ".join(qi(c) for c in self.data.columns)
        records = self.out.records(
            f"SELECT {columns}{', ' if columns else ''}{row} FROM {table} WHERE {bad} "
            f"ORDER BY {row} LIMIT {limit}"
        )
        return count, records

    def applicability(self):
        name = self.data.name
        if self.schema is None or not self.out.enabled("INP-009", name):
            return
        declared = [c for c, spec in self.schema.columns.items() if spec.na_values]
        if not declared:
            return
        try:
            for column in declared:
                if column in self.values:
                    self._population_result(column)
            for group in self.schema.na_groups:
                self._group(group)
        except Exception as exc:
            self.out.report(
                "INP-009",
                name,
                Status.ERROR,
                f"Not assessed: N/A applicability on {name}: {type(exc).__name__}: {exc}.",
                check=name,
                datasets=(name,),
            )

    def _population_result(self, column):
        name, rows = self.data.name, self.data.rows
        spec = self.schema.columns[column]
        na_count = self.population[column]
        self.out.report(
            "INP-009",
            name,
            Status.PASS,
            wording.not_applicable_population(name, column, na_count, rows),
            check=f"{name}.{column}: N/A population",
            check_id="applicability.population",
            datasets=(name,),
            columns=(column,),
            parameters={"na_values": list(spec.na_values), "trim_whitespace": spec.trim_whitespace},
            count=0,
            total=rows,
            unit="rows",
            metrics={"na_count": na_count, "applicable_count": rows - na_count},
        )

    def _group(self, group):
        data = self.data
        identity = {
            "check": wording.na_group_label(data.name, group),
            "check_id": "applicability.all_or_none",
            "datasets": (data.name,),
            "columns": tuple(group),
            "parameters": {
                "na_values": {column: self.schema.columns[column].na_values for column in group}
            },
        }
        absent = [column for column in group if column not in data.columns]
        if absent:
            self.out.report(
                "INP-009", data.name, Status.ERROR, wording.na_group_absent(absent), **identity
            )
            return
        partial = AllOrNoneApplicability.from_schema(group, self.schema).sql()
        count = self.con.execute(
            f"SELECT count(*) FROM {qi(data.table)} WHERE {partial}"
        ).fetchone()[0]
        if not count:
            self.out.report(
                "INP-009",
                data.name,
                Status.PASS,
                wording.na_group_satisfied(data.rows, data.name),
                count=0,
                total=data.rows,
                unit="rows",
                **identity,
            )
            return
        records = None
        limit = self.out.evidence_limit
        if limit:
            shown = list(dict.fromkeys([*self.schema.primary_key, *group]))
            shown = [column for column in shown if column in data.columns]
            reason = evidence_name("violates", [*shown, data.row_column])
            records = self.out.records(
                f"SELECT {self.as_read(shown)}{', ' if shown else ''}{qi(data.row_column)}, "
                f"{lit('partly not applicable')} AS {qi(reason)} FROM {qi(data.table)} "
                f"WHERE {partial} ORDER BY {qi(data.row_column)} LIMIT {limit}"
            )
        self.out.report(
            "INP-009",
            data.name,
            Status.FAIL,
            wording.na_group_violations(count, data.rows, data.name),
            records=records,
            count=count,
            total=data.rows,
            unit="rows",
            **identity,
        )

    # ------------------------------------------------------- grouped values
    def _grouped_literals(self, column, values):
        """Render a grouped-value domain as exact typed literals, or reject it."""
        from qavf.settings import ColumnCheck

        dtype = self.dtype(column)
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
        row = qi(self.data.row_column)
        predicates = [
            self.values[c].na
            for c in columns
            if c in self.values and self.schema.columns[c].na_values
        ]
        if not predicates:
            return "TRUE"
        na = " OR ".join(predicates)
        return f"{row} NOT IN (SELECT {row} FROM {qi(self.data.table)} WHERE {na})"

    def grouped_values(self, spec):
        """Count violating group values of one contract; return count, total, evidence."""
        from qavf.grouped_values import GroupedValuePlan

        columns = [*spec.group_by, spec.column]
        absent = [c for c in columns if c not in self.data.columns]
        if absent:
            raise ValueError(f"Grouped-value column(s) missing: {', '.join(absent)}")
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
        limit = self.out.evidence_limit
        records = (
            self.out.records(
                f"SELECT * FROM ({plan.violations_sql}) violations ORDER BY ALL LIMIT {limit}"
            )
            if count and limit
            else None
        )
        return count, total, records


@dataclass(frozen=True, slots=True)
class ViewColumns:
    """The columns a view exposes, where a foreign key's parent side looks them up."""

    columns: tuple[str, ...]


class ViewRows:
    """A view of a loaded dataset as the parent side of a foreign key.

    The keys are the dataset's canonical key values, limited to the rows the view's
    ``where`` selects (evaluated over the rows SQL reads as the dataset, as the
    registered view is), and its columns are the view's ``columns`` (or all). N/A
    keys are exempted within those rows only, as the Polars backend's view does.
    """

    def __init__(self, dataset, spec):
        self.dataset = dataset
        self.spec = spec
        self.data = ViewColumns(tuple(spec.columns or dataset.data.columns))

    def applicable_relation(self, fields):
        """The view's canonical key rows without N/A values, and the N/A rows exempted."""
        return self.dataset.applicable_relation(fields, self.spec.where)

    def dtype(self, column):
        """The canonical type of column, as its dataset declares it."""
        return self.dataset.dtype(column)
