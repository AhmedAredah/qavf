"""Primary keys, unique keys, duplicate rows and grouped values (INP-004) as SQL."""

from __future__ import annotations

from uuid import uuid4

from qavf import input_rules as wording
from qavf.audit_codec import evidence_json_size
from qavf.core.schema import evidence_name
from qavf.duplicate_evidence import DuplicateEvidencePolicy
from qavf.engine.severity import Status
from qavf.sources.sql import quote_identifier as qi


def _unclaimed_name(role, columns):
    """Prefix a generated column name until it collides with no data column."""
    while role.casefold() in {column.casefold() for column in columns}:
        role = "qavf_" + role
    return role


class KeyContract:
    """One primary or unique key: its columns, null policy and optional scope."""

    def __init__(self, columns, primary, nulls, where=None, name=None, rows_per_key=10):
        self.columns = list(columns)
        self.primary = primary
        self.nulls = nulls
        self.where = where
        self.name = name
        self.parameters = {"nulls": nulls, "rows_per_key": rows_per_key}
        if where is not None:
            self.parameters["where"] = where
        if name is not None:
            self.parameters["name"] = name

    @property
    def identity(self):
        return {
            "check_id": "key.primary" if self.primary else "key.unique",
            "columns": tuple(self.columns),
            "parameters": self.parameters,
        }

    def check(self, dataset):
        return wording.key_check(dataset, self.columns, self.primary)

    def label(self, dataset):
        return wording.key_label(dataset, self.columns)


class KeyChecks:
    """INP-004 for one loaded dataset, in the full backend's order and wording."""

    RULE = "INP-004"

    def __init__(self, checks):
        self.checks = checks
        self.out = checks.out
        self.con = checks.con
        self.data = checks.data
        self.schema = checks.schema
        self.failed = False
        self.identities = []

    def contracts(self):
        schema = self.schema
        cap = schema.duplicate_rows_per_key
        contracts = (
            [KeyContract(schema.primary_key, True, "forbid", rows_per_key=cap)]
            if schema.primary_key
            else []
        )
        contracts.extend(
            KeyContract(key.columns, False, key.nulls, key.where, key.name, cap)
            for key in schema.alternate_keys
        )
        return contracts

    def run(self):
        schema, name = self.schema, self.data.name
        if not (
            schema.primary_key or schema.unique_keys or schema.unique_rows or schema.grouped_values
        ):
            return
        try:
            for contract in self.contracts():
                self._contract(contract)
            if schema.unique_rows:
                self._unique_rows()
            for grouped in schema.grouped_values:
                self._grouped_values(grouped)
        except Exception as exc:
            self.failed = True
            self.out.report(
                self.RULE,
                name,
                Status.ERROR,
                f"Not assessed: keys on {name}: {type(exc).__name__}: {exc}.",
                check=name,
                datasets=(name,),
            )
        if not self.identities:
            return
        self.out.assessed(self.RULE, name, *self.identities)
        if not self.failed:
            self.out.passed(self.RULE, name, name)
            return
        # A partial success names the assessed contract, never the dataset as
        # though every configured key passed.
        for identity in self.identities:
            label = f"{name}: {identity['check_id']} ({', '.join(identity['columns'])})"
            self.out.passed(self.RULE, name, label)

    def _result(self, status, message, **fields):
        """Report one key contract that did not simply pass."""
        self.failed = True
        self.out.report(
            self.RULE, self.data.name, status, message, datasets=(self.data.name,), **fields
        )

    def _contract(self, contract):
        data = self.data
        absent = [column for column in contract.columns if column not in data.columns]
        if absent:
            self._result(
                Status.FAIL,
                wording.key_columns_absent(
                    "Primary key" if contract.primary else "Unique key",
                    data.name,
                    absent,
                    self.checks.available,
                ),
                check=contract.check(data.name),
                records=[{"dataset": data.name, "missing_key_column": c} for c in absent],
                unit="columns",
                **contract.identity,
            )
            return
        try:
            self._summarized(contract)
        except Exception as exc:
            label = contract.check(data.name)
            self._result(
                Status.ERROR,
                f"Not assessed: {label}: {type(exc).__name__}: {exc}.",
                check=label,
                **contract.identity,
            )

    def _summarized(self, contract):
        """Summarize canonical keys once and stream positions for only their sample."""
        out, data, columns = self.out, self.data, contract.columns
        frequency = _unclaimed_name("occurrences", columns)
        positions = _unclaimed_name("rows", columns)
        row = qi(data.row_column)
        keys = ", ".join(qi(c) for c in columns)
        order = ", ".join(f"{qi(c)} NULLS FIRST" for c in columns)
        missing = " OR ".join(f"{qi(c)} IS NULL" for c in columns)
        eligible = "TRUE" if contract.nulls == "equal" else f"NOT ({missing})"
        na = " OR ".join(
            self.checks.values[c].na
            for c in columns
            if c in self.checks.values and self.schema.columns[c].na_values
        )
        selected = self.checks.selection(contract.where)
        label = contract.label(data.name)
        exempt_na = (
            self.con.execute(
                f"SELECT count(*) FILTER (WHERE {na}) FROM {qi(data.table)} WHERE {selected}"
            ).fetchone()[0]
            if na
            else 0
        )
        self.out.exempt(self.RULE, data.name, "exempt_na_rows", label, exempt_na)
        projection = ", ".join(f"{self.checks.canonical(c)} AS {qi(c)}" for c in columns)
        source = (
            f"SELECT {row}, {projection} FROM {qi(data.table)} "
            f"WHERE NOT ({na or 'FALSE'}) AND ({selected})"
        )
        groups = (
            f"SELECT {keys}, count(*)::UBIGINT AS {qi(frequency)} FROM "
            f"({source}) source GROUP BY {keys}"
        )
        duplicates = f"({eligible}) AND {qi(frequency)} > 1"
        limit = out.evidence_limit
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
                f"{duplicates} ORDER BY {order} LIMIT {limit}) s) AS sample, "
                f"(SELECT list(s) FROM (SELECT * FROM grouped WHERE "
                f"{missing} ORDER BY {order} LIMIT "
                f"{limit if contract.nulls == 'forbid' else 0}) s) AS null_sample "
                f"FROM grouped"
            )
            total, duplicate_count, involved, null_count, scoped_rows = self.con.execute(
                f"SELECT total_keys, duplicated_keys, involved, null_rows, scoped_rows "
                f"FROM {qi(summary_table)}"
            ).fetchone()
            rows = scoped_rows + exempt_na
            evidence = (
                ", ".join(f"key.{qi(c)} AS {qi(c)}" for c in columns)
                + f", key.{qi(frequency)} AS {qi(frequency)}"
            )
            failed = False
            if contract.nulls == "forbid" and null_count:
                failed = True
                self._result(
                    Status.FAIL,
                    wording.null_keys(null_count, rows, label),
                    check=f"{data.name}: null keys" if contract.primary else f"{label}: null keys",
                    records=self._sample(evidence, summary_table, "null_sample") if limit else None,
                    count=null_count,
                    total=rows,
                    unit="rows",
                    **{**contract.identity, "check_id": contract.identity["check_id"] + ".nulls"},
                )
            elif contract.nulls == "exempt":
                self.out.exempt(
                    self.RULE, data.name, "rows_exempt_null_unique_keys", label, null_count
                )
            if not duplicate_count:
                if not failed:
                    self.identities.append({**contract.identity, "datasets": (data.name,)})
                return
            records = self._sample(evidence, summary_table, "sample")
            metrics = {
                "duplicated_keys": duplicate_count,
                "rows_involved": involved,
                "surplus_rows": involved - duplicate_count,
            }
            metrics.update(
                self._positions(
                    records, metrics, contract, summary_table, source, frequency, positions
                )
            )
            self._result(
                Status.FAIL,
                wording.duplicated_keys(duplicate_count, total, label, involved, rows),
                check=contract.check(data.name),
                records=records,
                count=duplicate_count,
                total=total,
                unit="keys",
                affected_rows=involved,
                metrics=metrics,
                **contract.identity,
            )
        finally:
            self.con.execute(f"DROP TABLE IF EXISTS {qi(summary_table)}")

    def _sample(self, evidence, summary_table, field):
        """Fetch the sampled key records stored in one summary-table list field."""
        return self.out.records(
            f"SELECT {evidence} FROM "
            f"(SELECT unnest({qi(field)}) AS key FROM {qi(summary_table)}) sample_keys"
        )

    def _positions(self, records, metrics, contract, summary_table, source, frequency, positions):
        """Attach the row positions of each sampled duplicated key, within a byte budget."""
        cap = self.schema.duplicate_rows_per_key
        row = qi(self.data.row_column)
        position_metadata = {
            **metrics,
            "duplicate_rows_per_key": cap,
            "duplicate_positions_available": False,
            "duplicate_positions_complete": False,
            "duplicate_positions_truncated": False,
        }
        budget = max(
            0,
            self.out.settings.engine.max_evidence_bytes
            - evidence_json_size(position_metadata)
            - 64,
        )
        accumulated = DuplicateEvidencePolicy(cap, budget).accumulator(
            records, frequency=frequency, positions=positions
        )
        if records:
            wanted = (
                f"SELECT generate_subscripts(sample, 1) - 1 AS sample_index, "
                f"unnest(sample) AS key FROM {qi(summary_table)}"
            )
            wanted = f"SELECT * FROM ({wanted}) sampled WHERE sample_index < {len(records)}"
            match = " AND ".join(
                f"source.{qi(c)} IS NOT DISTINCT FROM wanted.key.{qi(c)}" for c in contract.columns
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
                    keep = all(accumulated.append(index, position) for index, position in batch)
            except Exception:
                accumulated.available = False
        return accumulated.metrics

    def _unique_rows(self):
        data = self.data
        columns = list(data.columns)
        if not columns:
            return
        selected = ", ".join(qi(column) for column in columns)
        reason = evidence_name("violates", [*columns, data.row_column])
        duplicated = _unclaimed_name("__qavf_duplicated", columns)
        rows = (
            f"SELECT {selected}, {qi(data.row_column)}, count(*) OVER "
            f"(PARTITION BY {selected}) > 1 AS {qi(duplicated)} FROM "
            f"({self.checks.as_read_relation()}) source"
        )
        count = self.con.execute(
            f"SELECT count(*) FROM ({rows}) source WHERE {qi(duplicated)}"
        ).fetchone()[0]
        identity = {
            "check_id": "key.unique_rows",
            "columns": tuple(columns),
            "parameters": {},
        }
        if not count:
            self.identities.append({**identity, "datasets": (data.name,)})
            return
        limit = self.out.evidence_limit
        records = (
            self.out.records(
                f"SELECT {selected}, {qi(data.row_column)}, 'duplicate row' AS {qi(reason)} "
                f"FROM ({rows}) source WHERE {qi(duplicated)} ORDER BY "
                f"{qi(data.row_column)} LIMIT {limit}"
            )
            if limit
            else None
        )
        self._result(
            Status.FAIL,
            wording.duplicate_rows(count, data.rows, data.name),
            check=f"{data.name}: duplicate rows",
            records=records,
            count=count,
            total=data.rows,
            unit="rows",
            **identity,
        )

    def _grouped_values(self, spec):
        data = self.data
        columns = [*spec.group_by, spec.column]
        label = wording.grouped_values_label(data.name, spec.name, columns)
        identity = {
            "check_id": "key.grouped_values",
            "columns": tuple(columns),
            "parameters": spec.model_dump(mode="python"),
        }
        self.failed = True
        try:
            count, total, records = self.checks.grouped_values(spec)
        except Exception as exc:
            self._result(
                Status.ERROR,
                f"Not assessed: {label}: {type(exc).__name__}: {exc}.",
                check=label,
                **identity,
            )
            return
        if count:
            message = wording.group_values_violate(count, total)
        else:
            message = wording.group_values_satisfy(total)
        self._result(
            Status.FAIL if count else Status.PASS,
            message,
            check=label,
            records=records if count else None,
            count=count,
            total=total,
            unit="group values",
            **identity,
        )
