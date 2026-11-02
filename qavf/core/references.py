"""Foreign keys (INP-005) as SQL: orphaned keys, match policy and cardinality."""

from __future__ import annotations

import functools

from qavf import input_rules as wording
from qavf.core.key_policy import FLOATS, key_casts
from qavf.core.schema import evidence_name
from qavf.engine.severity import Status
from qavf.sources.sql import quote_identifier as qi


class ReferenceCheck:
    """One foreign key of a loaded child dataset against its parent.

    Keys are grouped once per side. Only distinct keys are compared, so totals
    count keys, and rows are the sum of each key's occurrences.
    """

    RULE = "INP-005"

    def __init__(self, child, fk, parent, reason=None):
        self.child = child
        self.out = child.out
        self.con = child.con
        self.fk = fk
        self.parent = parent
        self.reason = reason
        self.dataset = child.data.name
        self.label = wording.reference_label(self.dataset, fk.columns, fk.ref_dataset)
        self.identity = {
            "check_id": "key.reference",
            "columns": tuple(fk.columns),
            "datasets": (self.dataset, fk.ref_dataset),
            "parameters": fk.model_dump(mode="python"),
        }
        self.clean = True

    def run(self):
        try:
            self._assess()
        except Exception as exc:
            self._report(
                Status.ERROR,
                f"Not assessed: {self.label}: {type(exc).__name__}: {exc}.",
                check=self.label,
            )
            return
        if self.clean:
            self.out.passed(self.RULE, self.dataset, self.label, self.identity)

    def _report(self, status, message, **fields):
        self.clean = False
        self.out.report(
            self.RULE,
            self.dataset,
            status,
            message,
            severity=self.fk.severity,
            **self.identity,
            **fields,
        )

    def _assess(self):
        fk, child, parent = self.fk, self.child, self.parent
        if parent is None:
            self._report(
                Status.ERROR,
                wording.blocked_reference(self.label, self.reason),
                check=self.label,
            )
            return
        if any(c not in child.data.columns for c in fk.columns) or any(
            c not in parent.data.columns for c in fk.target_columns
        ):
            self._report(
                Status.ERROR, wording.reference_columns_absent(self.label), check=self.label
            )
            return
        child_sql, child_na = child.applicable_relation(fk.columns)
        parent_sql, parent_na = parent.applicable_relation(fk.target_columns)
        self.out.exempt(self.RULE, self.dataset, "exempt_na_rows", self.label, child_na)
        self.out.exempt(self.RULE, self.dataset, "parent_exempt_na_rows", self.label, parent_na)
        columns, targets = fk.columns, fk.target_columns
        frequency = evidence_name("rows", columns)
        keys = ", ".join(qi(c) for c in columns)
        groups = (
            f"SELECT {keys}, count(*) AS {qi(frequency)} FROM ({child_sql}) keys GROUP BY {keys}"
        )
        missing = " OR ".join(f"{qi(c)} IS NULL" for c in columns)
        every = " AND ".join(f"{qi(c)} IS NULL" for c in columns)
        exempt = every if fk.match == "full" else missing
        total, nulls, valid = self.con.execute(
            f"SELECT coalesce(sum({qi(frequency)}), 0), "
            f"coalesce(sum({qi(frequency)}) FILTER (WHERE {exempt}), 0), "
            f"count(*) FILTER (WHERE NOT ({missing})) FROM ({groups}) groups"
        ).fetchone()
        self.out.exempt(self.RULE, self.dataset, "rows_exempt_null_keys", self.label, nulls)
        if fk.match == "full":
            self._partial(groups, f"({missing}) AND NOT ({every})", frequency, total)
        if fk.cardinality in ("many_to_one", "one_to_one"):
            self._repeated("parent", parent_sql, targets, fk.ref_dataset)
        if fk.cardinality == "one_to_one":
            self._repeated("child", child_sql, columns, self.dataset)
        self._orphans(groups, missing, frequency, parent_sql, valid)

    def _sample(self, relation, columns, order_by):
        limit = self.out.evidence_limit
        if not limit:
            return None
        ordering = ", ".join(f"{qi(c)} NULLS FIRST" for c in order_by)
        selected = ", ".join(qi(c) for c in columns)
        return self.out.records(
            f"SELECT {selected} FROM ({relation}) sample ORDER BY {ordering} LIMIT {limit}"
        )

    def _partial(self, groups, partial, frequency, total):
        relation = f"SELECT * FROM ({groups}) groups WHERE {partial}"
        rows = self.con.execute(
            f"SELECT coalesce(sum({qi(frequency)}), 0) FROM ({relation}) partial"
        ).fetchone()[0]
        if not rows:
            return
        occurrences = evidence_name("occurrences", self.fk.columns)
        shown = f"SELECT *, {qi(frequency)} AS {qi(occurrences)} FROM ({relation}) partial"
        self._report(
            Status.FAIL,
            wording.partly_null_keys(rows, self.label),
            check=f"{self.label}: partly null keys",
            records=self._sample(shown, [*self.fk.columns, occurrences], self.fk.columns),
            count=rows,
            total=total,
            unit="rows",
        )

    def _repeated(self, side, relation, columns, name):
        selected = ", ".join(qi(c) for c in columns)
        present = " AND ".join(f"{qi(c)} IS NOT NULL" for c in columns)
        occurrences = evidence_name("occurrences", columns)
        groups = (
            f"SELECT {selected}, count(*) AS {qi(occurrences)} FROM ({relation}) keys "
            f"WHERE {present} GROUP BY {selected} HAVING count(*) > 1"
        )
        keys, rows = self.con.execute(
            f"SELECT count(*), coalesce(sum({qi(occurrences)}), 0) FROM ({groups}) groups"
        ).fetchone()
        if not keys:
            return
        self._report(
            Status.FAIL,
            wording.repeated_reference_keys(keys, name, columns, self.label, self.fk.cardinality),
            check=f"{self.label}: {side} keys not unique",
            records=self._sample(groups, [*columns, occurrences], columns),
            count=keys,
            affected_rows=rows,
            unit="keys",
        )

    def _orphans(self, groups, missing, frequency, parent_sql, valid):
        fk, child, parent = self.fk, self.child, self.parent
        comparisons = []
        for column, target in zip(fk.columns, fk.target_columns, strict=True):
            left = child.dtype(column)
            right = parent.dtype(target)
            exact_float = functools.partial(
                exact_float_keys, self.con, groups, column, left, parent_sql, target, right
            )
            cast_left, cast_right = key_casts(column, left, right, exact_float)
            lhs, rhs = f"child.{qi(column)}", f"parent.{qi(target)}"
            if cast_left:
                lhs = f"CAST({lhs} AS {cast_left})"
            if cast_right:
                rhs = f"CAST({rhs} AS {cast_right})"
            comparisons.append(f"{lhs} = {rhs}")
        orphans = (
            f"SELECT * FROM ({groups}) child WHERE NOT ({missing}) AND NOT EXISTS "
            f"(SELECT 1 FROM ({parent_sql}) parent WHERE {' AND '.join(comparisons)})"
        )
        keys, rows = self.con.execute(
            f"SELECT count(*), coalesce(sum({qi(frequency)}), 0) FROM ({orphans}) orphans"
        ).fetchone()
        if not keys:
            return
        self._report(
            Status.FAIL,
            wording.orphaned_keys(keys, valid, self.label, rows),
            check=self.label,
            records=self._sample(orphans, [*fk.columns, frequency], fk.columns),
            count=keys,
            total=valid,
            unit="keys",
            affected_rows=rows,
        )


def exact_float_keys(con, child_sql, column, left, parent_sql, target, right):
    """Prove both key columns hold only integers exactly representable as DOUBLE."""
    for sql, field, dtype in ((child_sql, column, left), (parent_sql, target, right)):
        value = qi(field)
        invalid = f"{value} < -9007199254740992 OR {value} > 9007199254740992"
        if dtype in FLOATS:
            invalid += f" OR NOT isfinite({value}) OR {value} != trunc({value})"
        if con.execute(f"SELECT count(*) FROM ({sql}) keys WHERE {invalid}").fetchone()[0]:
            return False
    return True
