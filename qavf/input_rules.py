"""Identity, labels and wording of the input rules (INP-001 to INP-009).

Both backends report the declared input contracts: the full backend with Polars
rules and the native backend with DuckDB SQL. A report must not change when the
backend does, so the rule metadata, check labels and messages live here, once,
without importing Polars.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from qavf.engine.severity import Severity
from qavf.settings import Bounds

CATEGORY = "input"


@dataclass(frozen=True, slots=True)
class InputRule:
    """What a report says about an input rule, whichever backend assesses it."""

    rule_id: str
    name: str
    description: str
    recommendation: str
    unconfigured: str
    """The message of the SKIPPED result when nothing is configured for the rule."""
    default_severity: Severity = Severity.ERROR
    category: str = CATEGORY
    check_name: str = ""
    """The rule's check name in configurations and reports."""


INPUT_RULES: dict[str, InputRule] = {
    rule.rule_id: rule
    for rule in (
        InputRule(
            "INP-001",
            "Datasets available",
            "Required datasets are accessible and their schemas can be inspected; optional "
            "datasets may be absent. Native sources are not fully scanned by this availability "
            "check, so it does not certify that every source record can be parsed.",
            "Check the dataset's source settings (paths, queries, connection details) and "
            "confirm the upstream step that produces it completed. Mark a dataset "
            "'required: false' if it is genuinely optional.",
            "No datasets are configured.",
            Severity.CRITICAL,
            check_name="datasets_available",
        ),
        InputRule(
            "INP-002",
            "Schema conformance",
            "Each dataset has its required columns with the declared types, no undeclared "
            "columns when strict, and a row count within the configured limits.",
            "Fix the step that produced the dataset so column names and types match the "
            "schema, or update the schema if the layout changed intentionally. A column read "
            "as text when another type is declared usually holds a few stray values: the "
            "listed rows show them. Declare placeholders such as 'N/A' in the column's "
            "missing_values. A code declared str but read as a number has lost its leading "
            "zeros: read it as text at the source ('text_columns' for CSV files).",
            "No available dataset has a schema configured.",
            Severity.CRITICAL,
            check_name="schema",
        ),
        InputRule(
            "INP-003",
            "Column QA limits",
            "Column values respect their configured QA limits: nulls, bounds, allowed and "
            "forbidden values, patterns, lengths, uniqueness and aggregate statistics.",
            "Inspect the listed rows. Out-of-range values usually come from unit mix-ups, sign "
            "errors, sentinel values such as -999, or failed joins that leave nulls. If a "
            "value is legitimate, widen the limit or downgrade that check's severity.",
            "No column QA limits are configured.",
            check_name="column_limits",
        ),
        InputRule(
            "INP-004",
            "Unique primary keys",
            "Each dataset's primary-key columns identify its rows uniquely.",
            "Duplicate keys double-count values when tables are joined. De-duplicate at the "
            "source, or add the missing dimension to the primary key.",
            "No primary or unique keys are configured.",
            check_name="unique_keys",
        ),
        InputRule(
            "INP-005",
            "Referential integrity",
            "Every configured foreign-key value exists in the referenced dataset. Keys are "
            "compared exactly; key types that cannot be compared exactly are reported.",
            "Orphaned keys are silently dropped by inner joins. Add the missing reference rows "
            "or correct the keys in the child dataset.",
            "No foreign keys are configured.",
            check_name="references",
        ),
        InputRule(
            "INP-006",
            "Row expressions",
            "Every row satisfies the dataset's configured SQL expressions, e.g. "
            "'pm25 <= pm10' or 'end_date >= start_date'.",
            "An expression failure means a relationship between columns that must always hold "
            "does not. Inspect the listed rows for mismatched units or rows combined from "
            "different sources.",
            "No row expressions are configured.",
            check_name="expression",
        ),
        InputRule(
            "INP-007",
            "SQL assertions",
            "Each configured SQL assertion, a query that selects offending rows across any "
            "datasets, returns nothing.",
            "Each returned row is a counter-example to the assertion. Run the assertion's "
            "query yourself to explore them.",
            "No SQL assertions are configured.",
            check_name="sql",
        ),
        InputRule(
            "INP-008",
            "Data freshness",
            "The newest value of each configured date or datetime column is no older than its "
            "max_age, measured from the run (or a configured reference time).",
            "Stale data usually means an upstream job did not run or a file was not refreshed. "
            "Check the producing pipeline's last run before trusting downstream results.",
            "No freshness checks are configured.",
            check_name="fresh",
        ),
        InputRule(
            "INP-009",
            "Not-applicable values",
            "Declared N/A values are counted independently; configured groups are "
            "all or none N/A in each row.",
            "Make every column in the group applicable, or mark every column "
            "with its declared N/A token.",
            "No not-applicable tokens or groups are configured.",
            check_name="not_applicable",
        ),
    )
}


# ------------------------------------------------------------------ vocabulary
def plural(count: int, noun: str) -> str:
    """``"1 row"`` / ``"3 rows"`` (``series`` is its own plural)."""
    return f"{count:,} {noun}{'' if count == 1 or noun.endswith('series') else 's'}"


def agree(count: int, singular: str, plural_form: str) -> str:
    """The form of a verb phrase that agrees with ``count`` things."""
    return singular if count == 1 else plural_form


def format_value(value: Any, limit: int = 6) -> str:
    """Compact rendering of a limit value for labels."""
    if isinstance(value, list):
        shown = ", ".join(format_value(v) for v in value[:limit])
        return f"[{shown}{', …' if len(value) > limit else ''}]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def check_label(check: Any) -> str:
    """A column check's label: its name, or its limits (``ge 0, isin [a, b]``)."""
    if check.name:
        return str(check.name)
    parts = []
    for name, value in check.constraints().items():
        if isinstance(value, Bounds):
            parts.append(f"{name} {value.phrase()}")
        elif value is True:
            parts.append(name)
        else:
            parts.append(f"{name} {format_value(value)}")
    return ", ".join(parts)


def row_limit_label(name: str, value: Any) -> str:
    """The ``violates`` reason of one row-level limit (``ge 0``, ``integral``)."""
    return "integral" if name == "integral" else f"{name} {format_value(value)}"


def show_value(value: int | float | Decimal | Fraction | None) -> str:
    """An observed statistic: exact values in full, floats to six digits."""
    if value is None:
        return "undefined"
    return str(value) if isinstance(value, int | Decimal | Fraction) else f"{value:.6g}"


def show_bound(value: int | float | Decimal | Fraction | None) -> str:
    """One side of a configured range; an open side is left blank."""
    return "" if value is None else show_value(value)


def outside_bounds(name: str, value: Any, bounds: Any) -> str:
    """A column-level problem: a statistic outside its configured range."""
    return f"{name} {show_value(value)} outside {bounds.interval(show_bound)}"


def missing_fraction_problem(missing: int, rows: int, limit: float) -> str:
    """A column-level problem: too large a share of missing values."""
    fraction = missing / rows if rows else 0.0
    return f"missing fraction {fraction:.4g} (null or NaN) exceeds {limit:g}"


def described(dtype: str) -> str:
    """``an int``, ``a date``: the declared type as a noun phrase."""
    return f"{'an' if dtype == 'int' else 'a'} {dtype}"


def folded_name(name: str) -> str:
    """A column name without case, spaces, hyphens or dots: ``Site ID`` is ``siteid``."""
    return re.sub(r"[\s_.\-]+", "", name).casefold()


def absent_column_hint(column: str, available: Iterable[str]) -> str:
    """How the data probably spells an absent column, as a clause to append; or ``""``.

    The same name in another case or separator style comes first; otherwise the
    closest spellings, if any are close.
    """
    names = list(available)
    same = [name for name in names if folded_name(name) == folded_name(column)]
    if same:
        return f"; the data has {same[0]!r}: map it with 'rename'"
    close = difflib.get_close_matches(column, names, n=3, cutoff=0.75)
    if close:
        return f"; did you mean {' or '.join(repr(name) for name in close)}? Map it with 'rename'"
    return ""


def required_not_assessed(entry: str, reason: str) -> str:
    """An ``engine.require`` entry that no result assessed."""
    return f"Not assessed: required check {entry!r} was not assessed: {reason}."


# -------------------------------------------------------------------- INP-001
def dataset_unavailable(name: str, required: bool, reason: str | None) -> str:
    return f"{'Required' if required else 'Optional'} dataset {name!r} is not available: {reason}"


def datasets_accessible(count: int) -> str:
    return f"{plural(count, 'dataset')} accessible."


def blocked_dataset(what: str, dataset: str, reason: str | None) -> str:
    """A check that could not run because its dataset is unavailable."""
    return f"Not assessed: {what} on {dataset!r}: {reason}."


# -------------------------------------------------------------------- INP-002
def schema_violations(count: int, dataset: str) -> str:
    return f"{plural(count, 'schema violation')} in {dataset}."


def schemas_conform(count: int) -> str:
    conform = agree(count, "conforms to its schema", "conform to their schemas")
    return f"{plural(count, 'dataset')} {conform}."


def typed_read_values(count: int, rows: int, dataset: str, column: str, kind: str) -> str:
    """Values a typed read could not convert, which it reads as missing."""
    return (
        f"{plural(count, 'value')} of {rows:,} in {dataset}.{column} "
        f"{agree(count, 'is', 'are')} not {kind}, so {agree(count, 'it is', 'they are')} "
        "read as missing."
    )


def text_values(count: int, rows: int, dataset: str, column: str, kind: str, read_as: str) -> str:
    """Values that kept a declared column as text."""
    return (
        f"{plural(count, 'value')} of {rows:,} in {dataset}.{column} "
        f"{agree(count, 'is', 'are')} not {kind}, so the column was read as {read_as}."
    )


def non_integral_values(count: int, rows: int, dataset: str, column: str) -> str:
    whole = agree(count, "is not a whole number", "are not whole numbers")
    return (
        f"{plural(count, 'value')} of {rows:,} in {dataset}.{column} {whole} in a column "
        "declared int."
    )


def row_count_problem(rows: int, bound: int, *, at_least: bool) -> str:
    return f"{rows} rows, expected {'at least' if at_least else 'at most'} {bound}"


# -------------------------------------------------------------------- INP-003
def column_values_violate(
    count: int, applicable: int, identity: str, label: str, problems: Sequence[str] = ()
) -> str:
    """Rows of a column that break a check, with its column-level problems."""
    return (
        f"{count:,} value{'s' if count != 1 else ''} of {applicable:,} "
        f"in {identity} {agree(count, 'violates', 'violate')} '{label}'."
        + (f" Column-level: {'; '.join(problems)}." if problems else "")
    )


def column_fails(identity: str, label: str, problems: Sequence[str]) -> str:
    """A check that only column-level limits break."""
    return f"{identity} fails '{label}': {'; '.join(problems)}."


def column_checks_passed(count: int) -> str:
    return f"{plural(count, 'column check')} passed."


def required_column_absent(dataset: str, column: str, available: Iterable[str] = ()) -> str:
    return (
        f"Not assessed: required column {dataset}.{column} is absent"
        f"{absent_column_hint(column, available)}."
    )


def check_not_assessed(label: str, dataset: str, column: str, reason: str) -> str:
    return f"Not assessed: '{label}' on {dataset}.{column}: {reason}."


def too_few_applicable(rows: int, where: str | None, minimum: int) -> str:
    return f"only {rows} row(s) match 'where: {where}', fewer than min_applicable_rows {minimum}"


def statistic_undefined(name: str, usable: int, minimum: int) -> str:
    return f"{name} is undefined: {usable} usable value(s), at least {minimum} required"


def statistic_minimum(name: str) -> int:
    """The usable values a statistic needs to be defined."""
    return 2 if name == "std" else 0 if name == "distinct_count" else 1


def unit_column_absent(unit_column: str, dataset: str, column: str) -> str:
    return f"Not assessed: unit column {unit_column!r} of {dataset}.{column} does not exist."


def all_not_applicable(check: str) -> str:
    return f"Not assessed: {check}: every value is N/A, so its numeric statistic is undefined."


# -------------------------------------------------------------------- INP-004
def key_check(dataset: str, columns: Sequence[str], primary: bool) -> str:
    """A key contract's check label: the dataset for its primary key."""
    return dataset if primary else f"{dataset}: unique ({', '.join(columns)})"


def key_label(dataset: str, columns: Sequence[str]) -> str:
    return f"{dataset} ({', '.join(columns)})"


def key_columns_absent(
    what: str, dataset: str, absent: Sequence[str], available: Sequence[str]
) -> str:
    return (
        f"{what} of {dataset} names column(s) that do not exist: {', '.join(absent)}"
        f"{''.join(absent_column_hint(c, available) for c in absent)}."
    )


def null_keys(count: int, rows: int, label: str) -> str:
    return (
        f"{plural(count, 'row')} of {rows:,} in {label} "
        f"{agree(count, 'has', 'have')} a null key value."
    )


def duplicated_keys(keys: int, total: int, label: str, involved: int, rows: int) -> str:
    return (
        f"{plural(keys, 'duplicated key')} of {total:,} in {label}, "
        f"shared by {plural(involved, 'row')} of {rows:,}."
    )


def duplicate_rows(count: int, rows: int, dataset: str) -> str:
    return (
        f"{plural(count, 'row')} of {rows:,} in {dataset} "
        f"{agree(count, 'is', 'are')} identical to another row in every column."
    )


def key_contracts_passed(count: int) -> str:
    return f"{plural(count, 'key contract')} passed."


def grouped_values_label(dataset: str, name: str | None, columns: Sequence[str]) -> str:
    return f"{dataset}: {name or 'grouped values (' + ', '.join(columns) + ')'}"


def group_values_violate(count: int, total: int) -> str:
    return (
        f"{plural(count, 'group value')} of {total:,} "
        f"{agree(count, 'violates its', 'violate their')} occurrence contract."
    )


def group_values_satisfy(total: int) -> str:
    return (
        f"{plural(total, 'group value')} "
        f"{agree(total, 'satisfies its', 'satisfy their')} occurrence contract."
    )


# -------------------------------------------------------------------- INP-005
def reference_label(dataset: str, columns: Sequence[str], parent: str) -> str:
    return f"{dataset}({', '.join(columns)}) -> {parent}"


def orphaned_keys(keys: int, total: int, label: str, rows: int) -> str:
    return f"{plural(keys, 'orphaned key')} of {total:,} in {label}, used by {plural(rows, 'row')}."


def repeated_reference_keys(
    keys: int, dataset: str, columns: Sequence[str], label: str, cardinality: str
) -> str:
    return (
        f"{plural(keys, 'key')} of {dataset}({', '.join(columns)}) "
        f"{agree(keys, 'occurs', 'occur')} more than once, but {label} is "
        f"{cardinality.replace('_', '-')}: a join would multiply rows."
    )


def partly_null_keys(rows: int, label: str) -> str:
    return (
        f"{plural(rows, 'row')} of {label} {agree(rows, 'has', 'have')} a partly null key "
        "(match: full allows only all-null keys)."
    )


def reference_columns_absent(label: str) -> str:
    return f"Not assessed: foreign key {label} refers to columns that do not exist."


def blocked_reference(label: str, reason: str | None) -> str:
    return f"Not assessed: foreign key {label}: {reason}."


def foreign_keys_resolved(count: int) -> str:
    return f"{plural(count, 'foreign key')} fully resolved."


# ---------------------------------------------------------------- INP-006/007
def expression_violations(count: int, rows: int, dataset: str, expression: str) -> str:
    return (
        f"{plural(count, 'row')} of {rows:,} in {dataset} "
        f"{agree(count, 'violates', 'violate')} '{expression}'."
    )


def expression_not_evaluated(label: str, dataset: str, reason: str) -> str:
    return f"Not assessed: expression '{label}' on {dataset} could not be evaluated: {reason}"


def expression_input_unavailable(label: str, dataset: str, reason: str | None) -> str:
    return f"Not assessed: expression '{label}' on {dataset} could not read its input: {reason}"


def expressions_hold(count: int) -> str:
    return f"{plural(count, 'row expression')} {agree(count, 'holds', 'hold')}."


def assertion_rows(name: str, count: int) -> str:
    return f"Assertion '{name}' returned {plural(count, 'offending row')}."


def assertion_not_evaluated(name: str, reason: str) -> str:
    return f"Not assessed: assertion '{name}' could not be evaluated: {reason}"


def assertions_hold(count: int) -> str:
    return f"{plural(count, 'SQL assertion')} {agree(count, 'holds', 'hold')}."


# -------------------------------------------------------------------- INP-009
def not_applicable_population(dataset: str, column: str, count: int, rows: int) -> str:
    return f"{dataset}.{column}: {count:,} of {rows:,} values are not applicable."


def na_group_label(dataset: str, columns: Sequence[str]) -> str:
    return f"{dataset}: all-or-none N/A ({', '.join(columns)})"


def na_group_satisfied(rows: int, dataset: str) -> str:
    return f"All {rows:,} rows in {dataset} satisfy the all-or-none N/A group."


def na_group_violations(count: int, rows: int, dataset: str) -> str:
    return f"{count:,} of {rows:,} rows in {dataset} have N/A in only part of the group."


def na_group_absent(absent: Sequence[str]) -> str:
    return f"Not assessed: N/A group columns are absent: {', '.join(absent)}."
