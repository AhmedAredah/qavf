"""Pydantic configuration models and YAML/JSON loading.

A configuration file describes *what* to validate (datasets, their sources,
their column-level QA limits and named ``checks``), the pass/fail policy of a
``run`` and *where* results go. ``qavf init`` writes a complete starting
configuration and ``qavf schema`` a JSON Schema for editor autocompletion.

String values may reference environment variables as ``${NAME}`` or
``${NAME:-default}``, which keeps credentials out of configuration files.
"""

from __future__ import annotations

import importlib
import logging
import math
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    field_serializer,
    field_validator,
    model_validator,
)

from qavf.checks.items import CheckItem, RunSettings
from qavf.checks.legacy import reject_legacy
from qavf.configuration import ConfigurationGraph
from qavf.configuration import expand_env as expand_env
from qavf.configuration import read_config_mapping as _read_config_mapping
from qavf.configuration import resolve_config_text as _resolve_config_text
from qavf.engine.severity import Severity
from qavf.number_grammar import AllowedValueDomain, NumberGrammar, Whitespace, converted_options
from qavf.scalars import normalize_scalar, normalize_scalar_sequence
from qavf.text_types import TEXT_DTYPES, TextType, Word
from qavf.value_policy import TextNormalization

if TYPE_CHECKING:
    from qavf.checks.selection import AppliedSelection
    from qavf.sources.base import SourceEnvironment

logger = logging.getLogger(__name__)

DType = Literal["int", "float", "decimal", "str", "bool", "date", "datetime"]


def _preserve_numeric_type(value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
    """Validate bounds without changing the meaning of an existing numeric value.

    Older Pydantic versions prefer Decimal over float in this union and convert a
    binary float through its decimal display. That changes an inclusive boundary.
    String bounds still follow normal parsing, including exact quoted Decimals.
    """
    value = normalize_scalar(value)
    if isinstance(value, bool):
        raise ValueError("numeric bounds must not be boolean")
    parsed = handler(value)
    return value if type(value) in (int, float, Decimal) else parsed


NumericBound = Annotated[int | Decimal | float, WrapValidator(_preserve_numeric_type)]
Bound = Annotated[int | Decimal | float | datetime | date, WrapValidator(_preserve_numeric_type)]
"""A numeric or temporal bound. Integers stay integers (exact beyond 2**53); YAML
dates (``2020-01-01``) parse as dates."""


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos in configuration fail loudly."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        use_attribute_docstrings=True,
        hide_input_in_errors=True,
    )

    @field_validator("*", mode="before")
    @classmethod
    def _scientific_scalar(cls, value: Any) -> Any:
        """Resolve foreign scalar identity before Pydantic coerces a field."""
        return normalize_scalar(value)


# ------------------------------------------------------------------ QA limits
def _require_finite(name: str, value: Any) -> None:
    if (isinstance(value, float) and not math.isfinite(value)) or (
        isinstance(value, Decimal) and not value.is_finite()
    ):
        raise ValueError(f"'{name}' must be a finite number, got {value}; omit it for no limit")


def _comparable(value: Any) -> Any:
    """Dates compare as midnight datetimes, so date and datetime bounds can be ordered."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    return value


def _check_range(lower: list[tuple[str, Any, bool]], upper: list[tuple[str, Any, bool]]) -> None:
    """Reject bounds no value can satisfy. Entries are ``(name, value, inclusive)``."""
    for low_name, low, low_inclusive in lower:
        for high_name, high, high_inclusive in upper:
            try:
                a, b = _comparable(low), _comparable(high)
                empty = a > b or (a == b and not (low_inclusive and high_inclusive))
            except TypeError as exc:
                raise ValueError(
                    f"'{low_name}' and '{high_name}' cannot be compared: {exc}"
                ) from exc
            if empty:
                raise ValueError(
                    f"'{low_name}: {low}' and '{high_name}: {high}' leave no allowed value"
                )


class Bounds(_Strict):
    """Inclusive bounds for an aggregate statistic; either side may be omitted."""

    ge: NumericBound | None = None
    le: NumericBound | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Bounds:
        if self.ge is None and self.le is None:
            raise ValueError("set 'ge', 'le' or both")
        _require_finite("ge", self.ge)
        _require_finite("le", self.le)
        if self.ge is not None and self.le is not None and self.ge > self.le:
            raise ValueError("'ge' must not exceed 'le'")
        return self


#: Row-level constraint fields shared by :class:`ColumnConstraints` subclasses.
ROW_CONSTRAINTS = (
    "ge",
    "le",
    "gt",
    "lt",
    "isin",
    "notin",
    "regex",
    "min_length",
    "max_length",
    "integral",
    "max_decimals",
)
#: Column-level (aggregate) constraint fields.
COLUMN_CONSTRAINTS = (
    "max_null_fraction",
    "not_null",
    "unique",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "sum",
    "distinct_count",
)


class ColumnConstraints(_Strict):
    """QA limits for a column. Every limit that is set must hold.

    Row-level limits flag individual values; column-level limits test the
    column as a whole.
    """

    # row level
    ge: Bound | None = None
    le: Bound | None = None
    gt: Bound | None = None
    lt: Bound | None = None
    isin: list[Any] | None = None
    """Allowed values."""
    notin: list[Any] | None = None
    """Forbidden values (sentinels such as -999)."""
    regex: str | None = None
    """Pattern every text value must match in full."""
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    integral: bool = False
    """Every value is a whole number (for float columns; integers always are)."""
    max_decimals: int | None = Field(default=None, ge=0)
    """At most this many decimal places; exact for decimals, bounded rounding noise for floats."""
    # column level
    max_null_fraction: float | None = Field(default=None, ge=0, le=1)
    """Largest acceptable share of missing values (requires ``nullable: true``
    on the column to be meaningful; non-nullable columns allow none)."""
    not_null: bool = False
    """Every value is present: with ``where``, a field required only in some rows
    (``{where: "status = 'closed'", not_null: true}``). Missing rows are listed."""
    unique: bool = False
    """No value occurs twice; missing values are exempt."""
    each_allowed_once: bool = False
    """Every distinct allowed value in ``isin`` must occur exactly once. Missing
    and N/A values are outside the domain; their own policies still apply."""
    convert_options: bool = False
    """Parse numeric string options in ``isin``/``notin`` with the declared
    column's number grammar. Off by default to preserve strict literal typing."""
    # modifiers
    normalize: list[TextNormalization] = Field(default_factory=list)
    """Compare text in ``isin``, ``notin`` and ``unique`` after normalizing it:
    ``case`` ignores letter case (``Valid`` is ``valid``), ``trim`` ignores
    surrounding whitespace. Applies to the limits set alongside it."""
    min: Bounds | None = None
    """Bounds on the column minimum."""
    max: Bounds | None = None
    """Bounds on the column maximum."""
    mean: Bounds | None = None
    median: Bounds | None = None
    std: Bounds | None = None
    sum: Bounds | None = None
    distinct_count: Bounds | None = None

    @field_validator("isin", "notin", "missing_values", mode="before", check_fields=False)
    @classmethod
    def _scientific_literals(cls, value: Any) -> Any:
        return normalize_scalar_sequence(value)

    def constraints(self) -> dict[str, Any]:
        """The limits that are set, by field name."""
        names = (*ROW_CONSTRAINTS, *COLUMN_CONSTRAINTS)
        values = {name: getattr(self, name) for name in names}
        return {k: v for k, v in values.items() if v is not None and v is not False}

    @model_validator(mode="after")
    def _satisfiable(self) -> ColumnConstraints:
        """Reject limits that are invalid, or that no value could ever satisfy.

        Boundary cases that admit a value (``ge == le``) are valid.
        """
        bounds = {name: getattr(self, name) for name in ("ge", "le", "gt", "lt")}
        bounds = {name: value for name, value in bounds.items() if value is not None}
        for name, value in bounds.items():
            _require_finite(name, value)
        kinds = {"date" if isinstance(value, date) else "number" for value in bounds.values()}
        if len(kinds) > 1:
            raise ValueError(
                f"bounds mix numbers and dates ({', '.join(sorted(bounds))}); use one kind"
            )
        _check_range(
            [(n, bounds[n], n == "ge") for n in ("ge", "gt") if n in bounds],
            [(n, bounds[n], n == "le") for n in ("le", "lt") if n in bounds],
        )
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError(
                f"'min_length: {self.min_length}' exceeds 'max_length: {self.max_length}'"
            )
        if self.isin is not None and not self.isin:
            raise ValueError("'isin: []' allows no value at all; list the allowed values")
        if self.each_allowed_once:
            if self.isin is None:
                raise ValueError("'each_allowed_once' requires 'isin'")
            if any(
                value is None or (isinstance(value, float) and math.isnan(value))
                for value in self.isin
            ):
                raise ValueError("'each_allowed_once' domain cannot include missing values")
        if self.convert_options and self.isin is None and self.notin is None:
            raise ValueError("'convert_options' requires 'isin' or 'notin'")
        if self.normalize and self.isin is None and self.notin is None and not self.unique:
            raise ValueError("'normalize' goes with 'isin', 'notin' or 'unique'")
        if len(set(self.normalize)) != len(self.normalize):
            raise ValueError("'normalize' lists a normalization twice")
        if sum(form in self.normalize for form in ("NFC", "NFD", "NFKC", "NFKD")) > 1:
            raise ValueError("'normalize' accepts one Unicode normalization form")
        if "case" in self.normalize and "casefold" in self.normalize:
            raise ValueError("choose either case or casefold normalization")
        if self.regex is not None:
            _check_regex(self.regex)
        for name in ("std", "distinct_count"):
            limit = getattr(self, name)
            if limit is not None and limit.le is not None and limit.le < 0:
                raise ValueError(f"'{name}' can never be negative, but 'le' is {limit.le}")
        return self


def _check_where(condition: str, what: str = "where condition") -> None:
    """Admit SQL syntax independently of optional installed execution engines.

    Name/function binding belongs to the selected backend and is isolated to
    each contract. A machine's optional packages cannot change configuration.
    """
    import duckdb

    from qavf.sources.sql import _parser

    try:
        statements = _parser().extract_statements(f"SELECT 1 WHERE ({condition})")
        if len(statements) != 1:
            raise ValueError(f"invalid {what}: expected one expression")
    except duckdb.Error as exc:
        raise ValueError(f"invalid {what}: {exc}") from None


def _check_regex(pattern: str) -> None:
    """Admit shared syntax; selected backends compile their own dialect."""
    from qavf.regex_policy import check_regex_syntax

    check_regex_syntax(pattern)


class ColumnCheck(ColumnConstraints):
    """A named group of QA limits with its own severity."""

    name: str | None = None
    """Label shown in reports; generated from the limits when omitted."""
    severity: Severity | None = None
    """Severity when this check fails; the column-check rule's severity when omitted."""
    description: str | None = None
    where: str | None = None
    """Apply the check only to rows matching this SQL condition (Polars SQL syntax),
    e.g. ``"status = 'valid'"``; other rows are outside its scope."""
    min_applicable_rows: int = Field(default=0, ge=0)
    """With ``where``: fewer matching rows than this makes the check not assessed."""

    @model_validator(mode="after")
    def _not_empty(self) -> ColumnCheck:
        if not self.constraints():
            raise ValueError("a check must set at least one limit (ge, le, isin, regex, …)")
        if self.where is not None:
            _check_where(self.where)
        elif self.min_applicable_rows:
            raise ValueError("'min_applicable_rows' goes with 'where'")
        return self


class ColumnSpec(ColumnConstraints):
    """Expectations for one column.

    Limits set directly on the column form an implicit check at the default
    severity; ``checks`` adds further checks with their own severities::

        vmt:
          dtype: float
          ge: 0                         # ERROR (rule default)
          checks:
            - {le: 1.0e9, severity: WARNING, name: implausibly large}
    """

    dtype: DType | None = None
    """Logical type. ``float`` also accepts integers; ``datetime`` also accepts dates."""
    number_grammar: NumberGrammar = Field(default_factory=NumberGrammar)
    """Lexical grammar for numeric text; defaults preserve strict integer spelling."""
    trim_whitespace: Whitespace = "ascii"
    """Trim surrounding U+0020 spaces (``ascii``, default), the shared Unicode
    whitespace set (``unicode``), or no characters (``none``) before type parsing."""
    decimal_precision: int | None = Field(default=None, ge=1, le=38)
    decimal_scale: int | None = Field(default=None, ge=0, le=38)
    """Optional exact fixed-point contract; precision and scale must be supplied together."""
    datetime_unit: Literal["ms", "us", "ns"] = "us"
    timezone: str | None = None
    """Optional IANA timezone for text timestamps; canonical values are UTC-naive."""
    ambiguous: Literal["raise", "earliest", "latest", "null"] = "raise"
    nonexistent: Literal["raise", "null"] = "raise"
    """DST folds require earliest/latest to be accepted; gaps are always invalid text."""
    required: bool = True
    nullable: bool = False
    """Allow missing values. NaN counts as missing, exactly like null."""
    allow_infinite: bool = False
    """Allow +/-infinity in a floating-point column (rejected by default)."""
    missing_values: list[Any] = Field(default_factory=list)
    """Values that mean "missing" (sentinels such as -999): counted as missing for
    ``nullable`` and ``max_null_fraction``. They are normalized to null for every
    QA consumer, including statistics, reconciliation, keys and expressions;
    each check then applies its documented null policy."""
    na_values: list[str] = Field(default_factory=list)
    """Valid, case-sensitive not-applicable tokens. They are distinct from missing
    values, satisfy nullability, and are excluded from value checks and aggregate
    populations. Token matching follows ``trim_whitespace``. No tokens are built in."""
    blank_as_missing: bool = False
    """Normalize empty and whitespace-only text to missing for every QA consumer. With
    a non-text ``dtype``, this applies while a column read as text is checked against
    that type."""
    format: str | None = None
    """For ``date`` and ``datetime`` written another way than ISO 8601: how values are
    written, e.g. ``'%m/%d/%Y'`` or ``'%d %b %Y %H:%M'`` (directives ``%Y %y %m %d %b
    %B %j``, and ``%H %I %M %S %p`` for datetimes)."""
    true_values: list[Word] = Field(default_factory=lambda: ["true"])
    """For ``bool`` written as text: the words meaning true (case and surrounding space
    ignored), e.g. ``[Y, yes, 1]``."""
    false_values: list[Word] = Field(default_factory=lambda: ["false"])
    """For ``bool`` written as text: the words meaning false."""
    unit: str | None = None
    """The column's unit (``kg``, ``vehicle-miles``). Recorded in reports; ``REC-001``
    refuses to reconcile columns declared in different units."""
    unit_column: str | None = None
    """A column holding each row's unit; with ``allowed_units`` (or ``unit``), rows in
    any other unit are violations."""
    allowed_units: list[str] | None = None
    description: str | None = None
    checks: list[ColumnCheck] = Field(default_factory=list)

    def text_type(self) -> TextType | None:
        """How this column's declared type is written as text; ``None`` when its type
        is not recognized in text by every engine (``str``, ``decimal``, undeclared)."""
        if self.dtype not in TEXT_DTYPES:
            return None
        return TextType.model_validate(
            {
                "dtype": self.dtype,
                "number_grammar": self.number_grammar,
                "trim_whitespace": self.trim_whitespace,
                "missing": [value for value in self.missing_values if isinstance(value, str)],
                "blank_as_missing": self.blank_as_missing,
                "format": self.format,
                "true_values": self.true_values,
                "false_values": self.false_values,
                "datetime_unit": self.datetime_unit,
                "timezone": self.timezone,
                "ambiguous": self.ambiguous,
                "nonexistent": self.nonexistent,
            }
        )

    def all_checks(self) -> list[ColumnCheck]:
        """The implicit check (if any limits are set inline) followed by ``checks``."""
        inline = self.constraints()
        if inline and self.normalize:
            inline["normalize"] = list(self.normalize)
        if inline:
            inline["convert_options"] = self.convert_options
            inline["each_allowed_once"] = self.each_allowed_once
        implicit = [ColumnCheck.model_validate(inline)] if inline else []
        return [self._resolved_check(check) for check in implicit + list(self.checks)]

    def _resolved_check(self, check: ColumnCheck) -> ColumnCheck:
        """Compile optional domain conveniences to the shared primitive contracts."""
        if not check.convert_options and not check.each_allowed_once and not self.timezone:
            return check
        raw = check.model_dump()
        if self.timezone:
            from qavf.text_policies import temporal_literal

            for name in ("ge", "le", "gt", "lt"):
                raw[name] = temporal_literal(raw[name], self)
            for name in ("isin", "notin"):
                if raw[name] is not None:
                    raw[name] = [temporal_literal(value, self) for value in raw[name]]
        if check.convert_options:
            if self.dtype not in ("int", "float", "decimal"):
                raise ValueError("'convert_options' requires a declared numeric dtype")
            for name in ("isin", "notin"):
                if raw[name] is not None:
                    raw[name] = converted_options(
                        raw[name], self.dtype, self.number_grammar, self.trim_whitespace
                    )
        if check.each_allowed_once:
            count = AllowedValueDomain.resolved(raw["isin"], check.normalize).cardinality
            if check.distinct_count and (
                (check.distinct_count.ge is not None and check.distinct_count.ge > count)
                or (check.distinct_count.le is not None and check.distinct_count.le < count)
            ):
                raise ValueError("'distinct_count' contradicts 'each_allowed_once' domain size")
            raw.update(unique=True, distinct_count={"ge": count, "le": count})
            if raw["name"] is None:
                raw["name"] = "each allowed value exactly once"
        return ColumnCheck.model_validate(raw)

    @model_validator(mode="after")
    def _compatible_with_dtype(self) -> ColumnSpec:
        """With a declared ``dtype``, reject limits that cannot apply to it."""
        if self.na_values:
            from qavf.applicability import NotApplicablePolicy

            NotApplicablePolicy.from_spec(self).validate(self)
        if self.not_null:
            raise ValueError(
                "'not_null' goes in a check, usually with 'where'; for the whole column "
                "use 'nullable: false'"
            )
        if self.format is not None and self.dtype not in ("date", "datetime"):
            raise ValueError("'format' goes with dtype date or datetime")
        if (self.decimal_precision is None) != (self.decimal_scale is None):
            raise ValueError("decimal_precision and decimal_scale must be supplied together")
        if self.decimal_precision is not None:
            assert self.decimal_scale is not None
            if self.dtype != "decimal":
                raise ValueError("decimal_precision/decimal_scale require dtype decimal")
            if self.decimal_scale > self.decimal_precision:
                raise ValueError("decimal_scale cannot exceed decimal_precision")
        if (self.datetime_unit != "us" or self.timezone is not None) and self.dtype != "datetime":
            raise ValueError("datetime_unit and timezone require dtype datetime")
        if self.timezone is None and (self.ambiguous != "raise" or self.nonexistent != "raise"):
            raise ValueError("DST policies require timezone")
        words = (self.true_values, self.false_values)
        if self.dtype != "bool" and words != (["true"], ["false"]):
            raise ValueError("'true_values' and 'false_values' go with dtype bool")
        self.text_type()  # validates format directives and true/false words
        if self.allowed_units is not None and not self.allowed_units:
            raise ValueError("'allowed_units' must not be empty")
        if self.unit_column is not None and self.unit is None and self.allowed_units is None:
            raise ValueError("'unit_column' needs 'unit' or 'allowed_units'")
        if self.allowed_units is not None and self.unit_column is None:
            raise ValueError("'allowed_units' goes with 'unit_column'")
        if self.dtype is None:
            return self
        numeric = self.dtype in ("int", "float", "decimal")
        temporal = self.dtype in ("date", "datetime")
        for check in self.all_checks():
            limits = check.constraints()
            for name in ("ge", "le", "gt", "lt"):
                if name not in limits:
                    continue
                is_date = isinstance(limits[name], date)
                if not (temporal if is_date else numeric):
                    kind = "date" if is_date else "numeric"
                    raise ValueError(
                        f"a {kind} bound '{name}' cannot apply to a {self.dtype} column"
                    )
            for name in ("regex", "min_length", "max_length"):
                if name in limits and self.dtype != "str":
                    raise ValueError(f"'{name}' applies to str columns, not {self.dtype}")
            if check.normalize and self.dtype != "str":
                raise ValueError(f"'normalize' applies to str columns, not {self.dtype}")
            for name in ("integral", "max_decimals"):
                if name in limits and not numeric:
                    raise ValueError(f"'{name}' applies to numeric columns, not {self.dtype}")
            for name in ("min", "max", "mean", "median", "std", "sum"):
                if name in limits and not numeric:
                    raise ValueError(
                        f"the '{name}' statistic needs a numeric column, not {self.dtype}"
                    )
        return self


class UniqueKeySpec(_Strict):
    """A composite alternate key and its explicit null-equality contract."""

    columns: list[str] = Field(min_length=1)
    nulls: Literal["exempt", "equal", "forbid"] = "exempt"
    """``exempt`` ignores rows with any null component; ``equal`` compares nulls
    as equal at the same position; ``forbid`` reports them as invalid key rows."""
    where: str | None = None
    """Assess only rows where this expression on canonical values is true."""
    name: str | None = None

    @model_validator(mode="after")
    def _condition(self) -> UniqueKeySpec:
        if self.where is not None:
            _check_where(self.where)
        return self


class GroupedValueSpec(_Strict):
    """Occurrence bounds for an allowed domain independently within each group."""

    column: str
    group_by: list[str] = Field(min_length=1)
    allowed_values: list[Any] = Field(min_length=1)
    min_occurrences: int = Field(default=1, ge=0)
    max_occurrences: int | None = Field(default=1, ge=0)
    allow_other_values: bool = False
    where: str | None = None
    expected_groups: list[list[Any]] | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> GroupedValueSpec:
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by columns must be distinct")
        if self.column in self.group_by:
            raise ValueError("the value column cannot also identify its group")
        if self.max_occurrences is not None and self.min_occurrences > self.max_occurrences:
            raise ValueError("min_occurrences must not exceed max_occurrences")
        if any(value is None for value in self.allowed_values):
            raise ValueError("allowed_values cannot contain missing values")
        if self.expected_groups is not None and any(
            len(group) != len(self.group_by) for group in self.expected_groups
        ):
            raise ValueError("each expected group must match the group_by width")
        if self.where is not None:
            _check_where(self.where)
        return self


class SchemaSpec(_Strict):
    """Expectations for a whole dataset."""

    columns: dict[str, ColumnSpec] = Field(default_factory=dict)
    na_groups: list[list[str]] = Field(default_factory=list)
    """Column groups whose rows contain either zero or all N/A tokens (INP-009).
    Missing or invalid values do not count as N/A. Each column must declare
    ``na_values``; groups may overlap and are assessed independently."""
    primary_key: list[str] = Field(default_factory=list)
    """Columns that together must identify every row uniquely; none may be null."""
    strict: bool = False
    """Reject columns that are not declared."""
    min_rows: int = Field(default=1, ge=0)
    max_rows: int | None = Field(default=None, ge=0)
    unique_keys: list[list[str] | UniqueKeySpec] = Field(default_factory=list)
    """Alternate keys. Legacy column lists exempt null-bearing rows. Structured
    entries select ``nulls: exempt``, ``equal`` or ``forbid`` explicitly."""
    grouped_values: list[GroupedValueSpec] = Field(default_factory=list)
    """Per-group domain coverage and occurrence bounds (INP-004). Observed groups
    are retained before ``where`` filtering; ``expected_groups`` also assesses
    completely absent groups. Null group components compare equal."""
    duplicate_rows_per_key: Annotated[int, Field(strict=True, gt=0)] | Literal["all"] = 10
    """Maximum zero-based row positions per sampled duplicate key, or ``all``.
    Key and row totals remain exact independently of this evidence limit."""
    unique_rows: bool = False
    """No two rows may be identical in every column."""
    evidence_columns: Literal["checked", "all"] = "checked"
    """Columns shown for the failing rows of a column check: ``checked`` (the checked
    column, the columns of its ``where`` and unit, and ``primary_key``) or ``all``
    (whole rows, which reads every column for each failing check)."""

    @model_validator(mode="after")
    def _consistent_key(self) -> SchemaSpec:
        from qavf.applicability import NotApplicablePolicy

        for contract in self.grouped_values:
            literals = [(contract.column, value) for value in contract.allowed_values]
            literals.extend(
                (column, value)
                for group in contract.expected_groups or []
                for column, value in zip(contract.group_by, group, strict=True)
            )
            for column, value in literals:
                spec = self.columns.get(column)
                if spec and NotApplicablePolicy.from_spec(spec).matches(value):
                    raise ValueError(
                        f"grouped_values literal {value!r} for {column!r} is a declared N/A token"
                    )
        seen_groups: set[frozenset[str]] = set()
        for group in self.na_groups:
            if len(group) < 2 or len(group) != len(set(group)):
                raise ValueError("na_groups entries require at least two distinct columns")
            if frozenset(group) in seen_groups:
                raise ValueError(f"N/A group {group!r} is listed twice")
            seen_groups.add(frozenset(group))
            for column in group:
                if column not in self.columns or not self.columns[column].na_values:
                    raise ValueError(f"N/A group column {column!r} must declare na_values")
        seen_keys: set[tuple[tuple[str, ...], str | None]] = set()
        for key in self.alternate_keys:
            columns = key.columns
            if not columns:
                raise ValueError("unique_keys entries must name at least one column")
            if len(set(columns)) != len(columns):
                raise ValueError(f"unique key {columns} repeats a column")
            identity = (tuple(columns), key.where)
            if identity in seen_keys:
                raise ValueError(f"unique key {columns} is listed twice")
            seen_keys.add(identity)
        if self.max_rows is not None and self.min_rows > self.max_rows:
            hint = (
                "" if "min_rows" in self.model_fields_set else " (the default; set it explicitly)"
            )
            raise ValueError(
                f"'min_rows: {self.min_rows}'{hint} exceeds 'max_rows: {self.max_rows}', so no "
                f"dataset could pass"
            )
        repeated = sorted({c for c in self.primary_key if self.primary_key.count(c) > 1})
        if repeated:
            raise ValueError(f"primary_key repeats column(s): {', '.join(repeated)}")
        for column in self.primary_key:
            spec = self.columns.get(column)
            if spec is None:
                continue
            if spec.nullable:
                raise ValueError(
                    f"primary-key column {column!r} cannot be nullable; use 'unique' on the "
                    f"column for a nullable unique constraint"
                )
            if not spec.required:
                raise ValueError(f"primary-key column {column!r} must be required")
        return self

    @property
    def alternate_keys(self) -> list[UniqueKeySpec]:
        """Resolve legacy list declarations without changing their public shape."""
        return [
            key if isinstance(key, UniqueKeySpec) else UniqueKeySpec(columns=key)
            for key in self.unique_keys
        ]


class ForeignKeySpec(_Strict):
    """Every distinct value of ``columns`` must exist in ``ref_dataset.ref_columns``."""

    columns: list[str] = Field(min_length=1)
    ref_dataset: str
    ref_columns: list[str] | None = Field(default=None, min_length=1)
    """Defaults to ``columns`` when the names match on both sides."""
    severity: Severity | None = None
    match: Literal["simple", "full"] = "simple"
    """How rows with null key components count. ``simple`` (SQL MATCH SIMPLE, the
    default): a row with any null component is exempt. ``full``: a partly null key
    is a violation; only all-null keys are exempt. Exempt rows are always counted."""
    cardinality: Literal["any", "many_to_one", "one_to_one"] = "any"
    """``many_to_one``: referenced keys must be unique in the parent (a join cannot
    multiply rows). ``one_to_one``: child keys must be unique too."""

    @property
    def target_columns(self) -> list[str]:
        """Referenced column names, resolved against the default."""
        return self.ref_columns or self.columns

    @model_validator(mode="after")
    def _same_arity(self) -> ForeignKeySpec:
        if self.ref_columns is not None and len(self.ref_columns) != len(self.columns):
            raise ValueError("ref_columns must have the same length as columns")
        for name, columns in (("columns", self.columns), ("ref_columns", self.ref_columns or [])):
            repeated = sorted({c for c in columns if columns.count(c) > 1})
            if repeated:
                raise ValueError(f"{name} repeats column(s): {', '.join(repeated)}")
        return self


class ExpressionCheck(_Strict):
    """A row-level SQL condition every row must satisfy, e.g. ``pm25 <= pm10``.

    Evaluated by DuckDB; the expression may use any SQL function and may
    reference other datasets by name in subqueries.
    """

    expr: str
    name: str | None = None
    severity: Severity | None = None
    description: str | None = None
    null_as: Literal["pass", "fail"] = "pass"
    """How a row whose expression evaluates to NULL counts (SQL CHECK semantics: pass)."""


class SqlAssertion(_Strict):
    """A DuckDB query that returns offending rows; it passes when it returns none.

    Every dataset is available as a table under its name::

        sql: SELECT county FROM inventory GROUP BY county HAVING sum(nox) = 0
    """

    name: str
    sql: str
    severity: Severity | None = None
    description: str | None = None


class ViewSpec(_Strict):
    """A named slice of a dataset, used as ``<dataset>.<view>`` wherever a check
    names a dataset. It has the dataset's columns and types."""

    where: str | None = None
    """SQL condition selecting the view's rows (all rows when omitted)."""
    columns: list[str] | None = Field(default=None, min_length=1)
    """Columns the view keeps (all when omitted)."""
    description: str | None = None

    @field_validator("where")
    @classmethod
    def _valid_where(cls, condition: str | None) -> str | None:
        if condition is not None:
            _check_where(condition, "view condition")
        return condition


class SourcePolicy(_Strict):
    """Loading safeguards independent of the input's location or representation.

    Only explicitly supplied fields override a source's own policy. Changing a
    file into an in-memory frame keeps its row budget; retries apply only where
    a source declares that retrying is safe.
    """

    retries: int = Field(default=0, ge=0, le=10)
    """Extra attempts for explicitly retryable failures of a retry-safe source."""
    retry_delay_s: float = Field(default=1.0, ge=0, le=300)
    """Delay before the first retry, doubled on each additional attempt."""
    row_budget: int | None = Field(default=None, ge=1)
    """Maximum input result rows, preserved across representations. This does
    not authorize sampling and does not bound rows scanned by a source query."""

    @model_validator(mode="after")
    def _finite_delay(self) -> SourcePolicy:
        _require_finite("retry_delay_s", self.retry_delay_s)
        return self


class DatasetSpec(_Strict):
    """Where a dataset comes from and what it must look like."""

    path: str | None = None
    """Shorthand for ``source: {type: file, path: ...}``."""
    format: Literal["csv", "parquet", "json", "ndjson", "excel"] | None = None
    """With ``path``: the file format, inferred from the suffix when omitted."""
    source: dict[str, Any] | None = None
    """Full source specification, e.g. ``{type: query, sql: ...}``. See ``qavf schema``."""
    load_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    """Location-independent safeguards. Explicit values override policy fields
    inside ``source`` and are preserved when the public API replaces an input."""
    required: bool = True
    description: str | None = None
    rename: dict[str, str] = Field(default_factory=dict)
    """Column renames applied after loading (``old: new``)."""
    schema_spec: SchemaSpec | None = Field(default=None, alias="schema")
    foreign_keys: list[ForeignKeySpec] = Field(default_factory=list)
    series: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Named time series for temporal checks (``series: <name>``): ``value``,
    ``time_column``, ``group_by``, ``aggregate`` and the other settings of an inline
    series."""
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Named per-entity metrics for spatial checks (``metric: <name>``): ``value``,
    ``entity``, ``group_by``, ``aggregate`` and ``per`` (normalizer, e.g. ``counts.aadt``)."""
    views: dict[str, ViewSpec] = Field(default_factory=dict)
    """Named slices of this dataset, usable as ``<dataset>.<view>`` wherever a check
    names a dataset."""
    checks: list[CheckItem] = Field(default_factory=list)
    """Checks of this dataset, e.g. ``{check: missing_periods, series: aadt_by_site}``.
    ``qavf list-checks`` lists the checks and their settings."""
    across: list[str] | dict[str, dict[str, Any] | list[str | int] | None] | None = None
    """Copy this dataset across several sources: a list of connection names, a mapping
    of copy names to the source settings that differ (``connection``, ``path``,
    ``secret``, ``table``, ...), or ``{key: [values]}`` filling ``{key}`` in the
    source's path or table. Each copy is a dataset of its own, ``<dataset>@<copy>``;
    ``<dataset>@*`` lists every copy."""

    @model_validator(mode="after")
    def _expanded(self) -> DatasetSpec:
        if self.across is not None:
            raise ValueError(
                "across is expanded when a configuration is validated as a whole; "
                "give the copies as datasets of their own here"
            )
        return self

    @property
    def expressions(self) -> list[ExpressionCheck]:
        """The enabled ``expression`` checks, as the row expressions INP-006 evaluates."""
        return [
            expression_check(item)
            for item in self.checks
            if item.check == "expression" and item.enabled
        ]

    @field_serializer("load_policy")
    def _serialize_load_policy(self, value: SourcePolicy) -> dict[str, Any]:
        # Preserve which policy fields override the source across settings round-trips.
        return value.model_dump(exclude_unset=True)

    @model_validator(mode="after")
    def _one_source(self) -> DatasetSpec:
        if self.path is not None and self.source is not None:
            raise ValueError("set either 'path' or 'source', not both")
        targets: dict[str, str] = {}
        for old, new in self.rename.items():
            other = targets.setdefault(new, old)
            if other != old:
                raise ValueError(
                    f"rename maps both {other!r} and {old!r} to {new!r}; each column needs "
                    f"its own name"
                )
        if self.format is not None and self.path is None:
            raise ValueError("'format' goes with 'path'; put it inside 'source' otherwise")
        return self

    def source_config(self) -> dict[str, Any] | None:
        """The effective source specification, expanding the ``path`` shorthand."""
        if self.path is not None:
            config: dict[str, Any] = {"type": "file", "path": self.path}
            if self.format is not None:
                config["format"] = self.format
        elif self.source is not None:
            config = dict(self.source)
        else:
            return None
        config.update(self.effective_load_policy().model_dump(exclude_unset=True))
        self._preserve_applicability(config)
        self._preserve_decimal_text(config)
        text = self._declared_text(config)
        if text:
            config["text_columns"] = list(dict.fromkeys([*config.get("text_columns", []), *text]))
        declared = self._declared_types(config)
        if declared:
            config["declared_columns"] = {**declared, **config.get("declared_columns", {})}
        return config

    def _preserve_decimal_text(self, config: dict[str, Any]) -> None:
        """Keep decimal spellings intact until their exact column scale is known.

        Unlike fixed-width text types, decimal conversion is resolved after
        ingestion. A reader's DECIMAL precision/scale hint must not round those
        values before the validation policy can inspect them.
        """
        if not self._csv(config) or self.schema_spec is None:
            return
        from qavf.sources.csv_diagnostics import CsvPolicy

        original = {new: old for old, new in self.rename.items()}
        governed = [
            original.get(column, column)
            for column, spec in self.schema_spec.columns.items()
            if spec.dtype == "decimal"
        ]
        if governed:
            config["options"] = CsvPolicy.protect_columns(config.get("options", {}), governed)

    def _preserve_applicability(self, config: Mapping[str, Any]) -> None:
        """A CSV reader must not erase declared N/A before validation sees it."""
        if not self._csv(config) or self.schema_spec is None:
            return
        from qavf.applicability import NotApplicablePolicy

        options = {str(name).casefold(): value for name, value in config.get("options", {}).items()}
        markers = options.get("nullstr", [])
        markers = markers if isinstance(markers, list) else [markers]
        original = {new: old for old, new in self.rename.items()}
        typed = options.get(
            "types", options.get("dtypes", options.get("column_types", options.get("columns", {})))
        )
        declared = config.get("declared_columns", {})
        for column, spec in self.schema_spec.columns.items():
            if not spec.na_values:
                continue
            policy = NotApplicablePolicy.from_spec(spec)
            if any(policy.matches(marker) for marker in markers):
                raise ValueError(f"CSV nullstr would erase na_values in column {column!r}")
            source_name = original.get(column, column)
            text_types = {"VARCHAR", "STRING", "TEXT"}
            if source_name in declared or (
                isinstance(typed, Mapping)
                and source_name in typed
                and str(typed[source_name]).upper() not in text_types
            ):
                raise ValueError(f"CSV column {column!r} with na_values must be read as text")
            if isinstance(typed, list) and any(
                str(dtype).upper() not in text_types for dtype in typed
            ):
                raise ValueError(
                    "CSV positional types with na_values must all be text; "
                    "use named types for other columns"
                )

    def _declared_types(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Columns of a CSV file source to read as text and convert to their declared
        type in the same scan, so a reader's sample never decides their type.

        A reader that typed a column from its first rows silently rounds a later
        ``2.5`` into an integer and reads ``0x10`` as 16. A ``datetime`` without a
        ``format`` keeps the reader's typing: it fails on a bad value rather than
        changing it, and it alone reads time-zone offsets (``...Z``) as instants.
        """
        if not self._csv(config) or self.schema_spec is None:
            return {}
        original = {new: old for old, new in self.rename.items()}
        declared = {}
        for column, spec in self.schema_spec.columns.items():
            if spec.na_values:
                continue  # Preserve applicability tokens until column checks consume them.
            text = spec.text_type()
            if text is None or (
                text.dtype == "datetime" and text.format is None and not text.governed_datetime
            ):
                continue
            declared[original.get(column, column)] = text.model_dump()
        return declared

    @staticmethod
    def _csv(config: Mapping[str, Any]) -> bool:
        if config.get("type") != "file":
            return False
        from qavf.sources.formats import infer_format

        try:
            return (config.get("format") or infer_format(str(config.get("path", "")))) == "csv"
        except ValueError:
            return False  # the source's own validation reports an unknown format

    def _declared_text(self, config: Mapping[str, Any]) -> list[str]:
        """Source names of columns that must retain raw CSV text.

        A type sniffed from a sample would turn codes such as ``007`` into numbers
        and drop their leading zeros before any check could see them. Exact
        decimals need their original fraction to resolve a lossless column scale.
        """
        if not self._csv(config) or self.schema_spec is None:
            return []
        original = {new: old for old, new in self.rename.items()}
        return [
            original.get(column, column)
            for column, spec in self.schema_spec.columns.items()
            if spec.dtype in ("str", "decimal") or spec.na_values
        ]

    def effective_load_policy(self) -> SourcePolicy:
        """Combine legacy source policy with explicit dataset-level safeguards."""
        source = self.source or {}
        inherited = {name: source[name] for name in SourcePolicy.model_fields if name in source}
        return SourcePolicy.model_validate(
            {**inherited, **self.load_policy.model_dump(exclude_unset=True)}
        )


class ConnectionSpec(_Strict):
    """A database DuckDB attaches for ``database`` and ``query`` sources."""

    type: Literal["duckdb", "sqlite", "postgres", "mysql"]
    path: str | None = None
    """Database file for ``duckdb`` and ``sqlite``; relative to the configuration file."""
    uri: str | None = None
    """libpq/MySQL connection string for ``postgres`` and ``mysql``, e.g.
    ``host=localhost user=qa database=movesdb`` (use ``${ENV}`` for passwords)."""
    read_only: bool = True
    snapshot: bool = False
    """Read every dataset that comes from this connection together, on one connection
    in one transaction, so related tables reflect a single moment. Guaranteed for DuckDB
    files (MVCC); for other databases it is what DuckDB's extension transaction gives
    (see the documentation), so add a batch check (``REC-005``) where it matters."""

    @property
    def file_based(self) -> bool:
        """Whether the database is a local file (``path``) rather than a server (``uri``)."""
        return self.type in ("duckdb", "sqlite")

    @model_validator(mode="after")
    def _target(self) -> ConnectionSpec:
        if self.file_based:
            if not self.path:
                raise ValueError(f"a {self.type} connection needs 'path'")
            if self.uri is not None:
                raise ValueError(f"a {self.type} connection takes 'path', not 'uri'")
        else:
            if not self.uri:
                raise ValueError(f"a {self.type} connection needs 'uri'")
            if self.path is not None:
                raise ValueError(f"a {self.type} connection takes 'uri', not 'path'")
        return self

    def attach_sql(self, name: str, env: SourceEnvironment) -> str:
        """The DuckDB ``ATTACH`` statement for this connection."""
        from qavf.sources.sql import quote_identifier, sql_literal

        # The target is chosen by type, never by which field happens to be set.
        target = str(env.resolve(str(self.path))) if self.file_based else str(self.uri)
        options = [] if self.type == "duckdb" else [f"TYPE {self.type}"]
        if self.read_only:
            options.append("READ_ONLY")
        suffix = f" ({', '.join(options)})" if options else ""
        return f"ATTACH {sql_literal(target)} AS {quote_identifier(name)}{suffix}"


class SecretSpec(_Strict):
    """Cloud storage credentials for remote files (``s3://``, ``az://``) and SQL reads.

    Each secret becomes a DuckDB secret on every QAVF-owned connection and supplies
    the matching ``storage_options`` for remote Parquet read through Polars. Use
    ``${ENV}`` for credential values; reports and logs mask them.
    """

    type: Literal["s3", "azure"]
    """``s3`` for AWS S3 and S3-compatible stores (Cloudflare R2, MinIO, Wasabi, GCS
    interoperability); ``azure`` for Azure Blob Storage and ADLS Gen2."""
    provider: Literal["config", "credential_chain"] = "config"
    """``config`` uses the values given here; ``credential_chain`` uses the platform's
    own credentials (environment, profile, instance or workload identity, CLI login)."""
    scope: list[str] = Field(default_factory=list)
    """URL prefixes this secret applies to (``s3://bucket/path``). Unscoped secrets
    apply to every URL of their type; at most one per type may be unscoped."""
    key_id: str | None = None
    """S3 access key id (``config`` provider)."""
    secret: str | None = None
    """S3 secret access key (``config`` provider)."""
    session_token: str | None = None
    """Optional S3 session token for temporary credentials."""
    region: str | None = None
    """S3 region, e.g. ``us-east-1`` (``auto`` for Cloudflare R2)."""
    endpoint: str | None = None
    """S3-compatible endpoint as ``host[:port]`` without a scheme, e.g.
    ``<account>.r2.cloudflarestorage.com`` or ``storage.googleapis.com``."""
    url_style: Literal["vhost", "path"] | None = None
    """S3 addressing; most S3-compatible servers need ``path``."""
    use_ssl: bool = True
    """Use HTTPS for a custom S3 endpoint."""
    connection_string: str | None = None
    """Azure storage connection string (``config`` provider)."""
    account_name: str | None = None
    """Azure storage account name (``credential_chain`` provider)."""
    chain: str | None = None
    """Credential sources to try for ``credential_chain``, e.g. ``env;config`` (S3) or
    ``cli;managed_identity`` (Azure); omitted means the platform default order."""

    @model_validator(mode="after")
    def _complete(self) -> SecretSpec:
        s3_only = ("key_id", "secret", "session_token", "region", "endpoint", "url_style")
        azure_only = ("connection_string", "account_name")
        # Judge by values, not by which fields were set: serialized settings (for
        # example an isolated-worker request) state every field, defaults included.
        foreign = sorted(
            name
            for name in (azure_only if self.type == "s3" else s3_only)
            if getattr(self, name) is not None
        )
        if self.type == "azure" and not self.use_ssl:
            foreign.append("use_ssl")
        if foreign:
            raise ValueError(f"a {self.type} secret does not take {', '.join(foreign)}")
        chained = self.provider == "credential_chain"
        if self.chain is not None and not chained:
            raise ValueError("'chain' applies only to provider 'credential_chain'")
        if self.type == "s3":
            if chained and (self.key_id or self.secret or self.session_token):
                raise ValueError("credential_chain secrets take no key_id, secret or session_token")
            if not chained and not (self.key_id and self.secret):
                raise ValueError("an s3 secret with provider 'config' needs key_id and secret")
            if self.endpoint is not None and ("://" in self.endpoint or "/" in self.endpoint):
                raise ValueError("endpoint is host[:port] without a scheme or path")
        elif chained:
            if self.connection_string:
                raise ValueError(
                    "credential_chain secrets take account_name, not connection_string"
                )
            if not self.account_name:
                raise ValueError("an azure credential_chain secret needs account_name")
        elif not self.connection_string:
            raise ValueError("an azure secret with provider 'config' needs connection_string")
        for prefix in self.scope:
            if "://" not in prefix:
                raise ValueError(
                    f"secret scope {prefix!r} must be a URL prefix such as s3://bucket"
                )
        return self


# --------------------------------------------------------------------- engine
def expression_check(item: CheckItem) -> ExpressionCheck:
    """The row expression of an ``expression`` check item.

    A ``where`` condition limits the rows assessed: rows it does not select pass.
    """
    settings = item.settings
    expr = settings.get("expr")
    if item.where is not None and isinstance(expr, str):
        expr = f"CASE WHEN coalesce(({item.where}), false) THEN ({expr}) ELSE true END"
    return ExpressionCheck(
        expr=expr,
        name=item.name,
        severity=item.severity,
        description=item.description,
        null_as=settings.get("null_as", "pass"),
    )


def sql_assertion(item: CheckItem) -> SqlAssertion:
    """The query of a top-level ``sql`` check item."""
    return SqlAssertion(
        name=item.name or "",
        sql=item.settings.get("sql"),
        severity=item.severity,
        description=item.description,
    )


class EngineSettings(_Strict):
    """Execution behaviour."""

    execution_mode: Literal["in_process", "isolated"] = "in_process"
    """Use a fresh supervised process (Linux, macOS, Windows) for transportable inputs."""
    run_timeout_s: float | None = Field(default=None, gt=0, strict=True)
    """Whole isolated worker wall time, including imports, discovery and serialization."""
    process_address_space_bytes: int | None = Field(default=None, ge=1, strict=True)
    """OS virtual address-space ceiling per isolated process; not a resident-memory cap."""
    process_cpu_time_s: int | None = Field(default=None, ge=1, strict=True)
    """OS CPU seconds per isolated process, summed over its threads."""
    process_file_bytes: int | None = Field(default=None, ge=0, strict=True)
    """OS maximum size of any individual file written by the isolated worker."""
    parallel: bool = True
    max_workers: int | None = Field(default=None, ge=1)
    """Thread count; defaults to ``min(8, cpu_count)``."""
    max_affected_records: int = Field(default=50, ge=0)
    """Cap on the offending-record sample kept per result."""
    rule_timeout_s: float | None = Field(default=None, gt=0)
    """Cooperative deadline: overdue work is unassessed and cancellation is requested.
    Arbitrary Python/native plugin calls cannot be forcibly terminated by a thread;
    overdue calls remain part of the active-worker budget until they stop."""
    duckdb_memory_limit: str | None = None
    """Buffer-manager limit per DuckDB connection, not a process RAM limit. Some
    operators allocate outside it or cannot spill; resource exhaustion is an error."""
    memory_budget_bytes: int | None = Field(default=None, ge=1, strict=True)
    """Run target for tracked materializations and cache state, not an OS-enforced
    process RSS cap. Caller-owned inputs and opaque plugin allocations may exceed it."""
    cache_max_bytes: int = Field(default=268_435_456, ge=0, strict=True)
    """Maximum tracked retained cache bytes; zero disables optional caching."""
    max_spill_files: int = Field(default=4096, ge=1, strict=True)
    """Maximum live owned IPC files, including leases and failed cleanup."""
    spill_max_bytes: int = Field(default=1_073_741_824, ge=0, strict=True)
    """Run temporary-storage budget. Zero disables spilling; insufficient storage
    makes affected checks unassessed rather than reducing validation coverage."""
    temp_directory: Path | None = None
    """Parent of the run-owned temporary directory; relative to the configuration."""
    max_report_bytes: int = Field(default=16_777_216, ge=1, strict=True)
    """Maximum encoded report size; essential outcomes exceeding it raise a resource error."""
    max_evidence_bytes: int = Field(default=1_048_576, ge=0, strict=True)
    """Maximum serialized evidence/diagnostic payload bytes per result. Truncation
    is disclosed and never changes exact violation counts or assessment status."""
    native_threads: int | None = Field(default=None, ge=1, strict=True)
    """Native execution thread allowance used where the backend supports it.
    Polars process-global settings must be configured before its pool initializes."""
    timeout_grace_s: float = Field(default=0.1, ge=0)
    """Bounded grace after cancellation; this is not a hard process-kill deadline."""
    concurrent_loads: bool = True
    """Load independent datasets concurrently when rules run in parallel. Turn off
    when a backend cannot take several queries at once or loads must not overlap."""
    canonical_columns: list[str] | str = Field(default_factory=list)
    """Column spellings to normalise to case-insensitively on load (``sourcetypeid``
    becomes ``sourceTypeID``). Either a list or an import reference such as
    ``my_plugin:COLUMNS`` to a list defined by a plugin."""

    @model_validator(mode="after")
    def _distinct_canonical(self) -> EngineSettings:
        for name in ("rule_timeout_s", "timeout_grace_s", "run_timeout_s"):
            _require_finite(name, getattr(self, name))
        if any(
            value is not None
            for value in (
                self.run_timeout_s,
                self.process_address_space_bytes,
                self.process_cpu_time_s,
                self.process_file_bytes,
            )
        ):
            if "execution_mode" in self.model_fields_set and self.execution_mode != "isolated":
                raise ValueError("process execution limits require execution_mode='isolated'")
            self.execution_mode = "isolated"
        if isinstance(self.canonical_columns, list):
            check_canonical_columns(self.canonical_columns)
        return self

    def resolved_canonical_columns(self) -> list[str]:
        """:attr:`canonical_columns` with an import reference resolved."""
        if isinstance(self.canonical_columns, list):
            return self.canonical_columns
        module, _, attribute = self.canonical_columns.partition(":")
        if not attribute:
            raise ValueError("canonical_columns must be a list or 'module:attribute'")
        return list(getattr(importlib.import_module(module), attribute))


class PluginSettings(_Strict):
    """Where to discover rules and sources."""

    packages: list[str] = Field(default_factory=lambda: ["qavf.rules"])
    """Importable packages whose modules are scanned for rules."""
    modules: list[str] = Field(default_factory=list)
    """Additional importable modules scanned for rules and sources."""
    paths: list[Path] = Field(default_factory=list)
    """Stand-alone ``.py`` files (or directories of them) scanned for rules."""
    entry_points: bool = True
    """Also load plugins advertised under the ``qavf.rules`` / ``qavf.sources`` groups.
    The same applies to comparison components (``qavf.compare.*`` groups)."""
    required: list[str] = Field(default_factory=list)
    """Required entry-point names, or qualified names such as ``rules:moves`` or
    ``compare.levels:spatial`` (also ``compare.measures``, ``compare.comparators``,
    ``compare.alignment``).
    Every advertised component of an unqualified name must load; at least one
    must exist. Missing or broken components stop the run with a configuration error."""


class ReportSettings(_Strict):
    """Report outputs."""

    title: str = "QA Report"
    output_dir: Path = Path("qa_output")
    html: bool = True
    html_max_cells: int = Field(default=10_000, ge=1, strict=True)
    html_max_columns: int = Field(default=64, ge=1, strict=True)
    html_max_bytes: int = Field(default=16_777_216, ge=1, strict=True)
    json_: bool = Field(default=True, alias="json")
    hash_inputs: bool = False
    """Record a SHA-256 digest of each local input file in the report's provenance
    (size and modification time are always recorded). Whole-file content checks
    surround each complete read operation and run again at finalization. Nested
    helpers share their operation's checks; separate operations always rehash.
    This verifies boundary consistency, not an immutable snapshot. Per-source
    provenance notes disclose actual hash passes and bytes read."""


class QAVFSettings(_Strict):
    """Root configuration object."""

    project: str = "QA run"
    vars: dict[str, Any] = Field(default_factory=dict)
    """Named values, used as ``${vars.<name>}`` in any value."""
    include: list[str] = Field(default_factory=list)
    """Other configuration files merged first, in order (paths relative to this file).
    This file wins on conflicting values; a dataset or named check defined twice is
    an error."""
    defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Default settings per check name, e.g. ``change_rate: {max: 0.15}``; an item's
    own settings win."""
    datasets: dict[str, DatasetSpec] = Field(default_factory=dict)
    connections: dict[str, ConnectionSpec] = Field(default_factory=dict)
    secrets: dict[str, SecretSpec] = Field(default_factory=dict)
    """Cloud storage credentials, keyed by a SQL-identifier name."""
    checks: list[CheckItem] = Field(default_factory=list)
    """Checks relating several datasets: ``sql``, ``compare``, ``totals_match``,
    ``key_sets_match``, ``batch_consistent``, ``coverage`` and ``drift``."""
    run: RunSettings = Field(default_factory=RunSettings)
    """Pass/fail policy: ``fail_on``, ``fail_on_incomplete`` and ``require``."""
    engine: EngineSettings = Field(default_factory=EngineSettings)
    plugins: PluginSettings = Field(default_factory=PluginSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Free-form settings for plugins, keyed by plugin name (e.g. ``moves: {version: MOVES4}``)."""

    source_path: Path | None = Field(default=None, exclude=True)
    """File this configuration was loaded from; set by :func:`load_settings`."""

    _selection: AppliedSelection | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _string_mapping_keys(cls, data: Any) -> Any:
        validate_config_mapping_keys(data)
        reject_legacy(data)
        if not isinstance(data, dict):
            return data
        from qavf.checks.across import AcrossExpansion
        from qavf.checks.reuse import ItemPreparation
        from qavf.composition import VariableSubstitution

        if data.get("include"):
            raise ValueError(
                "include is resolved when a configuration file is read (load_settings, "
                "settings_from_text or a configuration path); merge mappings yourself"
            )
        return ItemPreparation.apply(AcrossExpansion.apply(VariableSubstitution.apply(data)))

    @property
    def assertions(self) -> list[SqlAssertion]:
        """The enabled top-level ``sql`` checks, as the queries INP-007 evaluates."""
        return [sql_assertion(item) for item in self.checks if item.check == "sql" and item.enabled]

    @model_validator(mode="after")
    def _references_resolve(self) -> QAVFSettings:
        check_sql_names(self.datasets, "dataset")
        check_sql_names(self.connections, "connection")
        check_sql_names(self.secrets, "secret")
        unscoped: dict[str, str] = {}
        scoped: dict[tuple[str, str], str] = {}
        for name, secret in self.secrets.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ValueError(f"secret name {name!r} must be a plain SQL identifier")
            if not secret.scope:
                other = unscoped.setdefault(secret.type, name)
                if other != name:
                    raise ValueError(
                        f"secrets {other!r} and {name!r} are both unscoped {secret.type} "
                        "secrets; give at least one a scope so each URL has one credential"
                    )
            for prefix in secret.scope:
                other = scoped.setdefault((secret.type, prefix), name)
                if other != name:
                    raise ValueError(
                        f"secrets {other!r} and {name!r} have the same {secret.type} scope; "
                        "each URL prefix must select one credential"
                    )
        for name, spec in self.datasets.items():
            for fk in spec.foreign_keys:
                if fk.ref_dataset not in self.datasets:
                    raise ValueError(
                        f"dataset {name!r}: foreign key references unknown dataset "
                        f"{fk.ref_dataset!r}"
                    )
            source = spec.source or {}
            connection = source.get("connection")
            if isinstance(connection, str) and connection not in self.connections:
                raise ValueError(f"dataset {name!r}: unknown connection {connection!r}")
            attached = source.get("connections")
            if isinstance(attached, list):
                if not all(isinstance(item, str) for item in attached):
                    raise ValueError(f"dataset {name!r}: source connections must be names")
                if len(attached) != len(set(attached)):
                    raise ValueError(f"dataset {name!r}: duplicate source connections")
                unknown = set(attached) - self.connections.keys()
                if unknown:
                    raise ValueError(f"dataset {name!r}: unknown connections {sorted(unknown)!r}")
        if (
            self.checks
            or self.run.require
            or self.defaults
            or any(spec.checks for spec in self.datasets.values())
        ):
            problems = _check_errors(self)
            if problems:
                raise ValueError("invalid checks: " + "; ".join(problems))
        return self

    @property
    def selection(self) -> AppliedSelection | None:
        """The run-time selection this configuration was narrowed by, if any
        (see :func:`qavf.checks.selection.narrow`)."""
        return self._selection

    def apply_selection(self, selection: AppliedSelection | None) -> None:
        """Record the run-time selection this configuration was narrowed by."""
        self._selection = selection

    def views(self) -> dict[str, tuple[str, ViewSpec]]:
        """Every dataset view as ``<dataset>.<view>`` -> (dataset, definition)."""
        return {
            f"{name}.{view}": (name, spec)
            for name, dataset in self.datasets.items()
            for view, spec in dataset.views.items()
        }

    def knows_dataset(self, name: str) -> bool:
        """Whether ``name`` is a dataset or a view (``counts.recent``) of one."""
        return name in self.datasets or name in self.views()

    @property
    def base_dir(self) -> Path:
        """Directory that relative paths in the configuration resolve against."""
        return self.source_path.parent if self.source_path else Path.cwd()

    def resolve(self, path: str | Path) -> Path:
        """Resolve ``path`` against :attr:`base_dir` unless it is already absolute."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()


def _check_errors(settings: QAVFSettings) -> list[str]:
    """Problems with the check items of ``settings``, each with its path.

    Built-in checks are validated against their rules' parameters. The catalog
    imports the rule implementations (which import this module, and need the
    optional Polars backend), so it is imported here. Without Polars only the
    ``expression`` and ``sql`` checks can run (natively) and only they are
    validated now. Plugin checks are validated when the run is planned, once
    plugins are loaded (:func:`qavf.engine.compilation.compile_rules`).
    """
    try:
        from qavf.checks.compiler import configuration_errors
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("polars"):
            raise
        from qavf.checks.placement import row_check_errors as configuration_errors
    return configuration_errors(settings)


def validate_config_mapping_keys(value: Any) -> None:
    """Reject key coercions that could give distinct runtime settings one digest.

    Traverse raw mappings and existing models without serialization: serialization
    may already have coerced a key. Locations use entry indexes below known model
    fields so diagnostics never reveal arbitrary plugin keys or their values.
    """
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if not isinstance(item, (BaseModel, Mapping, list, tuple)) or id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, BaseModel):
            for name in type(item).model_fields:
                visit(getattr(item, name), f"{path}.{name}")
        elif isinstance(item, Mapping):
            for index, (key, child) in enumerate(item.items()):
                if not isinstance(key, str):
                    raise ValueError(
                        f"configuration mapping keys must be strings at {path} (key entry {index})"
                    )
                child_path = (
                    f"{path}.{key}"
                    if path == "configuration" and key in QAVFSettings.model_fields
                    else f"{path}[value {index}]"
                )
                visit(child, child_path)
        else:
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "configuration")


def check_canonical_columns(names: Iterable[str]) -> dict[str, str]:
    """Map each lower-cased canonical spelling to the spelling itself.

    Raises:
        ValueError: if two spellings differ only in case, which would make the
            target of a case-insensitive match ambiguous.
    """
    canonical: dict[str, str] = {}
    for name in names:
        other = canonical.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(
                f"canonical_columns lists both {other!r} and {name!r}, which differ only in "
                f"case; keep one spelling"
            )
    return canonical


def check_sql_names(names: Iterable[str], kind: str) -> None:
    """Reject names that SQL would treat as one identifier (DuckDB is case-insensitive).

    Raises:
        ValueError: naming the colliding names.
    """
    seen: dict[str, str] = {}
    for name in names:
        other = seen.setdefault(name.lower(), name)
        if other != name:
            raise ValueError(
                f"{kind} names {other!r} and {name!r} differ only in case; SQL treats them as "
                f"the same table, so one would silently replace the other. Rename one."
            )


def load_settings(path: str | Path) -> QAVFSettings:
    """Load and validate a YAML (``.yaml``/``.yml``) or JSON configuration file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: for an unknown suffix, a non-mapping document or an unset
            environment variable.
        pydantic.ValidationError: if the content violates the schema.
    """
    raw, source = _read_config_mapping(path)
    settings = _validated(raw, source)
    logger.info("Loaded configuration from %s", source)
    return settings


def settings_from_text(text: str, path: str | Path) -> QAVFSettings:
    """Validate configuration text as if it were the content of the file at ``path``.

    For editors: unsaved text is checked without writing it. ``path`` decides the
    format by its suffix and anchors relative paths; it need not exist. Raises
    the same errors as :func:`load_settings`.
    """
    source = Path(path).resolve()
    return _validated(_resolve_config_text(text, source), source)


def _validated(raw: dict[str, Any], source: Path) -> QAVFSettings:
    reject_legacy(raw, str(source))
    ConfigurationGraph().check(raw)
    settings = QAVFSettings.model_validate(raw)
    settings.source_path = source
    return settings
