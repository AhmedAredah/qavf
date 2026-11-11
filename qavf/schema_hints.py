"""Presentation hints for configuration editors, added to the JSON Schema.

Each hinted property gets ``ui_group`` (the form section it belongs to),
``ui_title`` (a plain-language label) and ``ui_advanced`` (hidden until the user
asks for advanced options). Hints never change validation; editors may ignore
them. Imports neither Polars nor Pydantic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

Hint = tuple[str, str, bool]
"""(group, title, advanced)."""

_LIMITS: dict[str, Hint] = {
    "ge": ("Value limits", "Minimum (≥)", False),
    "le": ("Value limits", "Maximum (≤)", False),
    "gt": ("Value limits", "Greater than (>)", False),
    "lt": ("Value limits", "Less than (<)", False),
    "isin": ("Value limits", "Allowed values", False),
    "notin": ("Value limits", "Forbidden values", False),
    "regex": ("Value limits", "Pattern (regular expression)", False),
    "min_length": ("Value limits", "Minimum length", False),
    "max_length": ("Value limits", "Maximum length", False),
    "integral": ("Value limits", "Whole numbers only", False),
    "max_decimals": ("Value limits", "Maximum decimal places", False),
    "unique": ("Value limits", "Values are unique", False),
    "not_null": ("Missing & N/A", "No missing values", True),
    "max_null_fraction": ("Missing & N/A", "Maximum missing fraction", False),
    "normalize": ("Value limits", "Normalize text before comparing", True),
    "each_allowed_once": ("Value limits", "Each allowed value at most once", True),
    "convert_options": ("Value limits", "Convert allowed values to the column type", True),
    "between": ("Value limits", "Between (inclusive)", False),
    "stats": ("Column statistics", "Column statistics", False),
    "description": ("General", "Description", False),
}

_STATS: dict[str, Hint] = {
    "min": ("Column statistics", "Smallest value", False),
    "max": ("Column statistics", "Largest value", False),
    "mean": ("Column statistics", "Mean", False),
    "median": ("Column statistics", "Median", False),
    "std": ("Column statistics", "Standard deviation", False),
    "sum": ("Column statistics", "Sum", False),
    "distinct_count": ("Column statistics", "Distinct values", False),
}

_BOUNDS: dict[str, Hint] = {
    "ge": ("Bounds", "At least (≥)", False),
    "le": ("Bounds", "At most (≤)", False),
    "gt": ("Bounds", "Greater than (>)", True),
    "lt": ("Bounds", "Less than (<)", True),
    "between": ("Bounds", "Between (inclusive)", False),
}

_COLUMN: dict[str, Hint] = {
    **_LIMITS,
    "dtype": ("Type & parsing", "Data type", False),
    "number_grammar": ("Type & parsing", "Number format", True),
    "trim_whitespace": ("Type & parsing", "Trim whitespace", True),
    "decimal_precision": ("Type & parsing", "Decimal precision", True),
    "decimal_scale": ("Type & parsing", "Decimal scale", True),
    "datetime_unit": ("Type & parsing", "Date-time unit", True),
    "timezone": ("Type & parsing", "Time zone", True),
    "ambiguous": ("Type & parsing", "Ambiguous local times", True),
    "nonexistent": ("Type & parsing", "Nonexistent local times", True),
    "format": ("Type & parsing", "Date/time format", False),
    "true_values": ("Type & parsing", "Words meaning true", True),
    "false_values": ("Type & parsing", "Words meaning false", True),
    "required": ("General", "Column must exist", False),
    "nullable": ("Missing & N/A", "Missing values allowed", False),
    "allow_infinite": ("Value limits", "Allow infinity", True),
    "missing_values": ("Missing & N/A", "Values meaning missing", False),
    "na_values": ("Missing & N/A", "Not-applicable tokens", False),
    "blank_as_missing": ("Missing & N/A", "Treat blank text as missing", False),
    "unit": ("Units", "Unit", False),
    "unit_column": ("Units", "Unit column", False),
    "allowed_units": ("Units", "Allowed units", False),
    "conditions": ("Conditions", "Conditional limits", False),
}

_COLUMN_CHECK: dict[str, Hint] = {
    **_LIMITS,
    "name": ("General", "Check name", False),
    "severity": ("General", "Severity", False),
    "where": ("General", "Only rows where", False),
    "min_applicable_rows": ("General", "Minimum applicable rows", True),
}

_DATASET: dict[str, Hint] = {
    "path": ("Source", "File path or glob", False),
    "format": ("Source", "File format", False),
    "source": ("Source", "Source", False),
    "load_policy": ("Source", "Load policy", True),
    "required": ("General", "Dataset must be available", False),
    "description": ("General", "Description", False),
    "rename": ("Source", "Rename columns", True),
    "schema": ("Schema", "Schema", False),
    "foreign_keys": ("Keys & relations", "Foreign keys", False),
    "across": ("Source", "Copies across sources", True),
    "series": ("Reuse", "Named series", False),
    "metrics": ("Reuse", "Named metrics", False),
    "views": ("Reuse", "Named views", True),
    "checks": ("Checks", "Checks", False),
}

_ROOT: dict[str, Hint] = {
    "project": ("Project", "Project name", False),
    "vars": ("Reuse", "Variables", True),
    "include": ("Reuse", "Included files", True),
    "defaults": ("Reuse", "Default check settings", True),
    "overrides": ("Reuse", "Overrides", True),
    "datasets": ("Datasets", "Datasets", False),
    "connections": ("Connections", "Database connections", False),
    "secrets": ("Connections", "Cloud credentials", True),
    "checks": ("Checks", "Checks across datasets", False),
    "run": ("Run", "Pass / fail policy", False),
    "engine": ("Engine", "Execution settings", True),
    "plugins": ("Plugins", "Plugins", True),
    "report": ("Report", "Report outputs", False),
    "extensions": ("Plugins", "Plugin settings", True),
}
"""Top-level sections of a configuration."""

_CHECK: dict[str, Hint] = {
    "check": ("Check", "Check", False),
    "name": ("Check", "Name", False),
    "severity": ("Check", "Severity", False),
    "where": ("Check", "Only rows where", True),
    "tags": ("Check", "Tags", True),
    "enabled": ("Check", "Enabled", False),
    "description": ("Check", "Description", False),
}
"""Fields every check item has; the check's own settings are hinted below or generically."""

_CHECK_SETTINGS: dict[str, Hint] = {
    "min_applicable_rows": ("Check", "Minimum rows matching where", True),
    "datasets": ("Datasets", "Datasets related", False),
    "strategy": ("Datasets", "Pairing strategy", False),
    "reference": ("Datasets", "Reference dataset", False),
    "max_pairs": ("Datasets", "Most pairs (all_pairs)", True),
    "consensus": ("Datasets", "Consensus rules", True),
    "series": ("Settings", "Series", False),
    "metric": ("Settings", "Metric", False),
}
"""Check settings shared by several checks, with a specific title."""

_INLINE: dict[str, Hint] = {
    "value": ("Definition", "Value column", False),
    "group_by": ("Definition", "Group by", False),
    "aggregate": ("Definition", "Aggregate", False),
    "weight": ("Definition", "Weight column", True),
    "order_by": ("Definition", "Order by (first/last)", True),
    "bucket": ("Definition", "Calendar bucket", True),
    "time_zone": ("Definition", "Time zone", True),
    "time_format": ("Definition", "Time format", True),
    "time_column": ("Definition", "Time column", False),
    "expected_trend": ("Definition", "Expected trend (this series)", True),
    "max_change_rate": ("Definition", "Maximum change (this series)", True),
    "entity": ("Definition", "Entity column", False),
    "per": ("Definition", "Per (normalizer)", False),
}
"""Fields of an inline series or metric definition (``Inline_<series|metric>_<check>``)."""

INLINE_PREFIX = "Inline_"

CHECK_PREFIX = "Check_"
"""Definitions of individual checks' items are named ``Check_<check name>``."""

_SOURCE: dict[str, Hint] = {
    "type": ("Source", "Source type", False),
    "retries": ("Load policy", "Retries", True),
    "retry_delay_s": ("Load policy", "Retry delay (s)", True),
    "row_budget": ("Load policy", "Row budget", True),
    "path": ("Source", "File path or glob", False),
    "format": ("Source", "File format", False),
    "options": ("Reader options", "Reader options", True),
    "columns": ("Reader options", "Columns to read", True),
    "text_columns": ("Reader options", "Columns read as text", True),
    "min_files": ("Reader options", "Minimum matching files", True),
    "declared_columns": ("Reader options", "Declared CSV columns", True),
    "max_rejected_rows": ("Reader options", "Maximum rejected rows", True),
    "rejects_budget": ("Reader options", "Rejected-row budget", True),
    "acknowledge_lossy": ("Reader options", "Accept lossy reading", True),
    "connection": ("Source", "Connection", False),
    "table": ("Source", "Table", False),
    "schema": ("Source", "Database schema", False),
    "query": ("Source", "Query", False),
    "sql": ("Source", "SQL query", False),
    "connections": ("Source", "Connections used", False),
    "callable": ("Source", "Python callable", False),
    "kwargs": ("Source", "Arguments", True),
}

_SOURCE_COMMON = ("type", "retries", "retry_delay_s", "row_budget")


def _source(*fields: str) -> dict[str, Hint]:
    """Hints for one source type: the common fields plus its own."""
    return {field: _SOURCE[field] for field in (*_SOURCE_COMMON, *fields)}


HINTS: dict[str, dict[str, Hint]] = {
    "ColumnSpec": _COLUMN,
    "ColumnCheck": _COLUMN_CHECK,
    "ColumnStats": _STATS,
    "Bounds": _BOUNDS,
    "Dataset": _DATASET,
    "NativeDataset": _DATASET,
    "FileParams": _source(
        "path",
        "format",
        "options",
        "columns",
        "text_columns",
        "min_files",
        "declared_columns",
        "max_rejected_rows",
        "rejects_budget",
        "acknowledge_lossy",
    ),
    "DatabaseParams": _source("connection", "table", "schema", "query"),
    "QueryParams": _source("sql", "connections"),
    "PythonParams": _source("callable", "kwargs"),
    "SchemaSpec": {
        "columns": ("Columns", "Columns", False),
        "primary_key": ("Keys & relations", "Primary key", False),
        "unique_keys": ("Keys & relations", "Unique keys", False),
        "unique_rows": ("Keys & relations", "Rows are unique", False),
        "duplicate_rows_per_key": ("Keys & relations", "Rows shown per duplicate key", True),
        "na_groups": ("Keys & relations", "N/A groups", False),
        "grouped_values": ("Keys & relations", "Grouped values", False),
        "strict": ("Table rules", "No extra columns", False),
        "min_rows": ("Table rules", "Minimum rows", False),
        "max_rows": ("Table rules", "Maximum rows", False),
        "evidence_columns": ("Table rules", "Evidence columns", True),
    },
    "ForeignKeySpec": {
        "columns": ("Foreign key", "Columns", False),
        "ref_dataset": ("Foreign key", "Referenced dataset", False),
        "ref_columns": ("Foreign key", "Referenced columns", False),
        "match": ("Foreign key", "Partial-null matching", True),
        "cardinality": ("Foreign key", "Cardinality", False),
        "severity": ("Foreign key", "Severity", False),
    },
    "UniqueKeySpec": {
        "columns": ("Unique key", "Columns", False),
        "nulls": ("Unique key", "Missing key values", False),
        "where": ("Unique key", "Only rows where", True),
        "name": ("Unique key", "Name", True),
    },
    "GroupedValueSpec": {
        "column": ("Grouped values", "Column", False),
        "group_by": ("Grouped values", "Group by", False),
        "allowed_values": ("Grouped values", "Allowed values", False),
        "min_occurrences": ("Grouped values", "Minimum occurrences", False),
        "max_occurrences": ("Grouped values", "Maximum occurrences", False),
        "allow_other_values": ("Grouped values", "Allow other values", True),
        "where": ("Grouped values", "Only rows where", True),
        "expected_groups": ("Grouped values", "Expected groups", True),
        "name": ("Grouped values", "Name", True),
    },
    "ExpressionItem": {
        **_CHECK,
        "expr": ("Settings", "Expression", False),
        "null_as": ("Settings", "A null result counts as", True),
        "min_applicable_rows": ("Settings", "Minimum applicable rows", True),
    },
    "SqlItem": {
        **{key: hint for key, hint in _CHECK.items() if key != "where"},
        "sql": ("Settings", "Query returning offending rows", False),
    },
    "CheckItem": _CHECK,
    "RunSettings": {
        "fail_on": ("Pass / fail", "Fail on severity", False),
        "fail_on_incomplete": ("Pass / fail", "Fail when checks cannot run", False),
        "require": ("Pass / fail", "Checks that must be assessed", True),
        "tags": ("Pass / fail", "Policies per tag", True),
    },
    "TagPolicy": {"fail_on": ("Pass / fail", "Fail on severity (this tag)", False)},
    "ViewSpec": {
        "where": ("View", "Rows where", False),
        "columns": ("View", "Columns", True),
        "description": ("View", "Description", False),
    },
    "_SideSettings": {
        "value": ("Datasets", "Value column (this dataset)", False),
        "keys": ("Datasets", "Key columns (this dataset)", False),
    },
    "ConsensusSettings": {
        "numeric": ("Consensus", "Numbers combine by", False),
        "trim": ("Consensus", "Trimmed share (trimmed_mean)", True),
        "categorical": ("Consensus", "Categories combine by", False),
        "min_agreement": ("Consensus", "Minimum agreeing share", False),
        "ties": ("Consensus", "On a tie", False),
    },
    "ConnectionSpec": {
        "type": ("Connection", "Database type", False),
        "path": ("Connection", "Database file", False),
        "uri": ("Connection", "Connection URI", False),
        "read_only": ("Connection", "Read only", True),
        "snapshot": ("Connection", "Read datasets from one snapshot", True),
    },
    "SecretSpec": {
        "type": ("Cloud access", "Service", False),
        "provider": ("Cloud access", "Credentials from", False),
        "scope": ("Cloud access", "Applies to URLs", False),
        "region": ("Cloud access", "Region", False),
        "endpoint": ("Cloud access", "Endpoint", True),
        "url_style": ("Cloud access", "URL style", True),
        "use_ssl": ("Cloud access", "Use SSL", True),
        "key_id": ("Credentials", "Key ID", False),
        "secret": ("Credentials", "Secret", False),
        "session_token": ("Credentials", "Session token", True),
        "connection_string": ("Credentials", "Connection string", False),
        "account_name": ("Credentials", "Account name", False),
        "chain": ("Credentials", "Credential chain", True),
    },
    "NumberGrammar": {
        "int_decimal": ("Number format", "Allow integers written as decimals", True),
        "underscores": ("Number format", "Allow underscores", True),
        "exponent": ("Number format", "Allow exponents", True),
        "special_values": ("Number format", "Allow NaN and infinity words", True),
        "decimal_separator": ("Number format", "Decimal separator", False),
        "grouping_separator": ("Number format", "Thousands separator", False),
    },
    "SourcePolicy": {
        "retries": ("Load policy", "Retries", False),
        "retry_delay_s": ("Load policy", "Retry delay (s)", False),
        "row_budget": ("Load policy", "Row budget", False),
    },
    "OverridesSpec": {
        "checks": ("Overrides", "Check settings by name or instance id", False),
        "datasets": ("Overrides", "Dataset settings by name", False),
    },
    "ReportSettings": {
        "title": ("Report", "Title", False),
        "output_dir": ("Report", "Output folder", False),
        "html": ("Report", "Write HTML report", False),
        "json": ("Report", "Write JSON report", False),
        "hash_inputs": ("Report", "Record input file hashes", True),
        "html_max_cells": ("Report limits", "HTML table cell limit", True),
        "html_max_columns": ("Report limits", "HTML column limit", True),
        "html_max_bytes": ("Report limits", "HTML size limit", True),
    },
    "PluginSettings": {
        "packages": ("Plugins", "Packages", False),
        "modules": ("Plugins", "Modules", False),
        "paths": ("Plugins", "Paths", True),
        "entry_points": ("Plugins", "Load installed plugins", False),
        "required": ("Plugins", "Required plugins", True),
    },
    "EngineSettings": {
        "parallel": ("Performance", "Run rules in parallel", False),
        "max_workers": ("Performance", "Maximum workers", True),
        "native_threads": ("Performance", "DuckDB threads", True),
        "concurrent_loads": ("Performance", "Load datasets concurrently", True),
        "duckdb_memory_limit": ("Performance", "DuckDB memory limit", True),
        "memory_budget_bytes": ("Performance", "Memory budget (bytes)", True),
        "cache_max_bytes": ("Performance", "Cache size (bytes)", True),
        "spill_max_bytes": ("Performance", "Spill limit (bytes)", True),
        "max_spill_files": ("Performance", "Spill file limit", True),
        "temp_directory": ("Performance", "Temporary folder", True),
        "canonical_columns": ("Performance", "Canonical columns", True),
        "execution_mode": ("Safety", "Execution mode", True),
        "rule_timeout_s": ("Safety", "Rule timeout (s)", False),
        "timeout_grace_s": ("Safety", "Timeout grace period (s)", True),
        "run_timeout_s": ("Safety", "Run timeout (s)", True),
        "process_address_space_bytes": ("Safety", "Process address space (bytes)", True),
        "process_cpu_time_s": ("Safety", "Process CPU time (s)", True),
        "process_file_bytes": ("Safety", "Process file size (bytes)", True),
        "max_affected_records": ("Evidence", "Evidence rows per check", False),
        "max_evidence_bytes": ("Evidence", "Evidence size per check (bytes)", True),
        "max_report_bytes": ("Evidence", "Report size limit (bytes)", True),
    },
}


def _check_hints(properties: Mapping[str, Any]) -> dict[str, Hint]:
    """The common check fields, plus a plain title for each of the check's settings."""
    return {
        field: _CHECK.get(field)
        or _CHECK_SETTINGS.get(field, ("Settings", field.replace("_", " ").capitalize(), False))
        for field in properties
    }


def _apply(properties: Any, hints: Mapping[str, Hint]) -> None:
    if not isinstance(properties, Mapping):
        return
    for field, (group, title, advanced) in hints.items():
        target = properties.get(field)
        if isinstance(target, dict):
            target.update(ui_group=group, ui_title=title, ui_advanced=advanced)


def apply_ui_hints(schema: dict[str, Any]) -> dict[str, Any]:
    """Add the hints to ``schema``'s definitions in place, and return it."""
    _apply(schema.get("properties"), ROOT_HINTS)
    definitions = schema.get("$defs", {})
    for name, hints in HINTS.items():
        _apply(definitions.get(name, {}).get("properties"), hints)
    for name, definition in definitions.items():
        properties = definition.get("properties")
        if name.startswith(CHECK_PREFIX) and isinstance(properties, Mapping):
            _apply(properties, _check_hints(properties))
        elif name.startswith(INLINE_PREFIX):
            _apply(properties, _INLINE)
    return schema


ROOT_HINTS = _ROOT
"""Hints for the configuration's top-level properties."""

__all__ = ["HINTS", "ROOT_HINTS", "apply_ui_hints"]
