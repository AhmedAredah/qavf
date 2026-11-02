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
    "min": ("Column statistics", "Minimum value bounds", False),
    "max": ("Column statistics", "Maximum value bounds", False),
    "mean": ("Column statistics", "Mean bounds", False),
    "median": ("Column statistics", "Median bounds", False),
    "std": ("Column statistics", "Standard deviation bounds", False),
    "sum": ("Column statistics", "Sum bounds", False),
    "distinct_count": ("Column statistics", "Distinct value count bounds", False),
    "description": ("General", "Description", False),
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
    "checks": ("Conditional checks", "Conditional checks", False),
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
    "expressions": ("Row rules", "Row expressions", False),
}

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
    "ExpressionCheck": {
        "expr": ("Row expression", "Expression", False),
        "name": ("Row expression", "Name", False),
        "severity": ("Row expression", "Severity", False),
        "description": ("Row expression", "Description", False),
        "null_as": ("Row expression", "A null result counts as", True),
    },
    "SqlAssertion": {
        "name": ("SQL assertion", "Name", False),
        "sql": ("SQL assertion", "Query returning offending rows", False),
        "severity": ("SQL assertion", "Severity", False),
        "description": ("SQL assertion", "Description", False),
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
        "fail_on": ("Pass / fail", "Fail on severity", False),
        "fail_on_incomplete": ("Pass / fail", "Fail when checks cannot run", False),
        "require": ("Pass / fail", "Checks that must be assessed", True),
        "run_unconfigured_rules": ("Rules", "Run rules that are not configured", False),
        "include_categories": ("Rules", "Only these categories", False),
        "exclude_categories": ("Rules", "Skip these categories", False),
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


def apply_ui_hints(schema: dict[str, Any]) -> dict[str, Any]:
    """Add the hints to ``schema``'s definitions in place, and return it."""
    definitions = schema.get("$defs", {})
    for name, hints in HINTS.items():
        properties = definitions.get(name, {}).get("properties")
        if not isinstance(properties, Mapping):
            continue
        for field, (group, title, advanced) in hints.items():
            target = properties.get(field)
            if isinstance(target, dict):
                target.update(ui_group=group, ui_title=title, ui_advanced=advanced)
    return schema


__all__ = ["HINTS", "apply_ui_hints"]
