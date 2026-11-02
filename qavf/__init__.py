"""QAVF: explicit QA contracts for files, frames and queries.

Quick start::

    from qavf import validate

    report = validate("counts.parquet", schema={"columns": {"count": {"ge": 0}}})
    report.write_json("qa_report.json")

Public objects are imported when requested, so importing QAVF's version does
not initialize dataframe, regression or report-rendering dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from qavf._version import __version__

if TYPE_CHECKING:
    from qavf.api import validate
    from qavf.capabilities import BackendCapabilities, Capability, backend_capabilities
    from qavf.compare.api import compare_datasets
    from qavf.compare.results import DatasetComparison
    from qavf.engine.compilation import ConfigurationError, build_registry
    from qavf.engine.context import DataContext
    from qavf.engine.planning import DataRequirements
    from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry, register
    from qavf.engine.runner import QARunner
    from qavf.engine.severity import Severity, Status
    from qavf.execution.errors import ExecutionLimitError
    from qavf.inputs.descriptors import BatchSource, Dataset, File, Loader, Query
    from qavf.inputs.resources import ResourceLimits
    from qavf.planning import ValidationPlan, explain
    from qavf.profiling import ProfileResult, profile_input
    from qavf.regression.compare_outputs import (
        assert_outputs_match,
        compare_frames,
        compare_outputs,
    )
    from qavf.regression.models import ComparisonResult
    from qavf.reports.html import render_html, write_html
    from qavf.reports.models import QAReport, RuleResult
    from qavf.run_control import CancellationToken, RunEvent
    from qavf.settings import QAVFSettings, load_settings
    from qavf.sources import DataSource, SourceParams

_EXPORTS = {
    "BackendCapabilities": "qavf.capabilities",
    "Capability": "qavf.capabilities",
    "backend_capabilities": "qavf.capabilities",
    "BatchSource": "qavf.inputs.descriptors",
    "CancellationToken": "qavf.run_control",
    "CheckIdentity": "qavf.reports.models",
    "ComparisonResult": "qavf.regression.models",
    "ConfigurationError": "qavf.engine.compilation",
    "DataContext": "qavf.engine.context",
    "DataRequirements": "qavf.engine.planning",
    "DataSource": "qavf.sources",
    "DatasetComparison": "qavf.compare.results",
    "Dataset": "qavf.inputs.descriptors",
    "ExecutionLimitError": "qavf.execution.errors",
    "File": "qavf.inputs.descriptors",
    "Finding": "qavf.engine.registry",
    "Loader": "qavf.inputs.descriptors",
    "ProfileResult": "qavf.profiling",
    "QAReport": "qavf.reports.models",
    "QARunner": "qavf.engine.runner",
    "QAVFSettings": "qavf.settings",
    "Query": "qavf.inputs.descriptors",
    "ResourceLimits": "qavf.inputs.resources",
    "RunEvent": "qavf.run_control",
    "Rule": "qavf.engine.registry",
    "RuleParams": "qavf.engine.registry",
    "RuleRegistry": "qavf.engine.registry",
    "RuleResult": "qavf.reports.models",
    "Severity": "qavf.engine.severity",
    "SourceParams": "qavf.sources",
    "Status": "qavf.engine.severity",
    "ValidationPlan": "qavf.planning",
    "assert_outputs_match": "qavf.regression.compare_outputs",
    "build_registry": "qavf.engine.compilation",
    "compare_datasets": "qavf.compare.api",
    "compare_frames": "qavf.regression.compare_outputs",
    "compare_outputs": "qavf.regression.compare_outputs",
    "explain": "qavf.planning",
    "load_settings": "qavf.settings",
    "profile_input": "qavf.profiling",
    "register": "qavf.engine.registry",
    "render_html": "qavf.reports.html",
    "validate": "qavf.api",
    "write_html": "qavf.reports.html",
}

__all__ = [
    "BackendCapabilities",
    "BatchSource",
    "CancellationToken",
    "Capability",
    "CheckIdentity",
    "ComparisonResult",
    "ConfigurationError",
    "DataContext",
    "DataRequirements",
    "DataSource",
    "Dataset",
    "DatasetComparison",
    "ExecutionLimitError",
    "File",
    "Finding",
    "Loader",
    "ProfileResult",
    "QAReport",
    "QARunner",
    "QAVFSettings",
    "Query",
    "ResourceLimits",
    "Rule",
    "RuleParams",
    "RuleRegistry",
    "RuleResult",
    "RunEvent",
    "Severity",
    "SourceParams",
    "Status",
    "ValidationPlan",
    "__version__",
    "assert_outputs_match",
    "backend_capabilities",
    "build_registry",
    "compare_datasets",
    "compare_frames",
    "compare_outputs",
    "explain",
    "load_settings",
    "profile_input",
    "register",
    "render_html",
    "validate",
    "write_html",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(module), name)
    except ModuleNotFoundError as exc:
        if exc.name == "polars":
            from qavf.optional import require_polars

            require_polars(name)
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
