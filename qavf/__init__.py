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
    from qavf.engine.compilation import ConfigurationError, build_registry
    from qavf.engine.context import DataContext
    from qavf.engine.planning import DataRequirements
    from qavf.engine.registry import Finding, Rule, RuleParams, RuleRegistry, register
    from qavf.engine.runner import QARunner
    from qavf.engine.severity import Severity, Status
    from qavf.execution.errors import ExecutionLimitError
    from qavf.inputs.descriptors import Dataset, File, Loader, Query
    from qavf.inputs.resources import ResourceLimits
    from qavf.profiling import ProfileResult, profile_input
    from qavf.regression.compare_outputs import (
        assert_outputs_match,
        compare_frames,
        compare_outputs,
    )
    from qavf.regression.models import ComparisonResult
    from qavf.reports.html import render_html, write_html
    from qavf.reports.models import QAReport, RuleResult
    from qavf.settings import QAVFSettings, load_settings
    from qavf.sources import DataSource, SourceParams

_EXPORTS = {
    "ComparisonResult": "qavf.regression.models",
    "ConfigurationError": "qavf.engine.compilation",
    "DataContext": "qavf.engine.context",
    "DataRequirements": "qavf.engine.planning",
    "DataSource": "qavf.sources",
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
    "Rule": "qavf.engine.registry",
    "RuleParams": "qavf.engine.registry",
    "RuleRegistry": "qavf.engine.registry",
    "RuleResult": "qavf.reports.models",
    "Severity": "qavf.engine.severity",
    "SourceParams": "qavf.sources",
    "Status": "qavf.engine.severity",
    "assert_outputs_match": "qavf.regression.compare_outputs",
    "build_registry": "qavf.engine.compilation",
    "compare_frames": "qavf.regression.compare_outputs",
    "compare_outputs": "qavf.regression.compare_outputs",
    "load_settings": "qavf.settings",
    "profile_input": "qavf.profiling",
    "register": "qavf.engine.registry",
    "render_html": "qavf.reports.html",
    "validate": "qavf.api",
    "write_html": "qavf.reports.html",
}

__all__ = [
    "ComparisonResult",
    "ConfigurationError",
    "DataContext",
    "DataRequirements",
    "DataSource",
    "Dataset",
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
    "Severity",
    "SourceParams",
    "Status",
    "__version__",
    "assert_outputs_match",
    "build_registry",
    "compare_frames",
    "compare_outputs",
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
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
