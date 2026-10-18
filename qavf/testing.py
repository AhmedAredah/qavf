"""Helpers for testing rules, configurations and plugins with pytest.

::

    from qavf.testing import run_rule

    def test_my_rule_flags_negative_values():
        frame = pl.DataFrame({"x": [1.0, -2.0]})
        result = run_rule("CUSTOM-001", {"data": frame}, params={"limit": 0})
        assert result.status is Status.FAIL
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

from qavf.engine.registry import Rule, RuleRegistry, default_registry
from qavf.engine.runner import QARunner
from qavf.reports.models import QAReport, RuleResult
from qavf.settings import QAVFSettings


@cache
def _scanned_rules(entry_points: bool) -> tuple[type[Rule], ...]:
    """Rules found by scanning packages (and entry points): the slow part, cached."""
    return tuple(RuleRegistry().discover(use_entry_points=entry_points))


def discovered_rules(entry_points: bool = True) -> RuleRegistry:
    """Built-in rules (plus installed plugins when ``entry_points``) and every rule
    registered with ``@register`` so far.

    Package scanning is cached; each call returns a new registry, so rules
    registered later are included and changes to a returned registry never leak
    into other tests. :func:`clear_discovery_cache` forgets the scan (for tests that
    install or remove plugins).
    """
    registry = RuleRegistry()
    for rule in (*_scanned_rules(entry_points), *default_registry):
        registry.register(rule)
    return registry


def clear_discovery_cache() -> None:
    """Rescan packages and entry points on the next :func:`discovered_rules` call."""
    _scanned_rules.cache_clear()


def make_settings(
    rules: Mapping[str, Any] | None = None,
    datasets: Mapping[str, Any] | None = None,
    extensions: Mapping[str, Any] | None = None,
    assertions: list[Mapping[str, Any]] | None = None,
    connections: Mapping[str, Any] | None = None,
    **engine: Any,
) -> QAVFSettings:
    """Settings for tests: sequential execution unless ``parallel=True`` is passed."""
    return QAVFSettings.model_validate(
        {
            "datasets": dict(datasets or {}),
            "rules": dict(rules or {}),
            "extensions": dict(extensions or {}),
            "assertions": list(assertions or []),
            "connections": dict(connections or {}),
            "engine": {"parallel": False, **engine},
        }
    )


def run_rules(
    frames: Mapping[str, Any],
    rules: Mapping[str, Any] | None = None,
    registry: RuleRegistry | None = None,
    **settings: Any,
) -> QAReport:
    """Run every discovered rule (configured by ``rules``) against in-memory frames."""
    return QARunner(
        make_settings(rules=rules, **settings),
        # An explicitly empty registry is a registry, not "use the default".
        registry=registry if registry is not None else discovered_rules(),
    ).run(frames=frames)


def rule_results(
    rule_id: str,
    frames: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    registry: RuleRegistry | None = None,
    **settings: Any,
) -> list[RuleResult]:
    """Run exactly one rule and return all its results (one per reported check)."""
    report = QARunner(
        make_settings(
            rules={rule_id: {"params": dict(params or {})}},
            run_unconfigured_rules=False,
            **settings,
        ),
        registry=registry if registry is not None else discovered_rules(),
    ).run(frames=frames)
    return report.results_for(rule_id)


def run_rule(
    rule_id: str,
    frames: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    registry: RuleRegistry | None = None,
    **settings: Any,
) -> RuleResult:
    """Run exactly one rule; return its most severe failure, or its first result."""
    results = rule_results(rule_id, frames, params, registry, **settings)
    failing = [r for r in results if r.failed]
    return max(failing, key=lambda r: r.severity.rank) if failing else results[0]
