"""The DataContext facade keeps every public member; narrow ports are views of it."""

import polars as pl

from qavf.engine.context import DataContext
from qavf.engine.derived_names import RowFilter
from qavf.engine.filtered import FilteredPopulation
from qavf.engine.ports import PORTS
from qavf.engine.row_locator import RowLocator

# Every public member DataContext had before its internal ports (0.7.0). Plugins
# rely on these; removing one is a breaking change.
PUBLIC_MEMBERS = frozenset(
    {
        "assertions",
        "available",
        "begin_rule",
        "build_source",
        "cancel_active_rules",
        "cancel_rule",
        "check_materialization",
        "check_retained_bytes",
        "close",
        "collect",
        "collect_many",
        "consume_sql",
        "contract_spec",
        "declared",
        "describe",
        "end_rule",
        "env",
        "evidence_bytes",
        "evidence_limit",
        "execution_kind",
        "extension",
        "finish_inactive_workers",
        "finish_worker",
        "from_settings",
        "get",
        "has",
        "is_materialized",
        "load_error",
        "load_source",
        "memo",
        "memory_budget_bytes",
        "names",
        "native_read_services",
        "observation_plan",
        "observation_storage",
        "observe",
        "pandas",
        "project",
        "projected_observations",
        "query_plans",
        "read_source",
        "reduce",
        "reference_time",
        "referenced",
        "release",
        "repeatable_scan",
        "reporting_snapshot",
        "resolve_path",
        "resource_usage",
        "run_source_operation",
        "scan",
        "schema",
        "source",
        "source_fingerprint",
        "source_read_services",
        "specs",
        "spill_file",
        "sql",
        "sql_missing_projection",
        "sql_sample",
        "thread_load_seconds",
        "unavailable",
        "unavailable_reason",
        "unused",
        "verify_inputs",
        "work_store",
    }
)


def _instance_members() -> set[str]:
    with DataContext(frames={"d": pl.DataFrame({"x": [1]})}) as ctx:
        return {name for name in dir(ctx) if not name.startswith("_")}


def test_facade_keeps_every_public_member():
    missing = PUBLIC_MEMBERS - _instance_members()
    assert not missing, f"DataContext lost public members: {sorted(missing)}"


def _port_members(port: type) -> set[str]:
    names: set[str] = set()
    for klass in port.__mro__:
        if klass is object or klass.__name__ in {"Protocol", "Generic"}:
            continue
        names.update(name for name in vars(klass) if not name.startswith("_"))
        names.update(
            name for name in getattr(klass, "__annotations__", {}) if not name.startswith("_")
        )
    return names


def test_every_port_member_is_on_the_facade():
    members = _instance_members()
    for port in PORTS:
        missing = _port_members(port) - members
        assert not missing, f"{port.__name__} names {sorted(missing)}, not on DataContext"


class _Reductions:
    """Only DatasetReductions.reduce: a filtered population needs nothing else."""

    def __init__(self, rows: dict[str, int]) -> None:
        self.rows = rows

    def reduce(self, name, build):
        frame = pl.DataFrame({"x": list(range(self.rows[name]))}).lazy()
        return build(frame).collect()


def test_filtered_population_needs_only_reductions():
    rows = RowFilter(dataset="d", where="x > 1")
    population = FilteredPopulation.count(rows, _Reductions({"d": 5, rows.view: 3}))
    assert (population.population, population.applicable) == (5, 3)


class _NoSources:
    def source(self, name):
        return None


def test_row_locator_needs_only_source_lookup():
    from qavf.engine.registry import Finding

    finding = Finding.unassessed("nothing", check="c", datasets=("d",))
    assert RowLocator(_NoSources()).locate(finding) is finding


def test_one_retry_policy_decides_for_both_backends():
    from qavf.sources.base import SourceError, SourceIntegrityError, SourceNotFoundError
    from qavf.sources.execution import RetryPolicy

    policy = RetryPolicy(retries=2, delay_s=0.5, safe=True)
    transient = SourceError("timeout", retryable=True)
    assert policy.pause(transient, 0) == 0.5
    assert policy.pause(transient, 1) == 1.0
    assert policy.pause(transient, 2) is None
    assert policy.pause(SourceError("bad"), 0) is None
    assert policy.pause(SourceNotFoundError("gone"), 0) is None
    assert policy.pause(SourceIntegrityError("changed"), 0) is None
    assert policy.pause(ValueError("unknown"), 0) is None
    assert RetryPolicy(retries=2, delay_s=0.5, safe=False).pause(transient, 0) is None
