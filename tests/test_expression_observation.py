"""One observed row population supplies expression counts and bounded evidence."""

import polars as pl
import pytest

from qavf import DataContext
from qavf.engine import connections
from qavf.rules.input_checks import ExpressionRule
from qavf.settings import DatasetSpec
from qavf.sources.base import SourceEnvironment
from qavf.sources.parameters import QueryParams
from qavf.sources.query import QuerySource


def spec(*expressions, **values):
    return DatasetSpec.model_validate(
        {"expressions": [{"expr": value} for value in expressions], **values}
    )


@pytest.mark.parametrize("backend", ["lazy", "native"])
@pytest.mark.parametrize("limit", [0, 1, 5])
def test_volatile_expression_counts_and_sample_share_one_population(backend, limit, monkeypatch):
    calls = []
    if backend == "lazy":

        def changing(_):
            calls.append(1)
            return pl.DataFrame({"x": [-1] * (1 if len(calls) == 1 else 3)})

        plan = (
            pl.DataFrame({"x": [0]})
            .lazy()
            .map_batches(
                changing,
                projection_pushdown=False,
                predicate_pushdown=False,
                slice_pushdown=False,
            )
        )
        inputs = {"frames": {"d": plan}}
    else:
        source = QuerySource(
            QueryParams(sql="SELECT -1 AS x FROM range(3) t(i) WHERE audit_changing(i)"),
            SourceEnvironment(),
        )
        inputs = {"sources": {"d": source}}

        def changing(i):
            if i == 0:
                calls.append(0)
            calls[-1] += 1
            return i < (1 if len(calls) == 1 else 3)

    if backend == "native":
        connect = connections.connect

        def with_function(*args, **kwargs):
            con = connect(*args, **kwargs)
            con.create_function(
                "audit_changing", changing, ["BIGINT"], "BOOLEAN", side_effects=True
            )
            return con

        monkeypatch.setattr(connections, "connect", with_function)
    with DataContext(**inputs, specs={"d": spec("x > 0")}, evidence_limit=limit) as ctx:
        [failure] = ExpressionRule().evaluate(ctx)
        assert failure.assessed and not failure.passed
        assert failure.total_records == failure.affected_count == 1
        assert failure.affected.height == min(limit, 1)
        assert len(calls) == 1


@pytest.mark.parametrize("name", ["d", "__qavf_expression_failures", 'quoted"name'])
@pytest.mark.parametrize("limit", [0, 2])
def test_expression_self_subqueries_names_and_missing_normalization(name, limit):
    table = '"' + name.replace('"', '""') + '"'
    frame = pl.DataFrame({"x": [-999, -1, 2]})
    schema = {"columns": {"x": {"missing_values": [-999]}}}
    with DataContext(
        frames={name: frame},
        specs={name: spec(f"x >= (SELECT max(x) FROM {table})", schema=schema)},
        evidence_limit=limit,
    ) as ctx:
        [failure] = ExpressionRule().evaluate(ctx)
    assert failure.assessed and not failure.passed
    assert failure.total_records == 3 and failure.affected_count == 1
    assert failure.affected.height == min(limit, 1)
    if limit:
        assert failure.affected["x"].to_list() == [-1]


@pytest.mark.parametrize(
    "frame", [pl.DataFrame({"x": []}, schema={"x": pl.Int64}), pl.DataFrame({"x": [1, 2]})]
)
def test_expression_empty_and_all_passing_inputs(frame):
    with DataContext(frames={"d": frame}, specs={"d": spec("x > 0")}) as ctx:
        [finding] = ExpressionRule().evaluate(ctx)
    assert finding.assessed and finding.passed
    assert finding.metrics["passed_checks"] == ["d: x > 0"]


def test_expression_error_preserves_sibling_expression_and_other_input_failure():
    def broken(_):
        raise RuntimeError("broken input")

    with DataContext(
        frames={
            "broken": pl.DataFrame({"x": [0]}).lazy().map_batches(broken),
            "good": pl.DataFrame({"x": [-1, 2]}),
        },
        specs={"broken": spec("x > 0"), "good": spec("invalid syntax ???", "x > 0")},
    ) as ctx:
        findings = ExpressionRule().evaluate(ctx)
    assert [(f.assessed, f.passed) for f in findings] == [
        (False, False),
        (False, False),
        (True, False),
    ]
    assert findings[-1].total_records == 2 and findings[-1].affected_count == 1


@pytest.mark.parametrize("backend", ["lazy", "native"])
def test_expression_retains_complete_source_row_budget_checks(backend):
    if backend == "lazy":
        inputs = {"frames": {"d": pl.DataFrame({"x": [-1, -2]}).lazy()}}
    else:
        inputs = {
            "sources": {
                "d": QuerySource(
                    QueryParams(sql="SELECT -i AS x FROM range(2) t(i)"), SourceEnvironment()
                )
            }
        }
    with DataContext(
        **inputs, specs={"d": spec("x > 0", load_policy={"row_budget": 1})}, evidence_limit=0
    ) as ctx:
        [finding] = ExpressionRule().evaluate(ctx)
    assert not finding.assessed and not finding.passed
    assert "row_budget" in finding.message


def test_expression_cache_spill_budget_failure_is_unassessed():
    with DataContext(
        frames={"d": pl.DataFrame({"x": [-1, -2]}).lazy()},
        specs={"d": spec("x > 0")},
        cache_max_bytes=0,
        spill_max_bytes=0,
    ) as ctx:
        [finding] = ExpressionRule().evaluate(ctx)
    assert not finding.assessed and not finding.passed
    assert "spill" in finding.message.lower()


@pytest.mark.parametrize("cache_bytes", [0, 1_000_000])
def test_contradictory_native_expressions_cannot_both_pass(cache_bytes, monkeypatch):
    calls = []

    def changing():
        calls.append(1)
        return 1 if len(calls) == 1 else -1

    connect = connections.connect

    def with_function(*args, **kwargs):
        con = connect(*args, **kwargs)
        con.create_function("audit_value", changing, [], "BIGINT", side_effects=True)
        return con

    monkeypatch.setattr(connections, "connect", with_function)
    source = QuerySource(QueryParams(sql="SELECT audit_value() AS x"), SourceEnvironment())
    with DataContext(
        sources={"d": source},
        frames={"other": pl.DataFrame({"lower": [0]})},
        specs={"d": spec("x > (SELECT lower FROM other) AND x = (SELECT x FROM main.d)", "x < 0")},
        cache_max_bytes=cache_bytes,
    ) as ctx:
        retained_before = ctx._resources.retained_budget.used
        findings = ExpressionRule().evaluate(ctx)
        [failure] = [f for f in findings if f.assessed and not f.passed]
        assert failure.check == "d: x < 0"
        assert failure.affected_count == failure.total_records == 1
        assert failure.affected["x"].to_list() == [1]
        assert not any(not f.assessed for f in findings)
        assert len(calls) == 1
        assert ctx._resources.retained_budget.used == retained_before
        tables = ctx._connection(()).execute("SHOW TABLES").fetchall()
        assert tables == []
