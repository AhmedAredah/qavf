"""Related key summaries share work without collecting complete key tables."""

from unittest.mock import patch

import duckdb
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules._execution import KeyRelation
from qavf.rules._key_summary import KeySelection


@pytest.mark.parametrize("key", ["__QAVF_SUMMARY_PADDING", "__QAVF_KEY_ROWS"])
def test_native_private_identifiers_do_not_shadow_case_variants(key):
    frame = pl.DataFrame({key: [0, 1, 1], "rows": [1, 2, 3]})
    with DataContext() as ctx, duckdb.connect() as native:
        native.register("__QAVF_KEY_ROWS", frame.to_arrow())
        relation = KeyRelation(
            ctx,
            ("__QAVF_KEY_ROWS",),
            (key,),
            "rows",
            frame.schema,
            sql='SELECT * FROM "__QAVF_KEY_ROWS"',
        )
        with patch.object(ctx, "sql", side_effect=lambda sql, **_: native.execute(sql).pl()):
            result = relation.summarize(2)
    assert (result.keys, result.rows, result.sample.height) == (3, 6, 2)
    assert result.sample.schema == frame.schema


@pytest.mark.parametrize("backend", ["native", "lazy"])
@pytest.mark.parametrize("values", [[], [None, None], [None, 1, 2, 3]])
@pytest.mark.parametrize("limit", [0, 1, 5])
def test_related_summaries_keep_exact_counts_and_bounded_samples(backend, values, limit):
    # Include names resembling all private summary outputs and the old CTE name.
    key = "__qavf_summary_keys"
    frequency = "__qavf_summary_rows"
    frame = pl.DataFrame(
        {key: pl.Series(values, dtype=pl.Int64), frequency: [i + 1 for i in range(len(values))]},
        schema_overrides={frequency: pl.UInt64},
    )
    selections = {
        "all": KeySelection(limit=limit),
        "missing": KeySelection(f'"{key}" IS NULL', pl.col(key).is_null(), limit),
        "valid": KeySelection(f'"{key}" IS NOT NULL', pl.col(key).is_not_null(), limit),
        "heavy": KeySelection(f'"{frequency}" > 1', pl.col(frequency) > 1, limit),
    }
    calls = []
    with DataContext() as ctx, duckdb.connect() as native:
        native.register("data", frame.to_arrow())
        relation = KeyRelation(
            ctx,
            ("key_rows",),
            (key,),
            frequency,
            frame.schema,
            sql="SELECT * FROM data" if backend == "native" else None,
            lazy=frame.lazy() if backend == "lazy" else None,
        )

        def sql(query, **_):
            calls.append(query)
            return native.execute(query).pl()

        def collect(_names, plan):
            calls.append(plan)
            return plan.collect(engine="streaming")

        with patch.object(ctx, "sql", sql), patch.object(ctx, "collect_many", collect):
            results = relation.summarize_many(selections)
    assert len(calls) == 1
    for name, selection in selections.items():
        expected = frame.filter(selection.predicate)
        result = results[name]
        assert result.keys == expected.height
        assert result.rows == sum(expected[frequency].to_list())
        assert result.sample.height == min(limit, expected.height)
        assert result.sample.schema == frame.schema
        assert result.sample.join(
            expected, on=frame.columns, how="anti", nulls_equal=True
        ).is_empty()


def test_native_batched_summaries_evaluate_grouped_input_once():
    seen = []

    def observed(value):
        seen.append(value)
        return value

    with DataContext() as ctx, duckdb.connect() as native:
        native.create_function(
            "observed", observed, parameters=["BIGINT"], return_type="BIGINT", side_effects=True
        )
        relation = KeyRelation(
            ctx,
            (),
            ("key",),
            "rows",
            {"key": pl.Int64(), "rows": pl.UInt64()},
            sql="SELECT observed(range) AS key, 2::UBIGINT AS rows FROM range(10)",
        )
        with patch.object(ctx, "sql", side_effect=lambda sql, **_: native.execute(sql).pl()):
            result = relation.summarize_many(
                {
                    "all": KeySelection(),
                    "small": KeySelection('"key" < 3', pl.col("key") < 3, 2),
                    "large": KeySelection('"key" >= 3', pl.col("key") >= 3, 1),
                }
            )
    assert sorted(seen) == list(range(10))
    assert (result["all"].keys, result["all"].rows) == (10, 20)
    assert (result["small"].keys, result["small"].rows) == (3, 6)
    assert (result["large"].keys, result["large"].rows) == (7, 14)


@pytest.mark.parametrize("backend", ["native", "lazy"])
def test_summary_batch_bounds_query_width(backend):
    calls = []
    with DataContext() as ctx, duckdb.connect() as native:
        frame = pl.DataFrame(
            {"key": [2**64 - 1], "rows": [2]}, schema={"key": pl.UInt64, "rows": pl.UInt64}
        )
        native.register("data", frame.to_arrow())
        relation = KeyRelation(
            ctx,
            (),
            ("key",),
            "rows",
            frame.schema,
            sql="SELECT * FROM data" if backend == "native" else None,
            lazy=frame.lazy() if backend == "lazy" else None,
        )

        def sql(query, **_):
            calls.append(1)
            return native.execute(query).pl()

        def collect(_names, plan):
            calls.append(1)
            return plan.collect(engine="streaming")

        with patch.object(ctx, "sql", sql), patch.object(ctx, "collect_many", collect):
            result = relation.summarize_many({str(i): KeySelection(limit=1) for i in range(33)})
    assert len(calls) == 3
    assert len(result) == 33
    assert all(item.keys == 1 and item.rows == 2 for item in result.values())
    assert all(item.sample["key"][0] == 2**64 - 1 for item in result.values())
