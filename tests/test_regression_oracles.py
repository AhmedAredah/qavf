"""Regression comparison against independent boundary oracles."""

from __future__ import annotations

from decimal import Decimal, Inexact, localcontext
from fractions import Fraction

import polars as pl
import pytest

from qavf import ResourceLimits, compare_datasets
from qavf.engine.sql_read import SqlReadSession
from qavf.engine.sqlite_store import Store
from qavf.regression import _comparison, compare_frames, compare_outputs
from qavf.typed_values import exact_scale


@pytest.mark.parametrize(
    "values,factor",
    [
        ([0, None, -(2**127), 2**127 - 1], 1),
        ([0, None, -(2**126), 2**126 - 1], 2),
        ([-1, 0, None], 2**127),
        ([0, None, 2**126], 4),
        ([-(2**127)], -1),
        ([2**126], -2),
        ([-(2**126)], -2),
    ],
)
def test_integer_scale_matches_python_integer_range_or_refuses(values, factor):
    expected = [None if value is None else value * factor for value in values]
    frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int128)})
    expression = exact_scale(pl.col("x"), pl.Int128, factor)
    if any(value is not None and not -(2**127) <= value < 2**127 for value in expected):
        with pytest.raises(ValueError, match="storage precision"):
            frame.lazy().select(expression).collect(engine="streaming")
    else:
        assert (
            frame.lazy().select(expression).collect(engine="streaming")["x"].to_list() == expected
        )


def test_overflowing_public_unit_conversion_cannot_pass():
    left = pl.DataFrame({"id": [1], "x": pl.Series([2**126], dtype=pl.Int128)})
    right = pl.DataFrame({"id": [1], "x": pl.Series([0], dtype=pl.Int128)})
    result = compare_datasets(
        left,
        right,
        keys="id",
        units={"x": {"factor": 4}},
        rtol=0,
        atol=0,
        entry_points=False,
    )
    assert result.error is not None


@pytest.mark.parametrize("precision", [3, 6, 28, 50])
@pytest.mark.parametrize(
    "values,dtype,factor",
    [
        ([1, None, -1], pl.Int64, Decimal("1.2345")),
        ([1], pl.Int64, Decimal("1.0000000000000000000000000001")),
        ([1], pl.Int64, Decimal("1.2345678901234567890123456789")),
        ([Decimal("1.2345678901234567890123456789")], pl.Decimal(38, 28), Decimal("1.000")),
        ([Decimal("1234567890123456789012345678.1234567890")], pl.Decimal(38, 10), Decimal("0.1")),
    ],
)
def test_decimal_scale_is_context_independent_with_fraction_oracle(
    values, dtype, factor, precision
):
    expected = [None if value is None else Fraction(value) * Fraction(factor) for value in values]
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    with localcontext() as context:
        context.prec = precision
        context.traps[Inexact] = True
        actual = frame.lazy().select(exact_scale(pl.col("x"), dtype, factor)).collect()["x"]
        assert [None if value is None else Fraction(value) for value in actual] == expected


def test_public_decimal_factor_retains_its_last_digit():
    frame = pl.DataFrame({"id": [1], "x": [1]})
    result = compare_datasets(
        frame,
        frame,
        keys="id",
        units={"x": {"factor": Decimal("1.0000000000000000000000000001")}},
        rtol=0,
        atol=0,
        entry_points=False,
    )
    assert result.error is None
    assert result.levels[-1].values_changed == 1


@pytest.mark.parametrize(
    "resources",
    [
        None,
        ResourceLimits(),
        ResourceLimits(evidence_bytes=0),
        ResourceLimits(evidence_rows=1),
        ResourceLimits(isolate=False),
    ],
)
def test_eager_evidence_only_policy_needs_no_work_store(monkeypatch, resources):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager empty/evidence-only policy must not build a disk index")

    monkeypatch.setattr(Store, "__init__", forbidden)
    left = pl.DataFrame({"id": [1, 2], "v": [10, 20]})
    right = left.with_columns(pl.col("v") + 1)
    result = compare_outputs(left, right, "id", resources=resources)
    assert result.execution == "in_memory" and result.values_changed == 2
    if resources is not None and resources.evidence_bytes == 0:
        assert result.differences == []
    if resources is not None and resources.evidence_rows == 1:
        assert len(result.differences) == 1


@pytest.mark.parametrize(
    "dtype,values,candidates",
    [
        (pl.Int64, [None, 0, 1, 5, -5], [1, 1, 2, 1, -3]),
        (pl.Int64, [-(2**63), 2**63 - 1, 2**53 + 1], [2**63 - 1, -(2**63), 2**53 + 2]),
        (pl.UInt64, [0, 2**64 - 1, 2**53 + 1], [2**64 - 1, 0, 2**53]),
        (pl.Int128, [-(2**127), 2**127 - 1], [2**127 - 1, -(2**127)]),
    ],
)
def test_exact_integer_counts_and_metrics_do_not_need_cell_dispatch(
    monkeypatch, dtype, values, candidates
):
    def forbidden(*args, **kwargs):
        raise AssertionError("exact same-dtype integer comparison must avoid scalar cell dispatch")

    monkeypatch.setattr(_comparison, "_cell_change", forbidden)
    left = pl.DataFrame({"id": range(len(values)), "v": pl.Series(values, dtype=dtype)})
    right = pl.DataFrame({"id": range(len(values)), "v": pl.Series(candidates, dtype=dtype)})
    result = compare_frames(left, right, "id", rtol=0, atol=0)
    pairs = [
        (a, b) for a, b in zip(values, candidates, strict=True) if a is not None and b is not None
    ]
    assert result.values_changed == sum(a != b for a, b in zip(values, candidates, strict=True))
    assert result.max_abs_diff == max(float(abs(Fraction(a) - Fraction(b))) for a, b in pairs)
    assert result.max_rel_diff == max(
        float(abs(Fraction(a) - Fraction(b)) / abs(a)) for a, b in pairs if a
    )


@pytest.mark.parametrize("limits", [{"max_differences": 0}, {"max_evidence_bytes": 0}])
def test_native_count_only_comparison_has_no_ordering_operator(tmp_path, monkeypatch, limits):
    left = pl.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]})
    right = pl.DataFrame({"id": [4, 2, 1], "v": [40, 21, 10]})
    paths = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(paths[0])
    right.write_parquet(paths[1])
    queries = []
    original = SqlReadSession.batches

    def capture(self, query, batch_rows=65536):
        queries.append(query)
        yield from original(self, query, batch_rows=batch_rows)

    monkeypatch.setattr(SqlReadSession, "batches", capture)
    result = compare_outputs(*paths, "id", rtol=0, atol=0, **limits)
    expected = compare_frames(left, right, "id", rtol=0, atol=0, **limits)
    assert result.model_dump(
        exclude={"name", "execution", "fallback_reason"}
    ) == expected.model_dump(exclude={"name", "execution", "fallback_reason"})
    joins = [query for query in queries if "FULL OUTER JOIN" in query]
    assert joins and all("ORDER BY" not in query and "row_number()" not in query for query in joins)
