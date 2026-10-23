"""Exact-value casts, temporal precision and regression comparison boundaries."""

import math
import random
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction

import polars as pl
import pytest

from qavf import ResourceLimits, compare_datasets
from qavf.missingness import missing_expr
from qavf.regression import ComparisonResult, compare_frames, compare_outputs
from qavf.settings import ColumnSpec
from qavf.typed_values import checked_cast
from qavf.value_policy import temporal_ticks


def _cast(values, source, target):
    original = pl.Series("v", values, dtype=source)
    converted, lost = checked_cast(pl.col("v"), source, target)
    result = original.to_frame().select(converted.alias("converted"), lost.alias("lost"))
    return original, result


@pytest.mark.parametrize(
    "unit,good,bad",
    [
        ("ms", "123000000", "123000001"),
        ("us", "123456000", "123456001"),
        ("ns", "123456789", "1234567891"),
    ],
)
def test_timestamp_text_precision_is_checked_before_accepting_parse(unit, good, bad):
    _, result = _cast(
        [f"2024-01-01T12:34:56.{good}", f"2024-01-01T12:34:56.{bad}", None],
        pl.String,
        pl.Datetime(unit),
    )
    assert result["lost"].to_list() == [False, True, False]


def test_wide_timestamp_text_does_not_require_nanosecond_storage():
    _, result = _cast(
        ["2500-01-01T00:00:00.000001000", "2500-01-01T00:00:00.000001001"],
        pl.String,
        pl.Datetime("us"),
    )
    assert result["converted"][0] == datetime(2500, 1, 1, microsecond=1)
    assert result["lost"].to_list() == [False, True]


def test_public_text_datetime_cast_cannot_hide_a_nanosecond_difference():
    left = pl.DataFrame({"id": [1], "v": ["2024-01-01T00:00:00.000000001"]})
    right = pl.DataFrame({"id": [1], "v": ["2024-01-01T00:00:00.000000000"]})
    result = compare_datasets(
        left,
        right,
        keys="id",
        cast={"v": "datetime"},
        rtol=0,
        atol=0,
        entry_points=False,
    )
    assert result.error is not None and "would change" in result.error


@pytest.mark.parametrize("direction", ["decimal_float", "float_decimal"])
def test_public_numeric_cast_cannot_hide_decimal_binary_difference(direction):
    decimal = pl.DataFrame({"id": [1], "v": pl.Series([Decimal("0.1")], dtype=pl.Decimal(38, 1))})
    binary = pl.DataFrame({"id": [1], "v": [0.1]})
    left, right = (decimal, binary) if direction == "decimal_float" else (binary, decimal)
    spec = (
        {"dtype": "float", "side": "baseline"}
        if direction == "decimal_float"
        else {
            "dtype": "decimal",
            "scale": 1,
            "side": "baseline",
        }
    )
    result = compare_datasets(
        left,
        right,
        keys="id",
        cast={"v": spec},
        rtol=0,
        atol=0,
        entry_points=False,
    )
    assert result.error is not None and "would change" in result.error


@pytest.mark.parametrize("float_type", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("scale", [0, 1, 2, 8, 20, 38])
def test_cross_decimal_float_loss_flags_match_exact_fraction_oracle(float_type, scale):
    rng = random.Random(31 + scale)
    dtype = pl.Decimal(38, scale)
    with localcontext() as context:
        context.prec = 90
        decimals = [Decimal(rng.randrange(-(10**15), 10**15)).scaleb(-scale) for _ in range(50)]
        decimals += [Decimal(0), Decimal(1).scaleb(-scale), None]
    for values, source, target in [(decimals, dtype, float_type)]:
        original, result = _cast(values, source, target)
        expected = [
            a is not None and (b is None or Fraction(a) != Fraction(b))
            for a, b in zip(original, result["converted"], strict=True)
        ]
        assert result["lost"].to_list() == expected
    floats = [math.ldexp(rng.randrange(-100000, 100000), -rng.randrange(0, 40)) for _ in range(50)]
    floats += [0.1, 0.25, 0.5, None]
    if scale == 38:
        floats = [None if value is None else value / 100000 for value in floats]
    original, result = _cast(floats, float_type, dtype)
    expected = [
        a is not None and (b is None or Fraction(a) != Fraction(b))
        for a, b in zip(original, result["converted"], strict=True)
    ]
    assert result["lost"].to_list() == expected


@pytest.mark.parametrize(
    "source,target,values",
    [
        (pl.Decimal(38, 2), pl.Float64, [Decimal("0.25"), Decimal("0.50"), None]),
        (pl.Float64, pl.Decimal(38, 2), [0.25, 0.5, None]),
        (pl.Float32, pl.Decimal(38, 2), [0.25, 0.5, None]),
    ],
)
def test_exact_dyadic_casts_remain_accepted(source, target, values):
    original, result = _cast(values, source, target)
    assert result["lost"].to_list() == [False, False, False]
    assert all(
        a is None or Fraction(a) == Fraction(b)
        for a, b in zip(original, result["converted"], strict=True)
    )


@pytest.mark.parametrize(
    "after,atol,expected",
    [
        ("0.3", 0.3, True),
        ("0.100000000000000003", 0.1, False),
    ],
)
def test_record_and_group_exact_tolerance_boundaries_agree(after, atol, expected):
    left = pl.DataFrame({"id": [1], "v": pl.Series([Decimal("0")], dtype=pl.Decimal(38, 18))})
    right = pl.DataFrame({"id": [1], "v": pl.Series([Decimal(after)], dtype=pl.Decimal(38, 18))})
    result = compare_datasets(
        left,
        right,
        keys="id",
        rtol=0,
        atol=atol,
        entry_points=False,
        levels=["records", {"level": "groups", "measures": ["sum(v)"], "decision": "material"}],
    )
    assert result.error is None
    record = next(c for c in result.level("records").checks if c.check == "values")
    group = next(c for c in result.level("groups").checks if c.check == "sum(v)")
    assert record.passed is expected and group.passed is expected


@pytest.mark.parametrize("zone", [None, "UTC", "America/New_York"])
def test_pandas_timestamp_missing_marker_preserves_nanoseconds(zone):
    pd = pytest.importorskip("pandas")
    stamp = pd.Timestamp("2024-01-01T00:00:00.000000001")
    dtype = pl.Datetime("ns", zone)
    expected = stamp.value if zone is None else stamp.tz_localize(zone).value
    assert temporal_ticks(stamp, dtype) == expected
    frame = pl.DataFrame({"v": pl.Series([expected - 1, expected], dtype=dtype)})
    assert frame.select(missing_expr(frame, "v", ColumnSpec(missing_values=[stamp])))[
        "v"
    ].to_list() == [False, True]


def test_zoned_pandas_marker_preserves_instant_and_nanosecond_remainder():
    pd = pytest.importorskip("pandas")
    stamp = pd.Timestamp("2024-01-01T00:00:00.000000001", tz="America/New_York")
    assert temporal_ticks(stamp, pl.Datetime("ns", "UTC")) == stamp.value
    # A finer-than-storage boundary is retained, never rounded onto an adjacent row.
    assert temporal_ticks(stamp, pl.Datetime("us", "UTC")).denominator == 1000


@pytest.mark.parametrize("ignored", [True, False])
def test_projected_fallback_renames_only_retained_baseline_columns(ignored):
    left = pl.DataFrame({"id": [1], "obj": pl.Series([{"a": 1}], dtype=pl.Object), "old": [1]})
    right = pl.DataFrame({"id": [1], "obj": pl.Series([{"a": 1}], dtype=pl.Object), "new": ["x"]})
    options = {
        "keys": "id",
        "column_map": {"old": "new"},
        "ignore_columns": ["new"] if ignored else [],
    }
    eager = compare_frames(left, right, **options)
    lazy = compare_outputs(left.lazy(), right.lazy(), **options)
    assert lazy.execution == "projected_fallback"
    assert lazy.equal == eager.equal
    assert lazy.dtypes_changed == eager.dtypes_changed


@pytest.mark.parametrize("tolerance", [Decimal("1e-400"), Decimal("1e1000"), 10**400])
@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float64])
def test_finite_exact_tolerances_roundtrip_without_underflow_or_infinity(tolerance, dtype):
    left = pl.DataFrame({"id": [1], "v": pl.Series([1], dtype=dtype)})
    right = pl.DataFrame({"id": [1], "v": pl.Series([2], dtype=dtype)})
    result = compare_frames(
        left, right, "id", rtol=tolerance, atol=0, column_tolerances={"v": (tolerance, 0)}
    )
    restored = ComparisonResult.model_validate_json(result.model_dump_json())
    assert restored.rtol == tolerance
    assert restored.column_tolerances["v"][0] == tolerance
    assert restored.equal == result.equal
    for difference in restored.differences:
        assert difference.rtol == tolerance


@pytest.mark.parametrize(
    "resources", [ResourceLimits(native_threads=1), ResourceLimits(max_workers=1)]
)
def test_eager_object_frames_keep_projected_fallback_under_managed_policy(resources):
    left = pl.DataFrame({"id": [1], "v": pl.Series([{"a": 1}], dtype=pl.Object)})
    right = pl.DataFrame({"id": [1], "v": pl.Series([{"a": 2}], dtype=pl.Object)})
    result = compare_outputs(left, right, keys="id", resources=resources)
    assert result.execution == "projected_fallback"
    assert result.values_changed == 1
    assert result.differences[0].path == ["a"]


def test_pandas_date_marker_does_not_discard_submicrosecond_time_of_day():
    pd = pytest.importorskip("pandas")
    frame = pl.DataFrame({"v": [date(2024, 1, 1)]})
    exact = ColumnSpec(missing_values=[pd.Timestamp("2024-01-01")])
    finer = ColumnSpec(missing_values=[pd.Timestamp("2024-01-01T00:00:00.000000001")])
    assert frame.select(missing_expr(frame, "v", exact))["v"].to_list() == [True]
    assert frame.select(missing_expr(frame, "v", finer))["v"].to_list() == [False]
