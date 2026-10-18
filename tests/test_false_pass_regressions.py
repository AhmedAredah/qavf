"""Independent public-API acceptance for the six reproduced false-pass defects.

Numerical oracles operate on exact stored values. Unsupported representations may
be explicitly unassessed where noted, but must never be silently accepted.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from qavf import validate
from qavf.engine.severity import Status
from qavf.reports.models import QAReport


def _check(
    rule: str,
    inputs: dict[str, Any],
    params: dict[str, Any] | None = None,
    datasets: dict[str, Any] | None = None,
) -> QAReport:
    return validate(
        inputs,
        config={
            "datasets": datasets or {key: {} for key in inputs},
            "plugins": {"entry_points": False},
            "engine": {
                "parallel": False,
                "run_unconfigured_rules": False,
                "fail_on": "WARNING",
            },
            "rules": {rule: {"params": params or {}}},
        },
    )


@pytest.mark.parametrize("union_by_name", [False, True])
@pytest.mark.parametrize("large_integer", [False, True])
def test_mixed_parquet_cannot_accept_values_changed_by_a_union(
    tmp_path: Path, union_by_name: bool, large_integer: bool
) -> None:
    if large_integer:
        # 2**53+1 cannot be exactly represented by Float64.
        integer, floating, limit = 2**53 + 1, 0.0, 2**53
    else:
        integer, floating, limit = 1, 1.25, 1
    pl.DataFrame({"x": pl.Series([integer], dtype=pl.Int64)}).write_parquet(tmp_path / "a.parquet")
    pl.DataFrame({"x": pl.Series([floating], dtype=pl.Float64)}).write_parquet(
        tmp_path / "b.parquet"
    )
    location = str(tmp_path / "*.parquet")
    report = validate(
        location,
        schema={"columns": {"x": {"le": limit}}},
        config={
            "datasets": {
                "d": {
                    "source": {
                        "type": "file",
                        "path": location,
                        "options": {"union_by_name": union_by_name},
                    }
                }
            },
            "plugins": {"entry_points": False},
            "engine": {"parallel": False},
        },
    )
    # A lossless implementation finds a violation; a strict reader may reject
    # the incompatible schemas. Either must make the run unacceptable.
    assert report.exit_code() != 0
    checks = report.results_for("INP-003")
    if not any(r.status is Status.FAIL for r in checks):
        assert any(
            r.status is Status.ERROR and "matching logical types" in r.message
            for r in report.results
        )
        assert not any(r.status is Status.PASS for r in checks)


_MISSING = [
    pytest.param([None, 0.0], {}, "x >= 0", id="null"),
    pytest.param([math.nan, 0.0], {}, "x >= 0", id="nan"),
    pytest.param([-999, 0], {"missing_values": [-999]}, "x <= 0", id="sentinel"),
    pytest.param([" ", "valid"], {"blank_as_missing": True}, "length(x) > 0", id="blank"),
]


@pytest.mark.parametrize("values,column,expression", _MISSING)
@pytest.mark.parametrize("null_as", ["fail", "pass"])
@pytest.mark.parametrize("adapter", ["frame", "lazy", "parquet"])
def test_expression_missing_policy_is_independent_of_input_adapter(
    tmp_path: Path,
    values: list[Any],
    column: dict[str, Any],
    expression: str,
    null_as: str,
    adapter: str,
) -> None:
    frame = pl.DataFrame({"x": values})
    source: Any = frame
    if adapter == "lazy":
        source = frame.lazy()
    elif adapter == "parquet":
        source = tmp_path / "missing.parquet"
        frame.write_parquet(source)
    report = _check(
        "INP-006",
        {"d": source},
        datasets={
            "d": {
                # The missing row is permitted by the column contract, so only
                # the expression's explicit NULL policy decides its outcome.
                "schema": {"columns": {"x": {"nullable": True, **column}}},
                "expressions": [{"expr": expression, "null_as": null_as}],
            }
        },
    )
    (result,) = report.results_for("INP-006")
    assert result.status is (Status.FAIL if null_as == "fail" else Status.PASS)
    assert result.affected_count == (1 if null_as == "fail" else 0)


def test_expression_normalizes_referenced_dataset_before_subquery() -> None:
    report = _check(
        "INP-006",
        {"d": pl.DataFrame({"x": [0]}), "ref": pl.DataFrame({"y": [-999]})},
        datasets={
            "d": {"expressions": [{"expr": 'x >= (SELECT max(y) FROM "ref")', "null_as": "fail"}]},
            "ref": {"schema": {"columns": {"y": {"nullable": True, "missing_values": [-999]}}}},
        },
    )
    (result,) = report.results_for("INP-006")
    assert result.status is Status.FAIL
    assert result.affected_count == 1


@pytest.mark.parametrize("adapter", ["frame", "lazy", "parquet"])
def test_sql_expression_rejects_ambiguous_case_distinct_columns(
    tmp_path: Path, adapter: str
) -> None:
    frame = pl.DataFrame({"x": [0], "X": [10]})
    source: Any = frame
    if adapter == "lazy":
        source = frame.lazy()
    elif adapter == "parquet":
        source = tmp_path / "case_distinct.parquet"
        frame.write_parquet(source)
    report = _check(
        "INP-006",
        {"d": source},
        datasets={"d": {"expressions": [{"expr": '"X" <= 1', "null_as": "fail"}]}},
    )
    (result,) = report.results_for("INP-006")
    # DuckDB identifiers are case-insensitive, even when quoted. Checking x=0
    # instead of X=10 is unsafe; ambiguity must be explicit at the binding edge.
    assert result.status is Status.ERROR
    assert "same identifier" in result.message.lower()
    assert report.exit_code() != 0


def test_sql_expression_accepts_an_unambiguous_uppercase_column() -> None:
    report = _check(
        "INP-006",
        {"d": pl.DataFrame({"X": [10]})},
        datasets={"d": {"expressions": [{"expr": '"X" <= 1', "null_as": "fail"}]}},
    )
    (result,) = report.results_for("INP-006")
    assert result.status is Status.FAIL
    assert result.affected_count == 1


@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_coverage_preserves_both_dst_instants_and_missing_entity(
    zone: str, order: list[int]
) -> None:
    first = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    second = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    # The intended two periods are proved in UTC, independently of local
    # datetime equality, which intentionally ignores fold for shared tzinfo.
    assert second.timestamp() - first.timestamp() == 3600
    times = [first, second, second]
    entities = [1, 1, 2]
    frame = pl.DataFrame(
        {
            "id": [entities[i] for i in order],
            "t": [times[i].astimezone(ZoneInfo(zone)) for i in order],
        }
    )
    report = _check(
        "SPA-003",
        {"d": frame},
        {"entity_column": "id", "time_column": "t", "expected_entities": [1, 2]},
    )
    (result,) = report.results_for("SPA-003")
    assert result.status is Status.FAIL
    assert result.metrics["slices_checked"] == 2
    assert result.affected_count == 1
    assert result.affected_records[0]["missing_entities"] == [2]


@pytest.mark.parametrize("exponent", [0, -200])
def test_spatial_normalization_never_accepts_an_erased_outlier(exponent: int) -> None:
    numerator = [10.0**exponent, 10.0**exponent, 100.0 * 10.0**exponent]
    denominator = 1e200
    exact = [Fraction(v) / Fraction(denominator) for v in numerator]
    assert exact[2] > 2 * exact[1]
    report = _check(
        "SPA-002",
        {
            "n": pl.DataFrame({"id": [1, 2, 3], "v": numerator}),
            "d": pl.DataFrame({"id": [1, 2, 3], "w": [denominator] * 3}),
        },
        {
            "metrics": [
                {
                    "dataset": "n",
                    "value": "v",
                    "entity_column": "id",
                    "normalize_by": {"dataset": "d", "value": "w"},
                }
            ]
        },
    )
    results = report.results_for("SPA-002")
    assert report.exit_code() != 0
    assert not any(r.status is Status.PASS for r in results)
    if exponent == 0:
        assert any(r.status is Status.FAIL and r.affected_count == 1 for r in results)
    else:
        # Ratios below Float64's supported range may be explicitly unassessed.
        assert any(r.status is Status.FAIL and r.affected_count == 1 for r in results) or any(
            r.status is Status.ERROR
            and r.affected_count == 3
            and all("underflow" in row["problem"] for row in r.affected_records)
            for r in results
        )


def _median(values: list[Fraction]) -> Fraction:
    ordered = sorted(values)
    middle, odd = divmod(len(values), 2)
    return ordered[middle] if odd else (ordered[middle - 1] + ordered[middle]) / 2


def _exact_level_shift_score(values: np.ndarray) -> tuple[int, float]:
    exact = [Fraction(float(v)) for v in values]
    candidates = []
    for split in range(4, len(exact) - 3):
        left, right = exact[:split], exact[split:]
        before, after = _median(left), _median(right)
        residuals = [abs(v - before) for v in left] + [abs(v - after) for v in right]
        scale = Fraction(14826, 10000) * _median(residuals)
        candidates.append((split, float(abs(after - before) / scale)))
    return max(candidates, key=lambda candidate: candidate[1])


@pytest.mark.parametrize("offset", [0.0, 1e16])
def test_level_shift_matches_exact_oracle_after_reversible_offset(offset: float) -> None:
    original = np.array([18, 18, 0, 12, 8, 10, 8, 12, -18, -2, -12, -20], dtype=float)
    shifted = original + offset
    assert np.array_equal(shifted - offset, original)
    expected_split, expected_score = _exact_level_shift_score(shifted)
    assert (expected_split, expected_score) == _exact_level_shift_score(original)
    assert expected_score > 5
    report = _check(
        "TMP-008",
        {"d": pl.DataFrame({"t": range(len(shifted)), "x": shifted})},
        {
            "series": [{"dataset": "d", "value": "x", "time_column": "t"}],
            # Offset invariance applies to the statistical score, not the
            # intentionally level-dependent relative materiality requirement.
            "min_relative_shift": 0,
            "threshold": 5,
        },
    )
    (result,) = report.results_for("TMP-008")
    assert result.status is Status.FAIL
    assert result.affected_count == 1
    assert result.affected_records[0]["shift_period"] == expected_split
    assert result.affected_records[0]["score"] == pytest.approx(expected_score, abs=0.0005)


def _exact_quantile_psi(reference: np.ndarray, current: np.ndarray) -> float:
    """Small exact interpolation oracle; no NumPy statistical kernel is used."""
    ordered = sorted(Fraction(float(v)) for v in reference)
    edges = []
    for tenth in range(11):
        position = Fraction((len(ordered) - 1) * tenth, 10)
        lower = position.numerator // position.denominator
        fraction = position - lower
        edges.append(
            ordered[lower]
            if not fraction
            else ordered[lower] * (1 - fraction) + ordered[lower + 1] * fraction
        )
    edges = sorted(set(edges))

    def counts(values: np.ndarray) -> Counter[int]:
        # Values in this fixture are inside reference support, including the
        # upper endpoint, which belongs to the final interior bin.
        exact = [Fraction(float(v)) for v in values]
        assert min(exact) >= edges[0] and max(exact) <= edges[-1]
        return Counter(bisect_right(edges[:-1], v) for v in exact)

    expected, actual = counts(reference), counts(current)
    terms = []
    for category in expected.keys() | actual.keys():
        before = max(expected[category] / len(reference), 1e-6)
        after = max(actual[category] / len(current), 1e-6)
        terms.append((after - before) * math.log(after / before))
    return math.fsum(terms)


@pytest.mark.parametrize("scale", [1.0, math.ldexp(1.0, 1023)])
def test_psi_does_not_hide_drift_after_quantile_overflow(scale: float) -> None:
    reference = np.array(
        [
            -1.5e308,
            -1.4e308,
            -1.3e308,
            -1.2e308,
            -1.1e308,
            -1e308,
            1e308,
            1.1e308,
            1.2e308,
            1.3e308,
            1.4e308,
            1.5e308,
        ]
    )
    current = reference.copy()
    current[6] = -1e308
    assert np.array_equal(reference / scale * scale, reference)
    assert np.array_equal(current / scale * scale, current)
    expected = _exact_quantile_psi(reference, current)
    assert expected > 0.2
    report = _check(
        "STA-002",
        {"r": pl.DataFrame({"x": reference / scale}), "c": pl.DataFrame({"x": current / scale})},
        {
            "comparisons": [
                {
                    "dataset": "c",
                    "reference": "r",
                    "column": "x",
                    "min_samples": 2,
                    "bins": 10,
                    "max_psi": 0.2,
                }
            ]
        },
    )
    (result,) = report.results_for("STA-002")
    assert report.exit_code() != 0
    if result.status is Status.FAIL:
        assert result.metrics["psi"] == pytest.approx(expected, abs=0.0000005)
    else:
        # Rejecting the extreme range explicitly is safe; the scaled case is
        # comfortably representable and must actually assess the drift.
        assert scale == 1.0 and result.status is Status.ERROR
        assert any(word in result.message for word in ("precision", "range", "scal", "quantile"))
