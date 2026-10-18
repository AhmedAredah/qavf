"""Independent numerical oracles for indexed, batch-based statistical checks."""

import math
from collections import Counter

import numpy as np
import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.engine.observation import Observation
from qavf.rules._distribution import observed_distribution
from qavf.rules._outlier_batches import scan_target
from qavf.rules.helpers import normalize_missing
from qavf.rules.statistical_checks import ColumnOutlierRule, OutlierTarget
from qavf.statistics.ordered import fit_outliers, quantiles, scaled
from qavf.stats import OutlierMethod, detect_outliers


def _counts(values):
    return sorted(Counter(map(float, values)).items())


def _fit(values, method):
    counts = _counts(values)

    def deviations(centre, origin, unit):
        return iter(_counts(abs(scaled(value, origin, unit) - centre) for value in values))

    return fit_outliers(
        lambda: iter(counts), size=len(values), minimum=min(values), maximum=max(values),
        method=method, threshold=None, min_group_size=3,
        original_median=float(np.median(values)), deviations=deviations,
    )


@pytest.mark.parametrize("size", [1, 2, 3, 8, 31, 137])
def test_counted_quantiles_match_numpy_linear_interpolation(size):
    values = np.random.default_rng(size).integers(-15, 20, size=size).astype(float)
    probabilities = [1.0, 0.0, 0.5, 0.25, 0.75, 0.1, 0.99, 0.5]
    actual = quantiles(iter(_counts(values)), size, probabilities)
    expected = np.quantile(values, probabilities, method="linear")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method", list(OutlierMethod))
def test_randomized_count_models_match_complete_array_verdicts(method):
    rng = np.random.default_rng(487)
    for size in (16, 31, 101):
        for iteration in range(12):
            values = rng.normal(size=size)
            if iteration % 2:
                values[-1] *= 20
            expected = detect_outliers(values, method=method)
            model = _fit(values, method)
            assert model.testable
            np.testing.assert_array_equal(
                [model.score(value)[1] for value in values], expected.mask
            )


@pytest.mark.parametrize("method", list(OutlierMethod))
@pytest.mark.parametrize("values", [
    [1.0] * 15 + [100.0],
    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 40.0] * 2,
    [float(2**53 + 2 * i) for i in range(15)] + [float(2**53 + 1000)],
    [1e-300 * i for i in range(1, 16)] + [1e-298],
    [1e300 * i for i in range(1, 16)] + [1e302],
    [0.0] * 16,
])
def test_indexed_outlier_model_matches_array_detector(method, values):
    expected = detect_outliers(values, method=method)
    model = _fit(values, method)
    assert model.testable
    observations = [model.score(value) for value in values]
    np.testing.assert_array_equal([flag for _, flag in observations], expected.mask)
    np.testing.assert_allclose([score for score, _ in observations], expected.scores, rtol=2e-14)
    assert model.data_units(model.lower) == pytest.approx(expected.lower, rel=2e-14, abs=0)
    assert model.data_units(model.upper) == pytest.approx(expected.upper, rel=2e-14, abs=0)


def _small_batches(monkeypatch, size):
    original = Observation.batches
    observed = []

    def batches(self, batch_rows=65_536, *, build=None):
        for frame in original(self, min(size, batch_rows), build=build):
            observed.append(frame.height)
            assert frame.height <= size
            yield frame

    monkeypatch.setattr(Observation, "batches", batches)
    return observed


@pytest.mark.parametrize("batch_rows", [1, 7, 61])
def test_many_small_groups_keep_exact_counts_without_full_projection(monkeypatch, batch_rows):
    # Each group has eight observations, and groups straddle every batch size.
    groups = 129
    frame = pl.DataFrame({
        "g": [group for slot in range(8) for group in range(groups)],
        "x": [100.0 if slot == 7 else 1.0 for slot in range(8) for _ in range(groups)],
    })
    observed = _small_batches(monkeypatch, batch_rows)
    with DataContext(frames={"data": frame}, evidence_limit=3) as context:
        def forbidden(*args, **kwargs):
            pytest.fail("batch statistical execution requested a full projection")

        monkeypatch.setattr(context, "project", forbidden)
        monkeypatch.setattr(context, "get", forbidden)
        original_empty = np.empty

        def bounded_empty(shape, *args, **kwargs):
            count = math.prod(shape) if isinstance(shape, tuple) else shape
            assert count <= 8, "outlier execution allocated a full observation array"
            return original_empty(shape, *args, **kwargs)

        monkeypatch.setattr(np, "empty", bounded_empty)
        scan = scan_target(
            context, OutlierTarget(dataset="data", value="x", group_by=("g",)),
            OutlierMethod.IQR, None, 8, 3, normalize_missing,
        )
        assert scan.tested == groups * 8
        assert scan.outliers == groups
        assert scan.invalid == scan.untested == 0
        assert len(scan.records) == 3
        assert [row["g"] for row in scan.records] == [0, 1, 2]
    assert observed and max(observed) <= batch_rows


@pytest.mark.parametrize("batch_rows", [2, 17])
def test_log_direction_materiality_and_invalid_counts_survive_batch_boundaries(
    monkeypatch, batch_rows
):
    values = [10.0] * 15 + [1000.0, 0.0, -1.0, None, float("nan"), float("inf")]
    _small_batches(monkeypatch, batch_rows)
    with DataContext(frames={"data": pl.DataFrame({"x": values})}, evidence_limit=2) as context:
        scan = scan_target(
            context, OutlierTarget(dataset="data", value="x", transform="log",
                                  direction="high", min_relative_deviation=0.1),
            OutlierMethod.MODIFIED_ZSCORE, None, 8, 2, normalize_missing,
        )
    assert scan.tested == 16
    assert scan.nonpositive == 2
    assert scan.invalid == 3
    assert scan.outliers == 1
    assert scan.records[0]["x"] == 1000
    assert scan.records[0]["relative_deviation"] == 99
    assert scan.records[0]["log_value"] == pytest.approx(math.log(1000))


def test_outlier_two_pass_algorithm_observes_volatile_source_only_once():
    calls = []

    def loader(frame):
        calls.append(1)
        return pl.DataFrame({"x": [1.0] * 15 + ([100.0] if len(calls) == 1 else [1.0])})

    lazy = pl.DataFrame({"x": [1.0] * 16}).lazy().map_batches(loader, schema={"x": pl.Float64})
    with DataContext(frames={"data": lazy}) as context:
        rule = ColumnOutlierRule({"targets": [{"dataset": "data", "value": "x"}]})
        findings = rule.evaluate(context)
    assert calls == [1]
    assert sum(f.affected_count or 0 for f in findings if f.assessed) == 1


@pytest.mark.parametrize("size", [9_999, 10_000, 10_001])
def test_ks_auto_threshold_matches_scipy_on_tied_samples(size, monkeypatch):
    from scipy import stats

    reference = np.arange(size, dtype=np.float64) % 19
    current = np.arange(size, dtype=np.float64) % 23
    expected = stats.ks_2samp(reference, current, method="auto")
    _small_batches(monkeypatch, 509)
    with (
        DataContext(frames={"ref": pl.DataFrame({"x": reference}),
                            "cur": pl.DataFrame({"x": current})}) as context,
        observed_distribution(context, "ref", "cur", "x", normalize_missing) as distribution,
    ):
        statistic, probability, policy = distribution.ks(stats)
    assert statistic == pytest.approx(expected.statistic, rel=1e-14, abs=1e-15)
    assert probability == pytest.approx(expected.pvalue, rel=1e-12, abs=1e-300)
    assert ("bounded_sample" in policy) == (max(len(reference), len(current)) <= 10_000)


def test_categorical_psi_preserves_rare_and_new_support_across_batches(monkeypatch):
    reference = ["common"] * 999 + ["rare"]
    current = ["common"] * 998 + ["new-a", "new-b"]
    expected = 0.0
    left, right = Counter(reference), Counter(current)
    for value in left.keys() | right.keys():
        p, q = max(left[value] / len(reference), 1e-6), max(right[value] / len(current), 1e-6)
        expected += (q - p) * math.log(q / p)
    _small_batches(monkeypatch, 31)
    with (
        DataContext(frames={"ref": pl.DataFrame({"x": reference}),
                            "cur": pl.DataFrame({"x": current})}) as context,
        observed_distribution(context, "ref", "cur", "x", normalize_missing) as distribution,
    ):
        psi, count, policy = distribution.psi(10)
    assert psi == pytest.approx(expected)
    assert count == 4 and policy == "distinct_values"


def test_dynamic_range_failure_is_unassessed_and_other_groups_still_run():
    frame = pl.DataFrame({"g": ["wide"] * 8 + ["good"] * 8,
                          "x": [1e-300, 2e-300] * 3 + [1e-300, 1e300]
                          + [1.0] * 7 + [100.0]})
    with DataContext(frames={"data": frame}) as context:
        findings = ColumnOutlierRule({"targets": [{
            "dataset": "data", "value": "x", "group_by": ["g"],
        }]}).evaluate(context)
    assert any(not f.assessed and f.affected_count == 8 for f in findings)
    assert any(f.assessed and not f.passed and f.affected_count == 1 for f in findings)


def test_exhausted_index_storage_never_publishes_an_assessed_success():
    with DataContext(
        frames={"data": pl.DataFrame({"x": [1.0] * 15 + [100.0]})},
        spill_max_bytes=8192,
    ) as context:
        findings = ColumnOutlierRule({"targets": [{
            "dataset": "data", "value": "x",
        }]}).evaluate(context)
    assert findings and all(not finding.assessed for finding in findings)
    assert any("spill budget" in finding.message for finding in findings)
