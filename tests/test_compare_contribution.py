"""The contribution level and the exact decompositions it rests on.

Contributions sum exactly to the total change; the minimal set is checked
against a brute-force search; the mix-versus-rate split is checked on a
hand-worked two-group example and, for random data, by the exact identity
``mix + rate + interaction == R_c - R_b`` (a hypothesis property).
"""

from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction
from itertools import combinations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from qavf import compare_datasets
from qavf.compare import DatasetComparison
from qavf.statistics.decomposition import contributions, minimal_set, rate_decomposition

F = Fraction


# -------------------------------------------------------------- the kernels
def test_contributions_sum_exactly_to_the_total_change():
    members = [("a", F(10), F(13)), ("b", F(5), None), ("c", None, F(1, 3)), ("d", F(2), F(2))]
    ranked = contributions(members)
    assert [c.key for c in ranked] == ["b", "a", "c", "d"]  # largest magnitude first
    assert sum((c.change for c in ranked), F(0)) == (F(13) + F(1, 3) + F(2)) - F(17)
    assert ranked[0].present == (True, False) and ranked[0].change == -5


def _brute_force(changes: list[Fraction], share: Fraction) -> int | None:
    total = sum(changes, F(0))
    if not total:
        return None
    for size in range(1, len(changes) + 1):
        for subset in combinations(changes, size):
            if (sum(subset, F(0)) / total) >= share:
                return size
    raise AssertionError("unreachable: the full set explains everything")


@given(
    st.lists(st.integers(-50, 50), min_size=1, max_size=7),
    st.sampled_from([F(1, 2), F(4, 5), F(9, 10), F(1)]),
)
@settings(max_examples=150, deadline=None)
def test_the_minimal_set_is_as_small_as_a_brute_force_search(values, share):
    changes = [F(v) for v in values]
    picked = minimal_set(changes, share)
    expected = _brute_force(changes, share)
    if expected is None:
        assert picked is None
        return
    assert picked is not None and len(picked) == expected
    total = sum(changes, F(0))
    assert sum((changes[i] for i in picked), F(0)) / total >= share


def test_a_hand_worked_two_group_rate_change():
    # Baseline: A 10/100 (rate 0.1), B 40/100 (rate 0.4): R = 50/200 = 0.25.
    # Candidate: A 60/300 (rate 0.2), B 40/100 (rate 0.4): R = 100/400 = 0.25.
    # Weights 1/2, 1/2 -> 3/4, 1/4. Mix: 1/4 * 0.1 - 1/4 * 0.4 = -0.075; rate:
    # 1/2 * 0.1 = 0.05; interaction: 1/4 * 0.1 = 0.025; together 0, the change.
    strata = [("A", F(10), F(100), F(60), F(300)), ("B", F(40), F(100), F(40), F(100))]
    split = rate_decomposition(strata)
    assert (split.baseline_rate, split.candidate_rate) == (F(1, 4), F(1, 4))
    assert (split.mix, split.rate, split.interaction) == (F(-3, 40), F(1, 20), F(1, 40))
    symmetric = rate_decomposition(strata, "symmetric")
    # Kitagawa: mix 1/4 * 0.15 - 1/4 * 0.4 = -1/16; rate 0.1 * (1/2 + 3/4) / 2 = 1/16.
    assert (symmetric.mix, symmetric.rate, symmetric.interaction) == (F(-1, 16), F(1, 16), 0)
    # A pure composition shift is all mix; a pure rate change all rate.
    mix_only = rate_decomposition([("A", F(10), F(100), F(30), F(300)), strata[1]])
    assert (mix_only.mix, mix_only.rate, mix_only.interaction) == (F(-3, 40), 0, 0)
    rate_only = rate_decomposition([("A", F(10), F(100), F(20), F(100)), strata[1]])
    assert (rate_only.mix, rate_only.rate, rate_only.interaction) == (0, F(1, 20), 0)


@given(
    st.lists(
        st.tuples(*[st.integers(0, 1_000)] * 4).filter(lambda t: t[1] > 0 or t[3] > 0),
        min_size=1,
        max_size=8,
    ),
    st.sampled_from(["three_term", "symmetric"]),
)
@settings(max_examples=200, deadline=None)
def test_mix_rate_and_interaction_sum_exactly_to_the_change(rows, method):
    strata = []
    for index, (nb, db, nc, dc) in enumerate(rows):
        # A side without denominator has no numerator either (an absent stratum).
        strata.append((index, F(nb if db else 0), F(db), F(nc if dc else 0), F(dc)))
    if not sum(s[2] for s in strata) or not sum(s[4] for s in strata):
        with pytest.raises(ValueError, match="sum to zero"):
            rate_decomposition(strata, method)
        return
    split = rate_decomposition(strata, method)
    totals = [sum((s[i] for s in strata), F(0)) for i in (1, 2, 3, 4)]
    change = totals[2] / totals[3] - totals[0] / totals[1]
    assert split.mix + split.rate + split.interaction == change == split.change
    assert sum((e.total for e in split.strata), F(0)) == change


def test_invalid_strata_are_refused():
    with pytest.raises(ValueError, match="negative denominator"):
        rate_decomposition([("a", F(1), F(-1), F(1), F(1))])
    with pytest.raises(ValueError, match="numerator but no denominator"):
        rate_decomposition([("a", F(1), F(0), F(1), F(1))])


# ---------------------------------------------------------------- the level
def _crashes(seed: int = 1, boost: float = 1.0):
    rng = np.random.default_rng(seed)
    rows = []
    for zone, share in (("n", 0.5), ("s", 0.3), ("e", 0.2)):
        for day in range(90):
            vmt = int(rng.integers(900, 1100) * share * 10)
            rate = 0.002 * (boost if zone == "n" else 1.0)
            rows.append((zone, date(2024, 1, 1) + timedelta(days=day), vmt,
                         int(rng.poisson(vmt * rate))))  # fmt: skip
    return pl.DataFrame(rows, schema=["zone", "day", "vmt", "fatal"], orient="row")


LEVEL = {
    "level": "contribution",
    "measure": "sum(fatal)",
    "dimensions": [
        "zone",
        {"columns": ["zone"], "time": {"time_column": "day", "bucket": "month"}},
    ],
    "rates": ["fatal / vmt"],
}


def test_the_level_explains_a_planted_regional_change_exactly():
    base, cand = _crashes(1), _crashes(2, boost=2.0)
    result = compare_datasets(base, cand, levels=[LEVEL])
    level = result.level("contribution")
    by_zone = level.dimensions[0]
    total = F(int(cand["fatal"].sum()) - int(base["fatal"].sum()))
    assert F(by_zone.exact_total_change) == total == F(level.total_change)
    assert sum((F(c.exact_change) for c in by_zone.contributions), F(0)) == total
    assert by_zone.contributions[0].group == {"zone": "n"} and by_zone.minimal_set == 1
    assert by_zone.contributions[0].in_minimal_set
    shares = [c.share for c in by_zone.contributions]
    assert sum(shares) == pytest.approx(1.0) and by_zone.contributions[-1].cumulative == 1.0
    zone_month = level.dimensions[1]
    assert zone_month.members == 9 and zone_month.keys == ["zone", "bucket"]
    [rate] = level.rates
    parts = {name: F(value) for name, value in rate.exact.items()}
    assert parts["mix"] + parts["rate"] + parts["interaction"] == parts["change"]
    assert rate.rate > abs(rate.mix)  # the rate rose, the composition barely moved
    checks = {c.check: c for c in level.checks}
    assert not checks["total"].passed and not checks["contribution: zone"].passed
    assert [r["zone"] for r in checks["contribution: zone"].affected_records] == ["n"]
    assert not checks["rate: fatal / vmt"].passed
    assert checks["rate: fatal / vmt"].significance.method == "poisson"
    assert DatasetComparison.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "decision", ["material_and_significant", "material", "significant", "equivalence"]
)
def test_identical_versions_contribute_nothing_under_every_policy(decision):
    base = _crashes(1)
    result = compare_datasets(base, base, levels=[LEVEL], decision=decision)
    level = result.level("contribution")
    assert level.total_change == 0 and level.dimensions[0].minimal_set is None
    rate = level.rates[0]
    assert rate.exact == {"change": "0", "mix": "0", "rate": "0", "interaction": "0"}
    checks = {c.check: c for c in level.checks}
    if decision == "equivalence":
        # Poisson has no equivalence form: the rate check fails closed.
        assert not checks["rate: fatal / vmt"].assessed
    else:
        assert all(c.passed for c in checks.values()), result.summary()


def test_non_additive_measures_and_unknown_dimensions_are_configuration_errors():
    frame = pl.DataFrame({"zone": ["a"], "x": [1.0]})
    with pytest.raises(ValidationError, match="not additive"):
        compare_datasets(frame, frame, levels=[{**LEVEL, "measure": "mean(x)"}])
    with pytest.raises(ValidationError, match="rate dimension"):
        compare_datasets(frame, frame, levels=[{**LEVEL, "rates": [
            {"numerator": "x", "denominator": "x", "dimension": "nowhere"}]}])  # fmt: skip
    missing = compare_datasets(
        frame, frame, levels=[{"level": "contribution", "dimensions": ["zz"]}]
    )
    assert not next(c for c in missing.level("contribution").checks if "zz" in c.check).assessed
