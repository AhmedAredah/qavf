"""Comparison levels against independent oracles (plain Python over the inputs)."""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest

from qavf import Dataset, ResourceLimits, compare_datasets


def _check(result, level, check):
    return next(c for c in result.level(level).checks if c.check == check)


# --------------------------------------------------------------- structure
def test_structure_reports_columns_types_units_nullability_rows_and_keys():
    baseline = Dataset(
        pl.DataFrame({"k": [1, 1, 2], "mass": [1, 2, 3], "code": ["a", "b", "c"], "gone": [0] * 3}),
        schema={"columns": {"mass": {"unit": "kg"}}},
    )
    candidate = Dataset(
        pl.DataFrame(
            {"k": [1, 2, 3, 4], "mass": [1, 2, None, 4], "code": [1, 2, 3, 4], "new": [0] * 4}
        ),
        schema={"columns": {"mass": {"unit": "g", "nullable": True}}},
    )
    result = compare_datasets(baseline, candidate, keys="k", levels=["structure"])
    structure = result.level("structure")
    assert structure.columns_removed == ["gone"] and structure.columns_added == ["new"]
    assert [(c.column, c.candidate) for c in structure.type_changes] == [("code", "Int64")]
    assert [(u.baseline, u.candidate) for u in structure.unit_changes] == [("kg", "g")]
    (nullability,) = structure.nullability_changes
    assert (nullability.baseline_missing, nullability.candidate_missing) == (0, 1)
    assert (structure.baseline_rows, structure.candidate_rows) == (3, 4)
    assert structure.duplicate_keys == {"baseline": 1, "candidate": 0}
    failed = {check.check for check in structure.checks if not check.passed}
    assert failed == {"columns", "types", "units", "nullability", "rows", "keys"}


def test_structure_tolerances_and_allowances():
    baseline = pl.DataFrame({"k": list(range(100)), "x": [1] * 100})
    candidate = pl.DataFrame({"k": list(range(101)), "x": [1] * 101, "extra": [0] * 101})
    levels = [{"level": "structure", "allow_added": True, "rows": {"rtol": 0.01}}]
    assert compare_datasets(baseline, candidate, keys="k", levels=levels).passed
    strict = compare_datasets(baseline, candidate, keys="k", levels=["structure"])
    assert {c.check for c in strict.level("structure").checks if not c.passed} == {
        "columns",
        "rows",
    }


def test_rename_suggestions_prefer_similar_names_and_values_and_are_never_applied():
    baseline = pl.DataFrame({"k": [1, 2, 3], "aadt_2023": [10, 20, 30], "notes": ["x", "y", "z"]})
    candidate = pl.DataFrame({"k": [1, 2, 3], "aadt2023": [10, 20, 30], "remarks": ["p", "q", "r"]})
    result = compare_datasets(baseline, candidate, keys="k", levels=["structure"])
    structure = result.level("structure")
    best = structure.rename_suggestions[0]
    assert (best.removed, best.added, best.value_similarity) == ("aadt_2023", "aadt2023", 1.0)
    assert best.score > 0.9
    # Suggestions are evidence only: the columns are still removed and added.
    assert (
        "aadt_2023" in structure.columns_removed
        and not _check(result, "structure", "columns").passed
    )
    suggested = {(s.removed, s.added) for s in structure.rename_suggestions}
    assert ("notes", "remarks") not in suggested  # dissimilar names, disjoint values
    limited = compare_datasets(
        baseline, candidate, keys="k", levels=[{"level": "structure", "max_suggestions": 0}]
    )
    assert limited.level("structure").rename_suggestions == []


# ---------------------------------------------------------------- coverage
def test_coverage_counts_one_sided_keys_overall_and_by_group():
    baseline = pl.DataFrame(
        {"site": [1, 2, 3, 4, 5, None], "county": ["a", "a", "b", "b", "b", "b"], "v": [1] * 6}
    )
    candidate = pl.DataFrame(
        {
            "site": pl.Series([1, 2, 6, 7, None], dtype=pl.Int32),
            "county": ["a", "a", "c", "c", "b"],
            "v": [1, None, 1, 1, 1],
        }
    )
    levels = [{"level": "coverage", "group_by": ["county"], "max_null_rate_change": 0.1}]
    result = compare_datasets(baseline, candidate, keys="site", levels=levels)
    coverage = result.level("coverage")
    base_keys, cand_keys = set(baseline["site"].to_list()), set(candidate["site"].to_list())
    assert coverage.only_in_baseline == len(base_keys - cand_keys) == 3
    assert coverage.only_in_candidate == len(cand_keys - base_keys) == 2
    assert coverage.shared_keys == len(base_keys & cand_keys) == 3  # null matches null
    assert coverage.null_key_rows == {"baseline": 1, "candidate": 1}
    keys = _check(result, "coverage", "keys")
    assert not keys.passed and keys.affected_count == 5
    assert [r["site"] for r in keys.affected_records] == [3, 4, 5, 6, 7]
    breakdown = keys.metrics["breakdown"]
    assert breakdown["only_in_baseline"] == {"groups": 1, "largest": [{"county": "b", "keys": 3}]}
    assert breakdown["only_in_candidate"]["largest"] == [{"county": "c", "keys": 2}]
    (rate,) = coverage.null_rates
    assert rate.column == "v" and rate.change == pytest.approx(1 / 5)
    assert not _check(result, "coverage", "null rates").passed
    allowed = compare_datasets(
        baseline,
        candidate,
        keys="site",
        levels=[
            {
                "level": "coverage",
                "allow_added": True,
                "allow_removed": True,
                "max_null_rate_change": 0.25,
            }
        ],
    )
    assert allowed.passed


def test_coverage_refuses_keys_that_cannot_align_exactly():
    baseline = pl.DataFrame({"k": [1.5, 2.0], "v": [1, 2]})
    candidate = pl.DataFrame({"k": [1, 2], "v": [1, 2]})
    result = compare_datasets(baseline, candidate, keys="k", levels=["coverage"])
    (check,) = result.level("coverage").checks
    assert not check.assessed and "cannot be compared exactly" in check.message


# ----------------------------------------------------------------- records
def test_records_classify_rows_and_count_changes_per_column():
    baseline = pl.DataFrame(
        {
            "k": [1, 2, 3, 4, 5, 6],
            "x": [1.0, 2.0, float("nan"), None, 5.0, 6.0],
            "c": ["a", "a", "b", "b", None, "a"],
            "n": [10, 20, 30, 40, 50, 60],
        }
    )
    candidate = pl.DataFrame(
        {
            "k": [1, 2, 3, 4, 5, 7],
            "x": [1.0, 2.5, 3.0, 4.0, None, 7.0],
            "c": ["a", "b", "c", "b", "a", "a"],
            "n": [10, 20, 30, 41, 50, 70],
        }
    )
    result = compare_datasets(baseline, candidate, keys="k", levels=["records"], rtol=0, atol=0)
    records = result.level("records")
    assert (records.rows_removed, records.rows_added, records.matched) == (1, 1, 5)
    columns = {column.column: column for column in records.columns}
    base, cand = baseline.head(5), candidate.head(5)

    def differs(a, b):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return False
        return a != b

    oracle = {
        name: sum(differs(a, b) for a, b in zip(base[name], cand[name], strict=True))
        for name in ("x", "c", "n")
    }
    assert {name: columns[name].changed for name in oracle} == oracle
    rows_changed = sum(any(differs(base[n][i], cand[n][i]) for n in oracle) for i in range(5))
    assert records.rows_changed == rows_changed and records.rows_unchanged == 5 - rows_changed
    assert columns["x"].nan_to_value == 1 and columns["x"].null_to_value == 1
    assert columns["x"].value_to_null == 1 and columns["c"].null_to_value == 1
    transitions = Counter((a, b) for a, b in zip(base["c"], cand["c"], strict=True) if a != b)
    assert {(t.baseline, t.candidate): t.count for t in columns["c"].transitions} == transitions
    assert columns["c"].distinct_transitions == len(transitions)
    assert columns["x"].max_abs_diff == 0.5 and columns["x"].max_rel_diff == 0.25
    assert columns["n"].max_abs_diff == 1.0 and columns["c"].rate == pytest.approx(3 / 5)
    assert not _check(result, "records", "rows").passed
    values = _check(result, "records", "values")
    assert values.affected_count == sum(oracle.values())


def test_records_large_integers_and_decimals_are_exact():
    big = 2**63 - 1
    baseline = pl.DataFrame(
        {
            "k": [1, 2],
            "i": pl.Series([big, -big], dtype=pl.Int64),
            "d": pl.Series(
                [Decimal("12345678901234567890.123456"), Decimal("1")], dtype=pl.Decimal(38, 6)
            ),
        }
    )
    candidate = pl.DataFrame(
        {
            "k": [1, 2],
            "i": pl.Series([big - 1, -big], dtype=pl.Int64),
            "d": pl.Series(
                [Decimal("12345678901234567890.123457"), Decimal("1")], dtype=pl.Decimal(38, 6)
            ),
        }
    )
    result = compare_datasets(baseline, candidate, keys="k", levels=["records"], rtol=0, atol=0)
    columns = {c.column: c for c in result.level("records").columns}
    assert columns["i"].changed == 1 and columns["d"].changed == 1
    assert columns["i"].max_abs_diff == 1.0
    assert columns["d"].max_abs_diff == pytest.approx(1e-6)
    (change, *_) = [
        r for r in _check(result, "records", "values").affected_records if r["column"] == "i"
    ]
    assert (change["baseline"], change["candidate"]) == (big, big - 1)
    # Within a relative tolerance far below binary64 resolution, still a change.
    tight = compare_datasets(baseline, candidate, keys="k", levels=["records"], rtol=1e-30, atol=0)
    assert tight.level("records").values_changed == 2


def test_records_top_changes_are_the_largest_beyond_tolerance_in_order():
    keys = list(range(20))
    baseline = pl.DataFrame({"k": keys, "v": [100.0] * 20})
    deltas = [((k * 7) % 20) * 0.5 for k in keys]
    candidate = pl.DataFrame({"k": keys, "v": [100.0 + d for d in deltas]})
    levels = [{"level": "records", "top_n": 3}]
    result = compare_datasets(baseline, candidate, keys="k", levels=levels, rtol=0, atol=1.0)
    records = _check(result, "records", "values").affected_records
    expected = sorted(
        ((d, k) for k, d in zip(keys, deltas, strict=True) if d > 1.0),
        key=lambda item: (-item[0], item[1]),
    )[:3]
    assert [(r["abs_diff"], r["k"]) for r in records] == expected
    changed = sum(d > 1.0 for d in deltas)
    assert _check(result, "records", "values").affected_count == changed


def test_records_need_unique_keys():
    frame = pl.DataFrame({"k": [1, 1], "v": [1, 2]})
    result = compare_datasets(frame, frame, keys="k", levels=["records"])
    (check,) = result.level("records").checks
    assert not check.assessed and "duplicate key" in check.message


def test_evidence_is_bounded_and_deterministic_while_counts_stay_complete():
    keys = list(range(200))
    baseline = pl.DataFrame({"k": keys, "v": keys, "c": ["a"] * 200})
    candidate = pl.DataFrame({"k": keys[::-1], "v": [k + 1 for k in keys[::-1]], "c": ["b"] * 200})
    limits = ResourceLimits(evidence_rows=4)
    first = compare_datasets(baseline, candidate, keys="k", resources=limits, rtol=0, atol=0)
    second = compare_datasets(
        baseline,
        candidate.sample(fraction=1, shuffle=True, seed=3),
        keys="k",
        resources=limits,
        rtol=0,
        atol=0,
    )
    values = _check(first, "records", "values")
    assert values.affected_count == 400 and len(values.affected_records) <= 4
    assert values.truncated
    assert first.model_dump_json() == second.model_dump_json()


# ------------------------------------------------------------------ groups
def _groups(baseline, candidate, **level):
    levels = [{"level": "groups", **level}]
    return compare_datasets(baseline, candidate, levels=levels)


def test_group_sums_are_exact_for_large_integers_decimals_and_floats():
    big = 2**62
    baseline = pl.DataFrame(
        {
            "g": ["a", "a", "b"],
            "i": [big, big, 1],
            "d": pl.Series([Decimal("0.1"), Decimal("0.2"), Decimal("1")], dtype=pl.Decimal(10, 1)),
            "f": [0.1, 0.2, 1.0],
        }
    )
    candidate = baseline.with_columns(
        pl.Series("i", [big, big + 1, 1]), pl.Series("f", [0.1, 0.2, 1.0])
    )
    measures = ["sum(i)", "sum(d)", "sum(f)", "mean(i)"]
    result = _groups(baseline, candidate, group_by=["g"], measures=measures, rtol=0, atol=0)
    by_label = {m.label: m for m in result.level("groups").measures}
    assert by_label["sum(i)"].material == 1 and by_label["sum(d)"].material == 0
    assert by_label["sum(f)"].material == 0 and by_label["mean(i)"].material == 1
    (record,) = _check(result, "groups", "sum(i)").affected_records
    assert record == {
        "g": "a",
        "measure": "sum(i)",
        "baseline": 2 * big,
        "candidate": 2 * big + 1,
        "diff": 1,
        "rel_diff": 1 / (2 * big),
    }
    (mean,) = _check(result, "groups", "mean(i)").affected_records
    assert Fraction(mean["diff"]).limit_denominator(4) == Fraction(1, 2)


def test_group_materiality_thresholds_follow_their_precedence():
    baseline = pl.DataFrame({"g": ["a", "b"], "v": [100, 100], "w": [100, 100]})
    candidate = pl.DataFrame({"g": ["a", "b"], "v": [101, 110], "w": [101, 110]})
    measures = [
        "sum(v)",
        {"measure": "sum", "column": "v", "name": "loose", "rtol": 0.1},
        {"measure": "sum", "column": "v", "name": "exact", "exact": True},
        "sum(w)",
    ]
    levels = [{"level": "groups", "group_by": ["g"], "measures": measures, "atol": 1}]
    result = compare_datasets(baseline, candidate, levels=levels, tolerances={"w": {"atol": 10}})
    material = {m.label: m.material for m in result.level("groups").measures}
    # Boundary values are not material: |101-100| <= atol 1; |110-100| <= rtol 0.1 * 110.
    assert material == {"sum(v)": 1, "loose": 0, "exact": 2, "sum(w)": 0}


def test_groups_on_one_side_rank_changes_and_share_change():
    baseline = pl.DataFrame({"g": ["a", "b", "c", "d"], "v": [40, 30, 20, 10]})
    candidate = pl.DataFrame({"g": ["a", "b", "c", "e"], "v": [10, 30, 45, 15]})
    result = _groups(
        baseline,
        candidate,
        group_by=["g"],
        measures=[{"measure": "sum", "column": "v", "comparators": ["share_change", "ratio"]}],
        rank={"measure": "sum(v)", "top_k": 2},
    )
    groups = result.level("groups")
    assert (groups.only_in_baseline, groups.only_in_candidate) == (1, 1)
    assert not _check(result, "groups", "groups").passed
    rank = groups.rank
    # Baseline top 2: a, b. Candidate top 2: c, b.
    assert (rank.entered, rank.left, rank.moved) == (1, 1, 0)
    changes = {
        r["g"]: r["change"] for r in _check(result, "groups", "rank: sum(v)").affected_records
    }
    assert changes == {"c": "entered", "a": "left"}
    records = {r["g"]: r for r in _check(result, "groups", "sum(v)").affected_records}
    base_total, cand_total = 40 + 30 + 20, 10 + 30 + 45  # groups present on both sides
    expected = Fraction(45, cand_total) - Fraction(20, base_total)
    assert records["c"]["share_change"] == pytest.approx(float(expected))
    assert records["c"]["ratio"] == pytest.approx(45 / 20)
    assert list(records) == ["a", "c"]  # largest absolute change first
    ignored = _groups(baseline, candidate, group_by=["g"], missing_groups="ignore")
    assert _check(ignored, "groups", "groups").passed


def test_order_statistics_and_counts_per_group():
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    baseline = pl.DataFrame({"g": ["x"] * 8, "v": values, "c": list("aabbccdd")})
    candidate = pl.DataFrame({"g": ["x"] * 8, "v": [v * 2 for v in values], "c": list("aaaaaaaa")})
    measures = [
        "median(v)",
        {"measure": "quantile", "column": "v", "p": 0.25},
        "distinct_count(c)",
        "min(v)",
        "max(v)",
        "count",
        "null_count(v)",
    ]
    result = _groups(baseline, candidate, measures=measures, rtol=0, atol=0)
    records = {c.check: c.affected_records for c in result.level("groups").checks}
    ordered = sorted(values)
    median = Fraction(ordered[3] + ordered[4], 2)
    assert records["median(v)"][0]["baseline"] == float(median)
    assert records["median(v)"][0]["candidate"] == float(2 * median)
    assert records["distinct_count(c)"][0]["baseline"] == 4
    assert records["min(v)"][0]["diff"] == 1 and records["max(v)"][0]["diff"] == 9
    assert records["count"] == [] and records["null_count(v)"] == []


def test_groups_refuse_unusable_columns():
    frame = pl.DataFrame({"g": [1], "s": ["x"]})
    result = _groups(frame, frame, group_by=["g"], measures=["sum(s)"])
    (check,) = result.level("groups").checks
    assert not check.assessed and "numeric" in check.message
    missing = _groups(frame, frame, group_by=["nope"])
    assert not missing.level("groups").checks[0].assessed


# ------------------------------------------------------- empty and filtered
def test_empty_or_filtered_out_inputs_are_not_assessed_unless_allowed():
    frame = pl.DataFrame({"k": [1, 2], "v": [1, 2]})
    empty = frame.clear()
    result = compare_datasets(frame, empty, keys="k")
    assert not result.assessed and "fewer than min_rows" in result.error
    scoped = compare_datasets(frame, frame, keys="k", where="v > 5")
    assert not scoped.assessed
    allowed = compare_datasets(empty, empty, keys="k", min_rows=0)
    assert allowed.passed, allowed.summary()
