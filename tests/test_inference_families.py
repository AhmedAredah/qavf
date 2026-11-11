"""Inference families across check items (review finding R12).

Items naming one ``inference.family`` have their p-values corrected together, as
one family over every member's tests, so splitting or merging items does not
change the correction. Items without a family keep correcting alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from qavf import explain, validate
from qavf.checks.selection import SelectionError
from qavf.engine.families import FamilyCorrection, FamilyTests, PooledFamily
from qavf.migrate import Migrator
from qavf.optional import OptionalDependencyError
from qavf.redaction import exception_details
from qavf.reports.html import render_html
from qavf.reports.models import QAReport, RuleResult
from qavf.settings import QAVFSettings, load_settings
from qavf.testing import rule_results

pytest.importorskip("scipy")

DRIFT = {"check": "drift", "datasets": ["ref", "cur"], "method": "ks"}


def _drift_frames() -> dict[str, pl.DataFrame]:
    """``x`` (p about 0.006) and ``y`` (p about 0.0003) have drifted: alone each fails at
    alpha 0.01; corrected as a family of two, ``x`` no longer does."""
    rng = np.random.default_rng(4)
    n = 400
    ref = pl.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
    cur = pl.DataFrame({"x": rng.normal(0.3, 1, n), "y": rng.normal(0.2, 1, n)})
    return {"ref": ref, "cur": cur}


def _drift_item(column: str, family: str | None, **settings: Any) -> dict[str, Any]:
    item = {**DRIFT, "datasets": ["ref", "cur"], "name": f"drift {column}", "column": column}
    return item | settings | ({"inference": {"family": family}} if family else {})


def _drift_config(family: str | None, **settings: Any) -> dict[str, Any]:
    return {
        "datasets": {"ref": {}, "cur": {}},
        "checks": [_drift_item(column, family, **settings) for column in ("x", "y")],
        "engine": {"parallel": False},
    }


def _members(report: QAReport, check: str) -> list[RuleResult]:
    return [result for result in report.results if result.check_name == check]


def _verdicts(results: list[RuleResult]) -> list[tuple[str, Any, Any, Any]]:
    """Each comparison's verdict (items are named, the 0.6 rule's are labelled)."""
    return [
        (
            result.status.value,
            result.metrics.get("p_value"),
            result.metrics.get("adjusted_p"),
            result.metrics.get("alpha_bonferroni"),
        )
        for result in results
    ]


@pytest.mark.parametrize("correction", ["bonferroni", "holm", "bh", "by"])
def test_drift_items_in_a_family_match_one_rule_with_both_comparisons(correction: str) -> None:
    frames = _drift_frames()
    report = validate(
        frames, config=_drift_config("drift", correction=correction), backend="polars"
    )
    members = _members(report, "drift")
    comparisons = [
        {"dataset": "cur", "reference": "ref", "column": column, "method": "ks"}
        for column in ("x", "y")
    ]
    pooled = rule_results("STA-002", frames, {"comparisons": comparisons, "correction": correction})
    assert _verdicts(members) == _verdicts(pooled)
    for member, item in zip(members, ("checks[0]", "checks[1]"), strict=True):
        family = member.inference
        assert family is not None
        assert family.family == "drift" and family.method == correction
        assert family.family_size == 2 and family.tests == 1
        assert family.members == ["checks[0]", "checks[1]"] and family.configured_members == 2
        assert family.adjusted == [member.metrics["adjusted_p"]]
        assert member.item_ids == [item] and not family.notes


def test_items_without_a_family_correct_alone() -> None:
    frames = _drift_frames()
    alone = validate(frames, config=_drift_config(None), backend="polars")
    family = validate(frames, config=_drift_config("drift"), backend="polars")
    alone_members, family_members = _members(alone, "drift"), _members(family, "drift")
    assert all(result.inference is None for result in alone_members)
    assert [r.status.value for r in alone_members] == ["FAIL", "FAIL"]
    assert [r.status.value for r in family_members] == ["PASS", "FAIL"]
    for result, column in zip(alone_members, ("x", "y"), strict=True):
        comparison = {"dataset": "cur", "reference": "ref", "column": column, "method": "ks"}
        (single,) = rule_results("STA-002", frames, {"comparisons": [comparison]})
        assert _verdicts([result]) == _verdicts([single])


def test_a_family_is_deterministic() -> None:
    frames = _drift_frames()
    config = _drift_config("drift", correction="holm")
    first, second = (validate(frames, config=config, backend="polars") for _ in range(2))
    for left, right in zip(first.results, second.results, strict=True):
        assert left.status == right.status and left.metrics == right.metrics
        assert left.inference == right.inference


def test_a_parallel_run_corrects_the_same_family() -> None:
    frames = _drift_frames()
    config = _drift_config("drift")
    serial = validate(frames, config=config, backend="polars")
    config["engine"] = {"parallel": True, "max_workers": 2}
    parallel = validate(frames, config=config, backend="polars")
    assert _verdicts(_members(serial, "drift")) == _verdicts(_members(parallel, "drift"))
    timeout = {**config, "engine": {"parallel": False, "rule_timeout_s": 60}}
    timed = validate(frames, config=timeout, backend="polars")
    assert _verdicts(_members(serial, "drift")) == _verdicts(_members(timed, "drift"))


def _trend_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(7)
    years = list(range(2000, 2012))
    rows = []
    for site, slope in (("a", -1.0), ("b", -0.6), ("c", 0.0)):
        for year in years:
            rows.append((site, year, 100 + slope * (year - 2000) + rng.normal(0, 1.5)))
    frame = pl.DataFrame(rows, schema=["site", "year", "v"], orient="row")
    return {"one": frame, "two": frame.with_columns(pl.col("v") * 2)}


def test_trend_items_in_a_family_match_one_rule_with_both_series() -> None:
    frames = _trend_frames()
    settings = {"expect": "increasing", "min_tau": 0.2, "alpha": 0.01, "correction": "holm"}
    series = {"value": "v", "time_column": "year", "group_by": ["site"]}
    item = {"check": "trend", "series": series, **settings, "inference": {"family": "t"}}
    config = {
        "datasets": {"one": {"checks": [item]}, "two": {"checks": [item]}},
        "engine": {"parallel": False},
    }
    report = validate(frames, config=config, backend="polars")
    members = _members(report, "trend")
    old = {
        "series": [
            {"dataset": name, **series, "expected_trend": "increasing"} for name in ("one", "two")
        ],
        "min_tau": 0.2,
        "alpha": 0.01,
        "correction": "holm",
    }
    (pooled,) = rule_results("TMP-002", frames, old)
    assert sum(r.affected_count for r in members) == pooled.affected_count
    records = sorted(
        (record["series"], record["site"], record["adjusted_p"])
        for result in members
        for record in result.affected_records
    )
    expected = sorted(
        (record["series"], record["site"], record["adjusted_p"])
        for record in pooled.affected_records
    )
    assert records == expected and records
    assert {r.inference.family_size for r in members if r.inference} == {6}


def test_trend_outliers_pool_every_residual_of_every_member() -> None:
    frame = pl.DataFrame(
        {
            "year": list(range(2000, 2010)) * 2,
            "site": ["a"] * 10 + ["b"] * 10,
            "v": [100 + 2 * i for i in range(10)] + [50 + i for i in range(10)],
        }
    ).with_columns(
        pl.when((pl.col("site") == "a") & (pl.col("year") == 2005))
        .then(pl.col("v") * 1.6)
        .otherwise(pl.col("v"))
        .alias("v")
    )
    series = {"value": "v", "time_column": "year", "group_by": ["site"]}
    item = {"check": "trend_outliers", "series": series, "log_transform": False}
    family = {**item, "inference": {"family": "residuals"}}
    config = {
        "datasets": {"one": {"checks": [family]}, "two": {"checks": [family]}},
        "engine": {"parallel": False},
    }
    report = validate({"one": frame, "two": frame}, config=config, backend="polars")
    members = _members(report, "trend_outliers")
    assert {r.inference.family_size for r in members if r.inference} == {40}
    for result in members:
        assert result.status.value == "FAIL"
        (record,) = result.affected_records
        assert record["year"] == 2005 and record["correction"] == "bonferroni"
        assert record["adjusted_p"] == pytest.approx(min(1.0, record["p_value"] * 40))
    alone = validate(
        {"one": frame},
        config={"datasets": {"one": {"checks": [item]}}, "engine": {"parallel": False}},
        backend="polars",
    )
    (single,) = _members(alone, "trend_outliers")
    assert single.affected_records[0]["adjusted_p"] == pytest.approx(
        min(1.0, single.affected_records[0]["p_value"] * 10)
    )


def _compare_frames() -> dict[str, pl.DataFrame]:
    rng = np.random.default_rng(1)
    groups = ["a", "b"]
    base = pl.DataFrame({"g": np.repeat(groups, 60), "x": rng.normal(100.0, 5.0, 120)})
    moved = base.with_columns(pl.col("x") + pl.Series(np.repeat([3.0, 0.0], 60)))
    return {"v1": base, "v2": moved, "w1": base, "w2": moved}


def _compare_item(datasets: list[str], **inference: Any) -> dict[str, Any]:
    return {
        "check": "compare",
        "datasets": datasets,
        "correction": "bonferroni",
        "levels": [{"level": "groups", "group_by": ["g"], "measures": ["mean(x)"]}],
        **inference,
    }


def test_compare_items_in_a_family_correct_every_group_together() -> None:
    family = {"inference": {"family": "versions"}}
    config = {
        "datasets": {name: {} for name in ("v1", "v2", "w1", "w2")},
        "checks": [_compare_item(["v1", "v2"], **family), _compare_item(["w1", "w2"], **family)],
        "engine": {"parallel": False},
    }
    frames = _compare_frames()
    report = validate(frames, config=config, backend="polars")
    groups = [r for r in _members(report, "compare") if r.check and r.check.endswith("mean(x)")]
    assert len(groups) == 2
    for result in groups:
        assert result.metrics["family_size"] == 4
        assert result.inference is not None and result.inference.family_size == 4
        # Group b moved immaterially: untested, it counts in the family as p = 1.
        assert result.inference.tests == 1
        for record in result.affected_records:
            assert record["adjusted_p"] == pytest.approx(min(1.0, record["p_value"] * 4))
    config["checks"] = [_compare_item(["v1", "v2"])]
    (alone,) = [
        r
        for r in _members(validate(frames, config=config, backend="polars"), "compare")
        if r.check and r.check.endswith("mean(x)")
    ]
    assert alone.metrics["family_size"] == 2 and alone.inference is None


def test_a_correction_mismatch_names_the_member_and_its_line(tmp_path: Path) -> None:
    config = _drift_config("drift")
    config["checks"][1]["correction"] = "holm"
    path = tmp_path / "qa.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError) as caught:
        load_settings(path)
    message = exception_details(caught.value)
    lines = path.read_text(encoding="utf-8").splitlines()
    line = [n for n, text in enumerate(lines, 1) if "family: drift" in text][1]
    assert f"(qa.yaml:{line})" in message and "&" not in "".join(lines)
    assert "checks[1].inference.family: inference family" in message
    assert "inference family 'drift' is corrected with bonferroni (checks[0])" in message
    assert "this item applies holm" in message


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"correction": "none"}, "correction is 'none', which corrects nothing"),
        ({"check": "outliers", "column": "x"}, "the outliers check runs no multiple-testing"),
    ],
)
def test_families_need_a_correction_and_a_check_that_applies_one(
    change: dict[str, Any], message: str
) -> None:
    item = {"check": "trend", "series": {"value": "v", "time_column": "t"}, "expect": "increasing"}
    item = {**item, "inference": {"family": "f"}, "correction": "bh"}
    if "check" in change:
        item = {**change, "inference": {"family": "f"}}
    else:
        item |= change
    with pytest.raises(ValidationError, match=message):
        QAVFSettings.model_validate({"datasets": {"a": {"checks": [item]}}})


def test_row_checks_take_no_family() -> None:
    item = {"check": "expression", "expr": "x > 0", "inference": {"family": "f"}}
    with pytest.raises(ValidationError, match="the expression check runs no statistical test"):
        QAVFSettings.model_validate({"datasets": {"a": {"checks": [item]}}})


def test_compare_with_materiality_alone_cannot_join_a_family() -> None:
    item = {**_compare_item(["v1", "v2"]), "decision": "material", "inference": {"family": "f"}}
    with pytest.raises(ValidationError, match="runs no significance test"):
        QAVFSettings.model_validate({"datasets": {"v1": {}, "v2": {}}, "checks": [item]})


def test_the_native_backend_refuses_a_family() -> None:
    with pytest.raises(OptionalDependencyError, match="requires the optional Polars backend"):
        validate(_drift_frames(), config=_drift_config("drift"), backend="duckdb")


def test_a_partial_selection_corrects_the_selected_members_and_says_so() -> None:
    frames = _drift_frames()
    config = _drift_config("drift")
    report = validate(frames, config=config, backend="polars", only="drift x")
    (member,) = _members(report, "drift")
    family = member.inference
    assert family is not None and family.partial
    assert family.family_size == 1 and family.members == ["checks[0]"]
    assert family.configured_members == 2
    assert family.notes == ["selected 1 of 2 members (not run: checks[1])"]
    assert member.status.value == "FAIL"  # corrected over itself alone
    plan = explain(config=config, only="drift x")
    (planned,) = plan.families
    assert planned.members == ("checks[0]",) and planned.partial
    assert "selected 1 of 2 members (not run: checks[1])" in planned.describe()


def test_a_partial_selection_can_be_refused() -> None:
    config = {**_drift_config("drift"), "run": {"partial_families": "refuse"}}
    with pytest.raises(SelectionError, match="selected 1 of 2 members of inference family"):
        validate(_drift_frames(), config=config, backend="polars", only="drift x")
    with pytest.raises(SelectionError, match="partial_families is refuse"):
        explain(config=config, only="drift y")
    report = validate(_drift_frames(), config=config, backend="polars")
    assert all(r.inference is not None for r in _members(report, "drift"))


def test_explain_lists_families_without_reading_data() -> None:
    plan = explain(config=_drift_config("drift", correction="holm"))
    (family,) = plan.families
    assert family.name == "drift" and family.method == "holm"
    assert family.members == ("checks[0]", "checks[1]") and not family.partial
    assert family.describe() == "drift (holm): checks[0], checks[1]"
    assert not plan.data_read


def test_an_override_can_put_an_item_in_a_family() -> None:
    config = _drift_config(None)
    config["overrides"] = {
        "checks": {
            "drift x": {"inference": {"family": "drift"}},
            "drift y": {"inference": {"family": "drift"}},
        }
    }
    report = validate(_drift_frames(), config=config, backend="polars")
    assert [r.status.value for r in _members(report, "drift")] == ["PASS", "FAIL"]


def test_the_report_keeps_the_family(tmp_path: Path) -> None:
    report = validate(_drift_frames(), config=_drift_config("drift"), backend="polars")
    restored = QAReport.model_validate_json(b"".join(report.iter_json_chunks()))
    assert restored == report
    assert [r.inference for r in _members(restored, "drift")] == [
        r.inference for r in _members(report, "drift")
    ]
    older = report.model_dump(mode="json")
    for result in older["results"]:
        result.pop("inference", None)
    assert all(r.inference is None for r in QAReport.model_validate(older).results)
    html = render_html(report)
    assert "Inference family: drift: bonferroni over 2 test(s) of 2 member(s)" in html


def test_migrate_keeps_a_rule_correction_family() -> None:
    comparisons = [
        {"dataset": "cur", "reference": "ref", "column": column, "method": "ks"}
        for column in ("x", "y")
    ]
    old = {
        "datasets": {"ref": {}, "cur": {}},
        "rules": {"STA-002": {"params": {"comparisons": comparisons, "correction": "holm"}}},
    }
    migration = Migrator().migrate(old)
    items = migration.config["checks"]
    assert [item["inference"] for item in items] == [{"family": "STA-002"}] * 2
    assert any("share inference family 'STA-002'" in note for note in migration.notes)
    frames = _drift_frames()
    report = validate(frames, config=migration.config, backend="polars")
    expected = rule_results("STA-002", frames, old["rules"]["STA-002"]["params"])
    assert _verdicts(_members(report, "drift")) == _verdicts(expected)


@pytest.mark.parametrize(
    "params",
    [
        {"comparisons": [{"dataset": "b", "reference": "a", "column": "x"}]},
        {
            "comparisons": [
                {"dataset": "b", "reference": "a", "column": "x"},
                {"dataset": "b", "reference": "a", "column": "y"},
            ],
            "correction": "none",
        },
    ],
)
def test_migrate_adds_no_family_where_none_was_corrected(params: dict[str, Any]) -> None:
    migration = Migrator().migrate(
        {"datasets": {"a": {}, "b": {}}, "rules": {"STA-002": {"params": params}}}
    )
    assert all("inference" not in item for item in migration.config["checks"])


class _Measured(FamilyTests):
    """Raw tests of a member, for the family correction alone."""

    def __init__(self, p_values: list[float], family_size: int) -> None:
        self._p = np.asarray(p_values, dtype=float)
        self._size = family_size

    @property
    def p_values(self) -> np.ndarray:
        return self._p

    @property
    def family_size(self) -> int:
        return self._size

    def finish(self, ctx: Any, correction: FamilyCorrection) -> list[Any]:
        return []


def test_a_family_counts_untested_hypotheses_and_leaves_out_non_finite_p_values() -> None:
    members = [("a", _Measured([0.01, 0.04], 3)), ("b", _Measured([float("nan"), 0.02], 2))]
    family, corrections = PooledFamily.correct("f", "bonferroni", members)
    assert family.family_size == 4 and family.non_finite == 1
    assert family.members == ("a", "b") and dict(family.tests) == {"a": 2, "b": 2}
    assert corrections["a"].adjusted.tolist() == pytest.approx([0.04, 0.16])
    assert np.isnan(corrections["b"].adjusted[0])
    assert corrections["b"].adjusted[1] == pytest.approx(0.08)
    holm, _ = PooledFamily.correct("f", "holm", [("a", _Measured([], 0))])
    assert holm.family_size == 0 and holm.tests == {"a": 0}
