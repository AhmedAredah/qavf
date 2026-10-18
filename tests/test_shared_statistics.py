"""Independent rule-level contracts for shared coherent statistical observations."""

from contextlib import contextmanager

import polars as pl
import pytest

from qavf.engine.context import DataContext
from qavf.rules.statistical_checks import ColumnOutlierRule, DistributionDriftRule
from qavf.settings import DatasetSpec


def _lazy(frame, calls, *, later=None):
    def observed(_):
        calls.append(1)
        return frame if len(calls) == 1 or later is None else later

    return frame.lazy().map_batches(
        observed,
        schema=dict(frame.schema),
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
    )


def _records(finding):
    if isinstance(finding.affected, pl.DataFrame):
        return finding.affected.to_dicts()
    return list(finding.affected or [])


def _outlier_frame():
    return pl.DataFrame(
        {
            "g": ["one"] * 16,
            "h": ["other"] * 16,
            "x": [1.0] * 15 + [100.0],
            "y": [2.0] * 15 + [200.0],
            "unused": ["PRIVATE_UNUSED_COLUMN"] * 16,
        }
    )


def test_outlier_targets_share_union_once_but_keep_evidence_projection(monkeypatch, tmp_path):
    frame = _outlier_frame()
    later = frame.with_columns(pl.lit(1.0).alias("x"), pl.lit(2.0).alias("y"))
    calls, selections = [], []
    with DataContext(
        frames={"data": _lazy(frame, calls, later=later)},
        temp_directory=tmp_path,
    ) as context:
        original = context.observe

        @contextmanager
        def observe(name, columns=None):
            selections.append((name, None if columns is None else set(columns)))
            with original(name, columns) as result:
                yield result

        monkeypatch.setattr(context, "observe", observe)
        findings = ColumnOutlierRule(
            {
                "targets": [
                    {"dataset": "data", "value": "x", "group_by": ["g"]},
                    {"dataset": "data", "value": "y", "group_by": ["h"]},
                ]
            }
        ).evaluate(context)
        assert calls == [1]
        assert selections == [("data", {"g", "h", "x", "y"})]
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assessed = [finding for finding in findings if finding.assessed]
    assert len(assessed) == 1 and not assessed[0].passed
    assert assessed[0].affected_count == 2
    assert assessed[0].total_records == 32
    records = _records(assessed[0])
    assert len(records) == 2
    by_target = {record["target"]: record for record in records}
    assert "x" in by_target["data.x"] and "g" in by_target["data.x"]
    assert "y" not in by_target["data.x"] and "h" not in by_target["data.x"]
    assert "y" in by_target["data.y"] and "h" in by_target["data.y"]
    assert "x" not in by_target["data.y"] and "g" not in by_target["data.y"]
    assert all("unused" not in record for record in records)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_first", [True, False])
@pytest.mark.parametrize("defect", ["missing_column", "bad_marker"])
def test_bad_outlier_sibling_does_not_poison_union_or_good_target(tmp_path, bad_first, defect):
    frame, calls = _outlier_frame(), []
    bad = "absent" if defect == "missing_column" else "y"
    targets = [{"dataset": "data", "value": value} for value in (bad, "x")]
    if not bad_first:
        targets.reverse()
    specs = {"data": DatasetSpec(schema={"columns": {"y": {"missing_values": ["NA"]}}})}
    with DataContext(
        frames={"data": _lazy(frame, calls)},
        specs=specs,
        temp_directory=tmp_path,
    ) as context:
        findings = ColumnOutlierRule({"targets": targets}).evaluate(context)
        assert calls == [1]
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assert any(not finding.assessed and bad in finding.message for finding in findings)
    assessed = [finding for finding in findings if finding.assessed]
    assert len(assessed) == 1 and not assessed[0].passed
    assert assessed[0].affected_count == 1
    assert assessed[0].metrics["targets"] == ["data.x"]
    assert all("y" not in row and "unused" not in row for row in _records(assessed[0]))
    assert not list(tmp_path.iterdir())


def test_nonadjacent_targets_reuse_each_source_until_last_consumer(tmp_path):
    frame = _outlier_frame()
    calls = {"left": [], "right": []}
    with DataContext(
        frames={name: _lazy(frame, seen) for name, seen in calls.items()},
        temp_directory=tmp_path,
    ) as context:
        findings = ColumnOutlierRule(
            {
                "targets": [
                    {"dataset": "left", "value": "x"},
                    {"dataset": "right", "value": "x"},
                    {"dataset": "left", "value": "y"},
                ]
            }
        ).evaluate(context)
        assert calls == {"left": [1], "right": [1]}
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assert sum(f.affected_count or 0 for f in findings if f.assessed) == 3
    assert not list(tmp_path.iterdir())


def test_drift_comparisons_share_each_source_once_and_preserve_first_data(tmp_path):
    reference = pl.DataFrame(
        {"x": [1.0] * 16, "y": [2.0] * 16, "unused": ["PRIVATE_UNUSED_COLUMN"] * 16}
    )
    current = reference.with_columns(pl.lit(3.0).alias("x"), pl.lit(4.0).alias("y"))
    calls = {"reference": [], "current": []}
    with DataContext(
        frames={
            "reference": _lazy(reference, calls["reference"]),
            "current": _lazy(current, calls["current"], later=reference),
        },
        temp_directory=tmp_path,
    ) as context:
        findings = DistributionDriftRule(
            {
                "comparisons": [
                    {
                        "dataset": "current",
                        "reference": "reference",
                        "column": column,
                        "name": column,
                        "method": "psi",
                        "min_samples": 2,
                    }
                    for column in ("x", "y")
                ]
            }
        ).evaluate(context)
        assert calls == {"reference": [1], "current": [1]}
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assert {f.check for f in findings} == {"x", "y"}
    assert all(f.assessed and not f.passed and f.metrics["psi"] > 0.2 for f in findings)
    assert all("unused" not in row for f in findings for row in _records(f))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("defect", ["missing_column", "bad_marker"])
def test_drift_bad_sibling_is_isolated_and_all_observations_close(tmp_path, defect):
    reference = pl.DataFrame({"x": [1.0] * 16, "y": [2.0] * 16})
    current = reference.with_columns(pl.lit(3.0).alias("x"))
    calls = {"reference": [], "current": []}
    bad = "absent" if defect == "missing_column" else "y"
    specs = {"current": DatasetSpec(schema={"columns": {"y": {"missing_values": ["NA"]}}})}
    with DataContext(
        frames={
            "reference": _lazy(reference, calls["reference"]),
            "current": _lazy(current, calls["current"]),
        },
        specs=specs,
        temp_directory=tmp_path,
    ) as context:
        findings = DistributionDriftRule(
            {
                "comparisons": [
                    {
                        "dataset": "current",
                        "reference": "reference",
                        "column": column,
                        "name": column,
                        "method": "psi",
                        "min_samples": 2,
                    }
                    for column in (bad, "x")
                ]
            }
        ).evaluate(context)
        assert calls == {"reference": [1], "current": [1]}
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assert any(f.check == bad and not f.assessed for f in findings)
    good = [f for f in findings if f.check == "x"]
    assert len(good) == 1 and good[0].assessed and not good[0].passed
    assert not list(tmp_path.iterdir())


def test_failed_shared_observation_is_not_replayed_and_unrelated_source_runs(tmp_path):
    calls = []

    def broken(_):
        calls.append(1)
        raise RuntimeError("cannot observe source")

    frame = _outlier_frame()
    bad = frame.lazy().map_batches(
        broken,
        schema=dict(frame.schema),
        projection_pushdown=False,
    )
    with DataContext(
        frames={"bad": bad, "good": frame.lazy()},
        temp_directory=tmp_path,
    ) as context:
        findings = ColumnOutlierRule(
            {
                "targets": [
                    {"dataset": "bad", "value": "x"},
                    {"dataset": "good", "value": "x"},
                    {"dataset": "bad", "value": "y"},
                ]
            }
        ).evaluate(context)
        assert calls == [1]
        assert context.resource_usage()["owned_spill_bytes"] == 0
    assert len([f for f in findings if not f.assessed]) == 2
    assessed = [f for f in findings if f.assessed]
    assert len(assessed) == 1 and assessed[0].affected_count == 1
    assert assessed[0].metrics["targets"] == ["good.x"]
    assert not list(tmp_path.iterdir())
