"""Acceptance tests for accepted baselines with a manifest."""

from __future__ import annotations

import json
import math

import polars as pl
import pytest

from qavf.cli import EXIT_USAGE, main
from qavf.engine.runner import QARunner
from qavf.engine.severity import Status
from qavf.regression import accept_baseline, read_manifest, verify_baseline
from qavf.regression.baselines import manifest_path
from qavf.settings import QAVFSettings
from qavf.testing import discovered_rules

FRAME = pl.DataFrame({"k": [2, 1, 3], "v": [20.0, 10.0, 30.0]})


def _reg001(frames, path, **params):
    spec = {"dataset": "out", "path": str(path), "keys": ["k"]}
    settings = QAVFSettings.model_validate(
        {
            "plugins": {"entry_points": False},
            "engine": {"run_unconfigured_rules": False},
            "datasets": {"out": {}},
            "rules": {"REG-001": {"params": {"baselines": [spec], **params}}},
        }
    )
    return (
        QARunner(settings, discovered_rules(entry_points=False)).run(frames=frames).get("REG-001")
    )


def test_accepting_writes_sorted_data_and_a_manifest(tmp_path) -> None:
    path = tmp_path / "b.csv"
    manifest = accept_baseline(FRAME, path, ["k"], reason="reviewed in #42", source="run 7")
    assert pl.read_csv(path)["k"].to_list() == [1, 2, 3]
    stored = json.loads(manifest_path(path).read_text(encoding="utf-8"))
    assert stored["keys"] == ["k"] and stored["rows"] == 3 and stored["reason"] == "reviewed in #42"
    assert stored["columns"] == {"k": "Int64", "v": "Float64"} and len(stored["sha256"]) == 64
    assert read_manifest(path) == manifest and verify_baseline(path, ["k"]) == []


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (FRAME.with_columns(pl.lit(1).alias("k")), "duplicated"),
        (pl.DataFrame({"k": [1, None], "v": [1.0, 2.0]}), "missing"),
        (pl.DataFrame({"k": [1.0, math.nan], "v": [1.0, 2.0]}), "missing"),
        (FRAME, "lacks"),
    ],
)
def test_invalid_baselines_are_rejected_before_writing(tmp_path, frame, match) -> None:
    keys = ["k"] if match != "lacks" else ["nope"]
    with pytest.raises(ValueError, match=match):
        accept_baseline(frame, tmp_path / "b.csv", keys)
    assert not (tmp_path / "b.csv").exists()


def test_replacing_an_accepted_baseline_must_be_explicit(tmp_path) -> None:
    path = tmp_path / "b.csv"
    accept_baseline(FRAME, path, ["k"])
    with pytest.raises(FileExistsError):
        accept_baseline(FRAME, path, ["k"])
    accept_baseline(FRAME.with_columns(pl.col("v") * 2), path, ["k"], replace=True)
    assert pl.read_csv(path)["v"].to_list() == [20.0, 40.0, 60.0]


def test_a_changed_baseline_with_an_unchanged_manifest_fails_identity(tmp_path) -> None:
    path = tmp_path / "b.csv"
    accept_baseline(FRAME, path, ["k"], reason="ok")
    assert _reg001({"out": FRAME}, path).status is Status.PASS
    # Edited in place, bypassing acceptance: it is no longer the accepted baseline.
    FRAME.with_columns(pl.col("v") + 1).sort("k").write_csv(path)
    result = _reg001({"out": FRAME.with_columns(pl.col("v") + 1)}, path)
    assert (
        result.status is Status.ERROR and "contents changed since it was accepted" in result.message
    )
    assert verify_baseline(path) != []


def test_comparing_on_other_keys_than_accepted_is_rejected(tmp_path) -> None:
    path = tmp_path / "b.csv"
    accept_baseline(FRAME.with_columns(pl.col("k").alias("j")), path, ["j"])
    result = _reg001({"out": FRAME.with_columns(pl.col("k").alias("j"))}, path)
    assert result.status is Status.ERROR and "accepted with keys" in result.message


def test_manifests_can_be_required(tmp_path) -> None:
    path = tmp_path / "b.csv"
    FRAME.write_csv(path)
    assert _reg001({"out": FRAME}, path).status is Status.PASS  # optional by default
    result = _reg001({"out": FRAME}, path, require_manifest=True)
    assert result.status is Status.ERROR and "no manifest" in result.message


def test_reg001_records_the_acceptance(tmp_path) -> None:
    path = tmp_path / "b.csv"
    accept_baseline(FRAME, path, ["k"], reason="signed off")
    result = _reg001({"out": FRAME}, path)
    manifest = result.metrics["baselines"]["out"]["manifest"]
    assert manifest["reason"] == "signed off" and manifest["keys"] == ["k"]


def test_cli_accepts_refuses_to_overwrite_and_shows_the_replacement_diff(tmp_path, capsys) -> None:
    source = tmp_path / "out.csv"
    FRAME.write_csv(source)
    target = tmp_path / "baseline.csv"
    args = ["baseline", str(source), str(target), "-k", "k", "--reason", "first"]
    assert main(args) == 0 and read_manifest(target).reason == "first"
    assert main(args) == EXIT_USAGE  # exists: needs --force
    assert "--force" in capsys.readouterr().err
    FRAME.with_columns(pl.col("v") + 1).write_csv(source)
    assert main([*args[:-1], "second", "--force"]) == 0
    assert "differs from baseline" in capsys.readouterr().out
    assert read_manifest(target).reason == "second"


def test_reading_a_missing_baseline_creates_no_directory(tmp_path):
    target = tmp_path / "absent" / "counts.csv"
    assert read_manifest(target) is None
    assert verify_baseline(target) == ["it has no manifest"]
    assert not target.parent.exists()
