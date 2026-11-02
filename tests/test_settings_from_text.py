"""Unsaved configuration text validates exactly as the file it would become."""

from __future__ import annotations

import subprocess
import sys

import pydantic
import pytest

import qavf
from qavf import load_settings, settings_from_text

TEXT = "project: demo\ndatasets:\n  data:\n    path: data/x.csv\n"


def test_text_matches_the_saved_file(tmp_path):
    path = tmp_path / "qavf.yaml"
    from_text = settings_from_text(TEXT, path)
    path.write_text(TEXT)
    assert from_text.model_dump() == load_settings(path).model_dump()
    assert from_text.base_dir == tmp_path.resolve()


def test_text_needs_no_file_and_keeps_its_errors(tmp_path):
    missing = tmp_path / "unsaved" / "qavf.yaml"
    with pytest.raises(ValueError, match="duplicate key"):
        settings_from_text("project: a\nproject: b\n", missing)
    with pytest.raises(pydantic.ValidationError):
        settings_from_text("datasets: 3\n", missing)
    assert not missing.parent.exists()


def test_suffix_selects_json(tmp_path):
    settings = settings_from_text('{"project": "j"}', tmp_path / "qavf.json")
    assert settings.project == "j"


def test_package_runs_as_a_module():
    out = subprocess.run(
        [sys.executable, "-m", "qavf", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"qavf {qavf.__version__}"
