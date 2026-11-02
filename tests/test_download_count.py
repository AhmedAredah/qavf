"""The README download badge sums every source and survives an unavailable one."""

from __future__ import annotations

import json

import pytest

from scripts import download_count


def _seven() -> int:
    return 7


def _five() -> int:
    return 5


def _unavailable() -> int:
    raise OSError("service down")


@pytest.mark.parametrize(
    ("count", "label"),
    [(0, "0"), (802, "802"), (1_000, "1k"), (1_260, "1.3k"), (2_500_000, "2.5M")],
)
def test_badge_label_abbreviates_large_counts(count, label):
    assert download_count.badge(count) == {
        "schemaVersion": 1,
        "label": "downloads",
        "message": label,
        "color": "blue",
    }


@pytest.mark.parametrize(("text", "count"), [("18", 18), ("1,204", 1204), ("1.2k", 1200)])
def test_container_counts_are_expanded(text, count):
    assert download_count._expand(text) == count


def test_total_skips_an_unavailable_source(monkeypatch, tmp_path):
    sources = {"a": _seven, "b": _unavailable, "c": _five}
    monkeypatch.setattr(download_count, "SOURCES", sources)
    target = tmp_path / "downloads.json"
    assert download_count.main([str(target)]) == 0
    assert json.loads(target.read_text())["message"] == "12"


def test_no_readable_source_keeps_the_previous_badge(monkeypatch, tmp_path):
    monkeypatch.setattr(download_count, "SOURCES", {"a": _unavailable})
    target = tmp_path / "downloads.json"
    assert download_count.main([str(target)]) == 1
    assert not target.exists()
