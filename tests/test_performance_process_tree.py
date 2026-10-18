"""Tree measurements include descendants and disclose missing observations."""

import subprocess
import sys
from pathlib import Path

from scripts.performance import process_tree


def _stat(pid, parent, pages):
    fields = ["0"] * 22
    fields[0], fields[1], fields[19], fields[21] = "S", str(parent), str(pid * 17), str(pages)
    return f"{pid} (name with ) brackets) " + " ".join(fields)


def test_proc_namespace_ids_and_all_thread_children_are_used(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    monkeypatch.setattr(process_tree.os, "sysconf", lambda key: 4096, raising=False)
    for pid, parent, pages, children in [
        (100, 1, 10, [200]),
        (200, 100, 20, [300]),
        (300, 200, 30, []),
        (400, 100, 40, []),
    ]:
        directory = tmp_path / str(pid)
        (directory / "task" / str(pid)).mkdir(parents=True)
        (directory / "stat").write_text(_stat(pid, parent, pages))
        (directory / "task" / str(pid) / "children").write_text(" ".join(map(str, children)))
    (tmp_path / "self").mkdir()
    (tmp_path / "self/stat").write_text(_stat(100, 1, 10))
    (tmp_path / "100/task/101").mkdir()
    (tmp_path / "100/task/101/children").write_text("400")
    assert process_tree.sample_tree(tmp_path) == process_tree.TreeSample(409600, 4, True)
    (tmp_path / "200/stat").unlink()
    assert process_tree.sample_tree(tmp_path) == process_tree.TreeSample(204800, 2, False)


def test_real_live_child_is_counted_or_platform_is_explicitly_unsupported():
    if sys.platform != "linux":
        assert process_tree.sample_tree() == process_tree.TreeSample(None, None, False)
        return
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; data=bytearray(8<<20); "
            "print('ready', flush=True); sys.stdin.buffer.read(1)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == b"ready\n"
        sample = process_tree.sample_tree()
        assert sample.processes is not None and sample.processes >= 2
        assert sample.rss_bytes is not None and sample.rss_bytes > 8 << 20
    finally:
        child.communicate(b"x", timeout=10)


def test_missing_proc_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    assert process_tree.sample_tree(Path(tmp_path)) == process_tree.TreeSample(None, None, False)


def test_parent_index_fallback_when_kernel_omits_task_children(tmp_path, monkeypatch):
    monkeypatch.setattr(process_tree.sys, "platform", "linux")
    monkeypatch.setattr(process_tree.os, "sysconf", lambda key: 4096, raising=False)
    for pid, parent, pages in [(100, 1, 10), (200, 100, 20), (300, 200, 30), (400, 1, 99)]:
        directory = tmp_path / str(pid)
        (directory / "task" / str(pid)).mkdir(parents=True)
        (directory / "stat").write_text(_stat(pid, parent, pages))
    (tmp_path / "self").mkdir()
    (tmp_path / "self/stat").write_text(_stat(100, 1, 10))
    assert process_tree.sample_tree(tmp_path) == process_tree.TreeSample(245760, 3, True)
