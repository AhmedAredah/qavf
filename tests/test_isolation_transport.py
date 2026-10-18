"""Bounded parent transport and equivalent automatic thread allocation."""

from __future__ import annotations

import gc
import json
import os
import sys
import tracemalloc

import pytest

from qavf import ExecutionLimitError, ResourceLimits, validate
from qavf.audit_codec import AuditSizeError, decode_audit, encode_audit, iter_audit_json
from qavf.engine.thread_budget import connection_threads, startup_threads
from qavf.execution.os_limits import SUPPORTED_PLATFORMS
from qavf.execution.protocol import request_bytes
from qavf.inputs.descriptors import Query
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("length", [0, 1, 2, 3, 12_287, 12_288, 12_289, 100_000])
def test_binary_stream_matches_canonical_encoding_across_chunk_boundaries(length):
    value = {"payload": bytes(index % 256 for index in range(length))}
    expected = json.dumps(encode_audit(value), separators=(",", ":")).encode()
    chunks = list(iter_audit_json(value))
    actual = b"".join(chunks)
    assert actual == expected
    assert decode_audit(json.loads(actual)) == value
    assert max(map(len, chunks)) <= 16_384
    assert b"".join(iter_audit_json(value, max_bytes=len(actual))) == actual
    with pytest.raises(AuditSizeError):
        b"".join(iter_audit_json(value, max_bytes=len(actual) - 1))


def test_oversized_binary_is_rejected_before_base64_expansion(monkeypatch):
    import qavf.audit_codec as codec

    def forbidden(_):
        pytest.fail("oversized binary expanded before admission")

    monkeypatch.setattr(codec.base64, "b64encode", forbidden)
    with pytest.raises(AuditSizeError, match="encoded bytes"):
        list(iter_audit_json({"data": b"x" * 100_000}, max_bytes=1024))


def test_scalar_dense_request_heap_is_proportional_to_encoded_bytes():
    # Track only encoder allocations, excluding caller input and imported libraries.
    settings = QAVFSettings()
    query = Query("SELECT $items AS items", parameters={"items": [0] * 100_000})
    gc.collect()
    tracemalloc.start()
    try:
        result = request_bytes(settings, {"data": query})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(json.loads(result)["queries"]["data"]["parameters"]["items"]) == 100_000
    assert peak < 4 * len(result) + 1_000_000


@pytest.mark.parametrize("cpus,workers", [(1, 1), (2, 2), (8, 8), (16, 4)])
def test_automatic_thread_policies_distinguish_global_and_connection_pools(cpus, workers):
    assert startup_threads(None, cpu_count=cpus) == min(8, cpus)
    assert connection_threads(None, workers, cpu_count=cpus) * workers <= cpus
    assert connection_threads(3, workers, cpu_count=cpus) == 3
    assert startup_threads(3, cpu_count=cpus) == 3


def test_isolation_preserves_automatic_per_connection_thread_sharing(tmp_path, monkeypatch):
    if sys.platform not in SUPPORTED_PLATFORMS:
        with pytest.raises(ExecutionLimitError, match="supports Linux"):
            validate(
                "unused.parquet", schema={"columns": {"x": {"ge": 0}}},
                resources=ResourceLimits(isolate=True),
            )
        return
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    plugin = tmp_path / "thread_probe.py"
    body = (
        "import os, threading\n"
        "from qavf.engine.registry import Rule, Finding\n"
        "os.cpu_count = lambda: 4\n"
        "barrier = threading.Barrier(2)\n"
    )
    for number in range(2):
        body += (
            f"class Probe{number}(Rule):\n"
            f"    rule_id = 'THREAD-{number}'\n"
            "    name = 'Thread policy'\n"
            "    description = 'Observe owned query connection allowance'\n"
            "    category = 'test'\n"
            "    def evaluate(self, ctx):\n"
            "        barrier.wait(timeout=10)\n"
            "        frame = ctx.sql(\"SELECT current_setting('threads') AS n\", datasets=[])\n"
            "        return Finding.ok('observed', total_records=1, affected_count=0, "
            "metrics={'threads': frame['n'][0]})\n"
        )
    plugin.write_text(body)
    config = {
        "datasets": {"data": {"source": {"type": "query", "sql": "SELECT 1 AS x"}}},
        "plugins": {"packages": [], "paths": [str(plugin)], "entry_points": False},
    }
    observed = []
    for isolated in (False, True):
        report = validate(
            config=config,
            resources=ResourceLimits(
                isolate=isolated, wall_time=20 if isolated else None,
                max_workers=2, temp_dir=tmp_path / "temporary",
            ),
        )
        assert report.verdict() == "passed"
        observed.append([result.metrics["threads"] for result in report.results])
    assert observed == [[2, 2], [2, 2]]
