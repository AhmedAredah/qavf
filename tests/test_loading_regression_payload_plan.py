"""Physical placement of exact unchanged-payload projection before native sorting."""

from __future__ import annotations

import json

import polars as pl

from qavf import ResourceLimits
from qavf.engine.sql_read import SqlReadSession
from qavf.regression import compare_outputs


def nodes_with_ancestors(nodes, ancestors=()):
    for node in nodes:
        yield node, ancestors
        yield from nodes_with_ancestors(node.get("children", []), (*ancestors, node))


def test_unchanged_payload_projection_is_physically_below_order_by(tmp_path, monkeypatch):
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    pl.DataFrame(
        {"id": [1, 2], "text": ["wide" * 256, "old"], "blob": [b"x" * 1024, b"old"]}
    ).write_parquet(left)
    pl.DataFrame(
        {"id": [2, 1], "text": ["new", "wide" * 256], "blob": [b"new", b"x" * 1024]}
    ).write_parquet(right)
    original = SqlReadSession.batches
    plans, pairs = [], []

    def inspect(self, query, batch_rows=65536):
        # Explain the actual comparison query on its owned, already-bound
        # connection. EXPLAIN is a test observation, not a production SQL API.
        explained = self._connection.execute("EXPLAIN (FORMAT JSON) " + query).fetchone()
        plans.append(json.loads(explained[1]))
        for batch in original(self, query, batch_rows):
            pairs.extend(batch.select("b0", "b1", "b2", "c1", "c2").rows())
            yield batch

    monkeypatch.setattr(SqlReadSession, "batches", inspect)
    result = compare_outputs(left, right, "id", resources=ResourceLimits(native_threads=1))
    assert result.execution == "native_stream" and result.values_changed == 2
    assert pairs == [(1, "", b"", "", b""), (2, "old", b"old", "new", b"new")]
    [plan] = plans
    nodes = list(nodes_with_ancestors(plan))
    order_nodes = [node for node, _ in nodes if node["name"] == "ORDER_BY"]
    assert len(order_nodes) == 1
    order = order_nodes[0]
    aliases = {"b1", "b2", "c1", "c2"}
    payload_projections = [
        (node, ancestors)
        for node, ancestors in nodes
        if node["name"] == "PROJECTION"
        and aliases.issubset(node.get("extra_info", {}).get("Projections", []))
    ]
    assert payload_projections, "physical plan must expose the transformed output projection"
    assert all(
        any(ancestor is order for ancestor in ancestors) for _, ancestors in payload_projections
    ), "the collapsed payloads must feed ORDER_BY, not be computed after sorting"
    assert any(
        node["name"] == "HASH_JOIN" and any(ancestor is order for ancestor in ancestors)
        for node, ancestors in nodes
    ), "the original-payload join remains below the final ordering"
