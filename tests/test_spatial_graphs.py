"""Independent graph semantics, cardinality admission and ownership checks."""

import math
from fractions import Fraction

import duckdb
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qavf._adjacency import CompactAdjacency
from qavf.spatial.extension import register
from qavf.spatial.weights import SpatialWeights, contiguity, from_table, grid_contiguity


def mapping(graph):
    return {i: dict(row) for i, row in graph.neighbours.items()}


@settings(max_examples=35, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 5), st.integers(0, 5), st.integers(1, 100)), max_size=80))
def test_table_duplicates_and_self_pairs_match_exact_oracle(edges):
    expected = {i: {} for i in range(6)}
    for i, j, w in edges:
        if i != j:
            expected[i][j] = expected[i].get(j, Fraction()) + Fraction(w)
    table = pl.DataFrame(
        edges, schema={"from": pl.Int64, "to": pl.Int64, "weight": pl.Float64}, orient="row"
    )
    count = sum(map(len, expected.values()))
    graph = from_table(pl.Series(range(6)), table, max_edges=count, max_work=6 + len(edges))
    assert isinstance(graph.neighbours, CompactAdjacency)
    assert mapping(graph) == expected
    assert graph.counts() == [len(expected[i]) for i in expected]
    assert graph.islands == [i for i in expected if not expected[i]]
    with pytest.raises(ValueError, match="max_work"):
        from_table(pl.Series(range(6)), table, max_work=5 + len(edges))
    if count:
        with pytest.raises(ValueError, match="max_edges"):
            from_table(pl.Series(range(6)), table, max_edges=count - 1)


def test_table_rejects_bad_weights_even_on_ignored_self_edges():
    ids = pl.Series(["a", "b"])
    for value in [None, -1.0, 0.0, math.inf, math.nan]:
        table = pl.DataFrame(
            {"from": ["a"], "to": ["a"], "weight": [value]}, schema_overrides={"weight": pl.Float64}
        )
        with pytest.raises(ValueError, match="finite and positive"):
            from_table(ids, table)
    with pytest.raises(ValueError, match="unique"):
        from_table(
            pl.Series(["a", "a"]), pl.DataFrame({"from": ["a"], "to": ["a"], "weight": [1.0]})
        )


def test_table_sums_duplicates_safely_and_does_not_confuse_user_helper_names():
    table = pl.DataFrame(
        {
            "from": ["a", "a", "a"],
            "to": ["b", "b", "b"],
            "weight": [1e16, 1.0, 1.0],
            "i": [1, 1, 1],
            "j": [0, 0, 0],
        }
    )
    graph = from_table(pl.Series(["a", "b"]), table, max_edges=1, max_work=5)
    assert graph.neighbours[0][1] == float(Fraction(10**16) + 2)
    assert not graph.neighbours[1]
    with pytest.raises(ValueError, match="overflow"):
        from_table(pl.Series(["a", "b"]), table.head(2).with_columns(pl.lit(1e308).alias("weight")))


def test_compact_standardization_does_not_overflow_row_totals():
    adjacency = CompactAdjacency.from_edges(
        4, np.array([0, 0, 1, 2]), np.array([1, 2, 0, 0]), np.array([1e308, 1e308, 0.0, 2.0])
    )
    graph = SpatialWeights(pl.Series(range(4)), adjacency, "oracle").standardized()
    assert mapping(graph) == {0: {1: 0.5, 2: 0.5}, 1: {}, 2: {0: 1.0}, 3: {}}
    assert graph.subset([2, 0]).ids.to_list() == [2, 0]
    assert mapping(graph.subset([2, 0])) == {0: {1: 1.0}, 1: {0: 1.0}}
    assert isinstance(graph.subset([2, 0]).neighbours, CompactAdjacency)
    assert graph.subset([]).n == 0
    for keep in ([0, 0], [-1], [4], [True], [0.5]):
        with pytest.raises(ValueError, match="subset"):
            graph.subset(keep)


@pytest.mark.parametrize(
    "values", [[1.0] + [2.0**-54] * 4096, [1e-300, 1.7e23, 1.7e23], [1e308, 1e308, 1e-15]]
)
def test_standardization_preserves_small_weights_against_fraction_oracle(values):
    total = sum(map(Fraction, values))
    expected = [float(Fraction(value) / total) for value in values]
    n = len(values) + 1
    compact = CompactAdjacency.from_edges(
        n, np.zeros(n - 1, dtype=int), np.arange(1, n), np.array(values)
    )
    for rows in (compact, {0: dict(enumerate(values, 1))}):
        graph = SpatialWeights(pl.Series(range(n)), rows, "oracle").standardized()
        assert list(graph.neighbours[0].values()) == expected


@settings(max_examples=30, deadline=None)
@given(
    st.lists(st.tuples(st.integers(-3, 3), st.integers(-3, 3)), max_size=25),
    st.sampled_from(["queen", "rook"]),
)
def test_grid_with_repeated_positions_matches_pairwise_integer_oracle(positions, rule):
    expected = {i: {} for i in range(len(positions))}
    for i, (x, y) in enumerate(positions):
        for j, (a, b) in enumerate(positions):
            dx, dy = abs(x - a), abs(y - b)
            if 0 < max(dx, dy) <= 1 if rule == "queen" else dx + dy == 1:
                expected[i][j] = 1.0
    edges = sum(map(len, expected.values()))
    work = len(positions) * (9 if rule == "queen" else 5)
    ix = pl.Series([p[0] for p in positions], dtype=pl.Int64)
    iy = pl.Series([p[1] for p in positions], dtype=pl.Int64)
    graph = grid_contiguity(
        pl.Series(range(len(positions))), ix, iy, rule, max_edges=edges, max_work=max(work, 1)
    )
    assert mapping(graph) == expected
    if edges:
        with pytest.raises(ValueError, match="max_edges"):
            grid_contiguity(graph.ids, ix, iy, rule, max_edges=edges - 1)
    if work > 1:
        with pytest.raises(ValueError, match="max_work"):
            grid_contiguity(graph.ids, ix, iy, rule, max_work=work - 1)


def test_grid_int64_edges_do_not_wrap_and_dense_duplicates_refuse_before_expansion(monkeypatch):
    graph = grid_contiguity(
        pl.Series([0, 1]), pl.Series([-(2**63), 2**63 - 1]), pl.Series([0, 0]), "rook"
    )
    assert graph.pairs == 0
    # 1000 members in each adjacent cell would create 2,000,000 directed edges.
    ids = pl.Series(range(2000))
    ix, iy = pl.Series([0] * 1000 + [1] * 1000), pl.Series([0] * 2000)
    original = pl.DataFrame.join
    expansions = []

    def join(self, other, *args, **kwargs):
        if "i" in self.columns and kwargs.get("how") == "cross":
            expansions.append(True)
        return original(self, other, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "join", join)
    with pytest.raises(ValueError, match="max_edges"):
        grid_contiguity(ids, ix, iy, max_edges=10)
    assert expansions == []


@pytest.fixture
def spatial():
    con = duckdb.connect(config={"threads": 1})
    try:
        con.execute("LOAD spatial")
    except duckdb.Error:
        con.close()
        pytest.skip("spatial extension is not already installed")
    try:
        yield con
    finally:
        con.close()


def boxes(con, bounds):
    frame = pl.DataFrame(
        {
            "zone_order": range(len(bounds)),
            "wkt": [
                f"POLYGON (({a} {b},{c} {b},{c} {d},{a} {d},{a} {b}))" for a, b, c, d in bounds
            ],
            "xmin": [float(v[0]) for v in bounds],
            "ymin": [float(v[1]) for v in bounds],
            "xmax": [float(v[2]) for v in bounds],
            "ymax": [float(v[3]) for v in bounds],
        }
    )
    register(con, "input_boxes", frame)
    con.execute(
        "CREATE OR REPLACE TABLE zones AS SELECT * EXCLUDE(wkt), "
        "ST_GeomFromText(wkt) AS geom FROM input_boxes"
    )
    con.unregister("input_boxes")
    return pl.Series(range(len(bounds)))


class ObservedConnection:
    def __init__(self, con):
        self.con, self.queries = con, []

    def execute(self, query):
        self.queries.append(query)
        return self.con.execute(query)

    def __getattr__(self, key):
        return getattr(self.con, key)


def test_polygon_work_gate_precedes_geometry_and_induced_graph_admits_only_selected(spatial):
    ids = boxes(spatial, [(0, 0, 1, 1)] * 100)
    observed = ObservedConnection(spatial)
    with pytest.raises(ValueError, match="max_work"):
        contiguity(observed, "zones", ids, max_work=101)
    assert len(observed.queries) == 1
    assert "ST_Intersects" not in observed.queries[0]
    graph = contiguity(observed, "zones", ids, keep=[99, 0], max_work=3, max_edges=2)
    assert graph.ids.to_list() == [99, 0]
    assert mapping(graph) == {0: {1: 1.0}, 1: {0: 1.0}}
    with pytest.raises(ValueError, match="max_edges"):
        contiguity(observed, "zones", ids, keep=[99, 0], max_work=3, max_edges=1)
    assert "LIMIT 1" in observed.queries[-1]
    assert not any(
        "__qavf_neighbours_" in row[0] for row in spatial.execute("SHOW TABLES").fetchall()
    )


@pytest.mark.parametrize("rule", ["queen", "rook"])
def test_polygon_graph_matches_rectangle_oracle_and_subset_order(spatial, rule):
    bounds = [(0, 0, 2, 2), (2, 0, 3, 1), (2, 2, 3, 3), (0.5, 0.5, 1, 1), (8, 8, 9, 9)]
    ids = boxes(spatial, bounds)
    expected = {i: {} for i in range(len(bounds))}
    for i, (a, b, c, d) in enumerate(bounds):
        for j, (e, f, g, h) in enumerate(bounds):
            dx, dy = min(c, g) - max(a, e), min(d, h) - max(b, f)
            if i != j and dx >= 0 and dy >= 0 and (rule == "queen" or dx > 0 or dy > 0):
                expected[i][j] = 1.0
    full = contiguity(spatial, "zones", ids, rule)
    assert mapping(full) == expected
    narrow = contiguity(spatial, "zones", ids, rule, keep=[4, 2, 0])
    assert mapping(narrow) == mapping(full.subset([4, 2, 0]))


def test_polygon_cancellation_releases_subset_registration_and_reader(spatial):
    ids = boxes(spatial, [(0, 0, 1, 1)] * 3)
    calls = 0

    def cancel_at_batch():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("cancel graph")

    with pytest.raises(RuntimeError, match="cancel graph"):
        contiguity(spatial, "zones", ids, keep=[2, 0], check_cancelled=cancel_at_batch)
    assert spatial.execute("SELECT 42").fetchone() == (42,)
    assert not any(
        "__qavf_neighbours_" in row[0] for row in spatial.execute("SHOW TABLES").fetchall()
    )


@pytest.mark.parametrize("work,edges", [(True, 10), (1.5, 10), (0, 10), (10, True), (10, -1)])
def test_direct_builders_validate_limits(work, edges):
    ids = pl.Series([0])
    with pytest.raises(ValueError, match="max_"):
        grid_contiguity(ids, ids, ids, max_work=work, max_edges=edges)
    with pytest.raises(ValueError, match="max_"):
        from_table(
            ids,
            pl.DataFrame({"from": [0], "to": [0], "weight": [1.0]}),
            max_work=work,
            max_edges=edges,
        )
