"""Outcome/evidence parity through the real eager, lazy, SQL and fallback paths."""

from decimal import Decimal
from math import nextafter

import polars as pl
import pytest

from qavf.engine.severity import Status
from qavf.testing import rule_results


@pytest.mark.parametrize("kind", ["float", "integer", "decimal", "time", "unicode"])
def test_normalized_column_outcomes_preserve_backend_semantics(tmp_path, kind):
    series, check = {
        "float": (
            pl.Series(
                "x", [None, float("nan"), float("inf"), nextafter(1.0, 0), 1.0, nextafter(1.0, 2)]
            ),
            {"le": 1},
        ),
        "integer": (pl.Series("x", [None, 2**60 - 1, 2**60, 2**60 + 1]), {"le": 2**60}),
        "decimal": (pl.Series("x", [None, Decimal("1.01"), Decimal("1.00")]), {"le": "1.00"}),
        "time": (pl.Series("x", [None, 1, 2, 3], dtype=pl.Datetime("ns")), {"unique": True}),
        "unicode": (pl.Series("x", [None, "é", "e\u0301", "A", " A"]), {"regex": "^[A-Z]$"}),
    }[kind]
    frame = series.to_frame().with_columns(pl.int_range(pl.len()).alias("row"))
    contract = {
        "nullable": True,
        "allow_infinite": True,
        "conditions": [{**check, "name": "edge", "where": "row > 0", "min_applicable_rows": 1}],
    }
    path = tmp_path / "data.parquet"
    frame.write_parquet(path)
    outputs = []
    for backend in ("eager", "lazy", "native"):
        frames = {} if backend == "native" else {"d": frame.lazy() if backend == "lazy" else frame}
        dataset = {"path": str(path)} if backend == "native" else {}
        dataset["schema"] = {"columns": {"x": contract}}
        results = rule_results("INP-003", frames, datasets={"d": dataset})
        assert not any(result.status is Status.ERROR for result in results), results
        outputs.append(
            [result.model_dump(exclude={"execution_time_s", "load_time_s"}) for result in results]
        )
    # Every assessment/evidence field must agree; timing is excluded above.
    assert outputs[0] == outputs[1] == outputs[2]
