"""Missing-value declarations retain their meaning across public input adapters."""

import polars as pl
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from qavf import Status

from .support.adapters import BACKENDS, adapted_frames
from .support.contracts import run_checks
from .support.oracles import MissingPolicy
from .support.strategies import NULLABLE_NUMBER, NULLABLE_TEXT

pytestmark = pytest.mark.assurance


@pytest.mark.parametrize("backend", BACKENDS)
@given(
    rows=st.lists(st.tuples(NULLABLE_NUMBER, NULLABLE_TEXT), max_size=24),
    use_markers=st.booleans(),
    blank_text=st.booleans(),
)
@example(rows=[], use_markers=True, blank_text=True)
@example(
    rows=[(None, None), (float("nan"), " "), (-999.0, "NA"), (-1.0, "a"), (2.0, "b")],
    use_markers=True,
    blank_text=True,
)
def test_missingness_and_bounds_match_python_oracle(backend, rows, use_markers, blank_text):
    numeric = MissingPolicy((-999.0,) if use_markers else ())
    text = MissingPolicy(("NA",) if use_markers else (), blank_text)
    frame = pl.DataFrame(rows, schema={"number": pl.Float64, "text": pl.String}, orient="row")
    columns = {
        "number": {
            "nullable": False,
            "missing_values": list(numeric.markers),
            "checks": [{"name": "nonnegative", "ge": 0}],
        },
        "text": {
            "nullable": False,
            "missing_values": list(text.markers),
            "blank_as_missing": blank_text,
        },
    }
    expected = {
        "data.number: not null": sum(numeric.missing(number) for number, _ in rows),
        "data.text: not null": sum(text.missing(value) for _, value in rows),
        "data.number: nonnegative": sum(
            not numeric.missing(number) and number < 0 for number, _ in rows
        ),
    }
    with adapted_frames({"data": frame}, backend) as inputs:
        results = run_checks(inputs, "INP-003", datasets={"data": {"schema": {"columns": columns}}})
    failures = {result.check: result for result in results if result.status is Status.FAIL}
    assert {name: result.affected_count for name, result in failures.items()} == {
        name: count for name, count in expected.items() if count
    }
    for name, result in failures.items():
        assert result.total_records == len(rows)
        assert result.unit == "rows"
        assert result.affected_rows == expected[name]
        assert len(result.affected_records) == min(4, expected[name])
        for record in result.affected_records:
            if name == "data.number: not null":
                assert numeric.missing(record["number"])
            elif name == "data.text: not null":
                assert text.missing(record["text"])
            else:
                assert not numeric.missing(record["number"]) and record["number"] < 0


@pytest.mark.parametrize("backend", BACKENDS)
@given(values=st.lists(NULLABLE_NUMBER, max_size=24), reverse=st.booleans())
@example(values=[None, float("nan"), -999.0, -1.0, 1.0], reverse=True)
def test_nullable_missing_values_do_not_become_bound_violations(backend, values, reverse):
    policy = MissingPolicy((-999.0,))
    ordered = list(reversed(values)) if reverse else values
    frame = pl.DataFrame({"x": ordered}, schema={"x": pl.Float64})
    with adapted_frames({"data": frame}, backend) as inputs:
        results = run_checks(
            inputs,
            "INP-003",
            datasets={
                "data": {
                    "schema": {
                        "columns": {
                            "x": {
                                "nullable": True,
                                "missing_values": [-999],
                                "ge": 0,
                            }
                        }
                    }
                },
            },
        )
    expected = sum(not policy.missing(value) and value < 0 for value in values)
    failures = [result for result in results if result.status is Status.FAIL]
    assert len(failures) == bool(expected)
    assert sum(result.affected_count for result in failures) == expected
