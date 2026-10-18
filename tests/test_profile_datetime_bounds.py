"""Profiled datetime bounds must contain the physical observations."""
from datetime import UTC, datetime

import polars as pl
import pytest
import yaml

from qavf import validate
from qavf.scaffold import profile_frame


@pytest.mark.parametrize("ticks", [[1, 1001], [-1001, -1], [-1, 1], [1000, 2000]])
@pytest.mark.parametrize("zone", [None, "UTC", "America/New_York"])
def test_nanosecond_profile_rounds_outwards_and_validates(ticks, zone):
    frame = pl.DataFrame({"t": pl.Series(ticks, dtype=pl.Int64).cast(pl.Datetime("ns", zone))})
    schema = profile_frame(frame)
    spec = schema["columns"]["t"]
    for key, expected in (("ge", min(ticks) // 1000), ("le", -(-max(ticks) // 1000))):
        bound = pl.Series([spec[key]]).cast(pl.Datetime("us", zone)).cast(pl.Int64)[0]
        assert bound == expected
    assert ("description" in spec) == any(tick % 1000 for tick in ticks)
    # YAML roundtrip exercises the same representation that qavf init writes.
    restored = yaml.safe_load(yaml.safe_dump(schema))
    report = validate(frame, schema=restored)
    assert not report.failures(), [(r.rule_id, r.message) for r in report.failures()]


def test_exact_microsecond_profile_needs_no_precision_disclosure():
    value = datetime(2026, 1, 1, 12, tzinfo=UTC)
    spec = profile_frame(pl.DataFrame({"t": [value]}))["columns"]["t"]
    assert spec == {"dtype": "datetime", "ge": value, "le": value}


def test_profile_without_bounds_does_not_emit_precision_disclosure():
    frame = pl.DataFrame({"t": pl.Series([1], dtype=pl.Datetime("ns"))})
    assert profile_frame(frame, bounds=False)["columns"]["t"] == {"dtype": "datetime"}
