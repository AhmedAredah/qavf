"""Exact audit values across pandas, NumPy, streaming and model boundaries."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from types import ModuleType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from qavf.audit_codec import (
    canonical_digest,
    decode_audit,
    encode_audit,
    iter_audit_json,
    normalize_evidence,
)
from qavf.reports.models import RuleResult

pytestmark = pytest.mark.assurance


def test_partial_optional_import_does_not_break_unrelated_datetime_subclasses(monkeypatch):
    class CustomDatetime(datetime):
        pass

    value = CustomDatetime(2026, 1, 1, 0, 0, 0, 1)
    monkeypatch.setitem(sys.modules, "pandas", ModuleType("pandas"))
    assert decode_audit(encode_audit(value)) == value


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("ticks", [0, 1, -1, 123_456_789])
@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
def test_pandas_temporals_retain_exact_units_and_ticks(kind, ticks, unit):
    native = np.array(ticks, dtype=f"{kind}[{unit}]")[()]
    value = pd.Timestamp(native) if kind == "datetime64" else pd.Timedelta(native)
    normalized = normalize_evidence(value)
    restored = decode_audit(encode_audit(value))
    streamed = decode_audit(json.loads(b"".join(iter_audit_json(value))))
    for actual in (normalized, restored, streamed):
        assert actual.dtype == native.dtype
        assert actual.astype("int64") == ticks
    assert canonical_digest(value) == canonical_digest(native)


@pytest.mark.parametrize("fold", [0, 1])
def test_pandas_zoned_microseconds_preserve_zone_fold_and_instant(fold):
    original = datetime(
        2026, 11, 1, 1, 30, 0, 123456, tzinfo=ZoneInfo("America/New_York"), fold=fold
    )
    value = pd.Timestamp(original)
    restored = decode_audit(encode_audit(value))
    assert type(restored) is datetime
    assert restored.tzinfo.key == original.tzinfo.key
    assert restored.fold == fold
    assert restored.timestamp() == original.timestamp()
    assert canonical_digest(restored) == canonical_digest(value)


@pytest.mark.parametrize(
    "value,reason",
    [
        (pd.NaT, "NaT is ambiguous"),
        (
            pd.Timestamp("2026-01-01T00:00:00.000000001", tz="America/New_York"),
            "timezone and nanosecond precision",
        ),
    ],
)
def test_unsupported_temporals_fail_before_persisting_lossy_values(value, reason):
    for operation in (
        normalize_evidence,
        encode_audit,
        lambda item: b"".join(iter_audit_json(item)),
        canonical_digest,
    ):
        with pytest.raises(ValueError, match=reason):
            operation({"nested": [value]})


def test_report_snapshot_round_trip_keeps_pandas_nanosecond_evidence():
    values = {"stamp": pd.Timestamp(1), "duration": pd.Timedelta(1, unit="ns")}
    result = RuleResult(
        rule_id="AUDIT-001",
        name="audit",
        description="exact temporal evidence",
        category="assurance",
        status="FAIL",
        severity="ERROR",
        message="one nanosecond matters",
        metrics=values,
        affected_records=[values],
    )
    restored = RuleResult.model_validate_json(result.model_dump_json())
    for payload in (restored.metrics, restored.affected_records[0]):
        assert payload["stamp"].dtype == np.dtype("datetime64[ns]")
        assert payload["duration"].dtype == np.dtype("timedelta64[ns]")
        assert payload["stamp"].astype("int64") == 1
        assert payload["duration"].astype("int64") == 1
        assert canonical_digest(payload) == canonical_digest(values)
