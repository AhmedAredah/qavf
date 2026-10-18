"""Small exact-value corpus and public-API outcome assertions shared by adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import polars as pl

from qavf import Status, validate
from qavf.reports.models import QAReport


def corpus(clean: bool) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ordinal": [0, 1, 2],
            "id": [2**53 + 1, 2**53 + 2, 2**53 + (3 if clean else 2)],
            "amount": [
                Decimal("12345678901234567890.1234"),
                Decimal("2.0002") if clean else Decimal("-2.0002"),
                Decimal("0.0000") if clean else None,
            ],
            "stamp": [
                datetime(2024, 3, 10, 6, 59, tzinfo=UTC) + timedelta(minutes=i) for i in range(3)
            ],
            "label": ["alpha", "O'Brien", "日本語"],
        },
        schema_overrides={"amount": pl.Decimal(24, 4), "stamp": pl.Datetime("us", "UTC")},
    )


def configuration() -> dict:
    return {
        "datasets": {
            "data": {
                "schema": {
                    "columns": {
                        "id": {"dtype": "int", "nullable": False},
                        "amount": {"dtype": "decimal", "ge": 0, "nullable": False},
                        "stamp": {"dtype": "datetime", "nullable": False},
                        "label": {"dtype": "str", "nullable": False},
                    },
                    "primary_key": ["id"],
                    "min_rows": 3,
                    "max_rows": 3,
                }
            }
        },
        "engine": {"parallel": False, "run_unconfigured_rules": False},
        "plugins": {"entry_points": False},
        "rules": {name: {} for name in ("INP-001", "INP-002", "INP-003", "INP-004")},
    }


def outcomes(report: QAReport) -> list[tuple[str, str, int]]:
    return sorted(
        (result.rule_id, result.status.value, result.affected_count) for result in report.results
    )


def assert_exact_contract(report: QAReport, *, clean: bool) -> None:
    """Check known defects independently, then compare adapter verdicts/counts."""
    reference = validate(corpus(clean), config=configuration())
    failures = sorted(
        (r.rule_id, r.affected_count) for r in report.results if r.status is Status.FAIL
    )
    assert failures == ([] if clean else [("INP-003", 1), ("INP-003", 1), ("INP-004", 1)])
    if not clean:
        duplicate = next(result for result in report.results if result.rule_id == "INP-004")
        assert duplicate.metrics["duplicated_keys"] == 1
        assert duplicate.metrics["rows_involved"] == 2
        assert duplicate.metrics["surplus_rows"] == 1
    assert outcomes(report) == outcomes(reference)
    assert report.exit_code() == (0 if clean else 1)
    assert all(result.status in {Status.PASS, Status.FAIL} for result in report.results)


def assert_unavailable(report: QAReport) -> None:
    assert report.exit_code() == 1
    assert any(result.status is Status.ERROR for result in report.results)
    assert not any(result.status is Status.PASS for result in report.results)
