"""Exercise an installed core wheel with optional dataframe backends absent.

Run ``python -I scripts/verify_core.py`` in a core-only wheel environment.
The checks deliberately remain active under ``python -O``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from decimal import Decimal
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from qavf import BatchSource, Dataset, File, Query, validate


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_dependencies() -> None:
    for name in ("polars", "pandas", "scipy", "pyproj"):
        require(importlib.util.find_spec(name) is None, f"core environment includes {name}")


def check_input_verdicts() -> None:
    schema = {"columns": {"v": {"dtype": "int", "ge": 0}}}
    for values, expected in (([1, 2], "passed"), ([1, -2], "failed")):
        frame = pa.table({"v": values})
        report = validate(frame, schema=schema)
        require(report.verdict() == expected, f"incorrect core verdict: {report.verdict()}")
        require(
            all(r.check_id and r.identity.rule_id == r.rule_id for r in report.results),
            "missing check identity",
        )
    with TemporaryDirectory() as directory:
        csv = Path(directory) / "data.csv"
        csv.write_text("v\n1\n2\n", encoding="utf-8")
        parquet = Path(directory) / "data.parquet"
        pq.write_table(pa.table({"v": [1, 2]}), parquet)
        for data in (csv, File(csv), parquet, Query("SELECT 1 AS v UNION ALL SELECT 2")):
            require(validate(data, schema=schema).verdict() == "passed", f"input failed: {data}")
        report = validate({"example": Dataset(csv, schema=schema)})
        require(report.verdict() == "passed", "Dataset descriptor failed")
        output = Path(directory) / "report.json"
        report.write_json(output)
        require(output.is_file(), "JSON report was not written")


def check_key_and_na_contracts() -> None:
    report = validate(
        pa.table({"k": ["07", "7", "+7", "7.0", "N/A"]}),
        schema={
            "columns": {
                "k": {"dtype": "int", "number_grammar": {"int_decimal": True}, "na_values": ["N/A"]}
            },
            "primary_key": ["k"],
            "duplicate_rows_per_key": "all",
        },
    )
    failures = [r for r in report.results if r.status.value in ("FAIL", "ERROR")]
    require(len(failures) == 1, "N/A was mistaken for missing or duplicates were lost")
    duplicate = failures[0]
    require(
        duplicate.rule_id == "INP-004"
        and duplicate.affected_count == 1
        and duplicate.affected_rows == 4,
        "incorrect exact duplicate totals",
    )
    require(duplicate.affected_records[0]["rows"] == [0, 1, 2, 3], "incomplete duplicate positions")
    report = validate(
        pa.table({"a": ["N/A", "1"], "b": ["N/A", "N/A"]}),
        schema={
            "columns": {name: {"dtype": "int", "na_values": ["N/A"]} for name in ("a", "b")},
            "na_groups": [["a", "b"]],
        },
    )
    failures = [r for r in report.results if r.rule_id == "INP-009" and r.affected_count]
    require(len(failures) == 1 and failures[0].affected_count == 1, "N/A group rule failed")


def check_optional_capability_error() -> None:
    try:
        validate(
            pa.table({"v": [1]}),
            config={
                "rules": {"STA-002": {"enabled": True}},
            },
        )
    except ImportError as exc:
        require("qavf[polars]" in str(exc), "missing backend error lacks actionable guidance")
    else:
        raise RuntimeError("an unsupported configured rule was silently skipped")


@contextmanager
def _recorded_batches(events):
    """Two Arrow batches whose opening and closing are recorded in events."""
    events.append("open")
    try:
        yield iter([pa.table({"v": [1]}), pa.table({"v": [2]})])
    finally:
        events.append("close")


def check_shared_services() -> None:
    """Managed Arrow sources and acceptance contracts work without dataframe extras."""
    events: list[str] = []
    batches = partial(_recorded_batches, events)
    report = validate(
        BatchSource(batches, pa.schema([("v", pa.int64())])),
        schema={"primary_key": ["v"]},
        config={"engine": {"require": ["INP-004"]}},
    )
    require(
        report.verdict() == "passed" and events == ["open", "close"],
        "managed Arrow batches lost ownership or required-key assessment",
    )
    report = validate(
        pa.table({"v": [1]}),
        config={"engine": {"require": ["INP-004"]}, "rules": {"INP-004": {"enabled": False}}},
    )
    require(report.verdict() == "incomplete", "a disabled required check was accepted")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "source.csv"
        path.write_text("v\n1\n2\n", encoding="utf-8")
        config = {
            "engine": {"execution_mode": "isolated", "run_timeout_s": 20},
            "report": {"hash_inputs": True},
        }
        for source in (path, Query("SELECT 1 AS v UNION ALL SELECT 2")):
            report = validate(source, schema={"primary_key": ["v"]}, config=config)
            require(report.verdict() == "passed", "isolated native input failed")
            require(
                report.metadata.resource_usage["execution_mode"] == "isolated",
                "requested native isolation was not used",
            )
            require(
                bool(report.metadata.config_digest and report.metadata.plan_digest),
                "native provenance digests are missing",
            )
            if source == path:
                fingerprint = report.metadata.datasets[0].fingerprint
                require(
                    len((fingerprint or {}).get("sha256", "")) == 64,
                    "isolated native file lacks its content hash",
                )


def check_temporal_decimal_and_text_policies() -> None:
    """Exact temporal/decimal values and lexical policies keep their declared domain."""
    report = validate(
        pa.table(
            {"v": pa.array([1704067200123456789, 1704067200123456788], type=pa.timestamp("ns"))}
        ),
        schema={
            "columns": {"v": {"dtype": "datetime", "datetime_unit": "ns"}},
            "primary_key": ["v"],
        },
    )
    require(report.verdict() == "passed", "physical Arrow nanosecond identities were rounded")
    report = validate(
        pa.table(
            {
                "v": [
                    "2024-01-01T00:00:00.123456789",
                    "2024-01-01T05:00:00.123456789Z",
                    "2024-01-01T05:00:00.123456788Z",
                ]
            }
        ),
        schema={
            "columns": {
                "v": {"dtype": "datetime", "datetime_unit": "ns", "timezone": "America/New_York"}
            },
            "primary_key": ["v"],
        },
    )
    duplicate = [r for r in report.results if r.rule_id == "INP-004" and r.affected_count]
    require(
        len(duplicate) == 1 and duplicate[0].affected_rows == 2,
        "timezone conversion lost nanosecond key equality",
    )
    report = validate(
        pa.table(
            {
                "v": pa.array(
                    [Decimal("99.990"), Decimal("100.000"), Decimal("1.234")],
                    type=pa.decimal128(6, 3),
                )
            }
        ),
        schema={"columns": {"v": {"dtype": "decimal", "decimal_precision": 4, "decimal_scale": 2}}},
    )
    require(
        sum(r.affected_count for r in report.results if r.rule_id == "INP-002") == 2,
        "decimal precision/scale rounded an invalid value",
    )
    report = validate(
        pa.table({"v": ["1.234,50", "1234,500", "1a234,50", "12.34,50"]}),
        schema={
            "columns": {
                "v": {
                    "dtype": "decimal",
                    "decimal_precision": 8,
                    "decimal_scale": 2,
                    "number_grammar": {"decimal_separator": ",", "grouping_separator": "."},
                }
            }
        },
    )
    require(
        sum(r.affected_count for r in report.results if r.rule_id == "INP-002") == 2,
        "literal locale separators or group widths were misinterpreted",
    )
    report = validate(
        pa.table({"v": ["Straße", "cafe\u0301"]}),
        schema={
            "columns": {
                "v": {
                    "dtype": "str",
                    "normalize": ["casefold", "NFC"],
                    "isin": ["STRASSE", "café"],
                    "each_allowed_once": True,
                }
            }
        },
    )
    require(report.verdict() == "passed", "Unicode normalization differs for values and options")


def check_grouped_and_conditional_contracts() -> None:
    report = validate(
        pa.table({"a": ["07", "7", "7"], "b": ["x", "x", "x"], "active": [True, False, False]}),
        schema={
            "columns": {"a": {"dtype": "int"}},
            "unique_keys": [{"columns": ["a", "b"], "where": "active"}],
        },
    )
    require(report.verdict() == "passed", "conditional composite key included excluded rows")
    report = validate(
        pa.table({"group": ["a", "a", "b", "b", "b"], "code": ["07", "8", "7", "7", "9"]}),
        schema={
            "columns": {"code": {"dtype": "int"}},
            "grouped_values": [{"column": "code", "group_by": ["group"], "allowed_values": [7, 8]}],
        },
    )
    failures = [r for r in report.results if r.rule_id == "INP-004" and r.affected_count]
    require(
        len(failures) == 1 and failures[0].affected_count == 3,
        "group coverage lost missing, duplicate or unexpected members",
    )


def check_cli() -> None:
    """Use fresh installed-package processes: no import blockers or mocks."""
    for arguments in (("--help",), ("--version",), ("run", "--help"), ("capabilities",)):
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "qavf.cli", *arguments],
            text=True,
            capture_output=True,
            timeout=30,
        )
        require(completed.returncode == 0, completed.stderr)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        schema_path = root / "schema.json"
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "qavf.cli", "schema", "-o", str(schema_path)],
            text=True,
            capture_output=True,
            timeout=30,
        )
        require(completed.returncode == 0, completed.stderr)
        require(
            json.loads(schema_path.read_text())["x-qavf-backend"] == "duckdb",
            "core CLI did not generate a native schema",
        )
        (root / "source.csv").write_text("v\n1\n2\n", encoding="utf-8")
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "datasets": {
                        "data": {
                            "path": "source.csv",
                            "schema": {
                                "columns": {"v": {"dtype": "int", "ge": 0}},
                            },
                        }
                    },
                    "report": {"output_dir": "reports"},
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-m", "qavf.cli", "run", "-c", str(config)],
            text=True,
            capture_output=True,
            timeout=60,
        )
        require(completed.returncode == 0, completed.stderr)
        require("QA PASSED" in completed.stdout, "native CLI did not produce a verdict")
        require((root / "reports" / "qa_report.json").is_file(), "CLI JSON is missing")
        require((root / "reports" / "qa_report.html").is_file(), "CLI HTML is missing")


def main() -> None:
    check_dependencies()
    check_input_verdicts()
    check_key_and_na_contracts()
    check_optional_capability_error()
    check_shared_services()
    check_temporal_decimal_and_text_policies()
    check_grouped_and_conditional_contracts()
    check_cli()
    require("polars" not in sys.modules, "core validation initialized Polars")
    require("pandas" not in sys.modules, "core validation initialized pandas")
    print(
        "Core-only installed-wheel checks passed: Arrow/files/Query/Dataset, "
        "converted keys, N/A groups, JSON identity, CLI, managed batches, isolation, provenance, "
        "temporal/decimal/Unicode policies, grouped/conditional keys "
        "and optional-capability errors."
    )


if __name__ == "__main__":
    main()
