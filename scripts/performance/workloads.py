"""Realistic public validation calls and exact fixture-specific outcome checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from qavf import Status, validate

from .fixtures import FEATURES, FIXTURE_VERSION, TEXT_COLUMNS, content_fingerprint, residue_count
from .models import PreparedWorkload, WorkloadOptions
from .storage import read_json

CASES = ("wide_validation", "relational_validation", "grouped_reconciliation")


def _wide_columns() -> dict[str, Any]:
    return {
        "amount_cents": {"checks": [{"name": "positive", "gt": 0}]},
        "quantity": {
            "nullable": False,
            "missing_values": [-999],
            "checks": [{"name": "nonnegative", "ge": 0}],
        },
        "status": {"checks": [{"name": "category", "isin": ["OPEN", "CLOSED", "HOLD"]}]},
        "customer_code": {"checks": [{"name": "code_format", "regex": "^C[0-9]{6}$"}]},
        **{
            f"measure_{index:02d}": {"checks": [{"name": "bounds", "ge": 0, "lt": 100}]}
            for index in range(FEATURES)
        },
        **{
            f"description_{index:02d}": {"checks": [{"name": "length", "max_length": 180}]}
            for index in range(TEXT_COLUMNS)
        },
    }


def _contracts(case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if case == "wide_validation":
        return {"transactions": {"schema": {"columns": _wide_columns()}}}, {"INP-003": {}}
    if case == "relational_validation":
        return {
            "transactions": {
                "schema": {
                    "primary_key": ["transaction_id"],
                    "columns": {"customer_id": {"missing_values": [-999]}},
                },
                "foreign_keys": [
                    {
                        "columns": ["customer_id"],
                        "ref_dataset": "customers",
                        "cardinality": "many_to_one",
                        "match": "simple",
                    }
                ],
            },
            "customers": {"schema": {"primary_key": ["customer_id"]}},
        }, {"INP-004": {}, "INP-005": {}}
    if case == "grouped_reconciliation":
        return {"transactions": {}, "ledger": {}}, {
            "REC-001": {
                "params": {
                    "reconciliations": [
                        {
                            "name": "grouped cents",
                            "on": ["group_id"],
                            "rtol": 0,
                            "atol": 0,
                            "left": {"dataset": "transactions", "value": "amount_cents"},
                            "right": {"dataset": "ledger", "value": "ledger_cents"},
                        },
                        {
                            "name": "grouped row counts",
                            "on": ["group_id"],
                            "rtol": 0,
                            "atol": 0,
                            "left": {"dataset": "transactions"},
                            "right": {"dataset": "ledger", "value": "row_count"},
                        },
                    ],
                }
            }
        }
    raise ValueError(f"unknown validation workload: {case}")


def build_workload(
    case: str, fixture: Path, work_dir: Path, options: WorkloadOptions
) -> PreparedWorkload:
    """Prepare representations, then expose one public ``validate`` operation.

    Eager input reads are charged to preparation; lazy/native reads occur inside
    the operation. File provenance hashing is only valid for native file inputs.
    """
    fixture = fixture.resolve()
    manifest = read_json(fixture / "manifest.json")
    if manifest["version"] != FIXTURE_VERSION:
        raise ValueError("unsupported validation fixture version")
    if options.backend not in {"native", "lazy", "eager"}:
        raise ValueError(f"unknown validation backend: {options.backend}")
    if options.input_format not in {"csv", "parquet"}:
        raise ValueError(f"unknown input format: {options.input_format}")
    if options.hash_inputs and options.backend != "native":
        raise ValueError("hash_inputs requires native file inputs; frames have no file provenance")
    datasets, rules = _contracts(case)
    inputs: dict[str, Any] = {}
    input_files = []
    expected_fingerprints = {}
    for name, spec in datasets.items():
        path = fixture / options.input_format / name / f"*.{options.input_format}"
        declared = manifest["datasets"][name]
        if options.hash_inputs:
            expected_fingerprints[name] = content_fingerprint(
                fixture, name, options.input_format, declared
            )
        if options.backend == "native":
            inputs[name] = path
            spec["source"] = {
                "type": "file",
                "path": str(path),
                "min_files": declared["partitions"],
            }
            if options.input_format == "csv":
                spec["source"]["options"] = {
                    "header": True,
                    "types": {
                        column: {"Int64": "BIGINT", "Float64": "DOUBLE", "String": "VARCHAR"}[dtype]
                        for column, dtype in declared["schema"].items()
                    },
                }
        else:
            if options.input_format == "parquet":
                plan = pl.scan_parquet(path)
            else:
                types = {"Int64": pl.Int64, "Float64": pl.Float64, "String": pl.String}
                plan = pl.scan_csv(
                    path,
                    schema_overrides={
                        column: types[dtype] for column, dtype in declared["schema"].items()
                    },
                )
            inputs[name] = plan.collect() if options.backend == "eager" else plan
        input_files.extend(declared["files"][options.input_format])
    temporary = work_dir / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    engine = {
        "parallel": False,
        "run_unconfigured_rules": False,
        "native_threads": options.threads,
        "max_workers": 1,
        "max_affected_records": options.evidence_rows,
        "max_evidence_bytes": options.evidence_bytes,
        "temp_directory": temporary,
    }
    for name in ("memory_budget_bytes", "cache_max_bytes", "spill_max_bytes"):
        if (value := getattr(options, name)) is not None:
            engine[name] = value
    config = {
        "project": f"Performance workload: {case}",
        "plugins": {"entry_points": False},
        "datasets": datasets,
        "rules": rules,
        "engine": engine,
        "report": {"hash_inputs": options.hash_inputs},
    }
    return PreparedWorkload(
        run=lambda: validate(inputs, config=config),
        verify=lambda report: verify_outcome(
            case, report, manifest, options.evidence_rows, expected_fingerprints
        ),
        metadata={
            "case": case,
            "input_rows": manifest["rows"],
            "dataset_rows": {name: manifest["datasets"][name]["rows"] for name in datasets},
            "dataset_columns": {
                name: len(manifest["datasets"][name]["schema"]) for name in datasets
            },
            "input_bytes": sum(item["bytes"] for item in input_files),
            "input_files": len(input_files),
            "rule_ids": list(rules),
            "contracts": {"datasets": datasets, "rules": rules},
            "load_boundary": "preparation" if options.backend == "eager" else "operation",
            "hash_provenance": "file content" if options.hash_inputs else "disabled",
        },
    )


def _expected(case: str, manifest: dict[str, Any]) -> tuple[dict[tuple[str, str], dict], set]:
    rows, truth = manifest["rows"], manifest["truth"]
    failures: dict[tuple[str, str], dict] = {}
    passed: set[tuple[str, str]] = set()
    if case == "wide_validation":
        defects = {
            "transactions.quantity: not null": truth["quantity_missing"],
            "transactions.quantity: nonnegative": truth["quantity_negative"],
            "transactions.status: category": truth["invalid_status"],
            "transactions.customer_code: code_format": truth["invalid_code"],
        }
        for label, count in defects.items():
            if count:
                failures["INP-003", label] = {"affected_count": count, "total_records": rows}
            else:
                passed.add(("INP-003", label))
        passed.add(("INP-003", "transactions.amount_cents: positive"))
        passed.update(
            ("INP-003", f"transactions.{column}: not null")
            for column in _wide_columns()
            if column != "quantity"
        )
        passed.update(
            ("INP-003", f"transactions.measure_{index:02d}: {name}")
            for index in range(FEATURES)
            for name in ("bounds", "finite")
        )
        passed.update(
            ("INP-003", f"transactions.description_{index:02d}: length")
            for index in range(TEXT_COLUMNS)
        )
    elif case == "relational_validation":
        duplicate = truth["duplicate_keys"]
        if duplicate:
            failures["INP-004", "transactions"] = {
                "affected_count": duplicate,
                "affected_rows": 2 * duplicate,
                "total_records": rows - duplicate,
            }
        else:
            passed.add(("INP-004", "transactions"))
        passed.add(("INP-004", "customers"))
        failures["INP-005", "transactions(customer_id) -> customers"] = {
            "affected_count": truth["orphan_keys"],
            "affected_rows": truth["orphan_rows"],
            "total_records": truth["foreign_key_distinct_keys"],
        }
    elif case == "grouped_reconciliation":
        for name, key in (
            ("grouped cents", "reconciliation_total_failures"),
            ("grouped row counts", "reconciliation_count_failures"),
        ):
            failures["REC-001", name] = {
                "affected_count": truth[key],
                "total_records": manifest["group_count"] + 1,
            }
    else:
        raise ValueError(f"unknown validation workload: {case}")
    return failures, passed


def _verify_evidence(case: str, result: Any, manifest: dict[str, Any]) -> None:
    """Verify sampled values against injected defects; sample ordering is immaterial."""
    for record in result.affected_records:
        if case == "wide_validation":
            predicates = {
                "transactions.quantity: not null": lambda row: row["quantity"] in (None, -999),
                "transactions.quantity: nonnegative": lambda row: row["quantity"] == -1,
                "transactions.status: category": lambda row: row["status"] == "INVALID",
                "transactions.customer_code: code_format": lambda row: (
                    row["customer_code"] == "broken"
                ),
            }
            valid = predicates[result.check](record)
        elif case == "relational_validation":
            if result.rule_id == "INP-004":
                key = record["transaction_id"]
                valid = (
                    key % 1000 == 998
                    and 0 <= key < manifest["rows"] - 1
                    and record["occurrences"] == 2
                )
            else:
                orphan = record["customer_id"] - 1_000_000_000
                valid = 0 <= orphan < 7 and record["rows"] == residue_count(
                    manifest["rows"], 2 + 1000 * orphan, 7000
                )
        else:
            group = record["group_id"]
            groups = manifest["group_count"]
            if group == groups + 7:
                valid = record["transactions_total"] is None and record["ledger_total"] == 0
            elif 0 <= group < groups:
                source = manifest["truth"]["group_totals"][group]
                value = source["amount_cents" if result.check == "grouped cents" else "row_count"]
                missing = group % 31 == 0
                changed = result.check == "grouped cents" and group % 17 == 0
                target = None if missing else value + changed
                valid = (
                    (missing or changed)
                    and record["transactions_total"] == value
                    and record["ledger_total"] == target
                )
            else:
                valid = False
        if not valid:
            raise AssertionError(f"invalid evidence for {result.label}: {record}")


def verify_outcome(
    case: str,
    report: Any,
    manifest: dict[str, Any],
    evidence_rows: int = 20,
    expected_fingerprints: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject missing checks, incomplete runs, wrong exact counts and invalid samples."""
    failures, passed = _expected(case, manifest)
    observed_failures: set[tuple[str, str]] = set()
    observed_passed: set[tuple[str, str]] = set()
    rule_ids = {key[0] for key in (*failures, *passed)}
    normalized = []
    if report.incomplete:
        raise AssertionError("benchmark validation is incomplete")
    content_identity = {}
    if expected_fingerprints:
        datasets = {dataset.name: dataset for dataset in report.metadata.datasets}
        if (
            len(datasets) != len(report.metadata.datasets)
            or datasets.keys() != expected_fingerprints.keys()
        ):
            raise AssertionError("hashed benchmark dataset coverage differs from fixture")
        for name, expected_fingerprint in expected_fingerprints.items():
            if datasets[name].fingerprint != expected_fingerprint:
                raise AssertionError(f"hashed benchmark content fingerprint differs: {name}")
            content_identity[name] = {
                field: expected_fingerprint[field] for field in ("files", "bytes", "listing_sha256")
            }
    for result in report.results:
        if result.rule_id not in rule_ids:
            raise AssertionError(f"unexpected benchmark rule: {result.rule_id}")
        if result.status not in (Status.PASS, Status.FAIL):
            raise AssertionError(f"unassessed benchmark check: {result.label}: {result.message}")
        if len(result.affected_records) > evidence_rows:
            raise AssertionError(f"evidence bound exceeded: {result.label}")
        if result.status is Status.PASS:
            coverage = result.metrics.get("passed_checks", [])
            if not coverage:
                raise AssertionError(f"passing benchmark result has no coverage: {result.label}")
            for label in coverage:
                key = result.rule_id, label
                if key in observed_passed:
                    raise AssertionError(f"duplicate passing benchmark contract: {key}")
                observed_passed.add(key)
            continue
        key = result.rule_id, result.check
        if key in observed_failures or key not in failures:
            raise AssertionError(f"unexpected or duplicate benchmark failure: {key}")
        observed_failures.add(key)
        expected = failures[key]
        for field, value in expected.items():
            if getattr(result, field) != value:
                raise AssertionError(
                    f"{key}: {field} is {getattr(result, field)!r}, expected {value!r}"
                )
        _verify_evidence(case, result, manifest)
        normalized.append(
            {"rule_id": result.rule_id, "check": result.check, "status": "fail", **expected}
        )
    if observed_failures != failures.keys() or observed_passed != passed:
        raise AssertionError(
            {
                "missing_failures": sorted(failures.keys() - observed_failures),
                "expected_passed": sorted(passed),
                "observed_passed": sorted(observed_passed),
            }
        )
    if case == "relational_validation":
        result = report.get("INP-005")
        expected = {
            "transactions(customer_id) -> customers": manifest["truth"]["null_customer_rows"]
        }
        if result.metrics.get("rows_exempt_null_keys") != expected:
            raise AssertionError("foreign-key missing-value coverage differs from fixture truth")
    return {
        "verified": True,
        "oracle": "independent integer generator arithmetic",
        "case": case,
        "verdict": report.verdict(),
        "failures": sorted(normalized, key=lambda item: (item["rule_id"], item["check"])),
        "passed_checks": [list(key) for key in sorted(passed)],
        "assessed_contracts": len(failures) + len(passed),
        **({"verified_content_identity": content_identity} if content_identity else {}),
    }
