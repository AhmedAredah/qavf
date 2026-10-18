"""Performance evidence must retain exact, independently specified QA coverage."""

import json
from fractions import Fraction

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scripts.performance import fixtures
from scripts.performance.fixtures import group_truth, prepare_fixture
from scripts.performance.models import WorkloadOptions
from scripts.performance.workloads import CASES, build_workload


@pytest.fixture(scope="module")
def transaction_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("transactions")
    # Partial last block and more than 31 groups exercise every injected defect.
    manifest = prepare_fixture(path, rows=4213, seed=2026, partitions=3, chunk_rows=511)
    return path, manifest


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("backend", ["native", "lazy", "eager"])
@pytest.mark.parametrize("input_format", ["parquet", "csv"])
def test_public_validation_preserves_truth_for_all_representations(
    transaction_fixture, tmp_path, case, backend, input_format
):
    path, _ = transaction_fixture
    workload = build_workload(
        case,
        path,
        tmp_path,
        WorkloadOptions(backend=backend, input_format=input_format, evidence_rows=2),
    )
    outcome = workload.verify(workload.run())
    assert outcome["verified"] and outcome["verdict"] == "failed"
    assert (
        outcome["assessed_contracts"]
        == {
            "wide_validation": 40,
            "relational_validation": 3,
            "grouped_reconciliation": 2,
        }[case]
    )
    json.dumps(workload.metadata)
    json.dumps(outcome)


@pytest.mark.parametrize("seed", [0, -12345, 9223372036854775807])
def test_grouped_oracle_matches_small_independent_fraction_reference(seed):
    rows, groups = 3117, 29
    truth = group_truth(rows, seed, groups)
    for group in truth:
        indices = range(group["group_id"], rows, groups)
        expected = sum(
            (Fraction(100 + (37 * index + seed % 100_000) % 100_000) for index in indices),
            Fraction(),
        )
        assert group["amount_cents"] == expected
        assert group["row_count"] == len(indices)


def test_fixture_is_chunk_bounded_and_seeded_without_layout_dependence(tmp_path, monkeypatch):
    original = fixtures._facts
    chunk_sizes = []

    def observed(start, stop, *args):
        chunk_sizes.append(stop - start)
        return original(start, stop, *args)

    monkeypatch.setattr(fixtures, "_facts", observed)
    small = prepare_fixture(tmp_path / "small", rows=209, seed=43, partitions=3, chunk_rows=17)
    assert max(chunk_sizes) <= 17 and len(chunk_sizes) > 3
    large = prepare_fixture(tmp_path / "large", rows=209, seed=43, partitions=7, chunk_rows=500)
    different = prepare_fixture(tmp_path / "different", rows=209, seed=44, chunk_rows=100)
    assert small["truth"] == large["truth"]
    left = pl.read_parquet(tmp_path / "small/parquet/transactions/*.parquet").sort("row_id")
    right = pl.read_parquet(tmp_path / "large/parquet/transactions/*.parquet").sort("row_id")
    assert_frame_equal(left, right)
    assert small["truth"]["group_totals"] != different["truth"]["group_totals"]
    assert sum(file["rows"] for file in small["datasets"]["transactions"]["files"]["csv"]) == 209


@pytest.mark.parametrize("mutation", ["count", "coverage", "sample", "missing_failure"])
def test_oracle_rejects_plausible_but_incorrect_reports(transaction_fixture, tmp_path, mutation):
    path, _ = transaction_fixture
    workload = build_workload("wide_validation", path, tmp_path, WorkloadOptions(evidence_rows=2))
    report = workload.run()
    results = list(report.results)
    if mutation == "count":
        results[0] = results[0].model_copy(update={"affected_count": 0})
    elif mutation == "sample":
        evidence = [{**results[0].affected_records[0], "quantity": 123}]
        results[0] = results[0].model_copy(update={"affected_records": evidence})
    elif mutation == "coverage":
        results[-1] = results[-1].model_copy(update={"metrics": {"passed_checks": []}})
    else:
        results.pop(0)
    with pytest.raises(AssertionError):
        workload.verify(report.model_copy(update={"results": results}))


def test_zero_evidence_keeps_exact_counts_and_file_hashing(transaction_fixture, tmp_path):
    path, manifest = transaction_fixture
    workload = build_workload(
        "relational_validation",
        path,
        tmp_path,
        WorkloadOptions(evidence_rows=0, hash_inputs=True),
    )
    report = workload.run()
    assert workload.verify(report)["verified"]
    assert not any(result.affected_records for result in report.results)
    assert (
        report.get("INP-004", "transactions").affected_count == manifest["truth"]["duplicate_keys"]
    )
    assert all(dataset.fingerprint["identity"] == "content" for dataset in report.metadata.datasets)


def test_grouped_oracle_rejects_a_balanced_group_presented_as_failure(
    transaction_fixture, tmp_path
):
    path, manifest = transaction_fixture
    workload = build_workload("grouped_reconciliation", path, tmp_path, WorkloadOptions())
    report = workload.run()
    results = list(report.results)
    total = manifest["truth"]["group_totals"][1]["amount_cents"]
    results[0] = results[0].model_copy(
        update={
            "affected_records": [
                {
                    "group_id": 1,
                    "transactions_total": total,
                    "ledger_total": total,
                    "problem": "totals differ",
                }
            ]
        }
    )
    with pytest.raises(AssertionError, match="invalid evidence"):
        workload.verify(report.model_copy(update={"results": results}))


def test_small_fixture_with_no_duplicate_still_has_full_coverage(tmp_path):
    prepare_fixture(tmp_path / "fixture", rows=32, seed=1, partitions=40, chunk_rows=200)
    workload = build_workload(
        "relational_validation",
        tmp_path / "fixture",
        tmp_path / "work",
        WorkloadOptions(),
    )
    outcome = workload.verify(workload.run())
    assert ["INP-004", "transactions"] in outcome["passed_checks"]
    assert outcome["assessed_contracts"] == 3


def test_fixture_refuses_overwrite_and_invalid_sizes(tmp_path):
    prepare_fixture(tmp_path, rows=32, seed=1)
    with pytest.raises(ValueError, match="empty"):
        prepare_fixture(tmp_path, rows=32, seed=1)
    with pytest.raises(ValueError, match="at least 32"):
        prepare_fixture(tmp_path / "too_small", rows=1, seed=1)
    with pytest.raises(ValueError, match="chunk_rows"):
        prepare_fixture(tmp_path / "zero_chunk", rows=32, seed=1, chunk_rows=0)


def test_lazy_input_cannot_claim_file_content_verification(transaction_fixture, tmp_path):
    path, _ = transaction_fixture
    with pytest.raises(ValueError, match="requires native file inputs"):
        build_workload(
            "wide_validation",
            path,
            tmp_path,
            WorkloadOptions(backend="lazy", hash_inputs=True),
        )


@pytest.fixture(scope="module")
def hashed_validation(transaction_fixture, tmp_path_factory):
    path, _ = transaction_fixture
    workload = build_workload(
        "wide_validation",
        path,
        tmp_path_factory.mktemp("hashed-run"),
        WorkloadOptions(hash_inputs=True),
    )
    report = workload.run()
    outcome = workload.verify(report)
    identity = outcome["verified_content_identity"]["transactions"]
    assert identity["files"] == 3 and identity["bytes"] > 0
    return workload, report


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity", "metadata"),
        ("listing_sha256", "0" * 64),
        ("files", 1),
        ("bytes", 0),
        ("paths", []),
    ],
)
def test_hashed_oracle_rejects_wrong_digest_or_incomplete_file_coverage(
    hashed_validation, field, value
):
    workload, report = hashed_validation
    dataset = report.metadata.datasets[0]
    dataset = dataset.model_copy(update={"fingerprint": {**dataset.fingerprint, field: value}})
    metadata = report.metadata.model_copy(update={"datasets": [dataset]})
    with pytest.raises(AssertionError, match="content fingerprint"):
        workload.verify(report.model_copy(update={"metadata": metadata}))


def test_hashed_oracle_rejects_missing_dataset_provenance(hashed_validation):
    workload, report = hashed_validation
    metadata = report.metadata.model_copy(update={"datasets": []})
    with pytest.raises(AssertionError, match="dataset coverage"):
        workload.verify(report.model_copy(update={"metadata": metadata}))


def test_hashed_oracle_rejects_candidate_that_ignores_hash_inputs(
    hashed_validation, transaction_fixture, tmp_path
):
    hashed, _ = hashed_validation
    path, _ = transaction_fixture
    unhashed = build_workload("wide_validation", path, tmp_path, WorkloadOptions())
    with pytest.raises(AssertionError, match="content fingerprint"):
        hashed.verify(unhashed.run())


def test_hash_setup_detects_changed_fixture_bytes_before_timing(tmp_path):
    fixture = tmp_path / "fixture"
    manifest = prepare_fixture(fixture, rows=32, seed=7)
    path = fixture / manifest["datasets"]["transactions"]["files"]["parquet"][0]["path"]
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="fixture content differs"):
        build_workload(
            "wide_validation", fixture, tmp_path / "work", WorkloadOptions(hash_inputs=True)
        )
