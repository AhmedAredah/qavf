"""Configured storage secrets read private S3 and Azure objects through both readers.

Every path is checked through a file dataset (remote Parquet is read by Polars with
``storage_options``) and a query dataset (DuckDB with ``CREATE SECRET``), against the
exact corpus contract. Credentials never appear in the published report.
"""

import os
from copy import deepcopy

import pytest

from qavf import validate

from .azure_service import azure_corpus
from .contracts import assert_exact_contract, assert_unavailable, configuration
from .policy import endpoint_parts
from .pytest_support import available_service
from .s3_service import s3_corpus

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def s3(live_required):
    with available_service(s3_corpus) as service:
        yield service


@pytest.fixture(scope="module")
def azure(live_required):
    with available_service(azure_corpus) as service:
        yield service


def _config(secrets: dict, url: str, reader: str) -> dict:
    config = deepcopy(configuration())
    dataset = config["datasets"]["data"]
    if reader == "file":
        dataset["path"] = url
    else:
        dataset["source"] = {"type": "query", "sql": f"SELECT * FROM read_parquet('{url}')"}
    config["secrets"] = secrets
    return config


def _s3_secret(bucket: str) -> dict:
    endpoint, use_ssl = endpoint_parts(os.environ["QAVF_S3_ENDPOINT"])
    return {
        "type": "s3",
        "key_id": os.environ["QAVF_S3_ACCESS_KEY"],
        "secret": os.environ["QAVF_S3_SECRET_KEY"],
        "region": os.environ.get("QAVF_S3_REGION", "us-east-1"),
        "endpoint": endpoint,
        "url_style": "path",
        "use_ssl": use_ssl,
        "scope": [f"s3://{bucket}"],
    }


def _assert_no_credentials(report, *values: str) -> None:
    published = report.model_dump_json()
    for value in values:
        assert value not in published


@pytest.mark.parametrize("reader", ["file", "query"])
@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_s3_secret_reads_private_objects(s3, reader, clean):
    url = f"s3://{s3.bucket}/{s3.key(clean)}"
    report = validate(config=_config({"lake": _s3_secret(s3.bucket)}, url, reader))
    assert_exact_contract(report, clean=clean)
    _assert_no_credentials(report, os.environ["QAVF_S3_SECRET_KEY"])


@pytest.mark.parametrize("reader", ["file", "query"])
def test_s3_object_without_its_secret_is_unavailable(s3, reader):
    # Without the configured endpoint the read resolves to AWS itself and fails:
    # a passing read above therefore used the secret.
    other = _s3_secret("some-other-bucket")
    url = f"s3://{s3.bucket}/{s3.key(True)}"
    assert_unavailable(validate(config=_config({"other": other}, url, reader)))


@pytest.mark.parametrize("reader", ["file", "query"])
@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_azure_secret_reads_private_blobs(azure, reader, clean):
    secret = {"type": "azure", "connection_string": azure.connection_string}
    report = validate(config=_config({"blob": secret}, azure.uri(clean), reader))
    assert_exact_contract(report, clean=clean)
    account_key = azure.connection_string.split("AccountKey=", 1)[1].split(";", 1)[0]
    _assert_no_credentials(report, account_key, azure.connection_string)


@pytest.mark.parametrize("reader", ["file", "query"])
def test_azure_wrong_key_is_unavailable_and_not_disclosed(azure, reader):
    wrong_key = "V3JvbmdLZXlGb3JRQVZGSW50ZWdyYXRpb25UZXN0cw=="
    head, _, tail = azure.connection_string.partition("AccountKey=")
    connection_string = head + "AccountKey=" + wrong_key + ";" + tail.split(";", 1)[1]
    secret = {"type": "azure", "connection_string": connection_string}
    report = validate(config=_config({"blob": secret}, azure.uri(True), reader))
    assert_unavailable(report)
    _assert_no_credentials(report, wrong_key)
