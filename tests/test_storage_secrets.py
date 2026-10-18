"""Storage secret configuration, generated SQL, Polars options and credential redaction."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.settings import QAVFSettings, SecretSpec
from qavf.sources import native
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.storage import create_secret_statements, secret_for, storage_options

S3 = {"type": "s3", "key_id": "AKIAEXAMPLEKEYID", "secret": "s3cr'et-value", "region": "us-east-1"}
AZURE = (
    "DefaultEndpointsProtocol=http;AccountName=acct;AccountKey=QUNDT1VOVEtFWQ==;"
    "BlobEndpoint=http://127.0.0.1:10000/acct;"
)


def spec(**values: object) -> SecretSpec:
    return SecretSpec.model_validate(values)


def test_s3_secret_sql_escapes_values_and_creates_one_secret_per_scope():
    # List-valued SCOPE is not accepted by every supported DuckDB version.
    secret = spec(**S3, endpoint="minio:9000", url_style="path", use_ssl=False,
                  scope=["s3://a", "s3://b/x"])
    options = (
        "TYPE s3, PROVIDER config, KEY_ID 'AKIAEXAMPLEKEYID', SECRET 's3cr''et-value', "
        "REGION 'us-east-1', ENDPOINT 'minio:9000', URL_STYLE 'path', USE_SSL false"
    )
    assert create_secret_statements("lake", secret) == [
        f"CREATE OR REPLACE SECRET \"lake__scope_0\" ({options}, SCOPE 's3://a')",
        f"CREATE OR REPLACE SECRET \"lake__scope_1\" ({options}, SCOPE 's3://b/x')",
    ]
    single = create_secret_statements("lake", spec(**S3, scope=["s3://a"]))
    assert single[0].startswith('CREATE OR REPLACE SECRET "lake" (')


def test_credential_chain_and_azure_secret_sql():
    chained = spec(type="s3", provider="credential_chain", chain="env")
    assert create_secret_statements("c", chained) == [
        "CREATE OR REPLACE SECRET \"c\" (TYPE s3, PROVIDER credential_chain, CHAIN 'env')"
    ]
    assert create_secret_statements(
        "b", spec(type="azure", provider="credential_chain", account_name="acct")
    ) == [
        "CREATE OR REPLACE SECRET \"b\" (TYPE azure, PROVIDER credential_chain, "
        "ACCOUNT_NAME 'acct')"
    ]
    assert "CONNECTION_STRING 'DefaultEndpointsProtocol=http;" in create_secret_statements(
        "b", spec(type="azure", connection_string=AZURE)
    )[0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"type": "s3"}, "needs key_id and secret"),
        ({"type": "s3", "provider": "credential_chain", "secret": "x"}, "take no key_id"),
        ({**S3, "chain": "env"}, "only to provider 'credential_chain'"),
        ({**S3, "endpoint": "https://minio:9000"}, "without a scheme"),
        ({**S3, "connection_string": AZURE}, "does not take connection_string"),
        ({**S3, "scope": ["bucket"]}, "must be a URL prefix"),
        ({"type": "azure"}, "needs connection_string"),
        ({"type": "azure", "connection_string": AZURE, "region": "x"}, "does not take region"),
        ({"type": "azure", "connection_string": AZURE, "use_ssl": False}, "does not take use_ssl"),
        ({"type": "azure", "provider": "credential_chain"}, "needs account_name"),
        ({"type": "gcs", "key_id": "a", "secret": "b"}, "Input should be"),
    ],
)
def test_incomplete_or_mixed_secrets_are_configuration_errors(values, message):
    with pytest.raises(ValidationError, match=message):
        SecretSpec.model_validate(values)


@pytest.mark.parametrize(
    ("secrets", "message"),
    [
        ({"a": S3, "b": S3}, "both unscoped s3 secrets"),
        ({"lake": S3, "LAKE": {**S3, "scope": ["s3://x"]}}, "differ only in case"),
        ({"my-lake": S3}, "plain SQL identifier"),
    ],
)
def test_ambiguous_secret_sets_are_rejected(secrets, message):
    with pytest.raises(ValidationError, match=message):
        QAVFSettings.model_validate({"secrets": secrets})


def test_url_selects_longest_scope_then_the_unscoped_secret_of_its_type():
    secrets = {
        "default": spec(**S3),
        "bucket": spec(**S3, scope=["s3://data"]),
        "path": spec(**{**S3, "region": "eu-west-1"}, scope=["s3://data/eu/"]),
        "blob": spec(type="azure", connection_string=AZURE),
    }
    assert secret_for("s3://data/eu/x.parquet", secrets) is secrets["path"]
    assert secret_for("s3://data/us/x.parquet", secrets) is secrets["bucket"]
    assert secret_for("s3://elsewhere/x.parquet", secrets) is secrets["default"]
    assert secret_for("abfss://c/x.parquet", secrets) is secrets["blob"]
    assert secret_for("gs://bucket/x.parquet", secrets) is None
    assert secret_for("https://host/x.parquet", secrets) is None


def test_polars_options_mirror_the_duckdb_secret():
    secret = spec(**S3, session_token="tok", endpoint="minio:9000", url_style="path",
                  use_ssl=False)
    assert storage_options("s3://b/x.parquet", {"s": secret}) == {
        "aws_access_key_id": "AKIAEXAMPLEKEYID",
        "aws_secret_access_key": "s3cr'et-value",
        "aws_session_token": "tok",
        "aws_region": "us-east-1",
        "aws_endpoint_url": "http://minio:9000",
        "allow_http": "true",
        "aws_virtual_hosted_style_request": "false",
    }
    blob = {"b": spec(type="azure", connection_string=AZURE)}
    assert storage_options("az://c/x.parquet", blob) == {
        "azure_storage_account_name": "acct",
        "azure_storage_account_key": "QUNDT1VOVEtFWQ==",
        "azure_storage_endpoint": "http://127.0.0.1:10000/acct",
        "allow_http": "true",
    }
    chained = spec(type="azure", provider="credential_chain", account_name="acct")
    assert storage_options("az://c/x", {"b": chained}) == {"azure_storage_account_name": "acct"}
    assert storage_options("s3://b/x", {}) is None


def test_every_owned_connection_creates_each_secret(monkeypatch):
    executed: list[str] = []
    monkeypatch.setattr(
        native, "create_secret_statements", lambda name, _: executed.append(name) or ["SELECT 1"]
    )
    azure = spec(type="azure", connection_string=AZURE)
    env = SourceEnvironment(secrets={"one": spec(**S3), "two": azure})
    for _ in range(2):
        native.connect(env).close()
    assert executed == ["one", "two", "one", "two"]


def test_secret_failure_never_discloses_the_statement(monkeypatch):
    monkeypatch.setattr(
        native,
        "create_secret_statements",
        lambda *_: ["CREATE SECRET leak (TYPE no_such_storage, SECRET 'TOPSECRETVALUE')"],
    )
    with pytest.raises(SourceError, match="could not create storage secret 'leak'") as raised:
        native.connect(SourceEnvironment(secrets={"leak": spec(**S3)}))
    assert "TOPSECRETVALUE" not in str(raised.value)


def test_reports_mask_secret_values(tmp_path, monkeypatch):
    # Local data keeps this offline; the live suites read actual remote objects.
    monkeypatch.setattr(native, "create_secret_statements", lambda *_: ["SELECT 1"])
    path = tmp_path / "data.csv"
    pl.DataFrame({"x": [1, 2]}).write_csv(path)
    config = {
        "datasets": {"data": {"path": str(path), "schema": {"columns": {"x": {"ge": 0}}}}},
        "secrets": {
            "lake": {**S3, "session_token": "SESSIONTOKENVALUE"},
            "blob": {"type": "azure", "connection_string": AZURE},
        },
    }
    report = validate(config=config)
    assert report.verdict() == "passed"
    published = report.model_dump_json()
    for value in ("AKIAEXAMPLEKEYID", "s3cr'et-value", "SESSIONTOKENVALUE", "QUNDT1VOVEtFWQ=="):
        assert value not in published
    recorded = report.metadata.config["secrets"]
    assert recorded["lake"]["region"] == "us-east-1"
    assert recorded["lake"]["key_id"] == recorded["blob"]["connection_string"] == "***"


@pytest.mark.parametrize("values", [S3, {"type": "azure", "connection_string": AZURE}])
def test_serialized_secrets_round_trip(values):
    # Isolated execution transports settings with every field, defaults included.
    secret = spec(**values)
    assert SecretSpec.model_validate(secret.model_dump()) == secret
    settings = QAVFSettings.model_validate({"secrets": {"s": values}})
    assert QAVFSettings.model_validate(settings.model_dump(mode="python")) == settings


def test_credential_chain_without_credentials_spares_the_connection(monkeypatch, caplog):
    # Newer DuckDB validates a credential chain on creation and fails without
    # credentials; only reads that need the secret may fail, not the connection.
    failing = ["CREATE SECRET chain (TYPE no_such_storage, SECRET 'CHAINSECRETVALUE')"]
    monkeypatch.setattr(native, "create_secret_statements", lambda *_: failing)
    chained = spec(type="s3", provider="credential_chain")
    with caplog.at_level("WARNING", logger="qavf.sources.native"):
        con = native.connect(SourceEnvironment(secrets={"chain": chained}))
    assert con.execute("SELECT 42").fetchone() == (42,)
    con.close()
    assert "found no usable credentials" in caplog.text
    assert "CHAINSECRETVALUE" not in caplog.text
