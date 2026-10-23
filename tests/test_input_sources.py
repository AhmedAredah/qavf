"""Input source selection, projection and typed reader boundaries."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import polars as pl
import pytest
from pydantic import ValidationError

from qavf import profile_input, validate
from qavf.plugins.registry import Component, ComponentRegistry
from qavf.settings import QAVFSettings, SecretSpec
from qavf.sources import FileParams, FileSource, SourceEnvironment, SourceError, native
from qavf.sources.storage import storage_options
from qavf.sources.typed import needs_typed_reader, needs_typed_sql_binding

CONFIG = {"plugins": {"entry_points": False}}


@pytest.mark.parametrize(
    ("suffix", "contents"),
    [
        ("csv", "x,X\n1,-99\n"),
        ("json", '[{"x":1,"X":-99}]'),
        ("ndjson", '{"x":1,"X":-99}\n'),
    ],
)
def test_projection_never_resolves_a_differently_cased_name(tmp_path, suffix, contents):
    path = tmp_path / f"case.{suffix}"
    path.write_text(contents)
    source = FileSource(FileParams(path=str(path), columns=["X"]), SourceEnvironment())
    # The native decoder has disambiguated X to X_1; refusal is preferable to
    # selecting physical x through SQL's case-insensitive identifier lookup.
    with pytest.raises(SourceError, match="requested column spelling"):
        source.load()


def test_native_projection_preserves_exact_case_order_and_quoted_names(tmp_path):
    path = tmp_path / "names.csv"
    pl.DataFrame({"Mixed": [1], "a'b": [2], "last": [3]}).write_csv(path)
    source = FileSource(
        FileParams(path=str(path), columns=["a'b", "Mixed"]), SourceEnvironment()
    )
    assert source.load().to_dicts() == [{"a'b": 2, "Mixed": 1}]


_TYPED_CASES = [
    (pl.Null, [None, None]),
    (pl.List(pl.Null), [[None], []]),
    (pl.Struct({"y": pl.Null}), [{"y": None}]),
]
if hasattr(pl, "UInt128"):
    _TYPED_CASES.append((pl.UInt128, [2**127 + 1, 2**128 - 1]))
if hasattr(pl, "Float16"):
    _TYPED_CASES.append((pl.Float16, [0.1, 65504.0]))


@pytest.mark.parametrize(("dtype", "values"), _TYPED_CASES)
def test_parquet_types_survive_reader_selection(tmp_path, dtype, values):
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    path = tmp_path / "logical.parquet"
    frame.write_parquet(path)
    source = FileSource(FileParams(path=str(path)), SourceEnvironment())
    assert needs_typed_reader(dtype)
    assert needs_typed_sql_binding(dtype)
    assert source.scan() is None
    actual = source.load()
    assert actual.schema == frame.schema
    assert actual.to_dicts() == frame.to_dicts()


def test_null_parquet_contract_and_profile_match_frame(tmp_path):
    frame = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Null)})
    path = tmp_path / "null.parquet"
    frame.write_parquet(path)
    schema = {"columns": {"x": {"dtype": "int", "nullable": True}}}
    assert validate(frame, schema=schema, config=CONFIG).verdict() == "failed"
    assert validate(path, schema=schema, config=CONFIG).verdict() == "failed"
    assert profile_input(frame, config=CONFIG).schema == profile_input(path, config=CONFIG).schema


def test_concurrent_cross_family_discovery_finishes_without_lock_inversion():
    class A(Component):
        name = "a"

    class B(Component):
        name = "b"

    a = ComponentRegistry("a", "a", Component, ())
    b = ComponentRegistry("b", "b", Component, ())
    ready = threading.Barrier(2)
    entered_a, entered_b = threading.Event(), threading.Event()
    errors = []

    def load_a():
        entered_a.set()
        entered_b.wait(0.1)
        b.discover(use_entry_points=True)
        return A

    def load_b():
        entered_b.set()
        entered_a.wait(0.1)
        a.discover(use_entry_points=True)
        return B

    a.entries = lambda: [SimpleNamespace(name="a", load=load_a)]
    b.entries = lambda: [SimpleNamespace(name="b", load=load_b)]

    def discover(registry):
        try:
            ready.wait(timeout=2)
            registry.discover(use_entry_points=True)
        except Exception as error:
            errors.append(error)

    workers = [threading.Thread(target=discover, args=(r,), daemon=True) for r in (a, b)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(3)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors
    assert a.plugins == {"a": "loaded"}
    assert b.plugins == {"b": "loaded"}


@pytest.mark.parametrize(
    ("protocol", "suffix", "expected"),
    [
        ("https", "core.usgovcloudapi.net", "https://acct.blob.core.usgovcloudapi.net"),
        ("http", "core.windows.net", "http://acct.blob.core.windows.net"),
    ],
)
def test_azure_protocol_and_suffix_preserve_endpoint_identity(protocol, suffix, expected):
    secret = SecretSpec(
        type="azure",
        connection_string=(
            f"DefaultEndpointsProtocol={protocol};AccountName=acct;"
            f"AccountKey=SYNTHETIC;EndpointSuffix={suffix}"
        ),
    )
    options = storage_options("az://container/data.parquet", {"azure": secret})
    assert options["azure_storage_endpoint"] == expected
    assert options.get("allow_http") == ("true" if protocol == "http" else None)


@pytest.mark.parametrize("connection", ["UseDevelopmentStorage=true", "garbage", ""])
def test_untranslatable_azure_credentials_never_become_default_discovery(connection):
    if not connection:
        with pytest.raises(ValidationError):
            SecretSpec(type="azure", connection_string=connection)
        return
    secret = SecretSpec(type="azure", connection_string=connection)
    with pytest.raises(SourceError):
        storage_options("az://container/data.parquet", {"azure": secret})


def test_sas_endpoint_requires_explicit_account_for_typed_reader():
    secret = SecretSpec(
        type="azure",
        connection_string="BlobEndpoint=https://custom.example;SharedAccessSignature=sv=x&sig=y",
    )
    with pytest.raises(SourceError, match="require AccountName"):
        storage_options("az://container/data.parquet", {"azure": secret})


def test_conflicting_equal_scopes_rejected_for_settings_and_direct_sources():
    secrets = {
        name: SecretSpec(type="s3", key_id=name, secret="SYNTHETIC", scope=["s3://bucket"])
        for name in ("one", "two")
    }
    with pytest.raises(ValidationError, match="same s3 scope"):
        QAVFSettings(secrets=secrets)
    with pytest.raises(SourceError, match="conflicting storage secret scopes"):
        storage_options("s3://bucket/data.parquet", secrets)
    with pytest.raises(SourceError, match="conflicting storage secret scopes"):
        native.connect(SourceEnvironment(secrets=secrets))


def test_explicit_credential_chain_creation_failure_cannot_fall_back(monkeypatch):
    monkeypatch.setattr(
        native,
        "create_secret_statements",
        lambda *_: ["CREATE SECRET chosen (TYPE no_such_storage)"],
    )
    secret = SecretSpec(type="s3", provider="credential_chain", chain="env")
    with pytest.raises(SourceError, match="could not create storage secret"):
        native.connect(SourceEnvironment(secrets={"chosen": secret}))
