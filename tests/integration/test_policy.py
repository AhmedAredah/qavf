"""Offline prerequisite-policy checks; these are not live integration evidence."""

import pytest

from .policy import IntegrationUnavailable, dependency, endpoint_parts, required_mode, settings


def test_live_services_require_explicit_opt_in():
    assert not required_mode({})
    assert required_mode({"QAVF_INTEGRATION_MODE": "required"})


@pytest.mark.parametrize("value", ["auto", "optional", "true", "", "REQUIRED"])
def test_malformed_mode_fails_instead_of_disabling(value):
    with pytest.raises(IntegrationUnavailable, match="must be disabled or required"):
        required_mode({"QAVF_INTEGRATION_MODE": value})


def test_missing_settings_do_not_reveal_present_secrets():
    with pytest.raises(IntegrationUnavailable) as error:
        settings({"PASSWORD": "sentinel-secret", "ENDPOINT": " "}, ("PASSWORD", "ENDPOINT"))
    assert "ENDPOINT" in str(error.value)
    assert "sentinel-secret" not in str(error.value)


def test_unavailable_dependency_is_a_failure(monkeypatch):
    def absent(name):
        raise ImportError("unavailable")

    monkeypatch.setattr("importlib.import_module", absent)
    with pytest.raises(IntegrationUnavailable, match="dependency 'psycopg' unavailable"):
        dependency("psycopg")


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///tmp/data",
        "http://user:password@localhost:9000",
        "http://localhost/bucket",
        "http://localhost?token=secret",
        "localhost:9000",
        "https://localhost/#fragment",
    ],
)
def test_invalid_endpoint_is_rejected_without_echoing_it(endpoint):
    with pytest.raises(IntegrationUnavailable) as error:
        endpoint_parts(endpoint)
    assert endpoint not in str(error.value)


@pytest.mark.parametrize(
    "endpoint, expected",
    [
        ("http://127.0.0.1:9000", ("127.0.0.1:9000", False)),
        ("https://objects.example.test/", ("objects.example.test", True)),
    ],
)
def test_endpoint_transport_is_explicit(endpoint, expected):
    assert endpoint_parts(endpoint) == expected


def test_required_postgres_cannot_skip_missing_configuration(monkeypatch):
    from .postgres_service import postgres_corpus

    monkeypatch.delenv("QAVF_POSTGRES_DSN", raising=False)
    with pytest.raises(IntegrationUnavailable, match="QAVF_POSTGRES_DSN"), postgres_corpus():
        pytest.fail("missing PostgreSQL settings were accepted")


def test_required_s3_cannot_skip_missing_configuration(monkeypatch):
    from .s3_service import s3_corpus

    monkeypatch.delenv("QAVF_S3_ENDPOINT", raising=False)
    with pytest.raises(IntegrationUnavailable, match="QAVF_S3_ENDPOINT"), s3_corpus():
        pytest.fail("missing S3 settings were accepted")


def test_postgres_service_failure_is_redacted(monkeypatch):
    from types import SimpleNamespace

    from . import postgres_service

    def unavailable(*args, **kwargs):
        raise OSError("connection failed: sentinel-password")

    monkeypatch.setenv("QAVF_POSTGRES_DSN", "password=sentinel-password")
    monkeypatch.setattr(postgres_service, "extension", lambda name: None)
    monkeypatch.setattr(
        postgres_service, "dependency", lambda name: SimpleNamespace(connect=unavailable)
    )
    with pytest.raises(IntegrationUnavailable) as error, postgres_service.postgres_corpus():
        pytest.fail("unavailable service was accepted")
    assert "service unavailable (OSError)" in str(error.value)
    assert "sentinel-password" not in str(error.value)


@pytest.mark.parametrize("delete_fails", [False, True])
def test_s3_partial_upload_cleans_only_owned_keys_and_bucket(monkeypatch, delete_fails):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from . import s3_service

    client = Mock()
    client.put_object.side_effect = OSError("network error: sentinel-password")
    if delete_fails:
        client.delete_object.side_effect = OSError("delete error: sentinel-password")
    boto3 = SimpleNamespace(client=lambda *args, **kwargs: client)
    botocore = SimpleNamespace(Config=lambda **kwargs: kwargs)
    monkeypatch.setenv("QAVF_S3_ENDPOINT", "http://127.0.0.1:9000")
    monkeypatch.setenv("QAVF_S3_ACCESS_KEY", "test")
    monkeypatch.setenv("QAVF_S3_SECRET_KEY", "sentinel-password")
    monkeypatch.setattr(s3_service, "extension", lambda name: None)
    monkeypatch.setattr(
        s3_service, "dependency", lambda name: boto3 if name == "boto3" else botocore
    )
    with pytest.raises(IntegrationUnavailable) as error, s3_service.s3_corpus():
        pytest.fail("failed upload was accepted")
    assert "sentinel-password" not in str(error.value)
    assert "fixture setup unavailable" in str(error.value)
    if delete_fails:
        assert error.value.__notes__ == ["S3 fixture cleanup failed (OSError)"]
    bucket = client.create_bucket.call_args.kwargs["Bucket"]
    assert bucket.startswith("qavf-integration-")
    client.delete_object.assert_called_once_with(Bucket=bucket, Key="clean.parquet")
    client.delete_bucket.assert_called_once_with(Bucket=bucket)
    client.close.assert_called_once_with()


def test_cleanup_attempts_all_actions_and_never_silently_succeeds():
    from unittest.mock import Mock

    from .cleanup import cleanup_owned

    actions = [Mock(side_effect=OSError("sentinel-password")), Mock(), Mock()]
    with pytest.raises(IntegrationUnavailable) as error:
        cleanup_owned("test", actions, None)
    for action in actions:
        action.assert_called_once_with()
    assert str(error.value) == "test fixture cleanup failed (OSError)"


def test_fixture_prerequisites_fail_without_environment_traceback():
    from .pytest_support import available_service

    def unavailable():
        raise IntegrationUnavailable("required service unavailable")

    with pytest.raises(pytest.fail.Exception) as error, available_service(unavailable):
        pytest.fail("unavailable service was accepted")
    assert str(error.value) == "required service unavailable"
    assert error.value.pytrace is False
