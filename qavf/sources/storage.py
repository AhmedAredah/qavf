"""Configured cloud storage secrets for DuckDB connections and Polars remote reads.

DuckDB receives each secret as ``CREATE SECRET``; remote Parquet read through Polars
receives translated static ``storage_options``. Selection uses the longest matching
scope, else the unscoped secret of the URL's storage type. Default credential
discovery remains backend-specific; explicit chain selections are rejected for
typed Parquet instead of being silently discarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from qavf.sources.base import SourceError
from qavf.sources.sql import quote_identifier, sql_literal

if TYPE_CHECKING:
    from qavf.settings import SecretSpec

#: URL schemes each secret type serves, as DuckDB and Polars spell them.
SCHEMES = {"s3": ("s3", "s3a", "s3n"), "azure": ("az", "azure", "abfs", "abfss")}


def create_secret_statements(name: str, spec: SecretSpec) -> list[str]:
    """Temporary ``CREATE SECRET`` statements; they contain credential values.

    A secret with several scopes becomes one DuckDB secret per scope: list-valued
    ``SCOPE`` is not accepted by every supported DuckDB version.
    """
    options = [f"TYPE {spec.type}", f"PROVIDER {spec.provider}"]

    def add(option: str, value: str | None) -> None:
        if value is not None:
            options.append(f"{option} {sql_literal(value)}")

    if spec.type == "s3":
        add("KEY_ID", spec.key_id)
        add("SECRET", spec.secret)
        add("SESSION_TOKEN", spec.session_token)
        add("REGION", spec.region)
        add("ENDPOINT", spec.endpoint)
        add("URL_STYLE", spec.url_style)
        if spec.endpoint is not None:
            options.append(f"USE_SSL {'true' if spec.use_ssl else 'false'}")
    else:
        add("CONNECTION_STRING", spec.connection_string)
        add("ACCOUNT_NAME", spec.account_name)
    add("CHAIN", spec.chain)
    scopes: list[str | None] = list(spec.scope) or [None]
    statements = []
    for index, scope in enumerate(scopes):
        # Encode every name, including single-scope secrets. The separating
        # underscore is not a hex digit, making (configured name, scope index)
        # injective even when a caller's name resembles an internal name.
        native = f"__qavf_secret_{name.encode('utf-8').hex()}_{index}"
        clause = [] if scope is None else [f"SCOPE {sql_literal(scope)}"]
        statements.append(
            f"CREATE OR REPLACE SECRET {quote_identifier(native)} ({', '.join(options + clause)})"
        )
    return statements


def validate_secret_scopes(secrets: Mapping[str, SecretSpec]) -> None:
    """Validate direct source environments as well as root configurations."""
    selected: dict[tuple[str, str], SecretSpec] = {}
    for spec in secrets.values():
        for prefix in spec.scope or [""]:
            previous = selected.setdefault((spec.type, prefix), spec)
            if previous != spec:
                raise SourceError("conflicting storage secret scopes for the requested URL")


def secret_for(url: str, secrets: Mapping[str, SecretSpec]) -> SecretSpec | None:
    """The secret DuckDB would select for ``url``, or ``None`` when none applies."""
    validate_secret_scopes(secrets)
    scheme = url.split("://", 1)[0].lower()
    candidates = [spec for spec in secrets.values() if scheme in SCHEMES[spec.type]]
    scoped = [
        (len(prefix), spec)
        for spec in candidates
        for prefix in spec.scope
        if url.startswith(prefix)
    ]
    if scoped:
        return max(scoped, key=lambda item: item[0])[1]
    return next((spec for spec in candidates if not spec.scope), None)


def _connection_string(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in value.split(";"):
        if not part.strip():
            continue
        key, separator, item = part.partition("=")
        key = key.strip().casefold()
        if not separator or not key or key in fields:
            raise SourceError("invalid or repeated Azure connection-string field")
        fields[key] = item.strip()
    return fields


def _azure_options(value: str) -> dict[str, str]:
    """Translate blob endpoint identity or reject unsupported modes explicitly."""
    fields = _connection_string(value)
    supported = {
        "accountname", "accountkey", "sharedaccesssignature", "blobendpoint",
        "defaultendpointsprotocol", "endpointsuffix", "queueendpoint", "tableendpoint",
        "fileendpoint",
    }
    if set(fields) - supported:
        raise SourceError(
            "typed Parquet cannot translate this Azure connection-string mode; "
            "supply AccountName, AccountKey or SharedAccessSignature, and an explicit "
            "BlobEndpoint for emulator or custom endpoints"
        )
    protocol = fields.get("defaultendpointsprotocol", "https").lower()
    if protocol not in ("http", "https"):
        raise SourceError("Azure DefaultEndpointsProtocol must be http or https")
    endpoint = fields.get("blobendpoint")
    if endpoint is None and (
        "endpointsuffix" in fields or "defaultendpointsprotocol" in fields
    ):
        account = fields.get("accountname")
        suffix = fields.get("endpointsuffix", "core.windows.net")
        if not account or not suffix:
            raise SourceError("Azure endpoint translation requires AccountName and EndpointSuffix")
        endpoint = f"{protocol}://{account}.blob.{suffix}"
    if not fields.get("accountname"):
        raise SourceError(
            "typed Parquet Azure connection strings require AccountName, including "
            "SAS strings with BlobEndpoint; add it explicitly instead of relying "
            "on account discovery from a custom endpoint"
        )
    if not (fields.get("accountkey") or fields.get("sharedaccesssignature")):
        raise SourceError("Azure connection string lacks explicit credentials")
    mapping = {
        "accountname": "azure_storage_account_name",
        "accountkey": "azure_storage_account_key",
        "sharedaccesssignature": "azure_storage_sas_key",
    }
    options = {key: fields[field] for field, key in mapping.items() if field in fields}
    if endpoint is not None:
        if not endpoint.lower().startswith(("http://", "https://")):
            raise SourceError("Azure BlobEndpoint must be an http or https URL")
        options["azure_storage_endpoint"] = endpoint
        if endpoint.lower().startswith("http://"):
            options["allow_http"] = "true"
    return options


def storage_options(url: str, secrets: Mapping[str, SecretSpec]) -> dict[str, str] | None:
    """Polars ``storage_options`` for ``url``; ``None`` keeps Polars' own discovery."""
    spec = secret_for(url, secrets)
    if spec is None:
        return None
    if spec.chain is not None:
        raise SourceError(
            "explicit credential_chain.chain is not supported by the typed Parquet reader; "
            "supply explicit credentials or omit chain to use backend-default discovery"
        )
    options: dict[str, str] = {}
    if spec.type == "s3":
        values = {
            "aws_access_key_id": spec.key_id,
            "aws_secret_access_key": spec.secret,
            "aws_session_token": spec.session_token,
            "aws_region": spec.region,
        }
        options.update({key: value for key, value in values.items() if value is not None})
        if spec.endpoint is not None:
            scheme = "https" if spec.use_ssl else "http"
            options["aws_endpoint_url"] = f"{scheme}://{spec.endpoint}"
            if not spec.use_ssl:
                options["allow_http"] = "true"
        if spec.url_style is not None:
            options["aws_virtual_hosted_style_request"] = str(spec.url_style == "vhost").lower()
    elif spec.connection_string is not None:
        options.update(_azure_options(spec.connection_string))
    else:
        options["azure_storage_account_name"] = str(spec.account_name)
    return options
