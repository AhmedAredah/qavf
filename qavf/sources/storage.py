"""Configured cloud storage secrets for DuckDB connections and Polars remote reads.

DuckDB receives each secret as ``CREATE SECRET``; remote Parquet read through Polars
receives the equivalent ``storage_options``. Both choose a secret by URL the same way:
the longest matching scope, else the one unscoped secret of the URL's storage type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

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
        native = name if len(scopes) == 1 else f"{name}__scope_{index}"
        clause = [] if scope is None else [f"SCOPE {sql_literal(scope)}"]
        statements.append(
            f"CREATE OR REPLACE SECRET {quote_identifier(native)} "
            f"({', '.join(options + clause)})"
        )
    return statements


def secret_for(url: str, secrets: Mapping[str, SecretSpec]) -> SecretSpec | None:
    """The secret DuckDB would select for ``url``, or ``None`` when none applies."""
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
    parts = (part.split("=", 1) for part in value.split(";") if "=" in part)
    return {key.strip(): item.strip() for key, item in parts}


def storage_options(url: str, secrets: Mapping[str, SecretSpec]) -> dict[str, str] | None:
    """Polars ``storage_options`` for ``url``; ``None`` keeps Polars' own discovery."""
    spec = secret_for(url, secrets)
    if spec is None:
        return None
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
        fields = _connection_string(spec.connection_string)
        mapping = {
            "AccountName": "azure_storage_account_name",
            "AccountKey": "azure_storage_account_key",
            "SharedAccessSignature": "azure_storage_sas_key",
            "BlobEndpoint": "azure_storage_endpoint",
        }
        options.update({key: fields[field] for field, key in mapping.items() if field in fields})
        if options.get("azure_storage_endpoint", "").startswith("http://"):
            options["allow_http"] = "true"
    else:
        options["azure_storage_account_name"] = str(spec.account_name)
    return options
