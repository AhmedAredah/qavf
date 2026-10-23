# Cloud storage

Datasets can be read directly from object storage by URL: `s3://bucket/key`,
`az://container/blob`, `abfss://…`, or `https://…`. Configure credentials once under
`secrets`. QAVF creates each secret as a DuckDB secret on every connection it owns
and translates static credential options to Polars, which reads remote Parquet.
Both use configured URL scopes to select the applicable secret.

```yaml
secrets:
  lake:
    type: s3
    provider: credential_chain      # environment, profile, instance or OIDC role
    region: us-east-1
datasets:
  orders:
    path: s3://my-lake/orders/*.parquet
    schema: {primary_key: [order_id]}
```

Use `${ENV_VAR}` for every credential value. Reports, logs and error messages mask
`key_id`, `secret`, `session_token`, `connection_string` and keys embedded in
connection strings; regions, endpoints, account names and scopes stay visible.

## Secret types

| Field | `s3` | `azure` |
|---|---|---|
| `provider` | `config` (keys below) or `credential_chain` | `config` (connection string) or `credential_chain` |
| `scope` | URL prefixes the secret applies to | URL prefixes the secret applies to |
| Credentials (`config`) | `key_id`, `secret`, optional `session_token` | `connection_string` |
| Location | `region`, `endpoint` (`host[:port]`), `url_style`, `use_ssl` | `account_name` (`credential_chain`) |
| `chain` | Credential sources to try, e.g. `env;config` | e.g. `cli;managed_identity` |

`credential_chain` uses the reader's platform credential discovery: AWS environment variables,
profiles, instance roles and web identity; Azure CLI login, managed identity and
environment variables. In GitHub Actions, sign in with
`aws-actions/configure-aws-credentials` or `azure/login` before running QAVF, and no
stored key is needed.

An explicit `chain` value is supported by native DuckDB reads. Typed Parquet reads
reject it before scanning because QAVF does not implement an equivalent Polars
provider sequence. Use static credentials or omit `chain` to allow the selected
backend's default discovery. Default discovery order and supported identity
providers can differ between readers; QAVF does not claim account-selection parity.
If native secret creation fails for an explicit chain, the source fails with a
redacted error; it does not continue without that requested provider sequence.

Typed Azure Parquet reads preserve `BlobEndpoint`, or construct the endpoint from
`AccountName`, `EndpointSuffix` and `DefaultEndpointsProtocol`. Unsupported
connection-string modes, including `UseDevelopmentStorage=true`, are rejected.
For an emulator supply its explicit account, key and `BlobEndpoint` instead.
For typed Parquet reads, include `AccountName` even in a SAS connection string
that supplies `BlobEndpoint`; QAVF does not infer the account from a custom endpoint.
These translations have offline contract tests; live sovereign-cloud and ADLS
Gen2 routing remain outside the verified integration scope below.

A URL uses the secret with the longest matching `scope`, otherwise the single
unscoped secret of its type. Two unscoped secrets of one type are rejected, so every
URL resolves to exactly one credential. Conflicting equal scoped prefixes of one
type are also rejected before selecting a reader. Names must be plain SQL identifiers.
Generated native identifiers encode both the configured name and scope index, so
names that resemble internal suffixes cannot overwrite another configured secret.

A directly supplied `DataSource` retains its own locations, connections and
secrets when bound to a run, including an explicitly empty secret mapping. The
run supplies execution limits and handles. Change the source definition to change
its credentials; run settings do not silently substitute them. Operations that
combine sources reject conflicting secret names or equal scopes, and connections
are reused only when their attachment and credential definitions match.

## Providers and S3-compatible services

```yaml
secrets:
  # Static keys, for example a CI service account.
  warehouse:
    type: s3
    key_id: ${AWS_ACCESS_KEY_ID}
    secret: ${AWS_SECRET_ACCESS_KEY}
    region: eu-west-1
    scope: [s3://warehouse-exports]
  # Cloudflare R2: S3 API with an account endpoint; use s3:// URLs.
  r2:
    type: s3
    key_id: ${R2_KEY_ID}
    secret: ${R2_SECRET}
    region: auto
    endpoint: <account-id>.r2.cloudflarestorage.com
    scope: [s3://r2-bucket]
  # MinIO or another self-hosted S3 server.
  minio:
    type: s3
    key_id: ${MINIO_KEY}
    secret: ${MINIO_SECRET}
    endpoint: minio.internal:9000
    url_style: path
    use_ssl: false
    scope: [s3://qa-inputs]
  # Azure Blob Storage or ADLS Gen2 (az://, abfss://).
  blob:
    type: azure
    connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  # Or Azure identity (CLI login, managed identity, workload identity).
  lake_identity:
    type: azure
    provider: credential_chain
    account_name: mydatalake
    scope: [abfss://curated]
```

Google Cloud Storage can be read through its S3-interoperability API: create HMAC
keys, configure an `s3` secret with `endpoint: storage.googleapis.com`, and address
objects as `s3://bucket/key`. Native `gs://` authentication is not configured by
QAVF; it uses each reader's own environment discovery.

## Scope of verification

The live integration suite reads private objects through both readers from an S3
server (moto) and the Azure Blob emulator (Azurite), checks exact values and
verdicts, and confirms that a wrong key is reported as unavailable without
disclosing it. The Azure tests use `az://` URLs. `abfss://` (ADLS Gen2) uses the same
secrets but needs a hierarchical-namespace account, which the emulator does not
provide, and `credential_chain` needs real cloud identities; neither is exercised.

Remote objects are read with the same exact semantics as local files, but they are
not hashed for source-identity checks the way local files are. Each secret applies to
the DuckDB connections QAVF creates; a DuckDB connection you pass in yourself keeps
its own secrets. Container images include the `httpfs`, `aws` and `azure` extensions.
