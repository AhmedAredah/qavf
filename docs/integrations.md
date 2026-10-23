# Live adapter assurance

The integration suite separates service setup, exact-value test data, prerequisite
policy, and public API assertions into small modules under `tests/integration/`.
No PostgreSQL driver or S3 SDK is added to QAVF's runtime dependencies: `psycopg`
and `boto3` belong to the test-only `integration` dependency group.

| Service | Real path exercised | Cases |
| --- | --- | --- |
| PostgreSQL | Database adapter; configured parameterized `Query`; native and snapshot execution; missing table | 7 |
| S3-compatible object store | Authenticated S3 `Query` on a borrowed DuckDB connection; public HTTP object file adapter; missing object | 5 |
| Loopback HTTP | Native remote Parquet file adapter with byte-range requests; missing file | 3 |

The shared corpus includes integers beyond binary64's exact range, 24-digit
Decimals, UTC timestamps around a daylight-saving transition, Unicode, quoted
text, missing values, a negative amount, and a duplicate key. Tests compare exact
readback values and QAVF verdicts/counts with an in-memory input. Independently
specified expectations require one missing amount, one negative amount, and one
duplicated key involving two rows. Missing data must produce a nonzero acceptance
code and explicit unassessed results, with no passing checks.

## Required mode

The default `QAVF_INTEGRATION_MODE=disabled` skips live tests with an explicit
reason. Enable `required` for an integration qualification run. Invalid mode
values, missing configuration, missing dependencies, unavailable extensions,
unreachable services, fixture failures, and cleanup failures are errors; they
cannot silently become a skip. Offline prerequisite-policy tests are unmarked
and are never counted as evidence of a live connection.

Install the locked test environment and native extensions:

```bash
uv sync --locked --all-extras --group integration
uv run --no-sync python scripts/verify_release.py extensions --integration
```

Run the genuine HTTP cases without external service containers:

```bash
QAVF_INTEGRATION_MODE=required uv run --no-sync python -m pytest tests/integration/test_http.py -q
```

For the complete live suite, start disposable PostgreSQL and S3-compatible
services and supply their endpoints. Routine CI does not provision these services;
the release workflow's integration job uses PostgreSQL, a
[moto](https://github.com/getmoto/moto) S3 server and the Azurite Blob emulator with
synthetic local credentials. Personal cloud accounts are not needed.

| Variable | Requirement |
| --- | --- |
| `QAVF_INTEGRATION_MODE` | Set to `required`. |
| `QAVF_POSTGRES_DSN` | PostgreSQL connection string for a disposable database; role must create/drop its own schema and tables. |
| `QAVF_S3_ENDPOINT` | HTTP(S) origin such as `http://127.0.0.1:9000`, without a path, query, or embedded credentials. |
| `QAVF_S3_ACCESS_KEY` / `QAVF_S3_SECRET_KEY` | Disposable service credentials permitted to create/delete a bucket, set its policy, and put/get/delete test objects. |
| `QAVF_S3_REGION` | Optional, defaults to `us-east-1`. |

```bash
QAVF_INTEGRATION_MODE=required uv run --no-sync python -m pytest tests/integration -m integration -q
```

Each PostgreSQL fixture owns a UUID-named schema. Each S3 fixture owns a
UUID-named bucket containing two synthetic Parquet objects. The bucket grants
anonymous `GetObject` solely to exercise QAVF's native HTTP file path; a separate
test uses authenticated `s3://` reads. Teardown deletes only those fixture-owned
resources. Borrowed DuckDB connections are checked to remain usable after
validation. The loopback HTTP server closes and joins its thread after the tests.

## Scope of the evidence

Passing loopback HTTP tests does not certify PostgreSQL or S3. Passing against
moto's S3 server establishes the tested S3-compatible behavior, not all Amazon S3 account
policies, authentication modes, or storage services. The current live suite does
not qualify MySQL/MariaDB, presigned-URL authentication, production TLS policies,
or database consistency guarantees beyond the exercised paths. Results must
state the actual service versions and selected tests. See
[release assurance](release_assurance.md) for the strict evidence gate.

Implementation references:

- [DuckDB PostgreSQL extension](https://duckdb.org/docs/stable/core_extensions/postgres.html)
- [DuckDB S3 API and secrets](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api.html)
- [Boto3 S3 client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [pytest skipping semantics](https://docs.pytest.org/en/stable/how-to/skipping.html)
