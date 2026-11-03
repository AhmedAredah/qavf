# Backend capabilities and core CLI

QAVF keeps application-specific assumptions out of backend selection. Choose the
backend from the configured validation contracts and input ownership requirements.

```python
from qavf import backend_capabilities, validate

native = backend_capabilities("duckdb")
print(native.rule_ids)
print(native.to_dict())  # detached JSON-ready catalog
report = validate("data.csv", config="qavf.yaml", backend="duckdb")
```

`backend_capabilities("duckdb" | "polars")` returns an immutable
`BackendCapabilities` record containing `required_packages`, `rule_ids`, `sources`,
`policies`, and `limitations`. Each source/policy is a `Capability` record with a
stable `name`, a Boolean `supported`, and explanatory `detail`. These APIs import
neither dataframe engines nor Pydantic and perform no plugin discovery. They
describe built-in support; runtime input/configuration checks still apply. The
package list names the backend's data engines, not all QAVF distribution dependencies.
Installation, server accessibility, DuckDB extension availability, and plugin
capabilities are separate concerns.

| Facility | DuckDB backend | Polars backend |
|---|---|---|
| Declared schemas, values, keys, references, row expressions, SQL assertions, N/A groups | INP-001–007 and INP-009 | Same built-in rules |
| Freshness, statistical, temporal, spatial, reconciliation, regression and comparison rules | Requires Polars backend | Supported; some methods need additional extras |
| Strict CSV, Parquet, JSON/NDJSON, local globs and DuckDB-readable remote URLs | Supported | Supported |
| Arrow tables/record batches, pandas frames, Polars eager/lazy frames, Loader | Supported; optional frame package needed only when supplied | Supported |
| Managed BatchSource streams | Supported with per-batch schema checks | Supported |
| Query, configured database sources, named connections and secrets | Supported; extensions may be needed | Supported |
| Configured Python source plugins, Excel, lossy CSV rejection capture | Requires Polars backend | Supported; Excel needs its DuckDB extension |
| Source budgets/retries, evidence/report limits, memory/spill byte limits | Supported; retries need replay-safe sources | Supported |
| Supervised execution and timeouts, required-check acceptance, local input hashes | Supported | Supported |
| Explicit spill-file count limit, custom rule/source plugins | Requires Polars backend | Supported |

Native validation accepts managed `BatchSource` streams so ownership and replay
safety are explicit. A raw borrowed Arrow `RecordBatchReader` is rejected by the
native backend. Whole-process RSS is not guaranteed by the native memory limit;
supervised process limits depend on the host platform. `rule_timeout_s` uses a
cooperative DuckDB interrupt per native rule stage; isolated execution and
`run_timeout_s` use the hard process supervisor. Full-backend rule cancellation
is also cooperative and arbitrary plugin code may outlive that deadline. The native rule parameter
vocabulary supports dataset selectors for INP-002–006 and INP-009. No backend
silently treats an unsupported configured check as passed.

```bash
qavf --help
qavf --version
qavf capabilities                         # JSON catalog for both backends
qavf schema --backend duckdb -o schema.json
qavf list-checks --backend duckdb --format json
qavf run --backend duckdb -c qavf.yaml
qavf run --backend polars -c advanced.yaml
```

`run`, `schema` and `list-checks` default to `--backend auto`, preserving the full
backend when Polars is installed and selecting the native backend otherwise.
The native schema describes the shared configuration plus supported native rule
and source vocabulary; cross-field and runtime capability constraints remain
runtime checks. It does not discover optional plugins. Explicit custom plugin
discovery in a native schema/list-checks request reports a capability error.

Existing command names and options remain available. Help for every command is
available in the core installation. `init`, `profile`, `compare`,
`compare-datasets`, `baseline` and `demo` load the optional backend only when
executed. Missing extras yield exit code 2 and installation guidance without an
import traceback. When Polars is already installed, a rejected native capability
instead recommends selecting `backend="polars"` / `--backend polars`.
