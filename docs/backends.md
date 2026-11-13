# Backends and installation

```bash
pip install qavf             # declared-contract validation on DuckDB and Arrow
pip install 'qavf[polars]'   # full backend: advanced rules, comparisons, plugins
pip install 'qavf[all]'      # every optional capability
```

`auto` uses the full Polars backend when Polars is installed and the native
DuckDB backend otherwise; `duckdb` selects the native backend even when Polars is
installed, and `polars` requires the full one. Because `auto` depends on what is
installed, the same configuration can run differently on two machines, so a
configuration can pin its backend:

```yaml
engine:
  backend: duckdb        # auto (default) | duckdb | polars
```

The backend is chosen in this order:

1. an explicit request: `--backend` on `run`, `explain`, `schema` and
   `list-checks`, or `validate(backend=...)`/`explain(backend=...)` in Python.
   An explicit `auto` also overrides the configuration;
2. the configuration's `engine.backend`, when it is not `auto`;
3. `auto`.

A pinned or requested `polars` without `qavf[polars]` fails when the
configuration is loaded, before any data is read, with the usual
optional-dependency error naming the setting (`engine.backend: polars
(qavf.yaml:4:3) requires the optional Polars backend; install 'qavf[polars]'`).
The chosen backend's own admission then applies: with `engine.backend: duckdb`
a check only the full backend runs is a capability error, as with
`--backend duckdb`.

`qavf explain` prints the backend and where it came from (`Backend: duckdb
(engine.backend: duckdb at qavf.yaml:4:3)`; `ValidationPlan.backend_source` is
`cli`, `argument`, `config` or `auto`), `qavf resolve --origins` names it in its
header (`backend` in JSON), and the run report records `metadata.backend` and
`metadata.backend_source` (the HTML footer shows both).

The native backend checks declared schemas, column limits, keys, foreign keys, row
expressions, SQL assertions and N/A groups over CSV, Parquet, JSON, Arrow, pandas,
loaders and queries, without importing Polars or pandas. Statistical, temporal,
spatial, reconciliation, regression and comparison rules, and custom plugins, need
`qavf[polars]`. A configured check the selected backend cannot run is reported as
a capability error with installation guidance; it is never treated as passed. See
[backend capabilities](backend_capabilities.md) for the full matrix and the core
CLI.

Both backends report identical results for the input rules (INP-001 to INP-009):
the same results, check identities, messages, counts, evidence rows (with
`violates`, and `_file`/`_file_row` for multi-file inputs) and passing-check
summaries. Switching backends changes only timings. The full backend also reports
its other rule families, which the native backend does not run. Two deliberate
differences remain: a regular expression outside the shared subset below is an
`ERROR` on the native backend, and a row expression or SQL assertion that cannot
be evaluated quotes the SQL engine's diagnostic, which quotes the SQL that backend
generated.

## Native execution guarantees

- Each contract is independent. A bad condition, expression, key or reference
  reports its own `ERROR` and the other contracts still run. When a fused query
  over several column checks fails, the checks are evaluated separately; a result
  is never guessed.
- Sums, means, medians and missing fractions are decided exactly: integers and
  decimals with DuckDB's exact reducer (with a bounded fallback on overflow), and
  floats with an exact accumulator. Non-integral exact values are shown as
  fractions (`mean 411/16 outside [0, 5]`). An empty population for min, max, sum,
  mean, median or standard deviation is not assessed.
- Floating columns reject infinity unless `allow_infinite: true`, also when the
  type is inferred. A missing key column is a structural failure.
- Regular expressions outside the subset that Rust and RE2 interpret identically
  report a specific `ERROR` on the native backend; select `backend="polars"` for
  those patterns.
- Exact violation and duplicate totals never depend on the evidence sample size or
  evidence byte limits.
- A foreign key between an integer and a float column is compared only when every
  value converts exactly; otherwise it is not assessed.
- Dataset views (`datasets.<d>.views`) are SQL views over the loaded dataset,
  `SELECT <columns or *> FROM "<d>" WHERE <where>`, so `sql` and `expression`
  checks read `"<d>.<view>"` with the same rows and columns as on the full backend.
- `connections.<name>.snapshot: true` reads every dataset of the connection in one
  transaction on both backends. The native backend copies each member into its own
  owned temporary table inside that transaction, plans the members as one
  dependency component, and follows the same rules as the Polars backend: a
  transient failure retries the whole group, a permanent failure makes only that
  member unavailable, and a failed or cancelled attempt rolls back and drops what
  it copied. The row budget stays exact per member.
- A cancelled run starts no further source attempt, and a retry delay ends as soon
  as the run is cancelled.

## Resources and timeouts

A row budget retains at most budget + 1 result rows and rejects the excess; it
does not bound scans, sorts or joins inside a SQL source. DuckDB memory and spill
limits bound the engine, not the whole process. `rule_timeout_s` interrupts a
native rule stage cooperatively; use `execution_mode: isolated` with
`run_timeout_s` for a hard process deadline. Borrowed handles and one-pass readers
are never consumed just to serialize them.

Reports record source descriptions, reader settings, renames, fingerprints, load
times, configuration and plan digests, the environment, rule provenance and
timings.
