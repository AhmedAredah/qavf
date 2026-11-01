# Backends and installation

```bash
pip install qavf             # declared-contract validation on DuckDB and Arrow
pip install 'qavf[polars]'   # full backend: advanced rules, comparisons, plugins
pip install 'qavf[all]'      # every optional capability
```

`validate(..., backend="auto")` is the default: it uses the full Polars backend
when Polars is installed and the native DuckDB backend otherwise.
`backend="duckdb"` selects the native backend even when Polars is installed, and
`backend="polars"` requires the full one. The CLI `run`, `schema` and
`list-rules` commands take the same choice as `--backend`.

The native backend checks declared schemas, column limits, keys, foreign keys, row
expressions, SQL assertions and N/A groups over CSV, Parquet, JSON, Arrow, pandas,
loaders and queries, without importing Polars or pandas. Statistical, temporal,
spatial, reconciliation, regression and comparison rules, and custom plugins, need
`qavf[polars]`. A configured check the selected backend cannot run is reported as
a capability error with installation guidance; it is never treated as passed. See
[backend capabilities](backend_capabilities.md) for the full matrix and the core
CLI.

## Native execution guarantees

- Each contract is independent. A bad condition, expression, key or reference
  reports its own `ERROR` and the other contracts still run. When a fused query
  over several column checks fails, the checks are evaluated separately; a result
  is never guessed.
- Sums, means, medians and missing fractions are decided exactly: integers and
  decimals with DuckDB's exact reducer (with a bounded fallback on overflow), and
  floats with an exact accumulator. Non-integral exact values appear in metrics as
  rational strings (`"1/3"`). An empty population for min, max, sum, mean, median
  or standard deviation is not assessed.
- Floating columns reject infinity unless `allow_infinite: true`, also when the
  type is inferred. A missing key column is a structural failure.
- Regular expressions outside the subset that Rust and RE2 interpret identically
  report a specific `ERROR` on the native backend; select `backend="polars"` for
  those patterns.
- Exact violation and duplicate totals never depend on the evidence sample size or
  evidence byte limits.
- A foreign key between an integer and a float column is compared only when every
  value converts exactly; otherwise it is not assessed.
- Evidence rows carry `_row`. The `_file`, `_file_row` and `violates` columns of the
  full backend are not added.

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
