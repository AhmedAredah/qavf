# Native execution and compatibility in 0.7.1

The API remains `validate(data, schema=..., config=..., backend=...)`. `auto`
continues to select the full backend when Polars is installed; `duckdb` explicitly
selects the core. Core validation, its CLI, JSON/HTML reports and process
supervision work without Polars or pandas. Dependency floors are unchanged.

## Responsibilities and resource behavior

`NativeRunner` owns one run and one DuckDB connection. `NativePlan` selects
required columns and groups dependent inputs; `InputReader` owns source
admission, snapshots and releases. Value/key/aggregate policies compile exact
contracts. `DatasetChecks` executes independent contracts, and `ResultCollector`
plus `SqlEvidenceReader` own bounded, lossless evidence. Source connection setup,
process supervision and audit serialization are shared services, independent of
optional dataframe imports. There is no application-specific rule vocabulary.

Structural checks see the original column names/types even when only a subset of
values is retained. FK-connected inputs share their snapshots. Arbitrary SQL
assertions/expressions conservatively retain all relevant data; named source SQL
dependencies are ordered before their consumers. Independent snapshots and their
registrations are released after their last consumer. Exact duplicate/violation
totals never depend on the sample cap or available evidence bytes.

A row budget limits retained result rows to at most budget + 1 for admission and
rejects excess rows. It does not bound upstream scans, sorts or joins in a SQL
source. DuckDB memory/spill limits remain engine limits, not total process-RSS
caps. Very small explicit limits are preserved. Unicode/temporal scalar helpers
are registered only for policies requiring them, without scanning the function
catalog or importing an optional dataframe library.

`rule_timeout_s` cooperatively interrupts a native rule stage and discards any
partial assessments from that stage. Arbitrary Python work may finish before the
interruption takes effect. Use `execution_mode: isolated` and `run_timeout_s` for
a process deadline; only transportable descriptors are accepted there. Borrowed
handles and one-pass readers are never consumed merely to serialize them.

## Compilation and outcome meaning

Input configuration syntax admission does not depend on installed optional
packages. Binding/function errors belong to the selected backend and the
individual contract. Failure of one fused query triggers separate evaluation so
independent checks continue. Comparison alignment retains its explicitly Polars
expression dialect.

Rust and RE2 regex engines do not have identical Unicode semantics. The native
backend accepts the proven shared subset and reports a specific ERROR for other
patterns; it never silently substitutes ASCII `\\w` for a Unicode class. Select
`backend="polars"` for those patterns. An ERROR is unassessed, not zero violations:
its count is not exact and its evidence completeness is unknown.

Undefined numeric aggregates (including all-null populations) are ERROR outcomes
in both backends. Empty distinct cardinality and missing-fraction contracts keep
their defined zero behavior. Declared N/A values remain valid and have separately
counted exemptions. Partial successful key contracts have their own identities;
they do not label an otherwise failing dataset clean.

## Provenance and compatibility limits

Reports include source descriptions, effective reader settings, renames,
fingerprints, load timestamps, configuration/plan digests, computing environment,
rule provenance and timings. Optional SHA-256 verification checks source identity
at read/finalization boundaries; it does not claim an immutable remote snapshot.
Native stage elapsed time is divided among its results and this timing scope is
explicit in metadata.

The capability matrix distinguishes core features from advanced optional rules,
custom Polars-based plugins, Excel readers and lossy rejection capture. Unsupported
configured work is rejected explicitly. See [backend capabilities](backend_capabilities.md),
[value policies](value_policies_071.md), [key contracts](key_contracts.md),
[source admission](source_admission.md), and [exact contract execution](contract_execution_071.md).
