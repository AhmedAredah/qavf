# Code organization and shared policies

Modules own one policy or lifecycle, and the package adds no runtime dependency
beyond its declared requirements. Ordinary paths and frames need no wrappers;
`File` and `Dataset` remain optional.

## Public entry points

Use `qavf.validate`, `qavf.File`, `qavf.Dataset`, `qavf.Query`, and `qavf.Loader`
for ordinary validation. See [the Python API](api.md) for the simplest call for
each input form.
`qavf.sources` exports source classes and parameter models; `qavf.regression`
exports comparison functions and result models; `qavf.compare` exports
`compare_datasets`, the comparison configuration and result models, and the four
component registries; `qavf.inference` exports the statistical tests, corrections,
intervals, effects, decision policies and the test registry. `qavf.rules.input_checks`
exports the input rule classes. Internal modules are implementation details;
the unreleased package does not retain aliases for former private locations.

## Ownership map

| Area | Modules | Responsibility |
| --- | --- | --- |
| Public input preparation | `api.py`, `inputs/` | A thin `validate` facade; separate descriptors, resource policy, source adapters, and normalization into one validated run request. |
| Source definitions | `sources/parameters.py`, `formats.py`, `frames.py` | Parameter validation, format inference and dataframe conversion. |
| Source execution | `sources/file.py`, `query.py`, `python.py`, `native.py` | File policy, shared SQL source lifecycle, Python loaders and native connection/budget primitives. |
| Source discovery | `sources/base.py`, `builtin.py` | Transactional registration and the explicit built-in source inventory. |
| Dataset catalog | `engine/catalog.py` | Source/declaration inventory and observed dataset metadata; keeps provenance independent of retained payloads. |
| Execution context | `engine/context.py`, `connections.py`, `sql.py` | Dataset reads and cache lifetime; worker-local connection ownership and cleanup; SQL analysis and bounded sampling. |
| Scoped reads | `engine/projection.py`, `owned_ipc.py`, `sql_read.py`, `source_read.py` | Selected reads, reductions and bounded batches; preserve owned-reader lifetime, retry reset, cancellation, source budgets and identity checks. |
| Coherent observations | `engine/observation.py`, `observation_plan.py`, `spill.py`, `sqlite_store.py` | One evaluated typed source projection, reusable derived reads, leased IPC and shared capped indexed work. |
| Native binding | `engine/relations.py` | Preserve verified local file-reader pushdown; retain existing binding semantics for other native source kinds. |
| Rule compilation | `engine/compilation.py` | Discovery, configuration and required-plugin validation, rule construction, and complete planning diagnostics. |
| Scheduling | `engine/runner.py`, `planning.py` | Rule execution, deadlines, actual consumer completion, resource release, and report admission. |
| Input checks | `rules/input_checks/` | Separate availability, columns, keys, references and expressions; private helpers own eager/lazy execution and evidence. |
| Rule failures | `rules/_contracts.py` | Consistent unassessed findings and exception isolation for independent contracts. |
| Key policy | `keys.py` | Lossless dtype compatibility shared by eager frames, execution relations and regression. |
| Reductions | `rules/_row_summary.py`, `_coordinate_sql.py`, `_grouped.py`, `_spatial_aggregation.py` | Coherent bounded summaries, selected normalized relations and safe grouped aggregation. |
| Accumulators | `aggregation.py`, `_float_accumulator.py`, `numeric.py`, `rules/_column_stream.py` | Exact numerical state, overflow bounds, safe integer narrowing and scalar stream dispatch. |
| Complete statistics | `statistics/ordered.py`, `rules/_distribution.py`, `_outlier_batches.py`, `_exact_grouped.py`, `_ordered_grouped.py` | Pure frequency mathematics, managed distribution/group indexes, bounded scoring and exact grouped accumulation. |
| Input profiling | `profiling/` | Bounded per-column state and common input preparation for API and CLI profiles. |
| Presentation | `reports/presentation.py`, `html.py` | Shared report context, sparse evidence selection and bounded HTML publication. |
| Regression | `regression/_plan.py`, `_inputs.py`, `_streaming.py`, `_comparison.py` | Schema-only planning (including baseline column maps), policy-preserving dispatch, native joined batches and shared exact comparison state; typed indexed alignment and bounded baseline staging; numeric/cell/evidence modules retain their policies. |
| Dataset comparison | `compare/spec.py`, `align.py`, `measures.py`, `comparators.py`, `significance.py`, `levels/`, `results.py`, `engine.py`, `registry.py`, `api.py` | Configuration validated at load; ordered alignment steps; per-group measures over the shared exact reductions; comparators and materiality; each measure's test, its natively reduced inputs and the family decisions; one module per level; typed bounded results; one observation and aligned snapshot per side; component registries with entry-point discovery; the public facade. Keyed record classification is the regression engine's. |
| Statistical inference | `inference/results.py`, `paired.py`, `independent.py`, `equivalence.py`, `categorical.py`, `counts.py`, `trend.py`, `spatial.py`, `multiple.py`, `intervals.py`, `resampling.py`, `pvalues.py`, `effects.py`, `decision.py`, `registry.py`, `components.py` | Arrays in, typed results out, independent of comparisons: one module per test family; corrections; intervals, seeded resampling and the work budget; effect sizes; decision policies; test components and their registry. SciPy supplies every distribution function. |
| Component registries | `plugins/registry.py` | One registry implementation, discovery transaction and `plugins.entry_points` scope for comparison components and statistical tests. |
| Typed values | `typed_values.py` | Lossless temporal normalization, checked casts with per-row loss flags, exact numeric scaling. |

Concrete SQL source classes share their loading, budget, snapshot and connection
ownership behavior through `SQLSource`; each supplies its own query construction.
`DataContext` composes a connection owner instead of inheriting connection behavior.
Input rule categories and comparison stages use explicit function dependencies.
Dataset comparison components and statistical tests share one small registry
class (`plugins/registry.py`, re-exported by `compare/registry.py`) that mirrors rule
and source discovery; there are no mixin hierarchies.

## Input and planning boundaries

Input preparation validates all names and contracts before execution. It does
not invoke a loader, collect a lazy frame, execute a query, or write a report.
Existing Pydantic models remain the authority for contract and source-option
validation; YAML/JSON parsing preserves duplicate-key checks, environment
expansion, configuration-relative paths, and omitted-versus-explicit fields.
No shape guessing distinguishes column dictionaries from named datasets.

The catalog owns dataset state independently of the frame cache. Metadata
queries do not load files or read spill payloads. A catalog update uses the
context's existing synchronization boundary; introducing a second state owner
or lock would make provenance diverge from execution. The context still owns
when sources execute, retry, verify identities, and release resources.

Rule compilation builds a complete plan and its diagnostics locally. The
runner publishes them after successful compilation; a failed compile cannot
leave a partly updated exclusion/configuration ledger. Runtime dataset-lifetime
planning stays separate because its responsibility is actual consumer completion,
including workers that continue after a timeout notice.

Rule discovery ignores re-exported classes and registers the 29 built-in
rules. Source discovery registers the same four source kinds inside its existing
transaction, so a later conflict cannot publish a partial built-in registry.

## Consolidated behavior

- Eager and lazy key compatibility use one dtype policy. Expensive exact-float
  proofs run only for combinations that require them. Eager integer compaction
  remains eager-only; lazy relations do not acquire compaction scans.
- Reconciliation and column checks share accumulator selection and conservative
  exact-sum overflow guards. Caller-specific missing-value counts and collection
  boundaries are retained.
- Foreign-key partial-null, cardinality and orphan findings share construction
  helpers across eager and grouped paths. Required-column and row-check findings
  also use shared construction.
- Temporal, statistical, reconciliation and spatial contracts share exception
  isolation and unassessed-finding construction. Original messages, detail styles,
  severities and continuation behavior remain intact. Process-control exceptions
  are not swallowed.
- Regression comparison uses neutral key and frame modules instead of importing
  implementation helpers from rule or built-in-source modules.

Similar-looking numerical operations are not interchangeable. Regression tolerance
comparison retains its own semantics in `regression/_numeric.py`; rule-level exact
arithmetic remains in `numeric.py` and `aggregation.py`. Native and typed-lazy
source paths also remain distinct where lossless type preservation requires it.

## Resource and correctness boundaries

The existing shared column batches, grouped-key summaries, operation-scoped file
verification, metadata reuse and hash counters remain in place. This organization
work introduces no additional persistent digest store or unrelated dataframe
copy. Grouping, joins and exact fallbacks still have the resource costs documented
in [scan optimization](scan_optimization.md) and [resource limits](resources.md).

The connection owner records a created native view immediately. If subsequent
metadata or fingerprint work fails, cleanup still removes that registration.
Worker-local ownership, attachment identity, cancellation and retry policies remain
explicit and independently tested.

Smaller modules improve ownership and maintenance; a larger module count alone
does not establish lower runtime or disk usage.

## Verification

Behavioral tests cover source-registration rollback and connection ownership,
failed view preparation, eager/lazy key policy parity and proof counts, exact sums,
contract error isolation, input rule discovery and foreign-key evidence parity,
alongside the assurance and scan suites. See [release assurance](release_assurance.md)
for the complete release matrix and [full-data loading](full_data_loading.md) for
the ownership and capability map.
