# QAVF remediation and verification record

The original implementation below was delivered as 0.7.0. The 0.7.1 continuation
also protects custom batch source plugins from partial acceptance when their
cleanup manager suppresses an error. The shared batch engine now owns this
completion invariant; source adapters own conversion and reader cleanup.
Six additional regression cases exercise failure, interruption, admission,
schema, row-budget, and successful-stream boundaries. Current compatibility
results are in `validation/continuation/README.md` in the source ZIP; the initial
0.7.0 measurements and test counts below remain historical evidence.

This update implements the concrete defects and safe improvements in the supplied
review and remediation investigation. It preserves the existing facade and exact
QA/QC semantics, adds owned input and planning interfaces, and makes incomplete
assessment explicit. “Ultimate optimization” is not a measurable universal
property: time, memory, disk, exactness, and source repeatability can conflict.
The tables distinguish implemented changes from deliberately retained policies
and verification that still requires another platform or environment.

The source is based on the supplied `qavf(3).zip`, not a fetched Git revision.
`UPDATE_MANIFEST.json` provides per-file changes and digests. Local command
results and environment details are in `validation/`; historical Windows records
are retained separately with their original statuses. Workflow definitions do not
constitute successful release evidence.

Minimum-dependency investigation also exposed an existing backend compatibility
defect: DuckDB 1.2 infers zoned CSV timestamps as naive, despite loaded ICU.
Native probes and the existing QAVF timezone regression establish correct
behavior in 1.3.0 and 1.4.0. The declared DuckDB floor is therefore raised to
1.3.0. This avoids silently losing timezone metadata or adding an unconditional
extra scan. Other declared dependency floors are unchanged.

## OOP and SOLID application

| Principle | Concrete design in this update |
| --- | --- |
| Single responsibility | Source adaptation, data-free planning, outcome/evidence admission, resource ownership, and report rendering have separate modules and owners. `DataContext` and `QARunner` remain orchestration facades. |
| Open/closed | Existing rule/source/comparison registries and plugin protocols remain extension points. Grouped-measure fusion selects compatible built-ins and preserves plugin dispatch. |
| Substitutability | Native, eager, lazy, and typed fallback paths retain the same outcomes, counts, and typed identities; differential and adversarial regression tests cover volatile sources and resource exhaustion. |
| Interface segregation | Ordinary users provide paths or frames. Optional descriptors add reader/ownership policy; `BatchSource` has an explicit stream contract. Public static facades expose actual accepted input families instead of `object`. |
| Dependency inversion | Neutral parameters, scalar normalization, typed values, and snapshots do not depend on a particular rule or report implementation. Old `RuleParams` imports remain valid through an alias. |
| Encapsulation and ownership | Reusable numeric helpers own their snapshots. Leases remain owned until resources close successfully. Rendering state belongs to one document, and caller inputs/configuration are not mutated. |

Pure vectorized functions remain appropriate for numerical kernels. No extra
inheritance framework, generic manager hierarchy, or dependency-injection
container was introduced. Resource limits never authorize silently validating
only part of the input.

## Defect coverage

| ID | Implemented remedy | Main regression coverage |
| --- | --- | --- |
| F01 | Collision-safe internal CSV aliases preserve user fields and quoted/reserved names. | `test_source_remediation.py` |
| F02 | Missing-fraction completion uses compact first-observation state; no volatile-source reread after an assessed failure. | `test_rule_remediation.py` |
| F03 | One owned observation covers value-level conformance count, decimal/text and nonintegral checks. This is per assessment, not a whole-run snapshot promise. | `test_rule_remediation.py` |
| F04 | Floating operands widen to Float64; converted factors and finite-product overflow/underflow are checked. | `test_remediation_math_spatial.py` |
| F05 | Bucket arithmetic uses a wide integer domain with checked output narrowing and floor behavior before the epoch. | `test_remediation_math_spatial.py` |
| F06 | Shared canonical serialized-byte admission bounds comparison evidence and optional typed detail before publication. Omitted detail is disclosed. | `test_compare_remediation.py` |
| F07 | Composite cleanup attempts all children, retains failed owners for retry, and releases leases only after actual completion. | `test_engine_remediation.py` |
| F08 | Plugin evidence receives row/byte admission before row-location enrichment; row ordinal fields are explicit. | `test_engine_remediation.py` |
| F09 | Native-safe exact keyed comparisons reuse owned aligned observations in RAM or IPC; unsupported typed cases retain the exact indexed fallback. | `test_compare_remediation.py` |
| F10 | Internal CSV diagnostic roles are separate from user column names. | `test_source_remediation.py` |
| F11 | Conversion includes requested whole-row evidence columns; evidence disabled by policy keeps projection narrow. | `test_rule_remediation.py` |
| F12 | Contribution/rate group keys retain nanosecond datetime, duration, and time identities. | `test_compare_remediation.py` and temporal identity tests |
| F13 | Empty transformed grids return a typed empty cell relation. | `test_remediation_math_spatial.py` |
| F14 | Known eager loader row counts are checked before conversion/interchange allocation. | `test_source_remediation.py` |
| F15 | Projection and bulk spill/compaction occur outside global metadata locks. Staged publication commits ownership and catalog state together; failed staging cleans up. | `test_engine_remediation.py` |
| F16 | Positive rejection-storage budget bounds raw diagnostic entries; saturation produces explicit lower bounds and cannot establish a false pass. | `test_source_remediation.py` and report lower-bound tests |
| F17 | Exact multiple-testing families retain compact numeric state; rich evidence uses a bounded deferred spool and is hydrated only after admission. Evidence refusal cannot change the verdict. | `test_rule_remediation.py` |
| F18 | Shared lossless scientific-scalar normalization precedes settings/parameter coercion. Ambiguous NumPy temporal or lossy extended-float values are explicitly rejected. | `test_rule_remediation.py` |
| F19 | Chart/render IDs belong to a document instance, preserving deterministic independent/concurrent rendering. | `test_report_render_state.py` |
| F20 | Navigation anchors identify each result occurrence even when the same result object is reused. | `test_report_render_state.py` |

## Optimization coverage and gates

| ID | Delivered change or explicit design decision | Remaining boundary |
| --- | --- | --- |
| O01 | Promote an already owned IPC observation without recopying it for cache publication. | Other representation changes may still require conversion. |
| O02 | Preallocate final kNN compact adjacency buffers; bounded query blocks and exclusive internal buffer transfer avoid list and final-copy overhead. | Exact dense ties can still require expensive work; O(Nk) output is unavoidable. |
| O03 | Allocate exact seasonal slope values once and select the median in place with pair and numeric-byte admission. | Still O(P) storage and potentially quadratic pair work; no unproved external median or approximation is substituted. |
| O04 | Create global order index and deviation table only for paths that need them. | The proposed ordinary-z-score moments shortcut remains gated on threshold parity; exact established reducers are retained. |
| O05 | Share compatible grouped sum/mean and multi-quantile state while preserving component contracts. | Incompatible plugin or numerical semantics use the existing separate reducers. |
| O06 | Stream categorical groups, admit retained result state, and release per-group test arrays after use. | Complete statistical families and one admitted high-cardinality group still have real memory costs. |
| O07 | Auxiliary adapter reuses primary source policy and the parent owner; direct crosswalks accept frames/descriptors, projection, reader options and row budgets. | Serializable run settings use named datasets for live sources; nested live payloads and isolated transport fail with actionable guidance. |
| O08 | Compute derived report analysis once per HTML, canonical JSON, or acceptance operation. | No global cache retains an entire report across operations. |
| O09 | Registry keyed lookup/containment/length use direct transactional maps, without copying the whole visible registry per lookup. | Iteration intentionally remains a stable snapshot. |
| O10 | A small native distinct probe skips full-batch distinct work when category retention is already impossible; low-cardinality probing is disabled after its first full pass. | CSV leading-zero probing remains a separate correctness-preserving raw-text scan; typed inference cannot reconstruct discarded zeros. |

O04's moments kernel, optional spill compression, and merging raw CSV profiling
are conditional proposals from the investigation, not omitted correctness fixes.
They were not enabled without the required equivalence and end-to-end resource
evidence. Existing safe paths remain available.

## Broader review coverage

| ID | Resolution |
| --- | --- |
| A01 | Public `PairedValues` owns immutable snapshots; private one-shot preparation avoids unnecessary copies while the caller already owns stable arrays. |
| A02 | Extracted collaborators follow existing policy/lifecycle boundaries; orchestration facades remain stable. See the ownership map in `code_organization.md`. |
| A03 | Shared scalar/typed/evidence contracts plus backend differential tests preserve semantic consistency without forcing every backend into one conditional kernel. |
| A04 | `.pyi` public facades expose paths, descriptors, compatible frame protocols, and named datasets; negative static examples reject integers and column dictionaries. Lazy package imports are preserved. |
| A05 | `BatchSource` owns a context-managed factory, declared schema, batch admission, cleanup, coherent snapshot, and explicit replay/retry declarations. Borrowed half-consumed readers are not silently accepted. |
| A06 | `explain()` returns immutable data-free planning facts and unresolved work; optional explicit profiles compose existing checks and publish effective provenance. |
| A07 | Profiles combine declared input validity with configured regression/change. Documentation and tests distinguish stability from validity and relationships from coverage. |
| A08 | Quoting, canonical typed values, redaction, escaping, plugin transactions, and source identity remain in use. Python plugins/loaders are trusted code; resource isolation is not a hostile-code sandbox. |
| A09 | Complete verification workflows are included. Fresh local source/wheel, core-only, and targeted dependency-floor checks are recorded. Minimum-version testing corrected the DuckDB floor to 1.3.0. Full floor extension checks, Windows/macOS, other Python versions, and live services require their actual environments. Historical failed evidence is unchanged. |
| A10 | Mechanical Ruff formatting is applied after the functional changes, followed by lint, format checks, types, and tests. |
| A11 | Historical evidence is losslessly compressed with digest verification; caches/build intermediates are excluded from the delivery. Native backends remain because no measured supported replacement justifies their removal. |
| A12 | Benchmark catalog includes records, temporal/spatial, dirty CSV, report, resampling, many-file, and large-cell workloads with independent outcome checks. Zero target memory is rejected; Windows worker timeout uses Job-object ownership with a real-platform test. |

## Resource and measurement interpretation

Initial 0.7.0 verification of the executable package used a clean virtual
environment with the committed lock and test directories outside the checkout:

| Check | Result |
| --- | --- |
| Complete installed-wheel suite, ordinary Python | 5,002 passed; 29 skipped |
| Complete installed-wheel suite, optimized Python (`-O`) | 5,002 passed; 29 skipped; expected pytest warning about disabled ordinary assertions |
| Corrected dependency-floor regression selection (DuckDB 1.3.0) | 237 passed |
| Benchmark smoke matrix, 1,000-row scale | 32 variants passed independent oracles |
| Ruff lint and formatting | Passed |
| Mypy package and scripts | Passed, 278 files |
| Static public input examples, including expected rejections | Passed |

The 29 skips are 27 explicitly unconfigured live-service cases, one check that
requires SciPy to be absent (separately covered in the core-only environment),
and one genuine Windows Job-object contract. These are not silent failures.
At the initial delivery, Windows/macOS, Python 3.13/3.14, live services, and complete
minimum-version extension-enabled certification remained external gates. Consult
the continuation record for their current status. See `validation/README.md`
for exact commands, core-only checks, build verification, source/wheel identity,
and the metadata-only dependency-floor update after the full suites.

The preliminary source integration run exposed three stale fixtures: a missing
evidence-byte setting in a test double, an old lock-contention expectation, and
a cache benchmark tied to one eviction choice. Their replacements preserve
the original invariants and pass in both final complete suites. The old
dependency-floor attempt is retained as blocked/failed historical evidence;
it is not represented as passing current-floor certification.

Memory targets are cooperative admission policies, not hard RSS ceilings. Exact
family decisions, retained caller inputs, one large native value, parser/native
working memory, and statistical pairs may exceed a convenient constant-space
model. Exhaustion remains explicit. Bounded evidence is separate from complete
assessment; if optional evidence cannot fit, outcomes/counts remain intact and
truncation is disclosed.

Small local probes compare identical deterministic inputs in fresh subprocesses.
`validation/math_benchmark_results.json` records traced allocation peaks;
`validation/math_benchmark_timings.json` records three uninstrumented timing
runs per variant. Tracemalloc peaks are not process RSS and instrumented times
must not be interpreted as production timings.

| Probe | Original traced peak | Updated traced peak | Original median seconds | Updated median seconds |
| --- | ---: | ---: | ---: | ---: |
| Exact seasonal trend, 4,000 values in one season | 192,770,525 B | 64,478,356 B | 0.11097 | 0.08358 |
| Exact kNN, 12,000 points, k=8 | 5,883,936 B | 2,799,327 B | 0.37419 | 0.35080 |

The seasonal effect is identical in the probe; kNN returns 96,000 directed edges
and independent brute-force tie tests verify identities/weights. These small
measurements support allocation improvements, not universal performance claims.
The expanded benchmark harness records sampled RSS, temporary disk, independent
oracles, and input/source identities for reproducible workload comparisons.

Historical evidence originally occupied 18,571,050 uncompressed bytes across 19
files and now occupies 1,940,231 bytes as a verified ZIP. This saves extracted
source delivery space; it does not change runtime wheel footprint. No installed
dependency size reduction is claimed.

## How to reproduce validation

From a clean extraction, install the locked development dependencies, then run:

```bash
uv sync --locked --all-extras
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy qavf scripts
uv run --no-sync mypy --warn-unused-ignores tests/typing/inputs.py
uv run --no-sync python scripts/generate_docs.py --check
uv run --no-sync python -m pytest
uv build --clear
uv run --no-sync python scripts/verify_dist.py
```

For actual release certification, follow `release_assurance.md` on an immutable
Git checkout. The local audit environment required a mypy worker-discovery
workaround because container PID discovery fails; typing rules were unchanged.
The local validation summary records the workaround and exact environment.
