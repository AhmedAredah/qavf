# QAVF 0.7.0 implementation and verification

Implemented locally against the supplied QAVF 0.6.0 snapshot. **Nothing was pushed
to GitHub or published.** The deliverable contains the complete updated source,
tests, changelog, installable wheel/source distribution, benchmark scripts and raw
measurement results.

## What changed

The investigation did not substantiate every original description literally.
Some capabilities already existed; those were retained and tested rather than
replaced with MOVES-specific behavior.

| Item | Verified finding and implementation |
|---|---|
| 1. Converted duplicate keys | Normal declared CSV conversion already handled integer aliases. Reader-forced text and other adapters exposed the gap. Keys now use shared canonical conversion across CSV, Arrow, pandas, Polars, Parquet and queries. `07`, `7`, `+7` compare equally. `7.0` is admitted through explicit `number_grammar.int_decimal: true`; strict integer spelling remains the default. |
| 2. Fixed/import overhead | Profiled import and warm calls. Made public validation imports lazy, cached bounded SQL syntax while keeping bindings and file resolution current, reduced duplicate-evidence work, fused native column predicates, removed redundant numeric trimming, and reused one immutable resource plan per run. |
| 3. Optional Polars | Polars is an optional extra. A real DuckDB/Arrow backend validates core contracts without Polars; this was tested using the installed wheel in an environment where Polars and pandas are absent. Existing advanced features remain available through `qavf[polars]`; `stats` and `spatial` include it. The pandas extra accepts 2.1.3. |
| 4. Memory/spill defaults | The old 1 GiB value was a spill allowance. Automatic planning now uses input metadata, key width, available memory/disk, connection reservations and reader headroom. Large keys reduce concurrency; explicit limits remain caps. The synthetic large-file probe now completes with an explicit 1 GiB run target and retains exact evidence. |
| 5. Duplicate evidence | Added `duplicate_rows_per_key` as a positive integer or `"all"`, retaining default 10. Sampled-group limits and byte budgets remain separate. Exact group/involved/surplus counts are retained, with protected completeness/truncation fields. |
| 6. Nullable composite keys | Nullable alternate keys already existed with null-exempt behavior. Added structured `unique_keys` contracts with `nulls: exempt`, `equal`, or `forbid`. Existing list forms retain their behavior; primary keys still prohibit missing components. |
| 7. N/A | Added column `na_values`, distinct from missing/invalid values, and `na_groups` with INP-009 for all-or-none applicability. Counts, conflicts, original evidence positions and adapter parity are tested. No application-specific tokens are built in. |
| 8. Allowed values/rules | Added opt-in `convert_options` for declared numeric equality and built-in `each_allowed_once` for exact domain coverage. Existing `normalize: [case]` plus `unique` already provided case-insensitive collisions; it remains built in and has regression coverage. |
| 9. Identity | `rule_id` already existed. Added `check_id`, scope, columns, parameters and constituent `checks` for summaries. Identity survives evidence trimming; older/plugin constructors retain truthful rule-level defaults. Single identities are not redundantly duplicated in serialized reports. |
| 10. Strict CSV | Prevents sniffing away malformed leading records; anchors unambiguous local dialects using a bounded first-record probe. Failure-only diagnostics distinguish logical record and physical line, including multiline and compressed files. Ambiguous/remote dialects retain native inference; explicit reader options remove ambiguity. |
| 11. Grammar/whitespace | Added grammar controls for integral decimal notation, underscores, exponents and special values, plus ASCII/Unicode/no-trim policies. Exact numeric conversion avoids floating-point rounding. Reader hints cannot normalize or round governed CSV values before grammar checks. |

Every item has regression coverage and a 0.7.0 changelog entry. Detailed contracts
are in `docs/upgrade_070.md`, `docs/key_contracts.md`, `docs/applicability.md`,
`docs/numerics.md`, `docs/check_identity.md`, `docs/strict_csv.md` and
`docs/resources.md`.

## API and architecture

Existing calls and schema list forms remain accepted. The additive selector
`validate(..., backend="auto"|"duckdb"|"polars")` controls execution. `auto` preserves
the existing backend when Polars is installed; it uses the native backend when
Polars is absent. Selecting `duckdb` explicitly makes the choice independent of
other installed packages.

The native backend supports core declared contracts over files, Arrow/pandas,
loaders and queries. Advanced statistical, temporal, spatial, regression, plugin
and full CLI execution requires the optional full backend. Unsupported configured
capabilities produce specific dependency/capability errors, not silent passes.
Native reports can emit finer-grained checks; clients should use structured
contracts rather than result-list positions.

The implementation separates source adaptation/ownership, canonical value
conversion, applicability, key policy, resource planning and bounded result
construction. Shared policy objects compile backend expressions. Resource plans
are immutable and reused within a run; evidence accumulators enforce limits while
streaming. These boundaries follow OOP/SOLID principles without allocating objects
per data row or copying entire datasets into Python for ordinary checks.

A representative native call is:

```python
from qavf import validate

report = validate(
    "input.csv",
    schema=schema,
    backend="duckdb",
    config={"engine": {"native_threads": 2}},
)
```

Two threads were a useful speed/memory balance in this benchmark, not a universal
recommendation. New grammar, N/A and null-equality behavior remains configurable.

## Before/after measurements

Three fresh subprocesses per configuration; medians shown. Deterministic synthetic
CSV: 1.89M rows, 14 integer columns, seven-column key, two rows per key. Each run
checks the same declared type/null/range contract, exact duplicate totals and
positions for 50 sampled key groups. The real downstream files were not supplied,
so these numbers do not claim to reproduce its exact workload.

| Large-file engine | Validation time | Peak process RSS |
|---|---:|---:|
| 0.6.0 default | 6.757 s | 577.6 MiB |
| 0.7.0 compatibility default | 5.047 s | 526.4 MiB |
| 0.7.0 native, two threads | 3.705 s | 569.8 MiB |
| Plain DuckDB, two threads | 4.417 s | 736.0 MiB |
| 0.7.0 native, automatic threads | 1.914 s | 779.6 MiB |

The compatibility backend improved large-file time by about **25%** and peak RSS
by about **9%**. Native automatic parallelism trades additional memory for speed;
the two-thread native result is the direct thread-matched SQL comparison.

For 2,400 rows, import plus first validation fell from **1.379 s** to **1.222 s**
on the compatibility backend and **0.347 s** on native execution with two threads.
Warm validation fell from **0.512 s** to **0.291 s**, or **0.095 s** on native.
The public `validate` import fell from about **338 ms to below 1 ms**; first-use
cost is included in the cold totals rather than hidden by lazy imports.

Measurements used Python 3.12.14, DuckDB 1.5.6, pandas 2.1.3, PyArrow 16.0.0,
NumPy 1.26.4 and Polars 1.32.0 on Linux. Native execution was also measured with
optional dataframe packages absent. CPU quota: eight; container memory: 8 GiB.
Caches were not forcibly flushed. Results are workload/environment-specific.

Reproduce with `scripts/benchmark_validation.py` and `scripts/profile_validation.py`.
Full commands, raw repetitions, profiles, memory-limit probes and interpretation
are in `docs/benchmark_070.md` and `docs/benchmarks_070/`.

## Verification

- **5,487 passed** in the full local non-integration suite; three expected skips.
  The optional spatial test was subsequently exercised after installing pyproj:
  its complete file passed **29 tests**. The remaining skipped scenarios concern
  a Windows Job Object and behavior when the statistics extra is absent.
- **528 passed** on DuckDB **1.3.1**, including every new feature and selected
  existing contracts. One Excel extension-download case was excluded because
  that external download is unavailable in this environment.
- **432 passed under `python -O`** for the new and cross-backend contracts.
- Installed core-wheel checks pass normally and under `python -O` on both
  DuckDB **1.3.1 and 1.5.6**, with Polars, pandas, SciPy and pyproj absent.
  Dependency consistency checks pass.
- Ruff checks/formatting pass. Mypy passes all **304 source files** and the public
  input typing fixture. A fresh cache and `--no-native-parser` avoid local mypy
  tooling/cache issues; no type-checking errors were suppressed wholesale.
- Generated documentation and lockfile checks pass. Wheel and source distribution
  were compared byte-for-byte with the package and declared development inputs.

These are local Linux/Python 3.12 checks. Live-service integration tests and the
Windows/macOS/Python 3.13/3.14 CI matrices were not run here. The original CI
workflows, omitted from the outer source archive, were recovered unchanged from
its bundled 0.6.0 source distribution. No GitHub write was involved.

The original upload was `qavf(4).zip`, SHA-256
`cec166336c32b05563e2f692dc3ce0a7b26a7f4f2a85b1a9a29bc611cceb540d`.
The baseline package files were verified against its included 0.6.0 wheel.
