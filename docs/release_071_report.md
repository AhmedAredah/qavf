# QAVF 0.7.1 implementation and verification

This local release addresses the 25 confirmed finding groups and six capability
recommendations in the additional 0.7.0 investigation. It includes the earlier
0.7.0 changes requested against 0.6.0. No GitHub push, pull request, release or
publication was performed.

All additions are generic library policies: exact canonical values, conditional
keys, grouped domains, source ownership, evidence, resource controls and explicit
backend capabilities. No MOVES schemas, column names, tokens, file layouts or
business rules were introduced.

## Changes and design

| Area | Findings | Result |
|---|---|---|
| Exact decisions | G01, G03, G04, G06, G07, G08, G10, G14 | Lossless FK alignment; correct NaN/infinity/N/A policies; exact numeric and null-fraction bounds; isolated decimal outliers; Unicode boolean and nullable-unit parity |
| Safe input admission | G02, G05, G17, G18, G19, G24, G25 | Independent row positions; strict source/header/name validation; simultaneous renames; protected borrowed readers; early row admission; consistent CSV diagnostics |
| Independent contracts | G09, G11, G12, G13 | Explicit regex dialect boundary; backend-neutral admission; per-contract error isolation; identities independent of display labels |
| Trustworthy evidence | G15, G16, G20, G21, G22 | Lossless temporal evidence; truthful unknown counts; source/configuration/plan provenance; condition context; bytes admitted before Python materialization |
| Working-set control | G23 | Required-column snapshots; original schema retained; dependency-aware ordering and prompt release; no repeated empty evidence/N/A scans |
| Core integration | D01, D02 | Polars-free CLI; capability catalog; shared connections, secrets, retries, batches, hashing, acceptance requirements and execution supervision |
| Optional domain policies | D03–D06 | Timezone/ns/DST, exact decimal precision/scale, conditional composite keys, grouped occurrence coverage, Unicode normalization and numeric separators |

The design separates run orchestration, immutable plans, source admission,
connection ownership, value/key/aggregate policies and evidence collection. The
optional backend reuses shared policies and lifecycle services. Default paths
avoid scalar UDF setup; special Unicode/temporal policies opt into it. Existing
public arguments and default backend selection remain available. Correcting
false decisions and unassessed outcomes necessarily changes those erroneous
results, as listed individually in CHANGELOG.md.

## Verification

- Full pinned environment: **5,744 passed, 29 skipped**, zero failures.
  Python 3.12.14, DuckDB 1.5.6, pandas 2.1.3, PyArrow 16.0.0, NumPy 1.26.4,
  Polars 1.32.0. Expected skips cover live integrations, the stats-absent path,
  and the real Windows Job Object contract; those external/platform paths were
  not executed here.
- DuckDB **1.3.1** targeted regression gate: **699 passed**, covering the new
  policies and existing keys, CSV, applicability, identity and resource contracts.
- Installed core wheels in separate environments with **no Polars, pandas,
  SciPy or pyproj**: passed on both DuckDB versions, in normal and `python -O`
  execution. These include CLI, owned batches, isolation, hashing, ns/timezone,
  decimal, Unicode, grouped-domain and conditional-key checks. Core dependency
  distributions were copied from the pinned installed packages for this offline
  environment; QAVF itself was installed from the built wheel.
- Audit closure: **224 outcomes** (56 cases × two backends × two DuckDB versions),
  **zero unexpected issues**. All 220 generated reports identify version 0.7.1;
  four expected configuration exceptions intentionally produce no report.
- **Ruff clean; mypy clean across 276 source files.** Wheel and source-distribution
  builds succeed. A final wording-only capability correction also passed its
  13 focused tests.

The five non-fatal full-suite warnings concern Polars schema resolution,
DuckDB's deprecated Arrow extraction spelling retained for the 1.3.1 floor, and
an expected small-sample statistical warning. Final logs are included in the
verification archive.

The audit reproductions are retained with their before/after closure assessment.
They distinguish corrected behavior from deliberately unsupported capabilities.
The test suites also cover additional integration findings exposed during this
work: exact missing fractions, empty aggregate populations, timezone-aware
nanosecond report values, lossless physical Arrow admission, named SQL-source
dependencies, case-insensitive and nested condition dependencies, omitted
resource defaults during isolation, and low-memory connection setup.

## Performance measurements

| Engine | Rows | Import + first call (s) | Validation (s) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|
| 0.6.0 full, automatic | 2,400 | 1.434 | 1.118 | 250.1 |
| 0.6.0 full, 2 native threads | 2,400 | 1.459 | 1.113 | 220.3 |
| 0.7.1 full, automatic | 2,400 | 1.217 | 1.217 | 250.2 |
| 0.7.1 full, 2 native threads | 2,400 | 1.220 | 1.219 | 223.1 |
| Plain DuckDB, 2 threads | 2,400 | 0.082 | 0.045 | 70.9 |
| 0.7.1 native, automatic | 2,400 | 0.363 | 0.363 | 101.4 |
| 0.7.1 native, 2 native threads | 2,400 | 0.410 | 0.410 | 103.1 |
| 0.6.0 full, automatic | 1,890,000 | 7.006 | 6.702 | 518.9 |
| 0.6.0 full, 2 native threads | 1,890,000 | 10.135 | 9.767 | 542.2 |
| 0.7.1 full, automatic | 1,890,000 | 5.028 | 5.028 | 494.0 |
| 0.7.1 full, 2 native threads | 1,890,000 | 7.033 | 7.033 | 433.3 |
| Plain DuckDB, 2 threads | 1,890,000 | 4.342 | 4.307 | 611.2 |
| 0.7.1 native, automatic | 1,890,000 | 3.485 | 3.485 | 599.8 |
| 0.7.1 native, 2 native threads | 1,890,000 | 4.891 | 4.891 | 629.3 |

| Engine | Additional public import (ms) | Warm default call (s) | Warm 2-thread call (s) |
|---|---:|---:|---:|
| before-060 | 319.524 | 0.526 | 0.513 |
| after-071-full | 0.072 | 0.298 | 0.303 |
| after-071-native | 0.068 | 0.113 | 0.113 |


The matched two-thread native result remains **13.6% slower and 3.0% higher
in peak RSS** than lean SQL on the large synthetic file. Automatic-thread
results use a different thread policy and should not be read as a matched SQL
comparison. The full backend improves its automatic large-file time by 25.0%
and peak RSS by 4.8% relative to 0.6.0.

| Core workload | 0.7.0 → 0.7.1 validation (ms) | Peak RSS (MiB) |
|---|---:|---:|
| 200k rows, one retained field | 61.6 → 62.2 | 104.7 → 107.7 |
| 200k × 32 fields, one checked key | 237.9 → 70.0 | 170.8 → 107.4 |
| Wide input, row budget = 2 | 139.9 → 37.0 | 156.0 → 88.3 |
| Two independent wide inputs | 384.1 → 87.5 | 241.7 → 109.5 |
| 50 × 256 KiB evidence, 1 KiB cap | 70.0 → 72.4 | 152.6 → 130.9 |

For the lighter 2,400-row single-key SQL fixture, warm core calls remain about
27 ms (27.14 → 26.74 ms); cold import-plus-call remains about 0.332 s.
Wide-evidence transfer into Python falls from **13,107,200 bytes to zero** while
all 50 violations remain counted exactly. Evidence admission adds a small time
cost (70.0 → 72.4 ms) while lowering peak RSS. These results do not imply a
universal speedup.

CLI help was separately measured in seven fresh processes: median 0.5383 s in
0.7.0 and 0.0358 s in 0.7.1. Profiles are included for original/updated warm
calls. They guided removal of unnecessary catalog scans and registrations,
connection setup overhead, repeated metadata reads, unused projections and
retained snapshots. Report serialization and exact contract evaluation retain
real costs because they are part of the library's promised output.


Every benchmark checks exact duplicate/violation totals and evidence correctness.
Measurements use synthetic inputs, fresh subprocesses and repeated medians; they
are not measurements of the downstream application's private data. Peak RSS is
whole-process memory, not a DuckDB allocation counter. Native memory and bytes
crossing into Python are reported separately where instrumented.

The updated plain-SQL baseline avoids Python parameter binding for the CSV path:
DuckDB 1.5.6 otherwise imports pandas/Arrow/NumPy solely to bind that parameter.
QAVF already uses a properly escaped SQL literal for this reader argument. The
fair baseline now uses the same mechanism. Historical results remain unchanged
and are identified separately; comparisons use the freshly measured baseline.

## Deliberate boundaries

- Native regexes outside the proven shared Rust/RE2 subset produce a clear ERROR;
  the optional backend provides the existing Rust Unicode semantics.
- Advanced statistical/spatial rules, custom Polars-based plugins, Excel and
  lossy rejection capture remain explicit optional capabilities. They are not
  silently omitted by the core.
- Exact timezone-aware Arrow nanoseconds require an explicit timezone contract
  for native UTC canonicalization. Unsupported nested or native Parquet aware-ns
  representations are rejected before precision loss; supply a lossless Arrow/managed BatchSource adapter or select the full backend
  for those inputs.
- Row limits bound admitted rows, not upstream SQL scans/sorts. DuckDB resource
  limits are not total-RSS limits. Cooperative rule deadlines do not forcibly
  terminate arbitrary Python callbacks; isolated execution supplies process
  deadlines for transportable inputs.
- Full performance parity with hand-written SQL is not claimed. QAVF additionally
  admits contracts, preserves exact identities, bounds evidence and builds a
  provenance report. The benchmark tables disclose workload-specific costs.

## Reproduction and installation

Install the included wheel with its declared core dependencies, or use the
`polars` extra for the full backend. The DuckDB floor remains 1.3.1; pandas is an
optional adapter. The pinned DuckDB 1.5.6 / pandas 2.1.3 / Arrow 16 environment was
used for the main compatibility run.

```
python -m pip install dist/qavf-0.7.1-py3-none-any.whl
python -m qavf --version
python -m qavf capabilities
python -m pytest
python scripts/benchmark_core_071.py --before /path/to/qavf-0.7.0 --repeats 3
```

`python -I scripts/verify_core.py` exercises an installed wheel in an environment
without Polars, pandas, SciPy or pyproj. The original CSV workload is reproducible
with `scripts/benchmark_validation.py`; exact invocation records and JSON results
are included in the verification evidence.
