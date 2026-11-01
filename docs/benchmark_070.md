# 0.7 validation benchmarks

Measured on 2026-11-01 against the supplied, unchanged 0.6.0 source snapshot.
The data are deterministic synthetic CSVs, not the downstream application's
unprovided real files. Both sizes have 14 integer columns and a seven-column key;
each key occurs twice. Every column has type, non-null and nonnegative checks.
Every timed run asserts exact duplicate totals and all positions for the first
50 sampled key groups. An incomplete result aborts the benchmark.

## Repeated-process results

Medians of three fresh subprocesses per cell. Times are seconds; memory is peak
process RSS in MiB. The small-file column includes import plus first validation;
the large-file column separates validation from import (0.6 import adds about
0.34 s; 0.7 public validation import is below 0.001 s).

| Engine | 2,400 rows, import + validation (s) | Tiny peak RSS (MiB) | 1.89M rows, validation (s) | Large peak RSS (MiB) |
|---|---:|---:|---:|---:|
| 0.6.0 default | 1.379 | 304.8 | 6.757 | 577.6 |
| 0.7.0 compatibility default | 1.222 | 303.2 | 5.047 | 526.4 |
| 0.7.0 native, 2 threads | 0.347 | 114.3 | 3.705 | 569.8 |
| 0.7.0 native, automatic threads | 0.355 | 112.3 | 1.914 | 779.6 |
| Plain DuckDB, 2 threads | 0.320 | 164.7 | 4.417 | 736.0 |

The compatibility backend retains the existing automatic behavior when Polars is
installed. Its large-file validation improved by about 25%, while peak RSS fell
by about 9%. With two threads, the native backend was faster and used less peak
memory than the two-thread SQL comparison. Automatic native parallelism was
faster still, but used more memory: it is a speed/memory tradeoff, not a claim
that every 0.7 mode reduces RSS. Caller-owned data and an already-running app's
other allocations add to these isolated process measurements.

Both QAVF and SQL rows above use the same pinned environment: Python 3.12.14,
DuckDB 1.5.6, pandas 2.1.3, PyArrow 16.0.0, NumPy 1.26.4 and Polars 1.32.0.
The host exposes nine CPUs with an eight-CPU container quota and an 8 GiB memory
limit. No OS-cache flush was performed. SQL medians varied across batches; raw
repetitions, import time, total process time and resource plans are retained below.
Plain SQL materializes typed values and duplicate groups and collects equivalent
sampled positions; it does not construct QAVF audit/configuration reports.

A separate environment with Polars and pandas genuinely absent also passed the
same workload and the installed-core verifier. Its results are in `core_*.json`.
Removing the Polars requirement enables omitting it from an application build;
the installed Polars distribution here was 126,992,194 bytes. This is not a
measured PyInstaller saving or a promise of the downstream app's reported 161 MB.
Build without Polars, or configure the freezer's exclusions, to omit its binaries.

## Import and warm-call profile

The public `validate` import is measured after bare `import qavf`; bare import
was already lazy in 0.6. Each warm value is the median of five calls after one
initial call. Lazy imports defer work to first use, so the cold totals above are
also reported. Profiling overhead is excluded from these warm timings.

| Engine | Additional public validate import (ms) | Warm 2,400-row validation (s) |
|---|---:|---:|
| 0.6.0 default | 338.306 | 0.512 |
| 0.7.0 compatibility default | 0.093 | 0.291 |
| 0.7.0 native, 2 threads | 0.067 | 0.095 |

Profiling identified repeated SQL syntax analysis, native connection work,
duplicate evidence aggregation and repeated numeric trimming. Changes cache only
SQL syntax (bindings and filesystem resolution stay current), retrieve positions
for selected key groups, reuse one immutable resource plan per run, and gate
ASCII numeric spelling before casting without a redundant trim. Column
predicates are fused in the native backend. Report construction and exact byte
budget accounting remain visible costs; they are not disabled in the benchmark.

## Memory and spill checks

The 0.6 default of 1 GiB was a spill allowance, not a universal RAM default.
Subdividing native budgets among connections could starve a wide-key operation.
In the earlier 0.6 constrained probe, a 1 GiB run memory target produced three
ERROR results; a 128 MB native limit could retain exact counts but lose position
evidence. Raising spill alone did not fix that lost evidence.

The final 0.7 probes passed all three cases: native memory limit 128 MB with
automatic spill, native 128 MB with explicit 8 GiB spill, and an explicit 1 GiB
run memory target. Each retained exact totals and the sampled row positions.
These are this workload's results, not a guarantee that every DuckDB operator
can spill or that a buffer-manager allowance bounds process RSS. Explicit caps
are preserved; exhausted budgets produce errors or disclosed evidence truncation.

## Reproduce

```bash
# Run each source tree in the same pinned environment. The script generates data
# unless --csv supplies an existing deterministic fixture.
python scripts/benchmark_validation.py --source /path/to/qavf-0.6.0 --rows 2400 --output before-small.json
python scripts/benchmark_validation.py --source /path/to/qavf-0.7.0 --rows 2400 --output after-small.json
python scripts/benchmark_validation.py --source /path/to/qavf-0.7.0 --rows 1890000 --output after-large.json
python scripts/benchmark_validation.py --source /path/to/qavf-0.7.0 --backend duckdb --rows 1890000 --output native-large.json
python scripts/profile_validation.py --source /path/to/qavf-0.7.0 --csv synthetic_2400.csv --output profile.json
```

For the native backend's two-thread setting:

```python
report = validate(
    input_path,
    schema=schema,
    backend="duckdb",
    config={"engine": {"native_threads": 2}},
)
```

This setting is worth measuring on the application's own data; it is not a
universal thread recommendation. The script's `qavf_serial` mode also sets
`parallel: false`, which is relevant to the compatibility backend.

Raw measurements and profile summaries are in [benchmarks_070](benchmarks_070/).
`before_*` refers to unchanged 0.6, `after_*` to the compatibility backend,
`native_pinned_*` to native execution in the pinned environment, and `core_*` to
an environment without optional dataframe dependencies. Resource probes are
`resource_before.json` and `resource_after.json`.
