# Realistic performance benchmarks

Run the modular benchmark suite from the package root. It measures representative
QA/QC work with independently known outcomes, rather than accepting speed alone.
Benchmark implementation lives under `scripts/performance/`; case fixtures,
independent correctness oracles, execution policy, resource observation, planning
and reporting have separate modules.

## Run

Prepare the existing locked development environment once:

```bash
uv sync --all-extras --locked
uv run --no-sync python scripts/verify_release.py extensions
```

Start with the complete small matrix; it writes JSON and a sibling HTML report:

```bash
uv run --no-sync python scripts/benchmark_performance.py --output benchmark-results/smoke.json
```

| Profile | Rows per primary input | Measured repetitions | Separate warmups | Purpose |
| --- | ---: | ---: | ---: | --- |
| `smoke` (default) | 10,000 | 1 | 0 | Exercise every default workload and its oracle |
| `standard` | 250,000 | 3 | 1 | Repeatable measurements and revision comparisons |
| `capacity` (explicit opt-in) | 5,000,000 | 3 | 1 | Larger inputs on a suitably provisioned host |

```bash
# Standard matrix: 24 variants, 96 fresh workers for one source tree.
uv run --no-sync python scripts/benchmark_performance.py --profile standard --output benchmark-results/standard.json

# Scaling selected workloads; row scales must be distinct and at least 100.
uv run --no-sync python scripts/benchmark_performance.py --profile standard --rows 10000 250000 1000000 --cases wide_validation relational_validation --backends native lazy --formats parquet --output benchmark-results/scaling.json

# Compare a previous extracted QAVF root with this candidate, using identical inputs.
uv run --no-sync python scripts/benchmark_performance.py --profile standard --baseline /path/to/previous/QAVF --output benchmark-results/comparison.json

# Explicit larger run: restrict the workload before expanding the matrix.
uv run --no-sync python scripts/benchmark_performance.py --profile capacity --cases relational_validation --backends native --formats parquet --timeout 300 --output benchmark-results/capacity.json

# Current public streaming/grouped/statistical paths, including isolated workers.
uv run --no-sync python scripts/benchmark_performance.py --profile standard --rows 10000 250000 1000000 --cases streaming_regression grouped_statistics scalar_statistics --execution-modes in_process isolated --report-format none --timeout 300 --output benchmark-results/current-capacity.json
```

Use `--list` for supported cases/adapters and `--help` for all options. `--candidate`
selects another extracted source root. Both revisions run under the same interpreter
and dependency environment; this is a product-code comparison, not a dependency upgrade test.
`--repeat`, `--warmups`, `--threads`, `--seed`, `--partitions`, `--chunk-rows`,
`--evidence-rows` and `--evidence-bytes` make the workload explicit. Default threads
are two, partitions four, and fixture chunks at most 65,536 rows. Generation writes
bounded chunks instead of constructing full Python row lists.

`--memory-bytes`, `--cache-bytes` and `--spill-bytes` configure QAVF budgets; they do
not impose an OS process-RSS ceiling or guarantee successful execution at any scale.
The cache workload requires enough allocation for one retained frame but not two;
incompatible overrides fail explicitly. Capacity may use substantial RAM and disk,
especially eager inputs and regression. The host limits are recorded, not enforced
as automatic admission controls. `--timeout` bounds each fixture/worker job, not
the whole suite; on POSIX a timed-out worker's owned process group is terminated.

## Workloads and independent truth

The default matrix contains **24 variants** with in-process QAVF execution:

| Case | Coverage | Adapters |
| --- | --- | --- |
| `wide_validation` | Mixed numeric/string columns, missing/sentinel values, categories, regex, bounds and lengths | Native, lazy and eager × Parquet and CSV |
| `relational_validation` | Duplicate keys, skewed foreign keys, missing keys and orphans | Native, lazy and eager × Parquet and CSV |
| `grouped_reconciliation` | Exact integer totals/counts, missing groups and changed ledger amounts | Native, lazy and eager × Parquet and CSV |
| `hashed_validation` | Wide validation with file-content provenance hashing | Native Parquet |
| `regression` | Key reordering, added/removed rows, numeric tolerance, string/list/struct changes and bounded evidence | Parquet files loaded for an in-memory comparison |
| `cache_spill` | Real IPC spill, repeated reads, release, promotion, hit, metadata preservation and cleanup | Public `DataContext` lifecycle |
| `streaming_regression` | Distinct scalar Parquet files, added/removed/changed keys and values; requires `native_stream` public comparison | Public `compare_outputs`, in-process or isolated |
| `grouped_statistics` | High-cardinality grouped binary-half sums, independently known changed groups, exact rational evidence | Public `validate`, native Parquet, in-process or isolated |
| `scalar_statistics` | Two-point population with independently known z-score outliers | Public `validate`, native Parquet, in-process or isolated |

`--execution-modes in_process isolated` adds the isolated versions of the three
new cases; older cases support only in-process execution. The default omits these
extra variants. The capacity fixture has approximately one group per 16 rows,
so its group state scales with input cardinality. Grouped sums use exact rational
evidence, checked without converting to imprecise decimal approximations. These
workloads expose memory/time scaling; they do not promise constant-memory grouping.

Fixtures include partitioned files, deterministic defects, unequal row distributions,
and wider unused payload columns. Pure integer formulas establish validation counts
and reconciliation totals independently of QAVF. Regression has independently
calculated change counts. Cache checks verify exact values, physical spill files,
counter transitions and cleanup. The suite rejects incomplete/unassessed findings,
wrong outcomes and breached evidence bounds; matching an old version is insufficient.
The hashed case independently verifies file contents and expected fingerprints during
setup, then verifies QAVF's reported content identity after execution. This deliberately
warms those file pages before operation timing; it is not a cold-storage hash test.

The original `regression` case explicitly loads both files with the requested source thread setting,
then uses public in-memory comparison. It makes no out-of-core regression claim.
The cache operation includes columnar value summaries and lifecycle snapshots;
its timing is not an isolated cache-hit latency measurement. Each worker records
its precise operation boundary in `workload` metadata.

## Protocol and resource meanings

One fixture set per family/scale is shared unchanged by all repetitions and source
revisions. File inventories, content hashes, production-code hashes, benchmark-code
hash and environment versions are recorded. Final verification rejects changes to
inputs or code during a suite. Workers execute sequentially in fresh processes,
alternating baseline/candidate order. Warmups are separate fresh workers and excluded
from summaries; a failed warmup still invalidates its group. Fresh processes do not
clear OS page caches, so these results are **not cold-disk measurements**.

Polars and BLAS/OpenMP environment settings are applied before worker startup;
QAVF native connections receive the same requested thread allowance. The actual
Polars pool size is verified. No benchmark workers compete concurrently within
the suite, although other host activity remains uncontrolled.

| Measurement | Scope and limitation |
| --- | --- |
| Setup wall/CPU | Workload imports and input preparation; eager file reads belong here |
| Operation wall/CPU | Actual validation/comparison/cache workload; native/lazy input reads occur here; some workload summaries and bounded result serialization are included |
| Oracle time | Separate `verify` callback; its assertions do not enter operation timing |
| Report wall/CPU and bytes | Writing the per-workload reports after verification |
| Process elapsed | Controller-observed worker launch through exit, including imports, phases, verification and result publication; fixture generation is separate |
| Sampled start/peak/end RSS | Whole-process resident memory, including native libraries; sampled peaks are lower bounds and Linux RSS is approximate |
| Sampled process-tree peak RSS | Linux parent plus live descendants, including isolated workers in other process groups; summed RSS counts shared pages in each process and is not PSS |
| Waited-tree CPU | Parent CPU plus `RUSAGE_CHILDREN` deltas; includes children reaped during the phase, excludes live/unwaited descendants; null where unavailable |
| Lifetime RSS high-water mark | All preceding process work, including startup/setup; never subtract it to estimate phase allocations |
| Owned disk footprint | Distinct regular-file logical bytes and allocated blocks below the worker's operation directory, including the new cases' isolated-worker scratch; excludes shared fixtures, symlinks and directory metadata |
| Linux I/O deltas | `rchar`/`wchar` include cached/syscall traffic; `read_bytes`/`write_bytes` are kernel accounting, with deferred writeback; `cancelled_write_bytes` stays separate |

The sampler defaults to 10 ms. Short-lived allocations/files can escape observation;
partial disk snapshots during file deletion are disclosed. Monitoring adds work:
CPU includes monitor activity and I/O includes its procfs reads. Probe overhead is
reported but overlaps the operation, so it must not be subtracted as a correction.
Unsupported metrics remain `null`, never invented zeroes. QAVF cache/spill/hash
telemetry complements these measurements; execution counters are not physical reads.
Process-tree enumeration uses procfs-visible IDs, every live thread's children,
and a PPID-index fallback on kernels lacking `task/children`. Tree sampling can
miss short-lived or reparented descendants; unavailable platforms return null.
Partial samples are counted. The fallback scans visible process metadata, whose
CPU/procfs overhead is included in the reported sampler costs. Linux I/O counters
remain parent-only and must not be interpreted as process-tree disk traffic.
See [Python clocks][clocks], [resource usage][resource], [Linux process accounting][proc],
[Polars threads][threads] and [DuckDB memory/spill behavior][duckdb].

## Results, failures, and retention

JSON is updated after fixture preparation and each worker, then finalized with
identity checks and cleanup status. HTML summarizes the same evidence. Timeouts,
crashes, missing/duplicate repetitions, observer errors and failed oracles remain
visible; partial results never become performance wins. The CLI returns nonzero
when the suite fails. A killed controller may leave only its last saved checkpoint.

Summaries retain raw measured arrays, median, minimum, maximum and median absolute
deviation (MAD). Comparisons require final suite success, unchanged identities,
complete groups, matching outcomes and environments, and at least three measured
repetitions each. Ratios are candidate/baseline. Overlapping observed ranges are
inconclusive; separation is advisory, not statistical significance. Smoke results
establish execution/correctness only. There are no wall-time pass/fail gates.

Temporary fixtures, worker logs and per-workload outputs are removed by default;
the requested suite JSON/HTML remain. `--work-dir /path/to/scratch` selects the parent.
For debugging, add `--keep-work`; the result records the exact retained directory.
Retaining work keeps generated CSV/Parquet, logs, reports and possibly failed-job
temporary files, which can be large. After inspecting them, remove only that
recorded run-owned directory. `--report-format none|json|html|both` controls timed
per-workload QA report writing; the suite's JSON/HTML summaries are always written.
Non-QAReport workloads write their bounded result as JSON unless `none` is selected.

Benchmarks run locally and are separate from release assurance; they do not certify
another platform, live services or universal performance. Synthetic results apply
to their recorded inputs, hardware and dependencies; add domain-specific fixtures
before claiming performance for another workload. See [pyperf methodology][pyperf]
and [DuckDB correctness-checked benchmarks][benchmarks].

[clocks]: https://docs.python.org/3/library/time.html#time.perf_counter
[resource]: https://docs.python.org/3/library/resource.html#resource.getrusage
[proc]: https://www.kernel.org/doc/html/latest/filesystems/proc.html
[threads]: https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html
[duckdb]: https://duckdb.org/docs/current/guides/performance/how_to_tune_workloads
[pyperf]: https://pyperf.readthedocs.io/en/latest/run_benchmark.html
[benchmarks]: https://duckdb.org/docs/current/dev/benchmark
