# Execution and resource contracts

QAVF prioritizes complete, reproducible assessment. Resource limits never authorize
sampling the input or omitting configured checks. Insufficient resources produce
an unassessed result or an explicit resource error.

For a parent-enforced whole-worker deadline and OS process limits, use
`ResourceLimits(wall_time=30, address_space="4GiB", native_threads=2)` with a file
or configured source. Any process control selects isolated execution; the default
remains in-process. See [enforceable execution limits](execution_limits.md) for
supported inputs, exact scopes and failure behavior.

## Choose a budget

```python
from qavf import ResourceLimits, validate

report = validate(
    "readings.parquet",
    config="qavf.yaml",
    resources=ResourceLimits(
        target_memory="2GiB",
        cache_memory="256MiB",
        max_spill="4GiB",
        max_workers=4,
        evidence_rows=20,
        evidence_bytes="64KiB",
        report_bytes="16MiB",
    ),
)
```

Byte strings distinguish decimal `MB` from binary `MiB`. Explicit resource values
override configuration values. Callers can inspect `report.metadata.resource_usage`
and `report.metadata.query_plans` to see execution choices and tracked usage.

| Setting | Enforced scope |
|---|---|
| `memory_budget_bytes` | Target for tracked state, native buffer allocations and post-collection size checks; not total process RSS |
| `cache_max_bytes` | Shared admission ceiling for retained frame, pandas and aggregate caches plus coherent in-memory observations |
| `spill_max_bytes` | Separately enforced run-owned frame spill and reserved DuckDB temporary-storage partitions; excludes transport and arbitrary plugin files |
| `temp_directory` | Parent of an isolated directory created and cleaned up for each run |
| `max_workers` | Active rule calls; timed-out calls retain their slots until they stop |
| `native_threads` | Native thread allowance where supported; Polars' pool is process-global |
| `max_affected_records` | Evidence rows per result; independent of exact violation count |
| `max_evidence_bytes` | Encoded evidence/diagnostic budget per result; essential coverage names remain in the outcome ledger |
| `max_report_bytes` | Exact persisted JSON byte budget; essential outcomes that cannot fit raise a resource error |
| source `row_budget` | Maximum rows returned by the source; not a bound on input scanned |
| `run_timeout_s` | Whole isolated worker wall time from launch through exit; parent preparation, decoding and cleanup remain outside the deadline |
| `process_address_space_bytes` | Linux virtual address space or Windows committed memory per isolated process; not process-tree RSS; rejected on macOS |
| `process_cpu_time_s` | CPU seconds per isolated process, summed over its threads (user-mode only on Windows) |
| `process_file_bytes` | Linux/macOS maximum size of each worker-written file; not a total disk quota; rejected on Windows |

SQL/Query adapters and returned LazyFrames apply `row_budget + 1` before result
collection, allowing an exact over-budget decision without building the full result.
For a lazy reader with a row budget, `scan()` admits one capped coherent observation;
subsequent `collect()`, `project()` and `reduce()` calls use those admitted values.
The budget probe cannot observe different values from the accepted read. The
observation remains owned until dataset release or context close. Use `scan()` to
build plans passed to `collect()`; an unrelated caller-created plan cannot be
attributed to a named source by inspecting its result.
A Python loader that creates an eager frame has already allocated it before QAVF
receives it. Upstream joins, sorting and aggregate state may still exceed the row
result limit. Native reservations use the worker capacity of the selected execution
mode, including one worker for sequential execution and at most the scheduled tasks.
When a memory target is present, automatic rule scheduling allows at most one worker
per 32 MiB of target memory, with a minimum of one worker. The metadata-only planner
also estimates the largest key operator from declared key widths and file sizes or
known frame row counts, and reduces concurrency when that operator needs a larger
share of the target. Each worker reserves two possible native handles.
`max_workers` is a ceiling, not a requirement to run that many checks concurrently.
A directly constructed `DataContext` defaults to two native handles; advanced users
can explicitly change `connection_slots`. Reservations remain conservative fixed
ceilings within the selected concurrency, not dynamically resized live connections.

An omitted memory target and an omitted `spill_max_bytes` use
data-aware automatic planning. The legacy `spill_max_bytes` schema default of
1 GiB remains a baseline, rather than a cap on an automatically planned large
key workload. An explicitly supplied value, including zero or exactly 1 GiB,
remains a cap. `ResourceLimits(max_spill=...)` is always explicit.
Automatic memory leaves at least 128 MiB per potential native handle when host
capacity permits, so a tiny CSV is not assigned a limit smaller than DuckDB's
reader buffers. Automatic targets are capped against available host/container
memory, and automatic spill against available disk space. Linux reclaimable
filesystem cache is distinguished from live allocations. Unknown remote sources
and streams retain conservative defaults without consuming the stream to estimate
its size. Estimates are not promises that every operator can spill.

Automatic native query threads are also limited by each connection's memory
share (128 MiB of planning headroom per thread, with at least one thread).
Reducing Python worker count therefore cannot accidentally increase native CSV
reader concurrency beyond that share. An explicit `native_threads` value is
preserved. The actual native thread allowance is recorded in resource diagnostics.

The decision is recorded under `metadata.resource_usage.resource_plan`, including
input bytes, estimated key state, worker/connection capacity, effective budgets and
whether the memory/spill choice was automatic. No sampling or extra full scan is
performed to make the decision. Explicit memory and native DuckDB limits are never
increased by the planner.

One immutable plan is reused by scheduling and context allocation throughout a
run. Reducing the worker count therefore does not recalculate a smaller automatic
budget. The key allowance accounts for both the retained/native split and the two
possible handles per worker, subject to available capacity and explicit caps.

HTML presentation adds markup and escaping and has no independent file-size ceiling.
Writing or rendering reports after validation returns runs in the parent, outside
the isolated worker's deadline and file-size limit. Standalone report models have
no runner budget unless validated through the runner. Isolated validation and
comparison also enforce `max_report_bytes` on their encoded worker result.

Memory already owned by the caller, temporary native allocations, Python/plugin
allocations and the operating system's memory mapping are outside the retained-cache
accounting. A large eager load can allocate before the post-collection target is
checked. Use a process/container memory limit when a hard RSS ceiling is required.
A native buffer-manager setting does not cover every allocation or ensure that an
operator can spill. The opt-in `address_space` limit adds an OS virtual-memory
ceiling in the isolated worker; it does not change the meaning of the memory target
or provide a process-tree RSS cap.

## Plan and ownership

File, query and database sources can expose an immutable `ScanPlan` (SQL,
parameters and connection names). The context owns connections and temporary
storage. SQL-native checks retain projection/predicate pushdown. A supplied Polars
`LazyFrame` retains its lazy plan; `DataContext.scan()` exposes it to compatible
rules. No loader runs merely because its input descriptor is constructed.

A rule requiring a materialized frame calls `DataContext.get()`. The context
records that eager fallback, caches within the byte allocation, and can evict exact
frames to IPC files. Complete declared rule dependencies release frames, caches and spill files after their
last actual consumer finishes. Unknown plugins and cross-dataset input rules keep a
conservative scope. Cache reuse is bounded, so eviction can cause recomputation
of optional derived aggregates. `DataContext.project()` reads only selected columns;
`reduce()` keeps cached/spilled input alive through a scoped reduction. Owned IPC
projection avoids a complete Python bytes buffer and does not promote the whole
frame back into cache. Projected IPC read-byte counts are unmeasured, not zero.

On Polars versions providing incremental `collect_batches`, coherent observations
retain admitted chunks in available cache memory and spill incrementally when the
shared ceiling is reached. Native batch readers use the same admission policy.
This does not replay a volatile source. Older supported Polars versions retain the
exact IPC staging fallback. `retained_total_bytes` and `peak_retained_total_bytes`
include caches and observation leases; `retained_observation_bytes` reports the
portion retained by observations. Shared frame/observation leases stay charged until
their final owner closes, including after the dataset's cache entry is released.
Memory targets still exclude transient decoding/native operator allocations.

Native-safe repeatable file regression uses a source-owned join and bounded batches.
DuckDB manages join/sort state under its memory/spill policy. In-memory and exact
projected fallbacks still retain selected input frames and key indexes. Results
expose `execution` and `fallback_reason`; optional `ResourceLimits` also applies to
`compare_outputs` and `assert_outputs_match`.

An availability check on a known spilled frame reads only metadata. Materializing
that frame still reads its IPC file. Memoization locks exist only while a computation
or a waiter is active. Retained caches account for native Series/array buffers,
detach supported mutable values on insertion/retrieval, and refuse opaque objects
whose native allocation cannot be measured. Nested cache insertion carries the size
measured during ownership compaction; size-only IPC accounting uses a counting sink
without allocating another complete encoded payload. A plugin can return a supported value
representation rather than assuming a small Python wrapper implies small storage.


Source construction, normal loading, ad-hoc baseline loads and snapshot reads use
common row-budget, retry and normalization policies. Retries require an explicit
transient error and a retry-safe source. Native scan plans may execute more than
once across separate checks; volatile or externally changing input should use an
explicit consistent snapshot. A SQL assertion's count and evidence are derived
from one evaluation within that assertion. That evaluation keeps only the exact
count and the first `max_affected_records` rows in `ORDER BY ALL` order (a bounded
top-N over whole rows), not every failing row; a result whose rows DuckDB cannot
compare as whole rows is staged in temporary storage instead. With
`max_affected_records: 0` it is only counted.

### Projection and SQL dependencies

The native backend reads only the columns a dataset's checks use and loads
datasets that share no check as separate components, each released after its
checks. Row expressions and `where` conditions take part: DuckDB's parser proves
which columns a predicate names, and which datasets its subqueries read. A
dataset read by another's subquery joins that dataset's component and keeps every
column, as does the expression's own dataset (a subquery may bind any of its
columns). Expression evidence shows whole rows, so a dataset with an expression
keeps every column unless `max_affected_records` is `0`.

SQL whose reads cannot be proven (SQL assertions, table functions, qualified or
unknown relations such as views, unparseable text) keeps the conservative plan:
one component with every column of every dataset. `qavf explain --backend duckdb`
lists each dataset's planned columns, its component and why it keeps every
column, and `projection_fallback` names the SQL that forced the conservative plan;
a native report records the same reasons under `resource_usage`. On the Polars
backend the same analysis bounds how long an expression keeps other datasets.
Should a projection ever miss a column, the expression fails to bind and is
reported not assessed; it never passes.

`scripts/benchmark_local_performance.py` measures both in fresh processes. On one
Windows workstation (20 Parquet datasets of 200,000 rows and 9 columns, one local
expression; and one SQL assertion failing all 2,000,000 rows, 20 evidence rows):

| Workload | Before | After |
|---|---|---|
| Expression, native: components / retained columns / Parquet bytes | 1 / 180 / 1,213 MB | 20 / 28 / 76 MB |
| Expression, native: wall time / peak RSS | 12.2 s / 2.56 GB | 5.6 s / 0.42 GB |
| Assertion, native: wall time / peak RSS | 9.5 s / 2.02 GB | 10.0 s / 1.67 GB |
| Assertion, Polars: outcome / wall time / peak RSS | not assessed (spill cap reached) / 7.9 s / 0.50 GB | 2,000,000 counted / 9.4 s / 0.47 GB |

The native assertion's peak is mostly its loaded input; the failing rows themselves
are no longer staged. Before, the Polars backend reached its default spill cap while
staging the failing rows. Numbers vary by machine; run the script to compare a change.

### Streamed snapshots and borrowed queries

A coherent database snapshot (`connections.<name>.snapshot: true`, Polars backend)
reads every member inside one transaction. Each member streams as bounded Arrow
batches (65,536 rows) into its own owned IPC spill file under the run's spill
budget; nothing is published until the transaction commits, and then the group's
files are published together as spilled cache entries. Staging therefore holds
about one batch in memory at a time and the group is bounded by spill space, not
by `memory_budget_bytes`. The row budget is exact (the query is limited to the
budget plus one row), cancellation is checked between batches, a failed or
cancelled attempt deletes its files, and a retry restarts the whole group. A
plugin source that has no SQL relation (`snapshot_relation` returns `None`) is
still read whole by `read`, but is written to its spill file before the next
member is read. Checks that later read a member as a Polars frame still
materialize that member.

On the native backend, a `Query` on a borrowed DuckDB connection is copied into
the owned table one bounded Arrow batch at a time: each batch is registered,
inserted and unregistered before the next is read, with the same exact row
budget, cancellation and types as a complete copy. The caller's connection is
only read. On the Polars backend the borrowed result is still read whole, since
that frame is the retained dataset. `qavf explain` reports the two as
`staging: spill_backed_snapshot` and `staging: bounded_batch_copy`.

`scripts/benchmark_streaming.py` measures both in fresh processes (median of 2).
On one Windows workstation that was busy with other work (three members of
1,000,000 rows and 6 columns; a borrowed query of 3,000,000 rows):

| Workload | Before | After |
|---|---|---|
| Snapshot, 3 × 1,000,000 rows: outcome / wall time / peak RSS / spill | not assessed (spill budget exhausted) / 6.2 s / 1.43 GB / 0 | passed / 12.1 s / 1.91 GB / 480 MB |
| Snapshot, 3 × 300,000 rows: wall time / peak RSS / spill | 7.1 s / 0.63 GB / 0 | 10.4 s / 0.74 GB / 144 MB |
| Borrowed query, native, 3,000,000 rows: wall time / peak RSS | 7.4 s / 1.41 GB | 9.3 s / 1.03 GB |

The snapshot's peak is now its checks reading members back, not staging; a group
that fits in memory pays one spill write and read for that bound.

Scoped batch consumers keep readers, source identities and connection reservations
alive until complete consumption. A retry creates fresh consumer state; an incomplete
reader cannot return a successful result. The new grouped and regression reductions
allow multiple passes only for verified built-in file sources. Arbitrary lazy/query
inputs retain a coherent projected observation within these operations. Complete
source row budgets still apply when the returned aggregate has only one row.

Source definitions bind to independent run sessions; their environment and file
manifest are never reassigned to another run. File identity is verified around
native/eager operations and at finalization. A detected change makes assessment
incomplete. Metadata checks, and optional hashes, detect changes; they do not make
an arbitrary external writer atomic. Use immutable files or a database snapshot
when a consistent snapshot is required.


Eligible row-local column checks on a non-repeatable lazy input now share a single
streaming reduction. Each batch emits exact counts and bounded typed evidence;
merging by original row position preserves deterministic samples without retaining
the complete projected input. This fast path admits at most 64 compatible checks.
Global statistics, incompatible scopes and larger groups use the coherent
observation path. Partial reduction state can grow as batches × checks × evidence
rows; upstream operators and variable-width values add memory. It is not a hard
process-memory limit. A failed source is not replayed by this optimization.

Polygon assignment, neighbor graphs and optional map outlines also perform explicit
cardinality admission before expensive transfer or expansion. See [spatial data](spatial.md)
for scopes, defaults and refusal behavior. Bounds count logical work/retained results,
not every internal native operation.

## Deadlines and cleanup

A rule deadline requests cooperative cancellation and interrupts owned SQL work.
It cannot forcibly stop arbitrary Python functions or native extension calls.
Overdue work continues to occupy capacity; queued checks that cannot start are
reported as unassessed. The engine does not spawn replacement workers beyond the
configured limit. Resources still in use stay owned until the worker exits and
can safely clean them up.

With isolated execution, `run_timeout_s` covers startup, imports, discovery, all
checks, final verification and serialization through worker exit. The parent
terminates the owned process group on timeout and accepts no partial result.
`ExecutionLimitError` distinguishes a failed supervised execution from a complete
report that records caught per-rule resource errors and incomplete coverage.
The deadline excludes parent preparation, decoding and cleanup; it is not a strict
maximum for the complete API call's return time.

Borrowed connections remain caller-owned in in-process execution; isolated
execution rejects them. Run-created connections and temporary files are cleaned
up by their owner on success and failure. The isolated supervisor also stops
ordinary descendants and cleans its owned directory after worker termination.
Failure to stop a process group, parent termination or machine failure can leave
temporary files under the configured parent directory. The worker is not a sandbox
against plugins that deliberately escape its process group.

## Algorithmic limits

Complete-grid checks compute expected cardinality and compare observed distinct
keys without constructing the Cartesian product. Only bounded missing-key evidence
is generated. Input predicates count violations natively and materialize bounded
samples. Duplicate, foreign-key and key-set checks retain grouped keys in their
backend when both sides can execute together. Comparisons spanning native SQL and
in-memory/lazy backends still materialize projected key tables under the normal
budgets. Complete-grid checks also retain exact grouped-key state for rank-based gap
sampling. These are explicit capability boundaries, not constant-memory claims.
Regression comparison uses native joined batches where the source and exact types
permit them, and selected-frame joins for the remaining cases. Nested difference samples
consume both row and encoded-byte allowances as they are constructed. Zero evidence
allowances skip sample construction; large cells use bounded previews and changed-leaf
details while comparison and violation counts remain exact. Report JSON is emitted
incrementally from immutable values; the writer does not construct another complete
encoded report graph.

The level-shift screen remains an exact two-segment robust calculation with a
quadratic worst case. Its explicit `max_work` guard makes excessive work unassessed;
the allowance is shared across that rule's configured targets/groups in the run,
not reset for each group. It does not silently substitute an approximate method. Other statistical models
require a complete series/group in memory. Select appropriate columns and group
sizes and inspect unassessed coverage before accepting a run.

Scalar native float sum, mean and sample deviation now consume bounded batches with
exact numerical state. Grouped floating, weighted and order-sensitive kernels retain
their exact projected paths. STA outlier evidence contains raw checked values, group
keys, diagnostics and a position from the same projected observation; unrelated
columns are omitted.

## Bounded observation spooling

Small producer batches are detached and coalesced into independent IPC files.
A merge tree preserves order and avoids repeatedly copying a growing prefix.
Chunks flush at 65,536 rows or when the shared working allowance cannot admit the
next fragment. A large incoming batch is written synchronously without a second
retention copy. The working allowance is at most 16 MiB and, when a memory target
is supplied, at most one eighth of that target. It is separate from the retained
cache allowance; disabling caching does not disable bounded working buffers.
Transient native concatenation/serialization allocations are not a hard RSS cap.

`ResourceLimits(max_spill_files=4096)` (or `engine.max_spill_files`) caps live
owned IPC files, including files retained by leases or awaiting failed cleanup.
Exhaustion produces an explicit resource error, never a partial accepted dataset.
Increase it deliberately for workloads whose byte allowance admits more chunks.
SQLite work stores and DuckDB temporary files have their own lifecycle and byte
policies; the IPC file-count setting does not claim to count those files.

Resource diagnostics include `owned_spill_files`, `peak_owned_spill_files`,
`owned_spill_file_limit`, `spool_working_bytes`, `peak_spool_working_bytes`, and
`spool_working_budget_bytes`. Owned spill byte counters track logical bytes and
outstanding SQLite growth reservations. They exclude filesystem block rounding,
inode/directory overhead, and native temporary allocations. A pending SQLite
writer can temporarily reserve all remaining owned growth credit. Independent
reads proceed; concurrent writes may wait for credit. This conservative policy
avoids replaying partially executed SQL or allowing stores to overspend a quota.
