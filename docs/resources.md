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
        target_memory="2GiB", cache_memory="256MiB", max_spill="4GiB",
        max_workers=4, evidence_rows=20, evidence_bytes="64KiB",
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
| `cache_max_bytes` | Total retained frame, pandas and aggregate-cache allocation |
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
A Python loader that creates an eager frame has already allocated it before QAVF
receives it. Upstream joins, sorting and aggregate state may still exceed the row
result limit. Native reservations use the worker capacity of the selected execution
mode, including one worker for sequential execution and at most the scheduled tasks.

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

Native-safe repeatable file regression uses a source-owned join and bounded batches.
DuckDB manages join/sort state under its memory/spill policy. In-memory and exact
projected fallbacks still retain selected input frames and key indexes. Results
expose `execution` and `fallback_reason`; optional `ResourceLimits` also applies to
`compare_outputs` and `assert_outputs_match`.

An availability check on a known spilled frame reads only metadata. Materializing
that frame still reads its IPC file. Memoization locks exist only while a computation
or a waiter is active. Retained caches account for native Series/array buffers,
detach supported mutable values on insertion/retrieval, and refuse opaque objects
whose native allocation cannot be measured. A plugin can return a supported value
representation rather than assuming a small Python wrapper implies small storage.


Source construction, normal loading, ad-hoc baseline loads and snapshot reads use
common row-budget, retry and normalization policies. Retries require an explicit
transient error and a retry-safe source. Native scan plans may execute more than
once across separate checks; volatile or externally changing input should use an
explicit consistent snapshot. A SQL assertion's count and evidence are derived
from one materialized evaluation within that assertion.

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
columns are omitted. See [full-data loading](full_data_loading.md) for the capability
map, remaining materialization, and current measurements.
