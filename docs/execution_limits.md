# Enforceable execution limits

Use `ResourceLimits` to run validation or output comparison in a fresh supervised
process on Linux, macOS or Windows. The parent can terminate a stalled worker,
and the worker applies the supported operating-system limits before importing
QAVF or native dependencies. Ordinary calls keep the default `in_process` behavior.

The following example uses Linux controls. On macOS, omit `address_space`; on
Windows, omit `file_size`. Unsupported controls are rejected before launching
the worker. `wall_time` alone selects supervision on all three platforms.

```python
from qavf import ResourceLimits, validate

report = validate(
    "readings.parquet",
    schema={"columns": {"value": {"dtype": "float", "ge": 0}}},
    resources=ResourceLimits(
        wall_time=30,
        address_space="4GiB",
        cpu_time=30,
        file_size="1GiB",
        target_memory="512MiB",
        max_spill="1GiB",
        native_threads=2,
    ),
)
print(report.verdict())
print(report.metadata.resource_usage["effective_process_limits"])
```

The example is a starting budget for a small file, not a capacity guarantee. The
Linux address-space allowance must fit the interpreter, libraries, thread stacks,
mapped files and working data. On Windows the same option caps per-process
committed memory, not all reserved virtual address space. Neither is an RSS limit.
A budget can fail during import before a dataset is read.
Use the returned verdict and assessment coverage to decide whether QA succeeded.

`wall_time`, `address_space`, `cpu_time` or `file_size` automatically selects
isolated execution. `ResourceLimits(isolate=True)` selects it without adding those
limits; inherited OS limits still apply. `isolate=False` conflicts with any of
these four controls and is rejected. Unsupported platforms raise an explicit
`ExecutionLimitError`; they do not silently fall back to in-process execution.

## Public and configuration names

Byte quantities in `ResourceLimits` accept integers or strings such as `"4GiB"`.
Configuration fields use integer bytes. `None` keeps the configured value.

| `ResourceLimits` argument | `engine` configuration field | Scope |
|---|---|---|
| `isolate=True` | `execution_mode: isolated` | Fresh worker supervised by the caller on Linux, macOS or Windows |
| `wall_time=30` | `run_timeout_s: 30.0` | Positive finite seconds from process launch through successful worker exit |
| `address_space="4GiB"` | `process_address_space_bytes: 4294967296` | Positive per-process virtual address-space ceiling on Linux (`RLIMIT_AS`), committed-memory ceiling on Windows; rejected on macOS; not RSS |
| `cpu_time=30` | `process_cpu_time_s: 30` | Positive integer CPU seconds per process: user and system time on Linux/macOS (`RLIMIT_CPU`), user-mode time on Windows |
| `file_size="1GiB"` | `process_file_bytes: 1073741824` | Nonnegative maximum size of each file written on Linux/macOS (`RLIMIT_FSIZE`); rejected on Windows; not total disk use |
| `native_threads=2` | `native_threads: 2` | Startup environment for supported native pools, plus backend settings |
| `max_spill="1GiB"` | `spill_max_bytes: 1073741824` | Shared owned frame, observation and indexed-work partition, plus separately reserved DuckDB temporary storage |
| `target_memory="512MiB"` | `memory_budget_bytes: 536870912` | Planning and tracked-allocation target; not an OS ceiling |
| `report_bytes="16MiB"` | `max_report_bytes: 16777216` | Maximum encoded report/result accepted from the isolated worker |

Linux and macOS process limits use Python's
[`resource.setrlimit`](https://docs.python.org/3.12/library/resource.html#resource.setrlimit).
QAVF lowers the soft and hard limits without relaxing a stricter inherited value
and disables core dumps. Windows uses a parent-owned Job Object with per-process
committed-memory and user-mode CPU limits; the worker joins it before importing
QAVF. The parent also polls members' user-mode CPU time while waiting.

The successful result records the effective controls. An omitted control adds no
new limit, and existing OS restrictions still apply. A reported value of `-1`
means that reported control is unlimited; it does not remove other OS restrictions.
Isolation alone does not create a memory or CPU ceiling. Ordinary child processes
inherit POSIX limits or join the Windows job, but their memory and CPU consumption
are not summed into a shared process-tree budget. A per-file ceiling permits
multiple smaller files; it is not a filesystem quota.

The equivalent Linux configuration can be used by `validate(config="qavf.yaml")`,
`QARunner(load_settings("qavf.yaml")).run()` or `qavf run -c qavf.yaml`:

```yaml
datasets:
  readings:
    path: readings.parquet
    schema:
      columns:
        value: {dtype: float, ge: 0}
engine:
  execution_mode: isolated
  run_timeout_s: 30.0
  process_address_space_bytes: 4294967296
  process_cpu_time_s: 30
  process_file_bytes: 1073741824
  memory_budget_bytes: 536870912
  spill_max_bytes: 1073741824
  native_threads: 2
```

Omitting `execution_mode` with a process control also selects isolation. Explicit
`execution_mode: in_process` with a process control is a configuration error.

## Inputs and entry points

Isolation transports descriptions of inputs so the child can open and own them.
It accepts paths, `File` descriptors, `Dataset` wrappers around supported inputs,
and configured sources. `Query` supports bound parameters and either no connection
or a configured connection name. Plugin packages, modules and file paths are
discovered in the worker. Configuration-relative file and plugin paths preserve
their usual meaning.

Live pandas/Polars/Arrow frames, lazy plans, `Loader` callbacks, `DataSource`
instances and borrowed database connections are rejected before conversion or
loading. Save them as an explicit file first, or describe their source through
configuration. Settings and query parameters must contain values supported by
QAVF's audit codec; objects are never transported with pickle. The request has a
16 MiB encoded-size ceiling. Request encoding writes into a capped byte buffer,
and binary parameter values are encoded in fixed-size chunks. Oversized binary
values are rejected before allocating their base64 representation. These bounds
cover transport assembly; caller-owned inputs and configuration normalization
remain outside the worker's memory limit.

`QARunner.run` supports isolation with configured discovery. Supplying live rule
or source registries, or a borrowed `DataContext`, is rejected. Constructing a
low-level `DataContext` with an isolated policy is also rejected: enforcement
belongs to the supervised entry point. Calling `plan()` or `plugin_status()`
separately performs work in the caller and is outside a later run's limits.

File comparisons use the same supervisor:

```python
from qavf import ResourceLimits, compare_outputs

comparison = compare_outputs(
    "baseline.parquet",
    "candidate.parquet",
    keys=["site_id"],
    resources=ResourceLimits(wall_time=30, native_threads=2),
)
print(comparison.equal)
print(comparison.execution_limits)
```

`assert_outputs_match` accepts the same policy and raises its usual
`AssertionError` for an assessed mismatch. `compare_frames` stays in-process.
Comparison's `execution` still identifies the data path, such as `native_stream`,
`typed_stream`, or `projected_fallback`; `execution_limits` describes supervision separately.
`report_bytes` caps the serialized comparison only in isolated execution.

## What the deadline covers

The wall clock includes fresh process startup, imports, plugin discovery, source
opening and reading, evaluation, input-identity verification, result serialization
and worker shutdown. An import, loader or native operation that does not cooperate
with rule cancellation can therefore still be stopped by the parent.

The deadline excludes caller-side imports and configuration normalization, bounded
request encoding and staging, result decoding, and parent filesystem cleanup.
Writing JSON or rendering HTML after `validate` returns, including CLI report
publication, happens in the parent and is outside the worker's limits. Initial
process creation may delay timeout handling; see Python's
[`subprocess` timeout contract](https://docs.python.org/3.12/library/subprocess.html#subprocess.run).
Process stopping allows two additional seconds of polling. Process-table
inspection, filesystem operations and OS scheduling are outside that allowance,
so `wall_time` is not a strict upper bound on the complete API call's return time.

`rule_timeout_s` remains a separate cooperative per-rule deadline. It requests
cancellation, interrupts owned SQL and records overdue work as unassessed. A
stalled in-process Python/native call can keep its worker slot. The isolated
whole-run deadline terminates the worker process instead of trying to kill a
Python thread. Without `wall_time`, isolation alone adds no wall-clock deadline.

## Failure and cleanup

A timeout, worker crash, failed OS-limit setup, invalid/missing result or failed
cleanup raises `ExecutionLimitError`. QAVF accepts no partial report or comparison
from such an execution. If a rule catches a resource failure and records it in a
complete report, that report retains the error and incomplete assessment status.
Neither mode samples data or silently removes checks to satisfy a budget.

The parent terminates the owned process group on Linux/macOS or Job Object on
Windows after success as well as failure, waits for its running members to stop,
then removes its temporary directory. These include ordinary worker descendants.
If stopping the owned processes fails, QAVF raises an error and leaves its
directory available for diagnosis. Parent termination or machine failure can
also leave files; the configured `temp_dir` identifies their parent directory.

Isolation is for trusted QA code. It is not a security sandbox: a plugin can access
the worker's environment and filesystem. On Linux/macOS, a process that deliberately
escapes the group with `setsid` is outside the cleanup guarantee. Use external process-tree
and filesystem controls when aggregate RSS, total disk use or hostile-code
containment is required.

## Resource accounting and module boundaries

The existing cache, spill, evidence and source row budgets continue to apply.
Frame and observation IPC writes, indexed work and baseline staging share the
owned-file disk partition; DuckDB receives its separately reserved temporary-storage
allowance. Owned-file growth is admitted against the shared quota. Transport files and
arbitrary plugin writes are outside this spill accounting. A native memory setting
does not cover every allocation; DuckDB documents this limitation in its
[out-of-memory guidance](https://duckdb.org/docs/current/guides/performance/oom).
See [resource contracts](resources.md) for the retained-memory and algorithmic
boundaries.

Successful validation exposes `execution_mode`, `effective_process_limits`,
`native_threads_at_startup` and `run_timeout_s` in
`report.metadata.resource_usage`. Supported native thread environment variables
are set before native imports. With automatic thread settings, the startup pool
allowance is capped at eight detected CPUs, while each backend connection receives
the CPU allowance divided among the planned rule workers, with a minimum of one.
An explicit `native_threads` value applies to both allowances. Isolation preserves
this same per-connection policy. Libraries that ignore those controls and custom
plugin pools are not an enforced process-wide thread-count limit.

The implementation separates policy from supervision and QA execution:

| Module | Responsibility |
|---|---|
| `qavf.inputs.resources`, `qavf.settings` | Public options, configuration and validation |
| `qavf.execution.inputs` | Reject live inputs before adapting them |
| `qavf.execution.policy` | Revalidate mutable settings before choosing execution |
| `qavf.execution.protocol` | Versioned bounded request/result encoding |
| `qavf.execution.supervisor`, `qavf.execution._windows_job` | Launch, whole-worker deadline, process-group/job ownership and storage cleanup |
| `qavf.execution.bootstrap`, `qavf.execution.os_limits` | Standard-library startup and OS limits before native imports |
| `qavf.execution.worker`, `qavf.execution.dispatch` | Run existing QA/comparison paths and reconstruct complete typed results |

