# QAVF 0.7.2: implementation of the comprehensive 0.7.1 review

Date: 2026-10-31. Baseline: `QAVF_0.7.1_Updated_Code_and_Tests.zip`, SHA-256
`25909d9e83b21acc1f70c8722074529b8ae34e3ae651289cbbf75709184d2f8f`.

This is a complete source update, with new tests and reproducible validation
scripts. It preserves existing public entry points and source-plugin hooks.
It addresses the concrete defects and design boundaries identified in R01–R08.
R09 is only complete for the environments actually recorded in the accompanying
[validation record](../validation/review_072/README.md); unexecuted platform and
service gates remain open. No universal optimum in time, memory and storage is
claimed. Those objectives conflict and depend on workload and exactness policy.

## Concern-to-change map

| Finding | Implementation | Regression evidence |
|---|---|---|
| R01: input/report collision | `sources/local_files.py` defines literal-first, hidden-inclusive discovery. File readers and `ProtectedInputs` use it. The guard retains preflight inputs and rechecks current matches and file identity immediately before atomic replacement. | Both CLI commands; literal brackets, question marks, asterisks, hidden and ordinary glob matches; links; a new match created during publication. |
| R02: one spill file per producer batch | `ObservationStorage` owns storage admission. `BatchSpool` detaches small fragments and merges equal-sized groups within byte/row/fragment bounds; complete chunks are published through one reusable writer implementation. An explicit live-file cap bounds manifest metadata. | Exact values/order/types across partitions and mutable producer wrappers; byte/file accounting and cleanup; before/after timing, RSS and disk measurements in validation. |
| R03: writer lifecycle and swallowed errors | `SpillFile` grants one writer lease, prevents reuse, keeps files alive until active handles finish, reserves before writes and reconciles partial writes. The first write/flush error remains latched and causes context exit to fail. | Closed/repeated/nested writers, swallowed quota failures, short and partial writes, failed flush, owner closure during writing, and leased-file limits. Existing deletion-failure retry tests remain. |
| R04: inaccurate structural typing | Replace `PandasFrame`/`ArrowTable` metadata lookalikes with real conversion protocols. Runtime admission supports both nominal installed frame types and implemented table interchange. Annotate `explain` and `profile_input` with the same input union. | Positive protocol and negative lookalike mypy examples, runtime conversion/rejection, existing installed public API contracts. |
| R05: partially imported optional modules | Admission uses only optional-module attributes already available as actual classes. It does not equate presence in `sys.modules` with completed initialization. | A real first pandas import paused behind a deterministic barrier while Arrow admission proceeds; partially published module and ordinary conversion tests. |
| R06: context and reader coupling | `SourceReadServices`, `SqlReadPolicy` and `NativeReadServices` expose focused dependencies. Scoped readers no longer inspect private context fields. `BatchReader`/`BatchSink` and the source compatibility adapter isolate batch consumption. `LocalManifest` owns frozen file identity; `ObservationStorage`/`BatchSpool` own admission and buffering. | Small fake-reader/sink, SQL-policy, manifest and spool tests run without a full context. Existing source, retry, identity, lifetime and plugin tests continue to define compatibility. |
| R07: distributed column semantics | `ColumnPlan` and `ColumnOutcome` hold normalized contracts, applicability and finding construction. Separate Polars and native compilers retain vectorized execution and explicit unsupported-operation fallbacks. Eager evaluation shares the same outcome contract. | Cross-backend comparison of every result/evidence field except timing for adjacent float boundaries, large integers, Decimal, nanosecond time, Unicode and conditional checks; existing broader numerical/regex/unit tests retained. |
| R08: global allocation lock during SQL | SQLite authorizer actions identify proven read operations. Reads bypass growth admission. Writes reserve credit briefly under the allocator lock, execute outside it under a page limit, and reconcile actual growth. Result-bearing writes are refused before execution because their cursors could outlive credit. | Barrier-controlled independent reads, simultaneous writes under one quota, read-only CTEs at full quota, refused `RETURNING` writes, and existing indexed-statistics/disk-full tests. |
| R09: incomplete deployment/performance evidence | Add review-specific regressions, larger measurements, current source/distribution identity and fresh local gates. Retain strict release scripts and explicit environment requirements. | The validation record lists exact executed commands, versions, artifacts and remaining gates. Workflow existence is not counted as a pass. |

## OOP and SOLID decisions

- Single responsibility: source binding/publication remain facade policy; scoped
  readers execute; storage objects own their reservations and cleanup; manifest
  identity and column outcomes have independent contracts.
- Open/closed and interface segregation: existing `DataSource` plugins retain
  their hooks. An adapter supplies only the owned-batch capabilities needed by the
  batch consumer. No plugin is required to inherit another manager hierarchy.
- Substitutability: capability admission, exact schema checks, row limits and
  complete-stream requirements remain enforced for plugins and built-in sources.
- Dependency inversion: readers accept explicit service/policy objects and
  callables; independent fixtures demonstrate those boundaries without assembling
  the engine. Facades remain composition roots, so their total line count alone
  is not a success criterion.
- Performance: abstractions operate per source, batch, check or operation.
  There is no new Python virtual call for every data value. Native/vectorized
  paths, exact fallbacks and bounded evidence remain intact.

This is an incremental internal design improvement. The compatibility facades
still contain orchestration, and specialized CSV diagnostics, content hashing,
snapshot handling and backend compilation retain their existing focused helpers.
Further splitting should follow measured coupling or an independently changing
policy, rather than a target class size.

## Input contract and usage

```python
from qavf import ResourceLimits, explain, validate

report = validate(
    "measurements.parquet",
    schema={"columns": {"temperature": {"ge": -50, "le": 80}}},
    resources=ResourceLimits(
        cache_memory="128MiB",
        target_memory="512MiB",
        max_spill="4GiB",
        max_spill_files=4096,
    ),
)
plan = explain("measurements.parquet", schema={"columns": {"temperature": {"ge": -50}}})
```

Paths, `File`, `Dataset`, `Query`, `Loader`, owned `BatchSource`, Polars/pandas
frames and Arrow tables remain supported. Metadata attributes such as `columns`,
`schema` and `to_numpy` alone do not constitute a conversion contract. Custom
providers implement the table Arrow C stream protocol or `__dataframe__`; the
latter is converted through PyArrow's dataframe-interchange implementation.
Conversion validates the actual object and supported dtypes. Static protocols
cannot prove that a provider returns a valid capsule or table. Prefer Arrow C
streams for new providers. Borrowed one-shot readers still require explicit
owned batch-source handling; type hints do not silently take ownership.

## Resource and release limits

Read [resource policy](resources.md) for byte scopes and concurrency tradeoffs.
The spool's persistent working state is bounded; native writer/merge working
allocations are not an exact process-RSS bound. Logical file bytes are separate
from allocated disk blocks. The IPC file cap counts active leases and pending
cleanup. A resource refusal remains an error/incomplete assessment.

Compression, approximated exact statistics, adaptive budgets, merged CSV/profile
passes and new installation tiers remain conditional research choices. They are
not enabled merely to make a benchmark smaller or faster. The package keeps its
existing exact QA/QC semantics and dependency policy.

External certification still requires actual Windows/macOS runs, configured
PostgreSQL/S3/Azure/HTTP integrations, and a full dependency-floor run with the
required signed DuckDB extensions. The source ZIP is not a PyPI publication or
proof that untested future dependency versions are compatible.

## Technical references

- [PEP 544: structural protocols](https://peps.python.org/pep-0544/).
- [PyArrow dataframe conversion](https://arrow.apache.org/docs/python/generated/pyarrow.interchange.from_dataframe.html).
- [Arrow IPC reading and writing](https://arrow.apache.org/docs/python/ipc.html).
- [SQLite compile-time authorization](https://sqlite.org/c3ref/set_authorizer.html).
- [Python sqlite3 API](https://docs.python.org/3.13/library/sqlite3.html).

References describe API contracts. The included regression tests and measurements
are the evidence for QAVF behavior and performance.
