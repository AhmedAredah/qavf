# Native source admission

The DuckDB backend uses the same input contracts for ordinary files, in-memory
Arrow/pandas/Polars tables, read-only queries, callable loaders and managed batch
factories. No domain-specific columns, tokens or file layouts are built in.

Column names must be nonblank strings and unique ignoring case. DuckDB has a
case-insensitive identifier catalog, so admitting both `x` and `X` would erase
part of the input's identity. QAVF rejects such inputs explicitly, including
CSV headers a native reader would otherwise silently rename. Duplicate and blank
CSV headers are checked for every member of a local file glob. Explicit
`header: false` permits headerless CSV files. Legitimate names such as `column0`
or `x_1` remain valid. Source mappings reject unknown settings instead of
silently dropping them.

Renames apply simultaneously: `{"a": "b", "b": "a"}` exchanges the names.
The resulting names must also be unique ignoring case. QAVF materializes its own
zero-based row position once, using a fresh internal name; a user column named
`rowid` or `__QAVF_ROW` has no special meaning.

The native planner retains columns needed by checks, keys, joins, expressions
and requested evidence. Full original schema metadata remains available for
strict-schema and column-shape checks. Unknown SQL dependencies conservatively
retain every column. Input registrations and intermediate query views are
released after snapshot creation, and snapshots can be released after their
final dependent check. Projection does not change row totals or schema coverage.

`row_budget` is an admission limit, never a sampling request. Known eager sizes
are rejected before registration or copying. Native queries and file scans
materialize at most `row_budget + 1` rows, then reject the entire dataset if it
exceeds the limit. A query may still scan more source rows to execute filters,
sorts, joins or aggregates; this setting is not an execution-time or process-RAM
limit. A managed batch that would exceed the limit is rejected before insertion.

Strict CSV diagnostics wrap metadata binding, optional-column probing,
projection binding and snapshot creation. Unterminated quotes and ragged records
report the logical CSV row and physical line numbers, including multiline
records. The diagnostic scan runs only after a native error.

## Owned batches without Polars

A managed `BatchSource` accepts a PyArrow schema or a mapping of names to PyArrow
types. Polars dtype declarations remain available when the optional backend is
installed. The factory must return a context manager, and every yielded batch
must have the declared schema and column order. Eager Arrow tables/record
batches, pandas frames and Polars frames are supported.

```python
from contextlib import contextmanager
import pyarrow as pa
from qavf import BatchSource, Dataset, validate

@contextmanager
def batches():
    # Open/close owned input resources within this context.
    yield iter([
        pa.table({"id": [1, 2], "amount": [10.0, 20.0]}),
        pa.table({"id": [3], "amount": [30.0]}),
    ])

source = BatchSource(batches, {"id": pa.int64(), "amount": pa.float64()})
report = validate(
    Dataset(source, load_policy={"row_budget": 100_000}),
    schema={"columns": {"id": {"dtype": "int"}}, "primary_key": ["id"]},
    backend="duckdb",
)
```

QAVF retains one coherent snapshot, closes the manager on completion or failure,
and never accepts a partial population. Even a context manager that suppresses
an admission exception cannot turn that prefix into a successful load. Borrowed
one-pass `RecordBatchReader` objects are rejected without consuming them; wrap
stream creation and closure in an owned `BatchSource` factory instead.

Retries use the existing explicitly transient source-error classification.
`Loader` and `BatchSource` callbacks require both `replayable=True` and
`retry_safe=True` before a configured retry may run. A failed batch snapshot is
dropped completely before a fresh factory attempt. Missing files, integrity
failures and ordinary validation errors are not retried.

Local files have a pinned manifest and metadata identity. With
`report.hash_inputs: true`, fingerprints include SHA-256 content identities.
QAVF verifies the pinned identity after admission and again before publishing
the report, including files whose snapshots have already been released.

## Physical timezone-aware nanoseconds

DuckDB's timezone-aware timestamp bridge has microsecond precision. Before
registration, QAVF preserves typed Arrow, pandas and Polars nanosecond instants
under an explicit `dtype: datetime` and `timezone` contract by viewing their
unchanged timestamp buffers as UTC-naive nanoseconds. The requested
`datetime_unit` is then checked exactly; an unrepresentable fraction fails
validation instead of being rounded. Managed Arrow batches follow the same
policy. Reports retain the source timestamp type and record the UTC
normalization.

Ungoverned timezone-aware nanosecond columns and nested timezone-aware
nanosecond fields receive an admission error. Native Parquet reads containing
such fields also receive a specific error: use PyArrow to read the data or an
owned Arrow `BatchSource`, then declare the datetime/timezone contract. This
prevents the native Parquet reader from discarding precision before validation.
A caller-authored SQL query is validated as its returned physical values; QAVF
cannot reconstruct precision already discarded by the caller's SQL operations.
