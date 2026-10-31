# Input ownership, bounded diagnostics and incremental reads

`validate` accepts paths, `File`, supported Polars/pandas/Arrow tables, `Query`,
`Loader`, `BatchSource`, and explicit `DataSource` adapters. Wrap any input in
`Dataset` to add a schema, rename map or loading policy. A plain mapping means
named datasets, so use a dataframe for a mapping of columns.

Descriptors are validated before data is converted or a callback is invoked.
`explain` uses the same contract checks without reading input data. Declared
batch schemas are metadata; inspecting one does not open the batch factory.
Caller-owned borrowed database connections remain open. A `Loader` result with
a known row count is rejected before interchange conversion if it exceeds its
row budget; unknown-size streams cannot provide that early guarantee.

This release requires **DuckDB 1.3.0 or newer**. DuckDB 1.2.0's CSV inference
turns timezone-marked timestamps into naive UTC values, losing their timezone
metadata even with ICU loaded. Version 1.3.0 preserves aware timestamps while
keeping genuinely unzoned inputs naive. Upgrade the dependency when replacing
an existing installation; the minimum-version timestamp regression verifies
that QAVF reports zoned CSV timestamps in UTC.

## Owned batch sources

Use `BatchSource` when a reader produces eager tables incrementally. Its factory
must return a context manager that owns the reader and yields an iterable of
Polars dataframes, Arrow record batches/tables, or supported eager frames.

```python
from contextlib import contextmanager
import polars as pl
import pyarrow.csv as csv
from qavf import BatchSource, Dataset, ResourceLimits, validate


@contextmanager
def readings():
    reader = csv.open_csv(
        "readings.csv",
        convert_options=csv.ConvertOptions(column_types={"value": "int64"}),
    )
    try:
        yield reader
    finally:
        reader.close()


source = BatchSource(readings, schema={"value": pl.Int64})
report = validate(
    Dataset(source, load_policy={"row_budget": 10_000_000}),
    schema={"columns": {"value": {"dtype": "int", "ge": 0}}},
    resources=ResourceLimits(target_memory="256MiB", max_spill="2GiB"),
)
```

The schema names, order and dtypes must match every batch, including batches
with zero rows. An empty stream retains its declared schema. Lazy plans and
borrowed one-pass readers are not accepted as individual batches. QAVF does not
coerce schema drift or publish a partial stream as complete.

One run opens the factory once and stages the whole population into its existing
owned observation store. Subsequent rules share that population. Retained
buffers are detached where required, and the store spills to owned IPC files
when its retention budget is exhausted. The row budget applies to cumulative
rows across all batches. Temp files are removed when their final owner closes.
The factory's context manager exits on success, read failure, schema failure,
resource failure and cooperative cancellation. Suppressing an exception in its
cleanup cannot turn a partial observation into a pass.

This guarantee is enforced by the shared batch engine for custom `DataSource`
batch implementations as well. A plugin context manager that suppresses a read,
schema, row-budget, admission, or interruption error cannot publish the prefix
read before that error as a complete dataset. Source adapters still own their
reader cleanup and schema conversion; they cannot override completion policy.

Retries are disabled by default. A retry requires all of:

- `BatchSource(..., replayable=True, retry_safe=True)`;
- a `Dataset` loading policy with positive `retries`;
- an error that QAVF classifies as retryable.

Every retry opens a fresh context manager and discards the previous attempt.
The caller promises that replay is safe and represents the intended population.
Callbacks are in-process inputs and cannot be transported into an isolated
worker; use a configured source plugin or a file for that execution mode.

`target_memory` governs admitted results and retention, not every allocation
inside a callback or native reader. Choose a suitable reader batch size. The
producer can allocate memory before yielding, and a blocking producer cannot be
interrupted cooperatively until it returns. The spill budget limits owned disk
usage; exceeding it reports incomplete assessment instead of silently dropping
rows.

## CSV diagnostic capture

Lossy CSV reads require `acknowledge_lossy=True`. `File.rejects_budget` (or the
file source parameter with the same name) defaults to **1,000 raw native error
entries** across the resolved files. This is independent of evidence sampling:
limiting displayed records alone does not limit the native rejection table.

DuckDB's `rejects_limit` counts error entries per file. A malformed line can
produce several entries. QAVF allocates the total budget across the manifest,
then clamps any requested per-file limit to that allocation. Reader
`rejects_limit=0` uses the bounded budget; rejection storage is never unbounded. The effective limit and budget appear in reader
provenance. Unresolved remote wildcards cannot establish a total file count and
are rejected for lossy capture; use explicit inputs or resolved manifests.

The affected line count is distinct from the raw entry count. If any file fills
its entry cap, the line count is conservatively a lower bound, even when it
equals the allowed threshold. `affected_count_exact=False` survives report
serialization and zero-evidence reports. A lower bound above
`max_rejected_rows` proves failure; a lower bound at or below it leaves the
assessment incomplete. Increase `rejects_budget` or fix the input to establish
the remaining count. Exact counts at or below the threshold still pass.

The diagnostic budget bounds entries, not their byte lengths or process RSS.
Reader line-size limits and supported process execution limits serve different
purposes. Strict read failures also capture only bounded diagnostics. CSV
conversion aliases reserve the complete column namespace, and typed evidence
keeps user columns intact even when named `value`, `_row`, `violates`, `_file`
or `_file_row`; colliding metadata receives a separate generated name.

## Auxiliary crosswalk inputs

The direct comparison API accepts the same descriptors through `Crosswalk.data`.
It projects and renames through the existing context's source policy, resource
owner and cleanup scope. `Crosswalk.path` also supports explicit CSV reader
options, `text_columns` and a row budget, so identifiers such as `001` can remain
text.

Run configurations must be serializable for snapshots, digests and isolated
execution. A live `Crosswalk.data` inside a run configuration is rejected with
guidance: supply the auxiliary input as a named dataset and reference it using
`crosswalk.dataset`. Direct in-process comparison calls retain descriptor
support. No hidden dataset is introduced or silently added to the QA task list.

Direct auxiliary descriptions are scoped reads. When multiple accesses must
reuse one coherent auxiliary population, provide one named dataset and use
`crosswalk.dataset` so the run owns and reuses that dataset's observation.
