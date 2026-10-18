# Exact typed regression and bounded HTML

`compare_outputs` accepts the same File, Dataset, Query, Loader, source and table
descriptions as validation. Supplying an empty `ResourceLimits()` does not change
which input descriptions are accepted. Two eager tables retain the direct
`compare_frames` path when no resource policy is requested.

## Comparison execution

| Result `execution` | Work and retained state |
| --- | --- |
| `native_stream` | Verified repeatable local files, including native-safe list/struct columns, use the SQL engine's join, ordering and spill controls. Exact existing comparison kernels inspect bounded joined batches. |
| `typed_stream` | Unsupported native representations, mixed key widths and nonrepeatable inputs use one coherent projected observation per side and a disk-resident exact key index. Original schemas, Decimal values, Int128 values and nanosecond temporal values survive. |
| `in_memory` | Explicit eager-frame comparison retains both caller-provided tables and the key alignment. |
| `projected_fallback` | Opaque Python Object cells have no portable typed storage contract and retain the explicit projected path. The reason remains visible. |

The typed path stores full canonical keys, not hashes, and checks duplicate/NaN
identity keys before publishing a result. Changed-row/value counts, tolerances,
schema changes and bounded evidence are decided by the shared
`ComparisonAccumulator`. Source order determines evidence order within bounded
batches. Batch boundaries can differ across resource policies; exact counts and
acceptance do not depend on evidence selection.

Typed observations and index pages share the run's owned spill quota. Temporary
SQLite databases use a small page cache, no memory mapping and no durability
journal. Their queries must use existing indexes: a temporary sorting/grouping
B-tree is rejected rather than creating unaccounted work files. Work databases
and observations are released on success, failure and cancellation.

Exact typed alignment trades additional disk I/O and per-row encoding work for
bounded decoded retention. It is not promised faster than an eager comparison
when both tables already fit comfortably in memory. Observations and indexed
payloads coexist during alignment, so allow space for both. A zero spill budget
cannot run this path. When a memory target is supplied, one encoded row is
limited to one quarter of that target; decoded source batches are checked too.
Without a memory target, one unusually large cell can still dominate memory.
These are planning and owned-retention controls, not an RSS guarantee; isolated
execution provides the separately documented OS/process controls.

Baseline acceptance shares input normalization and typed observation/verification
with comparison. Its publication lock, recovery journal, exact read-back check
and manifest digest remain independent of the streaming writer. See the baseline
API documentation for supported formats and input resource policy.

## HTML presentation

The canonical JSON report budget and HTML output budget are separate:

```yaml
report:
  html_max_cells: 10000
  html_max_columns: 64
  html_max_bytes: 16777216
```

HTML switches sparse, heterogeneous records to `Record / Field / Value` rows
instead of expanding every record against the union of all fields. Dense tables
are capped by data-cell count; omitted presentation evidence is disclosed and
the JSON report remains the complete recorded evidence. Individual cell trees
are previewed without building a complete nested display string.

`qavf.reports.presentation.HtmlLimits` provides the same controls to
`render_html` and `write_html`. Both consume one shared presentation generator.
Disk output streams table rows, provenance and metrics; it does not buffer a
whole evidence-table macro or dump the complete metadata graph. Output exceeding
the byte cap raises `ResourceLimitError`; atomic publication preserves any
previous complete HTML file.

Credential redaction normalizes the same supported container algebra as audit
snapshots before visiting strings. Sets, frozensets and NumPy arrays therefore
cannot become newly visible unredacted values during later serialization.
