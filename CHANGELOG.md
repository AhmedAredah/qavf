# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Progress events and cancellation: `validate(..., on_event=..., cancel=...)`
  reports each step as it runs, and `CancellationToken.cancel()` stops a run from
  any thread while still returning its report (`metadata.cancelled`).
- `qavf run --progress` writes progress as JSON lines on stderr; Ctrl-C now
  cancels the run and still writes its reports.
- `settings_from_text(text, path)` validates unsaved configuration text as the file
  it would become, and `python -m qavf` runs the command line.

## [0.6.2] - 2026-11-02

### Changed

- **The native backend reports exactly what the full backend reports** for the
  input rules: the same results, names, messages, check identities, counts and
  evidence. Each column check is one result that counts every row breaking any of
  its limits, with a `violates` column in its evidence; passing checks are listed
  in one summary per rule; multi-file inputs locate evidence rows with `_file` and
  `_file_row`. If you parse native-backend reports, expect these result shapes
  instead of one result per limit.
- Report messages agree in number ("1 dataset conforms to its schema",
  "1 row expression holds", "1 key ... occurs more than once").
- A check that was not assessed (`ERROR`, `SKIPPED`) no longer reports its
  affected count as exact.

### Fixed

- A column-level limit with one open side reads `[0, ]`, not `[0, undefined]`.
- A primary-key column absent from the data no longer makes the declared-type
  check of a CSV file unassessable.
- A missing unit column is reported as such on every execution path.

## [0.6.1] - 2026-11-01

### Changed

- **Polars is optional.** `pip install qavf` validates declared contracts (schemas,
  column limits, keys, foreign keys, row expressions, SQL assertions) on a native
  DuckDB/Arrow backend. Statistical, temporal, spatial, reconciliation, regression
  and comparison rules and custom plugins need `pip install 'qavf[polars]'` (also
  included in `stats`, `spatial` and `all`); without it they report a capability
  error instead of running. Choose a backend with `backend=` or `--backend`.
- Supports DuckDB 2.0 as well as 1.3.1 and later. QAVF 0.6.0 failed to read any
  dataset when installed with DuckDB 2.0.
- Automatic resource planning uses input size, key widths and available memory and
  disk on Linux, macOS and Windows; explicit limits still win, and no automatic
  memory cap is set where memory cannot be read.

### Added

- `na_values` and `na_groups` (INP-009) for not-applicable tokens, distinct from
  missing or invalid values.
- Key policies: `unique_keys` with `nulls: exempt|equal|forbid`, conditional keys
  (`where`), `grouped_values` occurrence bounds and `duplicate_rows_per_key`.
- Value policies: `number_grammar` (separators, exponents, integral decimals),
  `trim_whitespace`, `convert_options`, `each_allowed_once`, Unicode
  normalization and `casefold`, `decimal_precision`/`decimal_scale`, and
  `datetime_unit`/`timezone` with explicit daylight-saving rules.
- Structured check identity on every result: `check_id`, `scope`, `columns`,
  `parameters` and `checks`.
- `backend_capabilities()` and `qavf capabilities` describe what each backend runs.

### Fixed

- Foreign keys match only when types align losslessly; declared N/A values are
  exempt on both sides.
- NaN counts as missing in keys; inferred float columns reject infinity by default.
- Blank, duplicate or colliding CSV headers are rejected; malformed CSV records
  report their logical row and physical line instead of being skipped.
- Sum, mean, median and missing-fraction limits are decided exactly.
- Keys compare declared values (`07`, `7` and `+7` are one integer key) on every
  input type.
- A failing contract no longer stops the checks after it; a fused query that fails
  is re-run check by check.
- Conditional checks include their condition's columns in evidence; renames are
  applied simultaneously, so swaps work.
- Borrowed one-pass Arrow readers are rejected without being consumed.

### Performance

- Evidence size is bounded inside DuckDB before values reach Python; only the
  needed columns are read, and snapshots are released after their last use.
- Oversized in-memory inputs are refused before they are copied; imports are lazy.

## [0.6.0] - 2026-10-31

### Added

- **Declared types are read, not guessed.** CSV columns declared `int`, `float`,
  `decimal`, `bool`, `date` or `datetime` are converted strictly in the same scan.
  INP-002 lists each value that is not of the type with its row and text; INP-003
  still checks every limit on the remaining values.
- `format` for dates and datetimes not written as ISO 8601, `true_values` and
  `false_values` for booleans, and text `missing_values` (such as `N/A`) on typed
  columns.
- `text_columns` reads CSV columns as written, so codes such as `007` keep their
  leading zeros; columns declared `dtype: str` are included automatically.
- `normalize: {case, trim}` for `isin`, `notin` and `unique` on text.
- `not_null` in a check requires a field only in the check's rows.
- **Every row-level finding is located.** Evidence carries `_row`, and `_file` and
  `_file_row` for multi-file datasets. Unreadable CSV lines are reported with their
  file, line, column and text; `max_rejected_rows` and `rejects_budget` bound
  lossy reads.
- An absent column names its likely spelling in the data (`ID` for `id`).
- `schema.evidence_columns`: failing rows show the checked columns and the key
  (`checked`, the default) or whole rows (`all`).
- `BatchSource` for caller-owned batch factories, `explain(...)` to list the checks a
  configuration runs without reading data, and the `validity`, `regression` and
  `change` profiles, which always need an explicit QA contract.
- `affected_count_exact` on results: `False` marks a lower bound, shown as "at least".
- `max_spill_files` resource limit; `.xlsm` workbooks.

### Changed

- Licensed under the Apache License 2.0 (previously MIT).
- DuckDB 1.3.1 or newer is required: 1.2 loses time zones when reading CSV files, and
  1.3.0 exports decimals in an Arrow format that the supported PyArrow and Polars
  versions cannot read.
- Zoned timestamps load and compare in UTC on every machine.
- A value that is not of its declared type counts as missing for every check other
  than INP-002.
- Comparison error, bias, agreement, concordance and R² limits are hard limits under
  every decision policy; the identity-line test decides only slope and intercept.
- `qavf.engine.registry.RuleParams` moved to `qavf.parameters`; the old import
  still works.
- Lossy CSV reads (`ignore_errors` or `store_rejects`) need a list of files no larger
  than `rejects_budget` (default 1,000): an unresolved remote wildcard or a larger
  list is refused. Raise `rejects_budget` or list the files explicitly.

### Fixed

- A CSV value past the reader's type sample could be silently changed (`2.5` read
  as 3) while every check passed.
- Decimal text is converted exactly and never rounded into its limits.
- False passes: a comparison that could not compare every requested column, a
  custom batch source that suppressed a read error, and a saturated rejection count
  are now incomplete instead of passing.
- A report path naming an input of the run is refused before anything is written.
- A rule calling `sys.exit` is reported as an ERROR instead of ending the program.
- Exact counts above `2**53` are no longer rounded before statistical tests.
- NaN identifiers count as missing in spatial layers, crosswalks and weight tables.
- Streamed JSON reports match Pydantic's output, including plugin result models.
- Chart groups with similar labels stay distinct; maps find float zone ids.
- Plugin registries forget removed or failed entry points on `refresh()`.
- Shared configuration objects are no longer expanded; cyclic ones are refused.
- Temporary files that cannot be removed at once are retried and stay accounted.

### Performance

- Failing-row evidence reads only the columns it shows; unordered SQL evidence is
  sampled in one streamed pass.
- Spilled data is read without a Python copy and outside shared locks.
- Series-shape screens, cross-CRS grid assignment, chart text and over-budget
  in-memory frames are bounded before work starts.
- Faster exact integer tolerances in regression comparisons, field-by-field nested
  comparisons, sparse subgraph selection and coalesced small source batches.

## [0.5.0] - 2026-10-23

### Added

- **Dataset comparison.** `compare_datasets`, the `CMP-001` rule and
  `qavf compare-datasets` compare a candidate dataset with a baseline, including
  datasets whose columns differ. An alignment pipeline handles this: column mapping,
  lossless casts, unit scaling, recoding, derived columns, filters, time shifts and
  zone assignment. Comparison levels: structure, key coverage, keyed records,
  grouped measures, temporal buckets and series shape, spatial zones and locations,
  distributions, relationships and contributions (with an exact mix-versus-rate
  decomposition). Checks that cannot be assessed are reported as such, never passed.
- **Statistical inference.** `qavf.inference` provides paired, independent-sample,
  equivalence, categorical, count, trend and spatial tests, multiple-testing
  corrections, intervals, bootstraps, effect sizes and agreement metrics. By default
  a comparison fails only when a difference is both material and significant after
  Benjamini-Hochberg correction. TMP-002, TMP-003 and STA-002 use the same tests
  and report `p_value`, `adjusted_p` and `effect`.
- **Spatial building blocks** (`qavf.spatial`). Zone layers come from datasets,
  GeoParquet, GeoJSON or shapefiles. Point-to-zone assignment has explicit boundary
  and overlap policies. Also grids, crosswalks, hierarchies, neighbour weights,
  Moran's I and LISA. GEO-001 accepts a `within` zone layer and SPA-003 a
  `zone_layer`. Geometry needs DuckDB's `spatial` extension.
- **Temporal building blocks** (`qavf.temporal`, `qavf.statistics.series`).
  Calendar buckets follow local time zones and keep DST days intact. TMP-001 to
  TMP-008 accept `bucket`, `time_zone` and `time_format` for Date, Datetime and
  text time columns.
- Reports show a comparison section: a summary, per-check evidence and inline SVG
  charts within `HtmlLimits`. `DatasetComparison` can write its own HTML and JSON
  reports.
- Regression comparisons map renamed columns (`column_map=`, `rename`,
  `qavf compare --map OLD=NEW`).
- A new `spatial` extra (pyproj and SciPy) adds indexed neighbour searches and
  authoritative CRS resolution. `all` now includes it.
- Alignment steps, measures, comparators, levels and statistical tests are
  pluggable through registration decorators and entry points. The shared component
  registry lives in `qavf.plugins.registry`.

### Changed

- Fewer false passes: invalid or empty assessments, unsupported numeric values,
  lossy casts, undefined statistics and duplicate keys are now failures or
  explicitly unassessed. Some verdicts that were wrongly passing will change.
- New default resource limits, with explicit refusal instead of partial results:
  - polygon assignment: 1,000,000 point keys, 2,000,000 mappings, 20,000,000
    candidate pairs;
  - neighbour graphs: 1,000,000 edges;
  - profiling category state: 1 MiB.

  Raise these deliberately for larger workloads.
- Unknown CRS classifications need `qavf[spatial]`. The minimum supported pyproj
  is 3.6.1.
- Invalid fractional byte sizes are rejected rather than rounded. Colliding HTML and
  JSON report paths are refused before writing (CLI exit code 2). Typed Parquet
  reads refuse explicit cloud credential-provider chains.
- A core install without the `stats` extra should set `decision: material` for
  comparisons.

### Performance

- Neighbour weights use compact sparse arrays.
- Point-to-zone matching runs in batches, resolving overlaps natively.
- Map outlines are checked against their vertex budget before they are built.
- Compatible lazy column checks are evaluated in one scan.
- Rule provenance no longer scans every installed package.
- Fewer redundant source evaluations and serializations, plus bounded work for
  resampling, correlations, graphs and reports.

### Fixed

- Exact numeric handling: Int128 scaling, Decimal rounding, nanosecond and missing
  group identities, subnormal ratios and floating-point threshold boundaries.
- Statistics: BCa acceleration, contingency counts, resampling stability, DST bucket
  grids and dependent time ordering.
- Execution: deadline monitoring through cleanup, cancellation at native operation
  entry, plugin registration atomicity and retained-memory accounting.
- Compatibility with the minimum supported Polars and DuckDB versions.

## [0.4.1] - 2026-10-18

- Add a `secrets` configuration section for cloud storage: AWS S3 and S3-compatible
  services (Cloudflare R2, MinIO, Wasabi, GCS interoperability) and Azure Blob/ADLS,
  with static keys or the platform credential chain. Secrets apply to DuckDB reads and
  to remote Parquet read through Polars, and are masked in reports and logs.
- Mask `key_id`, `connection_string`, account keys and SAS tokens wherever settings
  or messages are redacted.
- Include the `aws` and `azure` DuckDB extensions in the container image.
- Never stop an isolated worker before its wall-time deadline: on Windows a process
  wait could wake a few nanoseconds early and end the worker before its allowance.

## [0.4.0] - 2026-10-18

First public release.

- Validation API (`validate`, `QARunner`) and CLI for files, globs, URLs, databases,
  SQL queries, Polars/pandas/Arrow frames and Python loaders.
- 28 built-in rules: input contracts, column limits, keys and foreign keys, row
  expressions and SQL assertions, freshness, temporal and spatial checks, outlier and
  drift detection, reconciliation and regression against accepted baselines.
- Fail-closed assessment: checks that cannot be assessed make a run incomplete rather
  than passing, with exact counts and bounded, deterministic evidence.
- HTML and JSON reports with full run provenance and redacted credentials.
- Resource budgets for memory, spill storage, evidence and report size, and optional
  isolated execution with OS-enforced limits on Linux, macOS and Windows.
- Plugin discovery for rules and sources from packages, files and entry points.
