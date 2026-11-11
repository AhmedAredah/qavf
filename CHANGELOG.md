# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.7.0] - Unreleased

Checks are configured by name, next to the data they test, and one check can relate
any number of datasets. Rule codes remain only as stable audit IDs in reports.

### Breaking changes

Configurations written for QAVF 0.6 are rejected with an error naming their first
old setting. Convert each configuration once with `qavf migrate old.yaml -o new.yaml`
(`--check` shows what would change; `--drop-disabled` leaves disabled rules out).
Reports written by earlier versions still load.

- **Checks are named items.** A dataset's `checks` list (checks of that dataset)
  and a top-level `checks` list (checks relating datasets) replace
  `rules: {TMP-001: ...}`: `{check: change_rate, series: aadt_by_site, max: 0.15}`.
  Every item takes `name`, `severity`, `where`, `tags`, `enabled` and `description`.
  Some settings are renamed (`max_change_rate` is `max`, `expected_trend` is
  `expect`, `expected_periods` is `expected`); [docs/rules.md](docs/rules.md) lists
  each check's settings.
- Row expressions are `check: expression` items, SQL assertions top-level
  `check: sql` items, baselines `matches_baseline` items and comparisons top-level
  `compare` items; the `expressions` and `assertions` sections are removed, and
  `Dataset(..., expressions=...)` is `Dataset(..., checks=...)`.
- **Only configured checks run.** Schema contracts (columns, keys, foreign keys)
  always run. `engine.run_unconfigured_rules`, `include_categories` and
  `exclude_categories` are removed; use `--tags` and `--skip-tags`.
- `engine.fail_on`, `fail_on_incomplete` and `require` move to a new `run` section;
  `require` lists check or dataset names.
- **Column statistics go under `stats`.** `stats: {mean: {between: [1000, 100000]}}`
  replaces `mean: {ge: 1000, le: 100000}` beside the column's value limits, for
  `min`, `max`, `mean`, `median`, `std`, `sum` and `distinct_count`.
- **A column's conditional limits are `conditions`**, so `checks` always means
  check items.
- `qavf list-checks` replaces `list-rules`.
- Plugins declare a `check_name` beside their `rule_id`; a missing name, or one
  that clashes with another check, is a registration error.
- Each check item is assessed on its own. A 0.6 rule configured with several
  series or comparisons becomes one item each, so multiple-testing corrections
  (trend, drift) now apply within each item rather than across all of
  a rule's series; p-values adjusted across a whole rule can differ from 0.6.
- Result labels read `missing_periods · counts · no missing years (TMP-004)`;
  results keep `rule_id` and `check_id` and add `check_name` and `item_name`.

### Added

- **Value ranges:** `between: [a, b]` (inclusive) on column value limits and on
  each statistic in `stats`, which also accept `gt` and `lt`. Setting `between`
  with `ge`, `le`, `gt` or `lt` on the same field is an error.
- **Reuse.** A dataset's `series` and `metrics` define a time series or per-entity
  metric once (`series: aadt_by_site`, `metric: crash_rate`; `other.name` reaches
  another dataset's), with the normalizer as `per: counts.aadt`. `views` name a
  slice (`where`, `columns`) usable as `counts.recent` wherever a dataset is named.
  Top-level `defaults` give settings per check name, `vars` named values
  (`${vars.years}`), and `include` merges shared files first; a dataset or named
  check defined twice is an error. `2016..2024` or `{from, to, step}` expands to
  every value.
- **Run-time selection:** `qavf run --tags`, `--skip-tags`, `--only "<name>"` and
  `--dataset`, and `validate(..., only=, tags=, skip_tags=, datasets=)`, on both
  backends. Built-in checks carry their family as a tag.
  `run.tags: {<tag>: {fail_on: <severity>|never}}` sets a pass/fail policy per tag.
- **One dataset across many sources:** `across` copies a dataset, with its schema,
  checks, series, metrics and views, once per source: a list of connection names,
  a mapping of copy names to the source settings that differ (connection, path or
  URL, secret, table), or `{city: [houston, dallas]}` filling `{city}` in a path or
  table. Copies are named `counts@houston`; `counts@*` selects every copy in a
  check's `datasets` (one string or in a list), `run.require` and `--dataset`.
- **Many datasets in one check:** `compare`, `totals_match`, `key_sets_match` and
  `drift` take any number of `datasets` and a `strategy` (`reference`, the default;
  `sequential`; `all_pairs`, refused above `max_pairs`; `consensus`). Two datasets
  with the default strategy report exactly as before.
- **Consensus:** each member against a consensus built key by key, configurable as
  `consensus: {numeric: median|mean|trimmed_mean, trim, categorical:
  majority|plurality, min_agreement, ties: report|reference}`. A key without a
  consensus value is reported as not assessed, never guessed. `drift` compares each
  member with the pooled data.
- **Whole-group levels:** with more than two members, `compare` assesses
  `coverage` (a key × member presence matrix), `structure` and `distribution`
  (each member against the pooled data) over all members at once.
- **Group reports:** each result of such a check carries a `group` role
  (`GroupRole`), and `report.groups` holds one `GroupReport` per check: a matrix of
  members by levels, the pairs compared and a ranking of the members. The HTML
  report shows them under "Datasets checked together". Results name the consensus
  and pooled data by role and members by name.
- [docs/checks.md](docs/checks.md) explains check items, column limits, reuse,
  selection and many datasets; [docs/rules.md](docs/rules.md) lists the checks by
  name with their settings.
- The configuration JSON Schema hints every 0.7 section and setting for editors
  (`ui_group`, `ui_title`, `ui_advanced`).
- `qavf init` writes inferred ranges as `between` and shows the 0.7 forms in its
  commented example; the demo and `config/qavf.full.yaml` use them, including one
  dataset across three files compared by consensus.
- Testing helpers: `qavf.testing.rule_results` and `run_rule` run one rule instance
  with any parameters; `checks_config` converts rule-code mappings and `with_checks`
  configures every rule of a test registry.
- `qavf explain -c CONFIG` validates a configuration and prints its plan without
  reading data (`--dataset`, `--tags`, `--skip-tags`, `--only`, `--backend`,
  `--format text|json`, `-o`); it exits 2 naming the invalid field. `explain()`
  takes `backend=`, `only=`, `tags=`, `skip_tags=` and `datasets=`, applying the same
  selection and backend admission as `validate`.
- `qavf init --full` writes the complete reference configuration.
- `include` entries expand `${ENV}` and `${ENV:-default}` before they are resolved
  (`include: ['${MODULE_DIR:-modules}/child.yaml']`); `${vars.name}` still cannot
  be used there, since includes are read before `vars` merge.
- A foreign key's `ref_dataset` may name a view (`roads.active`) or an `across`
  copy (or a copy's view), on both backends. The view keeps its parent's contract,
  so a key the parent holds but the view's `where` excludes is orphaned; the
  native backend loads the parent with the child, with the columns the `where`
  reads.
- `overrides:` patches known datasets and check instances without copying them:
  `checks: {<check name or instance id>: {max: 0.2, severity: WARNING}}` and
  `datasets: {counts: {schema: {min_rows: 100}}}` (an `across` copy is
  `counts@houston`). They apply after include, vars, defaults and `across`
  expansion and before validation; mappings merge, lists and scalars replace and
  `null` is an error. An unknown target names the closest match and a name shared
  by several instances asks for the instance id. Any file may declare them; they
  apply in include order with the root's last. `QAVFSettings.applied_overrides`
  records what each changed and from which file, and `qavf explain` lists them
  (`ValidationPlan.overrides`). The editor schema accepts the section and gives it
  UI hints.

### Changed

- `qavf init` writes a minimal starter (project, datasets with their inferred
  schemas, pointers to `explain`, `list-checks` and the docs) and never pins defaults
  such as `engine.execution_mode`, so a process limit added later selects isolation.
  Comparison examples use `decision: material`.
- Usage and configuration errors print the message and field path without the
  library file and line (shown with `-v`).
- `mypy` checks the bodies of unannotated functions too (`check_untyped_defs`).
- In an included file, a file-path setting written through the file's own
  `${vars.name}` (or `${ENV}`) is substituted before it is rebased, so it names the
  same file as the literal text would (it used to resolve against the root
  configuration's directory). A variable the file does not define, or one the root
  also defines, still resolves relative to the root, which owns it.
- A file reached through more than one include path (a diamond) is merged once
  instead of failing as a duplicate dataset; different files defining one dataset
  or check name, and include cycles, are still errors.
- On the Polars backend, a key sample (orphaned keys, repeated keys) evaluated as
  a lazy plan is ordered by every column, nulls first, as the native backend's
  is, instead of by the plan's engine-dependent row order. A check on a view
  (`roads.active`) applies its parent's column contract (types and N/A tokens) to
  key values.

### Performance

- A row expression or `where` condition no longer reads every column of every
  dataset into one native component. The parser proves the columns it names and the
  datasets its subqueries read; only those are retained or joined. Unprovable SQL
  keeps the conservative plan, and `qavf explain` and the report's `resource_usage`
  say why. The Polars backend releases datasets an expression does not read.
- A SQL check with evidence counts its failing rows and keeps its first rows by
  every column in one evaluation that holds only the evidence rows, on both
  backends, instead of staging every failing row first.
- `scripts/benchmark_local_performance.py` measures both.
- A `Query` on a borrowed DuckDB connection is copied into the native backend's
  owned table one bounded Arrow batch at a time, not as one complete Arrow table.
  The row budget stays exact, cancellation is checked between batches, the
  caller's connection is only read, and types are unchanged. `qavf explain` reports
  it as `staging: bounded_batch_copy`.
- A coherent database snapshot (`snapshot: true`) streams each member in bounded
  batches inside its transaction and stages it adaptively: members stay in memory
  while the group fits the frame cache and `memory_budget_bytes`, so a small
  snapshot costs what reading it whole did; the first batch past that writes the
  buffered members to owned spill files and streams the rest straight to spill.
  The group is published together after commit, and a large group is bounded by
  spill space, no longer by `memory_budget_bytes`; a failed or cancelled attempt
  deletes its files and a retry restarts the whole group. `qavf explain` reports
  `staging: adaptive_snapshot`.
- `scripts/benchmark_streaming.py` measures both, before and after a change.

### Fixed

- Accepting a baseline, writing a report or publishing an isolated worker's
  result no longer fails on Windows when a virus scanner or the search indexer
  briefly holds the file being replaced: every atomic replace in QAVF goes
  through one helper (`qavf/file_replace.py`) that retries the rename on a
  `PermissionError` with a short, bounded backoff (up to about 2 s) on Windows
  only, then raises. The release scripts share the same helper.
- `qavf run` prints result labels on a console that is not UTF-8 (e.g. a Windows
  cp1252 pipe): `·` falls back to `-` instead of showing as `�`.
- Two unnamed checks of one kind on a dataset no longer share an identity: a
  failing `expression` could take the other item's tags, and with them a
  `fail_on: never` tag policy, so the run passed. Every check item now has an
  instance id, its location (`datasets.d.checks[0]`, `checks[2]`), which results
  carry in the new `item_ids` field; tags, tag policies and `run.require` follow
  it. Reports written without `item_ids` still load.
- A `run.require` entry naming several items (one name in several datasets)
  requires each of them to be assessed; a disabled one no longer passes unnoticed.
  A selection still narrows requirements to the items it runs.
- A required unnamed `expression` with `where` that passed is no longer reported
  as not assessed.
- An explicit check name equal to another item's default name in the same list,
  or to a dataset name, is a configuration error.
- A view, dataset, `across` copy or derived dataset with the name of another (or a
  name differing only in case) is a configuration error naming both declarations.
  A view `roads.active` could silently replace a dataset `roads.active` on the
  Polars backend while the DuckDB backend read the dataset.
- Included files rebase only settings typed as file paths (dataset and source
  paths, connection files, baseline and zone-layer or crosswalk paths,
  `plugins.paths`). A column named `path` in `rename`, a schema or any other user
  mapping was rewritten to `modules/path`.
- `qavf run` no longer writes a report over a file the configuration includes, an
  accepted baseline or its manifest, a zone layer, crosswalk or plugin file; it
  had replaced them and reported QA PASSED. The same declared-input check now
  guards `schema -o`, `init -o` (even with `--force`, also for profiled inputs)
  and `migrate -o`. `schema -o` needs `--force` to replace an existing file.
- `qavf migrate -o` writes the format of the output suffix (YAML text went into
  `.json` files), rebases relative paths when the output is in another directory,
  and reads its output back before writing.
- The native DuckDB backend runs dataset views (`"<dataset>.<view>"` in `sql` and
  `expression` checks) with the Polars backend's rows and columns; it had accepted
  them and then failed with a missing table.
- `connections.<name>.snapshot: true` is refused on the native backend before any
  read; it had been ignored, so related tables could come from different commits.
- A cancelled native run no longer retries a source: it starts no further attempt
  and a retry delay ends at cancellation (a loader had run three more times).
- `qavf run` in `execution_mode: isolated`: `--progress` is a usage error instead
  of emitting nothing, and Ctrl-C, SIGTERM or Ctrl-Break stops the worker and exits
  1 with `QA CANCELLED`; it had printed a cancellation notice, let the worker
  finish and reported QA PASSED.
- Comparison tolerances resolve `rtol` and `atol` independently (measure, column,
  level, comparison). A column entry setting only `atol` had dropped the level's
  `rtol` for the comparison's default, so a group mean within the level's `rtol`
  failed (or a looser default passed it).
- The records level describes a `Decimal(38, s)` change that needs 39 digits (e.g.
  from 10^38-1 to -(10^38-1)) exactly; the overflow had made the whole level not
  assessed although its counts were complete. A description that still fails is
  reported as `description_unavailable` beside the counts and verdict.
- Comparison `tolerances` and `ignore_columns` naming a column neither aligned side
  has are a configuration error (with suggestions) in `compare_datasets` and the
  `compare` check; they had been dropped silently, so a typo passed. A measure of
  an ignored column is refused when the configuration loads.
- `derive` columns apply in mapping order, so a later one may use an earlier one;
  every level had been not assessed with a missing column. A reference to a
  column that does not exist is a configuration error before any row is read.
- Crosswalks keep temporal zone ids exact: two `Datetime(ns)` ids 1 ns apart, or the
  two instants of a DST fold, had merged in validation, `compose` and
  `aggregate_up`, so a valid crosswalk was rejected as not summing to 1.
- A zone layer `path` obeys the run's memory target: GeoParquet is read through the
  common input reader (as a path crosswalk is), and an `ST_Read` file's size is
  checked before the fetch and the file verified unchanged after it. The path form
  had been read directly, outside admission and source identity.
- A filtered `expression` is assessed on the rows its `where` selects: its failures
  read `1 row of 2 matching 'where: x > 0' in d violates 'x < 10'` with those rows
  as `total_records` and `population_rows`, `applicable_rows` and `excluded_rows`
  in its metrics, on both backends. It had counted every row of the dataset and
  shown a compiled `CASE WHEN` expression instead of the configured `expr`.
- `expression` takes `min_applicable_rows` (with `where`): fewer matching rows leave
  it not assessed instead of passing vacuously, as for column conditions.
- `{check: expression, enabled: false}` and `{check: sql, enabled: false}` stubs are
  accepted like other disabled checks without settings.
- `where` is documented, advertised in the JSON Schema and accepted only where it is
  honoured (`expression`, `compare`, and plugin checks whose parameters declare it,
  which now receive it); other checks reject it and point at views or a filtered
  dataset.
- A plugin check placed on dataset `a` that names `dataset: b` is a configuration
  error; it had run on `b` while reporting on `a`. Plugin results name the dataset
  they are placed on, and compiling no longer changes the item's settings.
- Range shorthand (`2016..2024`, `{from, to, step}`) expands only in check settings
  declared as lists, nested ones included (`key_grid` `dimensions.<c>.values`); a
  text setting such as an `outliers` `column: 2016..2024` and plugin settings keep
  the value as written. The JSON Schema offers ranges only on those settings.
- `across` rejects a repeated connection, copy name or template value (also as text:
  `2023` and `'2023'`) instead of collapsing the copies into one.
- Included files sharing YAML anchors merge each pair of shared parts once, and every
  file is measured before it is merged; a few kilobytes had taken seconds and
  hundreds of megabytes before the size limit applied.
- Both editor JSON Schemas (full and DuckDB) accept `${vars.name}` in every setting
  (column limits, `min_rows`, a check's `enabled` and `severity`, ...), a single
  `include` string, comparison levels and steps written by name or with their own
  parameters, a bucket written as text, and a single `datasets` selector list; the
  documented example configurations validate against them.
- `$${NAME}` and `$${vars.name}` write a literal `${NAME}` / `${vars.name}` that is
  never substituted, also when settings are serialized and read back.
- Booleans are no longer read as counts (`min_rows: true` had meant 1): row counts,
  budgets and limits reject `true`/`false`. A YAML boolean in a str column's
  `isin`/`notin` or in `true_values`/`false_values` is rejected with a hint to quote
  `yes`/`no`/`on`/`off`.
- File paths in an included file's `defaults` (a `matches_baseline` `path`, ...) are
  rebased like the same settings on a check item.
- A records level's `categorical` naming a column neither aligned side has is a
  configuration error with suggestions, like `tolerances`; it had left the level not
  assessed during evaluation.
- A `compare-datasets` spec's `resources` mapping (`{target_memory: 64MiB}`) is read
  as `ResourceLimits`, and an invalid one is a usage error naming the field (it had
  crashed with a traceback and exit 1). Relative paths in the spec resolve against
  its folder.
- On the Polars backend, a dataset selection no longer converts or consumes an
  excluded input (a one-shot Arrow stream stays unread; a failing one no longer
  fails the run).
- A frame too large for the cache is spilled directly, instead of first spilling
  every in-memory entry (which could falsely exhaust the spill budget).
- A native SQL check that keeps no evidence (`max_affected_records: 0`) counts its
  rows in one query, without a temporary table of every failing row.
- Rule and source registration rejects a class that is not a concrete `Rule` /
  `DataSource`, lacks its id or has a non-pydantic `Params`, before registering it;
  rule and source lookups no longer copy the registry.
- `from qavf import CheckIdentity` has its type for type checkers.
- `docs/api.md` names `Dataset(..., checks=)` and `docs/plugins.md` the testing
  helpers' `checks=` keyword (not the removed `expressions=` / `assertions=`).
- An invalid dataset setting in `validate`/`explain` names its full path
  (`datasets.<name>....`).

## [Unreleased]

### Added

- Progress events and cancellation: `validate(..., on_event=..., cancel=...)`
  reports each step as it runs, and `CancellationToken.cancel()` stops a run from
  any thread while still returning its report (`metadata.cancelled`).
- `qavf run --progress` writes progress as JSON lines on stderr; Ctrl-C now
  cancels the run and still writes its reports.
- `settings_from_text(text, path)` validates unsaved configuration text as the file
  it would become, and `python -m qavf` runs the command line.
- The configuration JSON Schema (`qavf schema`) labels its fields for editors:
  `ui_group`, `ui_title` and `ui_advanced` on each setting.

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
