# Configuring checks

A configuration describes your data and names the checks to run on it. Users never
type a rule code: each check has a name (`missing_periods`), and its code (`TMP-004`)
is only the stable audit ID in reports. [The check reference](rules.md) lists every
check and its settings; `qavf list-checks` prints the same catalog.

## Where checks go

- **`schema`** describes what a dataset is: columns (type, parsing, limits), keys,
  foreign keys, N/A groups and grouped values. These contracts always run and need
  no check item.
- **`datasets.<name>.checks`** tests that dataset (`change_rate`, `expression`, ...).
- **Top-level `checks`** relates datasets (`sql`, `compare`, `totals_match`,
  `key_sets_match`, `coverage`, `drift`, `batch_consistent`).

Only configured checks run. Every item takes these fields besides its own settings:

| Field | Meaning |
|---|---|
| `check` | The kind of check (required) |
| `name` | Your label, unique within its list and different from every other item's default name there and from every dataset name; shown in reports, usable in `run.require` and `--only`. Defaults to `<check> on <dataset>` |
| `severity` | `INFO`, `WARNING`, `ERROR` or `CRITICAL`, overriding the check's default |
| `tags` | Words grouping checks for run-time selection |
| `enabled` | `false` keeps a check configured but off; an item switched off before any settings are given may omit its required ones |
| `description` | Shown in reports |

`where` (a SQL condition: assess only the rows that match it) is taken only by
`expression`, `compare` and plugin checks whose parameters declare it; any other
check rejects it. To check a slice with those, check a view (`counts.recent`, where
a check names datasets) or define a dataset of the filtered rows.

A filtered `expression` is assessed on the rows its `where` selects (a `NULL`
condition excludes the row): `1 row of 2 matching 'where: x > 0' in d violates
'x < 10'`, with `population_rows`, `applicable_rows` and `excluded_rows` in its
metrics. With `min_applicable_rows: N`, fewer than `N` matching rows leave it not
assessed (an incomplete run) instead of passing vacuously:

```yaml
datasets:
  speeds:
    path: data/speeds.csv
    checks:
      - {check: expression, name: valid speeds, expr: speed < 120,
         where: "status = 'valid'", min_applicable_rows: 1}
```

A result keeps its `rule_id` and `check_id` and adds the check name and the item's
name; its label reads `missing_periods · counts · no missing years (TMP-004)`.

Names are labels, not identities: unnamed items of one kind on a dataset share
the default name. Each item is identified by its location, its *instance id*:
`datasets.counts.checks[2]`, `checks[0]` for the top-level list, and
`datasets.counts@a.checks[0]` for an item on an `across` copy. Results carry the
ids of the items they report on in `item_ids` (a summary of passing checks lists
all of them), and tags, tag policies and `run.require` follow the id, so two
items sharing a name never take each other's tags or requirements.

## Column limits

```yaml
datasets:
  counts:
    path: data/counts.csv
    schema:
      columns:
        year: {dtype: int, between: [2000, 2030]}
        aadt:
          dtype: int
          ge: 0                                   # graded at the default severity
          stats:
            mean: {between: [1000, 100000]}
            distinct_count: {ge: 10}
          conditions:
            - {le: 200000, severity: WARNING, name: implausibly high}
            - {where: "count_type = 'permanent'", not_null: true, name: permanent counts}
```

- **Value limits** test every value: `ge`, `le`, `gt`, `lt`, `between: [a, b]`
  (inclusive, the same as `ge: a, le: b`), `isin`, `notin`, `regex`, `min_length`,
  `max_length`, `integral`, `max_decimals`; plus `unique`, `max_null_fraction` and,
  in a condition, `not_null`.
- **`stats`** bounds statistics of the whole column: `min`, `max`, `mean`, `median`,
  `std`, `sum` and `distinct_count`, each with `ge`, `le`, `gt`, `lt` or `between`.
- **`conditions`** are further groups of limits, each with its own `name`,
  `severity` and `where` (and `min_applicable_rows` with `where`).

Setting `between` together with `ge`, `le`, `gt` or `lt` on the same field is an
error, as are bounds no value can satisfy.

## Reuse

Each building block is defined once and referenced by name.

```yaml
vars:
  years: 2016..2024                 # every year in the range
defaults:
  change_rate: {max: 0.15}          # unless an item sets its own
datasets:
  counts:
    path: data/counts.csv
    series:
      aadt_by_site: {value: aadt, group_by: [site_id], time_column: year}
    views:
      recent: {where: year >= 2020}  # used as counts.recent wherever a dataset is named
    checks:
      - {check: missing_periods, name: no missing years, series: aadt_by_site, expected: '${vars.years}'}
      - {check: change_rate, series: aadt_by_site}
  crashes:
    path: data/crashes.parquet
    metrics:
      crash_rate: {value: crashes, entity: site_id, group_by: [year], per: counts.aadt}
    checks:
      - {check: entity_outliers, metric: crash_rate}
```

| Block | Lives in | Holds |
|---|---|---|
| `series` | a dataset | A time series: `value`, `time_column`, `group_by`, `aggregate`, `weight`, `order_by`, `bucket`, time zone and format |
| `metrics` | a dataset | A per-entity rate or aggregate: `value`, `entity`, `group_by`, `aggregate`, `per` (a normalizer such as `counts.aadt`) |
| `views` | a dataset | A named slice (`where`, optional `columns`) with the parent's contract |
| `defaults` | top level | Default settings per check name |
| `include` | top level | Other YAML files merged first |
| `overrides` | top level | Explicit patches of known datasets and check instances |
| `vars` | top level | Named values, used as `${vars.name}` in any value |

Datasets, their views (`<dataset>.<view>`), `across` copies (`<dataset>@<copy>`) and
the datasets checks derive share one namespace, because SQL reads each of them as a
table by name, ignoring case. Two of them with one name, or names differing only in
case, are a configuration error naming both declarations. Dataset names may contain
dots (a schema-qualified `roads.archive` is fine), but a dataset `roads.active` and
a view `active` of dataset `roads` cannot both exist.

Resolution: `include` files merge first, in order, and the including file wins; a
dataset or check name repeated across files is an error. `vars` and `${ENV}`
substitute before validation, and an unknown variable is an error. A check's own
settings win over `defaults`. A `series` or `metric` name resolves within its
dataset first; `other.name` reaches another dataset's block. In a check setting
declared as a list (`missing_periods` `expected`, `key_grid`
`dimensions.<column>.values`, ...), `2016..2024` or `{from: 2016, to: 2024, step: 1}`
expands to every value; a plain list means just those values. Other settings keep
such a value as written (an `outliers` `column: 2016..2024` is a column name), and
so do plugin checks' settings.

To write a literal `${...}` (in SQL, a description or an allowed value), double the
dollar: `$${NAME}` and `$${vars.name}` are the text `${NAME}` and `${vars.name}`,
never substituted.

A relative file path written in an included file is relative to that file. Only
settings that are file paths are rebased: a dataset's `path`, its source's
`path`/`paths` (and those of `across` copies), a connection's database `path`,
check settings naming files (a `matches_baseline` `path` or `source`, a zone
layer's or crosswalk's `path`, a comparison level's layer) and `plugins.paths`.
Column names, `rename`, `schema`, `views`, `series`, `metrics`, `where`, `expr`,
SQL text and `extensions` are never rewritten, so a column named `path` means the
same in an included file as inline. Absolute paths and URLs are kept; settings of
plugin checks QAVF does not know are kept as written.

A path means the same whether it is written literally or through a variable. In
an included file, the file's own `${vars.name}` references and `${ENV}` references
in a file-path setting are substituted before the path is rebased, so
`path: roads.csv` and `path: ${vars.input_file}` with `vars: {input_file:
roads.csv}` both name `modules/roads.csv` below. A variable the file does not
define itself (or one that an including file, such as the root, also defines)
belongs to the file that sets it: a path that starts with it is kept as written
and resolved relative to the root configuration's directory. Text written around
such a reference is still the included file's (`data/${vars.f}` in
`modules/child.yaml` is `modules/data/<value>`).

```text
# qa.yaml                                # modules/child.yaml
include: [modules/child.yaml]            vars: {input_file: roads.csv}
vars: {region_file: regions.csv}         datasets:
                                           roads:   {path: '${vars.input_file}'}   # modules/roads.csv
                                           regions: {path: '${vars.region_file}'}  # regions.csv (the root's)
```

If `qa.yaml` also set `input_file: prod.csv`, the root would own that variable and
`roads` would read `prod.csv` next to `qa.yaml`. `include` entries themselves take
`${ENV}` and `${ENV:-default}` (`include: ['${MODULE_DIR:-modules}/child.yaml']`),
relative to the including file; they cannot use `${vars.name}`, because files are
read before any file's `vars` merge.

A file reached through more than one include path (a diamond: two modules that
both include `shared/base.yaml`) is merged once, so its datasets are not reported
as defined twice. Two different files that define one dataset or check name are
still an error, and an include cycle is still an error.

### Overrides

Includes never silently replace a dataset or a check. To change one setting of a
dataset or check another file defines (a production threshold, a team's severity)
without copying it, name it under `overrides`:

```yaml
overrides:
  checks:
    no missing years: {severity: WARNING}       # a check item's name
    datasets.counts.checks[1]: {max: 0.2}       # or its instance id
  datasets:
    counts: {schema: {min_rows: 1}}             # a dataset, or a copy: counts@houston
```

- Overrides apply after `include`, `vars`, `defaults` and `across` expansion, and
  before validation, so they patch the configuration as it will run and the
  result is validated like any other value.
- Mappings merge recursively; lists and scalars replace (an overridden `checks`
  list or `expected` list is the new list). `null` is an error: an override never
  deletes a setting.
- A target must exist. An unknown name is an error naming the closest match; a
  check name that several instances share (each `across` copy repeats its
  dataset's checks) is an error listing their instance ids, so override one by id
  (`datasets.counts@houston.checks[0]`), the item location reports show as
  `item_ids`.
- Any file may declare `overrides`. Each file's overrides apply in include order, and the
  root configuration's last, so the root has the final word. A file path an
  included file's override sets is relative to that file.
- `qavf explain` lists each applied override: the instance it changed, the
  settings it set and the file it came from.

### Views

Views behave the same on both backends: each is the parent dataset's rows that
match `where`, with `columns` (or all columns), and SQL in `sql` and `expression`
checks reads it as `"<dataset>.<view>"`.

A foreign key may reference a view or an `across` copy (or a copy's view) as its
`ref_dataset`: `{columns: [road_id], ref_dataset: roads.active, ref_columns: [id]}`
requires each `road_id` to be the `id` of an open road. The view keeps the
parent's contract (its column types and N/A tokens), so a key the parent holds
but the view's `where` excludes is orphaned, and match policy, cardinality and
evidence are those of a physical parent. The parent dataset is loaded with the
child on the native backend, with the columns the `where` reads.

## Selection and the run section

```yaml
run:
  fail_on: ERROR                     # lowest severity that fails the run
  fail_on_incomplete: true           # a check that could not run fails the run
  require: [no missing years, counts]  # check names, or a dataset for all its checks
  tags:
    exploratory: {fail_on: never}    # per-tag policy
```

Narrow a run without editing the file: `qavf run --tags nightly`, `--skip-tags slow`,
`--only "no missing years"` (repeatable) or `--dataset counts` (that dataset's checks
and schema). `validate(..., only=, tags=, skip_tags=, datasets=)` takes the same
selectors. Built-in checks carry their family as a tag (`input`, `temporal`,
`statistical`, `reconciliation`, `spatial`, `regression`, `comparison`), so
`--tags temporal` works without configuration. A `require` entry that matches
nothing is a configuration error.

A `require` entry naming a check stands for every item with that name (in any
dataset's list), and **each** of them must be assessed: one that is disabled,
skipped, produced no result or could not run leaves the requirement unmet. A
dataset entry requires every check on that dataset and its schema contracts. A
check may not be named like a dataset, so an entry is never ambiguous. A run-time
selection narrows requirements: items it leaves out are not required, and entries
matching no selected item are dropped.

## Many datasets

### One dataset across many sources

`across` defines one dataset once and reads it from several places. Each copy is a
full dataset named `<dataset>@<copy>`, with the dataset's schema and checks.

```yaml
connections:
  houston: {type: duckdb, path: data/houston.duckdb}
  dallas: {type: duckdb, path: data/dallas.duckdb}
datasets:
  volumes:
    source: {type: database, table: volumes}
    across: [houston, dallas]                      # a list: connection names
  stations:
    path: 'data/{city}/stations.csv'
    across: {city: [houston, dallas, austin]}      # a template filling {city}
  readings:
    path: data/readings.csv
    across:                                        # the general form: source overrides
      live: {}
      archive: {path: data/archive/readings.csv}
```

`volumes@*` selects every copy wherever datasets are named, as one string
(`datasets: volumes@*`) or in a list; `volumes@houston` selects one. Each copy
is named once: a repeated connection, copy name or template value (also as text,
like `2023` and `'2023'`) is a configuration error naming it.

### Strategies

`compare`, `totals_match`, `key_sets_match` and `drift` take any number of
`datasets` and a `strategy`; the two-sided check runs once per pair.

| `strategy` | Pairs for A, B, C, D | Use when |
|---|---|---|
| `reference` (default) | A–B, A–C, A–D | One source is authoritative; `reference:` names it (default: the first) |
| `sequential` | A→B, B→C, C→D | Versions or years in order |
| `all_pairs` | every pair | Peers without an authority; refused above `max_pairs` (default 28) |
| `consensus` | each against the consensus | Finding the odd one out, matched on `keys` |

```yaml
checks:
  - check: compare
    name: cities agree
    datasets: stations@*
    strategy: consensus
    keys: [station]
    consensus:                       # the defaults
      numeric: median                # median | mean | trimmed_mean
      trim: 0.1                      # share trimmed from each end for trimmed_mean
      categorical: majority          # majority (more than half) | plurality
      min_agreement: 0.5             # share of members that must agree
      ties: report                   # report | reference (the reference member's value)
```

A key without a consensus value (a tie, or too little agreement) is reported as not
assessed for that key, never guessed. `batch_consistent` and `coverage` already
compare all their members at once. Two datasets with the default strategy behave
exactly like a single comparison.

### Whole-group levels and derived datasets

With more than two members, `compare` assesses three levels over all members at
once: `coverage` (a key × member presence matrix), `structure` (column names and
types) and `distribution` (each member against the pooled data). To do so a check
derives datasets from its members: the consensus, the pooled rows of every member,
and each member's rows that have a consensus. These are internal to the check;
reports name them by role (`consensus`, `pooled`) or by member, and only the
run's audit metadata lists them.

Statistical tests are corrected for multiple testing within each check item, as
set by its `correction`; items are never pooled into one family.

### Reports

`report.groups` holds one `GroupReport` per check relating more than two members
(or using a strategy other than the default):

| Field | Holds |
|---|---|
| `name`, `check`, `rule_id`, `strategy`, `members` | The check item and its members |
| `levels`, `whole_group_levels` | The matrix columns; the levels assessed over all members |
| `matrix` | One row per member: the worst status at each level |
| `pairs` | Each pair compared (`reference`, `other`, `comparison` label, counts by status) |
| `ranking` | Members by failures, unassessed results and changed records; rank 1 has the most problems |
| `group_results` | Results about the group as a whole, such as keys without a consensus |

Each result of such a check carries a `group` (`GroupRole`): the item (`group`),
`check`, `strategy`, the `members` the result is about, every member
(`group_members`), the `pair` compared, the comparison `level` and whether it is a
`whole_group` level. The HTML report shows the ranking, the matrix and the pairs.

## Migrating from QAVF 0.6

A QAVF 0.6 configuration fails fast with an error naming its first old key;
`qavf migrate old.yaml -o new.yaml` converts it once (`--check` reports what would
change without writing; `--drop-disabled` leaves disabled rules out).

| QAVF 0.6 | QAVF 0.7 |
|---|---|
| `rules.<CODE>.params` lists | One check item per entry; inline series and metrics become named `series` and `metrics` |
| Rule-level `severity`, `enabled: false` and parameters | Settings of each generated item, with the catalog's renames |
| `datasets.<d>.expressions`, `assertions` | `check: expression` items; top-level `check: sql` items |
| `left`/`right`, `baseline`/`candidate`, `reference` | `datasets: [a, b]` with `strategy: reference` |
| `min`, `max`, `mean`, `median`, `std`, `sum`, `distinct_count` on a column | The same under the column's `stats` |
| A column's `checks` | The column's `conditions` |
| `engine.fail_on`, `fail_on_incomplete`, `require` | `run` (`require` entries become check names) |
| `engine.run_unconfigured_rules`, `include_categories`, `exclude_categories` | Removed; `migrate` prints the equivalent `--tags` / `--skip-tags` |

Reports written by older versions still load; only configurations change.
