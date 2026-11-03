# QAVF

[![CI](https://github.com/AhmedAredah/qavf/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedAredah/qavf/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qavf)](https://pypi.org/project/qavf/)
[![Downloads](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/AhmedAredah/qavf/badges/downloads.json)](https://pypistats.org/packages/qavf)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/AhmedAredah/qavf/blob/main/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/AhmedAredah/qavf/blob/main/LICENSE)

**Configurable data QA/QC for Python.** Validate files, dataframes and database
queries against declared limits, statistical expectations and stored baselines.
QAVF reports violation counts with bounded evidence. Counts are exact unless
explicitly labelled as lower bounds, and an insufficient lower bound cannot
establish a pass.

- **Any input.** CSV, Parquet, JSON, Excel, globs and URLs; DuckDB, SQLite, PostgreSQL
  and MySQL tables; SQL queries; Polars, pandas and Arrow frames; Python loaders.
- **Column limits.** Types, nullability, ranges, allowed values, patterns, lengths,
  uniqueness, keys, foreign keys and aggregate statistics, each with its own severity.
- **Statistical and temporal QA.** Trend, jump, seasonal and level-shift checks; entity
  outliers; coverage gaps; distribution drift; reconciliation of totals across datasets.
- **Regression testing.** Keyed, tolerance-aware comparison against accepted baselines,
  usable from pytest.
- **Dataset comparison.** Compare two versions of a dataset whose columns, types,
  units or codes differ: structure, key coverage, record-level changes, grouped
  measures and time series, distributions, relationships between columns and what
  explains the total change, a difference failing when it is both material (beyond a
  tolerance) and statistically significant.
- **Spatial data.** Coordinates inside a study area, points assigned to zone
  polygons or grid cells, per-zone comparisons with expected zones, rank and share
  changes, and Moran's I and LISA naming the clusters of change; features that moved
  or changed shape. Geometry uses DuckDB's `spatial` extension, and without it those
  checks are not assessed, never passed.
- **Reports and CI gates.** HTML and JSON reports, deterministic evidence and exit codes.
- **Extensible.** Rules, sources, comparison levels, measures, comparators and
  statistical tests are plain classes discovered from packages, files or entry points.

## Install

```bash
pip install "qavf[stats]"                                     # or: uv tool install "qavf[stats]"
brew install AhmedAredah/tap/qavf                             # macOS and Linux command line
docker run --rm -v "$PWD:/work" ghcr.io/ahmedaredah/qavf run -c qavf.yaml
```

Python 3.12 or newer on Linux, macOS or Windows. `pip install qavf` provides
DuckDB/Arrow validation without Polars. The `polars` extra installs the full
execution backend; `stats` adds that backend and SciPy-based tests; `pandas` adds
pandas adapters; `spatial` adds the full backend, indexed neighbour searches and
authoritative CRS resolution; `all` installs every extra. In GitHub Actions, use
[`AhmedAredah/qavf-action`](https://github.com/AhmedAredah/qavf-action).

## Quick start

```bash
qavf demo my-demo && qavf run -c my-demo/qavf.yaml   # open my-demo/qa_output/qa_report.html
```

For your own data, generate a configuration with inferred limits, review it, then run it:

```bash
qavf init -d data/counts.csv -d crashes=data/crashes.parquet --margin 0.1
qavf run -c qavf.yaml
```

```yaml
datasets:
  counts:
    path: data/counts.csv
    schema:
      primary_key: [site_id, year]
      columns:
        site_id: {dtype: str, regex: 'S\d{2}'}
        year: {dtype: int, between: [2000, 2030]}   # inclusive; or ge, le, gt, lt
        aadt:
          dtype: int
          ge: 0
          stats: {mean: {between: [1000, 100000]}}  # min max mean median std sum distinct_count
          conditions:                               # more limits, each with its own severity
            - {le: 200000, severity: WARNING, name: implausibly high}
        truck_share: {dtype: float, between: [0, 1]}
    foreign_keys:
      - {columns: [site_id], ref_dataset: sites}
  sites:
    path: data/sites.csv
    schema: {primary_key: [site_id]}
```

Other checks are named in `checks` lists: on a dataset for checks of that dataset,
at the top level for checks relating datasets. Only configured checks run; schema
contracts always do. Row expressions and SQL queries run in DuckDB, where each
dataset is a table:

```yaml
vars:
  years: 2020..2023                 # every year from 2020 to 2023
defaults:
  change_rate: {max: 0.15}          # unless an item sets its own
datasets:
  counts:
    path: data/counts.csv
    series:                         # defined once, named by the checks
      aadt_by_site: {value: aadt, group_by: [site_id], time_column: year}
    views:                          # a slice, used as counts.recent
      recent: {where: year >= 2022}
    checks:
      - check: missing_periods
        name: no missing years
        series: aadt_by_site
        expected: ${vars.years}
      - {check: change_rate, series: aadt_by_site, tags: [nightly]}
  crashes:
    path: data/crashes.parquet
    metrics:
      crash_rate: {value: crashes, entity: site_id, group_by: [year], per: counts.aadt}
    checks:
      - {check: expression, expr: fatal <= crashes, severity: CRITICAL}
      - {check: entity_outliers, metric: crash_rate}
checks:
  - check: sql
    name: every site has counts
    sql: SELECT site_id FROM sites WHERE site_id NOT IN (SELECT site_id FROM counts)
  - {check: key_sets_match, datasets: [counts.recent, crashes], keys: [site_id, year]}
run:
  tags:
    nightly: {fail_on: never}       # report, but never fail the run
```

`include: [shared.yaml]` merges shared files first (this file wins); relative file
paths in an included file are relative to that file. Narrow a run
without editing the file: `qavf run -c qa.yaml --tags nightly`, `--skip-tags slow`,
`--only "no missing years"` or `--dataset counts`; built-in checks carry their
family (`temporal`, `spatial`, `input`, ...) as a tag. The same selectors exist on
`validate(..., only=, tags=, skip_tags=, datasets=)`.

`qavf list-checks` lists every check, and [configuring checks](docs/checks.md) explains
items, reuse, selection and checks over many datasets. Configurations written for
QAVF 0.6 (`rules:`, `expressions:`, `assertions:`, column statistics such as `mean:`
beside the column's limits, a column's `checks:`) are rejected with a pointer to
`qavf migrate old.yaml -o new.yaml`, which converts them.

[`config/qavf.full.yaml`](https://github.com/AhmedAredah/qavf/blob/main/config/qavf.full.yaml)
documents every option. Paths are relative to the configuration file, and any string
may use `${ENV_VAR}` so credentials stay out of the file.

See [backends and installation](docs/backends.md), [key contracts](docs/key_contracts.md),
[not-applicable values](docs/applicability.md) and [value policies](docs/value_policies.md)
for nullable keys, N/A tokens, number grammar and what each backend supports.

## Cloud storage

Read `s3://`, `az://` and `abfss://` URLs with credentials configured once. AWS
roles, Azure identity and GitHub OIDC logins work without stored keys:

```yaml
secrets:
  lake: {type: s3, provider: credential_chain, region: us-east-1}
  blob: {type: azure, connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"}
datasets:
  orders: {path: "s3://my-lake/orders/*.parquet"}
```

S3-compatible stores (Cloudflare R2, MinIO, Wasabi) take an `endpoint`. See
[cloud storage](https://github.com/AhmedAredah/qavf/blob/main/docs/cloud_storage.md).

## Python API

```python
from qavf import Dataset, ResourceLimits, validate

report = validate(
    "readings.parquet",
    schema={"columns": {"value": {"dtype": "float", "ge": 0}}},
)
print(report.verdict())

report = validate(
    {
        "counts": Dataset(counts_frame, schema={"columns": {"count": {"ge": 0}}}),
        "sites": Dataset("sites.parquet", schema={"primary_key": ["site_id"]}),
    }
)

# Validate the configured inputs in a supervised worker with a wall-time deadline.
report = validate(config="qavf.yaml", resources=ResourceLimits(wall_time=60))
```

Add `address_space="4GiB"` to limit each process's virtual address space on Linux
or committed memory on Windows; macOS rejects this control. See
[execution limits](docs/execution_limits.md) for the supported controls and their scopes.

## How a run is judged

Each result is **PASS**, **FAIL**, **SKIPPED** (nothing configured) or **ERROR**
(configured but not assessed, for example missing data or invalid SQL).

| Verdict | When | Exit code |
|---|---|---|
| passed | At least one check was assessed, none reports ERROR, and no failure meets `run.fail_on` | 0 |
| failed | A failure at or above `run.fail_on` (default `ERROR`) | 1 |
| incomplete | No qualifying failure, but a check could not be assessed or nothing was assessed | 1 by default |

A qualifying failure takes precedence over incompleteness. Setting
`run.fail_on_incomplete: false` allows exit code `0` for an incomplete report;
its verdict remains `incomplete`.

Use `engine.require` to name checks that must be assessed for a run to pass.

### Reading evidence

A failed check lists the first offending rows, up to `engine.max_affected_records`, with
their values and why each fails (`violates`). `_row` is a row's 0-based position in
the dataset as read: the first data row of a file with a header is `_row: 0`. A
dataset read from several files adds `_file` and `_file_row`, the row's position
within that file. Duplicate keys list their positions in `rows`, and a line the
reader could not parse is named by its file and 1-based `line`.

A column check's failing rows show the columns the check reads (the checked column,
the columns of its `where` and its unit column) and the `primary_key`; other columns
are not read for evidence. Set `evidence_columns: all` in a dataset's `schema` to show
whole rows.

## Checks

Each check has a name used in configurations; reports also show its rule code.

| Check | Code | Default | Checks |
|---|---|---|---|
| `datasets_available` | INP-001 | CRITICAL | Sources accessible and schemas inspectable |
| `schema` | INP-002 | CRITICAL | Columns present with declared types; row-count limits |
| `column_limits` | INP-003 | ERROR* | Column QA limits (*per-check severities) |
| `unique_keys` | INP-004 | ERROR | Primary keys unique |
| `references` | INP-005 | ERROR* | Foreign keys resolve |
| `expression` | INP-006 | ERROR* | Row expressions hold |
| `sql` | INP-007 | ERROR* | SQL assertions return no rows |
| `fresh` | INP-008 | ERROR | Newest date/time value is recent enough |
| `not_applicable` | INP-009 | ERROR | N/A tokens apply to all or none of a column group |
| `change_rate` | TMP-001 | WARNING | Period-over-period change within a rate |
| `trend` | TMP-002 | WARNING | Series trend in their declared direction |
| `trend_outliers` | TMP-003 | WARNING | No period is a significant and material trend outlier |
| `missing_periods` | TMP-004 | ERROR | No group is missing a period |
| `stale_values` | TMP-005 | WARNING | No value stuck for several periods |
| `cumulative` | TMP-006 | ERROR | Cumulative series never decrease |
| `seasonal` | TMP-007 | WARNING | No value is anomalous for its season |
| `level_shift` | TMP-008 | WARNING | No series shifts to a sustained new level |
| `entity_outliers` | SPA-001 | WARNING | No entity is an outlier among its peers |
| `group_comparison` | SPA-002 | WARNING | Each entity within a factor of its group's median |
| `coverage` | SPA-003 | ERROR | Every expected entity is covered in every period |
| `coordinates` | GEO-001 | ERROR | Coordinates valid, present and inside a bounding box |
| `outliers` | STA-001 | WARNING | z-score, modified z-score or IQR outlier scans |
| `drift` | STA-002 | WARNING | Distributions have not drifted from a reference |
| `totals_match` | REC-001 | ERROR | Totals reconcile between datasets, key by key |
| `shares` | REC-002 | ERROR | Shares sum to their total within each group |
| `key_grid` | REC-003 | ERROR | The full expected key grid is present |
| `key_sets_match` | REC-004 | ERROR | Two datasets have the same key set |
| `batch_consistent` | REC-005 | ERROR | Related datasets carry one identical batch id |
| `matches_baseline` | REG-001 | ERROR | Outputs match baselines within tolerance |
| `compare` | CMP-001 | ERROR | Two datasets compare within tolerance after alignment |

Check names and settings: [docs/rules.md](https://github.com/AhmedAredah/qavf/blob/main/docs/rules.md)
or `qavf list-checks`.

## Regression testing

```python
from qavf.regression import assert_outputs_match


def test_output_matches_baseline(tmp_path):
    run_pipeline(output=tmp_path / "out.parquet")
    assert_outputs_match(
        "tests/baselines/out.parquet", tmp_path / "out.parquet", keys=["site_id", "year"], rtol=1e-6
    )
```

Rows match by key, numbers within tolerance, and changed values, added or removed rows
and schema changes are reported. Accept reviewed outputs with `qavf baseline`, which
records a manifest with the keys, column types and a SHA-256 of the file.

## Dataset comparison

Compare a new version of a dataset with the previous one, even when columns were
renamed, retyped, rescaled or recoded:

```python
from qavf import compare_datasets

result = compare_datasets(
    "v1/counts.parquet",
    "v2/counts.parquet",
    keys=["site_id", "year"],
    columns={"AADT": "aadt"},  # baseline name -> candidate name
    units={"aadt": {"factor": 1000}},  # v1 counted thousands of vehicles
    recode={"fclass": {"values": {"1": "interstate", "2": "arterial"}}},
    levels=[
        "structure",
        "coverage",
        "records",
        {"level": "groups", "group_by": ["county"], "measures": ["count", "sum(aadt)"]},
    ],
)
print(result.summary())
```

Each level reports graded checks with exact counts and bounded evidence: columns
added or removed (with suggested renames), keys on one side only, changed values per
column with their largest differences, and changes of grouped measures that are
material and statistically significant (with p-values, effects and intervals; the
`stats` extra, or `decision="material"` for tolerances alone). A lossy cast or a
missing mapped column makes the comparison not assessed, never a pass. In a configuration, `CMP-001` compares configured datasets. See
[dataset comparison](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison.md).

Spatially, the `assign_zones` step gives points the zone they fall in (a polygon
layer from GeoParquet, GeoJSON or a shapefile, or grid cells), and the `spatial`
level compares zones: one-sided and missing zones, measures per zone, rank and share
changes, and whether the changes cluster (Moran's I, and LISA naming hot and cold
spots after a multiple-testing correction). The `locations` level reports keyed
features that moved (haversine metres) or changed area or shape. See
[spatial data](https://github.com/AhmedAredah/qavf/blob/main/docs/spatial.md).

The `distribution` level compares each column's distribution per group, zone or time
bucket (PSI, KS, Anderson-Darling, Wasserstein, Jensen-Shannon, quantile deltas, and
categories that appeared or disappeared), `relationships` compares correlations, ratio
invariants such as `fatal / crashes` and the agreement of matched records (Bland-Altman,
concordance, error metrics, the candidate-on-baseline line), and `contribution` says
which members explain the total change, with an exact mix-versus-rate decomposition.
HTML reports show each comparison with a verdict per level, a drill-down with tests
and charts; `result.write_html(path)` and `qavf compare-datasets` write the same report
for one comparison.

### Many datasets

A check relating datasets (`compare`, `totals_match`, `key_sets_match`, `drift`) takes
any number of them and a `strategy`: `reference` (the default; each against
`reference:`, the first unless named), `sequential` (each against the next),
`all_pairs` (every pair, up to `max_pairs`, 28 by default) or `consensus` (each
against the consensus of all, matched on `keys`). `across` copies one dataset across
sources, and `counts@*` lists every copy:

```yaml
datasets:
  counts:
    path: 'data/{city}/counts.parquet'
    across: {city: [houston, dallas, austin, el_paso]}   # counts@houston, ...
    schema: {columns: {site_id: {dtype: str}, aadt: {dtype: int, ge: 0}}}
checks:
  - check: compare
    name: cities agree
    datasets: counts@*               # every copy; one selector or a list
    keys: [site_id]
    strategy: consensus
    consensus: {numeric: median, categorical: majority, min_agreement: 0.5, ties: report}
  - {check: key_sets_match, datasets: ['counts@*'], keys: [site_id], strategy: all_pairs}
```

The consensus takes the median of the numbers and the majority of other values, key
by key; a key without one (a tie, or too little agreement) is reported as not
assessed, never guessed. With more than two members, `compare` assesses `coverage`
(a key x member presence matrix), `structure` and `distribution` (each member against
the pooled data) over all members at once. Reports add, per check, a matrix of
members by levels (each cell the worst status), the pairs compared and a ranking of
the members by failures and changed records (`report.groups`, one `GroupReport` per
check); each result's `group` (a `GroupRole`) names the members it is about. Results
show the consensus and pooled data by role, never by an internal dataset name. Two
datasets with the default strategy report exactly as a single comparison. See
[configuring checks](docs/checks.md#many-datasets).

## CLI

| Command | Purpose |
|---|---|
| `qavf run -c CONFIG` | Run QA and write HTML/JSON reports |
| `qavf init -d [NAME=]PATH` | Profile data and write a starting configuration |
| `qavf profile PATH` | Print inferred QA limits for a file |
| `qavf schema` | Write the configuration JSON Schema for editor support |
| `qavf list-checks` | List the checks a configuration can use |
| `qavf migrate OLD -o NEW` | Convert a QAVF 0.6 configuration to named checks |
| `qavf capabilities` | Show the backend capability catalog as JSON |
| `qavf compare BASELINE CANDIDATE -k KEY` | Compare an output with a baseline (`--map OLD=NEW`) |
| `qavf compare-datasets BASELINE CANDIDATE -c SPEC` | Compare two datasets at several levels; write `--html`/`--json` reports |
| `qavf baseline SOURCE DEST -k KEY` | Accept an output as a baseline |
| `qavf demo DIR` | Write a runnable demo project |

`--help`, `--version`, `schema`, `list-checks` and native `run` work with the core
installation. `run`, `schema` and `list-checks` accept `--backend auto|duckdb|polars`;
`auto` uses Polars when installed, otherwise DuckDB. Advanced profiling, comparison,
scaffolding and demo commands require `qavf[polars]`. Inspect the import-light
[capability catalog](docs/backend_capabilities.md) before selecting a backend.

Exit codes: `0` passed, `1` QA failed or outputs differ, `2` usage, configuration or
resource error.

## Documentation

- [Python API](https://github.com/AhmedAredah/qavf/blob/main/docs/api.md) and
  [sources](https://github.com/AhmedAredah/qavf/blob/main/docs/sources.md)
- [Dataset comparison](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison.md),
  its [component reference](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison_reference.md)
  and [statistical inference](https://github.com/AhmedAredah/qavf/blob/main/docs/inference.md)
- [Spatial data](https://github.com/AhmedAredah/qavf/blob/main/docs/spatial.md): zone
  layers, CRS, assignment policies, grids, crosswalks, weights and the spatial levels
- [Configuring checks](https://github.com/AhmedAredah/qavf/blob/main/docs/checks.md),
  the [check reference](https://github.com/AhmedAredah/qavf/blob/main/docs/rules.md) and
  [writing plugins](https://github.com/AhmedAredah/qavf/blob/main/docs/plugins.md)
- [Cloud storage](https://github.com/AhmedAredah/qavf/blob/main/docs/cloud_storage.md)
- [Resource limits](https://github.com/AhmedAredah/qavf/blob/main/docs/resources.md) and
  [isolated execution](https://github.com/AhmedAredah/qavf/blob/main/docs/execution_limits.md)
- [Numerical contracts](https://github.com/AhmedAredah/qavf/blob/main/docs/numerics.md) and
  [baseline acceptance](https://github.com/AhmedAredah/qavf/blob/main/docs/baseline_acceptance.md)

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run mypy qavf scripts
```

See [releasing](https://github.com/AhmedAredah/qavf/blob/main/docs/releasing.md)
for the release checks and [CHANGELOG.md](CHANGELOG.md) for changes.

## License

[Apache License 2.0](LICENSE)
