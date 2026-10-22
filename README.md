# QAVF

[![CI](https://github.com/AhmedAredah/qavf/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedAredah/qavf/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qavf)](https://pypi.org/project/qavf/)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/AhmedAredah/qavf/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/AhmedAredah/qavf/blob/main/LICENSE)

**Configurable data QA/QC for Python.** Validate files, dataframes and database
queries against declared limits, statistical expectations and stored baselines.
QAVF reports exact violation counts with bounded evidence, and it never reports a
pass for a check it could not assess.

- **Any input.** CSV, Parquet, JSON, Excel, globs and URLs; DuckDB, SQLite, PostgreSQL
  and MySQL tables; SQL queries; Polars, pandas and Arrow frames; Python loaders.
- **Column limits.** Types, nullability, ranges, allowed values, patterns, lengths,
  uniqueness, keys, foreign keys and aggregate statistics, each with its own severity.
- **Statistical and temporal QA.** Trend, jump, seasonal and level-shift checks; entity
  outliers; coverage gaps; distribution drift; reconciliation of totals across datasets.
- **Regression testing.** Keyed, tolerance-aware comparison against accepted baselines,
  usable from pytest.
- **Dataset comparison.** Compare two versions of a dataset whose columns, types,
  units or codes differ: structure, key coverage, record-level changes and grouped
  measures, each judged against a tolerance.
- **Reports and CI gates.** HTML and JSON reports, deterministic evidence and exit codes.
- **Extensible.** Rules, sources and comparison levels, measures and comparators are
  plain classes discovered from packages, files or entry points.

## Install

```bash
pip install "qavf[stats]"                                     # or: uv tool install "qavf[stats]"
brew install AhmedAredah/tap/qavf                             # macOS and Linux command line
docker run --rm -v "$PWD:/work" ghcr.io/ahmedaredah/qavf run -c qavf.yaml
```

Python 3.12 or newer on Linux, macOS or Windows. The `stats` extra adds SciPy-based
tests; `pandas` adds pandas adapters; `all` installs both. In GitHub Actions, use
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
        aadt:
          dtype: int
          ge: 0
          checks:
            - {le: 200000, severity: WARNING, name: implausibly high}
        truck_share: {dtype: float, ge: 0, le: 1}
    foreign_keys:
      - {columns: [site_id], ref_dataset: sites}
  sites:
    path: data/sites.csv
    schema: {primary_key: [site_id]}
```

Row expressions and SQL assertions run in DuckDB, where each dataset is a table:

```yaml
datasets:
  crashes:
    path: data/crashes.parquet
    expressions:
      - {expr: fatal <= crashes, severity: CRITICAL}
assertions:
  - name: every site has counts
    sql: SELECT site_id FROM sites WHERE site_id NOT IN (SELECT site_id FROM counts)
```

[`config/qavf.full.yaml`](https://github.com/AhmedAredah/qavf/blob/main/config/qavf.full.yaml)
documents every option. Paths are relative to the configuration file, and any string
may use `${ENV_VAR}` so credentials stay out of the file.

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

report = validate({
    "counts": Dataset(counts_frame, schema={"columns": {"count": {"ge": 0}}}),
    "sites": Dataset("sites.parquet", schema={"primary_key": ["site_id"]}),
})

# Run in a supervised worker process with OS-enforced limits.
report = validate("readings.parquet", config="qavf.yaml",
                  resources=ResourceLimits(wall_time=60, address_space="4GiB"))
```

## How a run is judged

Each result is **PASS**, **FAIL**, **SKIPPED** (nothing configured) or **ERROR**
(configured but not assessed, for example missing data or invalid SQL).

| Verdict | When | Exit code |
|---|---|---|
| passed | Every configured check was assessed and no failure meets `engine.fail_on` | 0 |
| failed | A failure at or above `engine.fail_on` (default `ERROR`) | 1 |
| incomplete | Any configured check could not be assessed | 1 |

Use `engine.require` to name checks that must be assessed for a run to pass.

## Rules

| ID | Default | Checks |
|---|---|---|
| INP-001 | CRITICAL | Sources accessible and schemas inspectable |
| INP-002 | CRITICAL | Columns present with declared types; row-count limits |
| INP-003 | ERROR* | Column QA limits (*per-check severities) |
| INP-004 | ERROR | Primary keys unique |
| INP-005 | ERROR* | Foreign keys resolve |
| INP-006 | ERROR* | Row expressions hold |
| INP-007 | ERROR* | SQL assertions return no rows |
| INP-008 | ERROR | Newest date/time value is recent enough |
| TMP-001 | WARNING | Period-over-period change within a rate |
| TMP-002 | WARNING | Series trend in their declared direction |
| TMP-003 | WARNING | No period is a significant and material trend outlier |
| TMP-004 | ERROR | No group is missing a period |
| TMP-005 | WARNING | No value stuck for several periods |
| TMP-006 | ERROR | Cumulative series never decrease |
| TMP-007 | WARNING | No value is anomalous for its season |
| TMP-008 | WARNING | No series shifts to a sustained new level |
| SPA-001 | WARNING | No entity is an outlier among its peers |
| SPA-002 | WARNING | Each entity within a factor of its group's median |
| SPA-003 | ERROR | Every expected entity is covered in every period |
| GEO-001 | ERROR | Coordinates valid, present and inside a bounding box |
| STA-001 | WARNING | z-score, modified z-score or IQR outlier scans |
| STA-002 | WARNING | Distributions have not drifted from a reference |
| REC-001 | ERROR | Totals reconcile between datasets, key by key |
| REC-002 | ERROR | Shares sum to their total within each group |
| REC-003 | ERROR | The full expected key grid is present |
| REC-004 | ERROR | Two datasets have the same key set |
| REC-005 | ERROR | Related datasets carry one identical batch id |
| REG-001 | ERROR | Outputs match baselines within tolerance |
| CMP-001 | ERROR | Two datasets compare within tolerance after alignment |

Parameters for every rule: [docs/rules.md](https://github.com/AhmedAredah/qavf/blob/main/docs/rules.md)
or `qavf list-rules`.

## Regression testing

```python
from qavf.regression import assert_outputs_match


def test_output_matches_baseline(tmp_path):
    run_pipeline(output=tmp_path / "out.parquet")
    assert_outputs_match("tests/baselines/out.parquet", tmp_path / "out.parquet",
                         keys=["site_id", "year"], rtol=1e-6)
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
    "v1/counts.parquet", "v2/counts.parquet",
    keys=["site_id", "year"],
    columns={"AADT": "aadt"},                      # baseline name -> candidate name
    units={"aadt": {"factor": 1000}},              # v1 counted thousands of vehicles
    recode={"fclass": {"values": {"1": "interstate", "2": "arterial"}}},
    levels=["structure", "coverage", "records",
            {"level": "groups", "group_by": ["county"], "measures": ["count", "sum(aadt)"]}],
)
print(result.summary())
```

Each level reports graded checks with exact counts and bounded evidence: columns
added or removed (with suggested renames), keys on one side only, changed values per
column with their largest differences, and material changes of grouped measures. A
lossy cast or a missing mapped column makes the comparison not assessed, never a
pass. In a configuration, `CMP-001` compares configured datasets. See
[dataset comparison](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison.md).

## CLI

| Command | Purpose |
|---|---|
| `qavf run -c CONFIG` | Run QA and write HTML/JSON reports |
| `qavf init -d [NAME=]PATH` | Profile data and write a starting configuration |
| `qavf profile PATH` | Print inferred QA limits for a file |
| `qavf schema` | Write the configuration JSON Schema for editor support |
| `qavf list-rules` | List discovered rules |
| `qavf compare BASELINE CANDIDATE -k KEY` | Compare an output with a baseline (`--map OLD=NEW`) |
| `qavf baseline SOURCE DEST -k KEY` | Accept an output as a baseline |
| `qavf demo DIR` | Write a runnable demo project |

Exit codes: `0` passed, `1` QA failed or outputs differ, `2` usage, configuration or
resource error.

## Documentation

- [Python API](https://github.com/AhmedAredah/qavf/blob/main/docs/api.md) and
  [sources](https://github.com/AhmedAredah/qavf/blob/main/docs/sources.md)
- [Dataset comparison](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison.md),
  its [component reference](https://github.com/AhmedAredah/qavf/blob/main/docs/comparison_reference.md)
  and [statistical inference](https://github.com/AhmedAredah/qavf/blob/main/docs/inference.md)
- [Rules reference](https://github.com/AhmedAredah/qavf/blob/main/docs/rules.md) and
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

See [release assurance](https://github.com/AhmedAredah/qavf/blob/main/docs/release_assurance.md)
for the full verification matrix and [CHANGELOG.md](CHANGELOG.md) for changes.

## License

[MIT](LICENSE)
