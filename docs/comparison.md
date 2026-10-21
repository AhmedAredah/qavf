# Dataset comparison

`compare_datasets` (and the `CMP-001` rule) compares a *candidate* dataset with a
*baseline*: a new version of an output with the previous one, a migrated table with
its source, or two producers of the same quantity. The datasets may differ in column
names, types, units and codes; an **alignment pipeline** makes them comparable first,
then each configured **level** compares them at one level of detail.

```python
from qavf import compare_datasets

result = compare_datasets(
    "v1/inventory.parquet",
    "v2/inventory.parquet",
    keys=["county", "year", "pollutant"],
    columns={"emissions_t": "emissions"},          # baseline name -> candidate name
    units={"emissions": {"factor": 1000, "from_unit": "t", "to_unit": "kg"}},
    levels=[
        "structure",
        "coverage",
        {"level": "records", "top_n": 5},
        {"level": "groups", "group_by": ["pollutant"],
         "measures": ["count", "sum(emissions)", {"measure": "quantile", "column": "emissions", "p": 0.9}],
         "rank": {"measure": "sum(emissions)", "top_k": 5}},
    ],
    rtol=1e-6,
)
print(result.summary())
for level in result.levels:
    for check in level.checks:
        print(level.name, check.check, check.passed, check.message)
```

Inputs take every form `validate` accepts: paths, `File`, `Dataset` (its schema's
`missing_values`, `unit` and `nullable` declarations apply), `Query`, `Loader` and
Polars, pandas or Arrow frames. `resources=ResourceLimits(...)` applies the same
resource policy, including isolated execution for file and query inputs.

## Fail-closed results

A `DatasetComparison` holds one typed result per level (`StructureResult`,
`CoverageResult`, `RecordsResult`, `GroupsResult`), each with graded checks. A check
has an exact `affected_count` and bounded `affected_records`: evidence follows the
run's `max_affected_records` and `max_evidence_bytes` limits and is ordered
deterministically, while counts stay complete.

Nothing that could not be compared reads as agreement. A dataset that cannot be read,
a mapped column the baseline lacks, a lossy cast, an unmapped category with
`unmapped: error`, fewer aligned rows than `min_rows` (1 by default), duplicate keys
for the record level or a non-numeric column for a numeric measure all leave the
affected checks **not assessed**. `result.passed` is true only when every check was
assessed and passed.

## Alignment

Steps run in this order; after `columns`, every column reference (keys, tolerances,
level parameters) uses **candidate names**.

| Option | Step | Example |
|---|---|---|
| `columns` | Rename baseline columns to candidate names. A mapped column the baseline lacks refuses the alignment. | `{AADT: aadt}` |
| `cast` | Convert types losslessly (`int`, `float`, `decimal` with `scale`, `str`, `bool`, `date`, `datetime`) on `both` sides or one. A value that would change refuses the alignment. | `{site: str, rate: {dtype: decimal, scale: 4}}` |
| `units` | Multiply one side's column by an exact factor; integers and decimals stay exact. `from_unit` must match the declared `unit`; `to_unit` becomes it. | `{mass: {factor: 1000, to_unit: kg}}` |
| `recode` | Replace values through a map; unmapped values are kept, made null or refused. | `{fc: {values: {"1": interstate}}}` |
| `derive` | Add the same Polars SQL expression on both sides. | `{total: "cars + trucks"}` |
| `steps` | Plugin (or built-in) steps, `{step: name, ...}`. | `[{step: trim, columns: [name]}]` |
| `baseline_where`, `candidate_where`, `where` | Keep only rows in scope on one or both sides. | `"year >= 2020"` |

Dataset-level `rename`, `missing_values` and `canonical_columns` apply before the
pipeline, exactly as for every other rule.

## Levels

| Level | Compares | Checks |
|---|---|---|
| `structure` | Columns added or removed after mapping, suggested renames (similar names and similar value samples, never applied), type, declared-unit and nullability changes, row counts (exact unless `rows: {rtol: ...}`), key uniqueness per side | `columns`, `types`, `units`, `nullability`, `rows`, `keys` |
| `coverage` | Keys only in the baseline or only in the candidate, overall and by `group_by`; changes of each column's missing-value rate beyond `max_null_rate_change` | `keys`, `null rates` |
| `records` | Keyed match through the regression engine: rows added, removed, changed, unchanged; per-column change counts and rates; largest absolute and relative differences; null and NaN transitions; categorical transitions (`max_transitions` most frequent, with the total count); the `top_n` largest numeric changes beyond tolerance as evidence | `rows`, `values` |
| `groups` | Measures per `group_by` group on each side; groups on one side only; material changes per measure; optional top-k rank changes | `groups`, one per measure, `rank: <measure>` |

`allow_added` / `allow_removed` accept one-sided columns, keys or rows where they are
expected. Every level's parameters are listed in the
[component reference](comparison_reference.md).

### Measures and comparators

Measures: `count`, `sum`, `mean`, `median`, `quantile` (`p`), `min`, `max`, `std`,
`distinct_count`, `null_count`, `weighted_mean` (`weight`) and `share` (of the side's
total). Write `sum(x)` or `{measure: quantile, column: x, p: 0.9, name: p90}`.

Sums and means of integers and decimals are exact (overflow-checked 128-bit and
38-digit accumulators); sums and means of floats use the exact binary-float
accumulator; quantiles select order statistics natively and interpolate exactly. Only
`std` is computed in binary64. Missing values (null, NaN, declared markers) are left
out of every measure.

Comparators describe a change: `diff`, `abs_diff`, `rel_diff`, `ratio`, `log_ratio`
and `share_change` (change of the group's share of its side's total), `diff` and
`rel_diff` by default. They operate on exact rationals, so the difference of two large
integers is exact.

### Materiality

A change is material beyond `max(atol, rtol * max(|b|, |c|))`, evaluated exactly.
Tolerances resolve from the most specific setting: a measure's own `rtol`/`atol`/
`exact`, then the comparison's per-column `tolerances`, then the level's `rtol`/`atol`,
then the comparison's `rtol` (1e-6) and `atol` (1e-9). Record values use the
comparison's and per-column tolerances, with the regression engine's exact arithmetic.

Results carry an empty `significance` slot for statistical tests, which a later
release adds as a separate judgement from materiality.

## In a configuration: CMP-001

```yaml
datasets:
  v1: {path: releases/2024/counts.parquet, rename: {SiteID: site}}
  v2: {path: releases/2025/counts.parquet}
rules:
  CMP-001:
    params:
      comparisons:
        - name: counts 2024 vs 2025
          baseline: v1
          candidate: v2
          keys: [site]
          columns: {aadt_k: aadt}
          units: {aadt: {factor: 1000}}
          levels:
            - structure
            - {level: coverage, allow_added: true}
            - records
            - {level: groups, severity: WARNING, group_by: [county], measures: [sum(aadt)]}
```

Each check becomes one result, `CMP-001 [counts 2024 vs 2025: records: values]`, so
severities (per level with `severity`), `engine.require`, evidence limits, redaction
and the HTML/JSON reports work as for every rule. Datasets keep their sources,
secrets, load policies and isolation settings.

## Regression comparisons with renamed columns

The keyed regression API takes the same mapping: `compare_frames`, `compare_outputs`
and `assert_outputs_match` accept `column_map={"old": "new"}`, REG-001 baselines
accept `rename: {old: new}`, and `qavf compare` accepts `--map OLD=NEW` (repeatable).
Keys, ignored columns and tolerances then use the new names.

## Extending

Alignment steps, measures, comparators and levels are registered components; see
[writing plugins](plugins.md#comparison-components).
