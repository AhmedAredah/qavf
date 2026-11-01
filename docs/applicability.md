# Not-applicable values

N/A tokens are opt-in for any declared column and source. There are no
reserved spellings: `N/A`, `-`, and `not measured` are ordinary values unless that
column declares them in `na_values`.

```yaml
datasets:
  observations:
    path: observations.csv
    schema:
      columns:
        distance:
          dtype: float
          na_values: ["N/A"]
          nullable: false
          ge: 0
        duration:
          dtype: float
          na_values: ["not measured"]
          nullable: false
          ge: 0
      na_groups:
        - [distance, duration]
```

The settings apply equally to CSV, Parquet, Arrow, pandas, Polars, and query
inputs; no domain-specific names or file layout are required. A loaded input
must retain the original token strings. An upstream reader that already changed
tokens to null has discarded the distinction and QAVF cannot recover it.

| State | Example | Effect |
|---|---|---|
| Present | `7` | Converted and checked against the column's constraints. |
| Missing | Null or a configured `missing_values` marker | Checked against nullability and missing-fraction limits. |
| Invalid | `unexpected` in a numeric column | Fails declared-type conformance. |
| Not applicable | A configured `na_values` token | Valid; satisfies nullability; excluded from the column's value, uniqueness, and aggregate populations. |

Tokens are case-sensitive. Their surrounding whitespace follows
`trim_whitespace`: `ascii` (the default) removes U+0020 space, `unicode` removes
the documented shared Unicode whitespace set, and `none` preserves all
characters. A null is never an N/A token. Token matching precedes conversion.
Definitions that overlap after normalization are rejected, including missing
markers, duplicate N/A tokens, allowed/forbidden values, boolean words, and
valid spellings of a declared numeric or temporal type. Empty tokens are
rejected. Case-insensitive allowed-value checks also participate in collision
validation.

`max_null_fraction` uses the population after N/A exclusion. For example, one
N/A, one null, and two numbers means a missing fraction of `1/3`, not `1/4`.
An aggregate over an all-N/A population is undefined and is reported as not
assessed; QAVF does not invent a sum or mean of zero. A row-level constraint over
that population is vacuously satisfied.

`INP-009` reports exact `na_count` and `applicable_count` metrics for each
configured column. Each `na_groups` entry requires at least two declared
columns with N/A policies. For every row, zero N/A cells or all N/A cells pass;
any partial N/A group fails. Missing and invalid cells are not N/A and remain
subject to their own checks. Groups can overlap and are assessed independently.
Failure totals remain exact when evidence is bounded. Results identify the
group as `applicability.all_or_none`, with its columns and token parameters;
population results use `applicability.population`. The validity profile enables
this rule whenever N/A tokens are configured.

CSV columns with N/A tokens are retained as text until validation, so tokens
cannot disappear during declared-type conversion. Conflicting `nullstr` or
explicit numeric reader types are rejected with a configuration error. Use
`missing_values` for truly missing cells and `na_values` for valid
non-applicability; they are separate contracts.

Column checks exclude N/A from categorical options and column uniqueness, as
well as numeric checks. Composite-key N/A exemption is reported separately from
null exemption. Arbitrary SQL expressions retain the raw token and can implement
their own explicit applicability condition; N/A tokens are never silently
rewritten to SQL null.
