# Key equality and duplicate evidence

QAVF 0.7 compares key components using their declared logical types on every
input adapter. With `dtype: int`, `07`, `7` and `+7` identify the same key. Set
`number_grammar: {int_decimal: true}` to also admit `7.0`; the default retains
0.6's integer-literal grammar. Fractional values are never rounded to keys.
`dtype: str` preserves identifiers such as `07` and `7` as distinct values.
Decimal conversion is exact and does not pass through binary floating point.
Invalid text becomes null for key checking, consistently with column value
checks; the type check separately reports the original invalid value.

## Nullable composite keys

Primary keys keep their existing contract: every component is required and
non-null. Alternate keys can include nullable columns:

```yaml
schema:
  columns:
    site: {dtype: int}
    category: {dtype: str, nullable: true}
  unique_keys:
    - columns: [site, category]
      nulls: equal
```

| `nulls` | Rule |
| --- | --- |
| `exempt` | Rows containing any null component are excluded and counted in `rows_exempt_null_unique_keys`. |
| `equal` | Null equals null in the same component. `(1, null)` duplicates `(1, null)`, but differs from `(2, null)` and `(1, "x")`. |
| `forbid` | Any null component fails the key contract; complete keys must still be unique. |

The existing `unique_keys: [[site, category]]` form is unchanged and means
`nulls: exempt`. Declared missing-value markers participate as nulls after
conversion. Explicit `na_values` are valid, inapplicable values: a row with any
N/A key component is exempt under every null policy, including primary keys.
These exemptions are counted separately in `exempt_na_rows`, never as missing
or invalid keys. Configure N/A tokens only when this exemption is intended.

## Duplicate row positions

```yaml
schema:
  primary_key: [site, category]
  duplicate_rows_per_key: all  # Or any positive integer; default: 10.
```

`engine.max_affected_records` (or `ResourceLimits.evidence_rows`) still limits
the number of sampled duplicate **keys**. `duplicate_rows_per_key` controls
the zero-based source positions listed for each sampled key. All key counts,
rows involved, and surplus row counts remain exact even when samples or
positions are limited. Positions retain their original numbering when N/A
rows are excluded.

`all` requests every position for each sampled key, subject to the existing
`engine.max_evidence_bytes` limit. Position lists are bounded before Python
materialization; very large duplicate groups cannot bypass that limit.
`duplicate_positions_available`, `duplicate_positions_complete`, and
`duplicate_positions_truncated` describe the returned position evidence.
A failed optional lookup preserves the exact duplicate failure and counts,
and marks positions unavailable. Completeness refers to the sampled keys;
`affected_count` still describes all duplicate keys.

The native lookup joins only the selected keys back to the source in one
query. It does not collect row lists for every distinct key or rescan once
per sampled key. The optional dataframe path uses the same conversion policy
and a semi-join over the sample.

Results include `check_id` (`key.primary`, `key.unique`, or their `.nulls`
variants), `columns`, and `parameters` with `nulls` and `rows_per_key`.
Passing summaries expose constituent identities through `checks`.

## Conditional composite keys

A structured alternate key can set `where` and `name`. Its condition is evaluated
on converted values; only rows for which it is true participate. False and NULL
conditions do not participate. The existing `nulls` policy applies within that
population. Independent conditions on the same columns are allowed.

```yaml
schema:
  unique_keys:
    - columns: [site, category]
      where: "active = TRUE"
      nulls: equal
      name: active category identity
```

## Grouped occurrence and domain coverage

`grouped_values` checks an allowed domain separately for every group, without
custom SQL rules. `min_occurrences` and `max_occurrences` default to one, so each
allowed value must occur exactly once. Set `max_occurrences: null` for no upper
bound, or `min_occurrences: 0` for optional domain members. Unexpected nonmissing
values fail unless `allow_other_values: true`.

```yaml
schema:
  columns:
    category: {dtype: int}
  grouped_values:
    - column: category
      group_by: [site, year]
      allowed_values: [1, 2, 3]
      where: "active = TRUE"
      expected_groups: [[north, 2025], [south, 2025]]
      name: active category coverage
```

Observed groups are determined **before** `where` filtering. A group with no
selected rows still fails missing required members. `expected_groups` adds groups
that are entirely absent from the dataset. Every expected-group tuple must match
`group_by` in order and width. NULL group components compare equal. A group row
with a declared N/A component is exempt; missing/N/A domain values do not satisfy
an allowed member. Declared numeric conversion is also applied to allowed values
and expected group values. Canonically equal allowed values are one domain member. A literal that matches a
declared N/A token is rejected as a contradictory configuration.
Evidence identifies each violating group/value pair, its occurrence count and
reason; exact totals do not depend on the evidence cap.

## Exact key admission

Undeclared floating NaN is missing, including primary/alternate keys and foreign
keys. Foreign keys compare canonical values only after proving a lossless common
type. Strings and numbers are not implicitly equated without a declared logical
conversion. Cross integer/float keys require finite integral values with magnitude
at most `2**53`; otherwise the contract is reported as unassessed (`ERROR`). The
engine never silently rounds a key to make a join succeed. N/A keys on either side
are exempt from reference matching and cardinality checks.
