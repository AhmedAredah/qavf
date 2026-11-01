# QAVF 0.7: configuration and compatibility

Version 0.7 keeps existing Python calls, schema declarations and result fields.
New semantics are explicit configuration, except fixes for inconsistent typed-key
comparison, whitespace handling and malformed strict CSV reads. The changes apply
to general tabular validation; no MOVES-specific columns, tokens or rules are built in.

## Installation and execution

```bash
pip install qavf                 # DuckDB/Arrow declared-contract validation
pip install 'qavf[polars]'       # full execution backend and advanced CLI commands
pip install 'qavf[all]'          # every optional capability
```

`validate(..., backend="auto")` is the default. It retains the existing execution
backend if Polars is installed and uses the native DuckDB/Arrow backend otherwise.
`backend="duckdb"` selects the native backend even when Polars is installed;
`backend="polars"` explicitly requires the optional backend. This selector is a
Python API keyword and, from 0.7.1, the CLI `run`, `schema` and `list-rules` commands
accept the equivalent `--backend` option. Help/version and native validation/schema
commands no longer require Polars. DuckDB 1.3.1 remains supported, including the downstream stack
DuckDB 1.5.6, pandas 2.1.3 and PyArrow 16.

The native backend handles declared schemas, column constraints, keys, foreign
keys, row expressions, SQL assertions and N/A groups over CSV, Parquet, JSON,
Arrow, pandas, loaders and queries. Advanced statistical, temporal, spatial,
regression and plugin execution uses `qavf[polars]`. A configured capability the
native backend cannot execute raises a specific dependency/capability error.
It does not treat omitted work as a successful validation. The full backend
retains the existing plugins and advanced APIs.

A native report can have a different number of separately emitted check results;
use the structured check contracts and exact counts instead of relying on result
list positions. Existing automatic-backend behavior is retained when Polars is
installed.

## Converted keys and configurable evidence

```python
schema = {
    "columns": {
        "item": {"dtype": "int", "number_grammar": {"int_decimal": True}},
        "revision": {"dtype": "int", "nullable": True},
    },
    "unique_keys": [{"columns": ["item", "revision"], "nulls": "equal"}],
    "duplicate_rows_per_key": "all",
}
```

Keys compare the declared values: `07`, `7` and `+7` represent the same integer.
`7.0` joins that group only with `number_grammar.int_decimal: true`; fractional
values are never rounded into integer keys. The same contract applies when the
reader or an Arrow/pandas/Polars column supplies text. Declared string identifiers
retain their string semantics, including leading zeros.

Alternate-key null policies are:

| `nulls` | Behavior |
|---|---|
| `exempt` | Rows with any missing key component are outside uniqueness. This is the existing list-form behavior. |
| `equal` | Corresponding missing components compare equal; `(1, null)` duplicates `(1, null)`, but not `(2, null)`. |
| `forbid` | Missing components are failures; complete keys are also checked for duplicates. |

Primary keys continue to forbid missing components. Configured N/A tokens are a
separate state: a key containing one is outside key validation and is counted in
`exempt_na_rows`, rather than reported as missing.

`duplicate_rows_per_key` is a positive integer or `"all"`; its default remains 10.
The number of sampled key groups remains controlled by the existing evidence
limits. All positions are subject to configured evidence/resource limits, with
explicit completeness/truncation metadata. Key counts, involved rows and surplus
rows remain exact regardless of the evidence sample.

## Not-applicable values

```python
schema = {
    "columns": {
        "amount": {"dtype": "float", "na_values": ["N/A"]},
        "quantity": {"dtype": "int", "na_values": ["N/A"]},
    },
    "na_groups": [["amount", "quantity"]],
}
```

N/A tokens are valid, distinct from missing and invalid text. Type, nullability and
ordinary value checks exclude these rows from applicable populations and report
N/A counts. INP-009 passes rows with zero or all group members N/A; partial N/A
fails. Empty/missing values do not satisfy an N/A token. Token matching uses the
column's whitespace policy; no tokens are recognized implicitly. Overlap with
missing tokens or reader null tokens is rejected. Numeric statistics with no
applicable values remain unassessable, not a manufactured success.
See [applicability](applicability.md).

## Numbers, allowed values and text collisions

```python
schema = {
    "columns": {
        "code": {
            "dtype": "int",
            "number_grammar": {
                "int_decimal": True,
                "underscores": False,
                "exponent": False,
                "special_values": False,
            },
            "trim_whitespace": "ascii",
            "isin": ["07", "+8", "9.0"],
            "convert_options": True,
            "each_allowed_once": True,
        }
    }
}
```

`convert_options` applies the declared numeric conversion to configured options.
It does not coerce string identifiers. Invalid option spellings or precision loss
produce a configuration/check error. `each_allowed_once` combines allowed-domain
membership, uniqueness and complete domain coverage. A repeated value cannot
compensate for a missing domain member.

`trim_whitespace` is `ascii` (ASCII space only), `unicode`, or `none`. Number grammar
can control separators, exponents, special floating values and exact integral
decimal notation. NaN/missing and `allow_infinite` policies continue to apply;
permitting a spelling does not automatically permit it as a valid finite value.
Existing `normalize: [case]` with `unique: true` is the built-in case-insensitive
collision rule; add `trim` when that is part of the intended text equality.

## Machine-readable identity

Every result contains `rule_id`, `check_id`, `scope`, `columns`, `parameters`, and
`checks`. The latter holds constituent contracts when multiple passing checks are
summarized together. `datasets` supplies dataset scope. Display labels remain for
humans and existing consumers; clients need not parse them. Older/plugin result
constructors receive a truthful rule-level identity unless a producer provides a
more specific check contract. Identity is retained when optional evidence is
removed to meet a budget.

## Resources and strict CSV

Automatic planning uses input metadata, declared key widths and available memory
and temporary-disk capacity. It does not add a data scan merely to estimate a
budget. Large key operations reduce concurrency. Estimates are diagnostics, not
RSS guarantees; compressed/remote data and unknown cardinalities remain uncertain.
Explicit memory/spill caps are never silently raised. See [resource limits](resources.md).

Strict CSV diagnostics distinguish logical records from physical lines and identify
unterminated quotes and incorrect field counts. Specify dialect options when a
file is ambiguous; see [strict CSV](strict_csv.md). Successful validation does not
invoke a second full diagnostic parse.

## Architecture and validation

Input adaptation, canonical conversion, applicability, key policies, resource
planning and result construction have separate responsibilities. Shared policy
objects compile backend expressions; backend-specific code handles execution and
ownership. Existing extension classes remain available through the optional full
backend. This keeps OOP boundaries useful without creating an object for each row
or copying a full dataset into Python.

The release includes cross-backend regression tests, an actual installation with
Polars absent, DuckDB-floor tests, and a correctness-checking benchmark script.
See [benchmark measurements](benchmark_070.md) for workload and measurement limits.
