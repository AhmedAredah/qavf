# Numerical and assessment contracts

QA decisions take priority over approximation. A floating input means its stored
binary value; an integer or Decimal bound remains exact. A bound is never silently
rounded to the nearest floating value before comparison. Exact sum overflow guards
do not depend on the caller's Decimal context. Unsupported aggregate precision or
range produces an explicit unassessed outcome instead of a passing infinity or a
wrapped integer.

Statistics use finite, representable working values. Scaling must preserve the
observations needed by the chosen method. If it erases a small nonzero observation
in a mixed-range series, the group is unassessed; the standalone detector raises a
clear value error. This is a deliberate correctness boundary, not a claim to
support every possible precision or dynamic range with Float64 statistics.

Period-change kernels avoid overflowing intermediate division and integer time
subtraction. Exact integer decreases retain their full magnitude in evidence.
Key matching and expected domains align losslessly or fail explicitly; fractional
expected keys are never truncated into integers. Nested regression retains logical
types recursively, including datetime, duration and text distinctions.

Configured `missing_values` and `blank_as_missing` are applied consistently when
data enters rule execution. Null and NaN are missing. Reconciliation, coverage and
statistical rules therefore cannot treat a configured sentinel as a real balancing
value or entity. Missingness policy is separate from whether a particular check
allows missing observations.

Each configured statistical, temporal or geographic target has its own outcome.
A later invalid target cannot erase an earlier failure. Evidence caps control
retained examples and diagnostics; exact counts and assessment status remain.

Spatial group comparisons use observed group/partition combinations by default
and disclose that scope. Set `require_all_groups_per_partition: true` with explicit
groups when every group must exist in every observed partition. A coverage contract
is needed to require partitions that are completely absent from the input.

Spatial coverage currently requires temporal identifiers representable at microsecond
precision. Non-aligned Datetime(ns), Duration(ns) or Time identifiers are unassessed
before Python conversion can merge identities; use explicit physical integer keys
when finer-resolution identities are required.

## Numeric text grammar and option equality (0.7)

Declared numeric text is admitted by a shared grammar before conversion. The
default integer grammar accepts `07`, `7`, and `+7` as the same integer, but
continues to reject `7.0` to preserve existing type-checking contracts. Enable
integral decimal notation explicitly when it is valid for your input:

```yaml
columns:
  category:
    dtype: int
    number_grammar:
      int_decimal: true
      underscores: false
    trim_whitespace: ascii
    isin: ["07", "+8", "9.0"]
    convert_options: true
    each_allowed_once: true
```

Here `7`, `07`, `+7`, and `7.0` identify one value, including in keys. An integer
never converts through floating point: `9007199254740993.0` remains exact,
fractions such as `7.0000000000000000001` are invalid, and out-of-range values
are invalid. With `int_decimal: true`, a decimal point may be followed only by
zeros (or no digits); scientific integer notation remains invalid.

| Setting | Default | Meaning |
|---|---|---|
| `number_grammar.int_decimal` | `false` | Admit integral decimal notation for integers. |
| `number_grammar.underscores` | `false` | Admit one underscore between digits, such as `1_000`; doubled, leading, and trailing underscores remain invalid. |
| `number_grammar.exponent` | `true` | Admit scientific notation for floats. Exact decimal columns retain decimal notation. |
| `number_grammar.special_values` | `true` | Admit float `nan`, `inf`, and `infinity` spellings, case insensitively. NaN still means missing; `allow_infinite` still governs infinities. |
| `trim_whitespace` | `ascii` | Strip surrounding U+0020 spaces before type parsing. `unicode` strips the explicitly shared Unicode whitespace set; `none` preserves all characters. |
| `convert_options` | `false` | Parse string `isin`/`notin` options with the declared numeric grammar. Without it, wrong-type options retain the existing compatibility diagnostic. |

Whitespace and grammar settings govern text interpretation, not already typed
numeric arrays. Preserve text at ingestion when the original spelling matters.
CSV columns declared as decimals retain their raw spelling until QAVF resolves
an exact column scale. A named reader hint such as `DECIMAL(6,2)` cannot round
`1.234` to `1.23` before validation. Positional numeric reader hints that conflict
with declared validation raise a specific configuration error; use named hints
or `VARCHAR` instead.
Missing markers retain their existing exact comparison rule; they do not become
valid numbers merely because they appear in an allowed list.

`each_allowed_once` is a built-in domain contract requiring `isin`. It compiles
to membership, uniqueness and an exact distinct-count bound, so it shares the
existing SQL reductions and bounded evidence paths. Repeated configured options
that convert to one value describe one domain member. Missing and N/A cells are
outside this domain and follow their separate policies. A missing domain member,
a repeated member, or an unexpected member fails the contract. `where` scopes
the entire contract, including coverage.

Case-insensitive collisions already use the built-in `unique` constraint:

```yaml
columns:
  label:
    dtype: str
    unique: true
    normalize: [case]
```

Combine `normalize: [case]`, `isin`, and `each_allowed_once` to require every
distinct normalized text option exactly once. Normalization affects comparison;
evidence retains the observed text.
