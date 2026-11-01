# Exact value policies

These policies apply to generic column contracts on both DuckDB and the optional
Polars backend. They add no application-specific tokens, names, units, or keys.
Existing spelling, trimming, case, decimal inference, and naive microsecond
datetime defaults remain unchanged.

## Decimal precision and scale

```yaml
columns:
  amount:
    dtype: decimal
    decimal_precision: 8
    decimal_scale: 2
```

Supply both fields together: precision is 1–38, and scale is 0–precision.
`DECIMAL(8,2)` admits values from `-999999.99` through `999999.99` on an exact
0.01 grid. `1.230` is valid; `1.234` and `1000000` produce type violations.
Values are never rounded to make them conform. The same contract applies to
text and physically typed decimal inputs. An explicit scale avoids the aggregate
pass used to infer a text column's scale.

Without these fields, QAVF retains its existing inferred 38-digit domain.
Unrepresentable outliers no longer determine that domain: a 39-digit integer
cannot cause a valid `1.25` in another row to be rejected. A column still has one
fixed-point storage grid; if independently representable values require
incompatible integer and fractional widths together, values that do not fit
the inferred grid are reported. Declare precision/scale when the domain must
be independent of the file's contents.

## Unicode equality

```yaml
columns:
  label:
    dtype: str
    normalize: [NFC, trim, casefold]
    isin: ["STRASSE", "café"]
    each_allowed_once: true
```

`NFC`, `NFD`, `NFKC`, or `NFKD` selects one Unicode normalization form.
`casefold` uses full Unicode case folding, so `Straße` and `STRASSE` compare
equal. Existing `case` continues to mean lowercase, preserving its earlier
semantics. Choose either `case` or `casefold`, and at most one Unicode form.
Normalization runs in a fixed order: Unicode form, trimming, then case handling,
regardless of list order. It applies equally to data and configured options,
including the distinct domain used by `each_allowed_once`. It does not change
the source values or normalize string identifiers by default.

Custom boolean words use the same Unicode lowercase semantics in both
backends. For example, `İ` matches the configured true word `i` followed by a
combining dot; it does not match plain `i`.

## Number spellings and allowed values

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

## Explicit numeric separators

```yaml
columns:
  amount:
    dtype: decimal
    decimal_precision: 8
    decimal_scale: 2
    number_grammar:
      decimal_separator: ","
      grouping_separator: "."
    convert_options: true
    isin: ["1.234,50"]
```

Both `1.234,50` and `1234,5` mean exactly 1234.5. Grouping is optional, but when
used it requires a first group of 1–3 digits followed by groups of exactly three;
`12.34,5` is invalid. Grouping is allowed only in the integer part, never in a
fraction or exponent. Separators must be distinct single characters, and a
grouping separator cannot be combined with underscore separators. Decimal
separators cannot be whitespace; a grouping separator may be a space, including
an explicitly configured Unicode space. No locale is guessed.

Admission always precedes separator removal. The same policy converts allowed
options and key values. Existing `int_decimal`, `underscores`, `exponent`,
`special_values`, and `trim_whitespace` settings retain their meanings.

## Timezone and timestamp precision

```yaml
columns:
  recorded_at:
    dtype: datetime
    datetime_unit: ns
    timezone: America/New_York
    ambiguous: earliest
    nonexistent: raise
```

`datetime_unit` is `ms`, `us` (the default), or `ns`. Setting a nondefault unit
or a timezone enables the explicit text contract. ISO text can carry up to nine
fractional digits. Nonzero digits below the chosen resolution are violations;
QAVF does not truncate or round them. Thus `.123000` fits milliseconds, but
`.123001` does not. Nanosecond values must fit signed 64-bit epoch nanoseconds,
the common storage range supported by both engines.

`timezone` is an IANA name, such as `UTC` or `America/New_York`. It determines
the meaning of local text without an offset. Explicit `Z` or `±HH:MM` offsets
identify their own instant. With a timezone contract, canonical values are
UTC-naive timestamps at the requested resolution, avoiding DuckDB's
microsecond-only `TIMESTAMPTZ` representation. This representation also defines
key and value equality. The timezone remains in the declared configuration;
it is not guessed from the host environment. Configured aware bounds/options
are converted to UTC; naive bounds/options are interpreted in the declared zone.

During a daylight-saving fold, `ambiguous: earliest` or `latest` chooses the
earlier or later UTC instant. `raise` (default) and `null` reject an ambiguous
local spelling. Nonexistent local times in a DST gap are always rejected:
`nonexistent` accepts `raise` (default) or `null`, and never silently shifts the
time. Here rejection is a reported type violation; it does not abort unrelated
validation. Explicit offsets remove local-time ambiguity.

Without a timezone contract, offset text remains outside the declared text
domain. Without either opt-in setting, legacy naive ISO microsecond parsing
stays unchanged. With an explicit contract, physically zoned timestamps also
normalize to UTC-naive values before comparison, and declared resolution is
checked without rounding. Physically naive timestamps already inhabit that
canonical domain; the timezone interprets naive **text and configuration
literals**, rather than relocalizing an already typed naive timestamp.

The DuckDB backend preserves top-level Arrow nanosecond timezone columns through
an explicit `dtype: datetime` and `timezone` contract before binding them to SQL.
Without that contract, inputs that DuckDB cannot bind losslessly are rejected
with an admission diagnostic. Nested zoned nanosecond values must first be
projected into supported top-level columns. For Parquet containing this physical
type, supply a lossless Arrow table or managed Arrow batches instead of a native
Parquet scan. QAVF does not silently narrow them to microseconds.

Ordinary numeric parsing, standard ASCII boolean words, and legacy datetime
parsing remain native engine expressions. Exact decimal conversion uses native
expressions and bounded aggregate reductions. Advanced timezone parsing and
full Unicode case folding use shared scalar policies only when requested; they
do not materialize complete columns in Python.
