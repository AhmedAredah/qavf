# Strict CSV records and line numbers

CSV reads use DuckDB's strict mode by default. QAVF also fixes the default
`skip` to `0` and `quote` to a double quote (`"`): dialect detection must not
discard malformed early records, mistake a later record for the header, or
ignore a missing closing quote. Single-quote dialects and literal quote text
remain available through explicit `quote="'"` or `quote=""`. To read a
deliberate preamble, specify
`options={"skip": n}` and `acknowledge_lossy=True`. Explicit delimiters, quotes,
escapes, headers, null padding, and `strict_mode=False` retain their native
reader meaning. When the delimiter is omitted, a bounded first-record probe
(up to 64 KiB of decoded text per local file) anchors an unambiguous comma,
semicolon, tab, or pipe outside quoted fields. Ambiguous or mixed-file dialects
retain native inference. For reproducible field-count validation of ambiguous
text or remote input, set the delimiter and header explicitly.

When a local strict CSV read fails, QAVF distinguishes a missing closing quote
from too many or too few fields, and reports both the **CSV row** and the
**physical line** where that record starts. CSV rows are one-based nonempty
logical records after explicitly skipped lines, **including the header**;
comments do not count as records. Physical lines are one-based lines in the
original, uncompressed text, including headers, comments, and skipped lines.
A quoted newline advances the physical line but not the CSV row. These locations
are separate from the zero-based `_row` indexes used for validation evidence.

For example, a header on line 1, a quoted data record spanning lines 2–3, and a
ragged record on line 4 produce `CSV row 3, line 4`. Unterminated quoted fields
also report the final physical line. Structural diagnostics are a failure-only
streaming pass with a bounded preview; clean input incurs no extra full parse
beyond that bounded first-record probe.
Plain files and gzip, bzip2, and zstd streams are supported by the diagnostic
reader. For remote files or dialects it cannot reproduce, QAVF retains the native
reader's diagnostic rather than inventing a location.

Declared value parsing takes precedence over named CSV reader type hints. For
example, declaring `dtype: int` with `types: {k: BIGINT}` validates the original
`2.5` spelling and rejects it; DuckDB cannot round it to `3` before validation.
The same protection applies to named `dtypes`, `column_types`, and `columns`
layouts. Decimal values retain their original precision for validation. This
fixes a bypass of declared type and grammar checks. Explicit reader types on
columns without declared parsing retain their existing behavior, including
precedence over a caller's `text_columns` hint. Nontext positional type lists
conflict with declared parsing and produce a specific configuration error; use
named type hints or positional `VARCHAR` entries instead.
