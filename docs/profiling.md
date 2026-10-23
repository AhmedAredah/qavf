# Exact draft profiles

`profile_input` accepts the same explicit input descriptions as `validate`: a path,
`File`, supported frame, `Query`, `Loader`, or `Dataset`. It returns the exact row
and column counts plus an inferred schema to review. Observed data can contain
errors; inferred bounds are a starting contract, not evidence of correctness.

```python
from qavf import File, ResourceLimits, profile_input, validate

source = File("counts.parquet")
profile = profile_input(
    source,
    max_categories=20,
    max_category_bytes=1 << 20,
    max_category_value_bytes=64 << 10,
    resources=ResourceLimits(target_memory="256MiB", max_spill="1GiB"),
)
print(profile.rows, profile.columns)
report = validate(source, schema=profile.schema)
```

Native SQL sources use one complete batch stream. The reducer keeps exact counts
and extrema, string length bounds, and at most `max_categories` labels per text
column. After that threshold is exceeded it discards the category set permanently;
it never retains an entire high-cardinality set. Processing does not sample rows.
Retained categories also share `max_category_bytes` (default 1 MiB) across the
whole profile, with `max_category_value_bytes` (default 64 KiB) per value. These
nonnegative integer limits apply to `profile_input` and `profile_frame`; zero
disables category retention. Oversized values are detected before conversion to
Python strings. If either byte limit is exceeded, the column keeps exact string
lengths and missingness instead of an incomplete `isin` list, with a description
explaining the fallback. The category accounting includes Python string storage
and conservative set-entry overhead; it is not a process RSS ceiling.
A row-bounded decoded batch can still be large if individual values are large.

Typed/lazy sources use a coherent, owned observation under the run's temporary
storage budget. The observation is released when profiling finishes or fails.
Caller-owned frames and opaque loaders retain their existing memory ownership
and loading costs. Planning memory remains a target for tracked state, not a
process RSS cap. Profiling explicitly rejects isolated execution; process limits
are supported by `validate` and `compare_outputs`.

NaN/null observations make a column nullable. Positive infinity leaves the upper
bound open; negative infinity leaves the lower bound open. Both set
`allow_infinite`, so mixed finite/infinite data is accepted by the inferred bounds.
Integer and Decimal extrema stay exact. Nanosecond datetime extrema round outwards
to the microsecond resolution supported by configuration datetimes, with a note in
the generated column description. Empty inputs retain their column types; the
inferred default `min_rows: 1` still requires a nonempty production dataset.

`qavf profile` and `qavf init --dataset ...` use this same reducer. Invalid options
are rejected before reading inputs. `qavf.scaffold.profile_frame` remains the
convenience function for caller-owned Polars frames and returns only the schema.

Column regex contracts use Polars/Rust semantics on every input representation.
A conservative shared syntax subset retains SQL pushdown. Engine-specific
patterns use bounded native batches with the canonical Polars predicates; complex
aggregate-dependent scopes retain the existing exact projected fallback.
