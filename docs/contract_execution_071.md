# Native contract execution guarantees

The native input engine uses independent contracts with shared canonical-value
policies. A bad column condition, expression, key or reference returns its own
`ERROR`; later independent contracts continue. Column predicates share one SQL
count scan when possible. If that fused query fails, the engine evaluates its
predicates separately to identify the failing contract. It does not guess a
successful result. Expression names and conditions remain in machine-readable
parameters, and column evidence includes columns referenced by its condition.

Floating columns enforce `allow_infinite: false` even when `dtype` is inferred.
Nullable missing measurements do not require an associated unit. A missing key
column is a structural failure with an exact count in units of columns.

Native sum, mean and median decisions preserve integer, decimal and stored binary
float precision. Native integer/decimal sums use DuckDB's exact reducer, with a
bounded Python accumulator fallback for fixed-width overflow. Floating sums/means
stream into a bounded exact accumulator. Median selection is ordered in DuckDB,
so sorting can spill to disk; Python receives only the central one or two values.
Nonintegral exact observations use rational strings in metrics (for example
`"1/3"`). Null fractions also compare the exact ratio with the configured bound.
An empty applicable min/max/sum/mean/median/std population is unassessed;
standard deviation also requires at least two usable observations. Empty
`distinct_count` remains zero. The existing empty
missing-fraction convention is retained as zero. Standard deviation remains a
finite rounded statistic, computed with a stable reducer.

These policies are independent of a particular data domain or downstream
application. They need DuckDB and Arrow; they do not import Polars.
