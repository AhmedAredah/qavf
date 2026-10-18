# Repeated scans and hashing

QAVF reduces repeated scans and hashing without adding a public input wrapper,
configuration option, runtime dependency, or persistent data cache. Existing file,
query, frame, and lazy-frame inputs continue through the same validation API.

## Shared reductions

`rules/_column_batch.py` owns dependency normalization, bounded execution batches,
and scalar result dispatch. `rules/_column_plan.py` still owns each column contract's
meaning, exact fallback choice, and bounded evidence. Compatible contracts share a
reduction even when another contract needs different execution. A batch contains
at most 64 checks. Invalid compiled expressions are isolated using owned, empty,
typed data; this validation does not read the user's source. Arbitrary SQL scopes
and reductions with data-dependent exact-arithmetic failures remain independent.
An actual execution failure does not split and replay the source. Configured
source retry policy remains the only retry authority.

`rules/_key_summary.py` owns exact key counts, row-frequency totals, and bounded
samples for multiple predicates over the same grouped relation. Primary and unique
key totals, null checks, and duplicates share one grouped evaluation. Batch identity
shares its total, missing, and identifier summaries. Foreign keys share child
totals, null policies, valid-key totals, and child duplicates; the parent and
anti-join still require their own evaluations. Each batch accepts at most 16
selections. Native SQL uses a materialized CTE; typed lazy execution uses one
projection of reductions. Only counts and bounded samples cross into Python.

The engine still has to group keys, maintain distinct sets, and sometimes sort or
project values for exact statistics. Those operations can scale with input size.
This is bounded Python evidence collection, not an assertion of constant total
engine memory. DuckDB may spill intermediate data according to configured resource
limits. No complete key table or observation array is cached across rules.

## File identity boundaries

`sources/identity.py` owns streaming hash accounting and thread-local operation
scopes. `FileSource` owns manifest and content checks. `sources/execution.py` owns
retry/cancellation policy and enters each participant's verification scope.

With `report.hash_inputs: true`, local file content identity is established before
the first source read. Each outer operation checks established content on entry
and on exit, including a failed read. Nested helpers still check manifest metadata
but reuse their enclosing operation's content verification. New participants,
separate threads, retries, direct verification calls, and later operations have
their own boundaries. Known lazy schema metadata is reused within the run.

File size and modification time alone never authorize digest reuse across
operations. A persistent content change is detected even when size and modification
time are restored. This policy checks boundaries; it does not create an immutable
filesystem snapshot or promise detection of a transient change fully restored
between observations. Callers needing snapshot isolation must supply stable inputs.

Per-dataset `notes.identity_work` records cumulative counters:

| Counter | Meaning |
|---|---|
| `hash_passes` | Full-file hash attempts begun, including failed attempts |
| `hash_passes_completed` | Hash attempts that read successfully to completion |
| `bytes_hashed` | Bytes actually streamed into the digest, including partial attempts |
| `coalesced_identity_checks` | Nested checks that reused the enclosing content boundary |

Counters are per source instance, not a global cache. Hash buffers are fixed at
1 MiB; no additional file copy or persisted digest database is created. Content
verification still adds complete reads at independent operation boundaries, so
hashing cost can remain substantial for large files or many independent rules.

## Validation and measurements

`tests/test_scan_*.py` covers shared summary correctness, bounded evidence, exact
keys, normalization, error isolation, hash accounting, retries, and source changes.
The package's existing correctness and assurance suites remain applicable.

`scripts/benchmark_scan_work.py` compares both source trees in separate processes
with the same interpreter and dependencies. It records executed reductions, result
equivalence, wall time, and process peak RSS. Hash probes separately count full
file passes and bytes. Reduction counts are deterministic for these cases; timings
and RSS are observations on the recorded host, not universal improvement claims or
release thresholds.
