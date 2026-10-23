# Coherent projected observations

Rules that compare several passes over a dataset must use the same observation of
that dataset. A supplied lazy plan, query, plugin or remote source may return
different values on each evaluation. A cached schema does not make its rows
repeatable.

`DataContext.observe(name, columns=None)` owns a stable selected projection for a
scope. The source is read and checked before the observation is yielded. Existing
owned frames use their immutable buffers; existing IPC spills use reference-counted
leases without loading a complete projection or copying the spill. Other lazy and
native inputs retain selected chunks in available shared cache memory and spill to
owned IPC when that admission ceiling is exhausted. The source is never replayed
to switch storage. Incremental lazy reading uses `collect_batches` where provided
by the installed Polars version; older supported versions keep exact IPC staging.
Native SQL batches use memory-first admission on all supported versions.

```python
with ctx.observe("events", ["id", "value"]) as observation:
    total = observation.reduce(lambda data: data.select(pl.sum("value"))).item()
    for batch in observation.batches(batch_rows=65_536):
        consume(batch)
```

The sum and every batch refer to the same source evaluation. `schema()` returns the
observed schema. `scan()` returns a lazy plan whose file lifetime ends with the
scope; collect it inside the scope through `observation.collect(plan)`. Returned
frames and batches are detached from owned IPC mappings and may outlive the scope.
Do not retain an uncollected plan or iterator after leaving it.

`ctx.projected_observations({"left": ["id"], "right": ["id"]})` opens several
observations and returns a mapping with `collect(plan)` for combined plans. Use an
`ExitStack` with individual `ctx.observe()` scopes when each dataset should report
its own failure independently. A `SourceReadView` inside `ctx.read_source()` offers
`observe(columns)` for the same bound ad-hoc source and provenance.

A derived observation evaluates a sort, grouping or other transformation once and
allows several consumers to reuse it. A transformation returning its input plan
unchanged borrows the original observation without copying its payload:

```python
with ctx.observe("events", ["id", "value"]) as source:
    with source.derived(lambda data: data.sort("id")) as ordered:
        for batch in ordered.batches(batch_rows=16_384):
            consume(batch)
```

`batches(build=...)` is a convenience for one derived observation followed by batch
iteration. It uses the same typed IPC path on the minimum and current supported
Polars versions, including Int128, nested Int128, Enum and nanosecond timestamps.
An IPC scan carries projection and slice pushdown; this is not a guarantee about
physical filesystem reads, native working allocations or process RSS.

## Resource and lifecycle contracts

In-memory observations and caches share a retained-memory admission owner.
`retained_total_bytes` includes both; `retained_observation_bytes` identifies the
observation portion. Immutable frame and IPC leases remain charged until their last
owner closes, even when a dataset is released while a scoped observation is active.
An observation cannot expand beyond the available retention ceiling by adding more
chunks: it switches to disk and releases the staged memory after writing it.

Frame spills, observations and external indexes share `RunResources.spill_budget`.
They use the existing owned-storage half of `spill_max_bytes`; native DuckDB
connections retain the other half of that envelope. The allocator reserves bytes
before owned writes. A full budget raises `ResourceLimitError` and removes partial
observation files. Failed identity checks and retried attempts cannot publish a
partial observation. A leased frame spill stays charged until its last owner
closes, even if the cache entry is released or promoted meanwhile.

`owned_spill_bytes`, `peak_owned_spill_bytes` and `owned_spill_budget_bytes` describe
this shared storage. Existing frame-spill counters still describe frame ownership;
`frame_spill_reduce_count` includes scoped lazy observation leases and does not
claim an exact count of physical scans. Native observations currently retain one
small file descriptor record per result batch, so their Python metadata grows with
batch count. Their payloads and metadata files remain subject to the byte budget.

Where incremental IPC collection is supported, `batches()` reads a stored plan once
instead of starting a slice query for each batch. Older supported Polars versions
retain projected slice replay. The `batches()` size bounds decoded output rows per
yielded frame. A single large
value can exceed a memory target; blocking sorts, groups, joins, Python callbacks
and other upstream work can need more memory than the output batch. A row budget
caps a returned LazyFrame to `row_budget + 1` before collection or staging, including
transactional source plugins. It does not cap upstream rows scanned or callback
allocations. Derived IPC staging may temporarily coexist with its input observation
and therefore need additional spill capacity.

For a single pass over a SQL-compatible source, `consume_sql()` already offers a
scoped native batch consumer without mandatory observation staging. Use observations
when coherent reuse or exact typed traversal is required, rather than adding a
snapshot to every reduction.

## Reuse across validated contracts

`ctx.observation_plan(contracts)` accepts a sequence of dataset-to-column mappings
after the rule has validated each contract's schema and normalization settings.
It unions selected columns per input and opens each input lazily, only when a
contract asks for it. Each contract receives a non-owning `ObservationView` with
only its own columns; batches retain projection pushdown without copying IPC files.

```python
with ctx.observation_plan([{"events": ["x"]}, {"events": ["y"]}]) as plan:
    with plan.contract(0) as observed:
        first = summarize(observed["events"])
    with plan.contract(1) as observed:
        second = summarize(observed["events"])
```

Both summaries share one input evaluation. An input closes after its final
scheduled contract finishes, including when a contract fails. Plan exit closes
all remaining inputs; unrequested inputs are never read. Ordinary source failures
are cached by exception category and message so later contracts do not repeat a
failed read. The cache retains no source exception traceback or arbitrary payload;
interruptions deriving directly from `BaseException` propagate. Contract indices
may be used once and the plan belongs to one sequential rule invocation. Invalid
contracts must be excluded before planning so missing columns or invalid markers
cannot contaminate a valid sibling's shared projection.

## Access and capability policy

Serialized source operations acquire the operation guard before keyed pandas or
memo locks. Keyed waits check invocation cancellation. This keeps nested native
consumers and concurrent context accessors from acquiring those locks in opposing
orders. With `concurrent_loads=False`, a memo callback also runs under that guard
because it may load data itself.

Reader capability resolution is cached per context. Once resolved, backend
classification does not reread file contents. Actual reads retain their before/after
identity checks and final input verification; capability caching never acts as a
content-digest cache. Unlimited materialization checks return without serializing
nested columns merely to calculate an unused size.
