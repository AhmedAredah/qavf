# Reducing full-data loading

QAVF reduces data crossing into Python and retained decoded frames, while preserving
complete assessment, exact numerical decisions, bounded evidence, and source
identity checks. Ordinary `validate` calls select these paths automatically.

## Execution choices

| Work | Reduced path | Remaining state or fallback |
| --- | --- | --- |
| GEO coordinate checks | Count every problem and retain bounded original-row evidence in one source evaluation. With zero row evidence, collect coordinate dependencies only. | Exact coordinate predicates still inspect every row. Backend state and selected evidence cells consume memory. |
| Input `unique_rows` | Source-side duplicate count and bounded evidence, sharing a row-summary boundary. | Full-row uniqueness semantically requires every column. Combined key/whole-row checks share one snapshot for nonrepeatable inputs, preserving a coherent observation. |
| STA column outliers | Read the checked value and group keys once; keep raw values and a row position from that same observation for evidence. | Exact quantiles and statistical kernels still retain the selected series/groups. Unrelated input columns are no longer included in evidence. |
| Temporal, spatial and reconciliation aggregates | Reduce supported count/min/max and guarded integer/Decimal sums before collection for verified repeatable file inputs. Row-count-only reconciliation reduces even an owned spilled frame. | Reduced group tables remain in memory. Exact floating/grouped means, weighted, order-dependent or unsupported typed operations use a selected-column fallback. Arbitrary queries and supplied lazy plans use a coherent projected observation instead of repeated evaluation. |
| Scalar float sum, mean and sample deviation | Consume one selected native column in bounded batches; exact binary accumulation and the standard-library exact-ratio deviation algorithm preserve numerical decisions. | Composite contracts that need complete observations, median and unsupported reader types retain an exact projected fallback. |
| Keyed regression | Plan from schemas, validate keys and consume a native joined result in bounded batches for supported repeatable file sources. | The native join, sorting and key grouping use backend memory/spill. Native-safe nested values share that path. Other portable typed inputs use coherent observations and a disk-indexed typed stream; opaque Object cells retain an explicit projected fallback. |
| Cached or spilled input | `DataContext.project` reads requested columns; scoped `reduce` keeps the owned IPC reader alive through reduction. | Whole-frame `get`, eager loaders and explicit snapshots retain their full-load semantics. |

The STA evidence position is zero-based within the single projected evaluation.
It is not a persistent database or file identifier. Its internal field name is
collision-safe. Group keys, raw checked values and statistical diagnostics remain
available; adding an unrelated ID to `group_by` would change the statistical test.

## Resource-aware regression

```python
from qavf import ResourceLimits
from qavf.regression import compare_outputs

result = compare_outputs(
    "baseline.parquet",
    "candidate.parquet",
    keys=["site_id", "year"],
    resources=ResourceLimits(
        target_memory="1GiB", max_spill="2GiB", evidence_rows=20,
        evidence_bytes="64KiB",
    ),
)
print(result.equal, result.execution, result.fallback_reason)
```

`execution` distinguishes `native_stream`, `typed_stream`, `projected_fallback` and `in_memory`.
`fallback_reason` explains the capability boundary. `assert_outputs_match` accepts
the same resource argument. `REG-001` uses its parent run's actual resource owner,
including baseline source policy and manifest verification. `compare_frames`
remains an explicitly in-memory API. See [exact typed regression](regression_streaming.md)
for the indexed typed path, baseline writer and resource boundaries.

Native comparison replaces equal matched non-key `String`/`Binary` payloads with
typed empty values before ordering and transfer. Actual changes, nulls, keys and
unmatched-row values remain intact; comparison results and evidence are unchanged.
Wide changed values and the backend hash join still consume memory.

## Modular ownership

| Module | Responsibility |
| --- | --- |
| `engine/projection.py`, `engine/owned_ipc.py` | Selected reads and reductions with owned spill lifetimes. |
| `engine/relations.py` | Preserve native file-reader projection and filtering without changing other sources' binding semantics. |
| `engine/sql_read.py`, `engine/source_read.py` | Scoped native batches and ad-hoc reads, cancellation, complete consumption, retry reset and identity verification. |
| `rules/_row_summary.py`, `rules/_coordinate_sql.py` | Coherent count/evidence reduction and exact coordinate expressions. |
| `rules/_grouped.py`, `rules/_spatial_aggregation.py` | Selected normalized relations and shared safe grouped reductions. |
| `_float_accumulator.py`, `numeric.py`, `rules/_column_stream.py` | Exact reusable numerical state and scalar streaming dispatch. |
| `regression/_plan.py`, `_inputs.py`, `_streaming.py`, `_comparison.py` | Schema planning, policy-preserving dispatch, native execution and shared exact comparison state. |

Consumers receive results only after successful full consumption and verification.
Each retry constructs fresh reduction state. Partial streams cannot publish a
successful result. Connections, readers and temporary storage remain owned until
work finishes, including cancellation and exceptional exit. File identity checks
detect drift; they cannot make an external writer atomic.

Built-in local file plans retain native projection and filter pushdown. Their
generated reader SQL uses escaped literal paths from the verified file manifest;
arbitrary query parameters are never interpolated. Query, plugin, subclass and
remote sources retain their existing native binding behavior. Nonempty bindings
can materialize the full source before downstream projection, so narrower Python
transfer does not necessarily imply narrower backend decoding. Empty bindings
may remain lazy depending on DuckDB version; arbitrary queries do not acquire a
snapshot guarantee and may evaluate again across separate reads.

Batch row counts bound Python transfer rows, not bytes: one large cell can be
expensive. Native memory settings and decoded-size checks are planning targets,
not process RSS caps. Projected IPC read-byte telemetry is explicitly unmeasured;
it must not be interpreted as zero physical I/O. Complete-source row budgets are
checked before returning a small reduction. No path samples away violations.

## Evidence and reproduction

The loading tests (`tests/test_loading_*.py`) check each reduced path against an
independent result. `scripts/benchmark_performance.py` measures them with
deterministic chunked fixtures, separate processes, warmups kept apart from measured
repetitions, and JSON records of phase times, sampled RSS, owned disk, correctness
oracles and cleanup. OS page cache and shared-host contention are uncontrolled, and
sampled peaks can miss short spikes. See [resource contracts](resources.md) and
[benchmark methodology](performance_benchmarks.md).
