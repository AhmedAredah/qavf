# Accepting reviewed baselines

`accept_baseline` accepts one normalized input: a path, `File`, supported frame or
lazy frame, `Query`, `Loader`, or `Dataset`. The same descriptors and loading
policies work with profiling, validation and comparison. A `Dataset` can rename
columns or carry a row budget; a loader is invoked through the common input
boundary. Acceptance records what was approved, not proof that the data is correct.

```python
from qavf import File, ResourceLimits
from qavf.regression import accept_baseline

manifest = accept_baseline(
    File("reviewed.parquet"),
    "baselines/counts.parquet",
    keys="record_id",
    reason="Reviewed against the source survey",
    resources=ResourceLimits(target_memory="256MiB", max_spill="2GiB"),
)
```

Acceptance observes the input once, rejects missing or duplicate keys, writes rows
in key order, and compares the restored values exactly before publishing the data
and manifest together. CSV, TSV, Parquet, JSON arrays and NDJSON are supported;
text formats also support gzip. A format that changes accepted values is rejected.
Use Parquet for typed or nested values. The manifest records the source column
types, exact row count, file digest, keys and acceptance reason.

The writer uses lazy sinks or bounded JSON batches. Round-trip validation uses the
shared disk index and bounded observations; it does not reload both full tables
into memory. Input snapshots, the staged payload and the comparison index share
the run-owned spill budget. Sorting and decoding retain their underlying engine
costs; `target_memory` is a planning target, not a hard process RSS ceiling.
Caller-owned frames and opaque loaders retain their own loading costs.

Published outputs, durable recovery metadata and backup copies are not temporary
spill. Replacing a baseline normally hard-links the old files; when hard links
are unavailable, recovery uses a backup copy and therefore needs additional free
disk space. Existing locking, journal recovery and rollback remain in effect.
A spill failure before publication preserves the previously accepted pair and
cleans the transaction and run-owned temporary files.

Set `replace=True` to replace an accepted baseline. The CLI equivalent is
`qavf baseline reviewed.parquet baselines/counts.parquet -k record_id --force`;
it shows the changes before acceptance. CLI baseline, profile and init commands
use the common source boundary rather than an eager table read.

Baseline acceptance explicitly rejects isolated execution and process execution
limits. Those controls are currently supported by `validate` and
`compare_outputs`; they are never silently ignored by acceptance or profiling.
