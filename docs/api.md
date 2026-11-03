# Python API

Use `validate` to apply the same explicit contract to a path, dataframe, lazy
frame or query. It returns a `QAReport`; writing reports is a separate action.

```python
from qavf import validate

schema = {
    "primary_key": ["site_id", "year"],
    "columns": {"aadt": {"dtype": "int", "ge": 0}},
}
report = validate("counts.parquet", schema=schema)
report = validate(counts_dataframe, schema=schema)
report = validate(counts_lazy_frame, schema=schema)
```

Polars, pandas and Arrow tables are accepted. A Polars `LazyFrame` is preserved
during input normalization; the execution plan decides whether a particular
check can scan it or needs a materialized frame. Lazy input does not promise
that every statistical or plugin check can run in bounded memory.
Tabular Arrow C-stream objects are also accepted and consumed once during
adaptation. Their schema must describe a table; scalar-array streams are rejected.
Arrow/pandas conversion explicitly preserves chunks where supported instead of
requesting a contiguous rechunk. Some type conversions still allocate buffers.

Provide `schema=`, `config=`, or an explicit contract on a `Dataset` descriptor.
QAVF does not derive acceptance limits from the same data being checked. An empty
schema does not add business rules.

## File options without a configuration tree

```python
from qavf import File, validate

report = validate(
    File("counts.csv", options={"delim": ";", "header": True}),
    schema={"columns": {"count": {"dtype": "int", "ge": 0}}},
)
```

Use a plain path for ordinary files. `File` adds explicit `format`, reader
`options`, projected `columns`, `min_files` for local globs, and
`acknowledge_lossy`. These use the same validation as file-source configuration;
options that can omit rows still require explicit acknowledgement. A format can
be specified for an extensionless file. File paths retain their exact spelling,
including surrounding whitespace; QAVF never trims a path to choose another file.
An unsupported suffix requires an explicit format instead of a guessed path.
For local Parquet, reader selection considers the projected columns while still
checking the complete file set for schema conflicts. Discarding a timestamp or
nested column can therefore retain native SQL execution. A requested column
whose physical name collides case-insensitively still requires the typed reader.

Native file projections require the exact decoded column spelling. CSV/JSON
readers can disambiguate case-colliding physical names (for example `x` and `X`);
an unavailable requested spelling is rejected rather than selecting another
column. Supply a typed Polars/pandas/Arrow frame when those original names must
remain distinct. Local Parquet selects the typed path automatically for types
that native SQL cannot preserve, including nested Null and version-available
UInt128/Float16 columns.

When replacing a configured file, a plain path keeps that file's reader options.
An explicit `File` supplies a complete new reader specification. In both cases,
configured QA contracts and common loading safeguards remain in effect.

## Inline contracts for named datasets

```python
from qavf import Dataset, File, validate

report = validate(
    {
        "counts": Dataset(
            File("counts.csv", options={"delim": ";"}),
            schema={"columns": {"site_id": {"dtype": "str"}, "count": {"ge": 0}}},
            foreign_keys=[{"columns": ["site_id"], "ref_dataset": "sites"}],
            load_policy={"row_budget": 1_000_000},
        ),
        "sites": Dataset("sites.parquet", schema={"primary_key": ["site_id"]}),
    }
)
```

`Dataset` associates an input with its `schema`, `rename`, `foreign_keys`,
`checks`, `load_policy`, `required`, or `description`. Its input can be any
supported path, `File`, frame, `Query`, `Loader`, or `DataSource`. It does not load
the input. Names come from the surrounding mapping; all names and contracts are
resolved together before execution. A single `Dataset` also works without a
mapping. Wrapping a `Dataset` inside another `Dataset` is an error.

Optional descriptor fields use `None` for omission. Explicit empty collections
remain explicit. Set a contract field in one place: a descriptor cannot replace
the same field explicitly declared in configuration, and `schema=` cannot also
appear inside a descriptor. Common load-policy fields may be added; a field
already declared with a different value is rejected. Repeating the same policy
value is allowed. This prevents a source change from silently relaxing a budget
or removing configured checks. `required=False` describes availability policy;
it does not make incorrect assessed data acceptable.

## Existing configuration and multiple datasets

```python
report = validate(config="qavf.yaml")
report = validate(
    {"counts": counts_lazy_frame, "sites": "sites.parquet"},
    config="qavf.yaml",
)
```

`config` accepts a YAML/JSON path, a configuration dictionary or `QAVFSettings`.
Named inputs replace source locations while retaining configured schema,
foreign-key and expression checks. A plain-path replacement also keeps its
explicit format, projection and reader options. Location-independent safeguards
belong in `datasets.<name>.load_policy` (`row_budget`, `retries`, `retry_delay_s`)
and survive replacements with a frame, query or loader. File-specific options do
not transfer to a different input kind. The caller's configuration object is not
mutated. If datasets are already configured, an unknown supplied name is an
error. A single unnamed input is ambiguous when the configuration has multiple
datasets; provide names instead.

Mappings passed as `data` always mean named datasets. Convert record or column
dictionaries to a dataframe first. Strings and `Path` objects always mean files
or URLs; SQL must use `Query`. `schema=` applies only to a single dataset and
cannot replace a schema already present in `config`.

Relative paths follow the configuration file's directory. Without a configuration
file they follow the current working directory.

## Queries and loaders

```python
from qavf import Loader, Query, validate

report = validate(
    Query("SELECT $value AS count", parameters={"value": 5}),
    schema={"columns": {"count": {"ge": 0}}},
)

report = validate(
    Loader(load_counts, kwargs={"batch_id": batch_id}, replayable=True),
    schema=counts_schema,
)
```

`Query` accepts one SELECT query, including CTEs, and named bound parameters.
Scalar and list parameters used as file-reader paths resolve relative to the
configuration directory. Scalar expressions using the same parameter keep its
original value; only the file-argument occurrence receives the resolved path.
SQL paths are normalized as a complete expression or
rejected before execution; QAVF never partially rewrites a path expression.
It can name a configured database connection; SQL references its tables as
`connection_name.table_name`. Native query scans remain available to the engine.
A query may instead borrow an existing DuckDB connection. That path performs an
explicit eager load and never closes the borrowed connection. QAVF serializes
its own queries sharing that connection within one validation run; callers must
not use it concurrently while the run is active.

`Loader` invokes a callable only when its dataset is required. It accepts supported
frames/tables and lazy-frame results. `replayable` and `retry_safe` describe the
callable; a nonzero explicit retry policy is also required for retries, and only
transient errors qualify. `retry_safe=True` requires `replayable=True`. Borrowed one-pass Arrow
readers require an explicit source adapter; they are not silently consumed or
cached by the facade. Existing `DataSource` objects can also be passed directly.
Their bound sessions retain source-owned paths, connection definitions and storage
secrets, including empty mappings; run settings supply execution resource policy.
Binding snapshots the definitions without mutating the caller's source.

`Query.parameters` and `Loader.kwargs` copy and freeze the mapping's bindings.
They do not copy nested mutable values or external handles. Those values remain
caller-owned: do not mutate them during validation, and keep any handles open
until the run finishes. This permits loaders to use connections and other objects
that cannot safely be deep-copied.

## Resource policy

```python
from qavf import ResourceLimits, validate

report = validate(
    "counts.parquet",
    schema=counts_schema,
    resources=ResourceLimits(
        target_memory="2GiB",
        cache_memory="256MiB",
        max_spill="4GiB",
        temp_dir="./qa_tmp",
        max_workers=2,
        native_threads=2,
        evidence_rows=50,
        evidence_bytes="1MiB",
        report_bytes="16MiB",
    ),
)
```

Sizes accept integer bytes or decimal (`MB`, `GB`) and binary (`MiB`, `GiB`)
units. `None` keeps the configuration's value. Zero cache, spill or evidence
limits are allowed; a total memory target must be positive. The memory target
guides execution and retention; it is not a hard operating-system limit.
Resource limits never authorize silently validating only a sample of the input.
`report_bytes` caps the serialized report graph and must be positive. Evidence
and nonessential metadata may be dropped with disclosure; if the essential
outcome ledger itself cannot fit, validation raises an explicit resource error.

For a supervised worker on Linux, macOS or Windows, add `wall_time=30`, `address_space="4GiB"`,
`cpu_time=30`, or `file_size="1GiB"`. These controls also work with
`compare_outputs` and select isolation automatically. See
[enforceable execution limits](execution_limits.md) for supported inputs, exact
OS scopes, failure behavior and the distinction between address space and RSS.

The existing `QARunner`, configuration models and plugin contracts remain
available for advanced orchestration. `import qavf` itself defers heavyweight
integration imports until a public object is requested.

## Progress and cancellation

Pass `on_event` to follow a run and a `CancellationToken` to stop it from any
thread, for example from a GUI's Stop button:

```python
from qavf import CancellationToken, validate

token = CancellationToken()


def show(event):
    print(event.kind, event.completed, event.total, event.rule_ids, event.dataset)


report = validate("data.csv", config="qavf.yaml", on_event=show, cancel=token)
# elsewhere: token.cancel()
```

Events arrive in order: `run_started`, then `step_started`/`step_finished` for
each step, then `run_finished` with the report. A step is one rule on the Polars
backend and one rule stage of one dataset on the DuckDB backend, where `total` is
`None` because the number of steps is not known in advance. `step_finished`
carries that step's results. A callback that raises is logged and ignored.

Cancelling interrupts running DuckDB queries and stops further steps. The run
still returns a report: finished steps keep their results, the rest are `ERROR`
"Not assessed: the run was cancelled.", `report.metadata.cancelled` is true and
the verdict is `incomplete`. Both hooks need `engine.execution_mode: in_process`.

On the command line, `qavf run --progress` writes each event as one JSON line to
stderr. Ctrl-C (or SIGTERM, or Ctrl-Break on Windows) cancels the run and still
writes its reports; a second interrupt stops at once.

## Comparing two datasets

`compare_datasets(baseline, candidate, keys=..., columns=..., levels=...)` takes the
same input forms and `resources=` policy as `validate` and returns a typed
`DatasetComparison`. See [dataset comparison](comparison.md).

## Plugin dependency declarations

Rule authors can preserve parameter typing with `Rule[MyParams]` and declare
complete dataset dependencies through `data_requirements(settings)`. For
example, this method belongs inside a rule that reads only `counts`:

```python
from qavf import DataRequirements


def data_requirements(self, settings):
    return DataRequirements(
        frozenset({"counts"}),
        columns={"counts": ("site_id", "aadt")},
    )
```

Include every dataset accessed by the rule, SQL and helpers. The scheduler
releases cached data after the last actual consumer finishes; a timeout notice
does not release data while the timed-out worker is still active. Column hints
describe the contract and do not independently authorize column projection.
An undeclared plugin conservatively retains every dataset. Existing `requires`
declares availability prerequisites and is not treated as a complete access list.
Built-in rules use the same explicit hook as plugins. Dynamic or incomplete
declarations remain conservative; the planner never infers ownership from a
module name or a parameter's spelling.

## Explain checks without reading datasets

```python
from qavf import explain

plan = explain(
    "counts.parquet",
    schema={"columns": {"aadt": {"dtype": "float", "ge": 0}}},
    profile="validity",
)
print(plan.model_dump_json(indent=2))
```

The immutable `ValidationPlan` lists selected rules, declared dataset/column
dependencies, known kinds of work, resource settings, excluded checks, and
unresolved information. It does not open input files, call loaders, collect lazy
frames, or consume Arrow protocols. File-based configuration itself is read.
Observed schemas, row counts, exact physical scans, and native working memory
remain unknown until execution; declared columns do not imply a safe projection.

By default, configured and installed plugin discovery is not performed, and the
plan discloses this in `unresolved`. `load_plugins=True` opts into ordinary plugin
imports and configuration hooks, which are trusted Python execution and may do
their own I/O. QAVF does not request dataset evaluation. `complete` means planning information was resolved,
not that data has been validated.

`explain` takes the selection and backend arguments of `validate`: `backend=`,
`only=`, `tags=`, `skip_tags=` and `datasets=`. The plan applies the same selection
(`plan.selection`, with unselected items under `excluded`) and the same backend
admission (`plan.backend`), so a check the chosen backend cannot run raises
`OptionalDependencyError` here, as `validate` would, and an unknown check name or
dataset raises `SelectionError`.

```python
plan = explain(config="project.yaml", backend="duckdb", datasets="counts", tags="nightly")
```

On the command line, `qavf explain` validates a configuration and prints its plan
without reading any data, so it doubles as a configuration check (it needs
`qavf[polars]`):

```text
qavf explain -c project.yaml [--backend auto|duckdb|polars] [--dataset NAME] [--tags TAG]
             [--skip-tags TAG] [--only NAME] [--format text|json] [-o plan.json]
```

It exits 0 when the configuration is valid and 2, naming the field path (for example
`datasets.counts.schema.columns.aadt.dtype`), when it is not or when the backend
cannot run a configured check. `-o` never writes over the configuration or a file it
names. `--load-plugins` plans plugin checks too, importing their code.

## Explicit check profiles

`validate(..., profile="validity")` adds INP-001 through INP-007 for declared
schema, keys, references, expressions, and assertions. It requires an explicit
contract and keeps every explicitly configured check. It does not learn limits
from the same data being assessed. Conflicting disabled checks or category
filters are configuration errors.

Columns can declare valid `na_values` tokens separately from missing values.
`schema.na_groups` adds all-or-none N/A consistency across column groups; the
validity profile then enables INP-009. See [applicability](applicability.md) for
population, nullability, normalization, and evidence semantics.

`profile="regression"` also requires enabled `REG-001` with `baselines`;
`profile="change"` requires enabled `CMP-001` with `comparisons`. Both include
the declared input checks. Expanded checks appear in the effective configuration
and profile provenance. Unconfigured unrelated rules are not selected.

For valid and stable geography, declare coordinate/domain bounds and configure
location comparison. For stable ratios and complete groups, configure both
relationship and coverage levels. Profiles do not invent those domain rules or
silently broaden specialized comparisons.

## Editor-visible input contracts

The `validate` and `compare_datasets` stub facades describe supported paths,
descriptors, Polars frames, compatible pandas/Arrow objects, and named datasets.
A plain mapping names datasets; it is not a dictionary of dataframe columns.
The runtime adapter remains authoritative for ownership and supported dtypes.
Static annotations do not prove replayability or memory limits. A plain
`import qavf` continues to defer heavyweight integrations.
