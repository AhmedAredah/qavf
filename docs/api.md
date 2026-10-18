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

When replacing a configured file, a plain path keeps that file's reader options.
An explicit `File` supplies a complete new reader specification. In both cases,
configured QA contracts and common loading safeguards remain in effect.

## Inline contracts for named datasets

```python
from qavf import Dataset, File, validate

report = validate({
    "counts": Dataset(
        File("counts.csv", options={"delim": ";"}),
        schema={"columns": {"site_id": {"dtype": "str"}, "count": {"ge": 0}}},
        foreign_keys=[{"columns": ["site_id"], "ref_dataset": "sites"}],
        load_policy={"row_budget": 1_000_000},
    ),
    "sites": Dataset("sites.parquet", schema={"primary_key": ["site_id"]}),
})
```

`Dataset` associates an input with its `schema`, `rename`, `foreign_keys`,
`expressions`, `load_policy`, `required`, or `description`. Its input can be any
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
configuration directory. SQL paths are normalized as a complete expression or
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
