# Writing plugins

QAVF finds rules and sources by scanning, so a plugin is just Python code in a place
QAVF looks (comparison components register with decorators in the same modules; see
[comparison components](#comparison-components)):

| Mechanism | Configuration | Use it for |
|---|---|---|
| Package scan | `plugins.packages: [my_pkg.rules]` | rules in a package you import anyway |
| Module scan | `plugins.modules: [my_pkg.extra]` | a single module of rules and/or sources |
| File scan | `plugins.paths: [./qa_rules]` | project-local rules, no packaging |
| Entry points | *(automatic when installed)* | distributable plugin packages |

Every concrete `Rule` or `DataSource` subclass *defined in* a scanned module is
registered. Modules and subpackages whose name starts with `_` are private: they are
never imported during discovery and nothing inside them is scanned, so put helpers there.

## Domain packs

Model-specific identifiers, units, value domains and physical constraints belong in
versioned domain packs rather than in the generic core.
A pack is an ordinary plugin: rules (and sources) registered under the entry-point
groups, its own settings under `extensions.<name>`, and its version recorded in every
report (`metadata.rules[].distribution`, `metadata.plugins`). Pin the pack in your
project's dependencies and list it in `plugins.required` so a run never proceeds
without it.

## Rules

```python
from typing import ClassVar

import polars as pl
from pydantic import Field

from qavf import DataContext, Finding, Rule, RuleParams, Severity


class ShareParams(RuleParams):
    """Parameters; validated from `rules.ACME-002.params` in the configuration."""

    max_share: float = Field(default=0.25, gt=0, le=1)
    """Largest acceptable share. Attribute docstrings become schema descriptions."""


class HeavyShareRule(Rule):
    rule_id: ClassVar[str] = "ACME-002"
    name: ClassVar[str] = "Heavy-vehicle share"
    description: ClassVar[str] = "The heavy-vehicle share of traffic stays below a ceiling."
    category: ClassVar[str] = "acme"
    default_severity: ClassVar[Severity] = Severity.WARNING
    requires: ClassVar[tuple[str, ...]] = ("counts",)  # SKIPPED when unavailable
    recommendation: ClassVar[str] = "Check the vehicle classification for these sites."
    Params: ClassVar[type[RuleParams]] = ShareParams

    def evaluate(self, ctx: DataContext) -> Finding:
        counts = ctx.get("counts")
        over = counts.filter(pl.col("truck_share") > self.params.max_share)
        if over.is_empty():
            return Finding.ok("All shares within the ceiling.", total_records=counts.height)
        return Finding.fail(
            f"{over.height} rows exceed {self.params.max_share:.0%}.",
            affected=over,  # the engine samples and serialises it
            total_records=counts.height,
            metrics={"max_share": float(over["truck_share"].max())},
        )
```

Rules must not mutate the frames they read: rules run concurrently on threads.

### What the context offers

| Call | Returns |
|---|---|
| `ctx.get(name)` | Polars DataFrame (loads on first use; raises if unavailable) |
| `ctx.has(name)` / `ctx.unavailable_reason(name)` | availability / why not |
| `ctx.pandas(name)` | cached pandas view (read-only) |
| `ctx.sql(query)` | DuckDB query with datasets as tables |
| `ctx.specs[name]` | the dataset's configuration (schema, keys, …) |
| `ctx.extension("my_plugin")` | `extensions.my_plugin` from the configuration |
| `ctx.memo(key, fn)` | compute once, share between rules |
| `ctx.load_source({...})` | load an ad-hoc source (e.g. a baseline) |

`qavf.rules.helpers` has `usable`, `total`, `tag`, `stack`, `plural`, `outcome` and
`MetricSpec`, which the built-in rules use.

### One rule, several graded checks

Return a list of findings, each with a `check` label and optionally its own
`severity`. Each becomes its own result row in the report (`ACME-003 [check]`):

```python
from qavf.rules.helpers import outcome


def evaluate(self, ctx):
    findings = []
    for column in ("a", "b"):
        bad = ctx.get("t").filter(pl.col(column) < 0)
        if not bad.is_empty():
            findings.append(
                Finding.fail(
                    f"{column} negative",
                    check=column,
                    affected=bad,
                    severity=Severity.CRITICAL if column == "a" else None,
                )
            )
    return outcome(
        findings,
        passed=2 - len(findings),
        ok_message="checks passed",
        nothing_message="nothing to check",
    )
```

### Plugin settings

Read plugin-wide settings from `extensions` rather than repeating them in every rule:

```yaml
extensions:
  acme: {region_file: regions.csv}
```

```python
region_file = ctx.extension("acme").get("region_file")
```

## Sources

```python
from typing import ClassVar, Literal

import polars as pl

from qavf.sources import DataSource, SourceError, SourceParams


class HttpCsvParams(SourceParams):
    type: Literal["http_csv"] = "http_csv"
    url: str


class HttpCsvSource(DataSource):
    kind: ClassVar[str] = "http_csv"
    Params: ClassVar[type[SourceParams]] = HttpCsvParams

    def load(self) -> pl.DataFrame:
        try:
            return pl.read_csv(self.params.url)
        except Exception as exc:
            raise SourceError(str(exc)) from exc

    def describe(self) -> str:
        return self.params.url
```

```yaml
datasets:
  remote: {source: {type: http_csv, url: "https://example.org/data.csv"}}
```

Raise `SourceNotFoundError` when the data simply does not exist (optional datasets
may be absent) and `SourceError` for anything else. `self.env.resolve(path)`
resolves paths against the configuration directory.

### Run-local source ownership

`DataSource` objects are definitions that can be reused. QAVF calls `bind(env)` to
obtain an independent source session for each run. A parameter-only source that
inherits the base constructor receives this behavior automatically. A source with
a custom constructor or stateful external handle must implement `bind` explicitly:

```python
def bind(self, env):
    return type(self)(self.params.model_copy(deep=True), self.bound_environment(env))
```

Adapt that factory to the actual constructor and ownership of the handle. Do not
share mutable load notes, cached manifests or a run's connections between sessions.
QAVF does not blindly deep-copy arbitrary plugin objects. `verify_identity()` can
enforce a source snapshot; raise `SourceIntegrityError` on a detected change.
Native connections created through the supplied environment participate in run
cancellation and cleanup. Borrowed connections remain owned by the caller.

Rule dependency hooks must declare every dataset their SQL/helpers access. Return
`None` for a dynamic access list. Discovery registers a module atomically: a bad or
duplicate declaration prevents that module's partially discovered rules or sources
from remaining active. Public rule IDs are preserved exactly in configuration and
JSON Schema, including IDs whose punctuation differs.

## Comparison components

Dataset comparison ([`compare_datasets`, `CMP-001`](comparison.md)) is assembled from
four kinds of registered component. Each is a class with a `name` (how configuration
selects it), an optional `Params` model (a `RuleParams` subclass validated when the
configuration loads) and one method:

| Kind | Base class | Method | Registration | Entry-point group |
|---|---|---|---|---|
| Alignment step | `AlignmentStep` | `apply(plan, context) -> LazyFrame` | `register_alignment_step` | `qavf.compare.alignment` |
| Measure | `Measure` | `compute(data) -> DataFrame` (group keys + `data.alias`) | `register_measure` | `qavf.compare.measures` |
| Comparator | `Comparator` | `compare(baseline, candidate, context) -> value` | `register_comparator` | `qavf.compare.comparators` |
| Level | `ComparisonLevel` | `evaluate(inputs, name) -> LevelResult` | `register_level` | `qavf.compare.levels` |

Built-in components register through the same decorators, and the generated
[component reference](comparison_reference.md) lists them with their parameters.
Contracts:

- **Steps** transform one side's lazy relation (`sides` limits which) and refuse with
  `context.refuse(...)`/`AlignmentError` (or `context.require_none(plan, predicate,
  message)` for a data proof). A refusal makes the comparison not assessed; never
  coerce silently. A step that changes the data in a way the configuration alone does
  not say (rows it was asked to drop) records it with `context.note(message)`; notes
  follow the step descriptions in `DatasetComparison.alignment`.
- **Measures** reduce `data.relation` (a scoped relation over the aligned side) per
  `data.keys` group without collecting rows: use `grouped_reduce`, `grouped_aggregate`,
  `grouped_mean`, `grouped_quantile` (`qavf.rules._grouped`) or `exact_grouped`.
  Values may be numbers or exact rational strings. Override `check` to refuse a
  column, `columns` to declare extra columns read, and set `numeric`/`requires_column`.
- **Comparators** receive exact `Fraction`s and return a `Fraction`, a float or `None`
  (undefined). Materiality is judged separately.
- **Levels** read both sides through `LevelInputs` (`schema`, `scan`, `relation`,
  `collect`, `rows`, `missing_counts`, `evidence`) and return a `LevelResult` (or a
  subclass set as `Result`, which is also used to read the result back from JSON)
  with graded `CheckResult`s. Raise for anything that cannot be assessed. Override
  `configuration_errors(spec)` for problems visible in the configuration alone.
  A level that compares keyed reductions builds on the built-in ones instead of
  reducing again: `reduce_groups(inputs, keys, measures, relations)`,
  `one_sided_check` and `compare_measure` (`qavf.compare.levels.groups`) reduce,
  align and judge any keys, and `bucketed_sides(inputs, params)`
  (`qavf.compare.levels.temporal`) adds time-bucket labels to both sides. The
  bucketing itself (`bucket_labels`, `bucket_ordinal`, `bucket_grid`, calendar
  offsets) is `qavf.temporal`, shared with the TMP series rules.

A custom measure and a custom level:

```python
from typing import ClassVar

import polars as pl

from qavf.compare import (
    CheckResult,
    ComparisonLevel,
    LevelResult,
    Measure,
    register_level,
    register_measure,
)
from qavf.engine.registry import RuleParams
from qavf.rules._grouped import grouped_reduce


class RangeParams(RuleParams):
    scale: int = 1
    """Multiply the range, e.g. to change its unit."""


@register_measure
class Range(Measure):
    """Largest minus smallest value per group."""

    name: ClassVar[str] = "range"
    Params: ClassVar[type[RuleParams]] = RangeParams

    def compute(self, data):
        value = pl.col(data.column)
        spread = (value.max() - value.min()) * self.params.scale
        return grouped_reduce(data.relation, data.keys, {data.alias: spread})


@register_level
class RowParity(ComparisonLevel):
    """Both sides have an even number of rows, or both an odd number."""

    name: ClassVar[str] = "row_parity"

    def evaluate(self, inputs, name):
        parity = {side: inputs.rows(side) % 2 for side in ("baseline", "candidate")}
        same = parity["baseline"] == parity["candidate"]
        check = CheckResult(check="parity", passed=same, message=f"row parity {parity}")
        return LevelResult(level=self.name, name=name, checks=[check])
```

```yaml
levels:
  - row_parity
  - {level: groups, group_by: [county], measures: [{measure: range, column: aadt, scale: 2}]}
```

Packaged plugins advertise them like rules:

```toml
[project.entry-points."qavf.compare.measures"]
acme = "qavf_acme.measures"      # a module (scanned) or a single component class

[project.entry-points."qavf.compare.levels"]
acme = "qavf_acme.levels"
```

Comparison entry points follow `plugins.entry_points` exactly like rules and sources.
When it is true, the `qavf.compare.*` groups load while the configuration is compiled,
before `CMP-001` parameters are validated, so a plugin level named there resolves. When
it is false, no installed plugin code is imported and naming a plugin component is a
configuration error; looking components up never loads entry points by itself.
`compare_datasets(..., entry_points=False)` makes the same choice for the Python API,
and isolated workers apply the run's choice when they validate the configuration again.
Each entry point tried is recorded in `metadata.plugins` as `compare.<kind>:<name>`, a
broken one leaves no partial registration, and `plugins.required: [compare.levels:acme]`
stops a run whose level plugin is missing, broken or disabled. Components registered in
code with the decorators (for example from `plugins.modules`) are always available.

## Packaging a plugin

```toml
# pyproject.toml of your plugin
[project]
name = "qavf-acme"
dependencies = ["qavf>=0.1"]

[project.entry-points."qavf.rules"]
acme = "qavf_acme.rules"        # a module (scanned) or a single Rule class

[project.entry-points."qavf.sources"]
acme = "qavf_acme.sources"
```

Once installed, `qavf list-rules` shows the rules and `qavf schema` / `qavf init`
include their parameters. `qavf.docgen.rules_markdown(registry)` generates a
reference page for your rules.

## Testing plugins

```python
from qavf import Status
from qavf.testing import run_rule, rule_results


def test_flags_high_share():
    frame = pl.DataFrame({"truck_share": [0.1, 0.9]})
    result = run_rule("ACME-002", {"counts": frame}, params={"max_share": 0.5})
    assert result.status is Status.FAIL and result.affected_count == 1
```

`run_rule` / `rule_results` run one rule against in-memory frames; pass
`datasets=`, `extensions=`, `assertions=` or `connections=` for configuration.
Installed entry-point plugins are discovered automatically. A plugin that fails to
import is recorded (not just logged): `metadata.plugins` in every report lists each
entry point as `loaded` or with its error, and `plugins.required: [moves]` makes the
run stop with a configuration error if a named plugin is missing or broken. Pass `registry=` to use
exactly that registry (an empty `RuleRegistry()` runs nothing). `discovered_rules()`
returns a fresh registry on every call that includes rules registered with
`@register` since the last call; `clear_discovery_cache()` forces a rescan of packages
and entry points.


## Required components and discovery transactions

`plugins.required` accepts entry-point names or qualified `rules:name` / `sources:name`
/ `compare.<kind>:name` entries. An unqualified name requires all advertised components of that name; a source
component cannot satisfy a failed rule component. Duplicate entry-point names within a
group are rejected before either entry point loads.

QAVF stages public `@register` changes while discovering an extension and publishes
those changes only after successful loading and declaration validation. A failed import
cannot leave an active rule behind. This transaction governs QAVF registries; plugin
code remains responsible for any external effects it performs during import.

A source's `concurrent_loads`, retry safety and identity hooks apply to eager loads,
SQL operations, typed lazy collection and snapshot preparation. Use context operations
to execute a source-backed plan; collecting it independently bypasses the coordinator.
