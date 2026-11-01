# Structured check identity

Every `RuleResult` retains `rule_id`, the human-readable `check` label, and the
existing status/count fields. Version 0.7 adds `check_id`, `scope`, `columns`,
`parameters`, and `checks`. `CheckIdentity` carries the same contract fields plus
its datasets and rule ID. The complete identity is the combination of operation,
rule, scope, datasets, columns, and parameters; `check_id` alone names an operation,
such as `column.dtype` or `key.primary`.

For example, a primary-key duplicate result identifies its key columns, the null
policy, and the configured rows per key. A column-limit failure carries the
column-check parameters; built-in defaults are omitted from compact identities
and follow the QAVF version recorded in the report. A passing summary contains the constituent
contracts in `checks`, so consumers need not interpret `metrics.passed_checks` or
split display labels on punctuation. Schema summaries identify each dataset's
schema contract. SQL assertions explicitly carry the query contract with no
invented source-column names. Rule-level results and older/custom plugins use an
explicit rule scope and their validated rule parameters unless the producer
supplies a more specific identity.

`result.identity` exposes the result's own `CheckIdentity` without duplicating
those fields in JSON. `checks` contains constituent contracts for summaries;
ordinary single-check results can leave it empty.

These fields are immutable audit data and survive optional evidence trimming.
They are serialized for PASS, FAIL, ERROR, and SKIPPED results and restored when
loading reports. Reports produced by older versions remain loadable and acquire
a rule-level identity. Legacy labels, constructors and `rule_id` stay available.

Plugin authors can opt in without changing existing constructors:

```python
return Finding.fail(
    "Amount exceeds the limit.",
    check="amount maximum",  # Presentation only.
    check_id="amount.maximum",
    datasets=("orders",),
    columns=("amount",),
    parameters={"maximum": 100},
)
```

For a summary, set `checks` to a tuple of mappings containing `check_id`,
`datasets`, `columns`, and `parameters`; the runner supplies `rule_id`. Identity
parameters support the same lossless audit encoding as other report values,
including `Decimal`. Parameters are redacted through the existing report policy.

`evidence_complete` and `evidence_truncated` are also outcome fields. A duplicate
result is incomplete if any duplicate group was omitted or any involved position
was capped. Exact `affected_count` and `affected_rows` remain independent of
evidence completeness. The more detailed duplicate-position metrics distinguish
per-key truncation from the outer group sample when the evidence budget allows
those optional details.
