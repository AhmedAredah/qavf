"""Regression rule: generated outputs still match their accepted baselines."""

from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from qavf.engine.context import DataContext
from qavf.engine.planning import DataRequirements
from qavf.engine.registry import Finding, Rule, RuleParams
from qavf.engine.severity import Severity
from qavf.redaction import redact, redact_text
from qavf.regression._inputs import compare_context
from qavf.regression.baselines import read_verified_baseline
from qavf.regression.models import SchemaPolicy
from qavf.settings import QAVFSettings
from qavf.sources.base import SourceError

CATEGORY = "regression"


class ColumnTolerance(RuleParams):
    """A tolerance for one column of a baseline comparison."""

    rtol: float | None = Field(default=None, ge=0)
    """Relative tolerance; the baseline's (or rule's) when omitted."""
    atol: float | None = Field(default=None, ge=0)
    """Absolute tolerance; the baseline's (or rule's) when omitted."""
    exact: bool = False
    """Compare exactly (no tolerance), e.g. for identifiers or categorical codes."""

    @model_validator(mode="after")
    def _exact_alone(self) -> ColumnTolerance:
        if self.exact and (self.rtol is not None or self.atol is not None):
            raise ValueError("'exact' cannot be combined with 'rtol' or 'atol'")
        return self


class BaselineSpec(RuleParams):
    """One dataset and the baseline file it must match."""

    dataset: str
    name: str | None = None
    """Optional unique comparison identity when a dataset has several baselines."""
    path: str | None = None
    """Baseline file; relative paths resolve against the configuration file."""
    source: dict[str, Any] | None = None
    """Any source specification instead of ``path``, e.g. a baseline database table."""
    keys: tuple[str, ...] = Field(min_length=1)
    rtol: float | None = Field(default=None, ge=0)
    atol: float | None = Field(default=None, ge=0)
    ignore_columns: tuple[str, ...] = ()
    schema_policy: SchemaPolicy = Field(default="compatible", alias="schema")
    """``schema`` in configuration. How column types are compared: ``compatible``
    reports changes of kind (number to text, date to datetime, a time zone added),
    ``strict`` any dtype change, ``ignore`` none."""
    columns: dict[str, ColumnTolerance] = Field(default_factory=dict)
    """Per-column tolerances, e.g. ``{share: {atol: 0.001}, code: {exact: true}}``, so
    one mixed-unit table needs no globally loose tolerance."""

    @model_validator(mode="after")
    def _one_location(self) -> BaselineSpec:
        if (self.path is None) == (self.source is None):
            raise ValueError("set exactly one of 'path' or 'source'")
        repeated = sorted({k for k in self.keys if self.keys.count(k) > 1})
        if repeated:
            raise ValueError(f"keys repeat column(s): {', '.join(repeated)}")
        ignored = sorted(set(self.keys) & set(self.ignore_columns))
        if ignored:
            raise ValueError(f"key column(s) {', '.join(ignored)} cannot also be ignored")
        clash = sorted(set(self.columns) & (set(self.keys) | set(self.ignore_columns)))
        if clash:
            raise ValueError(
                f"column tolerances cannot apply to key or ignored column(s): {', '.join(clash)}"
            )
        return self


class BaselineParams(RuleParams):
    """Parameters for :class:`BaselineComparisonRule`."""

    baselines: tuple[BaselineSpec, ...] = ()
    """Datasets and the baselines they must match."""
    rtol: float = Field(default=1e-6, ge=0)
    atol: float = Field(default=1e-9, ge=0)
    max_differences: int = Field(default=1000, ge=0)
    require_manifest: bool = False
    """Only compare against baseline files accepted with a manifest (``qavf baseline``):
    a baseline without one is not used, so an unreviewed file cannot become the
    reference."""


class BaselineComparisonRule(Rule):
    """Outputs match their baselines within tolerance (exact keyed comparison)."""

    rule_id: ClassVar[str] = "REG-001"
    name: ClassVar[str] = "Outputs match baseline"
    description: ClassVar[str] = (
        "Generated outputs equal their stored baselines row-for-row by key, within the "
        "configured numeric tolerance."
    )
    category: ClassVar[str] = CATEGORY
    default_severity: ClassVar[Severity] = Severity.ERROR
    recommendation: ClassVar[str] = (
        "Decide whether the change is intended. If a code or input change explains it, "
        "review the differences and refresh the baseline (qavf baseline). If not, the "
        "utility has regressed: bisect recent changes using the listed keys."
    )
    Params: ClassVar[type[RuleParams]] = BaselineParams

    def data_requirements(self, settings: QAVFSettings) -> DataRequirements | None:
        """Candidate inputs; opaque baseline loaders conservatively retain all inputs."""
        if any(
            spec.source is not None and spec.source.get("type") not in {"file", "query", "database"}
            for spec in self.params.baselines
        ):
            return None
        return DataRequirements(frozenset(spec.dataset for spec in self.params.baselines))

    def evaluate(self, ctx: DataContext) -> Finding | list[Finding]:
        if not self.params.baselines:
            return Finding.skip("No baselines are configured.")
        counts = Counter(spec.dataset for spec in self.params.baselines)
        identities: set[str] = set()
        findings = []
        for index, spec in enumerate(self.params.baselines, start=1):
            identity = spec.name or (
                spec.dataset if counts[spec.dataset] == 1 else f"{spec.dataset}#{index}"
            )
            if identity in identities:
                findings.append(
                    Finding.unassessed(
                        f"Duplicate baseline comparison identity {identity!r}.", check=identity
                    )
                )
                continue
            identities.add(identity)
            findings.append(self._compare(ctx, spec, identity))
        return findings

    def _compare(self, ctx: DataContext, spec: BaselineSpec, identity: str) -> Finding:
        config = dict(spec.source or {"type": "file", "path": spec.path})
        p = self.params
        rtol = p.rtol if spec.rtol is None else spec.rtol
        atol = p.atol if spec.atol is None else spec.atol
        per_column = {
            column: (0.0, 0.0)
            if value.exact
            else (
                rtol if value.rtol is None else value.rtol,
                atol if value.atol is None else value.atol,
            )
            for column, value in spec.columns.items()
        }
        fingerprint: dict[str, Any] | None
        try:
            if not ctx.has(spec.dataset):
                raise ValueError("dataset unavailable")
            location = config.get("path")
            local_file = (
                config.get("type") == "file"
                and isinstance(location, str)
                and (not urlsplit(location).scheme or bool(Path(location).drive))
                and not glob.has_magic(location)
            )
            options = {
                "rtol": rtol,
                "atol": atol,
                "ignore_columns": spec.ignore_columns,
                "max_differences": min(p.max_differences, ctx.evidence_limit),
                "max_evidence_bytes": ctx.evidence_bytes,
                "name": identity,
                "column_tolerances": per_column,
                "schema": spec.schema_policy,
            }
            if local_file:
                assert isinstance(location, str)
                resolved = ctx.resolve_path(location)
                config["path"] = str(resolved)
                (result, _), _manifest, fingerprint = read_verified_baseline(
                    resolved,
                    spec.keys,
                    p.require_manifest,
                    lambda: compare_context(ctx, spec.dataset, spec.keys, source=config, **options),
                )
            else:
                if p.require_manifest:
                    raise ValueError(
                        "require_manifest needs one local accepted baseline file; "
                        "this source has no verifiable acceptance manifest"
                    )
                result, fingerprint = compare_context(
                    ctx,
                    spec.dataset,
                    spec.keys,
                    source=config,
                    fingerprint=True,
                    **options,
                )
        except (ValueError, SourceError, OSError) as exc:
            reason = redact_text(str(exc))
            return Finding.unassessed(
                f"Baseline comparison {identity} could not run: {reason}.",
                check=identity,
                affected=[{"comparison": identity, "dataset": spec.dataset, "kind": "unavailable"}],
                affected_count=1,
                unit="problems",
                datasets=(spec.dataset,),
                metrics={"unavailable": [reason]},
            )
        metrics = {
            "summaries": {identity: result.summary()},
            "comparisons": {identity: result.model_dump(mode="python", exclude={"differences"})},
            "unavailable": [],
            "baselines": {identity: redact(fingerprint)},
        }
        if result.equal:
            return Finding.ok(
                result.summary(), check=identity, metrics=metrics, datasets=(spec.dataset,)
            )
        records = [
            {
                "comparison": identity,
                "dataset": spec.dataset,
                **difference.model_dump(mode="python"),
            }
            for difference in result.differences
        ]
        total = (
            result.values_changed
            + result.rows_added
            + result.rows_removed
            + len(result.columns_added)
            + len(result.columns_removed)
            + len(result.dtypes_changed)
        )
        return Finding.fail(
            result.summary(),
            check=identity,
            affected=records,
            affected_count=total,
            unit="differences",
            metrics=metrics,
            datasets=(spec.dataset,),
        )
