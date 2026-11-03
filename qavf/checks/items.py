"""Configuration models for named checks: check items and the ``run`` section.

This module is imported by :mod:`qavf.settings`, so it must not import the rule
implementations. Check-specific settings are kept as extra fields of
:class:`CheckItem` and validated against the check catalog
(:mod:`qavf.checks.catalog`) when the configuration is loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qavf.engine.severity import Severity

COMMON_FIELDS = ("check", "name", "severity", "where", "tags", "enabled", "description")
"""Fields every check item accepts; everything else is the check's own settings."""


class CheckItem(BaseModel):
    """One configured check: ``{check: missing_periods, series: {...}, expected: [...]}``.

    The common fields below apply to every check. Any other key is a setting of
    the check named by ``check`` (see ``qavf list-checks``).
    """

    model_config = ConfigDict(
        extra="allow",
        use_attribute_docstrings=True,
        hide_input_in_errors=True,
    )

    check: str = Field(min_length=1)
    """Kind of check, e.g. ``missing_periods`` (see ``qavf list-checks``)."""
    name: str | None = Field(default=None, min_length=1)
    """Your label; unique within its list; shown in reports and usable in ``run.require``.
    Defaults to ``<check> on <dataset>``."""
    severity: Severity | None = None
    """Overrides the check's default severity."""
    where: str | None = None
    """SQL condition: assess only the rows that match it."""
    tags: list[str] = Field(default_factory=list)
    """Words grouping checks for run-time selection."""
    enabled: bool = True
    """Keep a check configured but switched off."""
    description: str | None = None
    """Shown in reports."""

    @field_validator("tags")
    @classmethod
    def _tag_words(cls, tags: list[str]) -> list[str]:
        for tag in tags:
            if not tag or not tag.strip() or any(ch.isspace() for ch in tag):
                raise ValueError(f"tag {tag!r} must be a single nonempty word")
        return tags

    @property
    def settings(self) -> dict[str, Any]:
        """The check's own settings (every key that is not a common field)."""
        return dict(self.model_extra or {})

    def default_name(self, datasets: tuple[str, ...]) -> str:
        """``<check> on <dataset>`` (several datasets joined with ``and``)."""
        return f"{self.check} on {' and '.join(datasets)}" if datasets else self.check

    def effective_name(self, datasets: tuple[str, ...]) -> str:
        """The item's ``name``, or its default name."""
        return self.name or self.default_name(datasets)


class ExpressionItem(CheckItem):
    """``expression``: a SQL condition every row of the dataset satisfies (INP-006)."""

    model_config = ConfigDict(extra="forbid", title="ExpressionItem")

    check: Literal["expression"]  # type: ignore[assignment]
    expr: str
    """Condition every row satisfies, e.g. ``pm25 <= pm10``; it may query other
    datasets by name in subqueries."""
    null_as: Literal["pass", "fail"] = "pass"
    """How a row whose expression evaluates to NULL counts (SQL CHECK semantics: pass)."""


class SqlItem(CheckItem):
    """``sql``: a query returning offending rows across any datasets (INP-007)."""

    model_config = ConfigDict(extra="forbid", title="SqlItem")

    check: Literal["sql"]  # type: ignore[assignment]
    name: str = Field(min_length=1)
    """Your label; unique among the top-level checks; shown in reports."""
    sql: str
    """Query returning offending rows; every dataset is a table under its name."""


class TagPolicy(BaseModel):
    """Pass/fail policy for the checks carrying one tag."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    fail_on: Severity | Literal["never"]
    """Lowest severity of a failure of these checks that fails the run; ``never``
    keeps them from failing it (failures are still reported)."""


class RunSettings(BaseModel):
    """Pass/fail policy of a run."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    fail_on: Severity = Severity.ERROR
    """Lowest severity of a failed check that fails the run (exit code 1)."""
    fail_on_incomplete: bool = True
    """A check that could not be assessed fails the run, whatever its severity."""
    require: list[str] = Field(default_factory=list)
    """Checks that must actually be assessed: check names, or a dataset name for every
    check on that dataset. A required check that is disabled, skipped or not assessed
    makes the run incomplete. An entry that matches nothing is a configuration error."""
    tags: dict[str, TagPolicy] = Field(default_factory=dict)
    """Pass/fail policy per tag, e.g. ``exploratory: {fail_on: never}``. Built-in checks
    also carry their family as a tag (``temporal``, ``input``, …). When several tags of a
    check have a policy, the strictest applies."""

    @field_validator("tags")
    @classmethod
    def _tag_words(cls, tags: dict[str, TagPolicy]) -> dict[str, TagPolicy]:
        for tag in tags:
            if not tag or any(ch.isspace() for ch in tag):
                raise ValueError(f"tag {tag!r} must be a single nonempty word")
        return tags

    def tag_fail_on(self) -> dict[str, str]:
        """Each tag's ``fail_on`` as text (a severity, or ``never``), for reports."""
        return {tag: str(policy.fail_on) for tag, policy in self.tags.items()}


def checks_union(models: Sequence[type[BaseModel]]) -> Any:
    """The type of a ``checks`` list entry: one of ``models``, chosen by ``check``.

    Used to describe configurations (the JSON Schema), not to validate them.
    """
    if len(models) == 1:
        return models[0]
    return Annotated[Union[tuple(models)], Field(discriminator="check")]  # noqa: UP007


__all__ = [
    "COMMON_FIELDS",
    "CheckItem",
    "ExpressionItem",
    "RunSettings",
    "SqlItem",
    "TagPolicy",
    "checks_union",
]
