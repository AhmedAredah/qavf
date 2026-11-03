"""The check catalog: every built-in check name, its rule code and its settings.

A check item is written in the vocabulary of the data (``{check: change_rate,
series: {...}, max: 0.15}``). Its :class:`CheckDefinition` turns it into the
parameters of the rule that implements it (``TMP-001`` with
``{series: [...], max_change_rate: 0.15}``), so the rule implementations are
unchanged. Each definition can also turn rule parameters back into items, which
``qavf migrate`` uses to convert QAVF 0.6 configurations.

Definitions differ only in how an item maps onto rule parameters:

* :class:`ElementCheck`: the item is one entry of a list parameter (a baseline,
  a share, a freshness column), plus rule-level settings.
* :class:`InlineCheck`: the item names an inline ``series:`` or ``metric:``
  mapping (temporal and spatial checks), plus rule-level settings.
* :class:`PairCheck` subclasses: top-level checks relating ``datasets: [a, b]``.
* :class:`WholeCheck`: the item's settings are the rule's parameters.
* :class:`ExpressionDefinition` and :class:`SqlDefinition`: row expressions and
  SQL queries, evaluated together by one rule each.
"""

from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from qavf.checks.items import CheckItem, ExpressionItem, SqlItem
from qavf.checks.placement import CheckSettingError, Scope, expression_errors, sql_errors
from qavf.engine.registry import Rule
from qavf.input_rules import INPUT_RULES
from qavf.rules.comparison_checks import DatasetComparisonRule
from qavf.rules.freshness_checks import FreshnessRule
from qavf.rules.geo_checks import CoordinateRule
from qavf.rules.input_checks import (
    ApplicabilityRule,
    ColumnCheckRule,
    DuplicateKeyRule,
    ExpressionRule,
    ReferentialIntegrityRule,
    RequiredDatasetsRule,
    SchemaConformanceRule,
    SqlAssertionRule,
)
from qavf.rules.reconciliation_checks import (
    ConsistentBatchRule,
    KeyGridRule,
    KeySetsMatchRule,
    SharesSumRule,
    TotalsReconcileRule,
)
from qavf.rules.regression_checks import BaselineComparisonRule
from qavf.rules.spatial_checks import CoverageRule, EntityOutlierRule, GroupComparisonRule
from qavf.rules.statistical_checks import ColumnOutlierRule, DistributionDriftRule
from qavf.rules.temporal_checks import (
    CumulativeRule,
    LevelShiftRule,
    MissingPeriodRule,
    SeasonalAnomalyRule,
    StaleValueRule,
    StepChangeRule,
    TrendDirectionRule,
    TrendOutlierRule,
)
from qavf.settings import ExpressionCheck, expression_check, sql_assertion

Placement = Literal["dataset", "top"]
FieldSpec = tuple[Any, Any]
"""A field's annotation and its ``FieldInfo``, as ``create_model`` takes them."""

LATER_VERSION = "arrives in a later version of QAVF"


@dataclass(frozen=True, slots=True)
class Decompiled:
    """A check item recovered from rule parameters (for ``qavf migrate``)."""

    dataset: str | None
    """The dataset the item belongs on, or ``None`` for a top-level item."""
    item: dict[str, Any]


def _reverse(renames: Mapping[str, str]) -> dict[str, str]:
    return {old: new for new, old in renames.items()}


def _field_copy(field: FieldInfo) -> FieldSpec:
    return field.annotation, copy.copy(field)


def _fields_of(
    model: type[BaseModel], renames: Mapping[str, str], exclude: Sequence[str] = ()
) -> dict[str, FieldSpec]:
    """``model``'s fields under their item names (``renames`` maps item -> model name)."""
    back = _reverse(renames)
    return {
        back.get(name, name): _field_copy(field)
        for name, field in model.model_fields.items()
        if name not in exclude
    }


def _location(loc: Sequence[Any]) -> str:
    return ".".join(str(part) for part in loc if not isinstance(part, int))


class CheckDefinition(ABC):
    """How one named check maps onto its rule's parameters."""

    configurable: ClassVar[bool] = True
    """False for schema contracts, which come from ``schema`` rather than items."""
    list_driven: ClassVar[bool] = False
    """True when one rule instance evaluates every item of this check."""

    def __init__(
        self, rule_cls: type[Rule], placement: Placement, *, where_field: str | None = None
    ) -> None:
        self.rule_cls = rule_cls
        self.placement: Placement = placement
        self.where_field = where_field

    @property
    def name(self) -> str:
        return self.rule_cls.check_name

    @property
    def rule_id(self) -> str:
        return self.rule_cls.rule_id

    @property
    def summary(self) -> str:
        return self.rule_cls.description

    @abstractmethod
    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        """The rule parameters for ``item``."""

    @abstractmethod
    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        """Check items equivalent to the rule parameters ``params``."""

    @abstractmethod
    def item_fields(self) -> dict[str, FieldSpec]:
        """The check's own settings, for the configuration JSON Schema."""

    def locate(self, loc: Sequence[Any]) -> str:
        """The item setting a rule-parameter error location refers to."""
        return _location(loc)

    def datasets(self, item: CheckItem, scope: Scope) -> tuple[str, ...]:
        """The datasets an item tests or relates."""
        if scope.dataset is not None:
            return (scope.dataset,)
        named = item.settings.get("datasets")
        if isinstance(named, list | tuple) and all(isinstance(n, str) for n in named):
            return tuple(named)
        return ()

    def where(self, item: CheckItem) -> dict[str, Any]:
        """The element field receiving the common ``where`` setting."""
        if item.where is None:
            return {}
        if self.where_field is None:
            raise CheckSettingError(
                "where", f"'where' is not supported by the {self.name} check yet"
            )
        return {self.where_field: item.where}

    def errors(self, item: CheckItem, scope: Scope) -> list[str]:
        """Problems with ``item``, each prefixed with the setting's path."""
        if self.placement == "dataset" and scope.dataset is None:
            return [f"{scope.path}.check: {self.name} goes in a dataset's checks list"]
        if self.placement == "top" and scope.dataset is not None:
            return [f"{scope.path}.check: {self.name} goes in the top-level checks list"]
        try:
            params = self.compile(item, scope)
        except CheckSettingError as exc:
            where = f"{scope.path}.{exc.field}" if exc.field else scope.path
            return [f"{where}: {exc.message}"]
        try:
            self.rule_cls.Params.model_validate(params)
        except ValidationError as exc:
            problems = []
            for error in exc.errors():
                field = self.locate(error["loc"])
                where = f"{scope.path}.{field}" if field else scope.path
                problems.append(f"{where}: {error['msg']}")
            return problems
        return []

    def item_model(self) -> type[BaseModel]:
        """A model of this check's items, for the configuration JSON Schema."""
        return _item_model(self)


class ElementCheck(CheckDefinition):
    """An item is one entry of the list parameter ``list_param``."""

    def __init__(
        self,
        rule_cls: type[Rule],
        placement: Placement,
        list_param: str,
        *,
        renames: Mapping[str, str] | None = None,
        rule_fields: Mapping[str, str] | None = None,
        shared: Sequence[str] = (),
        inject_dataset: bool = True,
        where_field: str | None = None,
    ) -> None:
        super().__init__(rule_cls, placement, where_field=where_field)
        self.list_param = list_param
        self.renames = dict(renames or {})
        """Item setting -> element field."""
        self.rule_fields = dict(rule_fields or {})
        """Item setting -> rule-level parameter."""
        self.shared = tuple(shared)
        """Rule-level parameters the element overrides; items set them on the element."""
        self.inject_dataset = inject_dataset

    @property
    def element_model(self) -> type[BaseModel]:
        annotation = self.rule_cls.Params.model_fields[self.list_param].annotation
        return annotation.__args__[0]  # type: ignore[union-attr,no-any-return]

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        rule: dict[str, Any] = {}
        element: dict[str, Any] = self.where(item)
        for key, value in item.settings.items():
            if key in self.rule_fields:
                rule[self.rule_fields[key]] = value
            elif self.inject_dataset and key == "dataset":
                raise CheckSettingError(key, "the dataset is the one the check is placed on")
            else:
                element[self.renames.get(key, key)] = value
        if self.inject_dataset:
            element["dataset"] = scope.dataset
        if item.name is not None and "name" in self.element_model.model_fields:
            element["name"] = item.name
        return {**rule, self.list_param: [element]}

    def locate(self, loc: Sequence[Any]) -> str:
        if loc and loc[0] == self.list_param:
            rest = [part for part in loc[1:] if not isinstance(part, int)]
            if rest:
                rest[0] = _reverse(self.renames).get(rest[0], rest[0])
            return ".".join(str(part) for part in rest)
        return _location([_reverse(self.rule_fields).get(loc[0], loc[0]), *loc[1:]] if loc else [])

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        rule_level = {new: params[old] for new, old in self.rule_fields.items() if old in params}
        shared = {name: params[name] for name in self.shared if params.get(name) is not None}
        back = _reverse(self.renames)
        items = []
        for raw in params.get(self.list_param) or ():
            element = dict(raw)
            dataset = element.pop("dataset", None) if self.inject_dataset else None
            item: dict[str, Any] = {"check": self.name}
            if element.get("name") is not None:
                item["name"] = element.pop("name")
            element.pop("name", None)
            if self.where_field is not None and element.get(self.where_field) is not None:
                item["where"] = element.pop(self.where_field)
            item |= {str(back.get(key, key)): value for key, value in element.items()}
            for key, value in shared.items():
                item.setdefault(key, value)
            item |= rule_level
            items.append(Decompiled(dataset, item))
        return items

    def item_fields(self) -> dict[str, FieldSpec]:
        exclude = ["name", *(("dataset",) if self.inject_dataset else ())]
        if self.where_field:
            exclude.append(self.where_field)
        fields = _fields_of(self.element_model, self.renames, exclude)
        rule = self.rule_cls.Params.model_fields
        fields |= {new: _field_copy(rule[old]) for new, old in self.rule_fields.items()}
        return fields


class InlineCheck(CheckDefinition):
    """An item carries an inline ``series:``/``metric:`` mapping plus rule settings."""

    def __init__(
        self,
        rule_cls: type[Rule],
        *,
        inline_key: str = "series",
        list_param: str = "series",
        inline_renames: Mapping[str, str] | None = None,
        into_inline: Mapping[str, str] | None = None,
        rule_fields: Mapping[str, str] | None = None,
        promote: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(rule_cls, "dataset")
        self.inline_key = inline_key
        self.list_param = list_param
        self.inline_renames = dict(inline_renames or {})
        """Inline mapping key -> element field (``entity`` -> ``entity_column``)."""
        self.into_inline = dict(into_inline or {})
        """Item setting -> element field (``expect`` -> ``expected_trend``)."""
        self.rule_fields = dict(rule_fields or {})
        """Item setting -> rule-level parameter, where renamed (``max``)."""
        self.promote = dict(promote or {})
        """Item setting -> element field that overrides it per element (migration only)."""

    @property
    def element_model(self) -> type[BaseModel]:
        annotation = self.rule_cls.Params.model_fields[self.list_param].annotation
        return annotation.__args__[0]  # type: ignore[union-attr,no-any-return]

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        self.where(item)
        settings = item.settings
        inline = settings.pop(self.inline_key, None)
        if isinstance(inline, str):
            raise CheckSettingError(
                self.inline_key,
                f"naming a {self.inline_key} defined elsewhere {LATER_VERSION}; "
                f"give the {self.inline_key} inline as a mapping",
            )
        if not isinstance(inline, Mapping):
            raise CheckSettingError(
                self.inline_key, f"the {self.name} check needs a '{self.inline_key}' mapping"
            )
        element = {self.inline_renames.get(key, key): value for key, value in inline.items()}
        if "dataset" in element:
            raise CheckSettingError(
                f"{self.inline_key}.dataset", "the dataset is the one the check is placed on"
            )
        element["dataset"] = scope.dataset
        for new, old in self.into_inline.items():
            if new in settings:
                element[old] = settings.pop(new)
        rule = {self.rule_fields.get(key, key): value for key, value in settings.items()}
        return {**rule, self.list_param: [element]}

    def locate(self, loc: Sequence[Any]) -> str:
        if loc and loc[0] == self.list_param:
            rest = [part for part in loc[1:] if not isinstance(part, int)]
            if rest and rest[0] in self.into_inline.values():
                return ".".join([_reverse(self.into_inline)[rest[0]], *map(str, rest[1:])])
            if rest:
                rest[0] = _reverse(self.inline_renames).get(rest[0], rest[0])
            return ".".join([self.inline_key, *map(str, rest)])
        return _location([_reverse(self.rule_fields).get(loc[0], loc[0]), *loc[1:]] if loc else [])

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        back_rule = _reverse(self.rule_fields)
        rule_level = {
            back_rule.get(key, key): value
            for key, value in params.items()
            if key != self.list_param
        }
        back_inline = _reverse(self.inline_renames)
        items = []
        for raw in params.get(self.list_param) or ():
            element = dict(raw)
            dataset = element.pop("dataset", None)
            item: dict[str, Any] = {"check": self.name}
            for new, old in self.into_inline.items():
                if element.get(old) is not None:
                    item[new] = element.pop(old)
                element.pop(old, None)
            promoted = {}
            for new, old in self.promote.items():
                if element.get(old) is not None:
                    promoted[new] = element.pop(old)
                element.pop(old, None)
            item[self.inline_key] = {
                back_inline.get(key, key): value for key, value in element.items()
            }
            item |= rule_level | promoted
            items.append(Decompiled(dataset, item))
        return items

    def item_fields(self) -> dict[str, FieldSpec]:
        inline = _inline_model(self)
        fields: dict[str, FieldSpec] = {
            self.inline_key: (inline, Field(description=f"The {self.inline_key} examined."))
        }
        element = self.element_model.model_fields
        fields |= {new: _field_copy(element[old]) for new, old in self.into_inline.items()}
        fields |= _fields_of(self.rule_cls.Params, self.rule_fields, (self.list_param,))
        return fields


def _two(item: CheckItem, scope: Scope) -> tuple[str, str]:
    """The two datasets of a pair check, the reference member first."""
    settings = item.settings
    datasets = settings.get("datasets")
    if not isinstance(datasets, list | tuple) or not all(isinstance(n, str) for n in datasets):
        raise CheckSettingError("datasets", "give the datasets to relate as a list of names")
    if len(datasets) > 2:
        raise CheckSettingError(
            "datasets", f"relating more than two datasets in one check {LATER_VERSION}"
        )
    if len(datasets) != 2 or datasets[0] == datasets[1]:
        raise CheckSettingError("datasets", "give exactly two different datasets")
    strategy = settings.get("strategy", "reference")
    if strategy != "reference":
        raise CheckSettingError("strategy", f"strategy {strategy!r} {LATER_VERSION}")
    reference = settings.get("reference", datasets[0])
    if reference not in datasets:
        raise CheckSettingError("reference", "the reference must be one of the datasets")
    first, second = datasets
    return (first, second) if reference == first else (second, first)


_PAIR_SETTINGS = ("datasets", "strategy", "reference")


class PairCheck(CheckDefinition):
    """A top-level check relating two datasets (``datasets: [reference, other]``)."""

    def __init__(self, rule_cls: type[Rule], list_param: str, *, where_field: str | None = None):
        super().__init__(rule_cls, "top", where_field=where_field)
        self.list_param = list_param

    @property
    def element_model(self) -> type[BaseModel]:
        annotation = self.rule_cls.Params.model_fields[self.list_param].annotation
        return annotation.__args__[0]  # type: ignore[union-attr,no-any-return]

    rule_fields: ClassVar[tuple[str, ...]] = ()
    """Rule-level parameters an item may set."""

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        reference, other = _two(item, scope)
        settings = {k: v for k, v in item.settings.items() if k not in _PAIR_SETTINGS}
        rule = {key: settings.pop(key) for key in self.rule_fields if key in settings}
        element = self.element(reference, other, settings) | self.where(item)
        if item.name is not None and "name" in self.element_model.model_fields:
            element["name"] = item.name
        return {**rule, self.list_param: [element]}

    def locate(self, loc: Sequence[Any]) -> str:
        if loc and loc[0] == self.list_param:
            return self.element_location([p for p in loc[1:] if not isinstance(p, int)])
        return _location(loc)

    def element_location(self, loc: list[Any]) -> str:
        return ".".join(str(part) for part in loc)

    @abstractmethod
    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        """The list entry relating ``reference`` and ``other``."""

    @abstractmethod
    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """``(reference, other, settings)`` of a list entry (for migration)."""

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        rule_level = {key: params[key] for key in self.rule_fields if key in params}
        items = []
        for raw in params.get(self.list_param) or ():
            element = dict(raw)
            item: dict[str, Any] = {"check": self.name}
            if element.get("name") is not None:
                item["name"] = element["name"]
            element.pop("name", None)
            if self.where_field is not None and element.get(self.where_field) is not None:
                item["where"] = element.pop(self.where_field)
            reference, other, settings = self.split(element)
            item["datasets"] = [reference, other]
            item |= settings | rule_level
            items.append(Decompiled(None, item))
        return items

    def pair_fields(self) -> dict[str, FieldSpec]:
        return {
            "datasets": (
                list[str],
                Field(
                    min_length=2,
                    max_length=2,
                    description="The two datasets related; the reference first.",
                ),
            ),
            "strategy": (
                Literal["reference"],
                Field(default="reference", description="How the datasets are paired."),
            ),
            "reference": (
                str | None,
                Field(default=None, description="The authoritative dataset (the first)."),
            ),
        }


class DriftCheck(PairCheck):
    """``drift``: a column's distribution in the other dataset versus the reference."""

    rule_fields = ("correction",)

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        return {"reference": reference, "dataset": other, **settings}

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        return element.pop("reference"), element.pop("dataset"), element

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        fields |= _fields_of(self.element_model, {}, ("name", "dataset", "reference"))
        fields |= _fields_of(self.rule_cls.Params, {}, (self.list_param,))
        return fields


class CompareCheck(PairCheck):
    """``compare``: the other dataset (candidate) against the reference (baseline)."""

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        return {"baseline": reference, "candidate": other, **settings}

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        return element.pop("baseline"), element.pop("candidate"), element

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        exclude = ("name", "baseline", "candidate", "where")
        return fields | _fields_of(self.element_model, {}, exclude)


class _SideSettings(BaseModel):
    """Per-dataset overrides of a two-dataset check."""

    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    """Column to sum on this dataset (overrides the check's ``value``)."""
    keys: list[str] | None = None
    """Key columns on this dataset, in the order of the check's ``keys``."""


class TotalsCheck(PairCheck):
    """``totals_match``: totals per key agree between the two datasets."""

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        sides = settings.pop("sides", None) or {}
        if not isinstance(sides, Mapping) or set(sides) - {reference, other}:
            raise CheckSettingError("sides", "give overrides keyed by the check's datasets")
        value = settings.pop("value", None)
        keys = settings.pop("keys", None)
        for dataset, override in sides.items():
            side_keys = override.get("keys") if isinstance(override, Mapping) else None
            if side_keys is not None and len(side_keys) != len(keys or ()):
                raise CheckSettingError(
                    "sides",
                    f"{dataset}: keys must list {len(keys or ())} column(s), one for each of "
                    "the check's keys",
                )
        element: dict[str, Any] = {
            "left": self.side(reference, value, sides.get(reference)),
            "right": self.side(other, value, sides.get(other)),
            **settings,
        }
        if keys is not None:
            element["on"] = keys
        return element

    @staticmethod
    def side(dataset: str, value: Any, override: Any) -> dict[str, Any]:
        override = dict(override or {})
        side: dict[str, Any] = {"dataset": dataset}
        side_value = override.pop("value", value)
        if side_value is not None:
            side["value"] = side_value
        return side | override

    def element_location(self, loc: list[Any]) -> str:
        if loc and loc[0] == "on":
            return "keys"
        if loc and loc[0] in ("left", "right"):
            return "sides"
        return super().element_location(loc)

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        left, right = dict(element.pop("left")), dict(element.pop("right"))
        settings: dict[str, Any] = {}
        if element.get("on"):
            settings["keys"] = element.pop("on")
        element.pop("on", None)
        sides: dict[str, Any] = {}
        if left.get("value") == right.get("value"):
            if left.get("value") is not None:
                settings["value"] = left.get("value")
            left.pop("value", None)
            right.pop("value", None)
        for side in (left, right):
            override = {key: value for key, value in side.items() if key != "dataset"}
            if override.get("keys") == []:
                override.pop("keys")
            if override:
                sides[side["dataset"]] = override
        if sides:
            settings["sides"] = sides
        return left["dataset"], right["dataset"], settings | element

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        fields["value"] = (
            str | None,
            Field(default=None, description="Column to sum on both datasets; omit to count rows."),
        )
        fields["keys"] = (
            list[str],
            Field(default_factory=list, description="Key columns totals are compared on."),
        )
        fields["sides"] = (
            dict[str, _SideSettings],
            Field(default_factory=dict, description="Per-dataset value or key overrides."),
        )
        exclude = ("name", "left", "right", "on")
        return fields | _fields_of(self.element_model, {}, exclude)


class KeySetsCheck(PairCheck):
    """``key_sets_match``: both datasets hold the same set of keys."""

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        sides = settings.pop("sides", None) or {}
        if not isinstance(sides, Mapping) or set(sides) - {reference, other}:
            raise CheckSettingError("sides", "give overrides keyed by the check's datasets")
        keys = settings.pop("keys", None)
        element = {
            "left": {"dataset": reference, "columns": self.columns(keys, sides.get(reference))},
            "right": {"dataset": other, "columns": self.columns(keys, sides.get(other))},
        }
        return element | settings

    @staticmethod
    def columns(keys: Any, override: Any) -> Any:
        if isinstance(override, Mapping) and "keys" in override:
            return override["keys"]
        return keys if keys is not None else []

    def element_location(self, loc: list[Any]) -> str:
        if loc and loc[0] in ("left", "right"):
            return "keys"
        return super().element_location(loc)

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        left, right = dict(element.pop("left")), dict(element.pop("right"))
        settings: dict[str, Any] = {"keys": list(left["columns"])}
        if list(right["columns"]) != list(left["columns"]):
            settings["sides"] = {right["dataset"]: {"keys": list(right["columns"])}}
        return left["dataset"], right["dataset"], settings | element

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        fields["keys"] = (list[str], Field(description="Key columns compared on both datasets."))
        fields["sides"] = (
            dict[str, _SideSettings],
            Field(default_factory=dict, description="Per-dataset key overrides."),
        )
        return fields


class WholeCheck(CheckDefinition):
    """The item's settings are the rule's parameters (with renames)."""

    def __init__(self, rule_cls: type[Rule], placement: Placement, renames: Mapping[str, str]):
        super().__init__(rule_cls, placement)
        self.renames = dict(renames)

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        self.where(item)
        return {self.renames.get(key, key): value for key, value in item.settings.items()}

    def locate(self, loc: Sequence[Any]) -> str:
        return _location([_reverse(self.renames).get(loc[0], loc[0]), *loc[1:]] if loc else [])

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        if not params:
            return []
        back = _reverse(self.renames)
        item = {"check": self.name} | {back.get(k, k): v for k, v in params.items()}
        return [Decompiled(None, item)]

    def item_fields(self) -> dict[str, FieldSpec]:
        return _fields_of(self.rule_cls.Params, self.renames)


class ExpressionDefinition(CheckDefinition):
    """``expression``: a SQL condition every row of the dataset satisfies."""

    list_driven = True

    def __init__(self) -> None:
        super().__init__(ExpressionRule, "dataset")

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        return expression_check(item).model_dump(mode="python")

    def errors(self, item: CheckItem, scope: Scope) -> list[str]:
        return expression_errors(item, scope)

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        item = {"check": self.name}
        item |= {key: value for key, value in params.items() if value is not None}
        if item.get("null_as") == "pass":
            item.pop("null_as")
        return [Decompiled(None, item)]

    def item_fields(self) -> dict[str, FieldSpec]:
        return _fields_of(ExpressionCheck, {}, ("name", "severity", "description"))

    def item_model(self) -> type[BaseModel]:
        return ExpressionItem


class SqlDefinition(CheckDefinition):
    """``sql``: a query returning offending rows across any datasets."""

    list_driven = True

    def __init__(self) -> None:
        super().__init__(SqlAssertionRule, "top")

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        return sql_assertion(item).model_dump(mode="python")

    def errors(self, item: CheckItem, scope: Scope) -> list[str]:
        return sql_errors(item, scope)

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        item = {"check": self.name}
        item |= {key: value for key, value in params.items() if value is not None}
        return [Decompiled(None, item)]

    def item_fields(self) -> dict[str, FieldSpec]:
        return {"sql": (str, Field(description="Query returning offending rows."))}

    def item_model(self) -> type[BaseModel]:
        return SqlItem


class ContractDefinition(CheckDefinition):
    """A schema contract: it runs from ``schema``, never from a check item."""

    configurable = False

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        raise CheckSettingError(
            "check", f"{self.name} runs from the dataset's schema; it is not a check item"
        )

    def errors(self, item: CheckItem, scope: Scope) -> list[str]:
        return [
            f"{scope.path}.check: {self.name} runs from the dataset's schema and "
            "needs no check item"
        ]

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        return []

    def item_fields(self) -> dict[str, FieldSpec]:
        return {}


class PluginDefinition(CheckDefinition):
    """A plugin check: the item's settings are the plugin rule's parameters."""

    def __init__(self, rule_cls: type[Rule]) -> None:
        super().__init__(rule_cls, "dataset")

    def errors(self, item: CheckItem, scope: Scope) -> list[str]:
        # Plugins may be placed on a dataset or at the top level.
        try:
            params = self.compile(item, scope)
            self.rule_cls.Params.model_validate(params)
        except CheckSettingError as exc:
            return [f"{scope.path}.{exc.field}: {exc.message}"]
        except ValidationError as exc:
            return [
                f"{scope.path}.{_location(error['loc'])}: {error['msg']}"
                if error["loc"]
                else f"{scope.path}: {error['msg']}"
                for error in exc.errors()
            ]
        return []

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        self.where(item)
        params = item.settings
        if scope.dataset is not None and "dataset" in self.rule_cls.Params.model_fields:
            params.setdefault("dataset", scope.dataset)
        return params

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        return [Decompiled(None, {"check": self.name, **params})]

    def item_fields(self) -> dict[str, FieldSpec]:
        return _fields_of(self.rule_cls.Params, {})


def _definitions() -> Iterator[CheckDefinition]:
    yield ExpressionDefinition()
    yield SqlDefinition()
    yield ElementCheck(FreshnessRule, "dataset", "checks", rule_fields={"now": "now"})
    temporal: tuple[tuple[type[Rule], dict[str, str], dict[str, str]], ...] = (
        (StepChangeRule, {"max": "max_change_rate"}, {}),
        (TrendDirectionRule, {}, {"expect": "expected_trend"}),
        (TrendOutlierRule, {}, {}),
        (MissingPeriodRule, {"expected": "expected_periods"}, {}),
        (StaleValueRule, {}, {}),
        (CumulativeRule, {}, {}),
        (SeasonalAnomalyRule, {}, {}),
        (LevelShiftRule, {}, {}),
    )
    for rule_cls, rule_fields, into_inline in temporal:
        yield InlineCheck(
            rule_cls,
            rule_fields=rule_fields,
            into_inline=into_inline,
            promote={"max": "max_change_rate"} if rule_cls is StepChangeRule else None,
        )
    yield ElementCheck(
        ColumnOutlierRule,
        "dataset",
        "targets",
        renames={"column": "value"},
        rule_fields={"min_group_size": "min_group_size"},
        shared=("method", "threshold"),
    )
    yield DriftCheck(DistributionDriftRule, "comparisons")
    yield TotalsCheck(TotalsReconcileRule, "reconciliations")
    yield ElementCheck(SharesSumRule, "dataset", "shares")
    yield ElementCheck(KeyGridRule, "dataset", "grids")
    yield KeySetsCheck(KeySetsMatchRule, "pairs")
    yield ElementCheck(ConsistentBatchRule, "top", "batches", inject_dataset=False)
    yield ElementCheck(CoordinateRule, "dataset", "points")
    for rule_cls in (EntityOutlierRule, GroupComparisonRule):
        yield InlineCheck(
            rule_cls,
            inline_key="metric",
            list_param="metrics",
            inline_renames={"entity": "entity_column"},
        )
    yield WholeCheck(
        CoverageRule,
        "top",
        {"entity": "entity_column", "reference": "reference_dataset"},
    )
    yield ElementCheck(
        BaselineComparisonRule,
        "dataset",
        "baselines",
        rule_fields={"max_differences": "max_differences", "require_manifest": "require_manifest"},
        shared=("rtol", "atol"),
    )
    yield CompareCheck(DatasetComparisonRule, "comparisons", where_field="where")
    for rule_cls in (
        RequiredDatasetsRule,
        SchemaConformanceRule,
        ColumnCheckRule,
        DuplicateKeyRule,
        ReferentialIntegrityRule,
        ApplicabilityRule,
    ):
        yield ContractDefinition(rule_cls, "dataset")


CATALOG: dict[str, CheckDefinition] = {definition.name: definition for definition in _definitions()}
"""Every built-in check, by check name."""

BY_RULE: dict[str, CheckDefinition] = {
    definition.rule_id: definition for definition in CATALOG.values()
}
"""Every built-in check, by rule code."""

CONTRACT_RULES: tuple[str, ...] = tuple(
    sorted(definition.rule_id for definition in CATALOG.values() if not definition.configurable)
)
"""Rule codes of the schema contracts, which always run."""

assert all(rule_id in INPUT_RULES for rule_id in CONTRACT_RULES)


def definition_for(rule_cls: type[Rule]) -> CheckDefinition:
    """The catalog definition of a built-in rule, or a plugin definition."""
    builtin = BY_RULE.get(rule_cls.rule_id)
    if builtin is not None and builtin.rule_cls is rule_cls:
        return builtin
    return PluginDefinition(rule_cls)


_COMMON_ITEM_FIELDS: dict[str, Any] = {
    name: (field.annotation, copy.copy(field))
    for name, field in CheckItem.model_fields.items()
    if name != "check"
}


@functools.cache
def _item_model(definition: CheckDefinition) -> type[BaseModel]:
    fields = dict(definition.item_fields())
    for name in _COMMON_ITEM_FIELDS:
        fields.pop(name, None)
    model_name = "Check_" + "".join(ch if ch.isalnum() else "_" for ch in definition.name)
    return create_model(  # type: ignore[call-overload,no-any-return]
        model_name,
        __config__=ConfigDict(extra="forbid", title=definition.name),
        __doc__=f"{definition.name} ({definition.rule_id}). {definition.summary}",
        check=(
            Literal[definition.name],  # type: ignore[valid-type]
            Field(description="Kind of check."),
        ),
        **_COMMON_ITEM_FIELDS,
        **fields,
    )


@functools.cache
def _inline_model(definition: InlineCheck) -> type[BaseModel]:
    fields = _fields_of(definition.element_model, definition.inline_renames, ("dataset",))
    for name in definition.into_inline:
        fields.pop(name, None)
    for name in definition.into_inline.values():
        fields.pop(name, None)
    for name in definition.promote.values():
        fields.pop(name, None)
    model_name = f"Inline_{definition.inline_key}_" + definition.name
    return create_model(  # type: ignore[call-overload,no-any-return]
        model_name, __config__=ConfigDict(extra="forbid"), **fields
    )


__all__ = [
    "BY_RULE",
    "CATALOG",
    "CONTRACT_RULES",
    "LATER_VERSION",
    "CheckDefinition",
    "Decompiled",
    "PluginDefinition",
    "definition_for",
]
