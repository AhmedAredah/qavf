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
* :class:`PairCheck` subclasses: top-level checks relating ``datasets: [a, b, ...]``,
  paired by a ``strategy`` (:mod:`qavf.compare.pairing`).
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
from pydantic_core import InitErrorDetails

from qavf.checks.items import (
    FILTER_SETTING,
    CheckItem,
    ExpressionItem,
    SqlItem,
    settings_required_when_enabled,
)
from qavf.checks.placement import (
    BLOCK_SECTIONS,
    CheckSettingError,
    Scope,
    expression_errors,
    placed_items,
    sql_errors,
)
from qavf.compare.consensus import ROWS
from qavf.compare.pairing import (
    CONSENSUS,
    DEFAULT_MAX_PAIRS,
    POOLED,
    ConsensusSettings,
    DatasetGroup,
    GroupPlan,
    PairingError,
    PairRole,
    Strategy,
    pair_label,
)
from qavf.engine.derived import DerivedParams, derived_name
from qavf.engine.derived_names import RowFilter
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
from qavf.settings import ExpressionCheck, QAVFSettings, expression_check, sql_assertion

Placement = Literal["dataset", "top"]
FieldSpec = tuple[Any, Any]
"""A field's annotation and its ``FieldInfo``, as ``create_model`` takes them."""


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
    plugin_components: ClassVar[bool] = False
    """True when settings may name components from installed plugins (comparison
    levels, steps, statistical tests), which load only when the run is planned."""

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

    @property
    def supports_where(self) -> bool:
        """Whether the check's items take the common ``where`` setting."""
        return self.where_field is not None

    @property
    def filters_rows(self) -> bool:
        """Whether a dataset item's ``where`` selects the rows the check reads through a
        hidden view of the dataset (the check does not filter rows itself)."""
        return self.configurable and self.placement == "dataset" and not self.supports_where

    @property
    def takes_where(self) -> bool:
        """Whether the check's items take ``where`` at all (natively or filtered)."""
        return self.supports_where or self.filters_rows

    def where(self, item: CheckItem) -> dict[str, Any]:
        """The element field receiving the common ``where`` setting."""
        if item.where is None:
            return {}
        if self.where_field is None:
            raise CheckSettingError("where", unsupported_where(self.name, top_level=True))
        return {self.where_field: item.where}

    def rows(self, item: CheckItem, scope: Scope) -> RowFilter | None:
        """The rows a filtered dataset item assesses (``None``: the check reads its
        dataset, or applies ``where`` itself).

        Raises:
            CheckSettingError: for a ``min_applicable_rows`` that is not a count, or
                given without ``where``.
        """
        if scope.dataset is None or not self.filters_rows:
            return None
        minimum = item.settings.get(FILTER_SETTING, 0)
        if item.where is None:
            if FILTER_SETTING in item.settings:
                raise CheckSettingError(FILTER_SETTING, f"'{FILTER_SETTING}' goes with 'where'")
            return None
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise CheckSettingError(FILTER_SETTING, "must be a whole number of rows, 0 or more")
        return RowFilter(scope.dataset, item.where, minimum)

    def compile_item(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        """The rule parameters for ``item``: a filtered dataset item's check reads the
        hidden view of its rows (:meth:`rows`), without ``where`` of its own."""
        rows = self.rows(item, scope)
        if rows is None:
            return self.compile(item, scope)
        return self.compile(item.without_filter(), scope.filtered(rows.view))

    def errors(self, item: CheckItem, scope: Scope, *, parameters: bool = True) -> list[str]:
        """Problems with ``item``, each prefixed with the setting's path.

        With ``parameters=False`` only the item's own structure is checked, not the
        rule parameters it compiles to.
        """
        if self.placement == "dataset" and scope.dataset is None:
            return [f"{scope.path}.check: {self.name} goes in a dataset's checks list"]
        if self.placement == "top" and scope.dataset is not None:
            return [f"{scope.path}.check: {self.name} goes in the top-level checks list"]
        try:
            params = self.compile_item(item, scope)
        except CheckSettingError as exc:
            where = f"{scope.path}.{exc.field}" if exc.field else scope.path
            return [f"{where}: {exc.message}"]
        if not parameters:
            return []
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
            element["dataset"] = scope.reads
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
            inline = scope.resolve(self.inline_key, inline)
        if not isinstance(inline, Mapping):
            raise CheckSettingError(
                self.inline_key,
                f"the {self.name} check needs a '{self.inline_key}': the name of one the "
                f"dataset defines, or a mapping",
            )
        element: dict[str, Any] = {
            self.inline_renames.get(str(key), str(key)): value for key, value in inline.items()
        }
        self.translate(element)
        if "dataset" in element:
            raise CheckSettingError(
                f"{self.inline_key}.dataset", "the dataset is the one the check is placed on"
            )
        element["dataset"] = scope.reads
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
            item[self.inline_key] = self.untranslate(
                {str(back_inline.get(key, key)): value for key, value in element.items()}
            )
            item |= rule_level | promoted
            items.append(Decompiled(dataset, item))
        return items

    def translate(self, element: dict[str, Any]) -> None:
        """Turn shorthand values of an inline mapping into rule element fields."""

    def untranslate(self, inline: dict[str, Any]) -> dict[str, Any]:
        """The shorthand form of an inline mapping recovered from rule parameters."""
        return inline

    def inline_fields(self) -> dict[str, FieldSpec]:
        """The settings of the inline mapping, for the configuration JSON Schema."""
        return _fields_of(self.element_model, self.inline_renames, ("dataset",))

    def item_fields(self) -> dict[str, FieldSpec]:
        inline = _inline_model(self)
        fields: dict[str, FieldSpec] = {
            self.inline_key: (
                inline | str,
                Field(
                    description=f"The {self.inline_key} examined: the name of one the dataset "
                    f"defines under {BLOCK_SECTIONS[self.inline_key]} (``other.name`` for another "
                    "dataset's), or the definition itself."
                ),
            )
        }
        element = self.element_model.model_fields
        fields |= {new: _field_copy(element[old]) for new, old in self.into_inline.items()}
        fields |= _fields_of(self.rule_cls.Params, self.rule_fields, (self.list_param,))
        return fields


class MetricCheck(InlineCheck):
    """A spatial check: its ``metric`` may name its normalizer as ``per: dataset.column``."""

    def __init__(self, rule_cls: type[Rule]) -> None:
        super().__init__(
            rule_cls,
            inline_key="metric",
            list_param="metrics",
            inline_renames={"entity": "entity_column", "per": "normalize_by"},
        )

    def translate(self, element: dict[str, Any]) -> None:
        per = element.get("normalize_by")
        if isinstance(per, str):
            dataset, _, column = per.rpartition(".")
            if not dataset or not column:
                raise CheckSettingError(
                    "metric.per", f"write the normalizer as <dataset>.<column>, got {per!r}"
                )
            element["normalize_by"] = {"dataset": dataset, "value": column}

    def untranslate(self, inline: dict[str, Any]) -> dict[str, Any]:
        per = inline.get("per")
        if isinstance(per, Mapping) and set(per) == {"dataset", "value"}:
            inline["per"] = f"{per['dataset']}.{per['value']}"
        return inline

    def inline_fields(self) -> dict[str, FieldSpec]:
        fields = super().inline_fields()
        annotation, info = fields["per"]
        info.description = (
            "Normalizer: ``<dataset>.<column>`` (e.g. ``counts.aadt``), or a mapping with "
            "``dataset``, ``value``, ``on`` and ``aggregate``."
        )
        fields["per"] = (annotation | str, info)
        return fields


def _group(item: CheckItem) -> DatasetGroup:
    """The members of a check relating datasets and how they pair."""
    try:
        return DatasetGroup.of(item.settings)
    except PairingError as exc:
        raise CheckSettingError(exc.field, exc.message) from None


_PAIR_SETTINGS = ("datasets", "strategy", "reference", "max_pairs", "consensus")
_CONSENSUS_SIDE_SETTINGS = (
    "columns",
    "baseline_where",
    "candidate_where",
    "cast",
    "units",
    "recode",
    "steps",
)
"""Comparison settings that align one side only, which a consensus has no use for."""


@dataclass(frozen=True, slots=True)
class PairLayout:
    """A check relating many datasets: one list entry per pair, the group plan and the
    datasets derived from the members (consensus, pooled data)."""

    elements: list[dict[str, Any]]
    plan: GroupPlan
    derived: dict[str, DerivedParams]


class PairCheck(CheckDefinition):
    """A top-level check relating datasets (``datasets: [reference, other, ...]``).

    Two datasets with the default strategy compile to exactly one list entry, as
    before; more datasets, or another strategy, to one entry per pair and a
    :class:`~qavf.compare.pairing.GroupPlan` naming which member each pair is about.
    """

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
        group = _group(item)
        settings = {k: v for k, v in item.settings.items() if k not in _PAIR_SETTINGS}
        rule = {key: settings.pop(key) for key in self.rule_fields if key in settings}
        if group.plain:
            ((reference, other),) = group.pairs()
            element = self.element(reference, other, settings) | self.where(item)
            if item.name is not None and "name" in self.element_model.model_fields:
                element["name"] = item.name
            return {**rule, self.list_param: [element]}
        layout = self.layout(item, scope, group, settings)
        return {**rule, self.list_param: layout.elements, "group": layout.plan.model_dump()}

    def derived(self, item: CheckItem, scope: Scope) -> dict[str, DerivedParams]:
        """The datasets the item derives from its members (none for a plain pair)."""
        group = _group(item)
        if group.plain:
            return {}
        settings = {k: v for k, v in item.settings.items() if k not in _PAIR_SETTINGS}
        for key in self.rule_fields:
            settings.pop(key, None)
        return self.layout(item, scope, group, settings).derived

    def layout(
        self, item: CheckItem, scope: Scope, group: DatasetGroup, settings: dict[str, Any]
    ) -> PairLayout:
        """One list entry per pair of ``group``, each labelled ``<name>: <a> vs <b>``."""
        if group.strategy == CONSENSUS:
            return self.consensus_layout(item, scope, group, settings)
        name = item.effective_name(group.members)
        elements, roles = [], []
        for reference, other in group.pairs():
            label = pair_label(name, reference, other)
            pair_settings = self.pair_settings(settings, group, (reference, other))
            element = self.element(reference, other, pair_settings) | self.where(item)
            elements.append(element | {"name": label})
            members = group.concerned(reference, other)
            roles.append(PairRole(label=label, reference=reference, other=other, members=members))
        plan = GroupPlan(
            name=name,
            check=self.name,
            strategy=group.strategy,  # type: ignore[arg-type]
            members=group.members,
            reference=group.reference if group.strategy == "reference" else None,
            pairs=tuple(roles),
        )
        return PairLayout(elements, plan, {})

    def consensus_layout(
        self, item: CheckItem, scope: Scope, group: DatasetGroup, settings: dict[str, Any]
    ) -> PairLayout:
        """Each member against the consensus of all members, matched on the keys."""
        name = item.effective_name(group.members)
        keys, consensus = self.consensus_source(settings)
        full = derived_name("full", scope.path)
        agreed = derived_name("consensus", scope.path)
        derived = {
            full: DerivedParams(
                role="consensus_full",
                members=group.members,
                keys=keys,
                reference=group.reference,
                consensus=group.consensus or ConsensusSettings(),
                **consensus,
            ),
            agreed: DerivedParams(role="consensus", full=full, keys=keys),
        }
        elements, roles = [], []
        for member in group.members:
            restricted = derived_name("member", scope.path, member)
            derived[restricted] = DerivedParams(role="member", full=full, member=member, keys=keys)
            label = pair_label(name, CONSENSUS, member)
            pair_settings = self.pair_settings(settings, group, ())
            element = self.consensus_element(agreed, restricted, pair_settings, consensus)
            elements.append(element | self.where(item) | {"name": label})
            roles.append(
                PairRole(label=label, reference=CONSENSUS, other=member, members=(member,))
            )
        plan = GroupPlan(
            name=name,
            check=self.name,
            strategy="consensus",
            members=group.members,
            reference=group.reference,
            pairs=tuple(roles),
            consensus=group.consensus,
            consensus_dataset=full,
            keys=keys,
        )
        return PairLayout(elements, plan, derived)

    def consensus_source(self, settings: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
        """The consensus keys and how the consensus combines the members' rows."""
        keys = settings.get("keys")
        if not isinstance(keys, list | tuple) or not keys:
            raise CheckSettingError("keys", "strategy consensus matches members on keys; give keys")
        return tuple(keys), {}

    def consensus_element(
        self, agreed: str, member: str, settings: dict[str, Any], consensus: dict[str, Any]
    ) -> dict[str, Any]:
        """The list entry relating the consensus and one member."""
        return self.element(agreed, member, settings)

    def pair_settings(
        self, settings: dict[str, Any], group: DatasetGroup, pair: tuple[str, ...]
    ) -> dict[str, Any]:
        """The settings of one pair's list entry (a copy: entries are built from them)."""
        return copy.deepcopy(settings)

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
                    description="The datasets related; the reference first (by default).",
                ),
            ),
            "strategy": (
                Strategy,
                Field(
                    default="reference",
                    description=(
                        "How the datasets are paired: the reference with each other "
                        "(reference), each with the next (sequential), every pair "
                        "(all_pairs) or each with the consensus of all (consensus)."
                    ),
                ),
            ),
            "reference": (
                str | None,
                Field(
                    default=None,
                    description=(
                        "The authoritative dataset (default: the first); with consensus, the "
                        "dataset whose value breaks ties under ties: reference."
                    ),
                ),
            ),
            "max_pairs": (
                int,
                Field(
                    default=DEFAULT_MAX_PAIRS,
                    ge=1,
                    description="all_pairs refuses more pairs than this.",
                ),
            ),
            "consensus": (
                ConsensusSettings | None,
                Field(default=None, description="How the consensus is built (strategy consensus)."),
            ),
        }


class DriftCheck(PairCheck):
    """``drift``: a column's distribution in the other dataset versus the reference."""

    plugin_components = True

    rule_fields = ("correction",)

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        return {"reference": reference, "dataset": other, **settings}

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        return element.pop("reference"), element.pop("dataset"), element

    def consensus_layout(
        self, item: CheckItem, scope: Scope, group: DatasetGroup, settings: dict[str, Any]
    ) -> PairLayout:
        """Each member's distribution against the pooled data of all members."""
        if "consensus" in item.settings:
            raise CheckSettingError(
                "consensus",
                "drift compares each member with the pooled data of all members; consensus "
                "settings apply to checks matched on keys",
            )
        name = item.effective_name(group.members)
        pooled = derived_name("pooled", scope.path)
        derived = {pooled: DerivedParams(role="pooled", members=group.members)}
        elements, roles = [], []
        for member in group.members:
            label = pair_label(name, POOLED, member)
            element = self.element(pooled, member, copy.deepcopy(settings)) | self.where(item)
            elements.append(element | {"name": label})
            roles.append(PairRole(label=label, reference=POOLED, other=member, members=(member,)))
        plan = GroupPlan(
            name=name,
            check=self.name,
            strategy="consensus",
            members=group.members,
            pairs=tuple(roles),
        )
        return PairLayout(elements, plan, derived)

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        fields |= _fields_of(self.element_model, {}, ("name", "dataset", "reference"))
        fields |= _fields_of(self.rule_cls.Params, {}, (self.list_param, "group"))
        return fields


class CompareCheck(PairCheck):
    """``compare``: the other dataset (candidate) against the reference (baseline).

    With more than two members, the ``coverage``, ``structure`` and ``distribution``
    levels assess all members together (each member against the pooled data, for
    ``distribution``) instead of pair by pair.
    """

    plugin_components = True

    def element(self, reference: str, other: str, settings: dict[str, Any]) -> dict[str, Any]:
        return {"baseline": reference, "candidate": other, **settings}

    def split(self, element: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        return element.pop("baseline"), element.pop("candidate"), element

    def consensus_source(self, settings: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
        if aligned := [key for key in _CONSENSUS_SIDE_SETTINGS if settings.get(key)]:
            raise CheckSettingError(
                aligned[0],
                f"with strategy consensus every member is aligned the same way; "
                f"{aligned[0]!r} aligns one side only",
            )
        keys, consensus = super().consensus_source(settings)
        ignored = settings.get("ignore_columns") or ()
        return keys, consensus | {"ignore": tuple(ignored)}

    def layout(
        self, item: CheckItem, scope: Scope, group: DatasetGroup, settings: dict[str, Any]
    ) -> PairLayout:
        layout = super().layout(item, scope, group, settings)
        if len(group.members) <= 2:
            return layout
        return WholeGroupLevels(self, item, scope, group, settings).apply(layout)

    def item_fields(self) -> dict[str, FieldSpec]:
        fields = self.pair_fields()
        exclude = ("name", "baseline", "candidate", "where")
        return fields | _fields_of(self.element_model, {}, exclude)


class WholeGroupLevels:
    """``compare`` with more than two members: the levels assessed over all members.

    ``coverage`` and ``structure`` run once over every member (in the rule, from the
    group plan); ``distribution`` compares each member with the pooled data of all
    members. Pairs keep the other levels; a pair left with none is dropped.
    """

    LEVELS = ("coverage", "structure", "distribution")

    def __init__(
        self,
        check: CompareCheck,
        item: CheckItem,
        scope: Scope,
        group: DatasetGroup,
        settings: dict[str, Any],
    ) -> None:
        self.check = check
        self.item = item
        self.scope = scope
        self.group = group
        self.settings = settings

    def configured(self) -> list[tuple[Any, Any]] | None:
        """``(raw entry, LevelConfig)`` of each configured (or default) level."""
        from qavf.compare.spec import DEFAULT_LEVELS, LevelConfig

        raw = self.settings.get("levels")
        if raw is None:
            raw = list(DEFAULT_LEVELS if self.settings.get("keys") else DEFAULT_LEVELS[:1])
        if not isinstance(raw, list | tuple):
            return None
        try:
            return [(entry, LevelConfig.model_validate(entry)) for entry in raw]
        except (ValidationError, ValueError, TypeError):
            return None  # reported by the pairs' own validation

    def apply(self, layout: PairLayout) -> PairLayout:
        configured = self.configured()
        if configured is None:
            return layout
        whole = {config.level: config for _, config in configured if config.level in self.LEVELS}
        if not whole:
            return layout
        kept = [entry for entry, config in configured if config.level not in self.LEVELS]
        elements, roles = [], []
        if kept:
            for element, role in zip(layout.elements, layout.plan.pairs, strict=True):
                elements.append(element | {"levels": copy.deepcopy(kept)})
                roles.append(role)
        derived = dict(layout.derived)
        if not kept:
            derived = {}  # no pair is compared: the consensus is not needed
        if "distribution" in whole:
            self.pooled(whole["distribution"], configured, elements, roles, derived)
        coverage, structure = whole.get("coverage"), whole.get("structure")
        plan = layout.plan.model_copy(
            update={
                "pairs": tuple(roles),
                "consensus_dataset": layout.plan.consensus_dataset if kept else None,
                "keys": tuple(self.settings.get("keys") or ()),
                "coverage": coverage.label if coverage else None,
                "coverage_severity": coverage.severity if coverage else None,
                "structure": structure.label if structure else None,
                "structure_severity": structure.severity if structure else None,
                "where": self.item.where,
                "ignore_columns": tuple(self.settings.get("ignore_columns") or ()),
            }
        )
        return PairLayout(elements, GroupPlan.model_validate(plan.model_dump()), derived)

    def pooled(
        self,
        config: Any,
        configured: list[tuple[Any, Any]],
        elements: list[dict[str, Any]],
        roles: list[PairRole],
        derived: dict[str, DerivedParams],
    ) -> None:
        """Each member's distribution against the pooled data of all members."""
        entry = next(raw for raw, level in configured if level is config)
        pooled = derived_name("pooled", self.scope.path)
        derived[pooled] = DerivedParams(role="pooled", members=self.group.members)
        name = self.item.effective_name(self.group.members)
        settings = {
            key: value
            for key, value in self.settings.items()
            if key not in _CONSENSUS_SIDE_SETTINGS
        }
        for member in self.group.members:
            label = pair_label(name, POOLED, member)
            element = self.check.element(pooled, member, copy.deepcopy(settings))
            element |= self.check.where(self.item)
            elements.append(element | {"levels": [copy.deepcopy(entry)], "name": label})
            roles.append(
                PairRole(
                    label=label,
                    reference=POOLED,
                    other=member,
                    members=(member,),
                    level=config.label,
                )
            )


class _SideSettings(BaseModel):
    """Per-dataset overrides of a two-dataset check."""

    model_config = ConfigDict(extra="forbid")

    value: str | None = None
    """Column to sum on this dataset (overrides the check's ``value``)."""
    keys: list[str] | None = None
    """Key columns on this dataset, in the order of the check's ``keys``."""


def _sides_of(settings: dict[str, Any], group: DatasetGroup, pair: tuple[str, ...]) -> Any:
    """The ``sides`` overrides of one pair, checked against every member."""
    sides = settings.get("sides")
    if not sides:
        return sides
    if not isinstance(sides, Mapping) or set(sides) - set(group.members):
        raise CheckSettingError("sides", "give overrides keyed by the check's datasets")
    if group.strategy == CONSENSUS:
        raise CheckSettingError(
            "sides",
            "with strategy consensus every member is read the same way; per-dataset "
            "overrides do not apply",
        )
    return {name: override for name, override in sides.items() if name in pair}


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

    def pair_settings(
        self, settings: dict[str, Any], group: DatasetGroup, pair: tuple[str, ...]
    ) -> dict[str, Any]:
        result = copy.deepcopy(settings)
        if "sides" in result:
            result["sides"] = _sides_of(result, group, pair)
        return result

    def consensus_source(self, settings: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
        keys, consensus = super().consensus_source(settings)
        value = settings.get("value")
        return keys, consensus | (
            {"total": "sum", "value": value, "columns": (value,)}
            if value is not None
            else {"total": "rows", "columns": (ROWS,)}
        )

    def consensus_element(
        self, agreed: str, member: str, settings: dict[str, Any], consensus: dict[str, Any]
    ) -> dict[str, Any]:
        element = self.element(agreed, member, settings)
        element["left"]["value"] = consensus.get("value") or ROWS
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

    def pair_settings(
        self, settings: dict[str, Any], group: DatasetGroup, pair: tuple[str, ...]
    ) -> dict[str, Any]:
        result = copy.deepcopy(settings)
        if "sides" in result:
            result["sides"] = _sides_of(result, group, pair)
        return result

    def consensus_source(self, settings: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
        keys, consensus = super().consensus_source(settings)
        return keys, consensus | {"columns": ()}

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

    @property
    def supports_where(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(ExpressionRule, "dataset")

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        return expression_check(item).model_dump(mode="python")

    def errors(self, item: CheckItem, scope: Scope, *, parameters: bool = True) -> list[str]:
        return expression_errors(item, scope)

    def decompile(self, params: Mapping[str, Any]) -> list[Decompiled]:
        item = {"check": self.name}
        item |= {key: value for key, value in params.items() if value is not None}
        if item.get("null_as") == "pass":
            item.pop("null_as")
        if not item.get("min_applicable_rows"):
            item.pop("min_applicable_rows", None)
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

    def errors(self, item: CheckItem, scope: Scope, *, parameters: bool = True) -> list[str]:
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

    def errors(self, item: CheckItem, scope: Scope, *, parameters: bool = True) -> list[str]:
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

    def errors(self, item: CheckItem, scope: Scope, *, parameters: bool = True) -> list[str]:
        # Plugins may be placed on a dataset or at the top level.
        try:
            params = self.compile_item(item, scope)
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

    @property
    def supports_where(self) -> bool:
        """A plugin takes ``where`` when its parameters declare it."""
        return "where" in self.rule_cls.Params.model_fields

    @property
    def filters_rows(self) -> bool:
        """A plugin placed on a dataset reads the rows of its ``where`` through a hidden
        view when its parameters name the dataset but declare no ``where`` (nor a
        ``min_applicable_rows`` of their own)."""
        fields = self.rule_cls.Params.model_fields
        return not self.supports_where and "dataset" in fields and FILTER_SETTING not in fields

    def compile(self, item: CheckItem, scope: Scope) -> dict[str, Any]:
        params = copy.deepcopy(item.settings)
        if item.where is not None:
            if not self.supports_where:
                raise CheckSettingError(
                    "where", unsupported_where(self.name, top_level=scope.dataset is None)
                )
            params["where"] = item.where
        if scope.dataset is not None and "dataset" in self.rule_cls.Params.model_fields:
            named = params.get("dataset", scope.dataset)
            if named != scope.dataset:
                raise CheckSettingError(
                    "dataset",
                    f"the item is placed on dataset {scope.dataset!r} but names dataset "
                    f"{named!r}; place it on {named!r} or leave 'dataset' out",
                )
            params["dataset"] = scope.reads
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
        yield MetricCheck(rule_cls)
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


def derived_datasets(settings: QAVFSettings) -> dict[str, DerivedParams]:
    """The datasets checks relating many datasets derive from their members.

    A consensus (with the members restricted to the keys that have one) and the
    pooled data of all members are datasets of the run like views; they are named
    after the item's place (:func:`~qavf.engine.derived.derived_name`). Items with
    invalid settings derive nothing; their errors are reported where items are
    validated.
    """
    derived: dict[str, DerivedParams] = {}
    for placed in placed_items(settings):
        definition = CATALOG.get(placed.item.check)
        if not isinstance(definition, PairCheck) or placed.scope.dataset is not None:
            continue
        try:
            derived |= definition.derived(placed.item, placed.scope)
        except CheckSettingError:
            continue
    return derived


_MIN_APPLICABLE_ROWS: FieldSpec = (
    int,
    Field(
        default=0,
        ge=0,
        strict=True,
        description="With ``where``: fewer matching rows than this make the check not "
        "assessed (an empty filtered population is never assessed).",
    ),
)
"""The ``min_applicable_rows`` setting of the checks filtered through a hidden view."""


def unsupported_where(check: str, *, top_level: bool) -> str:
    """Why a check item cannot take ``where``, and what selects its rows instead."""
    if top_level:
        return (
            f"'where' is not supported by the top-level {check} check: one condition is "
            "ambiguous across the datasets it relates. Name a view of each dataset instead "
            "(datasets.<dataset>.views.<view>, named as <dataset>.<view> in 'datasets')"
        )
    return (
        f"'where' is not supported by the {check} check, whose parameters declare neither "
        "'where' nor the 'dataset' it reads; check a dataset of the filtered rows instead"
    )


@functools.cache
def _item_model(definition: CheckDefinition) -> type[BaseModel]:
    fields = dict(definition.item_fields())
    for name in _COMMON_ITEM_FIELDS:
        fields.pop(name, None)
    common = {
        name: field
        for name, field in _COMMON_ITEM_FIELDS.items()
        if name != "where" or definition.takes_where
    }
    if definition.filters_rows:
        common[FILTER_SETTING] = _MIN_APPLICABLE_ROWS
    model_name = "Check_" + "".join(ch if ch.isalnum() else "_" for ch in definition.name)
    return create_model(  # type: ignore[call-overload,no-any-return]
        model_name,
        __config__=ConfigDict(
            extra="forbid",
            title=definition.name,
            json_schema_extra=settings_required_when_enabled,
        ),
        __doc__=f"{definition.name} ({definition.rule_id}). {definition.summary}",
        check=(
            Literal[definition.name],  # type: ignore[valid-type]
            Field(description="Kind of check."),
        ),
        **common,
        **fields,
    )


@functools.cache
def _inline_model(definition: InlineCheck) -> type[BaseModel]:
    fields = definition.inline_fields()
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


class NamedBlockCheck:
    """Validates a dataset's named ``series`` or ``metrics`` definition when the
    configuration loads, whether or not a check uses it.

    A definition is checked as a check naming it would read it: its keys renamed to
    the element fields (``entity`` -> ``entity_column``), shorthand translated
    (``per: counts.aadt``) and the dataset added, then validated against the
    element model the series or metric checks share. Errors keep their path within
    the definition, under the setting names as written.
    """

    def __init__(self, definition: InlineCheck) -> None:
        self.definition = definition

    @classmethod
    def for_section(cls, section: str) -> NamedBlockCheck | None:
        """The check for definitions under ``series`` or ``metrics``."""
        for definition in CATALOG.values():
            if (
                isinstance(definition, InlineCheck)
                and BLOCK_SECTIONS.get(definition.inline_key) == section
            ):
                return cls(definition)
        return None

    @classmethod
    def validate(cls, sections: Mapping[str, Mapping[str, Any]]) -> None:
        """Raise one ``pydantic.ValidationError`` listing every invalid definition.

        ``sections`` maps ``series``/``metrics`` to one dataset's definitions.
        """
        errors: list[InitErrorDetails] = []
        for section, blocks in sections.items():
            check = cls.for_section(section) if blocks else None
            if check is not None:
                for name, block in blocks.items():
                    errors.extend(check.errors(name, block))
        if errors:
            raise ValidationError.from_exception_data("DatasetSpec", errors)

    def errors(self, name: str, block: Any) -> list[InitErrorDetails]:
        """The problems of one definition, located at ``<section>.<name>.<setting>``."""
        definition = self.definition
        section = BLOCK_SECTIONS[definition.inline_key]
        if not isinstance(block, Mapping):
            return [_problem((section, name), f"define the {definition.inline_key} as a mapping")]
        element: dict[str, Any] = {
            definition.inline_renames.get(str(key), str(key)): value for key, value in block.items()
        }
        problems: list[InitErrorDetails] = []
        try:
            definition.translate(element)
        except CheckSettingError as exc:
            setting = (exc.field or "").rpartition(".")[2]
            problems.append(_problem((section, name, *([setting] if setting else [])), exc.message))
            element.pop(definition.inline_renames.get(setting, setting), None)
        if "dataset" in element:
            problems.append(
                _problem((section, name, "dataset"), "the dataset is the one defining it")
            )
        element["dataset"] = "dataset"  # the defining dataset; the models only need a name
        try:
            definition.element_model.model_validate(element)
        except ValidationError as exc:
            back = _reverse(definition.inline_renames)
            problems.extend(
                InitErrorDetails(
                    type=error["type"],
                    loc=(section, name, *_renamed_loc(error["loc"], back)),
                    input=error["input"],
                    ctx=error.get("ctx", {}),
                )
                for error in exc.errors(include_url=False)
            )
        return problems


def _problem(loc: tuple[str | int, ...], message: str) -> InitErrorDetails:
    return InitErrorDetails(
        type="value_error", loc=loc, input=None, ctx={"error": ValueError(message)}
    )


def _renamed_loc(loc: Sequence[Any], back: Mapping[str, str]) -> tuple[Any, ...]:
    """An element error location under the setting names a definition is written with."""
    if not loc or not isinstance(loc[0], str):
        return tuple(loc)
    return (back.get(loc[0], loc[0]), *loc[1:])


__all__ = [
    "BY_RULE",
    "CATALOG",
    "CONTRACT_RULES",
    "CheckDefinition",
    "Decompiled",
    "NamedBlockCheck",
    "PluginDefinition",
    "definition_for",
    "derived_datasets",
]
