"""Shared immutable parameters for rules and other extension components."""

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, field_validator

from qavf.scalars import normalize_scalar


def _not_boolean(value: Any) -> Any:
    """YAML reads ``true``/``yes``/``on`` as booleans, which are integers to Python."""
    if isinstance(value, bool):
        raise ValueError("a count must be a whole number, not true or false")
    return value


Count = Annotated[int, BeforeValidator(_not_boolean)]
"""A row count, budget or limit: a whole number, never a boolean."""


class RuleParams(BaseModel):
    """Strict parameters, shared without importing the rule execution engine.

    The historical ``qavf.engine.registry.RuleParams`` import is an alias of
    this class. Unknown keys and nonfinite numeric thresholds are rejected.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True, allow_inf_nan=False
    )

    @field_validator("*", mode="before")
    @classmethod
    def _scientific_scalar(cls, value: Any) -> Any:
        return normalize_scalar(value)


_SCHEMA_ANNOTATIONS = ("title", "description")


def text_or_mapping_schema(schema: dict[str, Any]) -> None:
    """``json_schema_extra`` of a model also written as one string (``day``,
    ``records``): the editor schema takes the string or the mapping."""
    mapping = {key: value for key, value in schema.items() if key not in _SCHEMA_ANNOTATIONS}
    for key in list(schema):
        if key not in _SCHEMA_ANNOTATIONS:
            del schema[key]
    schema["anyOf"] = [{"type": "string", "minLength": 1}, mapping]
