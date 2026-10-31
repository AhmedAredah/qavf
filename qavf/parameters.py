"""Shared immutable parameters for rules and other extension components."""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from qavf.scalars import normalize_scalar


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
