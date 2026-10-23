"""Neutral immutable model base shared by inference, comparisons and reports."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_serializer, model_validator

from qavf.audit_codec import decode_audit as decode_non_finite
from qavf.audit_codec import encode_audit as encode_non_finite
from qavf.audit_codec import freeze_evidence


class SnapshotModel(BaseModel):
    """A validated audit snapshot with detached, immutable nested containers."""

    model_config = ConfigDict(frozen=True)
    audit_fields: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _decode_audit_fields(cls, data: Any, info: ValidationInfo) -> Any:
        if info.mode == "json" and isinstance(data, dict):
            data = dict(data)
            for name in cls.audit_fields:
                if name in data:
                    data[name] = decode_non_finite(data[name])
        return data

    @field_serializer(
        "affected_records",
        "metrics",
        "notes",
        "fingerprint",
        "params",
        "extensions",
        "config",
        "resource_usage",
        "query_plans",
        when_used="json",
        check_fields=False,
    )
    def _encode_audit_field(self, value: Any) -> Any:
        return encode_non_finite(value)

    @model_validator(mode="after")
    def _snapshot(self):
        for name in type(self).model_fields:
            object.__setattr__(
                self,
                name,
                freeze_evidence(getattr(self, name), preserved_types=(SnapshotModel, Enum)),
            )
        return self

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        data = {name: getattr(self, name) for name in type(self).model_fields}
        data.update(update or {})
        return type(self).model_validate(data)
