from collections.abc import Mapping
from typing import Any

from qavf._input_types import ValidationInput
from qavf.inputs.normalize import Configuration
from qavf.inputs.resources import ResourceLimits
from qavf.profiles import ProfileName
from qavf.reports.models import QAReport
from qavf.settings import SchemaSpec

def validate(
    data: ValidationInput = ...,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None = ...,
    config: Configuration = ...,
    resources: ResourceLimits | None = ...,
    profile: ProfileName | None = ...,
    backend: str = ...,
) -> QAReport: ...
