from collections.abc import Mapping, Sequence
from typing import Any

from qavf._input_types import DatasetInput
from qavf.compare.results import DatasetComparison
from qavf.compare.spec import ComparisonSpec
from qavf.inputs.resources import ResourceLimits

def compare_datasets(
    baseline: DatasetInput,
    candidate: DatasetInput,
    *,
    keys: str | Sequence[str] = ...,
    columns: Mapping[str, str] | None = ...,
    levels: Sequence[str | Mapping[str, Any]] | None = ...,
    name: str | None = ...,
    resources: ResourceLimits | None = ...,
    entry_points: bool = ...,
    configuration: ComparisonSpec | Mapping[str, Any] | None = ...,
    **options: Any,
) -> DatasetComparison: ...
