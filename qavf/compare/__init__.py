"""Dataset comparison: align two datasets, then compare them at several levels.

Quick start::

    from qavf import compare_datasets

    result = compare_datasets(
        "v1/counts.parquet", "v2/counts.parquet",
        keys=["site", "year"], columns={"aadt": "volume"},
        levels=["structure", "coverage", "records",
                {"level": "groups", "group_by": ["county"], "measures": ["sum(volume)"]}],
    )
    print(result.summary())

Over time, the ``temporal`` level compares measures per calendar bucket (or
cyclic profile) and ``series_shape`` compares the shapes of the bucketed series;
the ``shift_time`` step compares one version at a shifted time (year over year).

Alignment steps, measures, comparators and levels are registered components;
plugins add their own with the ``register_*`` decorators or the
``qavf.compare.alignment``, ``qavf.compare.measures``, ``qavf.compare.comparators``
and ``qavf.compare.levels`` entry-point groups. In a configuration, the
``CMP-001`` rule runs comparisons between configured datasets.
"""

from qavf.compare.align import (
    ALIGNMENT_STEPS,
    AlignmentContext,
    AlignmentError,
    AlignmentStep,
    register_alignment_step,
)
from qavf.compare.api import compare_datasets
from qavf.compare.comparators import (
    COMPARATORS,
    Comparator,
    ComparatorContext,
    Materiality,
    register_comparator,
)
from qavf.compare.levels import LEVELS, ComparisonLevel, LevelInputs, register_level
from qavf.compare.measures import MEASURES, Measure, MeasureInput, register_measure
from qavf.compare.registry import ComponentRegistry
from qavf.compare.results import (
    CheckResult,
    CoverageResult,
    DatasetComparison,
    GroupsResult,
    LevelResult,
    RecordsResult,
    SeriesShape,
    ShapeResult,
    Significance,
    StructureResult,
    TemporalResult,
)
from qavf.compare.spec import ComparisonSpec

__all__ = [
    "ALIGNMENT_STEPS",
    "COMPARATORS",
    "LEVELS",
    "MEASURES",
    "AlignmentContext",
    "AlignmentError",
    "AlignmentStep",
    "CheckResult",
    "Comparator",
    "ComparatorContext",
    "ComparisonLevel",
    "ComparisonSpec",
    "ComponentRegistry",
    "CoverageResult",
    "DatasetComparison",
    "GroupsResult",
    "LevelInputs",
    "LevelResult",
    "Materiality",
    "Measure",
    "MeasureInput",
    "RecordsResult",
    "SeriesShape",
    "ShapeResult",
    "Significance",
    "StructureResult",
    "TemporalResult",
    "compare_datasets",
    "register_alignment_step",
    "register_comparator",
    "register_level",
    "register_measure",
]
