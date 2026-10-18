"""Regression testing of generated outputs against baselines."""

from qavf.regression.baselines import (
    BaselineManifest,
    accept_baseline,
    read_manifest,
    verify_baseline,
)
from qavf.regression.compare_outputs import (
    assert_outputs_match,
    compare_frames,
    compare_outputs,
    save_baseline,
)
from qavf.regression.models import ComparisonResult, Difference

__all__ = [
    "BaselineManifest",
    "ComparisonResult",
    "Difference",
    "accept_baseline",
    "assert_outputs_match",
    "compare_frames",
    "compare_outputs",
    "read_manifest",
    "save_baseline",
    "verify_baseline",
]
