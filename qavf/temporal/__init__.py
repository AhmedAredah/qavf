"""Calendar arithmetic on time columns: buckets, cyclic profiles and calendar offsets.

Everything here builds native Polars expressions; no row reaches Python. Rules
(``TMP-*`` series) and dataset comparisons (``temporal`` and ``series_shape``
levels, the ``shift_time`` alignment step) share these definitions, so a bucket
means the same thing everywhere.
"""

from qavf.temporal.buckets import (
    Bucket,
    TimeBucketing,
    bucket_grid,
    bucket_labels,
    bucket_scalar,
    calendar_position,
    sequence_positions,
)
from qavf.temporal.shift import CalendarOffset, shifted_time

__all__ = [
    "Bucket",
    "CalendarOffset",
    "TimeBucketing",
    "bucket_grid",
    "bucket_labels",
    "bucket_scalar",
    "calendar_position",
    "sequence_positions",
    "shifted_time",
]
