"""Bounded duplicate-position accumulation, independent of dataframe engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from qavf.audit_codec import evidence_json_size


@dataclass(frozen=True, slots=True)
class DuplicateEvidencePolicy:
    """Keep exact counts separate from the optional row-position payload."""

    rows_per_key: int | Literal["all"] = 10
    byte_budget: int = 1_048_576

    def accumulator(
        self,
        records: list[dict[str, Any]],
        *,
        frequency: str = "occurrences",
        positions: str = "rows",
    ) -> PositionAccumulator:
        return PositionAccumulator(self, records, frequency, positions)


class PositionAccumulator:
    """Consume ordered ``(sample_index, row_position)`` batches under a byte cap.

    The key sample is already bounded. Counts remain in each record, even when
    its position list is truncated by the configured byte or per-key limit.
    """

    def __init__(
        self,
        policy: DuplicateEvidencePolicy,
        records: list[dict[str, Any]],
        frequency: str,
        positions: str,
    ) -> None:
        self.policy, self.records = policy, records
        self.frequency, self.positions = frequency, positions
        self.available = True
        for record in records:
            record[positions] = []
        self.used = evidence_json_size(records)

    def append(self, sample_index: int, position: int) -> bool:
        """False stops the stream when the byte budget is exhausted."""
        values = self.records[sample_index][self.positions]
        if self.policy.rows_per_key != "all" and len(values) >= self.policy.rows_per_key:
            return True
        # Canonical evidence JSON uses compact comma-separated integer arrays.
        cost = len(str(position)) + bool(values)
        if self.used + cost > self.policy.byte_budget:
            return False
        values.append(position)
        self.used += cost
        return True

    @property
    def metrics(self) -> dict[str, Any]:
        truncated = any(
            len(record[self.positions]) < record[self.frequency] for record in self.records
        )
        return {
            "duplicate_rows_per_key": self.policy.rows_per_key,
            "duplicate_positions_available": self.available,
            "duplicate_positions_complete": self.available and not truncated,
            "duplicate_positions_truncated": truncated,
        }
