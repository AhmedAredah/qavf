"""The consensus of several datasets, key by key, and the pooled data of all of them.

The consensus member of a check with ``strategy: consensus`` holds, for every key
the members agree exists, the members' consensus value of each column: the
median, mean or trimmed mean of the numbers, and the majority (or plurality)
value of anything else (:class:`~qavf.compare.pairing.ConsensusSettings`). Each
member is then compared with it as with a reference.

Nothing is guessed. A key or a value without enough agreement, or with a tie
(unless ``ties: reference`` takes the reference member's), has no consensus: the
key is kept with its reason in :data:`NO_CONSENSUS` and is not assessed. The
same rule decides whether a key exists at all: present on enough members, it is
part of the consensus; absent from enough, it is not.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from qavf.compare.pairing import ConsensusSettings

NO_CONSENSUS = "__qavf_no_consensus"
"""Column of the consensus frame giving, per key, why it has no consensus (null: it has)."""
ROWS = "rows"
"""The consensus column of row counts (``totals_match`` counting rows)."""

_MEMBER = "__qavf_member"
_COUNT = "__qavf_count"
_TOP = "__qavf_top"
_WINNERS = "__qavf_winners"
_REASON = "__qavf_reason"
_PRESENT = "__qavf_present"
_HELD = "__qavf_held"
_REFERENCE = "__qavf_reference"
_REFERENCE_VALUE = "__qavf_reference_value"


class ConsensusError(ValueError):
    """Members that cannot be combined (missing keys, incompatible columns)."""


class ConsensusBuilder:
    """Builds the consensus frame of several members, matched on ``keys``."""

    def __init__(
        self,
        settings: ConsensusSettings,
        keys: Sequence[str],
        members: Sequence[str],
        *,
        reference: str | None = None,
        columns: Sequence[str] | None = None,
        ignore: Sequence[str] = (),
    ) -> None:
        if not keys:
            raise ConsensusError("a consensus is matched on keys; give the check's keys")
        self.settings = settings
        self.keys = list(keys)
        self.members = list(members)
        self.reference = self.members.index(reference) if reference in self.members else 0
        self.columns = list(columns) if columns is not None else None
        self.ignore = set(ignore)

    # ------------------------------------------------------------------ inputs
    def value_columns(self, frames: Sequence[pl.DataFrame]) -> list[str]:
        """Columns with a consensus value: those given, else every common non-key column."""
        for member, frame in zip(self.members, frames, strict=True):
            if missing := [key for key in self.keys if key not in frame.columns]:
                raise ConsensusError(f"{member} has no key column(s) {', '.join(missing)}")
        if self.columns is not None:
            for member, frame in zip(self.members, frames, strict=True):
                if missing := [c for c in self.columns if c not in frame.columns]:
                    raise ConsensusError(f"{member} has no column(s) {', '.join(missing)}")
            return list(self.columns)
        common = [c for c in frames[0].columns if all(c in frame.columns for frame in frames)]
        return [c for c in common if c not in self.keys and c not in self.ignore]

    def aggregated(self, frames: Sequence[pl.DataFrame], value: str | None) -> list[pl.DataFrame]:
        """Each member's total per key: ``sum(value)``, or its row count."""
        total = pl.col(value).sum().alias(value) if value is not None else pl.len().alias(ROWS)
        result = []
        for member, frame in zip(self.members, frames, strict=True):
            needed = [*self.keys, *([value] if value is not None else [])]
            if missing := [column for column in needed if column not in frame.columns]:
                raise ConsensusError(f"{member} has no column(s) {', '.join(missing)}")
            result.append(frame.drop_nulls(self.keys).group_by(self.keys).agg(total))
        return result

    def stacked(self, frames: Sequence[pl.DataFrame], columns: list[str]) -> pl.DataFrame:
        """Every member's rows (keys and value columns), tagged with the member's index."""
        parts = [
            frame.select([*self.keys, *columns])
            .drop_nulls(self.keys)
            .with_columns(pl.lit(index, dtype=pl.Int32).alias(_MEMBER))
            for index, frame in enumerate(frames)
        ]
        try:
            return pl.concat(parts, how="vertical_relaxed")
        except (pl.exceptions.PolarsError, TypeError) as exc:
            raise ConsensusError(f"the members' columns cannot be combined: {exc}") from None

    # ------------------------------------------------------------------- build
    def build(self, frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
        """Keys, each column's consensus value, and :data:`NO_CONSENSUS`, sorted by key."""
        if len(frames) != len(self.members):
            raise ConsensusError("give one frame per member")
        columns = self.value_columns(frames)
        stacked = self.stacked(frames, columns)
        duplicated = (
            stacked.group_by([*self.keys, _MEMBER])
            .len()
            .filter(pl.col("len") > 1)
            .group_by(self.keys)
            .agg(pl.col(_MEMBER).min())
        )
        stacked = stacked.unique(subset=[*self.keys, _MEMBER], keep="first", maintain_order=True)
        presence = self.presence(stacked)
        held = presence.filter(pl.col(_PRESENT)).select([*self.keys, _HELD])
        values = stacked.join(held.select(self.keys), on=self.keys, how="semi")
        result = held.select(self.keys)
        reasons = [presence.filter(pl.col(_REASON).is_not_null()).select([*self.keys, _REASON])]
        decided_keys = presence.filter(pl.col(_PRESENT) | pl.col(_REASON).is_not_null())
        duplicated = duplicated.join(decided_keys.select(self.keys), on=self.keys, how="semi")
        reasons.append(self.duplicate_reasons(duplicated))
        for column in columns:
            dtype = stacked.schema[column]
            decided = (
                self.numeric(values, held, column, dtype)
                if dtype.is_numeric()
                else self.categorical(values, held, column)
            )
            result = result.join(decided.select([*self.keys, column]), on=self.keys, how="left")
            reasons.append(
                decided.filter(pl.col(_REASON).is_not_null()).select([*self.keys, _REASON])
            )
        gaps = (
            pl.concat(reasons, how="vertical_relaxed")
            .group_by(self.keys, maintain_order=True)
            .agg(pl.col(_REASON).unique(maintain_order=True).str.join("; ").alias(NO_CONSENSUS))
        )
        result = result.join(gaps, on=self.keys, how="full", coalesce=True)
        return result.select([*self.keys, *columns, NO_CONSENSUS]).sort(self.keys)

    def duplicate_reasons(self, duplicated: pl.DataFrame) -> pl.DataFrame:
        """Keys repeated within a member: no single value of that member to count."""
        names = pl.Series(self.members, dtype=pl.String)
        return duplicated.select(
            [
                *self.keys,
                (
                    pl.lit("duplicate key in ")
                    + pl.col(_MEMBER).replace_strict(
                        list(range(len(self.members))), names, return_dtype=pl.String
                    )
                ).alias(_REASON),
            ]
        )

    def presence(self, stacked: pl.DataFrame) -> pl.DataFrame:
        """Per key: whether it is part of the consensus (``_PRESENT``), the number of
        members holding it (``_HELD``), and why it has no consensus (``_REASON``)."""
        n = len(self.members)
        held = pl.col(_HELD)
        absent = pl.lit(n) - held
        top = pl.max_horizontal(held, absent)
        tie = held == absent
        agree = self._agrees(top, pl.lit(n), tie)
        tie_to_reference = tie & pl.lit(self.settings.ties == "reference")
        present = (~tie & agree & (held > absent)) | (tie_to_reference & pl.col(_REFERENCE))
        reason = (
            pl.when(tie & ~tie_to_reference)
            .then(pl.lit("presence: tie"))
            .when(~tie & ~agree)
            .then(pl.lit("presence: too little agreement"))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        counted = stacked.group_by(self.keys).agg(
            pl.len().cast(pl.Int64).alias(_HELD),
            (pl.col(_MEMBER) == self.reference).any().alias(_REFERENCE),
        )
        return counted.with_columns(present.alias(_PRESENT), reason.alias(_REASON))

    def _agrees(self, top: pl.Expr, total: pl.Expr, tie: pl.Expr) -> pl.Expr:
        """Whether the most common outcome is a consensus under the categorical method."""
        share = top.cast(pl.Float64) / total.cast(pl.Float64)
        enough = share >= self.settings.min_agreement
        if self.settings.categorical == "majority":
            return enough & (top * 2 > total)
        return enough & ~tie

    def numeric(
        self, values: pl.DataFrame, held: pl.DataFrame, column: str, dtype: pl.DataType
    ) -> pl.DataFrame:
        """A number's consensus per key: reported by enough members, then combined."""
        reported = pl.col(column).count()
        if self.settings.numeric == "median":
            combined = pl.col(column).median()
        elif self.settings.numeric == "mean":
            combined = pl.col(column).mean()
        else:
            cut = (reported.cast(pl.Float64) * self.settings.trim).floor().cast(pl.Int64)
            kept = pl.col(column).drop_nulls().sort().slice(cut, reported - 2 * cut)
            combined = kept.mean()
        stats = values.group_by(self.keys).agg(
            combined.alias(column), reported.cast(pl.Int64).alias(_COUNT)
        )
        stats = stats.join(held, on=self.keys, how="left")
        share_reported = pl.col(_COUNT).cast(pl.Float64) / pl.col(_HELD).cast(pl.Float64)
        share_missing = (pl.col(_HELD) - pl.col(_COUNT)).cast(pl.Float64) / pl.col(_HELD).cast(
            pl.Float64
        )
        minimum = self.settings.min_agreement
        value = pl.when(share_reported >= minimum).then(pl.col(column))
        reason = (
            pl.when((share_reported < minimum) & (share_missing < minimum))
            .then(pl.lit(f"{column}: too little agreement"))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        decided = stats.with_columns(value.alias(column), reason.alias(_REASON))
        return decided.with_columns(_restored(decided[column], dtype).alias(column))

    def categorical(self, values: pl.DataFrame, held: pl.DataFrame, column: str) -> pl.DataFrame:
        """Any other value's consensus per key: the majority (or plurality) value."""
        counts = values.group_by([*self.keys, column]).agg(pl.len().cast(pl.Int64).alias(_COUNT))
        tops = counts.group_by(self.keys).agg(pl.col(_COUNT).max().alias(_TOP))
        winners = (
            counts.join(tops, on=self.keys, how="left")
            .filter(pl.col(_COUNT) == pl.col(_TOP))
            .sort([*self.keys, column], nulls_last=True)
            .group_by(self.keys, maintain_order=True)
            .agg(
                pl.col(column).first().alias(column),
                pl.len().cast(pl.Int64).alias(_WINNERS),
                pl.col(_TOP).first().alias(_TOP),
            )
        )
        reference = values.filter(pl.col(_MEMBER) == self.reference).select(
            [*self.keys, pl.col(column).alias(_REFERENCE_VALUE), pl.lit(True).alias(_REFERENCE)]
        )
        decided = winners.join(held, on=self.keys, how="left").join(
            reference, on=self.keys, how="left"
        )
        tie = pl.col(_WINNERS) > 1
        agree = self._agrees(pl.col(_TOP), pl.col(_HELD), tie)
        to_reference = (
            tie & pl.lit(self.settings.ties == "reference") & pl.col(_REFERENCE).fill_null(False)
        )
        value = (
            pl.when(to_reference)
            .then(pl.col(_REFERENCE_VALUE))
            .when(~tie & agree)
            .then(pl.col(column))
            .otherwise(pl.lit(None))
        )
        reason = (
            pl.when(to_reference)
            .then(pl.lit(None, dtype=pl.String))
            .when(tie)
            .then(pl.lit(f"{column}: tie"))
            .when(~agree)
            .then(pl.lit(f"{column}: too little agreement"))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        return decided.with_columns(value.alias(column), reason.alias(_REASON))


def _restored(values: pl.Series, dtype: pl.DataType) -> pl.Series:
    """Consensus numbers in the members' integer type when every one is whole."""
    if not dtype.is_integer() or values.dtype == dtype:
        return values
    present = values.drop_nulls()
    if present.len() and not bool((present == present.round(0)).all()):
        return values
    return values.round(0).cast(dtype, strict=False)


def pooled(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Every member's rows together, in the columns all members share."""
    common = [c for c in frames[0].columns if all(c in frame.columns for frame in frames)]
    try:
        return pl.concat([frame.select(common) for frame in frames], how="vertical_relaxed")
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise ConsensusError(f"the members' columns cannot be combined: {exc}") from None


__all__ = ["NO_CONSENSUS", "ROWS", "ConsensusBuilder", "ConsensusError", "pooled"]
