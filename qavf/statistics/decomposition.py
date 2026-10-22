"""Exact decompositions of a change: member contributions and mix versus rate.

Everything here is exact rational arithmetic, so the parts of a change sum to
the change itself with no residual:

* :func:`contributions`: each member's change of an additive total (a count
  or a sum), ``candidate - baseline`` with an absent side counting 0; the
  changes sum to the change of the total;
* :func:`minimal_set`: the fewest members whose changes reach a share of the
  total change (taking the largest changes in its direction first, which is
  optimal: no ``k`` members reach more than the ``k`` largest);
* :func:`rate_decomposition`: the change of an overall rate ``sum(N) / sum(D)``
  split into a **mix** effect (the composition of the denominator shifted
  between strata), a **rate** effect (the strata's own rates changed) and their
  **interaction** (shift-share with interaction), or the symmetric Kitagawa
  (1955) form without an interaction. With weights ``w = D / sum(D)`` and
  rates ``r = N / D`` per stratum, ``R = sum(w r)`` and::

      R_c - R_b = sum((w_c - w_b) r_b)            mix
                + sum(w_b (r_c - r_b))            rate
                + sum((w_c - w_b) (r_c - r_b))    interaction

  term by term, since ``w_c r_c - w_b r_b`` expands exactly into the three.
  Kitagawa's form weighs each difference by the other factor's mean:
  ``sum((w_c - w_b)(r_b + r_c) / 2) + sum((r_c - r_b)(w_b + w_c) / 2)``.

A stratum without denominator on one side (new or removed) takes the other
side's rate there, so it contributes only through the mix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

DecompositionMethod = Literal["three_term", "symmetric"]
ZERO = Fraction(0)


@dataclass(frozen=True, slots=True)
class Contribution:
    """One member's values on both sides (0 where absent) and its change."""

    key: Any
    baseline: Fraction
    candidate: Fraction
    present: tuple[bool, bool]
    """Whether the member exists on each side."""

    @property
    def change(self) -> Fraction:
        return self.candidate - self.baseline


def contributions(
    members: Iterable[tuple[Any, Fraction | None, Fraction | None]],
) -> list[Contribution]:
    """Each member's change, largest magnitude first (ties in the given order).

    ``members`` are ``(key, baseline, candidate)`` with ``None`` for a side the
    member is absent from.
    """
    found = [
        Contribution(key, b if b is not None else ZERO, c if c is not None else ZERO,
                     (b is not None, c is not None))
        for key, b, c in members
    ]  # fmt: skip
    order = sorted(range(len(found)), key=lambda i: (-abs(found[i].change), i))
    return [found[i] for i in order]


def minimal_set(changes: Sequence[Fraction], share: Fraction) -> list[int] | None:
    """Positions of the fewest changes summing to at least ``share`` of their total.

    The changes in the total's direction are taken largest first; ``None`` when
    the total change is zero (nothing to explain).

    Raises:
        ValueError: for a share outside ``(0, 1]``.
    """
    if not 0 < share <= 1:
        raise ValueError("the explained share must lie in (0, 1]")
    total = sum(changes, ZERO)
    if not total:
        return None
    sign = 1 if total > 0 else -1
    target = share * abs(total)
    order = sorted(range(len(changes)), key=lambda i: (-sign * changes[i], i))
    picked: list[int] = []
    reached = ZERO
    for position in order:
        if reached >= target:
            break
        picked.append(position)
        reached += sign * changes[position]
    return picked


@dataclass(frozen=True, slots=True)
class StratumEffect:
    """One stratum's weights, rates and its parts of the change."""

    key: Any
    baseline_weight: Fraction
    candidate_weight: Fraction
    baseline_rate: Fraction | None
    """``None`` where the stratum has no denominator on that side."""
    candidate_rate: Fraction | None
    mix: Fraction
    rate: Fraction
    interaction: Fraction

    @property
    def total(self) -> Fraction:
        return self.mix + self.rate + self.interaction


@dataclass(frozen=True, slots=True)
class RateDecomposition:
    """``change = mix + rate + interaction`` exactly, with each stratum's parts."""

    method: DecompositionMethod
    baseline_rate: Fraction
    candidate_rate: Fraction
    mix: Fraction
    rate: Fraction
    interaction: Fraction
    strata: list[StratumEffect]

    @property
    def change(self) -> Fraction:
        return self.candidate_rate - self.baseline_rate


def rate_decomposition(
    strata: Iterable[tuple[Any, Fraction, Fraction, Fraction, Fraction]],
    method: DecompositionMethod = "three_term",
) -> RateDecomposition:
    """Split the change of ``sum(N) / sum(D)`` over strata into mix, rate and interaction.

    ``strata`` are ``(key, N_b, D_b, N_c, D_c)``; a side's absent stratum has
    ``N = D = 0``. Strata without denominator on both sides are left out.

    Raises:
        ValueError: for a negative denominator, a numerator without denominator,
            a side whose denominators sum to zero, or an unknown method.
    """
    if method not in ("three_term", "symmetric"):
        raise ValueError(f"method must be 'three_term' or 'symmetric', got {method!r}")
    rows = []
    for key, nb, db, nc, dc in strata:
        if db < 0 or dc < 0:
            raise ValueError(f"stratum {key!r} has a negative denominator")
        if (not db and nb) or (not dc and nc):
            raise ValueError(f"stratum {key!r} has a numerator but no denominator")
        if db or dc:
            rows.append((key, nb, db, nc, dc))
    totals = [sum((row[i] for row in rows), ZERO) for i in (1, 2, 3, 4)]
    if not totals[1] or not totals[3]:
        raise ValueError("a side's denominators sum to zero; its rate is undefined")
    effects = []
    for key, nb, db, nc, dc in rows:
        wb, wc = db / totals[1], dc / totals[3]
        rb = nb / db if db else None
        rc = nc / dc if dc else None
        # An absent side takes the other side's rate: the stratum moves by mix alone.
        b_rate = rb if rb is not None else rc
        c_rate = rc if rc is not None else rb
        assert b_rate is not None and c_rate is not None
        if method == "three_term":
            mix = (wc - wb) * b_rate
            rate = wb * (c_rate - b_rate)
            interaction = (wc - wb) * (c_rate - b_rate)
        else:
            mix = (wc - wb) * (b_rate + c_rate) / 2
            rate = (c_rate - b_rate) * (wb + wc) / 2
            interaction = ZERO
        effects.append(StratumEffect(key, wb, wc, rb, rc, mix, rate, interaction))
    return RateDecomposition(
        method,
        totals[0] / totals[1],
        totals[2] / totals[3],
        sum((e.mix for e in effects), ZERO),
        sum((e.rate for e in effects), ZERO),
        sum((e.interaction for e in effects), ZERO),
        effects,
    )
