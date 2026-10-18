# Numerical and assessment contracts

QA decisions take priority over approximation. A floating input means its stored
binary value; an integer or Decimal bound remains exact. A bound is never silently
rounded to the nearest floating value before comparison. Exact sum overflow guards
do not depend on the caller's Decimal context. Unsupported aggregate precision or
range produces an explicit unassessed outcome instead of a passing infinity or a
wrapped integer.

Statistics use finite, representable working values. Scaling must preserve the
observations needed by the chosen method. If it erases a small nonzero observation
in a mixed-range series, the group is unassessed; the standalone detector raises a
clear value error. This is a deliberate correctness boundary, not a claim to
support every possible precision or dynamic range with Float64 statistics.

Period-change kernels avoid overflowing intermediate division and integer time
subtraction. Exact integer decreases retain their full magnitude in evidence.
Key matching and expected domains align losslessly or fail explicitly; fractional
expected keys are never truncated into integers. Nested regression retains logical
types recursively, including datetime, duration and text distinctions.

Configured `missing_values` and `blank_as_missing` are applied consistently when
data enters rule execution. Null and NaN are missing. Reconciliation, coverage and
statistical rules therefore cannot treat a configured sentinel as a real balancing
value or entity. Missingness policy is separate from whether a particular check
allows missing observations.

Each configured statistical, temporal or geographic target has its own outcome.
A later invalid target cannot erase an earlier failure. Evidence caps control
retained examples and diagnostics; exact counts and assessment status remain.

Spatial group comparisons use observed group/partition combinations by default
and disclose that scope. Set `require_all_groups_per_partition: true` with explicit
groups when every group must exist in every observed partition. A coverage contract
is needed to require partitions that are completely absent from the input.

Spatial coverage currently requires temporal identifiers representable at microsecond
precision. Non-aligned Datetime(ns), Duration(ns) or Time identifiers are unassessed
before Python conversion can merge identities; use explicit physical integer keys
when finer-resolution identities are required.
