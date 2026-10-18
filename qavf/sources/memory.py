"""Compose an explicit DuckDB buffer cap with a run-owned byte reservation.

DuckDB's displayed settings are rounded, so they cannot safely decide which of
two caps is smaller. This module compares the original, already validated setting
using DuckDB's numeric/unit semantics, then keeps the original setting whenever
it is stricter. It never treats the buffer cap as a whole-process memory limit.
"""

from __future__ import annotations

import re

_QUANTITY = re.compile(r"\s*([0-9.eE-]+)\s*([^\s]*)")
_FACTORS = {"b": 1, "byte": 1, "bytes": 1}
for _power, _prefix in enumerate(("kilo", "mega", "giga", "tera"), 1):
    _letter = _prefix[0]
    for _unit in (_letter, _letter + "b", _prefix + "byte", _prefix + "bytes"):
        _FACTORS[_unit] = 1000**_power
    _FACTORS[_letter + "ib"] = 1024**_power


def reservation_is_stricter(configured: str | None, reserved: int) -> bool:
    """Whether to replace an already-applied native setting with reserved bytes.

    The caller first lets DuckDB validate/apply ``configured``. The grammar here
    covers its supported decimal/binary unit aliases, scientific notation and
    unlimited sentinels on the minimum and current dependency versions. DuckDB
    uses a double for the quantity and truncates the byte product. Retaining the
    original text avoids re-rounding a lower cap. Unknown future syntax fails
    closed instead of silently discarding either limit.

    See DuckDB's ``DBConfig::ParseMemoryLimit`` (v1.2) and
    ``StringUtil::TryParseFormattedBytes`` (current).
    """
    if not configured or configured.startswith("-") or configured in {"none", "null"}:
        return True
    match = _QUANTITY.match(configured)
    if match is None:
        raise ValueError("cannot combine this DuckDB memory limit with a run byte reservation")
    amount = float(match[1])
    if amount < 0:
        return True
    factor = _FACTORS.get(match[2].lower())
    if factor is None:
        raise ValueError("cannot combine this DuckDB memory unit with a run byte reservation")
    # Compare before converting to int: an accepted very large quantity cannot
    # overflow a Python integer conversion, and equal/sub-byte caps stay intact.
    return amount * factor >= reserved + 1
