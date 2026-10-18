"""One column-name boundary shared by the regression APIs."""

from collections.abc import Sequence


def column_names(value: str | Sequence[str]) -> list[str]:
    names = [value] if isinstance(value, str) else list(value)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("column names must be nonempty strings")
    return names
