"""Small cleanup primitives: attempt independent owners before reporting failures."""

from collections.abc import Callable, Iterable


def raise_cleanup_errors(label: str, failures: list[Exception]) -> None:
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ExceptionGroup(label, failures)


def attempt_cleanup(steps: Iterable[Callable[[], object]]) -> list[Exception]:
    """Run every step; each owner retains its own unfinished work for retry."""
    failures = []
    for step in steps:
        try:
            step()
        except Exception as exc:
            failures.append(exc)
    return failures
