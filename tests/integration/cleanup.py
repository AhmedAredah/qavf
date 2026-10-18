"""Best-effort fixture cleanup that attempts every owned resource and fails closed."""

from collections.abc import Callable, Iterable

from .policy import IntegrationUnavailable


def cleanup_owned(
    service: str,
    actions: Iterable[Callable[[], object]],
    original: BaseException | None,
) -> None:
    errors: list[str] = []
    for action in actions:
        try:
            action()
        except Exception as exc:
            # Service exception strings can contain credentials or signed URLs.
            errors.append(type(exc).__name__)
    if not errors:
        return
    message = f"{service} fixture cleanup failed ({', '.join(errors)})"
    if original is not None:
        original.add_note(message)
    else:
        raise IntegrationUnavailable(message) from None
