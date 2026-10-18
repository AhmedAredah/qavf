"""Render prerequisite failures without exposing environment or credential arguments."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest

from .policy import IntegrationUnavailable


@contextmanager
def available_service[T](factory: Callable[[], AbstractContextManager[T]]) -> Iterator[T]:
    try:
        with factory() as service:
            yield service
    except IntegrationUnavailable as exc:
        message = "\n".join((str(exc), *getattr(exc, "__notes__", ())))
        pytest.fail(message, pytrace=False)
