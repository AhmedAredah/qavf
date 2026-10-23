"""Credential-safe native errors with explicit transient-failure classification."""

import re
from collections.abc import Callable

import duckdb

from qavf.redaction import redact_text
from qavf.sources.base import SourceError, SourceIntegrityError, SourceNotFoundError


def redacted_error(exc: BaseException, cls: type[SourceError] = SourceError) -> SourceError:
    """A ``cls`` error carrying ``exc``'s message with credentials masked.

    The original exception is not chained: its text (and its traceback, if
    logged) could still hold a password or access token.
    """
    message = redact_text(str(exc))
    if "__qavf_exact_column" in message and "empty set of columns" in message:
        message += (
            "\nThe requested column spelling is unavailable in the native reader. "
            "Use its exact decoded name, or supply a typed frame to preserve "
            "physical column names that differ only in case."
        )
    error = cls(message)
    error.retryable = bool(getattr(exc, "retryable", False))
    if isinstance(exc, duckdb.HTTPException):
        # Authentication/not-found responses are permanent; retry only explicit
        # transient HTTP statuses or network timeouts.
        message = str(exc).lower()
        code = re.search(r"(?:http(?: error)?[: ]+|status(?: code)?[: ]+)([0-9]{3})\b", message)
        error.retryable = (
            code is not None and int(code[1]) in (408, 429, 500, 502, 503, 504)
        ) or "timed out" in message
    error.__cause__ = None
    error.__suppress_context__ = True
    return error


def verify_postcondition(verify: Callable[[], None], previous: BaseException | None) -> None:
    """Check failed reads too, without turning terminal failures into retries."""
    try:
        verify()
    except BaseException as current:
        # User/process interruptions remain interruptions regardless of an
        # ordinary read or identity failure encountered during unwinding.
        if previous is not None and not isinstance(previous, Exception):
            raise previous from None
        if not isinstance(current, Exception):
            raise
        if previous is None or isinstance(current, SourceIntegrityError):
            raise
        prior = (
            previous
            if isinstance(previous, SourceError)
            else redacted_error(previous)
            if isinstance(previous, duckdb.Error)
            else None
        )
        if (
            prior is None
            or not prior.retryable
            or isinstance(prior, SourceIntegrityError | SourceNotFoundError)
        ):
            raise previous from None
        raise


