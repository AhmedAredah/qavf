"""Mask credentials and access tokens before configuration or provenance is recorded.

Reports are shared far more widely than the configuration that produced them, and
``${ENV}`` expansion puts real secrets into the loaded settings. Everything a
report records about where data came from passes through :func:`redact`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit, urlunsplit

MASK = "***"

#: Names (of settings keys, query parameters or ``key=value`` pairs) whose values are secret.
_SECRET_NAME = re.compile(
    r"(?:^|[_\-.])(?:pass(?:word|wd)?|pwd|secret|token|api[_\-]?key|access[_\-]?key|"
    r"private[_\-]?key|account[_\-]?key|key[_\-]?id|connection[_\-]?string|"
    r"sas[_\-]?(?:key|token)|shared[_\-]?access[_\-]?signature|"
    r"credentials?|auth(?:orization)?|sig(?:nature)?|session|"
    r"x-amz-(?:signature|credential|security-token)|x-goog-(?:signature|credential))"
    r"(?:$|[_\-.])",
    re.IGNORECASE,
)
_URL = re.compile(r"(?<![A-Za-z0-9+.\-])[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"]+")
_ASSIGNMENT = re.compile(
    r"(?<![A-Za-z0-9_\-.])(?P<quote>['\"]?)(?P<name>[A-Za-z_][A-Za-z0-9_\-.]*)(?P=quote)\s*(?P<separator>[=:])\s*"
)
_AUTH_HEADER = re.compile(r"\b(Authorization\s*[=:]\s*)[^\r\n]+", re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def is_secret_name(name: str) -> bool:
    """Whether a key, parameter or option called ``name`` holds a secret."""
    return bool(_SECRET_NAME.search(_CAMEL_BOUNDARY.sub("_", name)))


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return MASK
    netloc = parts.netloc
    if "@" in netloc:
        _, _, host = netloc.rpartition("@")
        # Some services place an API token in the username rather than password.
        netloc = f"{MASK}@{host}"
    query = parts.query
    pairs = parse_qsl(query, keep_blank_values=True)
    if any(is_secret_name(name) for name, _ in pairs):
        query = "&".join(
            f"{quote(name, safe='')}={MASK if is_secret_name(name) else quote(value, safe='')}"
            for name, value in pairs
        )
    if netloc == parts.netloc and query == parts.query:
        return url
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def _quoted_end(text: str, start: int) -> int:
    """Skip a quoted value, including SQL doubled quotes and backslash escapes."""
    quote_char = text[start]
    width = 1
    while start + width < len(text) and text[start + width] == quote_char:
        width += 1
    if width == 2 and (start + width == len(text) or text[start + width] in ";,&}]) "):
        return start + width
    index = start + width
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == quote_char:
            count = 1
            while index + count < len(text) and text[index + count] == quote_char:
                count += 1
            if width == 1 and count % 2 == 0:
                index += count
            elif count >= width:
                return index + count
            else:
                index += count
        else:
            index += 1
    return len(text)


def _container_end(text: str, start: int) -> int:
    """Skip a complete nested diagnostic container; malformed syntax consumes the rest."""
    pairs = {"{": "}", "[": "]", "(": ")"}
    stack = [pairs[text[start]]]
    index = start + 1
    while index < len(text):
        char = text[index]
        if char in "\"'":
            index = _quoted_end(text, index)
            continue
        if char == "\\":
            index += 2
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in "}])":
            if not stack or stack.pop() != char:
                return len(text)
            if not stack:
                return index + 1
        index += 1
    return len(text)


def _value_end(text: str, start: int) -> int:
    """Consume whole secret values, distinguishing structured containers from ODBC braces."""
    if start == len(text):
        return start
    char = text[start]
    if char in "\"'":
        return _quoted_end(text, start)
    if char in "[(":
        return _container_end(text, start)
    if char == "{":
        inner = text[start + 1 :].lstrip()
        # Python/JSON mappings and nested containers have structure, unlike an
        # ODBC brace-quoted scalar. Ambiguous/malformed structured values are
        # conservatively consumed to the end by the balanced parser.
        if not inner or inner[0] in "\"'{[}(" or re.match(r"[A-Za-z0-9_]+\s*:", inner):
            return _container_end(text, start)
        index = start + 1
        while index < len(text):
            if text[index : index + 2] == "}}":
                index += 2
            elif text[index] == "}":
                return index + 1
            else:
                index += 1
        return len(text)
    index = start
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index].isspace() or text[index] in ";,&}":
            break
        else:
            index += 1
    return min(index, len(text))


def redact_text(text: str) -> str:
    """Mask credentials in URLs, assignments, dictionary displays and auth headers."""
    text = _URL.sub(lambda match: _redact_url(match.group(0)), text)
    text = _AUTH_HEADER.sub(lambda match: match.group(1) + MASK, text)
    parts: list[str] = []
    position = 0
    for match in _ASSIGNMENT.finditer(text):
        if match.start() < position or not is_secret_name(match.group("name")):
            continue
        parts.extend((text[position : match.end()], MASK))
        position = _value_end(text, match.end())
    parts.append(text[position:])
    return "".join(parts)


def redact(value: Any) -> Any:
    """Recursively mask secrets in configuration-like data.

    Values under secret-named keys are replaced entirely; other strings are
    passed through :func:`redact_text`.
    """
    from qavf.audit_codec import normalize_evidence

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: MASK if is_secret_name(key) else visit(child) for key, child in item.items()
            }
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, Path):
            return Path(redact_text(str(item)))
        if isinstance(item, bytes):
            try:
                return redact_text(item.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                return item
        return redact_text(item) if isinstance(item, str) else item

    # Normalize once using the same supported-container algebra, collision,
    # depth and cycle checks as the eventual audit snapshot. In particular sets
    # and NumPy arrays must not become reportable *after* redaction has skipped
    # their leaves. Unsupported containers fail before provenance publication.
    return visit(normalize_evidence(value))


def exception_details(exc: BaseException) -> str:
    """Actionable diagnostics without raw traceback code or credential-bearing inputs.

    Validation errors retain field paths, error types and messages but omit input
    values and arbitrary validator contexts. Only a traceback's final filename and
    line number are included; traceback source code and locals are never rendered.
    """
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: "
            f"{error['msg']} ({error['type']})"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        )
    else:
        details = str(exc)
    trace = exc.__traceback__
    while trace is not None and trace.tb_next is not None:
        trace = trace.tb_next
    location = ""
    if trace is not None:
        filename = trace.tb_frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        location = f" at {filename}:{trace.tb_lineno}"
    return redact_text(f"{type(exc).__name__}{location}: {details}")


__all__ = ["MASK", "exception_details", "is_secret_name", "redact", "redact_text"]
