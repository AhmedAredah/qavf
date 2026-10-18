"""Shared opt-in gate; fixtures never skip missing prerequisites in required mode."""

import os

import pytest

from .policy import IntegrationUnavailable, required_mode


@pytest.fixture(scope="session")
def live_required() -> None:
    try:
        enabled = required_mode(os.environ)
    except IntegrationUnavailable as exc:
        pytest.fail(str(exc), pytrace=False)
    if not enabled:
        pytest.skip("live integrations disabled; set QAVF_INTEGRATION_MODE=required")
