"""Genuine loopback HTTP file reads without a remote account or service container."""

import pytest

from qavf import validate
from qavf.sources import SourceEnvironment, SourceRegistry

from .contracts import assert_exact_contract, assert_unavailable, configuration, corpus
from .http_service import http_corpus
from .policy import extension
from .pytest_support import available_service

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def http_origin(live_required):
    def configured():
        extension("httpfs")
        return http_corpus()

    with available_service(configured) as service:
        yield service


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_http_parquet_preserves_exact_values_and_qa_verdicts(http_origin, clean):
    origin, requests = http_origin
    name = "clean.parquet" if clean else "invalid.parquet"
    url = f"{origin}/{name}"
    source = (
        SourceRegistry()
        .discover(use_entry_points=False)
        .build(
            {"type": "file", "path": url},
            SourceEnvironment(),
        )
    )
    assert source.load().sort("ordinal").to_dicts() == corpus(clean).to_dicts()
    before = len(requests)
    assert_exact_contract(validate(url, config=configuration()), clean=clean)
    assert len(requests) > before, "validation did not read from the HTTP service"


def test_missing_http_file_is_unavailable(http_origin):
    origin, requests = http_origin
    assert_unavailable(validate(f"{origin}/missing.parquet", config=configuration()))
    assert "/missing.parquet" in requests
