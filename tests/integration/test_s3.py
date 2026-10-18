"""Actual S3 object reads through authenticated Query and public HTTP file inputs."""

import duckdb
import pytest

from qavf import Query, validate
from qavf.sources import SourceEnvironment, SourceRegistry

from .contracts import assert_exact_contract, assert_unavailable, configuration, corpus
from .pytest_support import available_service
from .s3_service import s3_corpus

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def s3(live_required):
    with available_service(s3_corpus) as service:
        yield service


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_s3_authenticated_query_preserves_values_and_verdicts(s3, clean):
    with duckdb.connect() as connection:
        connection.execute("LOAD httpfs")
        connection.execute("SET http_timeout=5")
        connection.execute("SET http_retries=0")
        connection.execute(s3.secret_sql)
        actual = connection.execute("SELECT * FROM read_parquet(?)", [s3.uri(clean)]).pl()
        assert actual.sort("ordinal").to_dicts() == corpus(clean).to_dicts()
        query = Query("SELECT * FROM read_parquet($path)", connection, {"path": s3.uri(clean)})
        assert_exact_contract(validate(query, config=configuration()), clean=clean)
        assert connection.execute("SELECT 42").fetchone() == (42,)


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_s3_public_object_file_preserves_values_and_verdicts(s3, clean):
    url = s3.public_url(s3.key(clean))
    source = (
        SourceRegistry()
        .discover(use_entry_points=False)
        .build(
            {"type": "file", "path": url},
            SourceEnvironment(),
        )
    )
    assert source.load().sort("ordinal").to_dicts() == corpus(clean).to_dicts()
    assert_exact_contract(validate(url, config=configuration()), clean=clean)


def test_missing_s3_object_is_unavailable(s3):
    assert_unavailable(validate(s3.public_url("missing.parquet"), config=configuration()))
