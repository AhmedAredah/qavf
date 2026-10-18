"""Actual remote reads through the configured PostgreSQL database/query adapters."""

import pytest

from qavf import Query, validate
from qavf.settings import ConnectionSpec
from qavf.sources import SourceEnvironment, SourceRegistry

from .contracts import assert_exact_contract, assert_unavailable, configuration, corpus
from .postgres_service import postgres_corpus
from .pytest_support import available_service

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def postgres(live_required):
    with available_service(postgres_corpus) as service:
        yield service


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
def test_postgres_database_read_preserves_exact_values(postgres, clean):
    connections = {
        name: ConnectionSpec.model_validate(value)
        for name, value in postgres.connection_config().items()
    }
    source = (
        SourceRegistry()
        .discover(use_entry_points=False)
        .build(
            {
                "type": "database",
                "connection": "warehouse",
                "schema": postgres.schema,
                "table": "clean" if clean else "invalid",
            },
            SourceEnvironment(connections=connections),
        )
    )
    actual = source.load().sort("ordinal")
    assert actual.to_dicts() == corpus(clean).to_dicts()


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "invalid"])
@pytest.mark.parametrize("snapshot", [False, True], ids=["native", "snapshot"])
def test_postgres_public_query_matches_exact_qa_contract(postgres, clean, snapshot):
    config = configuration()
    config["connections"] = postgres.connection_config(snapshot=snapshot)
    query = Query(
        f"SELECT * FROM warehouse.{postgres.table(clean)} WHERE ordinal >= $first",
        connection="warehouse",
        parameters={"first": 0},
    )
    assert_exact_contract(validate(query, config=config), clean=clean)


def test_missing_remote_table_is_unavailable(postgres):
    config = configuration()
    config["connections"] = postgres.connection_config()
    query = Query(f"SELECT * FROM warehouse.{postgres.schema}.missing", connection="warehouse")
    assert_unavailable(validate(query, config=config))
