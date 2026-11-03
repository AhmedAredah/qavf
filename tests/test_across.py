"""One dataset across many sources: ``across`` and ``<dataset>@*``.

Each copy is a full dataset: its own source, schema contract, checks and report
entry. The three forms of ``across`` (connection names, copy overrides and a
template) expand before validation, and ``<dataset>@*`` lists every copy wherever
datasets are listed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest
from pydantic import ValidationError

from qavf import validate
from qavf.settings import QAVFSettings

SCHEMA = {
    "columns": {"id": {"dtype": "int"}, "x": {"dtype": "int", "ge": 0}},
    "primary_key": ["id"],
}


def _csv(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path.as_posix()


def _settings(config: dict[str, Any]) -> QAVFSettings:
    return QAVFSettings.model_validate(config)


def _database(path: Path, rows: str) -> str:
    con = duckdb.connect(str(path))
    try:
        con.execute(f"CREATE TABLE counts AS SELECT * FROM (VALUES {rows}) t(id, x)")
    finally:
        con.close()
    return path.as_posix()


# ------------------------------------------------------------------------------ forms


def test_a_list_copies_the_dataset_across_connections() -> None:
    settings = _settings(
        {
            "connections": {
                "houston": {"type": "duckdb", "path": "h.duckdb"},
                "dallas": {"type": "duckdb", "path": "d.duckdb"},
            },
            "datasets": {
                "before": {"path": "b.csv"},
                "counts": {
                    "source": {"type": "database", "table": "counts"},
                    "across": ["houston", "dallas"],
                    "schema": SCHEMA,
                    "checks": [{"check": "expression", "name": "positive", "expr": "x > 0"}],
                },
                "after": {"path": "a.csv"},
            },
        }
    )
    assert list(settings.datasets) == ["before", "counts@houston", "counts@dallas", "after"]
    for copy in ("houston", "dallas"):
        spec = settings.datasets[f"counts@{copy}"]
        assert spec.source_config() == {"type": "database", "table": "counts", "connection": copy}
        assert spec.schema_spec is not None and spec.schema_spec.primary_key == ["id"]
        assert [item.name for item in spec.checks] == ["positive"]
        assert spec.across is None


def test_a_mapping_gives_each_copy_its_own_source_overrides() -> None:
    settings = _settings(
        {
            "connections": {"houston": {"type": "duckdb", "path": "h.duckdb"}},
            "secrets": {
                "s3_archive": {
                    "type": "s3",
                    "key_id": "k",
                    "secret": "s",
                    "scope": ["s3://bucket/"],
                }
            },
            "datasets": {
                "counts": {
                    "source": {"type": "database", "table": "counts", "connection": "houston"},
                    "across": {
                        "houston": None,
                        "old": {"table": "counts_2019"},
                        "archive": {
                            "type": "file",
                            "path": "s3://bucket/2024/counts.parquet",
                            "secret": "s3_archive",
                        },
                        "local": {"type": "file", "path": "data/counts.csv"},
                    },
                }
            },
        }
    )
    sources = {name: spec.source_config() for name, spec in settings.datasets.items()}
    assert sources == {
        "counts@houston": {"type": "database", "table": "counts", "connection": "houston"},
        "counts@old": {"type": "database", "table": "counts_2019", "connection": "houston"},
        "counts@archive": {"type": "file", "path": "s3://bucket/2024/counts.parquet"},
        "counts@local": {"type": "file", "path": "data/counts.csv"},
    }


def test_a_template_fills_its_key_in_the_path_or_table() -> None:
    settings = _settings(
        {
            "connections": {"db": {"type": "duckdb", "path": "x.duckdb"}},
            "datasets": {
                "files": {
                    "path": "data/{city}.csv",
                    "format": "csv",
                    "across": {"city": ["a", "b"]},
                },
                "tables": {
                    "source": {"type": "database", "connection": "db", "table": "counts_{year}"},
                    "across": {"year": [2023, 2024]},
                },
            },
        }
    )
    assert settings.datasets["files@a"].source_config() == {
        "type": "file",
        "path": "data/a.csv",
        "format": "csv",
    }
    assert settings.datasets["tables@2024"].source_config() == {
        "type": "database",
        "connection": "db",
        "table": "counts_2024",
    }


def test_all_copies_are_listed_wherever_datasets_are_listed() -> None:
    settings = _settings(
        {
            "datasets": {
                "counts": {
                    "path": "{c}.csv",
                    "across": {"c": ["a", "b", "c"]},
                    "views": {"recent": {"where": "x > 0"}},
                },
                "other": {"path": "o.csv"},
            },
            "checks": [
                {"check": "batch_consistent", "datasets": ["counts@*"], "column": "x"},
                {"check": "batch_consistent", "datasets": ["counts@*.recent"], "column": "x"},
            ],
            "run": {"require": ["counts@*", "other"]},
        }
    )
    assert settings.checks[0].settings["datasets"] == ["counts@a", "counts@b", "counts@c"]
    assert settings.checks[1].settings["datasets"] == [
        "counts@a.recent",
        "counts@b.recent",
        "counts@c.recent",
    ]
    assert settings.run.require == ["counts@a", "counts@b", "counts@c", "other"]


# ----------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("dataset", "extra", "message"),
    [
        (
            {"source": {"type": "database", "table": "t"}, "across": ["nowhere"]},
            {},
            "unknown connection 'nowhere'",
        ),
        (
            {"path": "x.csv", "across": {"a": {"path": "s3://b/x.csv", "secret": "nope"}}},
            {},
            "unknown secret 'nope'",
        ),
        (
            {
                "path": "x.csv",
                "across": {"a": {"path": "s3://other/x.csv", "secret": "s3_archive"}},
            },
            {
                "secrets": {
                    "s3_archive": {
                        "type": "s3",
                        "key_id": "k",
                        "secret": "s",
                        "scope": ["s3://bucket/"],
                    }
                }
            },
            "does not apply to 's3://other/x.csv'",
        ),
        ({"path": "x.csv", "across": {"city": ["a"]}}, {}, "appears in neither"),
        ({"path": "{city}/{year}.csv", "across": {"city": ["a"]}}, {}, r"\{year\} .* not filled"),
        ({"path": "x_{c}.csv", "across": {"c": ["a"]}}, {"clash": True}, "existing dataset"),
        ({"path": "x.csv", "across": []}, {}, "list the connection names"),
        ({"path": "x.csv", "across": {"a b": {"path": "y.csv"}}}, {}, "copy name 'a b'"),
        ({"across": ["db"]}, {}, "needs a 'path' or 'source'"),
        (
            {"source": {"type": "database", "table": "t"}, "across": ["db", "db"]},
            {},
            "connection 'db' is repeated",
        ),
        ({"path": "x_{y}.csv", "across": {"y": [2023, 2023]}}, {}, "value 2023 is repeated"),
        (
            {"path": "x_{y}.csv", "across": {"y": [2023, "2023"]}},
            {},
            "values 2023 and '2023' both name copy '2023'",
        ),
    ],
)
def test_across_errors(dataset: dict[str, Any], extra: dict[str, Any], message: str) -> None:
    config: dict[str, Any] = {
        "connections": {"db": {"type": "duckdb", "path": "x.duckdb"}},
        "datasets": {"counts": dataset},
    }
    if extra.pop("clash", False):
        config["datasets"]["counts@a"] = {"path": "y.csv"}
    config |= extra
    with pytest.raises(ValidationError, match=message):
        _settings(config)


def test_all_copies_of_a_dataset_without_across_is_an_error() -> None:
    with pytest.raises(ValidationError, match="'other@\\*' names no copies"):
        _settings(
            {
                "datasets": {"counts": {"path": "{c}.csv", "across": {"c": ["a"]}}, "other": {}},
                "checks": [{"check": "batch_consistent", "datasets": ["other@*"], "column": "x"}],
            }
        )


def test_across_on_a_dataset_given_alone_is_an_error() -> None:
    from qavf.settings import DatasetSpec

    with pytest.raises(ValidationError, match="across is expanded"):
        DatasetSpec.model_validate({"path": "x.csv", "across": ["a"]})


# ------------------------------------------------------------------------------- runs


def _copies(tmp_path: Path) -> dict[str, Any]:
    _csv(tmp_path / "a.csv", "id,x\n1,1\n2,2\n")
    _csv(tmp_path / "b.csv", "id,x\n1,1\n1,-2\n")
    _csv(tmp_path / "c.csv", "id,x\n1,1\n2,3\n")
    return {
        "datasets": {
            "counts": {
                "path": (tmp_path / "{site}.csv").as_posix(),
                "across": {"site": ["a", "b", "c"]},
                "schema": SCHEMA,
                "checks": [{"check": "expression", "name": "small", "expr": "x < 3"}],
            }
        }
    }


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_each_copy_runs_its_own_contract_and_checks(tmp_path: Path, backend: str) -> None:
    report = validate(config=_copies(tmp_path), backend=backend)
    assert {d.name for d in report.metadata.datasets} == {"counts@a", "counts@b", "counts@c"}
    failed = {(r.rule_id, tuple(r.datasets)) for r in report.results if r.failed}
    assert ("INP-004", ("counts@b",)) in failed  # duplicate key in b only
    assert ("INP-003", ("counts@b",)) in failed  # x >= 0 in b only
    assert ("INP-006", ("counts@c",)) in failed  # x < 3 in c only
    assert {d for _, datasets in failed for d in datasets} == {"counts@b", "counts@c"}


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_the_dataset_selection_takes_all_copies(tmp_path: Path, backend: str) -> None:
    config = _copies(tmp_path)
    config["datasets"]["other"] = {"path": _csv(tmp_path / "o.csv", "id\n1\n")}
    report = validate(config=config, backend=backend, datasets=["counts@*"])
    assert {d for r in report.results for d in r.datasets} == {"counts@b", "counts@c"}


def test_copies_across_connections_read_their_own_database(tmp_path: Path) -> None:
    config = {
        "connections": {
            "houston": {"type": "duckdb", "path": _database(tmp_path / "h.duckdb", "(1, 1)")},
            "dallas": {"type": "duckdb", "path": _database(tmp_path / "d.duckdb", "(1, -5)")},
        },
        "datasets": {
            "counts": {
                "source": {"type": "database", "table": "counts"},
                "across": ["houston", "dallas"],
                "checks": [{"check": "expression", "name": "positive", "expr": "x > 0"}],
            }
        },
    }
    report = validate(config=config, backend="polars")
    failed = [r.datasets for r in report.results_for("INP-006") if r.failed]
    assert failed == [["counts@dallas"]]
    assert {d.name for d in report.metadata.datasets} == {"counts@houston", "counts@dallas"}
