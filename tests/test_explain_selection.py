"""Selection- and backend-aware ``explain()`` and the ``qavf explain`` command (UX-01).

The plan applies the same selection and backend admission as ``validate``, never
reads data, and doubles as a configuration validator (exit 2 with field paths).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import qavf
from qavf.checks.selection import SelectionError
from qavf.cli import EXIT_OK, EXIT_USAGE, main
from qavf.optional import OptionalDependencyError

pytest.importorskip("polars")

CONFIG = """\
project: explained
datasets:
  a:
    path: missing_a.csv
    schema: {columns: {id: {dtype: int}}}
    checks:
      - {check: expression, name: positive, expr: "id >= 0", tags: [nightly]}
  b:
    path: missing_b.csv
    schema: {columns: {id: {dtype: int}}}
    checks:
      - {check: not_constant, name: varies, column: id}
plugins: {entry_points: false}
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "qavf.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def _contract_datasets(plan: qavf.ValidationPlan) -> set[str]:
    return {name for rule in plan.rules if rule.rule_id == "INP-002" for name in rule.datasets}


def test_dataset_selection_narrows_the_plan_like_validate(config: Path) -> None:
    plan = qavf.explain(config=config, datasets="a", backend="duckdb")
    assert plan.backend == "duckdb" and plan.selection == "--dataset a"
    assert _contract_datasets(plan) == {"a"}
    assert not any("not_constant" in item for item in plan.unresolved)  # switched off
    assert not plan.data_read


def test_tag_selection_keeps_only_tagged_checks(config: Path) -> None:
    plan = qavf.explain(config=config, tags="nightly", backend="polars")
    assert [rule.rule_id for rule in plan.rules] == ["INP-006"]
    assert plan.excluded["INP-002"] == "not selected for this run"


def test_skip_tags_and_only(config: Path) -> None:
    plan = qavf.explain(config=config, only="positive", backend="polars")
    assert [rule.rule_id for rule in plan.rules] == ["INP-006"]
    skipped = qavf.explain(config=config, skip_tags="nightly", datasets="a", backend="polars")
    assert "INP-006" not in {rule.rule_id for rule in skipped.rules}


def test_unknown_selection_fails_like_validate(config: Path) -> None:
    with pytest.raises(SelectionError, match="no check is named"):
        qavf.explain(config=config, only="nope")
    with pytest.raises(SelectionError, match="unknown dataset"):
        qavf.explain(config=config, datasets="nope")


def test_backend_admission_matches_execution(config: Path) -> None:
    # The DuckDB core cannot run not_constant: refused before any data is read,
    # exactly as validate(backend="duckdb") refuses it.
    with pytest.raises(OptionalDependencyError, match="not_constant"):
        qavf.explain(config=config, backend="duckdb")
    with pytest.raises(OptionalDependencyError, match="not_constant"):
        qavf.validate(config=config, backend="duckdb")
    with pytest.raises(ValueError, match="backend must be"):
        qavf.explain(config=config, backend="spark")


def test_cli_explain_text_and_json(config: Path, tmp_path: Path, capsys) -> None:
    assert main(["explain", "-c", str(config), "--dataset", "a", "--backend", "duckdb"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Configuration valid" in text and "No data was read." in text
    assert "--dataset a" in text
    output = tmp_path / "plan.json"
    arguments = ["explain", "-c", str(config), "--format", "json", "--tags", "nightly"]
    assert main([*arguments, "-o", str(output)]) == EXIT_OK
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["selection"] == "--tags nightly" and plan["data_read"] is False
    assert main([*arguments, "-o", str(output)]) == EXIT_USAGE  # exists, no --force
    capsys.readouterr()


def test_cli_explain_never_writes_over_the_configuration(config: Path, capsys) -> None:
    before = config.read_bytes()
    assert main(["explain", "-c", str(config), "-o", str(config), "--force"]) == EXIT_USAGE
    assert config.read_bytes() == before
    capsys.readouterr()


def test_cli_explain_reports_an_invalid_configuration_with_its_field_path(
    tmp_path: Path, capsys
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "datasets: {a: {path: a.csv, schema: {columns: {x: {dtype: nope}}}}}\n", encoding="utf-8"
    )
    assert main(["explain", "-c", str(bad)]) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "datasets.a.schema.columns.x.dtype" in error and ".py:" not in error


def test_cli_explain_unsupported_backend_check_is_a_usage_error(config: Path, capsys) -> None:
    assert main(["explain", "-c", str(config), "--backend", "duckdb"]) == EXIT_USAGE
    assert "not_constant" in capsys.readouterr().err
