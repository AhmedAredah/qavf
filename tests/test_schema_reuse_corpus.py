"""The editor JSON Schemas accept the reusable configurations the runtime accepts
(review finding CFG-05), and advertise ranges only where they expand (R7)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from qavf.cli_core import config_json_schema as native_json_schema
from qavf.engine.runner import build_source_registry
from qavf.scaffold import config_json_schema
from qavf.settings import QAVFSettings, settings_from_text
from qavf.testing import discovered_rules

ROOT = Path(__file__).resolve().parents[1]
_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.DOTALL)


def _full_schema() -> dict[str, Any]:
    return config_json_schema(discovered_rules(), build_source_registry(QAVFSettings()))


def _native_schema() -> dict[str, Any]:
    return native_json_schema(QAVFSettings())


FULL = _full_schema()
NATIVE = _native_schema()
ROOT_KEYS = set(FULL["properties"])


def _errors(schema: dict[str, Any], document: Any) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{list(e.absolute_path)}: {e.message[:200]}" for e in validator.iter_errors(document)]


REUSABLE = """
vars: {minimum: 3, active: false, allowed: [ok, bad], level: WARNING}
include: child.yaml
datasets:
  x:
    path: x.csv
    schema:
      min_rows: ${vars.minimum}
      columns:
        state: {dtype: str, isin: '${vars.allowed}'}
    checks:
      - {check: expression, expr: 'TRUE', enabled: '${vars.active}', severity: '${vars.level}'}
"""


@pytest.mark.parametrize("schema", [FULL, NATIVE], ids=["full", "native"])
def test_both_schemas_accept_variables_everywhere_and_a_single_include(schema, tmp_path):
    (tmp_path / "child.yaml").write_text("project: child\n", encoding="utf-8")
    settings_from_text(REUSABLE, tmp_path / "qa.yaml")  # the runtime accepts it
    assert not _errors(schema, yaml.safe_load(REUSABLE))


def _item(check: dict[str, Any]) -> dict[str, Any]:
    return {"datasets": {"d": {"checks": [check]}}}


@pytest.mark.parametrize(
    ("check", "accepted"),
    [
        ({"check": "key_grid", "dimensions": {"year": {"values": "2016..2019"}}}, True),
        ({"check": "key_grid", "dimensions": {"year": {"values": {"from": 1, "to": 3}}}}, True),
        ({"check": "missing_periods", "series": {"value": "v"}, "expected": "2016..2019"}, True),
        ({"check": "change_rate", "series": {"value": "v"}, "max": "1..3"}, False),
        ({"check": "change_rate", "series": {"value": "v"}, "max": {"from": 1, "to": 3}}, False),
        ({"check": "change_rate", "series": {"value": "v"}, "tags": "1..3"}, False),
        ({"check": "change_rate", "series": {"value": "v"}, "max": "${vars.limit}"}, True),
    ],
)
def test_ranges_are_advertised_only_on_list_settings(check, accepted):
    assert (not _errors(FULL, _item(check))) == accepted


def _documents() -> list[tuple[str, str]]:
    found = [(str(p.relative_to(ROOT)), p.read_text(encoding="utf-8")) for p in _files()]
    documents = []
    for name, text in found:
        if name.endswith((".yaml", ".yml")):
            documents.append((name, text))
            continue
        for index, block in enumerate(_YAML_BLOCK.findall(text)):
            documents.append((f"{name}#{index}", block))
    return documents


def _files() -> list[Path]:
    files = sorted((ROOT / "config").glob("*.yaml"))
    files.append(ROOT / "qavf" / "demo_config.yaml")
    files.append(ROOT / "README.md")
    # The plugin guide's examples name source types and checks of plugins that
    # are not installed here.
    files.extend(p for p in sorted((ROOT / "docs").glob("*.md")) if p.name != "plugins.md")
    return files


def _document(text: str) -> Any:
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict) or not document or not set(document) <= ROOT_KEYS:
        return None
    return json.loads(json.dumps(document, default=str))


def _runtime_accepts(text: str, directory: Path) -> bool:
    try:
        settings_from_text(text, directory / "qa.yaml")
    except (ValueError, OSError):
        return False
    return True


CORPUS = [(name, text) for name, text in _documents() if _document(text) is not None]


def test_the_corpus_is_not_empty():
    names = {name.split("#")[0] for name, _ in CORPUS}
    assert {"config/qavf.full.yaml", "README.md"} <= {n.replace("\\", "/") for n in names}


@pytest.mark.parametrize(("name", "text"), CORPUS, ids=[name for name, _ in CORPUS])
def test_documented_configurations_the_runtime_accepts_match_the_full_schema(name, text, tmp_path):
    if not _runtime_accepts(text, tmp_path):
        pytest.skip("a fragment the runtime needs more context for")
    assert not _errors(FULL, _document(text)), name


def _native_document(text: str) -> bool:
    document = _document(text)
    checks = list(document.get("checks") or [])
    for spec in (document.get("datasets") or {}).values():
        checks.extend((spec or {}).get("checks") or [])
    return all(isinstance(c, dict) and c.get("check") in ("expression", "sql") for c in checks)


@pytest.mark.parametrize(("name", "text"), CORPUS, ids=[name for name, _ in CORPUS])
def test_documented_row_check_configurations_match_the_native_schema(name, text, tmp_path):
    if not _native_document(text) or not _runtime_accepts(text, tmp_path):
        pytest.skip("not a configuration the native backend runs")
    assert not _errors(NATIVE, _document(text)), name


@pytest.mark.parametrize("schema", [FULL, NATIVE], ids=["full", "native"])
def test_both_schemas_accept_disabled_row_check_stubs(schema):
    stubs = {
        "datasets": {"d": {"checks": [{"check": "expression", "enabled": False}]}},
        "checks": [{"check": "sql", "enabled": False}],
    }
    assert not _errors(schema, stubs)
    assert _errors(schema, {"datasets": {"d": {"checks": [{"check": "expression"}]}}})


RESOURCES = """
vars: {memory: 2GiB, spill: 20GiB, workers: 4, backend: duckdb}
resources:
  target_memory: ${vars.memory}
  max_spill: ${vars.spill}
  max_workers: ${vars.workers}
  temp_dir: scratch
engine:
  backend: ${vars.backend}
  cache_max_bytes: 256MiB
overrides:
  resources: {max_spill: 30GiB}
"""


@pytest.mark.parametrize("schema", [FULL, NATIVE], ids=["full", "native"])
def test_both_schemas_accept_resources_with_sizes_and_variables(schema, tmp_path):
    settings_from_text(RESOURCES, tmp_path / "qa.yaml")  # the runtime accepts it
    assert not _errors(schema, yaml.safe_load(RESOURCES))
    assert _errors(schema, {"resources": {"target_memory": True}})
    assert _errors(schema, {"resources": {"memory": "2GiB"}})
    assert _errors(schema, {"engine": {"backend": "spark"}})
