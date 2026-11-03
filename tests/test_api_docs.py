"""Field and keyword lists in the API and plugin guides match the code they describe."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from qavf.inputs.descriptors import Dataset
from qavf.settings import QAVFSettings

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _sentence(path: Path, start: str) -> str:
    text = " ".join(path.read_text(encoding="utf-8").split())
    begin = text.index(start)
    return text[begin : text.index(".", begin)]


def test_documented_dataset_fields_are_the_descriptor_fields() -> None:
    sentence = _sentence(DOCS / "api.md", "`Dataset` associates an input with its")
    documented = set(re.findall(r"`(\w+)`", sentence)) - {"Dataset"}
    fields = {field.name for field in dataclasses.fields(Dataset) if field.init} - {"data"}
    assert documented == fields


def test_documented_testing_keywords_are_configuration_fields() -> None:
    sentence = _sentence(DOCS / "plugins.md", "`run_rule` / `rule_results` run one rule")
    documented = set(re.findall(r"`(\w+)=`", sentence))
    assert documented and documented <= set(QAVFSettings.model_fields)
