"""Normalize public inputs without instantiating optional execution backends."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from qavf.inputs.descriptors import Dataset
from qavf.optional import unsupported
from qavf.settings import DatasetSpec, QAVFSettings, SchemaSpec, _read_config_mapping

SUPPORTED_RULES = frozenset(f"INP-{n:03d}" for n in (*range(1, 8), 9))


def prepare(data, schema, config, resources, profile):
    """Validate a complete configuration before opening caller inputs."""
    source_path = None
    if isinstance(config, QAVFSettings):
        settings = config.model_copy(deep=True)
        source_path = settings.source_path
        raw = {
            name: getattr(settings, name)
            for name in QAVFSettings.model_fields
            if name != "source_path"
        }
    elif isinstance(config, str | os.PathLike):
        raw, source_path = _read_config_mapping(Path(config))
    elif config is None or isinstance(config, Mapping):
        raw = dict(config or {})
    else:
        raise TypeError("config must be settings, a configuration mapping, or a YAML/JSON path")
    configured = dict(raw.get("datasets", {}))
    if isinstance(data, Mapping):
        inputs = dict(data)
    elif data is None:
        inputs = {}
    else:
        if len(configured) > 1:
            raise ValueError("a single input is ambiguous; pass a mapping of dataset names")
        inputs = {next(iter(configured), "data"): data}
    if configured and inputs.keys() - configured.keys():
        raise ValueError("input names are not configured datasets")
    for name in inputs:
        configured.setdefault(name, {})
    if schema is not None:
        if len(configured) != 1:
            raise ValueError("schema requires exactly one dataset; use Dataset for named schemas")
        name = next(iter(configured))
        current = DatasetSpec.model_validate(configured[name])
        if current.schema_spec is not None:
            raise ValueError("schema conflicts with configured schema")
        configured[name] = current.model_copy(
            update={"schema_spec": SchemaSpec.model_validate(schema)}
        )
    for name, value in list(inputs.items()):
        if isinstance(value, Dataset):
            current = DatasetSpec.model_validate(configured[name])
            contract = value.contract()
            updates = {}
            for field in contract.model_fields_set:
                if field in current.model_fields_set:
                    raise ValueError(f"dataset {name!r}: {field} conflicts with configuration")
                updates[field] = getattr(contract, field)
            configured[name] = current.model_copy(update=updates)
            inputs[name] = value.data
    raw["datasets"] = configured
    if resources is not None:
        engine = raw.get("engine", {})
        original = (
            engine.model_dump(exclude_unset=True) if hasattr(engine, "model_dump") else dict(engine)
        )
        raw["engine"] = {**original, **resources.engine_overrides()}
    settings = QAVFSettings.model_validate(raw)
    settings.source_path = source_path
    if profile is not None:
        from qavf.profiles import apply_profile

        settings = apply_profile(settings, profile)
    _capabilities(settings)
    if not settings.datasets:
        raise ValueError("validation requires at least one dataset")
    for name, spec in settings.datasets.items():
        if name not in inputs:
            source = spec.source_config()
            if source is None:
                raise ValueError(f"dataset {name!r} has no input")
            inputs[name] = source
    return settings, inputs


def _capabilities(settings):
    """Never omit configured work because its optional implementation is absent."""
    for rule_id, rule in settings.rules.items():
        if rule.enabled and rule_id not in SUPPORTED_RULES:
            unsupported(f"rule {rule_id}")
        if rule.enabled and rule.params:
            allowed = (
                {"datasets"}
                if rule_id in {"INP-002", "INP-003", "INP-004", "INP-005", "INP-006", "INP-009"}
                else set()
            )
            if rule.params.keys() - allowed:
                unsupported(f"custom parameters for {rule_id}")
            selected = rule.params.get("datasets")
            if selected is not None and (
                not isinstance(selected, list | tuple)
                or any(name not in settings.datasets for name in selected)
            ):
                raise ValueError(f"{rule_id}.datasets must list configured dataset names")
    plugins = settings.plugins
    if plugins.modules or plugins.paths or plugins.required or plugins.packages != ["qavf.rules"]:
        unsupported("custom rule/source plugins")
    engine = settings.engine
    if "max_spill_files" in engine.model_fields_set:
        unsupported("an explicit spill-file count limit")
