"""Assemble named input contracts before adapting or executing any input."""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from pydantic import BaseModel

from qavf.settings import (
    DatasetSpec,
    QAVFSettings,
    SchemaSpec,
    SourcePolicy,
    _read_config_mapping,
    check_sql_names,
    validate_config_mapping_keys,
)
from qavf.sources.base import DataSource, SourceEnvironment, SourceError
from qavf.sources.frames import as_polars, supports_frame
from qavf.sources.parameters import FileParams

from .adapters import _LoaderInput, _QueryInput
from .descriptors import Dataset, File, Loader, Query
from .resources import ResourceLimits

Configuration = QAVFSettings | Mapping[str, Any] | str | os.PathLike[str] | None


def _plain(value: Any) -> Any:
    """Revalidate mutated model instances while retaining their omission serializers."""
    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_plain(child) for child in value)
    return value


def _configuration(config: Configuration) -> tuple[dict[str, Any], Path | None]:
    if config is None:
        return {}, None
    if isinstance(config, QAVFSettings):
        copied = config.model_copy(deep=True)
        # Keep nested model field sets: omission remains meaningful for policies
        # and disabled plugin parameters, including mutated parameter mappings.
        return {
            name: getattr(copied, name)
            for name in QAVFSettings.model_fields
            if name != "source_path"
        }, copied.source_path
    if isinstance(config, str | os.PathLike):
        return _read_config_mapping(Path(config))
    if isinstance(config, Mapping):
        return dict(config), None
    raise TypeError("config must be settings, a configuration mapping, or a YAML/JSON path")


def _validate_names(names: Mapping[str, object]) -> None:
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise TypeError("dataset names must be nonempty strings")
    check_sql_names(names, "dataset")


def _inputs(data: object, configured: Mapping[str, object]) -> dict[str, object]:
    if data is None:
        return {}
    if isinstance(data, Mapping):
        inputs = dict(data)
        _validate_names(inputs)
    else:
        if len(configured) > 1:
            raise ValueError("a single input is ambiguous; pass a mapping of dataset names")
        inputs = {next(iter(configured), "data"): data}
    if configured and (unknown := sorted(inputs.keys() - configured.keys())):
        raise ValueError(f"input names are not configured datasets: {', '.join(unknown)}")
    return inputs


def _declared(spec: DatasetSpec, name: str) -> bool:
    value = getattr(spec, name)
    if value is None:
        return False
    default = DatasetSpec.model_fields[name].get_default(call_default_factory=True)
    return name in spec.model_fields_set or value != default


def _source_policy(value: object) -> dict[str, Any]:
    """Existing source safeguards count even when a subclass defines a new default."""
    if not isinstance(value, DataSource):
        return {}
    defaults = SourcePolicy()
    return {
        name: getattr(value.params, name)
        for name in SourcePolicy.model_fields
        if name in value.params.model_fields_set
        or getattr(value.params, name) != getattr(defaults, name)
    }


def _merge_contract(name: str, spec: DatasetSpec, supplied: Dataset) -> DatasetSpec:
    inline = supplied.contract()
    updates = {}
    for field in inline.model_fields_set - {"load_policy"}:
        if _declared(spec, field):
            label = "schema" if field == "schema_spec" else field
            raise ValueError(f"dataset {name!r}: {label} conflicts with configured {label}")
        updates[field] = getattr(inline, field)
    if "load_policy" in inline.model_fields_set:
        configured = spec.effective_load_policy().model_dump(exclude_unset=True)
        inherited = {**_source_policy(supplied.data), **configured}
        additions = inline.load_policy.model_dump(exclude_unset=True)
        conflicts = sorted(
            field
            for field, value in additions.items()
            if field in inherited and inherited[field] != value
        )
        if conflicts:
            raise ValueError(
                f"dataset {name!r}: load_policy conflicts with existing policy: "
                f"{', '.join(conflicts)}"
            )
        updates["load_policy"] = SourcePolicy.model_validate({**configured, **additions})
    return spec.model_copy(update=updates)


def _replace_input(spec: DatasetSpec, value: object) -> DatasetSpec:
    policy = spec.effective_load_policy()
    raw = spec.model_dump(by_alias=True)
    raw.update(path=None, source=None, format=None)
    raw["load_policy"] = policy.model_dump(exclude_unset=True)
    if isinstance(value, File):
        raw["source"] = value.source_config()
    elif isinstance(value, str | os.PathLike):
        path = os.fspath(value)
        if not isinstance(path, str):
            raise TypeError("input paths must resolve to strings, not bytes")
        configured = spec.source_config()
        file_config = (
            configured
            if configured is not None and configured.get("type") == "file"
            else {"type": "file"}
        )
        raw["source"] = {**file_config, "path": path}
    return DatasetSpec.model_validate(raw)


def _adapt(inputs: Mapping[str, object], settings: QAVFSettings) -> dict[str, object]:
    if settings.engine.execution_mode == "isolated":
        from qavf.execution.inputs import isolated_inputs

        return isolated_inputs(inputs, settings)
    # Validate every representation before the first conversion. An invalid
    # sibling must not consume an earlier caller-owned Arrow protocol object.
    known = (File, str, os.PathLike, Query, Loader, DataSource, pl.DataFrame, pl.LazyFrame)
    for name, value in inputs.items():
        if isinstance(value, known):
            if (
                isinstance(value, Query)
                and isinstance(value.connection, str)
                and value.connection not in settings.connections
            ):
                raise ValueError("Query refers to an unknown configured connection")
            continue
        if callable(getattr(value, "read_next_batch", None)):
            raise TypeError("a borrowed one-pass reader requires an explicit source adapter")
        if not supports_frame(value):
            raise TypeError(
                f"unsupported input for {name!r}; provide a path, File, supported frame, "
                "Query, Loader, or DataSource. A mapping denotes named datasets, not columns."
            )
    env = SourceEnvironment(
        base_dir=settings.base_dir,
        connections=settings.connections,
        threads=settings.engine.native_threads,
        memory_limit=settings.engine.duckdb_memory_limit,
    )
    frames: dict[str, object] = {}
    borrowed_locks: dict[int, threading.Lock] = {}
    for name, value in inputs.items():
        policy = settings.datasets[name].effective_load_policy()
        if isinstance(value, File | str | os.PathLike):
            continue
        if isinstance(value, Query):
            lock = (
                borrowed_locks.setdefault(id(value.connection), threading.Lock())
                if isinstance(value.connection, duckdb.DuckDBPyConnection)
                else threading.Lock()
            )
            frames[name] = _QueryInput(value, env, lock, policy)
        elif isinstance(value, Loader):
            frames[name] = _LoaderInput(value, env, policy)
        elif isinstance(value, DataSource | pl.DataFrame | pl.LazyFrame):
            frames[name] = value
        else:
            if callable(getattr(value, "read_next_batch", None)):
                raise TypeError("a borrowed one-pass reader requires an explicit source adapter")
            try:
                frames[name] = as_polars(value)
            except SourceError as exc:
                raise TypeError(
                    f"unsupported input for {name!r}; provide a path, File, supported frame, "
                    "Query, Loader, or DataSource. A mapping denotes named datasets, not columns."
                ) from exc
    return frames


def prepare(
    data: object,
    *,
    schema: SchemaSpec | Mapping[str, Any] | None,
    config: Configuration,
    resources: ResourceLimits | None,
) -> tuple[QAVFSettings, dict[str, object]]:
    """Normalize once, retaining caller ownership and validating before adaptation."""
    raw, source_path = _configuration(config)
    validate_config_mapping_keys(raw)
    configured = raw.get("datasets", {})
    if not isinstance(configured, Mapping):
        raise TypeError("config.datasets must be a mapping of named dataset specifications")
    _validate_names(configured)
    inputs = _inputs(data, configured)
    if not inputs and not configured:
        raise ValueError("no input datasets were provided or configured")
    inline_contract = any(
        isinstance(value, Dataset)
        and bool(value.contract().model_fields_set & {"schema_spec", "foreign_keys", "expressions"})
        for value in inputs.values()
    )
    if schema is None and config is None and not inline_contract:
        raise ValueError(
            "provide schema=, config=, or a Dataset QA contract; acceptance limits are not inferred"
        )
    specs = {
        name: DatasetSpec.model_validate(value).model_copy(deep=True)
        for name, value in configured.items()
    }
    for name, value in inputs.items():
        spec = specs.get(name, DatasetSpec())
        if isinstance(value, Dataset):
            spec = _merge_contract(name, spec, value)
            inputs[name] = value.data
        specs[name] = spec
    if schema is not None:
        if len(specs) != 1:
            raise ValueError(
                "schema= is only valid for one dataset; use Dataset or config for named schemas"
            )
        name, spec = next(iter(specs.items()))
        if spec.schema_spec is not None:
            raise ValueError(
                "schema= conflicts with the schema already present in Dataset or config"
            )
        contract = SchemaSpec.model_validate(schema)
        specs[name] = spec.model_copy(update={"schema_spec": contract})
    for name, value in inputs.items():
        specs[name] = _replace_input(specs[name], value)
    raw["datasets"] = specs
    if resources is not None:
        if not isinstance(resources, ResourceLimits):
            raise TypeError("resources must be ResourceLimits")
        engine = raw.get("engine", {})
        if hasattr(engine, "model_dump"):
            engine = engine.model_dump()
        if not isinstance(engine, Mapping):
            raise TypeError("config.engine must be a mapping or EngineSettings")
        raw["engine"] = {**engine, **resources.engine_overrides()}
    settings = QAVFSettings.model_validate(_plain(raw))
    if source_path is not None:
        settings.source_path = source_path
    for spec in settings.datasets.values():
        source = spec.source_config()
        if source is not None and source.get("type") == "file":
            # Built-in file settings are statically checkable. Validate every
            # selected reader before converting any caller-owned frame.
            FileParams.model_validate(source)
    return settings, _adapt(inputs, settings)
