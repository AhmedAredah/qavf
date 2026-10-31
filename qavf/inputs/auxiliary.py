"""Primary input descriptions reused for scoped auxiliary reads under one run owner."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars as pl

from qavf.engine.source_read import SourceReadView
from qavf.inputs.adapters import _BatchInput, _FrameInput, _LoaderInput, _QueryInput
from qavf.inputs.descriptors import BatchSource, File, Loader, Query
from qavf.inputs.normalize import prepare_configuration
from qavf.sources.base import DataSource, SourceError, SourceRegistry
from qavf.sources.frames import supports_frame

if TYPE_CHECKING:
    from qavf.engine.context import DataContext


def project_input(ctx: DataContext, value: object, columns: Iterable[str]) -> pl.DataFrame:
    """Read a descriptor's projection with the parent context's resources and policy.

    ``Dataset`` contracts use the same merge/validation rules as primary inputs.
    Named dataset references remain ``ctx.project(name, columns)``; arbitrary
    mappings are not reinterpreted as source configurations or dataframe columns.
    This function never constructs another DataContext.
    """
    settings, inputs = prepare_configuration(
        {"auxiliary": value},
        schema=None,
        config={"plugins": {"entry_points": False}, "connections": ctx.env.connections},
        resources=None,
    )
    spec = settings.datasets["auxiliary"]
    raw = inputs["auxiliary"]
    policy = spec.effective_load_policy()
    if isinstance(raw, File | str | os.PathLike):
        config = spec.source_config()
        assert config is not None
        registry = ctx._source_registry or SourceRegistry().discover()
        source = registry.build(config, ctx.env)
    elif isinstance(raw, DataSource):
        source = raw.bind(ctx.env)
        additions = policy.model_dump(exclude_unset=True)
        source.params = type(source.params).model_validate(
            {**source.params.model_dump(), **additions}
        )
    elif isinstance(raw, Query):
        source = _QueryInput(raw, ctx.env, threading.Lock(), policy)
    elif isinstance(raw, Loader):
        source = _LoaderInput(raw, ctx.env, policy)
    elif isinstance(raw, BatchSource):
        source = _BatchInput(raw, ctx.env, policy)
    elif isinstance(raw, pl.LazyFrame) or supports_frame(raw):
        source = _FrameInput(raw, ctx.env, policy)
    else:
        raise TypeError("auxiliary input must be a supported frame, path or input descriptor")
    selected = list(dict.fromkeys(columns))

    def attempt() -> pl.DataFrame:
        source.prepare_identity(hash_contents=True)
        view = SourceReadView(ctx.source_read_services(), source)
        try:
            schema = view.schema()
            rename = {old: new for old, new in spec.rename.items() if old in schema}
            public = [rename.get(name, name) for name in schema]
            if len(public) != len(set(public)):
                raise SourceError("auxiliary Dataset.rename produces duplicate columns")
            inverse = {new: old for old, new in rename.items()}
            if missing := set(selected) - set(public):
                raise SourceError(f"auxiliary input lacks column(s) {sorted(missing)!r}")
            raw_columns = [inverse.get(name, name) for name in selected]
            frame = view.project(raw_columns)
            return frame.rename({old: new for old, new in rename.items() if old in raw_columns})
        finally:
            view.close()

    return ctx.run_source_operation([source], attempt)
