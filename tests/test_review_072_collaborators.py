"""Independent collaborator contracts for the 0.7.2 architecture boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from qavf.engine.batch_input import consume_batches
from qavf.engine.observation_store import BatchSpool, ObservationStorage
from qavf.engine.retention import RetainedBudget
from qavf.engine.spill import SpillBudget
from qavf.engine.sql_read import SqlReadPolicy, SqlReadSession
from qavf.sources.base import SourceError
from qavf.sources.frames import as_polars, supports_frame
from qavf.sources.local_files import LocalManifest


def test_owned_batch_capability_needs_no_engine():
    class Reader:
        schema: ClassVar = {"x": pl.Int64}
        row_budget = 2
        rows = None
        closed = False

        @contextmanager
        def open(self):
            try:
                yield iter([pl.DataFrame({"x": [1, 2]})])
            finally:
                self.closed = True

        def record_rows(self, rows):
            self.rows = rows

    class Sink:
        def __init__(self):
            self.frames = []

        def append(self, frame):
            self.frames.append(frame)

    reader, sink = Reader(), Sink()
    schema = consume_batches(reader, sink, lambda: None, lambda *_: None, {"x": "value"})
    assert schema == {"value": pl.Int64}
    assert reader.closed and reader.rows == 2
    assert sink.frames[0].to_dict(as_series=False) == {"value": [1, 2]}


def test_sql_reader_policy_needs_no_context(tmp_path):
    import duckdb

    policy = SqlReadPolicy(tmp_path, lambda: [], lambda: None, lambda *_: None)
    with duckdb.connect() as connection:
        session = SqlReadSession(policy, connection, {})
        assert session.frame("select 1 as x").to_dict(as_series=False) == {"x": [1]}
        session.close()
        with pytest.raises(RuntimeError, match="ended"):
            session.schema("select 1")


def test_manifest_identity_can_be_tested_without_file_reader(tmp_path):
    path = tmp_path / "data"
    initial = (1, 2, 3, 4, 5)
    manifest = LocalManifest.capture((path,), lambda _: initial)
    manifest.verify(lambda _: initial)
    with pytest.raises(SourceError, match="changed"):
        manifest.verify(lambda _: (2, 2, 3, 4, 5))


@pytest.mark.parametrize("representation", ["int", "string", "nested", "enum", "decimal", "time"])
def test_spool_detaches_mutable_fragments_and_preserves_types(tmp_path, representation):
    from decimal import Decimal

    series = {
        "int": pl.Series("x", [2**60, None, -(2**60)]),
        "string": pl.Series("x", ["é", None, "x" * 100]),
        "nested": pl.Series("x", [[1, None], None, []]),
        "enum": pl.Series("x", ["a", None, "b"], dtype=pl.Enum(["a", "b"])),
        "decimal": pl.Series(
            "x", [Decimal("1.23"), None, Decimal("2.00")], dtype=pl.Decimal(18, 2)
        ),
        "time": pl.Series("x", [1, None, 3], dtype=pl.Datetime("ns")),
    }[representation]
    original = series.to_frame()
    working = RetainedBudget(65536)
    budget = SpillBudget(1024 * 1024)
    storage = ObservationStorage(tmp_path, RetainedBudget(0), working, budget, lambda: None)
    files = []
    spool = BatchSpool(storage, files.append)
    try:
        for batch in original.iter_slices(1):
            spool.append(batch)
            # Modifying the producer's wrapper must not modify retained input.
            batch.replace_column(0, pl.Series("x", [None], dtype=batch.schema["x"]))
        spool.flush()
        result = pl.read_ipc(files[0].path.read_bytes())
        assert result.equals(original) and result.schema == original.schema
        assert len(files) == 1 and working.used == 0
    finally:
        spool.close()
        for owned in files:
            owned.close()
    assert not list(Path(tmp_path).iterdir()) and budget.used == 0


def test_frame_metadata_lookalike_is_not_a_conversion_protocol():
    class Lookalike:
        columns: ClassVar = ["x"]
        dtypes: ClassVar = ["int64"]
        schema: ClassVar = {"x": "int64"}
        num_rows = 1

        def to_numpy(self):
            return [[1]]

    assert not supports_frame(Lookalike())
    with pytest.raises(SourceError, match="expected a DataFrame"):
        as_polars(Lookalike())


def test_builder_reset_and_transfer_keep_spool_publication_connected(tmp_path):
    from qavf.engine.context import DataContext
    from qavf.engine.observation import _ObservationBuilder

    with DataContext(cache_max_bytes=0, temp_directory=tmp_path) as ctx:
        builder = _ObservationBuilder(ctx)
        try:
            for value in (1, 2):
                # Native operation retries reset the builder before consumption.
                builder.close()
                builder.append(pl.DataFrame({"x": [value]}))
                with builder.finish({"x": pl.Int64}) as observed:
                    assert observed.collect(observed.scan()).to_dict(as_series=False) == {
                        "x": [value]
                    }
        finally:
            builder.close()
    assert not list(tmp_path.iterdir())
