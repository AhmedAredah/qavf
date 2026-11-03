"""Source identity and bounded admission regressions for the native engine."""

from contextlib import contextmanager, suppress

import duckdb
import pyarrow as pa
import pytest
from pydantic import ValidationError

from qavf import BatchSource, File, Loader, Query, validate
from qavf.core.sources import InputReader
from qavf.settings import DatasetSpec, QAVFSettings
from qavf.sources.base import SourceError, SourceIntegrityError


@pytest.fixture
def source_reader():
    with duckdb.connect(config={"threads": 1}) as connection:
        settings = QAVFSettings.model_validate({"datasets": {"data": {}}})
        yield InputReader(connection, settings), connection


def test_user_rowid_never_controls_evidence_or_condition_scope():
    report = validate(
        pa.table({"rowid": [100, 100, 200], "v": [1, 1, 2]}),
        schema={"columns": {"v": {"dtype": "int", "le": 0}}, "primary_key": ["v"]},
        backend="duckdb",
    )
    duplicate = next(r for r in report.results if r.rule_id == "INP-004")
    assert duplicate.affected_records[0]["rows"] == [0, 1]
    scoped = validate(
        pa.table({"rowid": [0, 0], "x": [1, -1], "state": ["active", "inactive"]}),
        schema={
            "columns": {
                "x": {"dtype": "int", "conditions": [{"where": "state = 'active'", "ge": 0}]}
            }
        },
        backend="duckdb",
    )
    assert scoped.verdict() == "passed"


def test_internal_position_names_are_case_insensitive(source_reader):
    reader, connection = source_reader
    loaded = reader.load("data", pa.table({"_ROW": [100, 200]}), DatasetSpec())
    assert loaded.row_column == "qavf__row"
    assert connection.execute(f'SELECT "{loaded.row_column}" FROM "{loaded.table}"').fetchall() == [
        (0,),
        (1,),
    ]


@pytest.mark.parametrize("header", ["x,x", "x,X", "x,", ",x", '"x",x'])
def test_native_csv_rejects_ambiguous_headers(tmp_path, source_reader, header):
    reader, _ = source_reader
    path = tmp_path / "bad.csv"
    path.write_text(header + "\n1,9\n")
    with pytest.raises(SourceError, match=r"blank|duplicate|collid"):
        reader.load("data", File(path), DatasetSpec())


def test_no_header_option_and_real_generated_names_remain_valid(tmp_path, source_reader):
    reader, _ = source_reader
    path = tmp_path / "data.csv"
    path.write_text("1,2\n3,4\n")
    assert reader.load("data", File(path, options={"header": False}), DatasetSpec()).rows == 2
    path.write_text("x,x_1,column2\n1,2,3\n")
    assert reader.load("data", File(path), DatasetSpec()).columns == ("x", "x_1", "column2")


def test_arrow_case_collisions_rejected_before_registration(source_reader):
    reader, connection = source_reader
    with pytest.raises(SourceError, match="case-colliding"):
        reader.load("data", pa.table({"x": [1], "X": [2]}), DatasetSpec())
    assert connection.execute("SHOW TABLES").fetchall() == []


def test_complete_source_mapping_validated(source_reader, tmp_path):
    reader, _ = source_reader
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    with pytest.raises(ValidationError, match="columnz"):
        reader.load("data", {"type": "file", "path": str(path), "columnz": ["x"]}, DatasetSpec())
    with pytest.raises(ValidationError, match="paramter"):
        reader.load("data", {"type": "query", "sql": "SELECT 1", "paramter": {}}, DatasetSpec())


def test_renames_are_simultaneous_and_projection_metadata_complete(source_reader):
    reader, connection = source_reader
    reader.projections = {"data": ("a",)}
    loaded = reader.load(
        "data",
        pa.table({"a": [1], "b": [2], "wide": ["unused"]}),
        DatasetSpec(rename={"a": "b", "b": "a"}),
    )
    assert loaded.columns == ("a",)
    assert loaded.original_columns == ("b", "a", "wide")
    assert set(loaded.original_types) == {"a", "b", "wide"}
    assert connection.execute(f'SELECT a FROM "{loaded.table}"').fetchone() == (2,)
    reader.release(loaded)
    assert connection.execute("SHOW TABLES").fetchall() == []


def test_borrowed_one_pass_reader_rejected_without_consumption(source_reader):
    reader, _ = source_reader
    batch = pa.record_batch([[1, 2]], names=["x"])
    stream = pa.RecordBatchReader.from_batches(batch.schema, [batch])
    with pytest.raises(TypeError, match=r"one-pass.*BatchSource"):
        reader.load("data", stream, DatasetSpec())
    assert stream.read_next_batch().equals(batch)


def test_known_row_budget_rejected_before_registration(source_reader):
    reader, connection = source_reader
    with pytest.raises(SourceError, match="row_budget=2"):
        reader.load(
            "data", pa.table({"x": range(100_000)}), DatasetSpec(load_policy={"row_budget": 2})
        )
    assert connection.execute("SHOW TABLES").fetchall() == []


def test_query_row_budget_limits_materialization_and_cleans_up(source_reader):
    reader, connection = source_reader
    with pytest.raises(SourceError, match="row_budget=2"):
        reader.load(
            "data",
            Query("SELECT i FROM range(1000000) r(i)"),
            DatasetSpec(load_policy={"row_budget": 2}),
        )
    assert connection.execute("SHOW TABLES").fetchall() == []


@pytest.mark.parametrize("probe", ["optional", "projection", "ordinary"])
@pytest.mark.parametrize(
    "text, message", [("x,y\n1,2,3\n", "too many columns"), ('x,y\n1,"two\n', "unterminated")]
)
def test_csv_bind_failures_get_row_and_line(tmp_path, source_reader, probe, text, message):
    reader, _ = source_reader
    path = tmp_path / "bad.csv"
    path.write_text(text)
    spec = (
        DatasetSpec.model_validate({"schema": {"columns": {"z": {"required": False}}}})
        if probe == "optional"
        else DatasetSpec()
    )
    source = File(path, columns=["x"]) if probe == "projection" else File(path)
    with pytest.raises(SourceError, match=f"CSV row 2, line 2: {message}"):
        reader.load("data", source, spec)


def test_owned_batches_are_complete_projected_and_closed(source_reader):
    reader, connection = source_reader
    events = []

    @contextmanager
    def factory():
        events.append("open")
        try:
            yield iter(
                [
                    pa.table({"x": [1, 2], "unused": ["a", "b"]}),
                    pa.table({"x": [3], "unused": ["c"]}),
                ]
            )
        finally:
            events.append("close")

    reader.projections = {"data": ("x",)}
    source = BatchSource(factory, {"x": pa.int64(), "unused": pa.string()})
    loaded = reader.load("data", source, DatasetSpec())
    assert loaded.rows == 3 and loaded.columns == ("x",)
    assert loaded.original_columns == ("x", "unused")
    assert events == ["open", "close"]
    assert connection.execute(
        f'SELECT "{loaded.row_column}", x FROM "{loaded.table}"'
    ).fetchall() == [(0, 1), (1, 2), (2, 3)]


def test_suppressing_batch_manager_cannot_accept_partial_data(source_reader):
    reader, connection = source_reader

    @contextmanager
    def factory():
        with suppress(SourceError):
            yield iter([pa.table({"x": [1]}), pa.table({"x": [2]})])

    source = BatchSource(factory, pa.schema([("x", pa.int64())]))
    with pytest.raises(SourceError, match="row_budget"):
        reader.load("data", source, DatasetSpec(load_policy={"row_budget": 1}))
    assert connection.execute("SHOW TABLES").fetchall() == []


def test_managed_retries_discard_failed_snapshot(source_reader):
    reader, connection = source_reader
    attempts = []

    @contextmanager
    def factory():
        attempts.append(len(attempts))

        def batches():
            yield pa.table({"x": [10 if len(attempts) == 1 else 20]})
            if len(attempts) == 1:
                raise SourceError("transient", retryable=True)
            yield pa.table({"x": [30]})

        yield batches()

    source = BatchSource(factory, {"x": pa.int64()}, replayable=True, retry_safe=True)
    loaded = reader.load(
        "data", source, DatasetSpec(load_policy={"retries": 1, "retry_delay_s": 0})
    )
    assert len(attempts) == 2 and loaded.rows == 2
    assert connection.execute(f'SELECT x FROM "{loaded.table}"').fetchall() == [(20,), (30,)]


def test_hash_identity_survives_snapshot_release(source_reader, tmp_path):
    reader, _ = source_reader
    reader.settings.report.hash_inputs = True
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    loaded = reader.load("data", File(path), DatasetSpec())
    assert loaded.metadata["fingerprint"]["identity"] == "content"
    assert len(loaded.metadata["fingerprint"]["sha256"]) == 64
    reader.release(loaded)
    path.write_text("x\n2\n")
    with pytest.raises(SourceIntegrityError, match="changed"):
        reader.verify(loaded)


def test_mapping_policy_is_not_lost_during_adaptation(source_reader, tmp_path):
    reader, _ = source_reader
    path = tmp_path / "budget.csv"
    path.write_text("x\n1\n2\n3\n")
    with pytest.raises(SourceError, match="row_budget=2"):
        reader.load("data", {"path": str(path), "row_budget": 2}, DatasetSpec())


def test_loader_retry_requires_declared_replay_safety(source_reader):
    reader, _ = source_reader
    calls = []

    def factory():
        calls.append(1)
        raise SourceError("transient", retryable=True)

    with pytest.raises(SourceError, match="transient"):
        reader.load(
            "data", Loader(factory), DatasetSpec(load_policy={"retries": 1, "retry_delay_s": 0})
        )
    assert len(calls) == 1


def test_borrowed_query_projection_keeps_original_schema_and_connection(source_reader):
    reader, _ = source_reader
    reader.projections = {"data": ("x",)}
    with duckdb.connect(config={"threads": 1}) as borrowed:
        loaded = reader.load(
            "data", Query("SELECT 1 AS x, 'unused' AS wide", connection=borrowed), DatasetSpec()
        )
        assert loaded.columns == ("x",)
        assert loaded.original_columns == ("x", "wide")
        reader.release(loaded)
        assert borrowed.execute("SELECT 1").fetchone() == (1,)


def test_glob_checks_headers_in_every_file(source_reader, tmp_path):
    reader, _ = source_reader
    (tmp_path / "a.csv").write_text("x,y\n1,2\n")
    (tmp_path / "b.csv").write_text("x,x\n1,2\n")
    with pytest.raises(SourceError, match="duplicate"):
        reader.load("data", File(tmp_path / "*.csv"), DatasetSpec())


def test_core_only_managed_arrow_batch_does_not_import_polars(tmp_path):
    import subprocess
    import sys

    program = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "polars" or fullname.startswith("polars."):
            raise ImportError("Polars intentionally blocked")
sys.meta_path.insert(0, Block())
from contextlib import contextmanager
import pyarrow as pa
from qavf import BatchSource, validate
@contextmanager
def batches():
    yield iter([pa.table({"x": [1, 2]}), pa.table({"x": [3]})])
report = validate(
    BatchSource(batches, {"x": pa.int64()}),
    schema={"columns": {"x": {"dtype": "int", "ge": 0}}, "min_rows": 3},
    backend="duckdb",
)
assert report.verdict() == "passed", [(r.check, r.message) for r in report.results]
assert "polars" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", program], check=True)


@pytest.mark.parametrize("backend", ["duckdb", "polars"])
def test_arrow_batch_schema_works_across_backends(backend):
    @contextmanager
    def factory():
        yield iter([pa.table({"x": [1, 2]})])

    report = validate(
        BatchSource(factory, pa.schema([("x", pa.int64())])),
        schema={"columns": {"x": {"dtype": "int", "ge": 0}}},
        backend=backend,
    )
    assert report.verdict() == "passed"


def test_legacy_polars_batch_schema_accepts_arrow_string_batches(source_reader):
    import polars as pl

    reader, _ = source_reader

    @contextmanager
    def factory():
        yield iter([pa.table({"x": ["a", "b"]})])

    loaded = reader.load("data", BatchSource(factory, {"x": pl.String}), DatasetSpec())
    assert loaded.rows == 2 and loaded.columns == ("x",)


def test_standalone_source_type_contract_is_not_silently_ignored(source_reader, tmp_path):
    from qavf.optional import OptionalDependencyError

    reader, _ = source_reader
    path = tmp_path / "typed.csv"
    path.write_text("x\ninvalid\n")
    with pytest.raises(OptionalDependencyError, match="declared_columns"):
        reader.load(
            "data",
            {"path": str(path), "declared_columns": {"x": {"dtype": "int"}}},
            DatasetSpec(),
        )


def test_query_mapping_keeps_borrowed_connection_ownership(source_reader):
    reader, _ = source_reader
    reader.query_resolver = lambda *_: pytest.fail("borrowed connection must bypass attachments")
    with duckdb.connect(config={"threads": 1}) as borrowed:
        loaded = reader.load(
            "data",
            {"type": "query", "sql": "SELECT 1 AS x", "connection": borrowed},
            DatasetSpec(),
        )
        reader.release(loaded)
        assert borrowed.execute("SELECT 2").fetchone() == (2,)


def test_absent_optional_rename_retains_existing_noop_behavior(source_reader):
    reader, _ = source_reader
    loaded = reader.load(
        "data", pa.table({"present": [1]}), DatasetSpec(rename={"absent": "optional"})
    )
    assert loaded.columns == ("present",)


def test_zoned_nanoseconds_are_zero_copy_and_keep_source_metadata(source_reader):
    from qavf.core.temporal_admission import TemporalAdmission

    reader, connection = source_reader
    dtype = pa.timestamp("ns", tz="UTC")
    values = pa.array([123456789, 123456788], type=dtype)
    table = pa.table({"t": values})
    spec = DatasetSpec.model_validate(
        {
            "schema": {
                "columns": {"t": {"dtype": "datetime", "datetime_unit": "ns", "timezone": "UTC"}}
            }
        }
    )
    native, notes = TemporalAdmission(spec).eager(table)
    assert native["t"].chunk(0).buffers()[1].address == values.buffers()[1].address
    assert native["t"].cast(pa.int64()).to_pylist() == [123456789, 123456788]
    loaded = reader.load("data", table, spec)
    assert loaded.types["t"] == "TIMESTAMP_NS"
    assert loaded.original_types["t"] == "timestamp[ns, tz=UTC]"
    assert notes["t"]["canonical_timezone"] == "UTC"
    assert connection.execute(f'SELECT epoch_ns(t) FROM "{loaded.table}"').fetchall() == [
        (123456789,),
        (123456788,),
    ]


@pytest.mark.parametrize("kind", ["arrow", "pandas", "polars", "batch"])
def test_zoned_nanosecond_adapters_preserve_distinct_keys(kind):
    from qavf import Dataset

    table = pa.table({"t": pa.array([123456789, 123456788], type=pa.timestamp("ns", tz="UTC"))})
    value = table
    if kind == "pandas":
        value = table.to_pandas()
    elif kind == "polars":
        import polars as pl

        value = pl.from_arrow(table)
    elif kind == "batch":

        @contextmanager
        def batches():
            yield iter([table])

        value = BatchSource(batches, table.schema)
    report = validate(
        Dataset(
            value,
            schema={
                "columns": {"t": {"dtype": "datetime", "datetime_unit": "ns", "timezone": "UTC"}},
                "primary_key": ["t"],
            },
        ),
        backend="duckdb",
    )
    assert report.verdict() == "passed", [(r.check, r.message) for r in report.results]


@pytest.mark.parametrize("nested", [False, True])
def test_unguarded_or_nested_zoned_nanoseconds_rejected(source_reader, nested):
    reader, _ = source_reader
    dtype = pa.timestamp("ns", tz="UTC")
    if nested:
        dtype = pa.struct([("time", dtype)])
        table = pa.table({"t": pa.array([{"time": 123456789}], type=dtype)})
    else:
        table = pa.table({"t": pa.array([123456789], type=dtype)})
    with pytest.raises(SourceError, match="nanosecond"):
        reader.load("data", table, DatasetSpec())


def test_native_parquet_rejects_lossy_zoned_nanosecond_bridge(source_reader, tmp_path):
    import pyarrow.parquet as pq

    reader, _ = source_reader
    path = tmp_path / "zoned.parquet"
    pq.write_table(
        pa.table({"t": pa.array([123456789, 123456788], type=pa.timestamp("ns", tz="UTC"))}), path
    )
    with pytest.raises(SourceError, match=r"nanosecond Parquet.*PyArrow"):
        reader.load("data", File(path), DatasetSpec())
