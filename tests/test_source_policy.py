"""Complete source policy and registration ownership regression contracts."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from qavf import DataContext
from qavf.engine import connections as connections_module
from qavf.engine.context import DatasetNotAvailableError
from qavf.engine.resources import ResourceLimitError
from qavf.engine.session import Invocation
from qavf.settings import ConnectionSpec
from qavf.sources.base import (
    DataSource,
    ScanPlan,
    SourceEnvironment,
    SourceError,
    SourceIntegrityError,
    SourceParams,
)
from qavf.sources.native import OwnedConnection
from qavf.sources.parameters import QueryParams
from qavf.sources.query import QuerySource


class Native(DataSource):
    kind = 'native'
    Params = SourceParams

    def load(self):
        raise AssertionError('native input must stay native')

    def scan(self):
        return ScanPlan('SELECT backend_read(1) AS x')


@pytest.mark.parametrize('sample', [False, True])
@pytest.mark.parametrize('concurrent, expected', [(False, 1), (True, 2)])
def test_native_operations_obey_serialization(monkeypatch, sample, concurrent, expected):
    active = peak = 0
    lock = threading.Lock()
    start = threading.Barrier(2)
    reading = threading.Barrier(2)

    def read(x):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if concurrent:
                # Query setup can take longer than a scheduling sleep. Meet
                # inside the actual source operation to prove overlap.
                reading.wait(timeout=10)
            else:
                time.sleep(0.08)
        finally:
            with lock:
                active -= 1
        return x

    original = connections_module.connect

    def connect(*args, **kwargs):
        con = original(*args, **kwargs)
        con.create_function('backend_read', read, ['BIGINT'], 'BIGINT', side_effects=True)
        return con

    monkeypatch.setattr(connections_module, 'connect', connect)
    with DataContext(
        sources={n: Native(SourceParams(type='native'), SourceEnvironment()) for n in ('a', 'b')},
        concurrent_loads=concurrent,
    ) as ctx:
        def execute(name):
            start.wait()
            query = f'SELECT x FROM {name}'
            return ctx.sql_sample(query, 1)[0].item() if sample else ctx.sql(query).item()

        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(execute, ['a', 'b'])) == [1, 1]
    assert peak == expected


class Eager(DataSource):
    kind = 'eager'
    Params = SourceParams

    def load(self):
        return pl.DataFrame({'x': [1]})


def test_serial_mixed_native_eager_query_does_not_deadlock():
    with DataContext(
        sources={
            'eager': Eager(SourceParams(type='eager'), SourceEnvironment()),
            'native': QuerySource(QueryParams(sql='SELECT 1 AS x'), SourceEnvironment()),
        },
        concurrent_loads=False,
    ) as ctx, ThreadPoolExecutor(2) as pool:
        query = pool.submit(ctx.sql, 'SELECT a.x FROM native a JOIN eager b USING(x)')
        eager = pool.submit(ctx.get, 'eager')
        assert query.result(timeout=3).item() == 1
        assert eager.result(timeout=3).item() == 1


def test_waiting_serial_source_operation_is_cancellable():
    holding = threading.Event()
    release = threading.Event()
    waiting = threading.Event()
    invocation = Invocation()
    with DataContext(concurrent_loads=False) as ctx, ThreadPoolExecutor(2) as pool:
        def hold():
            with ctx._operations.guard():
                holding.set()
                release.wait(3)

        def wait():
            ctx.begin_rule(invocation)
            try:
                waiting.set()
                return ctx.sql('SELECT 1')
            finally:
                ctx.end_rule()

        first = pool.submit(hold)
        assert holding.wait(1)
        second = pool.submit(wait)
        assert waiting.wait(1)
        ctx.cancel_rule(invocation)
        try:
            with pytest.raises(ResourceLimitError, match='cancelled'):
                second.result(timeout=1)
        finally:
            release.set()
            first.result(timeout=1)


def test_sql_registrations_have_operation_lifetimes_and_telemetry():
    with DataContext(frames={'data': pl.DataFrame({'x': range(1000)})}) as ctx:
        for _ in range(3):
            assert ctx.sql('SELECT sum(x) FROM data').item() == 499500
            assert ctx.resource_usage()['active_sql_registration_bytes_upper_bound'] == 0
            with pytest.raises(duckdb.CatalogException):
                ctx._connection([]).execute('SELECT * FROM data')
        usage = ctx.resource_usage()
        assert usage['sql_frame_registrations'] == 3
        assert usage['peak_sql_registration_bytes_upper_bound'] == 8000


def test_sql_handoff_exports_arrow_columns_before_binding(monkeypatch):
    # Whole-frame export can deadlock beside active Polars kernels on the
    # dependency floor. Keep the boundary deterministic instead of timing it.
    def forbid_frame_export(*args, **kwargs):
        raise AssertionError('SQL handoff must not use DataFrame.to_arrow')

    exports = []

    def register(connection, name, value):
        assert isinstance(value, pa.Table)
        exports.append((value.schema, value.get_total_buffer_size()))
        return connection._connection.register(name, value)

    monkeypatch.setattr(pl.DataFrame, 'to_arrow', forbid_frame_export)
    monkeypatch.setattr(OwnedConnection, 'register', register, raising=False)
    frame = pl.DataFrame({
        'label': ['a' * 20, None, ''],
        'number': [1, 2, 3],
        'time': pl.Series([1, 2, None], dtype=pl.Datetime('ns')),
    })
    with DataContext(frames={'data': frame}) as ctx:
        result = ctx.sql('SELECT * FROM data ORDER BY number')
        assert result.equals(frame) and result.schema == frame.schema
        assert len(exports) == 1
        assert exports[0][0].field('time').type == pa.timestamp('ns')
        usage = ctx.resource_usage()
        assert usage['peak_sql_registration_bytes_upper_bound'] == exports[0][1]
        assert usage['active_sql_registration_bytes_upper_bound'] == 0
        with pytest.raises(duckdb.CatalogException):
            ctx._connection([]).execute('SELECT * FROM data')


def test_sql_binding_failure_releases_earlier_registrations():
    with DataContext(frames={'a': pl.DataFrame({'x': [1]})}) as ctx:
        with pytest.raises(DatasetNotAvailableError):
            ctx.sql('SELECT * FROM a', datasets=['a', 'missing'])
        assert ctx.resource_usage()['active_sql_registration_bytes_upper_bound'] == 0
        with pytest.raises(duckdb.CatalogException):
            ctx._connection([]).execute('SELECT * FROM a')


@pytest.fixture
def snapshot_source(tmp_path):
    path = tmp_path / 'db.duckdb'
    with duckdb.connect(str(path)) as con:
        con.execute('CREATE TABLE t AS SELECT 1 AS x')
    env = SourceEnvironment(connections={
        'w': ConnectionSpec(type='duckdb', path=str(path), snapshot=True)
    })
    return QuerySource(QueryParams(sql='SELECT * FROM w.t', retries=2, retry_delay_s=0), env)


def test_snapshot_transient_failure_retries_complete_attempt(snapshot_source):
    with DataContext(sources={'data': snapshot_source}) as ctx:
        source = ctx._catalog.sources['data']
        original = source.read
        calls = []

        def read(con):
            calls.append(1)
            if len(calls) == 1:
                raise SourceError('temporary', retryable=True)
            return original(con)

        source.read = read
        assert ctx.get('data').item() == 1
        assert len(calls) == 2
        assert ctx.describe()[0].notes['retries_used'] == 1


def test_snapshot_retry_unsafe_source_is_not_retried(snapshot_source):
    snapshot_source.retry_safe = False
    with DataContext(sources={'data': snapshot_source}) as ctx:
        calls = []

        def read(con):
            calls.append(1)
            raise SourceError('temporary', retryable=True)

        ctx._catalog.sources['data'].read = read
        with pytest.raises(DatasetNotAvailableError):
            ctx.get('data')
        assert len(calls) == 1


@pytest.mark.parametrize('failure_at', [1, 2])
def test_snapshot_identity_failure_never_publishes(snapshot_source, failure_at):
    with DataContext(sources={'data': snapshot_source}) as ctx:
        checks = []

        def verify():
            checks.append(1)
            if len(checks) >= failure_at:
                raise SourceIntegrityError('identity changed', retryable=True)

        ctx._catalog.sources['data'].verify_identity = verify
        with pytest.raises(DatasetNotAvailableError):
            ctx.get('data')
        assert len(checks) == failure_at
        assert not ctx.is_materialized('data')
        assert ctx.verify_inputs() == {}  # No successful snapshot was published.


def test_snapshot_identity_checked_before_after_and_at_finalization(snapshot_source):
    with DataContext(sources={'data': snapshot_source}) as ctx:
        checks = []
        ctx._catalog.sources['data'].verify_identity = lambda: checks.append(1)
        assert ctx.get('data').item() == 1
        assert len(checks) == 2
        assert ctx.verify_inputs() == {}
        assert len(checks) == 3


class Typed(DataSource):
    kind = 'typed'
    Params = SourceParams

    def load(self):
        raise AssertionError('typed plan must stay lazy')

    def scan_polars(self):
        return pl.DataFrame({'x': [1, 2], 'ignored': ['wide', 'wide']}).lazy()


def test_typed_lazy_projection_checks_identity_and_budget():
    source = Typed(SourceParams(type='typed'), SourceEnvironment())
    with DataContext(sources={'data': source}) as ctx:
        assert ctx.execution_kind('data') == 'lazy'
        checks = []
        ctx._catalog.sources['data'].verify_identity = lambda: checks.append(1)
        plan = ctx.scan('data').select(pl.col('x').sum())
        checks.clear()
        assert ctx.collect('data', plan).item() == 3
        assert len(checks) == 2
        assert not ctx.is_materialized('data')
        assert ctx.describe()[0].notes['execution'] == 'typed_lazy_scan'
    limited = Typed(SourceParams(type='typed', row_budget=1), SourceEnvironment())
    with (
        DataContext(sources={'data': limited}) as ctx,
        pytest.raises(SourceError, match='row_budget'),
    ):
        ctx.scan('data')


@pytest.mark.parametrize('from_file', [False, True])
def test_sql_binding_cannot_merge_distinct_nanosecond_instants(tmp_path, from_file):
    frame = pl.DataFrame({
        'time': pl.Series([1, 2], dtype=pl.Datetime('ns', 'UTC')),
    })
    if from_file:
        from qavf.sources.file import FileSource
        from qavf.sources.parameters import FileParams
        path = tmp_path / 'instants.parquet'
        frame.write_parquet(path)
        arguments = {'sources': {
            'data': FileSource(FileParams(path=str(path)), SourceEnvironment())
        }}
    else:
        arguments = {'frames': {'data': frame}}
    with DataContext(**arguments) as ctx:
        with pytest.raises(SourceError, match='cannot preserve'):
            ctx.sql('SELECT count(DISTINCT time) FROM data')
        assert ctx.collect('data', ctx.scan('data').select(pl.col('time').n_unique())).item() == 2


def test_collect_accepts_budget_checked_eager_scan_fallback():
    source = Eager(SourceParams(type='eager', row_budget=1), SourceEnvironment())
    with DataContext(sources={'data': source}) as ctx:
        assert ctx.collect('data', ctx.scan('data').select('x')).item() == 1


@pytest.mark.parametrize('unit', ['ms', 'us', 'ns'])
def test_sql_binding_preserves_naive_timestamp_units(unit):
    frame = pl.DataFrame({'time': pl.Series([1, 2], dtype=pl.Datetime(unit))})
    with DataContext(frames={'data': frame}) as ctx:
        assert ctx.sql('SELECT count(DISTINCT time) FROM data').item() == 2
        result = ctx.sql('SELECT time FROM data ORDER BY time')
        assert result.schema == frame.schema and result.equals(frame)
