"""Public regressions for collision-safe CSV reads and bounded rejection truthfulness."""

from pathlib import Path

import polars as pl
import pytest

from qavf import Dataset, File, ResourceLimits, Status, validate
from qavf.sources.base import SourceEnvironment, SourceError
from qavf.sources.file import FileSource
from qavf.sources.parameters import FileParams


def _config(rule: str = "INP-002") -> dict:
    """Configuration for checking ``rule``; schema contracts always run."""
    return {"plugins": {"entry_points": False}, "engine": {"parallel": False}}


@pytest.mark.parametrize("helper", ["__qavf_text_0", "__QAVF_TEXT_0", "__qavf_text_0_"])
@pytest.mark.parametrize("project", [False, True])
def test_conversion_helpers_never_replace_physical_csv_values(tmp_path, helper, project):
    path = tmp_path / "values.csv"
    path.write_text(f"x,{helper},payload\n-1,100,kept\n")
    source = File(
        path,
        options={"header": True, "types": {helper: "VARCHAR"}},
        columns=["x", helper] if project else None,
    )
    report = validate(
        source,
        schema={"columns": {"x": {"dtype": "int", "ge": 0}}},
        config=_config("INP-003"),
    )
    result = report.get("INP-003", "data.x: ge 0")
    assert result.status is Status.FAIL and result.affected_count == 1
    assert result.affected_records[0]["x"] == -1
    expected = ["x", helper] if project else ["x", helper, "payload"]
    assert report.metadata.datasets[0].columns == expected


@pytest.mark.parametrize("key", ["value", "_row", "violates", "_file", "_file_row"])
def test_csv_type_evidence_preserves_metadata_named_user_keys(tmp_path, key):
    for name, value in (("a.csv", "first"), ("b.csv", "second")):
        (tmp_path / name).write_text(f"{key},x\n{value},bad\n")
    report = validate(
        File(tmp_path / "*.csv", options={"header": True}),
        schema={"primary_key": [key], "columns": {key: {"dtype": "str"}, "x": {"dtype": "int"}}},
        config=_config(),
    )
    result = report.get("INP-002", "data.x: dtype int")
    assert result.status is Status.FAIL and result.affected_count == 2
    assert [r[key] for r in result.affected_records] == ["first", "second"]
    for index, record in enumerate(result.affected_records):
        row_name = "qavf__row" if key == "_row" else "_row"
        file_name = "qavf__file" if key == "_file" else "_file"
        file_row = "qavf__file_row" if key == "_file_row" else "_file_row"
        assert record[row_name] == index
        assert record[file_row] == 0
        assert Path(record[file_name]).name == ("a.csv" if index == 0 else "b.csv")
        assert record["x"] == "bad"


@pytest.mark.parametrize("column", ["value", "_row", "violates"])
def test_failing_column_can_itself_have_a_metadata_name(tmp_path, column):
    path = tmp_path / "values.csv"
    path.write_text(f"{column}\nbad\n")
    report = validate(File(path), schema={"columns": {column: {"dtype": "int"}}}, config=_config())
    record = report.get("INP-002", f"data.{column}: dtype int").affected_records[0]
    assert record[column] == "bad"
    assert record["qavf__row" if column == "_row" else "_row"] == 0
    assert record["qavf_violates" if column == "violates" else "violates"] == "not an int"


@pytest.mark.parametrize("key", ["value", "_row", "_file"])
def test_type_evidence_retains_a_primary_key_that_is_also_converted(tmp_path, key):
    path = tmp_path / "values.csv"
    path.write_text(f"{key},x\n123,bad\n")
    report = validate(
        File(path),
        schema={"primary_key": [key], "columns": {key: {"dtype": "int"}, "x": {"dtype": "int"}}},
        config=_config(),
    )
    record = report.get("INP-002", "data.x: dtype int").affected_records[0]
    assert record[key] == "123"
    assert record["qavf__row" if key == "_row" else "_row"] == 0


@pytest.mark.parametrize(
    ("cap", "verdict", "count", "exact"),
    [
        (0, "failed", 10, True),
        (1, "incomplete", 1, False),
        (2, "incomplete", 1, False),
        (3, "failed", 2, False),
    ],
)
@pytest.mark.parametrize("evidence_bytes", [0, 4096])
def test_reject_entry_caps_never_make_an_unknown_threshold_pass(
    tmp_path, cap, verdict, count, exact, evidence_bytes
):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n" + "x,y\n" * 10 + "1,2\n")
    source = File(
        path,
        options={
            "header": True,
            "types": {"a": "INTEGER", "b": "INTEGER"},
            "store_rejects": True,
            "rejects_limit": cap,
            "parallel": False,
        },
        acknowledge_lossy=True,
        max_rejected_rows=1,
    )
    report = validate(
        source,
        schema={"columns": {}, "min_rows": 1},
        config=_config(),
        resources=ResourceLimits(evidence_bytes=evidence_bytes),
    )
    result = report.get("INP-002", "data: unreadable lines")
    assert report.verdict() == verdict
    assert result.affected_count == count
    assert result.affected_count_exact is exact
    assert result.model_dump()["affected_count_exact"] is exact


def test_native_capture_is_bounded_even_with_reader_zero_limit(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x\n" + "not-an-int\n" * 2500 + "1\n")
    source = FileSource(
        FileParams(
            path=str(path),
            options={
                "columns": {"x": "INTEGER"},
                "header": True,
                "store_rejects": True,
                "rejects_limit": 0,
            },
            acknowledge_lossy=True,
            rejects_budget=17,
        ),
        SourceEnvironment(),
    )
    assert source.load()["x"].to_list() == [1]
    assert source.load_notes["effective_reader_options"]["rejects_limit"] == 17
    rejected = source.rejected_rows()
    assert rejected is not None and rejected.count == 17 and not rejected.exact
    assert len(rejected.sample) <= 17


def test_capture_budget_is_shared_across_resolved_files(tmp_path):
    for index in range(3):
        (tmp_path / f"{index}.csv").write_text("x\n" + "bad\n" * 10 + "1\n")
    source = FileSource(
        FileParams(
            path=str(tmp_path / "*.csv"),
            options={"columns": {"x": "INTEGER"}, "header": True, "store_rejects": True},
            acknowledge_lossy=True,
            rejects_budget=7,
        ),
        SourceEnvironment(),
    )
    assert source.load().height == 3
    assert source.load_notes["rejection_capture_limit_per_file"] == 2
    assert source.rejected_rows().count == 6
    assert not source.rejected_rows().exact


def test_too_many_files_for_capture_budget_fails_before_reading(tmp_path):
    for index in range(3):
        (tmp_path / f"{index}.csv").write_text("x\n1\n")
    source = FileSource(
        FileParams(
            path=str(tmp_path / "*.csv"),
            options={"store_rejects": True},
            acknowledge_lossy=True,
            rejects_budget=2,
        ),
        SourceEnvironment(),
    )
    with pytest.raises(SourceError, match="exceeds rejects_budget"):
        source.load()


def test_unsaturated_threshold_equality_still_passes(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x\nbad\n1\n")
    report = validate(
        File(
            path,
            options={"columns": {"x": "INTEGER"}, "header": True, "store_rejects": True},
            acknowledge_lossy=True,
            max_rejected_rows=1,
        ),
        schema={"columns": {}},
        config=_config(),
    )
    assert report.verdict() == "passed"


def test_auxiliary_descriptor_projects_and_renames_without_new_context(tmp_path):
    from qavf.engine.context import DataContext
    from qavf.inputs.auxiliary import project_input

    path = tmp_path / "crosswalk.csv"
    path.write_text("raw,parent,payload\n001,002,unused\n")
    with DataContext() as ctx:
        result = project_input(
            ctx,
            Dataset(File(path, text_columns=["raw", "parent"]), rename={"raw": "child"}),
            ["child", "parent"],
        )
        assert result.to_dicts() == [{"child": "001", "parent": "002"}]
        assert result.schema == {"child": pl.String, "parent": pl.String}


def test_auxiliary_query_uses_parent_connection_configuration(tmp_path):
    import duckdb

    from qavf import Query
    from qavf.engine.context import DataContext
    from qavf.inputs.auxiliary import project_input
    from qavf.settings import ConnectionSpec

    path = tmp_path / "lookup.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE pairs AS SELECT '001' AS child, '002' AS parent")
    env = SourceEnvironment(connections={"lookup": ConnectionSpec(type="duckdb", path=str(path))})
    with DataContext(env=env) as ctx:
        result = project_input(
            ctx, Query("SELECT * FROM lookup.pairs", connection="lookup"), ["child", "parent"]
        )
    assert result.to_dicts() == [{"child": "001", "parent": "002"}]


@pytest.mark.parametrize("model", [False, True])
def test_run_config_rejects_live_crosswalk_before_copying_or_consuming_it(model):
    from qavf.inputs.normalize import prepare_configuration
    from qavf.settings import QAVFSettings
    from qavf.spatial.hierarchy import Crosswalk

    class Untouched:
        def __deepcopy__(self, memo):
            pytest.fail("configuration copied a live input")

        def __arrow_c_stream__(self):
            pytest.fail("configuration consumed a live input")

    crosswalk = Crosswalk(data=Untouched())
    config = {
        # A plugin check: its settings are validated once plugins load.
        "datasets": {"data": {"checks": [{"check": "plugin_probe", "crosswalk": crosswalk}]}},
    }
    if model:
        config = QAVFSettings.model_validate(config)
    with pytest.raises(ValueError, match="named dataset"):
        prepare_configuration(None, schema=None, config=config, resources=None)
