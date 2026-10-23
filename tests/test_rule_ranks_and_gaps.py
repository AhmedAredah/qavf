"""Exact ranks, narrow streams and bounded gap evidence."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

import polars as pl
import pytest

from qavf import compare_datasets
from qavf.engine.context import DataContext
from qavf.engine.sql_read import SqlReadSession
from qavf.rules._grouped import ProjectedRelation, grouped_quantile
from qavf.rules._period_gaps import PeriodDomain
from qavf.rules.input_checks import ColumnCheckRule
from qavf.rules.temporal_checks import MissingPeriodRule
from qavf.settings import QAVFSettings


@pytest.mark.parametrize("storage", ["frame", "spill"])
@pytest.mark.parametrize(
    "probability",
    [0, 1, 1 / 3, 2 / 3, Decimal("1e-200"), Decimal("0." + "3" * 100)],
)
def test_grouped_quantile_ranks_match_arbitrary_precision_oracle(
    storage, probability, tmp_path
):
    values = list(range(3000))
    frame = pl.DataFrame(
        {"g": ["long"] * len(values) + ["short"] * 3 + ["empty"], "v": [*values, 2, 4, 9, None]}
    )
    with (
        DataContext(
            frames={"d": frame},
            cache_max_bytes=0 if storage == "spill" else 2**20,
            temp_directory=tmp_path,
        ) as ctx,
        ctx.observe("d", frame.columns) as observation,
    ):
        relation = ProjectedRelation(
            ctx,
            "d",
            observation.schema(),
            lazy=observation.scan(),
            observation=observation,
            collector=observation.collect,
        )
        result = grouped_quantile(relation, ["g"], "v", "q", probability)
    actual = dict(result.iter_rows())
    for group, supplied in (("long", values), ("short", [2, 4, 9])):
        position = (len(supplied) - 1) * Fraction(str(probability))
        low = position.numerator // position.denominator
        high = min(low + 1, len(supplied) - 1)
        expected = Fraction(supplied[low]) + (supplied[high] - supplied[low]) * (position - low)
        assert Fraction(actual[group]) == expected
    assert actual["empty"] is None


def test_public_quantile_comparison_does_not_pass_using_overflowed_rank():
    baseline = pl.DataFrame({"v": range(3000)})
    candidate = pl.DataFrame({"v": [2154.6666666666665] * 3000})
    result = compare_datasets(
        baseline,
        candidate,
        levels=[
            {
                "level": "groups",
                "decision": "material",
                "measures": [{"measure": "quantile", "column": "v", "p": 1 / 3}],
                "atol": 1e-6,
                "rtol": 0,
            }
        ],
        entry_points=False,
    )
    check = next(
        check for check in result.level("groups").checks if check.check.startswith("quantile")
    )
    assert check.assessed and not check.passed
    assert check.affected_records[0]["baseline"] == pytest.approx(999.6666666666666)


@pytest.mark.parametrize("evidence", [0, 2])
@pytest.mark.parametrize("source_kind", ["file", "query"])
def test_canonical_regex_stream_projects_dependencies_and_keeps_one_observation(
    tmp_path, monkeypatch, evidence, source_kind
):
    frame = pl.DataFrame(
        {
            "s": ["abc", "!", "é", "?", "none", "#"],
            "scope": [1, 1, 1, 0, 1, 1],
            "payload": ["unused" * 1000] * 6,
        }
    )
    if source_kind == "file":
        path = tmp_path / "wide.parquet"
        frame.write_parquet(path)
        source = {"type": "file", "path": str(path)}
    else:
        source = {
            "type": "query",
            "sql": "SELECT CASE WHEN random() < 0.5 THEN '!' ELSE 'é' END AS s, "
            "1 AS scope, repeat('unused', 1000) AS payload FROM range(20)",
        }
    settings = QAVFSettings.model_validate(
        {
            "datasets": {
                "d": {
                    "source": source,
                    "schema": {
                        "columns": {
                            "s": {
                                "nullable": True,
                                "missing_values": ["none"],
                                "checks": [{"regex": r"\w+", "where": "scope = 1"}],
                            }
                        }
                    },
                }
            },
            "engine": {"max_affected_records": evidence},
        }
    )
    batches = SqlReadSession.batches
    observed = []
    queries = []

    def bounded(self, statement, *args, **kwargs):
        queries.append(statement)
        for batch in batches(self, statement, batch_rows=2):
            assert "payload" not in batch.columns
            if not evidence:
                assert batch.columns == ["s", "scope"]
            observed.extend(batch.to_dicts())
            yield batch

    monkeypatch.setattr(SqlReadSession, "batches", bounded)
    with DataContext.from_settings(settings) as ctx:
        findings = ColumnCheckRule().evaluate(ctx)
    assert len(queries) == 1, findings
    expected = [
        (index, row)
        for index, row in enumerate(observed)
        if row["scope"] == 1 and row["s"] in {"!", "#"}
    ]
    assert all(f.assessed for f in findings)
    failed = [f for f in findings if not f.passed]
    if expected:
        [finding] = failed
        assert finding.affected_count == len(expected)
        affected = (
            finding.affected.to_dicts()
            if isinstance(finding.affected, pl.DataFrame)
            else finding.affected or []
        )
        assert len(affected) == min(evidence, len(expected))
        for actual, (position, supplied) in zip(affected, expected[:evidence], strict=True):
            assert actual["s"] == supplied["s"]
            assert set(actual) == {"s", "scope", "_row", "violates"}
            assert actual["_row"] == position
    else:
        assert not failed


@pytest.mark.parametrize("evidence", [0, 1, 3])
def test_missing_periods_count_all_sparse_groups_with_bounded_gap_evidence(evidence, monkeypatch):
    frame = pl.DataFrame({"g": range(20), "year": [0] * 20, "v": [1] * 20})
    observed_limits = []
    original = PeriodDomain.missing

    def counted(self, periods, limit):
        observed_limits.append(limit)
        return original(self, periods, limit)

    monkeypatch.setattr(PeriodDomain, "missing", counted)
    rule = MissingPeriodRule(
        {
            "series": [{"dataset": "d", "value": "v", "group_by": ["g"]}],
            "expected_periods": range(10000),
            "expected_groups": [{"g": 30}],
        }
    )
    with DataContext(frames={"d": frame}, evidence_limit=evidence) as ctx:
        [finding] = rule.evaluate(ctx)
    assert finding.assessed and not finding.passed
    assert finding.affected_count == 21
    assert finding.total_records == 21
    assert len(finding.affected) == evidence
    if evidence:
        assert finding.affected[0]["missing_group"]
        assert finding.affected[0]["missing_period_count"] == 10000
        assert finding.affected[0]["missing_periods"] == list(range(evidence))
    for record in finding.affected[1:]:
        assert record["missing_period_count"] == 9999
        assert record["missing_periods"] == list(range(1, evidence + 1))
    assert sum(limit > 0 for limit in observed_limits) == max(0, evidence - 1)


def test_calendar_gap_ranks_keep_dst_fold_identity_and_only_convert_samples(monkeypatch):
    start = datetime(2026, 11, 1, 4, tzinfo=UTC)
    labels = pl.Series([start + timedelta(hours=hour) for hour in range(25)]).dt.convert_time_zone(
        "America/New_York"
    )
    domain = PeriodDomain(labels)
    present = labels.gather([index for index in range(25) if index != 2])
    to_list = pl.Series.to_list

    def bounded(self):
        assert self.dtype.is_integer(), "calendar labels must only be converted for the sample"
        assert len(self) <= len(present), "the full expected domain must stay typed"
        return to_list(self)

    monkeypatch.setattr(pl.Series, "to_list", bounded)
    count, sample = domain.missing(present, 1)
    assert count == 1
    assert sample[0].timestamp() == (start + timedelta(hours=2)).timestamp()
    assert domain.missing(present, 0) == (1, [])


def test_missing_periods_support_named_season_buckets():
    frame = pl.DataFrame(
        {
            "time": [datetime(2026, month, 15) for month in (1, 4, 7)],
            "v": [1, 1, 1],
        }
    )
    rule = MissingPeriodRule(
        {"series": [{"dataset": "d", "value": "v", "time_column": "time", "bucket": "season"}]}
    )
    with DataContext(frames={"d": frame}) as ctx:
        [finding] = rule.evaluate(ctx)
    assert finding.assessed and not finding.passed
    assert finding.affected_count == 1
    assert finding.affected[0]["missing_periods"] == ["autumn"]
