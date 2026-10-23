"""Independent outcome admission and observation-lifetime regression gates."""

from __future__ import annotations

import dataclasses
import threading
import time
from types import SimpleNamespace

import polars as pl
import pytest

from qavf import DataContext, Finding, QARunner, QAVFSettings, Rule, RuleRegistry, Status
from qavf.sources.base import DataSource, SourceEnvironment, SourceParams


class ProbeRule(Rule):
    rule_id = "REAUDIT-EXECUTION"
    name = description = category = "probe"


def runner(rule, **engine):
    registry = RuleRegistry()
    registry.register(rule)
    return QARunner(QAVFSettings(plugins={"entry_points": False}, engine=engine), registry)


def test_non_finding_cannot_truth_test_a_string_verdict_or_erase_a_valid_failure():
    class Malformed(ProbeRule):
        def evaluate(self, ctx):
            raw = dataclasses.asdict(Finding.fail("invalid boolean verdict"))
            raw["passed"] = "false"
            return [Finding.fail("independent failure"), SimpleNamespace(**raw)]

    report = runner(Malformed).run()
    assert [result.status for result in report.results] == [Status.FAIL, Status.ERROR]
    assert report.exit_code() == 1


def test_finding_subclass_cannot_bypass_base_verdict_validation():
    class Unchecked(Finding):
        def __post_init__(self):
            pass

    class Malformed(ProbeRule):
        def evaluate(self, ctx):
            return Unchecked(passed="false", message="invalid verdict")

    report = runner(Malformed).run()
    assert report.results[0].status is Status.ERROR
    assert report.exit_code() == 1


@pytest.mark.parametrize("labels", ["not-target", {"target": True}, None, [True], [""], [" "]])
def test_malformed_coverage_is_unassessed_and_cannot_satisfy_a_required_check(labels):
    class MalformedCoverage(ProbeRule):
        def evaluate(self, ctx):
            return Finding.ok("different check", metrics={"passed_checks": labels})

    report = runner(MalformedCoverage, require=["REAUDIT-EXECUTION[target]"]).run()
    assert all(result.status is Status.ERROR for result in report.results)
    assert any(result.check == "required: REAUDIT-EXECUTION[target]" for result in report.results)
    assert report.exit_code() == 1


@pytest.mark.parametrize("labels,expected", [(("target",), 0), (["not-target"], 1), ([], 1)])
def test_coverage_uses_exact_labels_and_survives_zero_evidence_budget(labels, expected):
    class Coverage(ProbeRule):
        def evaluate(self, ctx):
            return Finding.ok("summary", metrics={"passed_checks": labels})

    report = runner(Coverage, require=["REAUDIT-EXECUTION[target]"], max_evidence_bytes=0).run()
    assert report.exit_code() == expected
    assert report.results[0].metrics["passed_checks"] == list(labels)


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_timeout_report_does_not_wait_for_reduction_callback_or_release_its_input(cache_bytes):
    entered, release, returned, worker_done = (threading.Event() for _ in range(4))
    reports, errors = [], []

    class SlowReduction(ProbeRule):
        def evaluate(self, ctx):
            def build(data):
                entered.set()
                assert release.wait(30)
                return data.select(pl.sum("x"))

            try:
                ctx.reduce("d", build)
                return Finding.ok("complete")
            finally:
                worker_done.set()

    execute = runner(SlowReduction, rule_timeout_s=0.03, timeout_grace_s=0.001)
    # Metadata discovery is unrelated to the worker/telemetry lock invariant.
    from qavf.engine.runner import _rule_info, environment

    environment()
    _rule_info(SlowReduction())
    with DataContext(frames={"d": pl.DataFrame({"x": range(1000)})},
                     cache_max_bytes=cache_bytes) as ctx:
        def run():
            try:
                reports.append(execute.run(ctx))
            except BaseException as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(10)
            # A watchdog checks ownership, not a one-second whole-report SLA.
            assert returned.wait(10), "report telemetry waited for the overdue reduction"
            assert not release.is_set() and not worker_done.is_set()
            assert errors == []
            assert reports[0].results[0].status is Status.ERROR
            # The overdue reduction's storage remains leased until it really ends.
            before = ctx.resource_usage()
            ctx.release(["d"])
            after = ctx.resource_usage()
            assert after["retained_total_bytes"] == before["retained_total_bytes"]
            assert after["owned_spill_bytes"] == before["owned_spill_bytes"]
        finally:
            release.set()
            assert worker_done.wait(2)
            thread.join(2)
            deadline = time.monotonic() + 2
            while ctx._active and time.monotonic() < deadline:
                time.sleep(0.001)
        assert not thread.is_alive() and not ctx._active
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


class Eager(DataSource):
    kind, Params = "reaudit_eager", SourceParams

    def load(self):
        # This is newly allocated by the source; there is no caller-owned frame.
        return pl.DataFrame({"x": range(1000)})


def test_timeout_report_uses_pending_snapshots_during_blocked_cache_admission(monkeypatch):
    from qavf.engine import resources
    from qavf.engine.runner import _rule_info, environment

    entered, release, returned = (threading.Event() for _ in range(3))
    original = resources.compact_polars_with_size
    reports, errors = [], []

    def blocked_compaction(value):
        entered.set()
        assert release.wait(30)
        return original(value)

    class Loads(ProbeRule):
        def evaluate(self, ctx):
            ctx.get("d")
            return Finding.ok("complete")

    execute = runner(Loads, rule_timeout_s=0.03, timeout_grace_s=0.001)
    environment()
    _rule_info(Loads())
    source = Eager(SourceParams(type=Eager.kind), SourceEnvironment())
    with DataContext(sources={"d": source}) as ctx:
        monkeypatch.setattr(resources, "compact_polars_with_size", blocked_compaction)

        def run():
            try:
                reports.append(execute.run(ctx))
            except BaseException as error:
                errors.append(error)
            finally:
                returned.set()

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(10)
            # A watchdog checks ownership, not a one-second whole-report SLA.
            assert returned.wait(10), "finalization waited for timed-out cache admission"
            assert errors == []
            assert reports[0].results[0].status is Status.ERROR
            usage = reports[0].metadata.resource_usage
            assert usage["resource_usage_pending"] is True
            assert usage["metadata_snapshot_pending"] is True
        finally:
            release.set()
            thread.join(2)
            deadline = time.monotonic() + 2
            while ctx._active and time.monotonic() < deadline:
                time.sleep(0.001)
        assert not thread.is_alive() and not ctx._active


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_first_eager_observation_remains_charged_after_dataset_release(cache_bytes):
    source = Eager(SourceParams(type=Eager.kind), SourceEnvironment())
    with DataContext(sources={"d": source}, cache_max_bytes=cache_bytes) as ctx:
        with ctx.observe("d") as observed:
            before = ctx.resource_usage()
            ctx.release(["d"])
            after = ctx.resource_usage()
            assert before["retained_total_bytes"] + before["owned_spill_bytes"] >= 8000
            assert after["retained_total_bytes"] == before["retained_total_bytes"]
            assert after["owned_spill_bytes"] == before["owned_spill_bytes"]
            assert observed.reduce(lambda data: data.select(pl.sum("x"))).item() == 499500
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_ad_hoc_eager_observation_uses_shared_admission(cache_bytes):
    class Registry:
        def build(self, config, env):
            return Eager(SourceParams(type=Eager.kind), env)

    with DataContext(source_registry=Registry(), cache_max_bytes=cache_bytes) as ctx:
        def consume(view):
            with view.observe() as observed:
                usage = ctx.resource_usage()
                assert usage["retained_total_bytes"] + usage["owned_spill_bytes"] >= 8000
                assert observed.reduce(lambda data: data.select(pl.sum("x"))).item() == 499500
                if cache_bytes == 0:
                    assert usage["retained_total_bytes"] == 0
                    assert usage["owned_spill_bytes"] > 0

        ctx.read_source({"type": Eager.kind}, consume)
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_borrowed_contract_view_retains_charge_after_last_contract(cache_bytes):
    with (
        DataContext(
            frames={"d": pl.DataFrame({"x": range(1000), "y": range(1000)}).lazy()},
            cache_max_bytes=cache_bytes,
        ) as ctx,
        ctx.observation_plan([{"d": ["x"]}]) as plan,
    ):
        with plan.contract(0) as observations:
            borrowed = observations["d"].borrow()
            with pytest.raises(ValueError, match="does not contain"):
                observations["d"].borrow(["y"])
            before = ctx.resource_usage()
        with borrowed:
            after = ctx.resource_usage()
            assert after["retained_total_bytes"] == before["retained_total_bytes"]
            assert after["owned_spill_bytes"] == before["owned_spill_bytes"]
            assert borrowed.reduce(lambda data: data.select(pl.sum("x"))).item() == 499500
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_invalid_stored_observation_projection_does_not_leak_its_lease(cache_bytes):
    with DataContext(frames={"d": pl.DataFrame({"x": range(1000)})},
                     cache_max_bytes=cache_bytes) as ctx:
        with pytest.raises(ValueError, match="at least one"), ctx.observe("d", []):
            pytest.fail("invalid empty projection accepted")
        ctx.release(["d"])
        assert ctx.resource_usage()["retained_total_bytes"] == 0
        assert ctx.resource_usage()["owned_spill_bytes"] == 0


@pytest.mark.parametrize("cache_bytes", [0, 20_000])
def test_timeout_ownership_survives_slow_report_verification(monkeypatch, cache_bytes):
    original = DataContext.verify_inputs

    def delayed(self, *args, **kwargs):
        time.sleep(1.2)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataContext, "verify_inputs", delayed)
    test_timeout_report_does_not_wait_for_reduction_callback_or_release_its_input(cache_bytes)


def test_completed_worker_notifies_deadline_monitor():
    class Fast(ProbeRule):
        def evaluate(self, ctx):
            return Finding.ok("complete")

    returned = threading.Event()
    reports, errors = [], []

    def run():
        try:
            reports.append(runner(Fast, rule_timeout_s=30).run())
        except BaseException as error:
            errors.append(error)
        finally:
            returned.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert returned.wait(10), "completion did not notify deadline monitor"
    thread.join(1)
    assert errors == []
    assert reports[0].results[0].status is Status.PASS
