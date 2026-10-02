"""Tests for reading Talon's run history through its wait provider (#3398)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from kestrel_sovereign.features.strategic_memory.run_history import (
    RunHistory,
    RunHistoryUnreadable,
    TalonRun,
    read_run_history,
)
from kestrel_sovereign.waits.engine import WaitRegistry


def _run(job_id="x", repo="o/r", issue=3093, disposition="clarifying",
         completed_at="2026-09-24T08:41:00+00:00"):
    """One entry of a ``finished_runs()`` report."""
    return {
        "job_id": job_id,
        "repo": repo,
        "issue": issue,
        "disposition": disposition,
        "completed_at": completed_at,
    }


class _Provider:
    """A ``talon`` wait provider that reports its finished runs read-only.

    ``poll()`` reaps, pushes preserved work and rewrites the registry, and
    ``active_handles()`` cannot say whether it read the whole registry
    (#3398 review). Reading run history must call neither. Each records the
    call in ``forbidden`` before failing, so a test can assert none happened
    even where the failure itself is caught.
    """

    def __init__(self, runs=(), kind="talon", complete=True, reason=None, error=None):
        self.kind = kind
        self.signal = None
        self._report = {"complete": complete, "runs": list(runs)}
        if reason is not None:
            self._report["reason"] = reason
        self._error = error
        self.reads = 0
        self.forbidden = []

    async def finished_runs(self):
        self.reads += 1
        if self._error is not None:
            raise self._error
        return self._report

    async def active_handles(self):
        self.forbidden.append("active_handles")
        raise AssertionError("run history must not enumerate through active_handles()")

    async def poll(self, handle):
        self.forbidden.append(f"poll:{handle}")
        raise AssertionError("run history must not poll: poll() reaps, pushes and writes")


class _SilentlyUnreadableRegistry:
    """What the Talon provider without ``finished_runs()`` does with a corrupt
    ``jobs.json``: ``_reload_persisted_jobs()`` logs the failure and returns,
    so ``active_handles()`` answers ``[]`` -- an empty registry, as far as
    anything reading it can tell."""

    kind = "talon"
    signal = None

    def __init__(self):
        self.listed = 0

    async def active_handles(self):
        self.listed += 1
        return []

    async def poll(self, handle):
        raise AssertionError("run history must not poll: poll() reaps, pushes and writes")


def _agent(*providers):
    registry = WaitRegistry()
    for provider in providers:
        registry.register(provider)
    return SimpleNamespace(wait_registry=registry)


@pytest.mark.asyncio
async def test_finished_runs_are_read_per_issue():
    provider = _Provider([
        _run("job-a", issue=3093, disposition="clarifying"),
        _run("job-b", issue=3101, disposition="completed"),
    ])

    history = await read_run_history(_agent(provider))

    run = history.latest("o/r", 3093)
    assert run == TalonRun(
        job_id="job-a",
        repo="o/r",
        issue_number=3093,
        disposition="clarifying",
        completed_at=datetime(2026, 9, 24, 8, 41, tzinfo=timezone.utc),
    )
    assert run.ended_with_question
    assert not history.latest("o/r", 3101).ended_with_question
    assert provider.reads == 1
    assert provider.forbidden == []


@pytest.mark.asyncio
async def test_the_most_recent_finished_run_per_issue_wins():
    """The live #3093 record: three runs, the last one decides."""
    provider = _Provider([
        _run("09-16", disposition="blocked", completed_at="2026-09-16T08:30:00Z"),
        _run("09-24", disposition="clarifying", completed_at="2026-09-24T08:41:00Z"),
        _run("09-23", disposition="blocked", completed_at="2026-09-23T08:35:00Z"),
    ])

    run = (await read_run_history(_agent(provider))).latest("o/r", 3093)

    assert run.job_id == "09-24" and run.disposition == "clarifying"


@pytest.mark.asyncio
async def test_repository_names_are_matched_without_case():
    provider = _Provider([_run("j", repo="KestrelSovereignAI/kestrel-sovereign")])

    history = await read_run_history(_agent(provider))

    assert history.latest("kestrelsovereignai/KESTREL-SOVEREIGN", 3093).job_id == "j"


@pytest.mark.asyncio
@pytest.mark.parametrize("issue", ["3093", "#3093", 3093])
async def test_an_issue_number_may_be_recorded_as_text(issue):
    provider = _Provider([_run(issue=issue)])

    assert (await read_run_history(_agent(provider))).latest("o/r", 3093) is not None


@pytest.mark.asyncio
async def test_a_sync_finished_runs_is_read_too():
    class SyncProvider(_Provider):
        def finished_runs(self):
            self.reads += 1
            return self._report

    history = await read_run_history(_agent(SyncProvider([_run("j")])))

    assert history.latest("o/r", 3093).job_id == "j"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"job_id": "j", "repo": "o/r", "pr": 3311,
                      "completed_at": "2026-09-24T00:00:00Z"}, id="iterate-on-a-pr"),
        pytest.param({"job_id": "j", "repo": "o/r", "prd": "prd.json",
                      "completed_at": "2026-09-24T00:00:00Z"}, id="batch"),
        pytest.param(_run(issue=None), id="issue-none"),
        pytest.param(_run(issue=""), id="issue-empty"),
    ],
)
async def test_a_job_that_names_no_issue_is_not_a_run_on_an_issue(entry):
    history = await read_run_history(_agent(_Provider([entry])))

    assert history.runs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(_run(issue=True), id="boolean-issue"),
        pytest.param(_run(issue="o/r#1"), id="unparseable-issue"),
        pytest.param(_run(issue=0), id="issue-zero"),
        pytest.param(_run(repo=""), id="no-repo"),
        pytest.param({**_run(), "repo": None}, id="repo-none"),
    ],
)
async def test_a_job_naming_an_issue_it_does_not_identify_is_unreadable(entry):
    """Which issue it ran on is unknown, so it may be the run that asked the
    question. Skipping it would report that issue as free."""
    with pytest.raises(RunHistoryUnreadable, match="does not identify an issue"):
        await read_run_history(_agent(_Provider([entry])))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry, match",
    [
        pytest.param("job-a", "not a record", id="not-a-record"),
        pytest.param({**_run(), "job_id": ""}, "no job id", id="no-job-id"),
    ],
)
async def test_a_malformed_run_record_is_unreadable(entry, match):
    with pytest.raises(RunHistoryUnreadable, match=match):
        await read_run_history(_agent(_Provider([entry])))


@pytest.mark.asyncio
async def test_a_missing_disposition_reads_as_unknown_not_as_a_question():
    entry = _run()
    del entry["disposition"]

    run = (await read_run_history(_agent(_Provider([entry])))).latest("o/r", 3093)

    assert run.disposition == "unknown" and not run.ended_with_question


@pytest.mark.asyncio
async def test_no_talon_provider_means_no_runs():
    assert (await read_run_history(SimpleNamespace())).runs == {}
    assert (await read_run_history(_agent())).runs == {}


@pytest.mark.asyncio
async def test_only_the_talon_kind_is_read():
    """A sibling provider can spread peer-returned data into what it reports.
    A peer naming an issue there must not stand down its dispatch."""
    a2a = _Provider([_run(disposition="blocked")], kind="a2a")

    history = await read_run_history(_agent(a2a))

    assert history.runs == {}
    assert a2a.reads == 0


# ---------------------------------------------------------------------------
# Completeness: an unknown history is never an empty one (#3398 review P1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_registry_read_that_fails_silently_is_unconfirmed_not_empty():
    """Review P1: on a corrupt or unreadable ``jobs.json`` the Talon provider's
    enumeration logs and returns ``[]``. Read as a history, that is "no
    previous runs", and the blocked, unanswered issue is dispatched again --
    the #3093 failure. A provider with no read that reports completeness has
    an unconfirmed history, and its enumeration is not consulted at all."""
    provider = _SilentlyUnreadableRegistry()

    with pytest.raises(RunHistoryUnreadable, match=r"no read-only finished_runs\(\)"):
        await read_run_history(_agent(provider))

    assert provider.listed == 0


@pytest.mark.asyncio
async def test_an_incomplete_read_is_unreadable_even_with_runs_in_it():
    """The runs Talon could read are not the history: the one it could not
    may be the run that asked the question."""
    provider = _Provider(
        [_run("readable", issue=1, disposition="completed")],
        complete=False,
        reason="jobs.json: JSONDecodeError",
    )

    with pytest.raises(RunHistoryUnreadable, match="jobs.json: JSONDecodeError"):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report",
    [
        pytest.param({"runs": []}, id="completeness-not-reported"),
        pytest.param({"complete": "true", "runs": []}, id="completeness-as-text"),
        pytest.param({"complete": 1, "runs": []}, id="completeness-as-number"),
        pytest.param({"complete": None, "runs": []}, id="completeness-unknown"),
    ],
)
async def test_only_a_read_reported_complete_is_complete(report):
    provider = _Provider()
    provider._report = report

    with pytest.raises(RunHistoryUnreadable, match="whole job registry"):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report, match",
    [
        pytest.param([], "returned list, not a report", id="not-a-mapping"),
        pytest.param(None, "returned NoneType, not a report", id="nothing"),
        pytest.param({"complete": True}, "no list of runs", id="no-runs"),
        pytest.param({"complete": True, "runs": "j"}, "no list of runs", id="runs-not-a-list"),
    ],
)
async def test_a_malformed_report_is_unreadable(report, match):
    provider = _Provider()
    provider._report = report

    with pytest.raises(RunHistoryUnreadable, match=match):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
async def test_a_failed_read_is_unreadable():
    provider = _Provider(error=OSError("jobs.json locked"))

    with pytest.raises(RunHistoryUnreadable, match="jobs.json locked"):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_at", ["", None, "yesterday"])
async def test_a_finished_run_without_a_completion_time_is_unreadable(completed_at):
    provider = _Provider([_run(completed_at=completed_at)])

    with pytest.raises(RunHistoryUnreadable, match="no readable completion time"):
        await read_run_history(_agent(provider))


def test_from_runs_keeps_the_latest_run():
    early = TalonRun("a", "o/r", 1, "blocked", datetime(2026, 9, 1, tzinfo=timezone.utc))
    late = TalonRun("b", "O/R", 1, "completed", datetime(2026, 9, 2, tzinfo=timezone.utc))

    assert RunHistory.from_runs([late, early]).latest("o/r", 1) is late
    assert RunHistory.from_runs([early, late]).latest("o/r", 1) is late
