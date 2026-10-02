"""Tests for reading Talon's job registry through its wait provider (#3398)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from kestrel_sdk.tools import Outcome, WaitStatus

from kestrel_sovereign.features.strategic_memory.run_history import (
    RunHistory,
    RunHistoryUnreadable,
    TalonRun,
    read_run_history,
)
from kestrel_sovereign.waits.engine import WaitRegistry


def _job(repo="o/r", issue=3093, disposition="clarifying",
         completed_at="2026-09-24T08:41:00+00:00", status="complete"):
    """A terminal poll in the shape ``TalonWaitable.poll`` returns."""
    outcome = Outcome.PARTIAL if disposition in ("blocked", "clarifying") else Outcome.DONE
    return WaitStatus(
        outcome,
        f"Talon job {disposition}",
        data={
            "job_id": "x",
            "status": status,
            "repo": repo,
            "issue": issue,
            "completed_at": completed_at,
            "disposition": disposition,
        },
    )


class _Provider:
    """A stand-in for ``TalonWaitable``: ``active_handles`` + ``poll``."""

    def __init__(self, jobs, kind="talon", list_error=None, poll_errors=()):
        self.kind = kind
        self.signal = None
        self._jobs = jobs
        self._list_error = list_error
        self._poll_errors = set(poll_errors)
        self.polled = []

    async def active_handles(self):
        if self._list_error is not None:
            raise self._list_error
        return list(self._jobs)

    async def poll(self, handle):
        self.polled.append(handle)
        if handle in self._poll_errors:
            raise OSError("registry unreadable")
        return self._jobs[handle]


def _agent(*providers):
    registry = WaitRegistry()
    for provider in providers:
        registry.register(provider)
    return SimpleNamespace(wait_registry=registry)


@pytest.mark.asyncio
async def test_finished_runs_are_read_per_issue():
    provider = _Provider({
        "job-a": _job(issue=3093, disposition="clarifying"),
        "job-b": _job(issue=3101, disposition="completed"),
    })

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
    assert sorted(provider.polled) == ["job-a", "job-b"]


@pytest.mark.asyncio
async def test_the_most_recent_finished_run_per_issue_wins():
    """The live #3093 record: three runs, the last one decides."""
    provider = _Provider({
        "09-16": _job(disposition="blocked", completed_at="2026-09-16T08:30:00Z"),
        "09-24": _job(disposition="clarifying", completed_at="2026-09-24T08:41:00Z"),
        "09-23": _job(disposition="blocked", completed_at="2026-09-23T08:35:00Z"),
    })

    run = (await read_run_history(_agent(provider))).latest("o/r", 3093)

    assert run.job_id == "09-24" and run.disposition == "clarifying"


@pytest.mark.asyncio
async def test_repository_names_are_matched_without_case():
    provider = _Provider({"j": _job(repo="KestrelSovereignAI/kestrel-sovereign")})

    history = await read_run_history(_agent(provider))

    assert history.latest("kestrelsovereignai/KESTREL-SOVEREIGN", 3093).job_id == "j"


@pytest.mark.asyncio
@pytest.mark.parametrize("issue", ["3093", "#3093", 3093])
async def test_an_issue_number_may_be_recorded_as_text(issue):
    provider = _Provider({"j": _job(issue=issue)})

    assert (await read_run_history(_agent(provider))).latest("o/r", 3093) is not None


@pytest.mark.asyncio
async def test_a_running_job_is_not_a_finished_run():
    running = WaitStatus(
        Outcome.PENDING,
        "running",
        data={"repo": "o/r", "issue": 3093, "status": "running", "completed_at": ""},
    )

    history = await read_run_history(_agent(_Provider({"j": running})))

    assert history.latest("o/r", 3093) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"job_id": "j", "status": "finished_unknown"}, id="pruned-job"),
        pytest.param({"repo": "o/r", "pr": 3311, "completed_at": "2026-09-24T00:00:00Z"},
                     id="iterate-on-a-pr"),
        pytest.param({"repo": "o/r", "prd": "prd.json", "completed_at": "2026-09-24T00:00:00Z"},
                     id="batch"),
        pytest.param({"repo": "", "issue": 1, "completed_at": "2026-09-24T00:00:00Z"},
                     id="no-repo"),
        pytest.param({"repo": "o/r", "issue": True, "completed_at": "2026-09-24T00:00:00Z"},
                     id="boolean-issue"),
        pytest.param({"repo": "o/r", "issue": "o/r#1", "completed_at": "2026-09-24T00:00:00Z"},
                     id="unparseable-issue"),
    ],
)
async def test_a_job_that_is_not_a_run_on_an_issue_is_skipped(data):
    status = WaitStatus(Outcome.FAILED, "ended", data=data)

    history = await read_run_history(_agent(_Provider({"j": status})))

    assert history.runs == {}


@pytest.mark.asyncio
async def test_a_missing_disposition_reads_as_unknown_not_as_a_question():
    status = WaitStatus(
        Outcome.DONE, "done",
        data={"repo": "o/r", "issue": 1, "completed_at": "2026-09-24T00:00:00Z"},
    )

    run = (await read_run_history(_agent(_Provider({"j": status})))).latest("o/r", 1)

    assert run.disposition == "unknown" and not run.ended_with_question


@pytest.mark.asyncio
async def test_no_talon_provider_means_no_runs():
    assert (await read_run_history(SimpleNamespace())).runs == {}
    assert (await read_run_history(_agent())).runs == {}


@pytest.mark.asyncio
async def test_only_the_talon_kind_is_read():
    """A sibling provider can spread peer-returned data into its poll. A peer
    naming an issue there must not stand down its dispatch."""
    a2a = _Provider({"task": _job(disposition="blocked")}, kind="a2a")

    history = await read_run_history(_agent(a2a))

    assert history.runs == {}
    assert a2a.polled == []


@pytest.mark.asyncio
async def test_a_registry_that_cannot_list_its_jobs_is_unreadable():
    provider = _Provider({}, list_error=RuntimeError("jobs.json locked"))

    with pytest.raises(RunHistoryUnreadable, match="jobs.json locked"):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
async def test_a_provider_without_enumeration_is_unreadable():
    class PollOnly:
        kind = "talon"
        signal = None

        async def poll(self, handle):
            raise AssertionError("never polled")

    with pytest.raises(RunHistoryUnreadable, match="cannot list"):
        await read_run_history(_agent(PollOnly()))


@pytest.mark.asyncio
async def test_one_unreadable_job_makes_the_history_unreadable():
    """The job that could not be read may be the very run that asked the
    question; a partial history would report that issue as free."""
    provider = _Provider(
        {"ok": _job(issue=1), "broken": _job(issue=3093)}, poll_errors={"broken"}
    )

    with pytest.raises(RunHistoryUnreadable, match="broken"):
        await read_run_history(_agent(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("completed_at", ["", None, "yesterday"])
async def test_a_finished_run_without_a_completion_time_is_unreadable(completed_at):
    provider = _Provider({"j": _job(completed_at=completed_at)})

    with pytest.raises(RunHistoryUnreadable, match="no readable completion time"):
        await read_run_history(_agent(provider))


def test_from_runs_keeps_the_latest_run():
    early = TalonRun("a", "o/r", 1, "blocked", datetime(2026, 9, 1, tzinfo=timezone.utc))
    late = TalonRun("b", "O/R", 1, "completed", datetime(2026, 9, 2, tzinfo=timezone.utc))

    assert RunHistory.from_runs([late, early]).latest("o/r", 1) is late
    assert RunHistory.from_runs([early, late]).latest("o/r", 1) is late
