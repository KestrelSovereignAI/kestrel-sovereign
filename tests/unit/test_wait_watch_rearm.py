"""A re-registered wait watch wakes on the next distinct terminal EVENT (#3399).

On 2026-09-29 an agent re-ran a failed CI job and set
``wait("ci:<pr>", mode="signal")``. The call acknowledged ``watching: true``,
but the watch could never fire: it had fired once before, and
``list_watched`` only polled rows that had never delivered anything. Even a
re-armed watch would have stayed silent, because the CI dedup token was the
bare outcome — a re-run that fails again is ``failed`` twice.

Two changes, both pinned here against the real reconciler and store:

* registering re-arms the watch over the wake in flight or already
  delivered, and
* a provider can name its terminal event (``TERMINAL_EVENT_KEY``); the
  reconciler dedups on that identity, and the CI provider names one built
  from the head SHA plus the completed check runs' ids and attempts.

A watch fires on a new execution, never on a change of view. The Checks API
and its Actions fallback name one execution's records differently, so the
execution set travels as a view-scoped detail
(``TERMINAL_EVENT_DETAIL_KEY``/``TERMINAL_EVENT_VIEW_KEY``). Within one view
the sets are compared; across views the outcome is not consulted at all — the
reconciler re-baselines onto the new view without a wake, so a credential that
gains or loses the Checks API is not a new event.

The dispatcher and providers are test doubles; the CI scenarios run the real
``CIWaitable.poll`` and ``classify_ci_state`` over fabricated GitHub records,
and the read-path scenarios run the real ``fetch_check_rollup`` fallback and
Actions projection behind a fake HTTP layer.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from kestrel_sdk.signals import Status
from kestrel_sdk.tools import Outcome

import kestrel_sovereign.signals.sources.github_pr_watch as prw
from kestrel_sovereign.features.scheduler.ci_wait_provider import CIWaitable
from kestrel_sovereign.signals.sources.github_pr_watch import (
    CHECKS_SOURCE_CHECK_RUNS,
    CHECKS_SOURCE_WORKFLOW_RUNS,
    CheckRollup,
    PRWatchAuthError,
)
from kestrel_sovereign.waits.engine import (
    TERMINAL_EVENT_DETAIL_KEY,
    TERMINAL_EVENT_FINAL_KEY,
    TERMINAL_EVENT_KEY,
    TERMINAL_EVENT_VIEW_KEY,
    WaitRegistry,
)
from kestrel_sovereign.waits.reconciler import (
    TerminalToken,
    WaitReconciler,
    register_wait_watch,
)
from tests.unit import test_wait_reconciler as doubles


@pytest.fixture
def make_agent(tmp_path, sqlite_database_factory):
    async def create(*providers):
        db = await sqlite_database_factory(tmp_path / "agent.db")
        registry = WaitRegistry()
        for provider in providers:
            registry.register(provider)
        dispatcher = doubles._CapturingDispatcher()
        agent = SimpleNamespace(
            did="did:test:agent",
            agent_id="did:test:agent",
            _raw_storage=SimpleNamespace(db=db),
            wait_registry=registry,
            dispatcher=dispatcher,
        )
        agent._wait_reconciler = WaitReconciler(agent)
        return agent

    return create


async def _settle(agent) -> None:
    """Two ticks: detect + enqueue, then harvest the delivery."""
    await agent._wait_reconciler.reconcile()
    await agent._wait_reconciler.reconcile()


def _wakes(agent):
    return agent.dispatcher.signals


def _event(identity: str, detail: Optional[str] = None, view: Optional[str] = None) -> dict:
    data = {TERMINAL_EVENT_KEY: identity}
    if detail is not None:
        data[TERMINAL_EVENT_DETAIL_KEY] = detail
        data[TERMINAL_EVENT_VIEW_KEY] = view
    return data


async def _delivered(agent, kind, handle) -> TerminalToken:
    row = await agent._wait_reconciler._store.get(kind, handle)
    return TerminalToken.parse(row.last_signaled_outcome)


# ---------------------------------------------------------------------------
# Re-arm, over a provider that names its terminal events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rearmed_watch_fires_on_the_next_distinct_terminal_event(make_agent):
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, summary="attempt 1 failed", data=_event("run-1"))
    agent = await make_agent(jobs)

    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    # The re-run fails again: same outcome, different event. The spent watch
    # stays quiet until the agent asks for the next wake.
    jobs.set("h1", Outcome.FAILED, summary="attempt 2 failed", data=_event("run-2"))
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    await register_wait_watch(agent, "job:h1")
    await _settle(agent)

    assert len(_wakes(agent)) == 2
    assert _wakes(agent)[1].payload[TERMINAL_EVENT_KEY] == "run-2"
    assert (await _delivered(agent, "job", "h1")).event == "run-2"
    assert await agent._wait_reconciler._store.list_watched() == [], (
        "a re-armed watch is still one wake per registration"
    )


@pytest.mark.asyncio
async def test_a_rearmed_watch_does_not_refire_on_the_delivered_event(make_agent):
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("run-1"))
    agent = await make_agent(jobs)
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)

    await register_wait_watch(agent, "job:h1")
    for _ in range(3):
        await _settle(agent)

    assert len(_wakes(agent)) == 1
    watched = await agent._wait_reconciler._store.list_watched()
    assert [(w.kind, w.handle) for w in watched] == [("job", "h1")], (
        "quiet is not retired: the watch must still be armed for the next event"
    )

    jobs.set("h1", Outcome.DONE, data=_event("run-2"))
    await _settle(agent)
    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_identity_excludes_the_outcome_so_a_reclassified_event_is_not_news(
    make_agent,
):
    """#3390's shape: a classifier change that re-labels an old event must not
    mint a new token for it and replay it."""
    jobs = doubles._FakeProvider()  # monitorable: the implicit auto-wake loop
    jobs.set("j1", Outcome.FAILED, data={"status": "blocked", **_event("job-1@t0")})
    agent = await make_agent(jobs)
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    jobs.set("j1", Outcome.PARTIAL, data={"status": "clarifying", **_event("job-1@t0")})
    await _settle(agent)

    assert len(_wakes(agent)) == 1


# ---------------------------------------------------------------------------
# Providers without an identity, and the upgrade between the two
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_provider_without_an_identity_keeps_the_outcome_token(make_agent):
    tasks = doubles._PollOnlyProvider(kind="task")
    tasks.set("t1", Outcome.FAILED, data={"status": "failed"})
    agent = await make_agent(tasks)
    await register_wait_watch(agent, "task:t1")
    await _settle(agent)
    row = await agent._wait_reconciler._store.get("task", "t1")
    assert row.last_signaled_outcome == "failed:failed"

    # Same outcome class: dedup'd exactly as before, re-armed or not.
    await register_wait_watch(agent, "task:t1")
    await _settle(agent)
    assert len(_wakes(agent)) == 1
    assert TERMINAL_EVENT_KEY not in _wakes(agent)[0].payload

    tasks.set("t1", Outcome.DONE, data={"status": "completed"})
    await _settle(agent)
    assert len(_wakes(agent)) == 2
    row = await agent._wait_reconciler._store.get("task", "t1")
    assert row.last_signaled_outcome == "done:completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["", "   ", 7, None])
async def test_an_unusable_identity_falls_back_to_the_outcome_token(
    make_agent, identity,
):
    tasks = doubles._PollOnlyProvider(kind="task")
    tasks.set("t1", Outcome.DONE, data={TERMINAL_EVENT_KEY: identity})
    agent = await make_agent(tasks)
    await register_wait_watch(agent, "task:t1")
    await _settle(agent)

    row = await agent._wait_reconciler._store.get("task", "t1")
    assert row.last_signaled_outcome == "done"


@pytest.mark.asyncio
async def test_an_upgrade_to_identities_replays_no_delivered_wake(make_agent):
    """Every row a provider delivered before it named events holds an outcome
    token. Its first identity-bearing poll must re-key the row, not read the
    identity as news and replay the wake (#3390)."""
    jobs = doubles._FakeProvider()
    jobs.set("j1", Outcome.DONE, data={"status": "complete"})
    agent = await make_agent(jobs)
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    jobs.set("j1", Outcome.DONE, data={"status": "complete", **_event("job-1@t0")})
    await _settle(agent)

    assert len(_wakes(agent)) == 1
    assert (await _delivered(agent, "fake", "j1")).event == "job-1@t0"

    # From here on identities are compared: a corrected record is news.
    jobs.set("j1", Outcome.DONE, data={"status": "complete", **_event("job-1@t1")})
    await _settle(agent)
    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_an_upgrade_adoption_keeps_a_rearmed_watch_armed(make_agent):
    ci = doubles._PollOnlyProvider(kind="ci")
    ci.set("o/r#1", Outcome.FAILED)
    agent = await make_agent(ci)
    await register_wait_watch(agent, "ci:o/r#1")
    await _settle(agent)
    await register_wait_watch(agent, "ci:o/r#1")  # baseline: legacy "failed"

    ci.set("o/r#1", Outcome.FAILED, data=_event("checks@a:1"))
    await _settle(agent)
    assert len(_wakes(agent)) == 1, "the delivered failure is not replayed"
    assert len(await agent._wait_reconciler._store.list_watched()) == 1

    ci.set("o/r#1", Outcome.FAILED, data=_event("checks@a:2"))
    await _settle(agent)
    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_after_an_upgrade_every_later_event_wakes_as_its_own_transition(
    make_agent,
):
    """Review finding on #3399: an upgraded row keeps its delivered outcome
    token as its attempt target. Re-keying only the delivered token let every
    later failure match that stale target by outcome, so it was emitted as a
    retry under the old string and the event after it was read as old news.
    Each new head commit's failure must wake, as attempt 1, under its own
    token — including when the watch is re-registered inside the previous
    wake's turn, before its delivery is harvested."""
    ci = doubles._PollOnlyProvider(kind="ci")
    ci.set("o/r#1", Outcome.FAILED)  # a pre-#3399 provider: outcome token
    agent = await make_agent(ci)
    store = agent._wait_reconciler._store
    await register_wait_watch(agent, "ci:o/r#1")
    await _settle(agent)
    await register_wait_watch(agent, "ci:o/r#1")
    ci.set("o/r#1", Outcome.FAILED, data=_event("head@s1", "set-1", "check_runs"))
    await _settle(agent)  # the upgrade re-keys the delivered failure
    assert len(_wakes(agent)) == 1
    row = await store.get("ci", "o/r#1")
    assert row.attempts_signaled_target == row.last_signaled_outcome

    for n, head in enumerate(["s2", "s3", "s4"], start=2):
        ci.set("o/r#1", Outcome.FAILED, data=_event(f"head@{head}", "set-1", "check_runs"))
        await agent._wait_reconciler.reconcile()  # this failure's wake is enqueued
        await register_wait_watch(agent, "ci:o/r#1")  # inside its turn
        ci.set("o/r#1", Outcome.PENDING)  # the next push's CI is running
        await _settle(agent)

        assert len(_wakes(agent)) == n
        wake = _wakes(agent)[-1]
        assert wake.payload[TERMINAL_EVENT_KEY] == f"head@{head}"
        assert wake.payload["delivery_attempt"] == 1, "news, not a retry"
        assert (await _delivered(agent, "ci", "o/r#1")).event == f"head@{head}"
        assert len(await store.list_watched()) == 1


@pytest.mark.asyncio
async def test_after_a_view_switch_every_rerun_wakes_once(make_agent):
    """Review finding on #3399: after a re-baseline onto a new view, the
    delivered transition's attempt target still named the old view, so each
    later re-run matched it across views and was emitted as a retry of it —
    only every other re-run woke, and the attempt count climbed to the lock."""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("head@a", "run-1", "check_runs"))
    agent = await make_agent(jobs)
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    await register_wait_watch(agent, "job:h1")
    jobs.set("h1", Outcome.FAILED, data=_event("head@a", "wf-1#1", "workflow_runs"))
    await _settle(agent)  # re-baselined, no wake
    assert len(_wakes(agent)) == 1

    for attempt in range(2, 14):
        await register_wait_watch(agent, "job:h1")
        jobs.set(
            "h1", Outcome.FAILED,
            data=_event("head@a", f"wf-1#{attempt}", "workflow_runs"),
        )
        await _settle(agent)
        await _settle(agent)
        assert len(_wakes(agent)) == attempt, f"re-run {attempt} woke once"
        assert _wakes(agent)[-1].payload["delivery_attempt"] == 1


# ---------------------------------------------------------------------------
# A final event: nothing can follow it, so a re-armed watch is retired
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_watch_rearmed_over_a_final_event_is_retired_not_polled(
    make_agent, caplog,
):
    """Review finding on #3399: a re-armed watch waits for an event other
    than the delivered one. Over a merged PR there is none, so polling it
    every tick forever would spend GitHub requests on a wake that cannot
    come."""
    ci = doubles._PollOnlyProvider(kind="ci")
    merged = {**_event("merged@a"), TERMINAL_EVENT_FINAL_KEY: True}
    ci.set("o/r#1", Outcome.DONE, data=merged)
    agent = await make_agent(ci)
    store = agent._wait_reconciler._store
    await register_wait_watch(agent, "ci:o/r#1")
    await _settle(agent)
    assert len(_wakes(agent)) == 1
    assert await store.list_watched() == []

    await register_wait_watch(agent, "ci:o/r#1")
    assert len(await store.list_watched()) == 1, "registration re-arms"
    with caplog.at_level("INFO", logger="kestrel_sovereign.waits.reconciler"):
        await _settle(agent)

    assert len(_wakes(agent)) == 1
    assert await store.list_watched() == []
    assert any("retired" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_watch_rearmed_while_its_final_wake_is_in_flight_is_retired(
    make_agent,
):
    ci = doubles._PollOnlyProvider(kind="ci")
    ci.set("o/r#1", Outcome.DONE, data={**_event("merged@a"), TERMINAL_EVENT_FINAL_KEY: True})
    agent = await make_agent(ci)
    store = agent._wait_reconciler._store
    await register_wait_watch(agent, "ci:o/r#1")
    await agent._wait_reconciler.reconcile()  # the merge's wake is enqueued
    await register_wait_watch(agent, "ci:o/r#1")  # inside that wake's turn
    await _settle(agent)
    await _settle(agent)

    assert len(_wakes(agent)) == 1
    assert await store.list_watched() == []


# ---------------------------------------------------------------------------
# The ledger token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", [
    TerminalToken("failed"),
    TerminalToken("failed:blocked"),
    TerminalToken("failed", "checks@abc:1"),
    TerminalToken("partial", "checks@abc:1", "d|1\"x", "workflow_runs"),
])
def test_a_token_reads_back_as_written(token):
    assert TerminalToken.parse(token.render()) == token


def test_an_outcome_token_is_written_exactly_as_before():
    assert TerminalToken("done:completed").render() == "done:completed"


@pytest.mark.parametrize("stored", ["event:", "event:not-json", 'event:{"event":"x"}'])
def test_an_unreadable_identity_token_matches_no_event(stored):
    parsed = TerminalToken.parse(stored)
    assert parsed == TerminalToken(stored)
    assert not parsed.names_same_event(TerminalToken("failed", "x"))


@pytest.mark.parametrize(("delivered", "current", "same", "switch"), [
    # Same head SHA, same view: new iff the execution set differs.
    (TerminalToken("failed", "head@a", "set-1", "checks"),
     TerminalToken("failed", "head@a", "set-1", "checks"), True, False),
    (TerminalToken("failed", "head@a", "set-1", "checks"),
     TerminalToken("failed", "head@a", "set-2", "checks"), False, False),
    # Same head SHA, different view: never new, whatever either outcome says.
    (TerminalToken("done", "head@a", "set-1", "checks"),
     TerminalToken("partial", "head@a", "set-9", "actions"), True, True),
    (TerminalToken("failed", "head@a", "set-1", "checks"),
     TerminalToken("done", "head@a", "set-9", "actions"), True, True),
    # New head SHA: always new, through any view, with any outcome.
    (TerminalToken("failed", "head@a", "set-1", "checks"),
     TerminalToken("failed", "head@b", "set-1", "checks"), False, False),
    (TerminalToken("failed", "head@a", "set-1", "checks"),
     TerminalToken("failed", "head@b", "set-9", "actions"), False, False),
    # The outcome is no part of an identity: a reclassified event is old news.
    (TerminalToken("failed:blocked", "job@t0"),
     TerminalToken("partial:clarifying", "job@t0"), True, False),
    # No identity on one side: the outcome is all the two share.
    (TerminalToken("failed"),
     TerminalToken("failed", "head@a", "set-1", "checks"), True, False),
    (TerminalToken("failed"),
     TerminalToken("done", "head@a", "set-1", "checks"), False, False),
])
def test_one_execution_is_one_event_through_every_view(delivered, current, same, switch):
    assert delivered.names_same_event(current) is same
    assert current.names_same_event(delivered) is same
    assert current.switches_view_from(delivered) is switch


# ---------------------------------------------------------------------------
# One event, two read paths: details compare only within a view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_view_change_alone_is_not_a_new_event(make_agent):
    """The same execution named by another read path's record ids re-keys the
    row instead of waking, and the watch stays armed for the real next event."""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "check-run-101", "checks"))
    agent = await make_agent(jobs)
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    await register_wait_watch(agent, "job:h1")

    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "workflow-9001#1", "actions"))
    await _settle(agent)
    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "check-run-101", "checks"))
    await _settle(agent)

    assert len(_wakes(agent)) == 1
    assert (await _delivered(agent, "job", "h1")).detail == "check-run-101"
    assert len(await agent._wait_reconciler._store.list_watched()) == 1

    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "check-run-102", "checks"))
    await _settle(agent)
    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_across_a_view_change_the_outcome_is_not_consulted(make_agent, caplog):
    """The ruling on #3399's review: the outcome is never part of the identity,
    and the same head SHA through a different view is not a new event, even
    when the two views classify it differently. The watch is re-baselined onto
    the new view, so a genuine re-run seen through that view still fires."""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.DONE, data=_event("sha-1", "check-run-101", "checks"))
    agent = await make_agent(jobs)
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    await register_wait_watch(agent, "job:h1")

    jobs.set("h1", Outcome.PARTIAL, data=_event("sha-1", "workflow-9001#1", "actions"))
    with caplog.at_level("INFO", logger="kestrel_sovereign.waits.reconciler"):
        await _settle(agent)

    assert len(_wakes(agent)) == 1
    switches = [r for r in caplog.records if "re-baselined" in r.getMessage()]
    assert len(switches) == 1 and switches[0].levelname == "INFO"
    assert "'actions'" in switches[0].getMessage()
    delivered = await _delivered(agent, "job", "h1")
    assert (delivered.view, delivered.detail) == ("actions", "workflow-9001#1")
    store = agent._wait_reconciler._store
    [watch] = await store.list_watched()
    assert watch.watch_baseline == delivered.render(), "the baseline moved with it"

    jobs.set("h1", Outcome.PARTIAL, data=_event("sha-1", "workflow-9001#2", "actions"))
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["done", "partial"]
    assert await store.list_watched() == []


@pytest.mark.asyncio
async def test_across_a_view_change_a_new_event_identity_is_news(make_agent):
    """Only the detail is view-scoped: a new head commit is a new event
    whatever path reads it, even with the same outcome."""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "check-run-101", "checks"))
    agent = await make_agent(jobs)
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    await register_wait_watch(agent, "job:h1")

    jobs.set("h1", Outcome.FAILED, data=_event("sha-2", "workflow-9002#1", "actions"))
    await _settle(agent)

    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_a_view_switch_does_not_swallow_an_owed_rerun_wake(make_agent):
    """Review finding on #3399: a re-run judged new and emitted, whose wake
    soft-failed, is OWED. When the next poll reads that same execution
    through another view, the cross-view rule says it is the same event as
    the one delivered before the re-run — but the owed wake must still be
    retried, not re-keyed away. (A host restart to rotate the token is the
    realistic trigger: the in-flight wake is lost and the credential's view
    changes in the same tick.)"""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("head@a", "set-1", "check_runs"))
    agent = await make_agent(jobs)
    store = agent._wait_reconciler._store
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    await register_wait_watch(agent, "job:h1")

    jobs.set("h1", Outcome.FAILED, data=_event("head@a", "set-2", "check_runs"))
    agent.dispatcher._status = Status.FAILED  # the re-run's wake soft-fails
    await _settle(agent)
    owed = await store.get("job", "h1")
    assert owed.last_signaled_outcome != owed.attempts_signaled_target

    agent.dispatcher._status = Status.OK
    jobs.set("h1", Outcome.FAILED, data=_event("head@a", "wf-1#2", "workflow_runs"))
    await _settle(agent)

    row = await store.get("job", "h1")
    assert row.last_signaled_outcome == owed.attempts_signaled_target, (
        "the owed re-run wake was delivered under its own token"
    )
    assert _wakes(agent)[-1].payload["delivery_attempt"] > 1, "a retry of it"
    assert await store.list_watched() == [], "the re-armed watch fired"
    delivered_count = len(_wakes(agent))

    await register_wait_watch(agent, "job:h1")
    await _settle(agent)  # the same execution again: re-baselined, no wake
    assert len(_wakes(agent)) == delivered_count
    assert (await _delivered(agent, "job", "h1")).view == "workflow_runs"
    assert len(await store.list_watched()) == 1


@pytest.mark.asyncio
async def test_a_retry_keeps_the_token_its_transition_started_under(make_agent):
    """A wake that soft-fails and is retried after its event is re-read
    through another view is the same transition: its attempt count carries
    on and the delivery it records still matches a watch armed over it."""
    jobs = doubles._PollOnlyProvider(kind="job")
    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "check-run-101", "checks"))
    agent = await make_agent(jobs)
    agent.dispatcher._status = Status.FAILED  # soft-fail: retried next tick
    await register_wait_watch(agent, "job:h1")
    await _settle(agent)
    store = agent._wait_reconciler._store
    first = await store.get("job", "h1")
    assert first.last_signaled_outcome is None

    jobs.set("h1", Outcome.FAILED, data=_event("sha-1", "workflow-9001#1", "actions"))
    agent.dispatcher._status = Status.OK
    await _settle(agent)

    row = await store.get("job", "h1")
    assert row.last_delivery_attempts == len(_wakes(agent)) > 1, (
        "every emit was an attempt of the one transition"
    )
    assert row.last_signaled_outcome == first.attempts_signaled_target
    assert TerminalToken.parse(row.last_signaled_outcome).view == "checks"


# ---------------------------------------------------------------------------
# The CI provider, end to end through the reconciler
# ---------------------------------------------------------------------------

HEAD = "1b2c3d4e5f60718293a4b5c6d7e8f90123456789"
PR_OPEN = {"state": "open", "merged": False, "head": {"sha": HEAD}}


def _run(run_id, *, status="completed", conclusion="failure", name="unit-tests"):
    return {"id": run_id, "name": name, "status": status, "conclusion": conclusion}


class _GitHub:
    """What GitHub currently answers for one PR."""

    def __init__(self):
        self.pr = dict(PR_OPEN)
        self.rollup = CheckRollup(check_runs={"check_runs": []}, combined_status={})

    def runs(self, *runs, source=None):
        rollup = {"total_count": len(runs), "check_runs": list(runs)}
        if source is None:
            self.rollup = CheckRollup(check_runs=rollup, combined_status={})
        else:
            self.rollup = CheckRollup(
                check_runs=rollup,
                combined_status={},
                source=source,
                unreadable=("check-runs",),
            )

    async def fetch(self, repo, number, token):
        return dict(self.pr), self.rollup


@pytest.fixture
async def ci_rig(make_agent, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    github = _GitHub()
    provider = CIWaitable(feature=None)
    provider._fetch = github.fetch
    agent = await make_agent(provider)
    return SimpleNamespace(agent=agent, github=github, ref="ci:o/r#3380")


@pytest.mark.asyncio
async def test_a_ci_rerun_that_fails_again_wakes_a_rearmed_watch(ci_rig):
    agent, github = ci_rig.agent, ci_rig.github
    github.runs(_run(101))
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)
    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed"]

    github.runs(_run(102))  # the re-run, failed again
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed", "failed"]
    first, second = (
        (w.payload[TERMINAL_EVENT_KEY], w.payload[TERMINAL_EVENT_DETAIL_KEY])
        for w in _wakes(agent)
    )
    assert first[0] == second[0], "same head commit"
    assert first[1] != second[1], "the re-run's check run is a new execution set"


@pytest.mark.asyncio
async def test_a_stale_read_after_a_rerun_request_does_not_fire_on_the_old_event(
    ci_rig,
):
    """The 2026-09-29 sequence: the re-run is requested, the watch is set, and
    GitHub still answers with the old failed run. Waking on that cost a turn;
    the re-run's own pass then never woke anyone."""
    agent, github = ci_rig.agent, ci_rig.github
    github.runs(_run(101))
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)  # stale: still run 101
    github.runs(_run(102, status="queued", conclusion=None))
    await _settle(agent)  # the re-run is pending
    assert len(_wakes(agent)) == 1

    github.runs(_run(102, conclusion="success"))
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed", "done"]


@pytest.mark.asyncio
async def test_an_actions_api_rerun_is_a_new_event_by_its_attempt(ci_rig):
    """Read through the Actions fallback, a re-run keeps its workflow run id
    and increments ``run_attempt``; the attempt is what makes it news."""
    agent, github = ci_rig.agent, ci_rig.github
    attempt_1 = {**_run(9001), "run_attempt": 1}
    github.runs(attempt_1, source=CHECKS_SOURCE_WORKFLOW_RUNS)
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    github.runs({**_run(9001), "run_attempt": 2}, source=CHECKS_SOURCE_WORKFLOW_RUNS)
    await _settle(agent)
    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_a_new_head_commit_is_a_new_event_with_the_same_runs(ci_rig):
    agent, github = ci_rig.agent, ci_rig.github
    github.runs(_run(101))
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    github.pr = {**PR_OPEN, "head": {"sha": "f" * 40}}
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    assert len(_wakes(agent)) == 2


@pytest.mark.asyncio
async def test_rearming_from_inside_the_failure_wake_still_wakes_on_the_rerun(
    ci_rig,
):
    """The order an orchestrator actually uses: the failure's wake turn
    re-runs the job and re-registers the watch, so the registration lands
    before the reconciler has harvested that very wake."""
    agent, github = ci_rig.agent, ci_rig.github
    github.runs(_run(101))
    await register_wait_watch(agent, ci_rig.ref)
    await agent._wait_reconciler.reconcile()  # the failure's wake is enqueued
    assert len(_wakes(agent)) == 1

    await register_wait_watch(agent, ci_rig.ref)  # inside that wake's turn
    await agent._wait_reconciler.reconcile()  # harvest: delivered
    github.runs(_run(102, status="in_progress", conclusion=None))
    await _settle(agent)
    assert len(_wakes(agent)) == 1

    github.runs(_run(102, conclusion="success"))
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed", "done"]


@pytest.mark.asyncio
async def test_a_ci_watch_rearmed_over_a_merge_is_retired(ci_rig):
    agent, github = ci_rig.agent, ci_rig.github
    store = agent._wait_reconciler._store
    github.pr = {**PR_OPEN, "state": "closed", "merged": True}
    github.runs(_run(101, conclusion="success"))
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["done"]
    assert await store.list_watched() == []


@pytest.mark.asyncio
async def test_a_ci_watch_rearmed_over_a_close_waits_for_a_reopen(ci_rig):
    """A closed PR can be reopened, so a close is not final: the re-armed
    watch stays armed and wakes when the reopened PR's CI settles."""
    agent, github = ci_rig.agent, ci_rig.github
    store = agent._wait_reconciler._store
    github.pr = {**PR_OPEN, "state": "closed", "closed_at": "2026-09-29T10:00:00Z"}
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)
    await register_wait_watch(agent, ci_rig.ref)
    await _settle(agent)
    assert len(await store.list_watched()) == 1

    github.pr = dict(PR_OPEN)
    github.runs(_run(101, conclusion="success"))
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed", "done"]


# ---------------------------------------------------------------------------
# The CI read path changing under a watch (review finding on #3399)
# ---------------------------------------------------------------------------

_ACTIONS_JOB = 31337  # the job-level check run the Checks API reports


class _GitHubHTTP:
    """GitHub's HTTP answers for one PR, under a credential that can or cannot
    read the Checks API. Only ``_github_get`` is faked: the Checks-to-Actions
    fallback, the Actions projection and the classifier all run for real."""

    def __init__(self):
        self.checks_readable = True
        self.check_runs = []
        self.workflow_runs = []

    def execution(self, *, job_id, attempt, conclusion="failure"):
        """One GitHub Actions execution, as each API names it."""
        self.check_runs = [{
            "id": job_id, "name": "unit-tests",
            "status": "completed", "conclusion": conclusion,
        }]
        self.workflow_runs = [{
            "id": 9001, "run_attempt": attempt, "name": "CI",
            "status": "completed", "conclusion": conclusion,
        }]

    async def get(self, url, *, token, timeout, ref):
        if "/pulls/" in url:
            return dict(PR_OPEN)
        if "/check-runs" in url:
            if not self.checks_readable:
                raise PRWatchAuthError("403 for check-runs", status_code=403)
            return {"total_count": len(self.check_runs), "check_runs": self.check_runs}
        if "/actions/runs" in url:
            return {
                "total_count": len(self.workflow_runs),
                "workflow_runs": self.workflow_runs,
            }
        if "/status" in url:
            return {"state": "pending", "total_count": 0, "statuses": []}
        raise AssertionError(f"unrouted URL in test: {url}")


@pytest.fixture
async def http_rig(make_agent, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    github = _GitHubHTTP()
    monkeypatch.setattr(prw, "_github_get", github.get)
    agent = await make_agent(CIWaitable(feature=None))
    return SimpleNamespace(agent=agent, github=github, ref="ci:o/r#3380")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conclusion", "outcome", "through_actions"),
    [
        ("failure", "failed", "failed"),
        # The review's P1: one successful execution is DONE through Checks and
        # PARTIAL through the narrower Actions fallback. Neither reading is a
        # new execution.
        ("success", "done", "partial"),
    ],
)
async def test_checks_to_actions_to_checks_is_one_event_then_a_rerun_wakes(
    http_rig, conclusion, outcome, through_actions,
):
    agent, github = http_rig.agent, http_rig.github
    store = agent._wait_reconciler._store
    github.execution(job_id=_ACTIONS_JOB, attempt=1, conclusion=conclusion)
    await register_wait_watch(agent, http_rig.ref)
    await _settle(agent)
    assert [w.payload["outcome"] for w in _wakes(agent)] == [outcome]
    assert _wakes(agent)[0].payload[TERMINAL_EVENT_VIEW_KEY] == CHECKS_SOURCE_CHECK_RUNS
    await register_wait_watch(agent, http_rig.ref)  # re-run requested

    github.checks_readable = False  # the credential loses the Checks API
    await _settle(agent)
    delivered = await _delivered(agent, "ci", "o/r#3380")
    assert (delivered.view, delivered.outcome) == (
        CHECKS_SOURCE_WORKFLOW_RUNS, through_actions,
    ), "the Actions read was actually taken"
    assert len(await store.list_watched()) == 1, "still armed after the switch"
    github.checks_readable = True
    await _settle(agent)

    assert len(_wakes(agent)) == 1, "an unchanged execution is not news"
    assert (await _delivered(agent, "ci", "o/r#3380")).view == CHECKS_SOURCE_CHECK_RUNS
    assert len(await store.list_watched()) == 1

    # The genuine re-run, with the same conclusion as the run it re-ran.
    github.execution(job_id=_ACTIONS_JOB + 1, attempt=2, conclusion=conclusion)
    await _settle(agent)
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == [outcome, outcome]
    assert await store.list_watched() == [], "fired once, then spent"


@pytest.mark.asyncio
async def test_a_rerun_read_only_through_the_actions_fallback_still_wakes(http_rig):
    agent, github = http_rig.agent, http_rig.github
    github.execution(job_id=_ACTIONS_JOB, attempt=1)
    await register_wait_watch(agent, http_rig.ref)
    await _settle(agent)
    await register_wait_watch(agent, http_rig.ref)

    github.checks_readable = False
    await _settle(agent)  # the same execution, through Actions: absorbed
    assert len(_wakes(agent)) == 1

    github.execution(job_id=_ACTIONS_JOB + 1, attempt=2)
    await _settle(agent)

    assert [w.payload["outcome"] for w in _wakes(agent)] == ["failed", "failed"]
