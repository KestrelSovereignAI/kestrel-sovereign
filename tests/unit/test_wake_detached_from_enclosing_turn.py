"""#4023: a wake enqueued from inside a turn is never part of that turn.

Between 2026-11-10 and 11-11 Claw parked seven times with the same shape: a
turn logged ``end``, the next turn logged ``begin`` in the same millisecond,
and nothing reached ``process_input`` until the host restarted. The producing
turn had enqueued a COGNITION wake (most often its own ``a2a.task_complete``
after ``respond_to_a2a_task``). The wake's Task copied the turn's context, so
it was enrolled in the turn's native privacy lifetime and carried its delegated
CONVERSATION ownership. The turn's effect-journal retirement, which runs after
CONVERSATION is released and holds the privacy-transition lock, joined the
wake; the wake took CONVERSATION and waited for that lock.

Every dispatch and every supervisor awaiting one is now created inside
``detached_from_enclosing_turn()``. Every agent here is real, on real storage;
only the model is scripted. Each phase must settle within ``SETTLE_SECONDS``;
a park does not.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from kestrel_sdk.signals import (
    RedactionPolicy,
    ResourceLock,
    Signal,
    SignalMode,
    SourceRegistration,
)

from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.shared.booted_agent import booted_agent

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(60)]

REPLY = "Noted."
#: Each phase of a scenario must settle well inside this. A park does not.
SETTLE_SECONDS = 10.0
SENDER = "did:kestrel:sender-4023"


def _script_replies(agent, monkeypatch, generate=None):
    async def generate_with_messages(**kwargs):
        if generate is not None:
            return await generate(kwargs)
        return LLMResponse(content=REPLY)

    async def stream_with_tool_detection(**kwargs):
        yield REPLY

    monkeypatch.setattr(agent.llm_service, "generate_with_messages", generate_with_messages)
    monkeypatch.setattr(
        agent.llm_service, "stream_with_tool_detection", stream_with_tool_detection
    )


def _prompt_text(kwargs) -> str:
    return "\n".join(str(m.get("content")) for m in kwargs.get("messages") or [])


def _last_role(kwargs):
    messages = kwargs.get("messages") or [{}]
    return messages[-1].get("role")


def _diagnostics(*agents) -> str:
    """Which locks each agent holds, and which turn-shaped Tasks are still live."""
    lines = []
    for agent in agents:
        lines.append(
            f"{agent.agent_name}: holds {agent._get_lock_manager().active_hold_diagnostics()}, "
            f"privacy lock held: {agent._get_privacy_transition_lock().locked()}"
        )
    live = sorted(
        task.get_name()
        for task in asyncio.all_tasks()
        if not task.done()
        and task.get_name().startswith(
            ("invocation", "turn", "signal_dispatch", "durable_cognition", "a2a_", "restart_")
        )
    )
    lines.append(f"live tasks: {live}")
    return "\n".join(lines)


async def _settle(agents, tasks, label):
    done, pending = await asyncio.wait(set(tasks), timeout=SETTLE_SECONDS)
    if pending:
        diagnostics = _diagnostics(*agents)
        for task in pending:
            task.cancel()
        # A parked turn can absorb its cancellation; report, do not wait.
        await asyncio.wait(pending, timeout=1.0)
        pytest.fail(f"{label}: {len(pending)} task(s) never finished\n{diagnostics}")
    return done


async def _until(agents, predicate, label):
    """Poll an async predicate; fail naming every lock hold if it never holds."""
    deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
    while not await predicate():
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"{label}\n{_diagnostics(*agents)}")
        await asyncio.sleep(0.02)


async def _wait_until_queued(agent):
    while agent._get_lock_manager().active_hold_diagnostics()[0]["blocked_acquirers"] == 0:
        await asyncio.sleep(0.01)


async def _submit_task(recipient, *, creator_agent_id=SENDER):
    """An inbound A2A work order; TaskManager announces it on the calling Task."""
    from kestrel_sdk.native_a2a.helpers.proto_helpers import new_text_message
    from kestrel_sdk.native_a2a.types import Role, SendMessageRequest

    manager = recipient.task_manager
    return await manager.create_task(
        SendMessageRequest(
            tenant=manager.task_store.scope.tenant,
            message=new_text_message(
                "Please do this", context_id="context-4023", role=Role.ROLE_USER
            ),
        ),
        creator_agent_id=creator_agent_id,
    )


async def _deliveries(agent, consumer_id):
    return await agent.dispatcher.list_durable_deliveries(consumer_id=consumer_id)


async def _submission_acknowledged(agent):
    from kestrel_sovereign.signals.sources.a2a_task_submitted import (
        DURABLE_COGNITION_CONSUMER_ID,
    )

    statuses = [d.status for d in await _deliveries(agent, DURABLE_COGNITION_CONSUMER_ID)]
    return statuses == ["acknowledged"]


async def _until_completion_wake_settled(agent):
    """Leave the agent only once its ``a2a.task_complete`` wake has finished.

    Shutting down while that turn runs cancels it, perhaps mid-COMMIT, and the
    agent's teardown rightly reports a cancelled COMMIT as an unknown outcome.
    """
    from kestrel_sovereign.signals.sources.a2a import DURABLE_COGNITION_CONSUMER_ID

    async def settled():
        wakes = await _deliveries(agent, DURABLE_COGNITION_CONSUMER_ID)
        return bool(wakes) and all(wake.is_terminal for wake in wakes)

    await _until((agent,), settled, "the ended task's completion wake never settled")


def _cognition_registration(tmp_path: Path, name: str) -> SourceRegistration:
    template = tmp_path / f"{name}.md"
    template.write_text("A signal woke you. Reply briefly.\n", encoding="utf-8")
    return SourceRegistration(
        name=name,
        schema=dict,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=template,
        log_redaction=RedactionPolicy(
            summarize=lambda _payload: "test wake",
            store_raw_trusted=False,
            redact_caller_identifier=True,
        ),
    )


async def test_a_task_its_own_turn_completes_leaves_the_next_turn_runnable(
    tmp_path, monkeypatch
):
    """Self-wake: the most frequent Claw park (11-11 00:06Z, 02:28Z, 04:22Z,
    07:39Z). An inbound task's turn answers it with ``respond_to_a2a_task``,
    and the completion wakes the same agent (``a2a.task_complete``) from
    inside that turn. The wake's turn must reach its model call, and the
    producing turn must retire, delivery included, without waiting on it.
    """
    from kestrel_sdk.native_a2a.types import TaskState

    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    async with booted_agent(
        tmp_path / "agent",
        name="Recipient",
        provenance="test:4023-self-wake",
        features={"TaskFeature"},
    ) as agent:
        completion_wake = asyncio.Event()
        while_wake_ran = {}
        responded = []

        async def generate(kwargs):
            if _last_role(kwargs) == "tool":
                return LLMResponse(content=REPLY)
            text = _prompt_text(kwargs)
            if "[A2A_COMPLETE]" in text and not completion_wake.is_set():
                # Still inside the wake's turn: the producing turn must finish,
                # retirement and delivery included, without it.
                deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
                while not await _submission_acknowledged(agent):
                    if asyncio.get_running_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.02)
                while_wake_ran["producer_acknowledged"] = await _submission_acknowledged(
                    agent
                )
                completion_wake.set()
                return LLMResponse(content=REPLY)
            submitted = re.search(r"native A2A task `([^`]+)`", text)
            if "[A2A_TASK_SUBMITTED]" in text and submitted and not responded:
                responded.append(submitted.group(1))
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="respond-4023",
                            name="respond_to_a2a_task",
                            arguments={
                                "task_id": submitted.group(1),
                                "content": "Done.",
                                "state": "completed",
                            },
                        )
                    ],
                )
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        task = await _submit_task(agent)

        async def completed():
            seen = await agent.task_manager.get_task_for_creator(
                task.id, SENDER, recipient_agent_id=agent.did
            )
            return seen.status.state == TaskState.TASK_STATE_COMPLETED

        await _until((agent,), completed, "the inbound task's turn never completed it")
        assert responded == [task.id]
        try:
            await asyncio.wait_for(completion_wake.wait(), SETTLE_SECONDS)
        except TimeoutError:
            pytest.fail(
                "the completion wake never reached its model call\n" + _diagnostics(agent)
            )
        assert while_wake_ran == {"producer_acknowledged": True}

        (after,) = await _settle(
            (agent,),
            (asyncio.create_task(agent.process_input("after", session_id="s")),),
            "the turn after the completion wake",
        )
        assert after.result() == REPLY
        await _until_completion_wake_settled(agent)
        assert not agent._get_privacy_transition_lock().locked()


async def test_a_turn_that_submits_to_a_cohosted_peer_retires_and_the_peer_turn_runs(
    tmp_path, monkeypatch
):
    """Announced wake: a turn on one co-hosted agent submits an A2A task to
    another. TaskManager announces the task on the submitting Task, which is
    the sender's own turn. The peer's ``a2a.task_submitted`` wake is the
    peer's turn, so the sender's turn retires without waiting for it, and the
    peer's turn runs.

    Enrolled, the wake's supervisor made the sender's retirement wait for the
    peer's whole turn. That is a park as soon as the peer's turn depends on its
    sender. The peer's turn here waits, bounded, for the sender to retire, so
    an enrolled wake parks both until that bound expires.
    """
    async with booted_agent(
        tmp_path / "sender",
        name="Sender",
        provenance="test:4023-cohosted-sender",
        features=set(),
    ) as sender, booted_agent(
        tmp_path / "peer",
        name="Peer",
        provenance="test:4023-cohosted-peer",
        features=set(),
    ) as peer:
        submitted = []
        producing = {}
        while_peer_ran = {}
        peer_ran = asyncio.Event()

        async def sender_generate(kwargs):
            if not submitted:
                submitted.append(await _submit_task(peer, creator_agent_id=sender.did))
            return LLMResponse(content=REPLY)

        async def peer_generate(kwargs):
            if "[A2A_TASK_SUBMITTED]" in _prompt_text(kwargs) and not peer_ran.is_set():
                # Still inside the peer's turn: the sender's turn must retire
                # without it.
                deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
                while not producing["turn"].done():
                    if asyncio.get_running_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.02)
                while_peer_ran["sender_retired"] = producing["turn"].done()
                peer_ran.set()
            return LLMResponse(content=REPLY)

        _script_replies(sender, monkeypatch, sender_generate)
        _script_replies(peer, monkeypatch, peer_generate)
        producing["turn"] = asyncio.create_task(
            sender.process_input("send it", session_id="s")
        )

        (turn,) = await _settle(
            (sender, peer), (producing["turn"],), "the submitting turn"
        )
        assert turn.result() == REPLY
        assert len(submitted) == 1
        try:
            await asyncio.wait_for(peer_ran.wait(), SETTLE_SECONDS)
        except TimeoutError:
            pytest.fail(
                "the peer's wake never reached its model call\n"
                + _diagnostics(sender, peer)
            )
        assert while_peer_ran == {"sender_retired": True}
        await _until(
            (sender, peer),
            lambda: _submission_acknowledged(peer),
            "the peer's submission wake never finished",
        )
        assert not sender._get_privacy_transition_lock().locked()
        assert not peer._get_privacy_transition_lock().locked()


async def test_a_restart_sweep_inside_a_turn_leaves_the_turn_queued_behind_it_runnable(
    tmp_path, monkeypatch
):
    """Restart sweep: ``!restart coordinator`` sweeps post-restart rows from
    inside a turn and starts an ACK supervisor that awaits the
    ``restart.completed`` wake. Enrolled in the sweeping turn, the supervisor
    made that turn's retirement hold the privacy lock while it waited for the
    wake, the wake waited for CONVERSATION, and the turn queued behind the
    sweep held CONVERSATION while it waited for the privacy lock.
    """
    from kestrel_sovereign.features.restart_coordinator.store import (
        get_request,
        insert_request,
        update_status,
    )

    monkeypatch.setenv("KESTREL_TEST_AUTO_APPROVE", "1")
    monkeypatch.setenv("KESTREL_API_KEY", "test-4023-restart-ack")
    async with booted_agent(
        tmp_path / "agent",
        name="Restarted",
        provenance="test:4023-restart",
        features={"RestartCoordinatorFeature"},
    ) as agent:
        feature = next(
            f for f in agent.features.values()
            if type(f).__name__ == "RestartCoordinatorFeature"
        )
        request = await insert_request(
            feature._db,
            requested_by_agent=agent.did,
            reason="test #4023 restart wake",
            allow_agent_request=True,
        )
        assert await update_status(
            feature._db, request.id, status="completed", expected_current_status="pending"
        )
        queued = asyncio.Event()
        swept = []

        async def generate(kwargs):
            if _last_role(kwargs) == "tool" or swept:
                return LLMResponse(content=REPLY)
            # Only the first turn sweeps; it does so once the second turn is
            # waiting for CONVERSATION.
            swept.append(True)
            await asyncio.wait_for(queued.wait(), SETTLE_SECONDS)
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="sweep-1", name="restart_coordinator", arguments={})],
            )

        _script_replies(agent, monkeypatch, generate)
        first = asyncio.create_task(agent.process_input("sweep", session_id="s"))
        while agent._live_turn_id is None:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(agent.process_input("two", session_id="s"))
        await _wait_until_queued(agent)
        queued.set()

        await _settle((agent,), (first, second), "the sweeping and queued turns")
        assert [task.result() for task in (first, second)] == [REPLY, REPLY]

        async def delivered():
            return (await get_request(feature._db, request.id)).wake_delivered

        await _until(
            (agent,), delivered, "the restart wake's ACK supervisor never recorded delivery"
        )
        assert not agent._get_privacy_transition_lock().locked()


@pytest.mark.parametrize("delivery", ["enqueue_signal", "enqueue_durable_cognition"])
async def test_a_signal_a_turn_dispatches_leaves_the_turn_queued_behind_it_runnable(
    tmp_path, monkeypatch, delivery
):
    """The dispatcher's own enqueue Tasks: a turn enqueues a COGNITION wake for
    its own agent while another turn waits for CONVERSATION. Enrolled, the
    enqueuing turn's retirement held the privacy lock until the wake's turn
    ended, the queued turn took CONVERSATION and waited for the privacy lock,
    and the wake waited for CONVERSATION.
    """
    from kestrel_sovereign.signals import DurableConsumerRegistration

    async with booted_agent(
        tmp_path / "agent", name="Waker", provenance="test:4023-dispatch", features=set()
    ) as agent:
        agent.signal_registry.register(_cognition_registration(tmp_path, "test.wake"))
        if delivery == "enqueue_durable_cognition":
            await agent.dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id="test-wake-consumer",
                    source="test.wake",
                    agent_id=agent.did,
                    max_attempts=0,
                )
            )

        async def dispatch(signal):
            if delivery == "enqueue_signal":
                return await agent.dispatcher.enqueue_signal(signal)
            return await agent.dispatcher.enqueue_durable_cognition(
                signal, source_event_id="test-wake-1", consumer_id="test-wake-consumer"
            )

        queued = asyncio.Event()
        woke = asyncio.Event()
        handles = []

        async def generate(kwargs):
            if "A signal woke you" in _prompt_text(kwargs):
                woke.set()
            elif not handles:
                # The second turn is already waiting for CONVERSATION when this
                # turn hands the wake to the dispatcher.
                await asyncio.wait_for(queued.wait(), SETTLE_SECONDS)
                handles.append(
                    await dispatch(
                        Signal(
                            source="test.wake",
                            kind="tick",
                            mode=SignalMode.COGNITION,
                            payload={},
                            target_agent=agent.did,
                        )
                    )
                )
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        first = asyncio.create_task(agent.process_input("one", session_id="s"))
        while agent._live_turn_id is None:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(agent.process_input("two", session_id="s"))
        await _wait_until_queued(agent)
        queued.set()

        await _settle((agent,), (first, second), "the dispatching and queued turns")
        assert [task.result() for task in (first, second)] == [REPLY, REPLY]
        (handle,) = handles
        await _settle((agent,), (handle.task,), "the dispatched wake")
        assert woke.is_set()
        assert not agent._get_privacy_transition_lock().locked()


@pytest.mark.parametrize(
    "delivery", ["enqueue_signal", "enqueue_durable_cognition", "enqueue_durable_terminal"]
)
async def test_a_dispatch_enqueued_in_a_turn_is_not_part_of_it(tmp_path, monkeypatch, delivery):
    """Each dispatcher enqueue Task is created outside the enqueuing turn's
    native privacy lifetime, so the turn's retirement never joins it."""
    from kestrel_sovereign.signals import DurableConsumerRegistration

    async with booted_agent(
        tmp_path / "agent", name="Enqueuer", provenance="test:4023-enqueue", features=set()
    ) as agent:
        agent.signal_registry.register(_cognition_registration(tmp_path, "test.wake"))
        consumer_id = "test-wake-consumer"
        if delivery != "enqueue_signal":
            await agent.dispatcher.register_durable_consumer(
                DurableConsumerRegistration(
                    consumer_id=consumer_id,
                    source="test.wake",
                    agent_id=agent.did,
                    max_attempts=0,
                )
            )
        privacy_lock = agent._get_privacy_transition_lock()
        observed = []
        handles = []

        async def generate(kwargs):
            if not handles:
                signal = Signal(
                    source="test.wake",
                    kind="tick",
                    mode=SignalMode.COGNITION,
                    payload={},
                    target_agent=agent.did,
                )
                enqueue = getattr(agent.dispatcher, delivery)
                options = (
                    {} if delivery == "enqueue_signal"
                    else {"source_event_id": "test-wake-1", "consumer_id": consumer_id}
                )
                handle = await enqueue(signal, **options)
                handles.append(handle)
                observed.append(
                    (
                        privacy_lock._scope_for(asyncio.current_task()) is not None,
                        privacy_lock._scope_for(handle.task),
                    )
                )
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        (turn,) = await _settle(
            (agent,),
            (asyncio.create_task(agent.process_input("enqueue", session_id="s")),),
            "the enqueuing turn",
        )
        assert turn.result() == REPLY
        (handle,) = handles
        await _settle((agent,), (handle.task,), "the enqueued dispatch")

    assert observed == [(True, None)]


@pytest.mark.parametrize(
    "producer",
    [
        "_on_task_submitted",
        "_on_background_task_complete",
        "supervise_terminal_delivery",
        "harvest_detached_delivery",
    ],
)
async def test_a_delivery_supervisor_started_in_a_turn_is_not_part_of_it(
    tmp_path, monkeypatch, producer
):
    """Every supervisor that awaits a dispatch's delivery, started from turn
    context, holds neither of the turn's two authorities: it is not a member
    of the turn's native privacy lifetime, and it holds none of the turn's
    delegated locks.

    Each producer runs where a tool runs: in an admitted member of the turn's
    lifetime whose context carries the turn's delegated CONVERSATION hold,
    built as ``OrchestratorEngineMixin`` builds a tool operation's Task."""
    import contextvars

    import kestrel_sovereign.agent.event_manager as event_manager
    from kestrel_sdk.native_a2a.types import Task, TaskState, TaskStatus

    from kestrel_sovereign._async_ownership import await_owned_task
    from kestrel_sovereign.signals.delivery import (
        harvest_detached_delivery,
        supervise_terminal_delivery,
    )

    async with booted_agent(
        tmp_path / "agent", name="Producer", provenance="test:4023-detached", features=set()
    ) as agent:
        observed = []
        locks = agent._get_lock_manager()
        privacy_lock = agent._get_privacy_transition_lock()

        def observe():
            observed.append(
                (
                    privacy_lock._scope_for(asyncio.current_task()),
                    locks.is_owned_by_current_task(ResourceLock.CONVERSATION),
                )
            )

        async def deliver(**kwargs):
            observe()

        class _Handle:
            async def wait(self):
                observe()
                return SimpleNamespace(status=SimpleNamespace(value="ok"), error=None)

        async def enqueue():
            return _Handle()

        monkeypatch.setattr(agent, "_deliver_a2a_wake_until_durable", deliver)
        monkeypatch.setattr(
            event_manager, "require_native_task_snapshot", lambda candidate, _did: candidate
        )
        candidate = SimpleNamespace(
            task=Task(
                id="task-4023",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            ),
            ownership=SimpleNamespace(creator_agent_id="did:kestrel:creator-4023"),
            input_revision=1,
        )
        feature = next(iter(agent.features.values()))
        in_tool = []

        async def produce():
            in_tool.append(
                (
                    privacy_lock._scope_for(asyncio.current_task()) is not None,
                    locks.is_owned_by_current_task(ResourceLock.CONVERSATION),
                )
            )
            if producer == "supervise_terminal_delivery":
                supervise_terminal_delivery(
                    feature, _Handle(), label="test", task_name="test-4023-supervisor"
                )
            elif producer == "harvest_detached_delivery":
                harvest_detached_delivery(
                    agent._track_background_task,
                    enqueue,
                    label="test",
                    task_name="test-4023-harvest",
                )
            else:
                getattr(agent, producer)(candidate)

        async def generate(kwargs):
            start = privacy_lock.capture_native_work_factory(acquire_span=False)
            context = contextvars.copy_context()
            locks.delegate_current_task_ownership(context)
            tool = start(produce, context=context, name="test-4023-tool")
            outcome = await await_owned_task(tool)
            assert outcome.error is None, outcome.error
            return LLMResponse(content=REPLY)

        _script_replies(agent, monkeypatch, generate)
        assert await agent.process_input("produce", session_id="s") == REPLY
        deadline = asyncio.get_running_loop().time() + SETTLE_SECONDS
        while not observed and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

    assert in_tool == [(True, True)], (
        "the producer did not run inside the turn's privacy lifetime and CONVERSATION hold"
    )
    assert observed == [(None, False)]


async def test_detachment_is_scoped_to_its_block():
    """The context manager clears both carriers only for Tasks created inside
    it, and restores the creator's own context on exit."""
    from kestrel_sovereign._async_ownership import _OWNED_COORDINATORS, bind_owned_coordinator
    from kestrel_sovereign.signals.delivery import detached_from_enclosing_turn
    from kestrel_sovereign.signals.lock_manager import _delegated_lock_ownership
    from kestrel_sovereign.storage.privacy_wrapper import (
        bind_transition_lock_reentry,
        current_bound_reentry_token,
    )

    coordinator, token, delegation = object(), object(), {1: frozenset({object()})}
    reset = _delegated_lock_ownership.set(delegation)
    try:
        with bind_owned_coordinator(coordinator), bind_transition_lock_reentry(token):

            async def carried():
                return (
                    _OWNED_COORDINATORS.get(),
                    current_bound_reentry_token(),
                    _delegated_lock_ownership.get(),
                )

            with detached_from_enclosing_turn():
                detached = asyncio.create_task(carried())
            attached = asyncio.create_task(carried())
            assert await detached == ((), None, {})
            assert await attached == ((coordinator,), token, delegation)
            assert _OWNED_COORDINATORS.get() == (coordinator,)
            assert current_bound_reentry_token() is token
            assert _delegated_lock_ownership.get() is delegation
    finally:
        _delegated_lock_ownership.reset(reset)
