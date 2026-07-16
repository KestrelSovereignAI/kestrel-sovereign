"""Shared transport contract for operator-notice turn injection (#2531)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign.agent.operator_signals import (
    OperatorSignalBatch,
    OperatorSignalEvent,
    OperatorSignalProducer,
)
from kestrel_sovereign.agent.streaming import StreamingMixin
from kestrel_sovereign.kestrel_agent import KestrelAgent


class _StopAfterLLM(RuntimeError):
    """Terminate a turn after capturing its exact outbound LLM contract."""


class _CapturingLLM:
    def __init__(self):
        self.calls = []

    async def generate_with_messages(self, **kwargs):
        self.calls.append(kwargs)
        raise _StopAfterLLM

    async def stream_with_tool_detection(self, **kwargs):
        self.calls.append(kwargs)
        raise _StopAfterLLM
        yield  # pragma: no cover - makes this an async generator


class _FixedProducer(OperatorSignalProducer):
    def __init__(self, batch):
        self.batch = batch
        self.calls = []

    async def collect_for_turn(self, **kwargs):
        self.calls.append(kwargs)
        return self.batch


class _PrivacyAgent:
    def __init__(self, *, fail_operator_persistence):
        self.fail_operator_persistence = fail_operator_persistence
        self.persist_calls = []
        self.durable_calls = []
        self.privacy_config = SimpleNamespace(allows_cloud_llm=lambda: True)
        self.privacy_mode = SimpleNamespace(name="NORMAL")

    async def get_conversation_history(self, **_kwargs):
        return []

    async def add_conversation(self, role, content, **kwargs):
        call = {"role": role, "content": content, **kwargs}
        self.persist_calls.append(call)
        if self.fail_operator_persistence and (kwargs.get("metadata") or {}).get(
            "operator_signal"
        ):
            raise RuntimeError("operator persistence unavailable")
        self.durable_calls.append(call)


class _ContextManager:
    def __init__(self, state_of_mind):
        self.state_of_mind = state_of_mind
        self.calls = []

    async def build_context(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            system_prompt="system prompt",
            messages=[{"role": "assistant", "content": "prior answer"}],
            total_tokens=42,
            budget_summary={"total_budget": 1000, "total_used": 42},
            warnings=[],
            dynamic_user_context="retrieved context",
            episode_count=0,
            memory_count=1,
            rag_chunks=2,
            degraded_mode=False,
            state_of_mind=self.state_of_mind,
        )


class _ObservabilityStore:
    async def log_metric(self, **_kwargs):
        return None

    async def log_tool_call(self, **_kwargs):
        return "llm-event"


def _operator_batch(role):
    if role is None:
        return None
    return OperatorSignalBatch(
        role=role,
        content=f"{role} operator notice",
        keep_trailing_system=role == "system",
        events=[
            OperatorSignalEvent(
                source="operator.contract_test",
                content="operator fact",
                payload={"kind": "test"},
            )
        ],
        route_label="contract/test-model",
        fallback=role == "user",
    )


def _turn_agent(*, role, fail_operator_persistence):
    llm = _CapturingLLM()
    state_of_mind = object()
    context_manager = _ContextManager(state_of_mind)
    privacy_agent = _PrivacyAgent(fail_operator_persistence=fail_operator_persistence)
    batch = _operator_batch(role)
    producer = _FixedProducer(batch) if batch is not None else None
    agent = SimpleNamespace(
        did="did:test:operator-injection",
        hooks_manager=None,
        llm_service=llm,
        privacy_agent=privacy_agent,
        context_manager=context_manager,
        features={},
        operator_signal_producer=producer,
        observability_store=_ObservabilityStore(),
        extension=None,
        user_prompt_template="{context}\n{query}",
        _privacy_mode=SimpleNamespace(value="NORMAL"),
        _session_briefed=False,
        _maybe_compact_codex_thread=AsyncMock(),
        _get_governing_constitution=AsyncMock(return_value="constitution"),
        _assemble_post_build_system_prompt=(
            lambda system_prompt, _context, *, user_prompt: system_prompt
        ),
        _build_all_tools=lambda: [],
        _lazy_attachment_hint=lambda attachments: "",
        _make_inline_tool_executor=lambda _session_id: None,
    )
    return agent, llm, context_manager, privacy_agent, producer


async def _run_until_llm(transport, agent):
    if transport == "streaming":
        async for _ in StreamingMixin._process_input_streaming_traced_locked(
            agent,
            "current question",
            "test-model",
            "session-1",
            None,
        ):
            pass
        return

    await KestrelAgent._process_input_traced_locked(
        agent,
        "current question",
        "test-model",
        "session-1",
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["non_streaming", "streaming"])
@pytest.mark.parametrize(
    ("role", "fail_operator_persistence"),
    [
        pytest.param("system", False, id="inline-system"),
        pytest.param("user", False, id="fallback-user-persisted"),
        pytest.param("user", True, id="fallback-user-persist-failed"),
        pytest.param(None, False, id="producer-absent"),
    ],
)
async def test_operator_injection_shared_transport_contract(
    transport,
    role,
    fail_operator_persistence,
):
    """Both transports preserve one matrix, including failed-turn history.

    The capturing LLM always raises, reproducing the #2009 failure boundary:
    an inline system notice was sent in-flight but must not survive in durable
    history when no assistant row follows it.
    """
    agent, llm, context_manager, privacy_agent, producer = _turn_agent(
        role=role,
        fail_operator_persistence=fail_operator_persistence,
    )

    with pytest.raises(_StopAfterLLM):
        await _run_until_llm(transport, agent)

    assert len(llm.calls) == 1
    llm_call = llm.calls[0]
    expected_roles = ["system", "assistant", "user"]
    if role is not None:
        expected_roles.append(role)
    assert [message["role"] for message in llm_call["messages"]] == expected_roles
    if role is not None:
        assert llm_call["messages"][-1] == {
            "role": role,
            "content": f"{role} operator notice",
        }
    assert llm_call["keep_trailing_system"] is (role == "system")

    assert len(context_manager.calls) == 1
    if producer is None:
        operator_persists = [
            call
            for call in privacy_agent.persist_calls
            if (call.get("metadata") or {}).get("operator_signal")
        ]
        assert operator_persists == []
        return

    assert len(producer.calls) == 1
    assert producer.calls[0]["state_of_mind"] is context_manager.state_of_mind
    operator_persists = [
        call
        for call in privacy_agent.persist_calls
        if (call.get("metadata") or {}).get("operator_signal")
    ]
    if role == "system":
        assert operator_persists == []
        assert [call["role"] for call in privacy_agent.durable_calls] == ["user"]
        return

    assert len(operator_persists) == 1
    assert operator_persists[0] == {
        "role": "user",
        "content": "user operator notice",
        "metadata": {
            "sent_form": True,
            "operator_signal": True,
            "operator_signal_sources": ["operator.contract_test"],
            "operator_signal_fallback": True,
        },
        "session_id": "session-1",
        "rendered_content": "user operator notice",
    }
    durable_operator_calls = [
        call
        for call in privacy_agent.durable_calls
        if (call.get("metadata") or {}).get("operator_signal")
    ]
    assert len(durable_operator_calls) == (0 if fail_operator_persistence else 1)
