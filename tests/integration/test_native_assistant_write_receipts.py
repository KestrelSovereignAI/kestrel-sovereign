"""A booted agent retains the exact original assistant-write identity.

Two actual writes carry the same signal correlation. This is a persistence
identity test, not a provider, native MCP install or reflection acceptance.
"""
from uuid import uuid4

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from kestrel_sdk.payer_policy import PayerKind, PayerPolicy, PayerSpec

from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage.turn_provenance import WakeReplyReceipt
from tests.shared.booted_agent import booted_agent


@pytest_asyncio.fixture
async def assistant_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KESTREL_HOME", str(tmp_path))
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))
    monkeypatch.setenv("KESTREL_DB_BACKEND", "sqlite")
    monkeypatch.setenv("KESTREL_DATA_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("KESTREL_DISABLE_CONVERSATION_EMBEDDINGS", "true")
    (tmp_path / "kestrel.toml").write_text('[llm]\nroute_priority=[]\n')
    policy = PayerPolicy.model_validate({
        **PayerPolicy.host_env_default().model_dump(),
        "llm": PayerSpec(vendor="local", kind=PayerKind.NONE),
    })
    async with booted_agent(tmp_path, name="OwnedAssistantReceipts",
            provenance="test:1990-original-assistant-receipt", features=set(),
            payer_policy=policy) as agent:
        assert agent.llm_service.disabled
        assert not agent.llm_service.providers
        yield agent


@pytest.mark.asyncio
@pytest.mark.parametrize("with_wake", [False, True])
async def test_assistant_receipt_identifies_its_original_native_write(assistant_agent, monkeypatch, with_wake):
    agent = assistant_agent
    original = agent.privacy_agent.add_conversation
    writes = []

    async def interleave_original_and_later_write(*args, **kwargs):
        first = await original(*args, **kwargs)
        later = await original("assistant", "Later independently persisted reply", **kwargs)
        writes.extend((first, later))
        return first

    monkeypatch.setattr(agent.privacy_agent, "add_conversation", interleave_original_and_later_write)
    wake = WakeReplyReceipt(signal_id=uuid4().hex) if with_wake else None
    receipt = await agent._persist_assistant_conversation("Original owned reply",
        session_id=str(uuid4()), wake_reply=wake)
    assert len(writes) == 2 and all(type(row) is int and row > 0 for row in writes)
    assert writes[0] != writes[1]
    assert receipt == writes[0] and (wake is None or wake.message_id == writes[0]), (
        "Assistant persistence discarded its own row receipt or selected a later correlation match"
    )
    rows = await agent._raw_storage.conversation.get_messages_by_ids(writes)
    assert {row["id"]: row["content"] for row in rows} == {
        writes[0]: "Original owned reply", writes[1]: "Later independently persisted reply"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [PrivacyMode.EPHEMERAL, PrivacyMode.ISOLATED, PrivacyMode.DEIDENTIFIED])
async def test_volatile_assistant_never_reuses_a_prior_durable_wake_receipt(assistant_agent, mode):
    agent = assistant_agent
    signal_id = uuid4().hex
    prior = WakeReplyReceipt(signal_id=signal_id)
    row_id = await agent._persist_assistant_conversation("Earlier durable reply", wake_reply=prior)
    assert type(row_id) is int and row_id == prior.message_id
    await agent.set_privacy_mode(mode)
    assert agent.privacy_mode == mode and not agent.privacy_agent.can_store("conversation")
    before = await agent._raw_storage.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (agent.did,))
    wake = WakeReplyReceipt(signal_id=signal_id)
    receipt = await agent._persist_assistant_conversation("Current volatile reply", wake_reply=wake)
    assert receipt is None and wake.message_id is None, "Volatile writes must not reuse a durable correlation match"
    assert await agent._raw_storage.db.fetchone(
        "SELECT COUNT(*) FROM conversation_history WHERE agent_id = ?", (agent.did,)) == before
    history = await agent.privacy_agent.get_conversation_history()
    assert history[-1]["content"] == "Current volatile reply"
