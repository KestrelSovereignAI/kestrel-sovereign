"""Privacy contracts for routine memory/request diagnostics (#2332, #3318)."""

import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.logging_config import (
    CONTENT_DIGEST_HEX_CHARS,
    content_log_summary,
)
from tests.utils.log_records import records_containing


ROOT = Path(__file__).resolve().parents[2]

# Starts every line of the user message below, so a log that keeps any line,
# any prefix past a few characters, or the whole message carries it. None of
# its letters is a hex digit, so no digest or id can spell it by chance.
MARKER = "ZqXjVw"
SESSION = "3318-log-privacy-session"


def test_request_logs_do_not_embed_history_or_response_snippets():
    agent_source = (ROOT / "kestrel_sovereign/kestrel_agent.py").read_text()
    orchestrator_source = (
        ROOT / "kestrel_sovereign/agent/orchestrator_engine.py"
    ).read_text()

    assert "[SESSION-DEBUG]" not in agent_source
    assert "history[0].get('content'" not in agent_source
    assert "response.content[:150]" not in agent_source
    assert "final_content[:300]" not in orchestrator_source
    assert "response[:300]" not in orchestrator_source
    assert "result_json[:200]" not in orchestrator_source


def test_content_log_summary_states_length_and_digest_and_none_of_the_text():
    text = f"{MARKER} the heron came back to the mill pond"
    summary = content_log_summary(text)

    match = re.fullmatch(r"chars=(\d+) digest=([0-9a-f]+)", summary)
    assert match, summary
    assert int(match[1]) == len(text)
    assert len(match[2]) == CONTENT_DIGEST_HEX_CHARS
    assert MARKER not in summary and "heron" not in summary


def test_content_log_summary_matches_equal_text_and_tells_different_text_apart():
    assert content_log_summary("yes") == content_log_summary("yes")
    assert content_log_summary("yes") != content_log_summary("yes.")


def test_content_log_summary_digest_is_not_an_unkeyed_hash():
    """An unkeyed hash of a short message is a lookup away from its text."""
    for text in ("yes", "no", ""):
        digest = content_log_summary(text).split("digest=", 1)[1]
        unkeyed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert digest != unkeyed[:CONTENT_DIGEST_HEX_CHARS]


def test_content_log_summary_accepts_a_lone_surrogate():
    """JSON admits lone surrogates; describing one must not fail the turn."""
    assert content_log_summary("a\ud800b").startswith("chars=3 digest=")


def _user_message(entry_point: str) -> str:
    """A long multi-line message, like the 393-word one #3318 found logged."""
    return "\n".join(
        f"{MARKER}{line:02d} {entry_point} note: the heron by the old mill pond "
        f"still has not come back, and the reeds need cutting before frost."
        for line in range(12)
    )


@asynccontextmanager
async def _booted_agent(tmp_path, monkeypatch):
    """A real agent on real storage; only the provider call is scripted.

    ``KESTREL_DATA_KEY`` makes the conversation store encrypt the turn, as it
    does in production: the log is then the only plaintext copy left to find.
    """
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from tests.shared.genesis_audit import complete_deterministic_genesis_audit

    monkeypatch.setenv("KESTREL_DATA_KEY", "test-master-key-for-encryption-32chars!")
    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="LogPrivacy"
    )
    llm_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=llm_service,
    )
    try:
        await agent.initialize()
        await complete_deterministic_genesis_audit(
            agent, provenance="test:turn_log_privacy"
        )
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)

        async def generate_with_messages(*_args, **_kwargs):
            return LLMResponse(content="Noted.")

        async def stream_with_tool_detection(*_args, **_kwargs):
            yield "Noted."

        monkeypatch.setattr(llm_service, "generate_with_messages", generate_with_messages)
        monkeypatch.setattr(
            llm_service, "stream_with_tool_detection", stream_with_tool_detection
        )
        yield agent
    finally:
        await agent.shutdown()
        await llm_service.close()


@pytest.mark.asyncio
async def test_a_turn_logs_no_part_of_the_users_message(tmp_path, monkeypatch, caplog):
    """#3318: ContextBuilder logged every turn's user message verbatim at INFO.

    Both entry points run a whole turn, through shutdown so work the turn left
    behind is covered too, with every logger at DEBUG. The assertion is on the
    marker in any field of any record, not on the wording of one log line.
    """
    plain = _user_message("non-streaming")
    streamed = _user_message("streaming")
    queried = []

    async with _booted_agent(tmp_path, monkeypatch) as agent:
        builder = agent.context_manager.context_builder
        real_retrieve_context = builder.retrieve_context

        async def recording_retrieve_context(query, *args, **kwargs):
            queried.append(query)
            return await real_retrieve_context(query, *args, **kwargs)

        monkeypatch.setattr(builder, "retrieve_context", recording_retrieve_context)
        caplog.set_level(logging.DEBUG)
        caplog.clear()

        assert await agent.process_input(plain, session_id=SESSION) == "Noted."
        chunks = [
            chunk
            async for chunk in agent.process_input_streaming(
                streamed, session_id=SESSION
            )
        ]
        assert "".join(chunks) == "Noted."

    # The message reached the code that used to log it, that code still logged
    # (facts only), and DEBUG records were captured: a clean log below is the
    # fix rather than a turn that never got that far or a capture that missed.
    assert queried == [plain, streamed]
    retrieval_lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == "kestrel_sovereign.agent.context_builder"
    ]
    for message in (plain, streamed):
        assert any(content_log_summary(message) in line for line in retrieval_lines)
    assert any(record.levelno == logging.DEBUG for record in caplog.records)
    leaked = records_containing(caplog.records, MARKER)
    assert not leaked, [
        f"{record.levelname} {record.name} {record.pathname}:{record.lineno}"
        for record in leaked
    ]
