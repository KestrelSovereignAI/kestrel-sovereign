"""Real external rows reach existing consolidation after original-agent shutdown.

This does not enable generation or claim LLM reflection/native account acceptance.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.encoders import jsonable_encoder
from kestrel_sdk.storage.database.interface import QueryError

from kestrel_sovereign.agent.custody import ReleaseOutcome
from kestrel_sovereign.features.memory.feature import MemoryFeature
from kestrel_sovereign.features.memory.reflection_hook import ReflectionSleepHook
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.storage import memory_consolidator
from kestrel_sovereign.storage.db.sqlite import _aiosqlite_worker_is_alive
from tests.integration.test_native_mcp_log_turn import exchange_arguments
from tests.integration.test_native_mcp_memory_tools import observe_native_call
from tests.integration.test_native_mcp_memory_writes import (
    assistant_agent as _assistant_agent,
)
from tests.integration.test_native_mcp_memory_writes import publication as _publication
from tests.integration.test_native_mcp_memory_writes import pytestmark as write_marks
from tests.integration.test_native_mcp_tool_context import issue_native_tokens
from tests.utils.native_go_peer import retire_peer
from tests.utils.native_memory_restart_process import close_restart_agent

assistant_agent = _assistant_agent
publication = _publication
pytestmark = write_marks


def restart_request(agent, receipt_path, *, receipts=(), mode=PrivacyMode.NORMAL):
    return {"did": agent.did, "storage_path": agent.storage_path,
        "payer_policy": agent._resolve_payer_policy().model_dump(mode="json"),
        "receipts": receipts, "mode": mode.value, "receipt_path": str(receipt_path)}


@asynccontextmanager
async def native_restart_process(*, observation_seconds=20, retain_storage=False):
    # Reopen the same runtime imported by the parent, including wheel-only
    # acceptance. The test checkout supplies the child driver, not Core.
    source_paths = list(dict.fromkeys([
        str(Path(memory_consolidator.__file__).resolve().parents[2]),
        str(Path(__file__).resolve().parents[2]),
    ]))
    if os.environ.get("PYTHONPATH"):
        source_paths.append(os.environ["PYTHONPATH"])
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(source_paths)}
    arguments = [sys.executable, "-P", "-B", "-m", "tests.utils.native_memory_restart_process"]
    if retain_storage:
        arguments.append("--retain-storage")
    process = await asyncio.create_subprocess_exec(*arguments, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True, env=environment)
    try:
        async with asyncio.timeout(observation_seconds):
            yield process
    finally:
        await retire_peer(process, process_group=True)


@pytest.mark.parametrize("mode", [PrivacyMode.NORMAL, PrivacyMode.EPHEMERAL])
async def test_original_external_rows_reach_privacy_governed_consolidation_after_restart(
    publication, mode, monkeypatch,
):
    # A normal source-checkout CWD must not shadow the installed parent Core.
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    original, receipts, persisted = await persist_external_exchanges(publication)
    owner, _ = publication
    await retire_original_publication(owner)
    receipt_path = Path(original.storage_path).parent / (mode.value + "-restarted-memory.json")
    assert not receipt_path.exists()
    request = restart_request(original, receipt_path, receipts=receipts, mode=mode)
    async with native_restart_process() as process:
        stdout, stderr = await process.communicate(json.dumps(request).encode())
        assert process.returncode == 0, (stderr.decode()[-4000:], stdout.decode()[-1000:])
    result = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["pid"] == process.pid and result["pid"] != os.getpid()
    assert result["core_source"] == str(Path(memory_consolidator.__file__).resolve()), (
        "Restart probe imported another Core source tree")
    assert result["did"] == original.did and result["generation_disabled"]
    assert result["cleanup"]["primary_sqlite_closed"]
    assert result["cleanup"]["original_primary_workers"] >= 1
    assert {row["id"]: row for row in result["rows"]} == {
        row["id"]: row for row in jsonable_encoder(persisted)}, (
            "Restart changed the original native exchange bytes or provenance")
    report = result["report"]
    assert "error" not in report, report
    again = result["repeat_report"]
    assert "error" not in again and again["episodes_created"] == 0
    if mode == PrivacyMode.EPHEMERAL:
        assert report["privacy_blocked"] and report["episodes_created"] == 0
        assert not result["stored_episode_ids"], (
            "Restarted private memory processing persisted an external derivative")
    else:
        assert report["episodes_created"] == 1, (
            "Canonical consolidation did not consume the original external rows after restart", report)
        episodes = result["episodes"]
        assert len(episodes) == 1 and set(episodes[0]["key_message_ids"]) == set(map(str, receipts))
        assert "orbital" in episodes[0]["title"].lower() + episodes[0]["summary"].lower()
        assert [episode["id"] for episode in result["repeat_episodes"]] == [episodes[0]["id"]], (
            "Reprocessing an external exchange duplicated its canonical episode")


async def persist_external_exchanges(publication):
    """Store two real native exchanges and retain their original receipts."""
    owner, http = publication
    owner.publish()
    original = owner.agent
    token = (await issue_native_tokens(owner, http, scope="memory:write"))["access_token"]
    receipts = []
    for number in range(2):
        result, failure, _ = await observe_native_call(owner, token, "log_turn",
            exchange_arguments(user_content=f"Remember the orbital garden plan {number}.",
                assistant_content=f"The external assistant discussed orbital garden {number}."))
        assert failure is None and not result.is_error
        data = result.structured_content["data"]
        assert data["memory_processing"] == "not_scheduled"
        receipts.extend((data["user_message_id"], data["assistant_message_id"]))
    assert len(set(receipts)) == 4
    persisted = await original._raw_storage.conversation.get_messages_by_ids(receipts)
    assert len(persisted) == 4
    assert not await original.memory_system.get_episodes()
    return original, receipts, persisted


async def retire_original_publication(owner):
    """Close the original SDK and agent before opening the same database."""
    original = owner.agent
    # End the real SDK and agent owners, then let a fresh Python process
    # reconstruct the same identity/database. No store is copied or substituted.
    await owner.close()
    closed = await close_restart_agent(original)
    assert closed["primary_sqlite_closed"] and closed["original_primary_workers"] >= 1
    assert owner._resident.done() and not owner.registry.is_current(owner.binding)


async def test_attestation_schema_failure_prevents_memory_hook_activation(publication):
    owner, _ = publication
    agent = owner.agent
    db = agent._raw_storage.db
    assert not await db.table_exists("memory_attestation_state")
    await db.execute("CREATE VIEW memory_attestation_state AS SELECT 'conflicting schema' AS agent_id")
    feature = MemoryFeature(agent)
    with pytest.raises(RuntimeError, match="^memory_attestation_state was not created$"):
        await feature.initialize()
    assert not any(isinstance(hook, ReflectionSleepHook) for hook in agent.sleep_hooks)


async def test_restart_process_observation_has_its_own_deadline(publication):
    owner, _ = publication
    await owner.close()
    await close_restart_agent(owner.agent)
    receipt = Path(owner.agent.storage_path).parent / "bounded-restart.json"
    request = restart_request(owner.agent, receipt)
    with pytest.raises(TimeoutError):
        async with native_restart_process(observation_seconds=0.001) as process:
            await process.communicate(json.dumps(request).encode())
    assert process.returncode is not None, "Timed-out restart child was not reaped"


async def test_failed_feature_close_cannot_acknowledge_restart_shutdown(publication, monkeypatch):
    owner, _ = publication
    await owner.close()
    agent = owner.agent
    feature = next(f for f in agent.features.values() if type(f).__name__ == "MemoryAgencyFeature")
    native_errors = []

    async def native_failed_close():
        try:
            await agent._raw_storage.db.execute("SELECT absent_restart_close_column FROM conversation_history")
        except QueryError as error:
            native_errors.append(error)
            raise

    refusal = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(feature, "shutdown", native_failed_close)
            try:
                await close_restart_agent(agent)
            except RuntimeError as error:
                refusal = error
        assert native_errors, "Feature close did not reach the original native SQL fault"
        assert refusal is not None, "A failed original feature close was accepted as completed restart shutdown"
    finally:
        await close_restart_agent(agent)


async def test_retained_native_storage_cannot_acknowledge_restart_shutdown(publication, monkeypatch):
    owner, _ = publication
    await owner.close()
    agent = owner.agent
    backend = agent._raw_storage.db.backend
    original_connection = backend._connection
    assert backend.is_connected and _aiosqlite_worker_is_alive(original_connection)
    retained = []

    async def retain_native_storage():
        retained.append(backend._connection)
        return ReleaseOutcome.RETAINED

    refusal = None
    try:
        with monkeypatch.context() as fault:
            fault.setattr(agent._raw_storage, "close", retain_native_storage)
            try:
                await close_restart_agent(agent)
            except RuntimeError as error:
                refusal = error
        assert retained and retained[0] is original_connection
        assert backend.is_connected and _aiosqlite_worker_is_alive(original_connection)
        assert refusal is not None, "An original live SQLite worker was accepted as completed restart shutdown"
    finally:
        await close_restart_agent(agent)


async def test_original_child_cleanup_failure_never_publishes_success(publication):
    owner, _ = publication
    await owner.close()
    await close_restart_agent(owner.agent)
    receipt = Path(owner.agent.storage_path).parent / "failed-child-cleanup.json"
    request = restart_request(owner.agent, receipt)
    async with native_restart_process(retain_storage=True) as process:
        _stdout, stderr = await process.communicate(json.dumps(request).encode())
        assert process.returncode != 0 and not receipt.exists(), (
            "Failed original child cleanup was published as successful restart evidence")
        assert b"Restart shutdown did not retire its original cleanup and storage owners" in stderr
