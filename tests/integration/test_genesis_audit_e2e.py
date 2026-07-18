"""
Integration tests for genesis audit (CW-001 fix).
Tests that genesis audit properly validates constitutions during agent creation.
"""
import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from unittest.mock import AsyncMock
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.llm.service import LLMService
from kestrel_sovereign.privacy import PrivacyMode
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.storage import Storage


@pytest.mark.asyncio
async def test_inception_boundary_persists_pass_and_event(tmp_path):
    """A configured creation audit is durable before inception returns."""
    observed_prompt = None

    async def deterministic_safe_auditor(prompt):
        nonlocal observed_prompt
        observed_prompt = prompt
        return {
            "risk_level": 1,
            "reasoning": "Deterministic safe constitution fixture.",
        }

    credentials = await create_kestrel_identity_async(
        str(tmp_path / "agent"),
        agent_name="Audited",
        identity_method="did:pkh",
        genesis_auditor=deterministic_safe_auditor,
        genesis_audit_provenance="test:deterministic_safe",
    )

    assert observed_prompt is not None
    assert "Kestrel Constitution" in observed_prompt
    async with Storage(credentials.db_path, agent_id=credentials.agent_did) as storage:
        node = await storage.get_node(credentials.agent_did)
        receipt = node.properties["genesis_audit"]
        assert receipt["status"] == "passed"
        assert receipt["audited"] is True
        assert receipt["provenance"] == "test:deterministic_safe"
        assert receipt["constitution_hash"] == node.properties["constitution_hash"]

        history = await storage.get_conversation_history(limit=10)
        events = [
            row
            for row in history
            if (row.get("metadata") or {}).get("event") == "genesis_audit"
        ]
        assert len(events) == 1
        assert events[0]["metadata"]["result"]["status"] == "passed"


@pytest.mark.asyncio
async def test_risk_three_inception_removes_all_identity_artifacts(tmp_path):
    """A rejected governing constitution never leaves a partial agent."""
    output_dir = tmp_path / "rejected"

    async def deterministic_rejection(_prompt):
        return {"risk_level": 3, "reasoning": "Unsafe deterministic fixture."}

    with pytest.raises(ValueError, match="Risk Level: 3"):
        await create_kestrel_identity_async(
            str(output_dir),
            agent_name="Rejected",
            identity_method="did:pkh",
            genesis_auditor=deterministic_rejection,
            genesis_audit_provenance="test:deterministic_risk3",
        )

    assert not (output_dir / "kestrel_prime.db").exists()
    assert list(output_dir.glob("*.pem")) == []
    assert list(output_dir.glob("*.key.enc")) == []
    assert list(output_dir.glob("*.json")) == []


@pytest.mark.asyncio
async def test_pending_first_turn_blocks_then_completes_once_across_restart(tmp_path):
    """The real non-streaming boundary admits no cognition before pass."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "deferred"),
        agent_name="Deferred",
        identity_method="did:pkh",
    )

    first_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=first_service,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None

    audit_calls = 0

    async def unavailable_then_safe(_prompt):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 1:
            return {
                "risk_level": 1,
                "reasoning": "No configured route.",
                "audited": False,
            }
        return {"risk_level": 1, "reasoning": "Configured safe auditor."}

    agent.get_audit_response = unavailable_then_safe
    cognition_calls = 0

    async def cognition_after_receipt(*_args, **_kwargs):
        nonlocal cognition_calls
        cognition_calls += 1
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["genesis_audit"]["status"] == "passed"
        return "PONG"

    agent._process_input_traced_locked = cognition_after_receipt
    try:
        blocked = await agent.process_input("PONG")
        assert "GENESIS AUDIT PENDING" in blocked
        assert cognition_calls == 0

        response = await agent.process_input("PONG")
        assert response == "PONG"
        assert audit_calls == 2
        assert cognition_calls == 1
    finally:
        await agent.shutdown()

    # A completed receipt survives process restart and cannot silently rerun.
    restarted = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await restarted.initialize()
    restarted.bootstrap_service = None
    restarted.get_audit_response = AsyncMock(
        side_effect=AssertionError("completed genesis audit must not rerun")
    )
    restarted._process_input_traced_locked = AsyncMock(return_value="PONG")
    try:
        assert await restarted.process_input("PONG") == "PONG"
        restarted.get_audit_response.assert_not_awaited()
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_deferred_audit_uses_exact_hash_bound_bytes(tmp_path):
    """Runtime-only prompt appendages are not misbound to the base file hash."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "exact-bytes"),
        agent_name="ExactBytes",
        identity_method="did:pkh",
    )
    service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=service,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None

    marker = "RUNTIME_EXTENSION_MUST_NOT_ENTER_GENESIS_AUDIT"

    class RuntimeExtension:
        def get_constitution_amendments(self):
            return marker

    agent.extension = RuntimeExtension()
    observed_prompt = None

    async def capture_audit(prompt):
        nonlocal observed_prompt
        observed_prompt = prompt
        return {"risk_level": 1, "reasoning": "Exact stored bytes passed."}

    agent.get_audit_response = capture_audit
    agent._process_input_traced_locked = AsyncMock(return_value="PONG")
    try:
        assert await agent.process_input("PONG") == "PONG"
        node = await agent.storage.get_node(agent.agent_id)
        stored = await agent.storage.retrieve_file(
            node.properties["constitution_hash"]
        )
        assert stored.decode("utf-8") in observed_prompt
        assert marker not in observed_prompt
    finally:
        await agent.shutdown()
        await service.close()


@pytest.mark.asyncio
async def test_deferred_audit_refuses_hash_mismatched_storage(tmp_path):
    """A content-address lookup returning different bytes cannot be audited."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "mismatched-bytes"),
        agent_name="MismatchedBytes",
        identity_method="did:pkh",
    )
    service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=service,
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None
    auditor = AsyncMock(
        return_value={"risk_level": 1, "reasoning": "Must not be called."}
    )
    agent.get_audit_response = auditor
    original_retrieve_file = agent.storage.retrieve_file
    agent.storage.retrieve_file = AsyncMock(return_value=b"tampered bytes")
    try:
        response = await agent.process_input("PONG")
        assert "GENESIS AUDIT BLOCKED" in response
        auditor.assert_not_called()
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["genesis_audit"]["status"] == "pending"
        assert (
            node.properties["genesis_audit"]["last_error"]
            == "constitution_hash_mismatch"
        )
    finally:
        agent.storage.retrieve_file = original_retrieve_file
        await agent.shutdown()
        await service.close()


@pytest.mark.asyncio
async def test_concurrent_first_turns_share_one_genesis_audit(tmp_path):
    """Two racing cognition requests cannot run or outrun two audits."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "concurrent"),
        agent_name="Concurrent",
        identity_method="did:pkh",
    )
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None
    audit_calls = 0

    async def slow_safe_auditor(_prompt):
        nonlocal audit_calls
        audit_calls += 1
        await asyncio.sleep(0.05)
        return {"risk_level": 1, "reasoning": "One serialized audit."}

    agent.get_audit_response = slow_safe_auditor
    agent._process_input_traced_locked = AsyncMock(return_value="PONG")
    try:
        responses = await asyncio.gather(
            agent.process_input("first"),
            agent.process_input("second"),
        )
        assert responses == ["PONG", "PONG"]
        assert audit_calls == 1
        assert agent._process_input_traced_locked.await_count == 2
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_streaming_first_turn_is_blocked_before_stream_setup(tmp_path):
    """The primary streaming chat path enforces the same pending receipt."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "streaming"),
        agent_name="Streaming",
        identity_method="did:pkh",
    )
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.get_audit_response = AsyncMock(
        return_value={
            "risk_level": 1,
            "reasoning": "No configured route.",
            "audited": False,
        }
    )
    try:
        chunks = [chunk async for chunk in agent.process_input_streaming("PONG")]
        assert len(chunks) == 1
        assert "GENESIS AUDIT PENDING" in chunks[0]
        agent.get_audit_response.assert_awaited_once()
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_deferred_risk_three_failure_is_durable_and_not_retried(tmp_path):
    """A lazy audit rejection blocks every later turn without overwrite."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "runtime-rejected"),
        agent_name="Runtime Rejected",
        identity_method="did:pkh",
    )
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None
    audit = AsyncMock(
        return_value={"risk_level": 3, "reasoning": "Unsafe runtime fixture."}
    )
    agent.get_audit_response = audit
    agent._process_input_traced_locked = AsyncMock(return_value="must-not-run")
    try:
        first = await agent.process_input("PONG")
        assert "GENESIS AUDIT FAILED" in first
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["genesis_audit"]["status"] == "failed"

        second = await agent.process_input("PONG")
        assert "GENESIS AUDIT FAILED" in second
        audit.assert_awaited_once()
        agent._process_input_traced_locked.assert_not_awaited()
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_unincepted_shape_cannot_bypass_genesis_readiness(tmp_path):
    """Missing birth metadata is never treated as an implicit test bypass."""
    credentials = await create_kestrel_identity_async(
        str(tmp_path / "legacy-shape"),
        agent_name="Legacy Shape",
        identity_method="did:pkh",
    )
    async with Storage(credentials.db_path, agent_id=credentials.agent_did) as storage:
        node = await storage.get_node(credentials.agent_did)
        node.properties.pop("genesis_audit")
        node.properties.pop("created_at", None)
        await storage.add_node(node)

    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None
    agent.get_audit_response = AsyncMock(
        return_value={"risk_level": 1, "reasoning": "Migrated safely."}
    )
    agent._process_input_traced_locked = AsyncMock(return_value="PONG")
    try:
        assert await agent.process_input("PONG") == "PONG"
        agent.get_audit_response.assert_awaited_once()
        node = await agent.storage.get_node(agent.agent_id)
        assert node.properties["genesis_audit"]["status"] == "passed"
        assert (
            node.properties["genesis_audit"]["provenance"]
            == "runtime:first_cognition"
        )
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_legacy_completed_receipt_migrates_without_rerunning_auditor(tmp_path):
    """A valid historical completion is upgraded, not silently overwritten."""

    async def original_auditor(_prompt):
        return {"risk_level": 1, "reasoning": "Historical safe result."}

    credentials = await create_kestrel_identity_async(
        str(tmp_path / "legacy-receipt"),
        agent_name="Legacy Receipt",
        identity_method="did:pkh",
        genesis_auditor=original_auditor,
        genesis_audit_provenance="test:legacy_seed",
    )
    async with Storage(credentials.db_path, agent_id=credentials.agent_did) as storage:
        node = await storage.get_node(credentials.agent_did)
        receipt = node.properties["genesis_audit"]
        for key in ("status", "completed_at", "provenance", "audited"):
            receipt.pop(key)
        node.properties["genesis_audit"] = receipt
        await storage.add_node(node)

    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=credentials.db_path,
        llm_service=LLMService(),
        privacy_mode=PrivacyMode.NORMAL,
    )
    await agent.initialize()
    agent.bootstrap_service = None
    agent.get_audit_response = AsyncMock(
        side_effect=AssertionError("legacy completed receipt must not rerun")
    )
    agent._process_input_traced_locked = AsyncMock(return_value="PONG")
    try:
        assert await agent.process_input("PONG") == "PONG"
        agent.get_audit_response.assert_not_awaited()
        node = await agent.storage.get_node(agent.agent_id)
        receipt = node.properties["genesis_audit"]
        assert receipt["status"] == "passed"
        assert receipt["audited"] is True
        assert receipt["provenance"] == "runtime:migrated_legacy_receipt"
    finally:
        await agent.shutdown()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest_asyncio.fixture
async def llm_service():
    """Initialize LLM service for testing with proper cleanup."""
    service = LLMService()
    yield service
    await service.close()


@pytest_asyncio.fixture
async def kestrel_agent(temp_db, llm_service):
    """Create a KestrelAgent with the new API."""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_genesis_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()
    yield agent
    await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_with_valid_constitution(kestrel_agent):
    """Test that genesis audit passes with a properly crafted constitution"""
    # Mock the audit to pass
    original_get_audit_response = kestrel_agent.get_audit_response

    async def mock_audit_pass(prompt):
        return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

    kestrel_agent.get_audit_response = mock_audit_pass

    try:
        # Perform genesis audit
        result = await kestrel_agent.perform_genesis_audit()
        assert result is True, "Genesis audit should pass with mocked response"
    finally:
        # Restore original method
        kestrel_agent.get_audit_response = original_get_audit_response

    print("✅ Genesis audit passed with mocked response")


@pytest.mark.asyncio
async def test_genesis_audit_with_malicious_constitution(temp_db, llm_service):
    """Test that genesis audit FAILS with a malicious constitution"""
    # Create agent
    agent = KestrelAgent(
        did="did:pkh:eip155:1:malicious_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock the audit to fail for malicious constitution
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_fail(prompt):
            return {"risk_level": 3, "reasoning": "High risk: constitution allows harm to users and violates ethical principles"}

        agent.get_audit_response = mock_audit_fail

        try:
            # Attempt genesis audit - should FAIL
            with pytest.raises(ValueError) as exc_info:
                await agent.perform_genesis_audit()

            # Verify error message mentions audit failure
            error_message = str(exc_info.value)
            assert "genesis audit" in error_message.lower(), "Error should mention genesis audit"
            assert "risk level" in error_message.lower() or "non-compliant" in error_message.lower()

            print(f"✅ Genesis audit correctly REJECTED malicious constitution")
            print(f"   Error: {error_message[:200]}")
        finally:
            # Restore audit method
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_with_missing_constitution(temp_db, llm_service):
    """Test that genesis audit FAILS when constitution cannot be loaded"""
    # Create agent
    agent = KestrelAgent(
        did="did:pkh:eip155:1:no_constitution_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # The audit reads the exact content-addressed bytes, not the augmented
        # runtime prompt representation.
        original_retrieve_file = agent.storage.retrieve_file
        agent.storage.retrieve_file = AsyncMock(return_value=None)

        # Attempt genesis audit - should FAIL
        with pytest.raises(ValueError) as exc_info:
            await agent.perform_genesis_audit()

        # Verify error message
        error_message = str(exc_info.value)
        assert "cannot load constitution" in error_message.lower()

        print(f"✅ Genesis audit correctly FAILED with missing constitution")
        print(f"   Error: {error_message[:200]}")

        agent.storage.retrieve_file = original_retrieve_file
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_stores_result_in_node(temp_db, llm_service):
    """Test that genesis audit result is properly stored in the agent's graph node"""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_storage_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock constitution loading to return a valid constitution
        original_get_constitution = agent._get_governing_constitution

        async def mock_get_constitution():
            return """
# Safe Constitution

This agent follows ethical AI principles:
1. Respect user privacy
2. Do not harm users
3. Be truthful and helpful
4. Allow user sovereignty
"""
        agent._get_governing_constitution = mock_get_constitution

        # Mock the audit to pass
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_pass(prompt):
            return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

        agent.get_audit_response = mock_audit_pass

        try:
            # Perform genesis audit
            await agent.perform_genesis_audit()

            # Retrieve agent node and verify audit data
            updated_node = await agent.storage.get_node(agent.did)
            assert updated_node is not None

            audit_data = updated_node.properties.get("genesis_audit")
            assert audit_data is not None, "Audit data should be stored in node properties"

            # Verify audit data structure
            assert "timestamp" in audit_data
            assert "risk_level" in audit_data
            assert "reasoning" in audit_data
            assert "constitution_hash" in audit_data

            assert isinstance(audit_data["risk_level"], int)
            assert 1 <= audit_data["risk_level"] <= 3

            print(f"✅ Genesis audit result properly stored in graph node")
            print(f"   Audit data: {audit_data}")
        finally:
            # Restore original methods
            agent._get_governing_constitution = original_get_constitution
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


@pytest.mark.asyncio
async def test_genesis_audit_logs_to_conversation_history(temp_db, llm_service):
    """Test that genesis audit result is logged to conversation history"""
    agent = KestrelAgent(
        did="did:pkh:eip155:1:test_logging_agent",
        storage_path=temp_db,
        llm_service=llm_service,
        privacy_mode=PrivacyMode.NORMAL
    )
    await agent.initialize()

    try:
        # Mock constitution loading to return a valid constitution
        original_get_constitution = agent._get_governing_constitution

        async def mock_get_constitution():
            return """
# Safe Constitution

This agent follows ethical AI principles:
1. Respect user privacy
2. Do not harm users
3. Be truthful and helpful
4. Allow user sovereignty
"""
        agent._get_governing_constitution = mock_get_constitution

        # Mock the audit to pass
        original_get_audit_response = agent.get_audit_response

        async def mock_audit_pass(prompt):
            return {"risk_level": 1, "reasoning": "Constitution meets all safety and ethical requirements"}

        agent.get_audit_response = mock_audit_pass

        try:
            # Perform genesis audit
            await agent.perform_genesis_audit()

            # Retrieve conversation history
            history = await agent.storage.get_conversation_history(limit=10)

            # Find genesis audit entry (history might be strings or dicts)
            genesis_entries = []
            for msg in history:
                if isinstance(msg, dict):
                    # Parse metadata if it's a JSON string
                    metadata = msg.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            import json
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    if metadata.get("event") == "genesis_audit":
                        genesis_entries.append(msg)
                elif isinstance(msg, str) and "genesis_audit" in msg:
                    genesis_entries.append(msg)

            assert len(genesis_entries) > 0, "Genesis audit should be logged to conversation history"

            genesis_entry = genesis_entries[0]
            assert genesis_entry["role"] == "system"
            assert "genesis audit passed" in genesis_entry["content"].lower()

            # Parse metadata if needed
            metadata = genesis_entry.get("metadata", {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata)
            assert "result" in metadata

            print(f"✅ Genesis audit logged to conversation history")
            print(f"   Entry: {genesis_entry['content'][:100]}")
        finally:
            # Restore original methods
            agent._get_governing_constitution = original_get_constitution
            agent.get_audit_response = original_get_audit_response
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
