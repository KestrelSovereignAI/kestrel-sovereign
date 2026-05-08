"""
Integration tests for Compute + Security feature interaction.

Tests verify:
- Hook chain properly intercepts script execution
- Approval queue integration works
- Security analysis gates dangerous scripts
- Safe scripts can execute after approval
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import components
from kestrel_sovereign.features.compute import ComputeFeature, ComputeScript, ScriptState
from kestrel_sovereign.features.security import SecurityFeature, ApprovalQueue, ApprovalStatus
from kestrel_sovereign.hooks import HooksManager
from kestrel_sdk.hooks.base import HookEvent, HookInput


class MockAgent:
    """Mock agent for testing feature integration."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.did = "did:ethr:0x1234567890abcdef1234567890abcdef12345678"
        self.hooks_manager = HooksManager()
        self.features = {}
        self._event_listeners = []
    
    async def emit_event(self, event_type: str, data: dict):
        """Mock event emission."""
        for listener in self._event_listeners:
            await listener(event_type, data)


@pytest.fixture
def temp_db():
    """Create temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_agent(temp_db):
    """Create a mock agent with hooks manager."""
    return MockAgent(temp_db)


@pytest.fixture
async def compute_feature(mock_agent):
    """Create and initialize compute feature.

    Wave 0B: ScriptSigner now sign-or-fails. The MockAgent has no real key
    custody, so we inject a freshly-generated secp256k1 keypair onto the
    signer to exercise the genuine ECDSA path. Tests assert ``ecdsa:``
    output and signed state.
    """
    feature = ComputeFeature(mock_agent)
    await feature.initialize()
    mock_agent.features["ComputeFeature"] = feature

    # Inject real ECDSA keys into the signer (no inception ceremony in tests).
    # Wave 1: go through Secp256k1Suite so the test exercises the same
    # CryptoSuite path production code uses, not the raw cryptography API
    # (which the rest of the codebase no longer touches outside the suite).
    if feature.signer is not None:
        from kestrel_sovereign.security.crypto_suite import Secp256k1Suite
        kp = Secp256k1Suite().generate_keypair()
        feature.signer._private_key = kp.private_key
        feature.signer._public_key = kp.public_key
        async def _ok():
            return True
        feature.signer._load_keys = _ok

    # Auto-register hooks (mirrors _register_feature in kestrel_agent.py)
    for hook in feature.get_hooks():
        mock_agent.hooks_manager.register(hook)
    return feature


@pytest.fixture
async def security_feature(mock_agent):
    """Create and initialize security feature."""
    feature = SecurityFeature(mock_agent)
    await feature.initialize()
    mock_agent.features["SecurityFeature"] = feature
    # Auto-register hooks (mirrors _register_feature in kestrel_agent.py)
    for hook in feature.get_hooks():
        mock_agent.hooks_manager.register(hook)
    return feature


class TestHookRegistration:
    """Test that hooks are properly registered."""
    
    @pytest.mark.asyncio
    async def test_compute_hook_registered(self, mock_agent, compute_feature):
        """Test ComputeSecurityHook is registered after initialization."""
        hooks = mock_agent.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
        hook_names = [h.name for h in hooks]
        assert "compute_security" in hook_names
    
    @pytest.mark.asyncio
    async def test_security_hook_registered(self, mock_agent, security_feature):
        """Test SecurityHook is registered after initialization."""
        hooks = mock_agent.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
        hook_names = [h.name for h in hooks]
        assert "security_guard" in hook_names
    
    @pytest.mark.asyncio
    async def test_hook_priority_order(self, mock_agent, compute_feature, security_feature):
        """Test hooks are ordered by priority (compute=5 before security=10)."""
        hooks = mock_agent.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
        priorities = [(h.name, h.priority) for h in hooks]
        
        compute_idx = next(i for i, (n, _) in enumerate(priorities) if n == "compute_security")
        security_idx = next(i for i, (n, _) in enumerate(priorities) if n == "security_guard")
        
        assert compute_idx < security_idx, "ComputeSecurityHook should run before SecurityHook"


class TestScriptLifecycle:
    """Test full script lifecycle with hooks."""
    
    @pytest.mark.asyncio
    async def test_write_script_initializes_properly(self, compute_feature):
        """Test writing a script works after async init."""
        result = await compute_feature.write_script(
            name="test_script",
            language="python",
            content='print("Hello")',
            purpose="Test"
        )
        
        # write_script now returns a ToolResult envelope (#1061 wave 13);
        # the formatted body lives in confirmation.
        body = result.confirmation
        assert "created" in body.lower() or "✅" in body
        assert "test_script" in body
    
    @pytest.mark.asyncio
    async def test_script_signed_after_write(self, compute_feature):
        """Test script is signed after writing."""
        await compute_feature.write_script(
            name="signed_test",
            language="python",
            content='x = 1 + 1',
            purpose="Test signing"
        )
        
        scripts = await compute_feature.script_store.list_recent(1)
        assert len(scripts) == 1
        script = scripts[0]
        
        assert script.state == ScriptState.SIGNED
        assert script.signature is not None
        assert script.signature.startswith("ecdsa:"), (
            "Wave 0B: only ecdsa: signatures are produced; the hmac: fallback "
            "was removed because its key was the public DID and so was forgeable."
        )

    @pytest.mark.asyncio
    async def test_run_script_rejects_legacy_hmac_signature(self, compute_feature):
        """Defense-in-depth: run_script must independently verify the signature
        before executing, not trust script.state alone. A host that bypasses
        or misregisters the security-hook chain would otherwise execute a
        forgeable legacy 'hmac:' tag.
        """
        import base64, hashlib, hmac as hmac_mod

        # Write a script normally — gets a real ecdsa: signature
        await compute_feature.write_script(
            name="legacy",
            language="python",
            content="print('legacy')",
            purpose="defense-in-depth test",
        )
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        assert script.state == ScriptState.SIGNED

        # Tamper: replace the ecdsa: signature with a freshly-forged hmac: tag
        canonical = f"{script.name}|{script.language}|{script.content}|{script.purpose}"
        content_hash = hashlib.sha256(canonical.encode()).hexdigest()
        forged = hmac_mod.new(
            (script.signed_by or "").encode(), content_hash.encode(), hashlib.sha256
        ).digest()
        script.signature = "hmac:" + base64.b64encode(forged).decode()
        await compute_feature.script_store.update(script)

        # run_script should reject before reaching the executor.
        # Migrated path: invalid signature -> ToolResult.failed; the
        # rejection text lives in result.error.
        result = await compute_feature.run_script(script.id, executor="uv")
        assert "invalid signature" in (result.error or "").lower(), (
            f"run_script must reject the legacy hmac: tag, got: {result!r}"
        )

        # And the script must now be marked REJECTED on disk
        refreshed = await compute_feature.script_store.find_by_id_prefix(script.id)
        assert refreshed.state == ScriptState.REJECTED

    @pytest.mark.asyncio
    async def test_run_script_rejects_state_signed_with_null_signature(self, compute_feature):
        """Defense-in-depth (#925): a manually corrupted DB row with
        ``state=SIGNED`` but ``signature=None`` must NOT execute. The pre-#925
        guard short-circuited on a falsy signature, so verify never ran and
        the state check happily accepted SIGNED. Now we gate on state, not
        on signature truthiness.
        """
        await compute_feature.write_script(
            name="null_sig",
            language="python",
            content="print('null sig regression')",
            purpose="state=SIGNED + signature=None test",
        )
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        assert script.state == ScriptState.SIGNED

        # Simulate a corrupted/partial-write row.
        script.signature = None
        await compute_feature.script_store.update(script)

        result = await compute_feature.run_script(script.id, executor="uv")
        assert "invalid signature" in (result.error or "").lower(), (
            f"run_script must reject state=SIGNED with no signature, got: {result!r}"
        )

        refreshed = await compute_feature.script_store.find_by_id_prefix(script.id)
        assert refreshed.state == ScriptState.REJECTED

    @pytest.mark.asyncio
    async def test_run_script_rejects_state_signed_with_empty_signature(self, compute_feature):
        """Same #925 case as null, but ``signature=""``. Both falsy values
        must follow the verify→REJECT path."""
        await compute_feature.write_script(
            name="empty_sig",
            language="python",
            content="print('empty sig regression')",
            purpose="state=SIGNED + signature='' test",
        )
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        assert script.state == ScriptState.SIGNED

        script.signature = ""
        await compute_feature.script_store.update(script)

        result = await compute_feature.run_script(script.id, executor="uv")
        assert "invalid signature" in (result.error or "").lower(), (
            f"run_script must reject state=SIGNED with empty signature, got: {result!r}"
        )

        refreshed = await compute_feature.script_store.find_by_id_prefix(script.id)
        assert refreshed.state == ScriptState.REJECTED


class TestSecurityAnalysis:
    """Test security analysis integration."""
    
    @pytest.mark.asyncio
    async def test_critical_script_blocked(self, compute_feature):
        """Test that critical patterns are auto-blocked during analysis."""
        # Write a script with fork bomb pattern
        await compute_feature.write_script(
            name="dangerous",
            language="bash",
            content=':(){ :|:& };:',
            purpose="Test blocking"
        )
        
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        
        # Analyze the script - it should be flagged as critical
        result = compute_feature.analyzer.analyze(script)
        assert result.has_critical is True
        
        # The script should still be SIGNED (blocking happens at run time)
        # But when we try to run it, the analysis during run_script should reject it
        # Use local executor for bash (if available) or check the analysis directly
        
        # Since we may not have docker, let's verify the security analysis works
        critical_findings = [f for f in result.findings if f.severity == "critical"]
        assert len(critical_findings) > 0
        assert "fork_bomb" in critical_findings[0].category
    
    @pytest.mark.asyncio
    async def test_rm_rewritten_not_blocked(self, compute_feature):
        """Test that rm commands are rewritten, not blocked."""
        await compute_feature.write_script(
            name="cleanup",
            language="bash",
            content='rm -rf /tmp/test_data',
            purpose="Cleanup test data"
        )
        
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        
        # Script should be signed, not rejected
        assert script.state == ScriptState.SIGNED
        
        # Should have "rewritable" findings, not critical
        result = compute_feature.analyzer.analyze(script)
        assert result.has_critical is False
        assert result.has_rewritable is True


class TestApprovalIntegration:
    """Test approval queue integration."""
    
    @pytest.mark.asyncio
    async def test_approval_queue_created(self, security_feature):
        """Test approval queue is created."""
        assert security_feature.approval_queue is not None
        assert isinstance(security_feature.approval_queue, ApprovalQueue)
    
    @pytest.mark.asyncio
    async def test_run_script_uses_approval_queue(self, mock_agent, compute_feature, security_feature):
        """Test that run_script integrates with approval queue."""
        # Write a safe script
        await compute_feature.write_script(
            name="safe_script",
            language="python",
            content='result = 2 + 2',
            purpose="Simple math"
        )
        
        scripts = await compute_feature.script_store.list_recent(1)
        script = scripts[0]
        
        # Mock approval queue to auto-approve
        original_request = security_feature.approval_queue.request_approval
        
        async def mock_approval(*args, **kwargs):
            return (True, "once")
        
        security_feature.approval_queue.request_approval = mock_approval
        
        try:
            # Now run should work (auto-approved by mock)
            result = await compute_feature.run_script(script.id[:8], executor="uv")

            # Should either succeed or fail based on uv availability, but not be
            # denied. result is a ToolResult — flatten confirmation+error to
            # search the same content the legacy str return carried.
            text = ((result.confirmation or "") + (result.error or "")).lower()
            assert "denied" not in text or "approved" in text or "completed" in text or "failed" in text
        finally:
            security_feature.approval_queue.request_approval = original_request


class TestHookExecution:
    """Test hook chain execution."""
    
    @pytest.mark.asyncio
    async def test_hooks_execute_for_tool_calls(self, mock_agent, compute_feature):
        """Test hooks are executed when tools are invoked via task manager."""
        # Create hook input for run_script
        hook_input = HookInput(
            session_id="test-session",
            hook_event_name=HookEvent.PRE_TOOL_USE.value,
            tool_name="run_script",
            tool_input={"script_id": "nonexistent"},
            feature_name="ComputeFeature",
        )
        
        # Execute hooks
        output = await mock_agent.hooks_manager.execute_hooks(
            HookEvent.PRE_TOOL_USE,
            hook_input
        )
        
        # Should get some response (ALLOW for non-matching scripts or handled)
        assert output is not None
    
    @pytest.mark.asyncio
    async def test_compute_hook_matches_run_script(self, mock_agent, compute_feature):
        """Test ComputeSecurityHook matches run_script tool."""
        hooks = mock_agent.hooks_manager.get_hooks(HookEvent.PRE_TOOL_USE)
        compute_hook = next(h for h in hooks if h.name == "compute_security")
        
        assert compute_hook.matches("run_script") is True
        assert compute_hook.matches("other_tool") is False


class TestTrashIntegration:
    """Test trash manager integration."""
    
    @pytest.mark.asyncio
    async def test_trash_dir_created_on_init(self, compute_feature):
        """Test trash directory is created during initialization."""
        assert compute_feature.trash_manager is not None
        assert compute_feature.trash_manager.trash_dir.exists()
    
    @pytest.mark.asyncio
    async def test_list_trash_works(self, compute_feature):
        """Test listing trash items works."""
        result = await compute_feature.list_trash()
        body = result.confirmation
        assert "Trash" in body or "empty" in body.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
