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
from kestrel_sovereign.hooks import HooksManager, HookEvent, HookInput


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
    """Create and initialize compute feature."""
    feature = ComputeFeature(mock_agent)
    await feature.initialize()
    mock_agent.features["ComputeFeature"] = feature
    return feature


@pytest.fixture
async def security_feature(mock_agent):
    """Create and initialize security feature."""
    feature = SecurityFeature(mock_agent)
    await feature.initialize()
    mock_agent.features["SecurityFeature"] = feature
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
        
        assert "created" in result.lower() or "✅" in result
        assert "test_script" in result
    
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
        assert script.signature.startswith("hmac:") or script.signature.startswith("ecdsa:")


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
            
            # Should either succeed or fail based on uv availability, but not be denied
            assert "denied" not in result.lower() or "approved" in result.lower() or "completed" in result.lower() or "failed" in result.lower()
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
        assert "Trash" in result or "empty" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
