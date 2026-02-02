import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, timezone
from kestrel_sovereign.features.runpod.manager import RunPodManager
from kestrel_sovereign.features.runpod.providers import DirectRunPodProvider
from kestrel_sovereign.features.runpod.models import RunPodManagerError, PodStatus, RunPodSession

@pytest.fixture
def mock_runpod():
    with patch("kestrel_sovereign.features.runpod.providers.runpod") as mock:
        yield mock

@pytest.fixture
def mock_paramiko():
    with patch("kestrel_sovereign.features.runpod.providers.paramiko") as mock:
        yield mock

@pytest.fixture
def mock_utils():
    with patch("kestrel_sovereign.features.runpod.providers.get_pod_ssh_ip_port") as mock_get_ip, \
         patch("kestrel_sovereign.features.runpod.providers.find_ssh_key_file") as mock_find_key:
        yield mock_get_ip, mock_find_key

class TestRunPodLogs:
    def test_exec_command_success(self, mock_runpod, mock_paramiko, mock_utils):
        mock_get_ip, mock_find_key = mock_utils
        
        # Setup mocks
        mock_runpod.get_pod.return_value = {"id": "pod-123"}
        mock_get_ip.return_value = ("1.2.3.4", 2222)
        mock_find_key.return_value = "/path/to/key"
        
        mock_ssh = MagicMock()
        mock_paramiko.SSHClient.return_value = mock_ssh
        
        # Mock stdout/stderr
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b"command output"
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b""
        mock_ssh.exec_command.return_value = (None, mock_stdout, mock_stderr)
        
        provider = DirectRunPodProvider(api_key="test-key")
        output = provider.exec_command("pod-123", "echo hello")
        
        # Verify
        mock_runpod.get_pod.assert_called_with("pod-123")
        mock_get_ip.assert_called()
        mock_find_key.assert_called()
        mock_ssh.connect.assert_called_with("1.2.3.4", port=2222, username="root", key_filename="/path/to/key")
        mock_ssh.exec_command.assert_called_with("echo hello")
        assert output == "command output"

    @pytest.mark.asyncio
    async def test_get_logs_success(self):
        # Mock the provider
        mock_provider = MagicMock(spec=DirectRunPodProvider)
        mock_provider.exec_command.return_value = "log line 1\nlog line 2"
        
        with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
            manager = RunPodManager()
        
        # Setup active session
        session = RunPodSession(
            pod_id="pod-123",
            task_profile="llm",
            model_name="test-model",
            status=PodStatus.READY,
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            profile=MagicMock(),
            ttl_seconds=3600,
            pod_type="community"
        )
        # Manually set the private _session attribute since we are testing internals
        manager._session = session
        
        logs = await manager.get_logs(tail=50)
        
        # Verify
        expected_command = "docker logs --tail 50 $(docker ps -q | head -n 1)"
        # Since exec_command is run in a thread, we check if it was called
        # Note: asyncio.to_thread calls the function. 
        # Since we mocked the provider instance method, we can check the call.
        mock_provider.exec_command.assert_called_with("pod-123", expected_command)
        assert logs == "log line 1\nlog line 2"

    @pytest.mark.asyncio
    async def test_get_logs_no_session(self):
        mock_provider = MagicMock(spec=DirectRunPodProvider)
        with patch.object(RunPodManager, "_build_provider", return_value=mock_provider):
            manager = RunPodManager()
        manager._session = None
        
        with pytest.raises(RunPodManagerError, match="No active session"):
            await manager.get_logs()
