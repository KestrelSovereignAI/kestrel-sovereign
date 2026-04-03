"""
Unit tests for MCP container healthcheck functionality.

Tests the exponential backoff healthcheck without requiring Docker.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

docker_errors = pytest.importorskip("docker.errors")
NotFound = docker_errors.NotFound


class TestMCPHealthcheck:
    """Tests for container healthcheck with exponential backoff."""
    
    @pytest.fixture
    def mock_docker_client(self):
        """Create a mock Docker client."""
        client = MagicMock()
        client.ping.return_value = True
        client.containers = MagicMock()
        return client
    
    @pytest.fixture
    def manager(self, mock_docker_client):
        """Create MCPToolManager with mocked Docker."""
        with patch('kestrel_feature_mcp.manager.docker') as mock_docker:
            mock_docker.from_env.return_value = mock_docker_client
            from kestrel_feature_mcp.manager import MCPToolManager
            return MCPToolManager()
    
    @pytest.mark.asyncio
    async def test_check_port_ready_success(self, manager):
        """Test port check succeeds when connection works."""
        with patch('asyncio.open_connection') as mock_conn:
            mock_writer = MagicMock()
            mock_writer.close = MagicMock()
            mock_writer.wait_closed = AsyncMock()
            mock_conn.return_value = (AsyncMock(), mock_writer)

            result = await manager._check_port_ready('localhost', 8080)

            assert result is True
            mock_writer.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_port_ready_connection_refused(self, manager):
        """Test port check returns False when connection refused."""
        with patch('asyncio.open_connection') as mock_conn:
            mock_conn.side_effect = ConnectionRefusedError()
            
            result = await manager._check_port_ready('localhost', 8080)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_port_ready_timeout(self, manager):
        """Test port check returns False on timeout."""
        with patch('asyncio.open_connection') as mock_conn:
            mock_conn.side_effect = asyncio.TimeoutError()
            
            result = await manager._check_port_ready('localhost', 8080)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_port_ready_os_error(self, manager):
        """Test port check returns False on OS error."""
        with patch('asyncio.open_connection') as mock_conn:
            mock_conn.side_effect = OSError("Network unreachable")
            
            result = await manager._check_port_ready('localhost', 8080)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_immediate_success(self, manager, mock_docker_client):
        """Test container becomes ready immediately."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.ports = {'8000/tcp': [{'HostPort': '32768'}]}
        mock_docker_client.containers.get.return_value = mock_container
        
        with patch.object(manager, '_check_port_ready', new_callable=AsyncMock) as mock_port_check:
            with patch.object(manager, '_check_sse_ready', new_callable=AsyncMock) as mock_sse_check:
                mock_port_check.return_value = True
                mock_sse_check.return_value = True
                
                result = await manager._wait_for_container_ready('test-container')
                
                assert result is True
                mock_port_check.assert_called_once_with('localhost', 32768)
                mock_sse_check.assert_called_once_with('localhost', 32768)
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_with_retries(self, manager, mock_docker_client):
        """Test container becomes ready after a few retries."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.ports = {'8000/tcp': [{'HostPort': '32768'}]}
        mock_docker_client.containers.get.return_value = mock_container
        
        # Port check always succeeds, SSE fails twice then succeeds
        call_count = 0
        async def sse_check_side_effect(host, port):
            nonlocal call_count
            call_count += 1
            return call_count >= 3
        
        with patch.object(manager, '_check_port_ready', new_callable=AsyncMock) as mock_port:
            with patch.object(manager, '_check_sse_ready', side_effect=sse_check_side_effect):
                mock_port.return_value = True
                result = await manager._wait_for_container_ready('test-container', timeout=10.0)
                
                assert result is True
                assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_timeout(self, manager, mock_docker_client):
        """Test timeout when container never becomes ready."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        mock_container.ports = {'8000/tcp': [{'HostPort': '32768'}]}
        mock_docker_client.containers.get.return_value = mock_container
        
        with patch.object(manager, '_check_port_ready', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False  # Never ready
            
            with pytest.raises(TimeoutError) as exc_info:
                await manager._wait_for_container_ready('test-container', timeout=1.0)
            
            assert 'did not become ready' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_container_exited(self, manager, mock_docker_client):
        """Test error when container exits unexpectedly."""
        mock_container = MagicMock()
        mock_container.status = 'exited'
        mock_container.logs.return_value = b'Error: something went wrong'
        mock_docker_client.containers.get.return_value = mock_container
        
        with pytest.raises(RuntimeError) as exc_info:
            await manager._wait_for_container_ready('test-container')
        
        assert 'exited unexpectedly' in str(exc_info.value)
        assert 'something went wrong' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_container_not_found(self, manager, mock_docker_client):
        """Test error when container doesn't exist."""
        mock_docker_client.containers.get.side_effect = NotFound("Container not found")
        
        with pytest.raises(RuntimeError) as exc_info:
            await manager._wait_for_container_ready('test-container')
        
        assert 'not found' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_port_not_mapped(self, manager, mock_docker_client):
        """Test retries when port is not mapped yet."""
        mock_container = MagicMock()
        mock_container.status = 'running'
        
        # Port not mapped initially, then becomes available
        call_count = 0
        def get_ports():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {}  # No ports yet
            return {'8000/tcp': [{'HostPort': '32768'}]}
        
        type(mock_container).ports = property(lambda self: get_ports())
        mock_docker_client.containers.get.return_value = mock_container
        
        with patch.object(manager, '_check_port_ready', new_callable=AsyncMock) as mock_port:
            with patch.object(manager, '_check_sse_ready', new_callable=AsyncMock) as mock_sse:
                mock_port.return_value = True
                mock_sse.return_value = True
                
                result = await manager._wait_for_container_ready('test-container', timeout=10.0)
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_wait_for_container_ready_status_changes(self, manager, mock_docker_client):
        """Test waits for container to enter running state."""
        mock_container = MagicMock()
        mock_container.ports = {'8000/tcp': [{'HostPort': '32768'}]}
        
        # Status changes from 'created' to 'running'
        statuses = ['created', 'created', 'running']
        call_count = 0
        def get_status():
            nonlocal call_count
            result = statuses[min(call_count, len(statuses) - 1)]
            call_count += 1
            return result
        
        type(mock_container).status = property(lambda self: get_status())
        mock_docker_client.containers.get.return_value = mock_container
        
        with patch.object(manager, '_check_port_ready', new_callable=AsyncMock) as mock_port:
            with patch.object(manager, '_check_sse_ready', new_callable=AsyncMock) as mock_sse:
                mock_port.return_value = True
                mock_sse.return_value = True
                
                result = await manager._wait_for_container_ready('test-container', timeout=10.0)
                
                assert result is True


class TestMCPHealthcheckConstants:
    """Test healthcheck configuration constants."""
    
    def test_constants_defined(self):
        """Verify healthcheck constants are properly defined."""
        from kestrel_feature_mcp.manager import (
            HEALTHCHECK_INITIAL_DELAY,
            HEALTHCHECK_MAX_DELAY,
            HEALTHCHECK_TIMEOUT,
            HEALTHCHECK_BACKOFF_FACTOR
        )
        
        # Verify reasonable values
        assert 0 < HEALTHCHECK_INITIAL_DELAY < 5
        assert HEALTHCHECK_INITIAL_DELAY < HEALTHCHECK_MAX_DELAY
        assert HEALTHCHECK_MAX_DELAY < HEALTHCHECK_TIMEOUT
        assert HEALTHCHECK_BACKOFF_FACTOR >= 1.5
    
    def test_exponential_backoff_sequence(self):
        """Verify exponential backoff produces expected delays."""
        from kestrel_feature_mcp.manager import (
            HEALTHCHECK_INITIAL_DELAY,
            HEALTHCHECK_MAX_DELAY,
            HEALTHCHECK_BACKOFF_FACTOR
        )
        
        delay = HEALTHCHECK_INITIAL_DELAY
        delays = [delay]
        
        for _ in range(10):
            delay = min(delay * HEALTHCHECK_BACKOFF_FACTOR, HEALTHCHECK_MAX_DELAY)
            delays.append(delay)
        
        # Delays should increase
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]
        
        # Should cap at max delay
        assert delays[-1] == HEALTHCHECK_MAX_DELAY
