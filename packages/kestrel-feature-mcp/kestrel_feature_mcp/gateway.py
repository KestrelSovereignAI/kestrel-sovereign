"""
Docker MCP Gateway Manager.

This module manages the Docker MCP Gateway process, which provides
a unified SSE endpoint for accessing any MCP server from Docker's
311+ server catalog.

The gateway handles stdio→SSE conversion automatically, so we can
use any MCP server regardless of its native transport.

Usage:
    gateway = DockerMCPGateway(port=9000)
    auth_token = await gateway.start(["fetch", "sqlite"])
    # Connect via SSE at http://localhost:9000/sse with Bearer token
    await gateway.stop()
"""

import asyncio
import logging
import re
import shutil
import subprocess
import time
from typing import List, Optional, Set

from kestrel_sdk.config.constants import (
    SSH_COMMAND_TIMEOUT_SHORT,
    SSH_COMMAND_TIMEOUT_DEFAULT,
    HTTP_TIMEOUT_SHORT,
)

logger = logging.getLogger(__name__)

# Gateway startup configuration
GATEWAY_STARTUP_TIMEOUT = 30.0  # seconds
GATEWAY_STARTUP_POLL_INTERVAL = 0.5  # seconds


class DockerMCPGatewayError(Exception):
    """Error related to Docker MCP Gateway operations."""
    pass


class DockerMCPNotInstalledError(DockerMCPGatewayError):
    """Docker MCP Toolkit is not installed."""
    pass


class DockerMCPGateway:
    """
    Manages the Docker MCP Gateway process.

    The Docker MCP Gateway is a unified SSE endpoint that can expose
    any MCP server from Docker's catalog. It handles the stdio→SSE
    conversion automatically.

    Attributes:
        port: TCP port for the SSE endpoint
        process: The gateway subprocess
        auth_token: Bearer token for authentication
        enabled_servers: Set of currently enabled server names
    """

    def __init__(self, port: int = 9000, session_timeout: Optional[float] = None):
        """
        Initialize the gateway manager.

        Args:
            port: TCP port for the SSE endpoint (default: 9000)
            session_timeout: Optional timeout in seconds after which gateway
                           auto-stops. None = no timeout (default).
                           Recommended: 300-900 for agent sessions.
        """
        self.port = port
        self.session_timeout = session_timeout
        self.process: Optional[subprocess.Popen] = None
        self.auth_token: Optional[str] = None
        self.enabled_servers: Set[str] = set()
        self._output_lines: List[str] = []
        self._timeout_task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None

    @staticmethod
    def check_docker_mcp_available() -> bool:
        """
        Check if Docker MCP Toolkit is installed and available.

        Returns:
            True if docker mcp command is available
        """
        try:
            result = subprocess.run(
                ["docker", "mcp", "--help"],
                capture_output=True,
                timeout=SSH_COMMAND_TIMEOUT_SHORT
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_docker_mcp_version() -> Optional[str]:
        """
        Get the Docker MCP Toolkit version.

        Returns:
            Version string or None if not available
        """
        try:
            result = subprocess.run(
                ["docker", "mcp", "version"],
                capture_output=True,
                text=True,
                timeout=SSH_COMMAND_TIMEOUT_SHORT
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

    async def _enable_server(self, server_name: str) -> bool:
        """
        Enable a server in Docker MCP.

        Args:
            server_name: Name of the server to enable

        Returns:
            True if server was enabled successfully
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "mcp", "server", "enable", server_name],
                capture_output=True,
                text=True,
                timeout=SSH_COMMAND_TIMEOUT_DEFAULT
            )
            if result.returncode == 0:
                logger.info(f"Enabled MCP server: {server_name}")
                self.enabled_servers.add(server_name)
                return True
            else:
                logger.warning(f"Failed to enable {server_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout enabling server {server_name}")
            return False

    async def _disable_server(self, server_name: str) -> bool:
        """
        Disable a server in Docker MCP.

        Args:
            server_name: Name of the server to disable

        Returns:
            True if server was disabled successfully
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "mcp", "server", "disable", server_name],
                capture_output=True,
                text=True,
                timeout=SSH_COMMAND_TIMEOUT_DEFAULT
            )
            if result.returncode == 0:
                logger.info(f"Disabled MCP server: {server_name}")
                self.enabled_servers.discard(server_name)
                return True
            else:
                logger.warning(f"Failed to disable {server_name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout disabling server {server_name}")
            return False

    def _parse_auth_token(self, line: str) -> Optional[str]:
        """
        Parse the auth token from gateway output.

        The gateway outputs a line like:
        > Use Bearer token: Authorization: Bearer <token>

        Args:
            line: Output line from gateway

        Returns:
            The bearer token if found, None otherwise
        """
        # Pattern: "Authorization: Bearer <token>" where token is alphanumeric
        # The line format is: "> Use Bearer token: Authorization: Bearer <token>"
        # We need to match the LAST "Bearer" followed by an alphanumeric token (not "token:")
        match = re.search(r'Authorization:\s*Bearer\s+([a-zA-Z0-9]+)', line)
        if match:
            return match.group(1)
        return None

    async def start(self, servers: List[str]) -> str:
        """
        Start the Docker MCP Gateway with specified servers.

        Args:
            servers: List of server names to enable

        Returns:
            The bearer token for authentication

        Raises:
            DockerMCPNotInstalledError: If Docker MCP is not available
            DockerMCPGatewayError: If gateway fails to start
        """
        if not self.check_docker_mcp_available():
            raise DockerMCPNotInstalledError(
                "Docker MCP Toolkit is not installed. "
                "Please install Docker Desktop 29+ with MCP Toolkit enabled."
            )

        # Stop any existing gateway
        await self.stop()

        # Enable requested servers
        for server in servers:
            await self._enable_server(server)

        if not self.enabled_servers:
            raise DockerMCPGatewayError(
                f"Failed to enable any servers from: {servers}"
            )

        # Build gateway command
        servers_arg = ",".join(self.enabled_servers)
        cmd = [
            "docker", "mcp", "gateway", "run",
            "--transport=sse",
            f"--port={self.port}",
            f"--servers={servers_arg}"
        ]

        logger.info(f"Starting Docker MCP Gateway: {' '.join(cmd)}")

        # Start gateway process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Wait for gateway to be ready and capture auth token
        start_time = time.monotonic()
        self._output_lines = []

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= GATEWAY_STARTUP_TIMEOUT:
                await self.stop()
                raise DockerMCPGatewayError(
                    f"Gateway startup timeout after {GATEWAY_STARTUP_TIMEOUT}s. "
                    f"Output:\n{''.join(self._output_lines[-20:])}"
                )

            # Check if process died
            if self.process.poll() is not None:
                remaining = self.process.stdout.read()
                if remaining:
                    self._output_lines.append(remaining)
                raise DockerMCPGatewayError(
                    f"Gateway process exited with code {self.process.returncode}. "
                    f"Output:\n{''.join(self._output_lines)}"
                )

            # Read available output (non-blocking would be better, but this works)
            try:
                # Read one line with a short timeout
                line = await asyncio.wait_for(
                    asyncio.to_thread(self.process.stdout.readline),
                    timeout=GATEWAY_STARTUP_POLL_INTERVAL
                )
                if line:
                    self._output_lines.append(line)
                    logger.debug(f"Gateway: {line.rstrip()}")

                    # Check for auth token
                    token = self._parse_auth_token(line)
                    if token:
                        self.auth_token = token
                        self._start_time = time.monotonic()
                        logger.info(f"Gateway ready at http://localhost:{self.port}/sse")

                        # Start timeout task if configured
                        if self.session_timeout:
                            self._start_timeout_task()

                        return token

                    # Check for "Start sse server" message (gateway is ready)
                    if "Start sse server" in line or "Gateway URL:" in line:
                        # Keep reading for token
                        continue

            except asyncio.TimeoutError:
                # No output yet, continue waiting
                continue

    def _start_timeout_task(self):
        """Start a background task that stops the gateway after timeout."""
        async def timeout_handler():
            try:
                await asyncio.sleep(self.session_timeout)
                if self.is_running:
                    logger.warning(
                        f"Gateway session timeout ({self.session_timeout}s) - auto-stopping"
                    )
                    await self.stop()
            except asyncio.CancelledError:
                pass  # Normal cancellation when stop() is called

        self._timeout_task = asyncio.create_task(timeout_handler())
        logger.debug(f"Session timeout set: {self.session_timeout}s")

    def _cancel_timeout_task(self):
        """Cancel the timeout task."""
        if self._timeout_task:
            self._timeout_task.cancel()
            self._timeout_task = None

    def reset_timeout(self):
        """Reset the session timeout (call on activity to keep alive)."""
        if self.session_timeout and self.is_running:
            self._cancel_timeout_task()
            self._start_time = time.monotonic()  # Reset start time
            self._start_timeout_task()
            logger.debug("Session timeout reset")

    @property
    def time_remaining(self) -> Optional[float]:
        """Get remaining time before session timeout (None if no timeout)."""
        if not self.session_timeout or not self._start_time:
            return None
        elapsed = time.monotonic() - self._start_time
        remaining = self.session_timeout - elapsed
        return max(0, remaining)

    async def stop(self):
        """Stop the gateway process."""
        # Cancel timeout task first
        self._cancel_timeout_task()

        if self.process:
            logger.info("Stopping Docker MCP Gateway...")
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait),
                        timeout=HTTP_TIMEOUT_SHORT
                    )
                except asyncio.TimeoutError:
                    logger.warning("Gateway didn't terminate gracefully, killing...")
                    self.process.kill()
                    await asyncio.to_thread(self.process.wait)
            except Exception as e:
                logger.warning(f"Error stopping gateway: {e}")
            finally:
                self.process = None
                self.auth_token = None
                self._start_time = None
                logger.info("Gateway stopped")

    async def restart(self, servers: List[str] = None) -> str:
        """
        Restart the gateway with optionally different servers.

        Args:
            servers: List of servers, or None to use current servers

        Returns:
            New auth token
        """
        if servers is None:
            servers = list(self.enabled_servers)

        await self.stop()
        return await self.start(servers)

    async def add_server(self, server_name: str) -> str:
        """
        Add a server to the gateway (requires restart).

        Args:
            server_name: Name of server to add

        Returns:
            New auth token after restart
        """
        servers = list(self.enabled_servers)
        servers.append(server_name)
        return await self.restart(servers)

    async def remove_server(self, server_name: str) -> str:
        """
        Remove a server from the gateway (requires restart).

        Args:
            server_name: Name of server to remove

        Returns:
            New auth token after restart
        """
        servers = list(self.enabled_servers)
        if server_name in servers:
            servers.remove(server_name)
        if not servers:
            await self.stop()
            raise DockerMCPGatewayError("Cannot remove last server, stopping gateway")
        return await self.restart(servers)

    @property
    def sse_url(self) -> str:
        """Get the SSE endpoint URL."""
        return f"http://localhost:{self.port}/sse"

    @property
    def is_running(self) -> bool:
        """Check if the gateway is running."""
        return self.process is not None and self.process.poll() is None


async def list_available_servers() -> List[str]:
    """
    List all available servers from Docker MCP catalog.

    Returns:
        List of server names
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "mcp", "catalog", "show", "docker-mcp"],
            capture_output=True,
            text=True,
            timeout=SSH_COMMAND_TIMEOUT_DEFAULT
        )
        if result.returncode != 0:
            return []

        # Parse server names from output
        # Format is "  **ServerName**" on its own line
        servers = []
        for line in result.stdout.split('\n'):
            # Look for bold server names
            match = re.search(r'\*\*([a-zA-Z0-9_-]+)\*\*', line)
            if match:
                servers.append(match.group(1))

        return servers
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


async def list_enabled_servers() -> List[str]:
    """
    List currently enabled servers in Docker MCP.

    Returns:
        List of enabled server names
    """
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "mcp", "server", "ls"],
            capture_output=True,
            text=True,
            timeout=SSH_COMMAND_TIMEOUT_DEFAULT
        )
        if result.returncode != 0:
            return []

        # Parse enabled servers from table output
        # Format: NAME  OAUTH  SECRETS  CONFIG  DESCRIPTION
        servers = []
        in_table = False
        for line in result.stdout.split('\n'):
            if 'NAME' in line and 'DESCRIPTION' in line:
                in_table = True
                continue
            if in_table and line.strip() and not line.startswith('-'):
                # First column is the server name
                parts = line.split()
                if parts:
                    servers.append(parts[0])

        return servers
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
