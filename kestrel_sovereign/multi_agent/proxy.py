"""
Kestrel Host Proxy - Routes requests to the correct agent process.

The proxy is a thin layer that forwards HTTP requests from the host
to individual agent processes running on their own ports. It handles:
- Path rewriting: /api/agents/{agent_id}/foo → /foo on agent port
- Header forwarding (except Host)
- Streaming/SSE response passthrough
- Agent offline detection (503)
"""

import logging
from typing import Union

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse, Response

from kestrel_sovereign.multi_agent.config import (
    LocalAgentConfig,
    RemoteAgentConfig,
    MultiAgentConfig,
)

logger = logging.getLogger(__name__)

# Timeout for proxied requests (seconds)
PROXY_CONNECT_TIMEOUT = 5.0
PROXY_READ_TIMEOUT = 300.0  # Long timeout for streaming/LLM responses


def get_agent_base_url(
    agent_config: Union[LocalAgentConfig, RemoteAgentConfig],
) -> str:
    """Get the base URL for an agent.

    Args:
        agent_config: The agent's configuration.

    Returns:
        Base URL string (e.g. "http://localhost:8801").
    """
    if isinstance(agent_config, RemoteAgentConfig):
        return agent_config.url.rstrip("/")
    return f"http://localhost:{agent_config.port}"


def resolve_agent(
    agent_id: str, config: MultiAgentConfig
) -> Union[LocalAgentConfig, RemoteAgentConfig, None]:
    """Resolve an agent ID (alias or name) to its config.

    Args:
        agent_id: Agent name/alias from the URL path.
        config: The multi_agent configuration.

    Returns:
        Agent config if found, None otherwise.
    """
    return config.agents.get(agent_id)


def build_proxy_headers(request: Request) -> dict[str, str]:
    """Build headers to forward to the agent, excluding hop-by-hop headers.

    Args:
        request: The incoming request.

    Returns:
        Dictionary of headers to forward.
    """
    excluded = {"host", "transfer-encoding", "connection"}
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in excluded
    }


async def proxy_request_streaming(
    request: Request,
    agent_id: str,
    path: str,
    config: MultiAgentConfig,
    client: httpx.AsyncClient,
) -> Response:
    """Proxy a request to the target agent with streaming support.

    Auto-detects SSE responses by Content-Type and streams them.
    Non-streaming responses are returned as regular responses.

    Args:
        request: The incoming FastAPI request.
        agent_id: The agent name/alias.
        path: The path to forward.
        config: The multi_agent configuration.
        client: Shared httpx async client.

    Returns:
        StreamingResponse for SSE, regular Response otherwise.
    """
    agent_config = resolve_agent(agent_id, config)
    if agent_config is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Agent '{agent_id}' not found in multi_agent config"},
        )

    base_url = get_agent_base_url(agent_config)
    target_url = f"{base_url}/{path.lstrip('/')}"

    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    headers = build_proxy_headers(request)
    body = await request.body()

    timeout = httpx.Timeout(
        connect=PROXY_CONNECT_TIMEOUT,
        read=PROXY_READ_TIMEOUT,
        write=PROXY_READ_TIMEOUT,
        pool=PROXY_CONNECT_TIMEOUT,
    )

    try:
        req = client.build_request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body if body else None,
            timeout=timeout,
        )
        proxy_resp = await client.send(req, stream=True)
    except httpx.ConnectError:
        logger.warning(f"Agent '{agent_id}' is offline at {base_url}")
        return JSONResponse(
            status_code=503,
            content={
                "detail": f"Agent '{agent_id}' is offline",
                "agent_id": agent_id,
                "target_url": base_url,
            },
        )
    except httpx.TimeoutException:
        logger.warning(f"Timeout proxying to agent '{agent_id}' at {target_url}")
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"Timeout connecting to agent '{agent_id}'",
                "agent_id": agent_id,
            },
        )

    content_type = proxy_resp.headers.get("content-type", "")
    is_streaming = "text/event-stream" in content_type

    if is_streaming:
        response_headers = {
            key: value
            for key, value in proxy_resp.headers.items()
            if key.lower()
            not in {"transfer-encoding", "connection", "content-length"}
        }

        async def stream_generator():
            try:
                async for chunk in proxy_resp.aiter_bytes():
                    yield chunk
            finally:
                await proxy_resp.aclose()

        return StreamingResponse(
            stream_generator(),
            status_code=proxy_resp.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: read full response and close
        content = await proxy_resp.aread()
        await proxy_resp.aclose()

        response_headers = {
            key: value
            for key, value in proxy_resp.headers.items()
            if key.lower()
            not in {"transfer-encoding", "connection", "content-length"}
        }

        return Response(
            content=content,
            status_code=proxy_resp.status_code,
            headers=response_headers,
            media_type=proxy_resp.headers.get("content-type"),
        )
