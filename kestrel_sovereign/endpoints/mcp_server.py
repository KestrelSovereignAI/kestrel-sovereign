"""Private first-party consent routes; the optional MCP SDK is loaded on opt-in."""

from fastapi import APIRouter, HTTPException, Request

from .agent_helpers import get_agent


def _controller(request):
    agent = get_agent(request)
    publication = vars(agent).get("_native_mcp_publication")
    if publication is None:
        raise HTTPException(status_code=404, detail="mcp_server_unavailable")
    from kestrel_sovereign.mcp_server.publication import NativeMCPPublication

    if not isinstance(publication, NativeMCPPublication) or publication.agent is not agent:
        raise TypeError("Original native MCP publication required")
    return publication.controller


def create_consent_router(resolve_controller):
    """One route contract for the published host and native component tests."""
    router = APIRouter(prefix="/api/mcp/server/consent", tags=["mcp-server-consent"])

    @router.get("/requests/{handle}")
    async def describe(handle: str, request: Request):
        return resolve_controller(request).describe_response(handle, request)

    @router.post("/requests/{handle}/decision")
    async def decide(handle: str, request: Request):
        return resolve_controller(request).decision_response(handle, request)

    return router


router = create_consent_router(_controller)
