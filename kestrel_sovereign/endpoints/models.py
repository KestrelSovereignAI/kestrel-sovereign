"""Model, wallet, and IPFS status endpoints."""
from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import aiohttp
import httpx
import re
import time
import logging

from kestrel_sovereign.kestrel_config.defaults import get_ipfs_api_url
from kestrel_sovereign.llm.model_metadata import ModelCategory
from kestrel_sovereign.sql_utils import safe_column_name
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.features.bootstrap.feature import rename_agent_core
from kestrel_sovereign.endpoints.agent_helpers import get_agent, get_caller

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Validation: agent names must be alphanumeric + hyphens/underscores, 1-64 chars
_AGENT_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


class CreateAgentRequest(BaseModel):
    """Request body for creating a new agent."""
    name: str = Field(..., description="Agent name (alphanumeric, hyphens, underscores)", min_length=1, max_length=64)


@router.get("/api/agents")
async def get_agents(request: Request):
    """Get list of agents (A2A agent cards).

    In multi-agent mode, returns all agents with mode: "multi_agent".
    In single-agent mode, returns one agent with mode: "standalone".

    Each agent surfaces ``is_demo`` (from the inception service) and the
    response carries a top-level ``server_demo_mode`` flag so the page
    can render a DEMO MODE banner — the browser-side defence in #868.
    """
    server_demo_mode = bool(getattr(request.app.state, "demo_mode", False))

    def _is_demo(a) -> bool:
        return getattr(a, "is_demo", False) is True

    # Multi-agent mode: return all agents from AgentManager
    agent_manager = getattr(request.app.state, 'agent_manager', None)
    if agent_manager:
        agents_list = []
        for name, agent in agent_manager.list_agents().items():
            try:
                agent_card = await agent.get_agent_card()
                card_dict = agent_card.model_dump()
                card_dict["id"] = agent.agent_id
                card_dict["name"] = name
                card_dict["status"] = "online"
                card_dict["is_demo"] = _is_demo(agent)
                agents_list.append(card_dict)
            except Exception as e:
                logger.warning(f"Error getting agent card for '{name}': {e}")
                agents_list.append({
                    "id": agent.agent_id,
                    "name": name,
                    "status": "error",
                    "is_demo": _is_demo(agent),
                })
        return {
            "agents": agents_list,
            "mode": "multi_agent",
            "server_demo_mode": server_demo_mode,
        }

    # Single-agent mode
    try:
        agent = get_agent(request)
        agent_card = await agent.get_agent_card()
        card_dict = agent_card.model_dump()
        card_dict["id"] = agent.agent_id
        card_dict["status"] = "online"
        card_dict["is_demo"] = _is_demo(agent)
        return {
            "agents": [card_dict],
            "mode": "standalone",
            "server_demo_mode": server_demo_mode,
        }
    except Exception as e:
        logger.error(f"Error getting agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agents.")


@router.post("/api/agents")
@limiter.limit("5/minute")
async def create_agent(request: Request, body: CreateAgentRequest):
    """Create a new agent via inception.

    Runs the inception service to generate a DID and database,
    then loads the agent into the multi-agent manager.

    Only available in multi-agent mode.
    """
    agent_manager = getattr(request.app.state, 'agent_manager', None)
    if agent_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Agent creation is only available in multi-agent mode.",
        )

    name = body.name.strip()
    if not _AGENT_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail="Agent name must start with a letter and contain only letters, numbers, hyphens, or underscores.",
        )

    try:
        agent = await agent_manager.create_agent(name)
        return {
            "success": True,
            "agent": {
                "id": agent.agent_id,
                "name": name,
                "status": "online",
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating agent.")


@router.delete("/api/agents/{agent_name}")
@limiter.limit("10/minute")
async def delete_agent(request: Request, agent_name: str):
    """Remove an agent from the multi-agent manager.

    Shuts down the agent but does NOT delete its data directory.
    The agent can be re-loaded by restarting the server.

    Only available in multi-agent mode.
    """
    agent_manager = getattr(request.app.state, 'agent_manager', None)
    if agent_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Agent management is only available in multi-agent mode.",
        )

    removed = await agent_manager.remove_agent(agent_name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    return {
        "success": True,
        "name": agent_name,
        "message": f"Agent '{agent_name}' shut down and removed.",
    }


@router.get("/api/identity")
async def get_identity(request: Request):
    """Get agent identity information including avatar."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        agent_node = await storage.get_node(agent.agent_id) if storage else None

        # Get avatar_hash from agent node properties (like constitution_hash)
        avatar_hash = agent_node.properties.get("avatar_hash") if agent_node else None
        avatar_url = f"/api/files/{avatar_hash}" if avatar_hash else None

        # Get agent name and description from node properties
        agent_name = agent_node.properties.get("name") if agent_node else None
        description = agent_node.properties.get("description") if agent_node else None

        # Fall back to agent_metadata table for description
        if description is None:
            try:
                row = await agent._raw_storage.db.fetchone(
                    "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                    (agent.agent_id, "description"),
                )
                if row:
                    description = row[0]
            except Exception:
                pass

        # Hybrid identity surfacing (Quantum Hardening epic, follow-up to PR #999).
        # ``did`` stays the legacy did:pkh for backward compat with everything
        # already reading it. Post-ceremony agents additionally expose the new
        # did:web URI on ``signing_did``, plus ``is_hybrid`` / chain depth /
        # explicit ``legacy_did`` alias. Pre-ceremony agents return
        # ``is_hybrid=False`` and ``signing_did`` equals ``did``.
        identity_runtime = getattr(agent, "identity", None)
        is_hybrid = bool(identity_runtime and identity_runtime.is_hybrid)
        signing_did = (
            identity_runtime.new_did
            if is_hybrid and identity_runtime.new_did
            else agent.agent_id
        )
        chain_len = (
            len(identity_runtime.succession_chain)
            if identity_runtime and identity_runtime.succession_chain
            else 0
        )

        return {
            "did": agent.agent_id,
            "legacy_did": agent.agent_id,
            "signing_did": signing_did,
            "is_hybrid": is_hybrid,
            "succession_chain_length": chain_len,
            "name": agent_name,
            "description": description,
            "node_type": agent_node.node_type if agent_node else "agent",
            "created_at": agent_node.properties.get("created_at") if agent_node else None,
            "constitution_hash": agent_node.properties.get("constitution_hash") if agent_node else None,
            "avatar_hash": avatar_hash,
            "avatar_url": avatar_url,
        }
    except Exception as e:
        logger.error(f"Error getting identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving identity.")


# ------------------------------------------------------------------
# Identity update endpoints
# ------------------------------------------------------------------

class UpdateIdentityRequest(BaseModel):
    """Request body for updating agent identity."""
    name: Optional[str] = Field(None, min_length=1, max_length=64)
    description: Optional[str] = Field(None, max_length=500)


class SetAvatarUrlRequest(BaseModel):
    """Request body for setting avatar from URL."""
    url: str = Field(..., min_length=1)


class GenerateAvatarRequest(BaseModel):
    """Request body for AI avatar generation."""
    description: str = Field(..., min_length=1, max_length=1000)
    num_outputs: int = Field(2, ge=1, le=4)


_AVATAR_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_AVATAR_MAX_SIZE = 10 * 1024 * 1024  # 10 MB


@router.patch("/api/identity")
@limiter.limit("10/minute")
async def update_identity(request: Request, body: UpdateIdentityRequest):
    """Update agent name and/or description."""
    try:
        agent = get_agent(request)

        if body.name is None and body.description is None:
            raise HTTPException(status_code=422, detail="At least one of 'name' or 'description' required.")

        updated_fields = []

        if body.name is not None:
            try:
                await rename_agent_core(agent, body.name)
                updated_fields.append("name")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))

        if body.description is not None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            await agent._raw_storage.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (agent.agent_id, "description", body.description, now),
            )
            # Also update node properties
            agent_node = await agent.storage.get_node(agent.agent_id)
            if agent_node:
                agent_node.properties["description"] = body.description
                await agent.storage.add_node(agent_node)
            updated_fields.append("description")

        # Return updated identity
        agent_node = await agent.storage.get_node(agent.agent_id)
        avatar_hash = agent_node.properties.get("avatar_hash") if agent_node else None

        return {
            "success": True,
            "updated_fields": updated_fields,
            "did": agent.agent_id,
            "name": agent_node.properties.get("name") if agent_node else None,
            "description": agent_node.properties.get("description") if agent_node else None,
            "avatar_hash": avatar_hash,
            "avatar_url": f"/api/files/{avatar_hash}" if avatar_hash else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating identity.")


@router.post("/api/identity/avatar")
@limiter.limit("10/minute")
async def set_avatar(request: Request, file: Optional[UploadFile] = File(None)):
    """Set agent avatar from file upload or URL.

    Accepts either:
    - multipart/form-data with a 'file' field (image upload)
    - application/json with {"url": "https://..."} (set from URL)
    """
    try:
        agent = get_agent(request)
        storage = agent.storage
        if not storage or not hasattr(storage, 'files'):
            raise HTTPException(status_code=503, detail="File storage not available.")

        image_data: bytes
        source_url: Optional[str] = None

        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type and file:
            # File upload
            if file.content_type and file.content_type not in _AVATAR_MIME_TYPES:
                raise HTTPException(status_code=400, detail=f"Invalid image type '{file.content_type}'. Allowed: {', '.join(_AVATAR_MIME_TYPES)}")
            image_data = await file.read()
            if len(image_data) > _AVATAR_MAX_SIZE:
                raise HTTPException(status_code=400, detail=f"File too large. Maximum {_AVATAR_MAX_SIZE // (1024*1024)} MB.")
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Empty file.")
        elif "application/json" in content_type:
            # URL-based
            body = await request.json()
            url = body.get("url")
            if not url:
                raise HTTPException(status_code=422, detail="Missing 'url' field.")
            source_url = url
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    image_data = resp.content
            except httpx.HTTPError as e:
                raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")
            if len(image_data) > _AVATAR_MAX_SIZE:
                raise HTTPException(status_code=400, detail="Downloaded image too large.")
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Downloaded image is empty.")
        else:
            raise HTTPException(status_code=400, detail="Send multipart/form-data with file or application/json with url.")

        avatar_hash = await storage.files.store_avatar(
            image_data=image_data,
            agent_id=agent.agent_id,
            avatar_type="primary",
            source_url=source_url,
        )

        return {
            "success": True,
            "avatar_hash": avatar_hash,
            "avatar_url": f"/api/files/{avatar_hash}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting avatar: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting avatar.")


@router.post("/api/identity/avatar/generate")
@limiter.limit("3/minute")
async def generate_avatar(request: Request, body: GenerateAvatarRequest):
    """Generate avatar options using AI image generation.

    Requires VisualIdentityFeature to be enabled (REPLICATE_API_TOKEN set).
    """
    try:
        agent = get_agent(request)

        if not hasattr(agent, 'features'):
            raise HTTPException(status_code=503, detail="Agent features not available.")

        visual_identity = agent.features.get("VisualIdentityFeature")
        if not visual_identity:
            raise HTTPException(status_code=503, detail="Image generation not available. VisualIdentityFeature not loaded.")

        result = await visual_identity.generate_avatar(body.description, body.num_outputs)

        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            raise HTTPException(status_code=503, detail=f"Avatar generation failed: {error_msg}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating avatar: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating avatar.")


@router.get("/api/constitution")
async def get_constitution(request: Request):
    """Get the agent's constitution."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        constitution_text = await agent._get_governing_constitution()

        agent_node = await storage.get_node(agent.agent_id) if storage else None
        constitution_hash = agent_node.properties.get("constitution_hash") if agent_node else None
        constitution_node = await storage.get_node(constitution_hash) if constitution_hash and storage else None

        return {
            "text": constitution_text,
            "hash": constitution_hash,  # Include hash for UI compatibility
            "metadata": constitution_node.properties if constitution_node else {},
            "verified": constitution_text is not None,
        }
    except Exception as e:
        logger.error(f"Error getting constitution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving constitution.")


@router.get("/api/ipfs/status")
async def get_ipfs_status(request: Request):
    """Check IPFS node connectivity and status."""
    status = {
        "local_node": {"available": False, "error": None, "peer_id": None, "version": None},
        "gateways": [],
        "pinned_content": [],
    }

    local_api_url = get_ipfs_api_url() + "/api/v0"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.post(f"{local_api_url}/id") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    status["local_node"]["available"] = True
                    status["local_node"]["peer_id"] = data.get("ID")
                    status["local_node"]["agent_version"] = data.get("AgentVersion")

            if status["local_node"]["available"]:
                async with session.post(f"{local_api_url}/version") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        status["local_node"]["version"] = data.get("Version")

                async with session.post(f"{local_api_url}/pin/ls?type=recursive") as resp:
                    if resp.status == 200:
                        from kestrel_sovereign.kestrel_config.constants import MAX_PINNED_ITEMS_DISPLAY
                        data = await resp.json()
                        pins = data.get("Keys", {})
                        status["pinned_content"] = [
                            {"cid": cid, "type": info.get("Type")}
                            for cid, info in list(pins.items())[:MAX_PINNED_ITEMS_DISPLAY]
                        ]
    except Exception as e:
        logger.error(f"IPFS local node check failed: {e}")
        status["local_node"]["error"] = "Connection failed"

    test_cid = "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
    gateways = [
        {"name": "ipfs.io", "url": f"https://ipfs.io/ipfs/{test_cid}"},
        {"name": "dweb.link", "url": f"https://dweb.link/ipfs/{test_cid}"},
        {"name": "cloudflare-ipfs", "url": f"https://cloudflare-ipfs.com/ipfs/{test_cid}"},
    ]

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for gw in gateways:
            gw_status = {"name": gw["name"], "available": False, "latency_ms": None, "error": None}
            try:
                start = time.time()
                async with session.head(gw["url"]) as resp:
                    latency = (time.time() - start) * 1000
                    gw_status["available"] = resp.status == 200
                    gw_status["latency_ms"] = round(latency, 2)
            except Exception as e:
                logger.error(f"IPFS gateway {gw['name']} check failed: {e}")
                gw_status["error"] = "Connection failed"
            status["gateways"].append(gw_status)

    try:
        agent = get_agent(request)
        if hasattr(agent, 'storage') and agent.storage:
            storage = agent.storage
            if hasattr(storage, 'sovereign_adapter') and storage.sovereign_adapter:
                adapter = storage.sovereign_adapter
                if hasattr(adapter, 'filecoin_adapter') and adapter.filecoin_adapter:
                    status["filecoin_adapter"] = {
                        "configured": True,
                        "cache_dir": str(adapter.filecoin_adapter.cache_dir) if hasattr(adapter.filecoin_adapter, 'cache_dir') else None,
                    }
                else:
                    status["filecoin_adapter"] = {"configured": False}
    except Exception:
        pass  # Agent not available — skip filecoin status

    return status


@router.get("/api/wallet")
async def get_wallet(request: Request):
    """Get wallet balance and transaction history."""
    try:
        agent = get_agent(request)

        if hasattr(agent, 'wallet_agent') and agent.wallet_agent:
            wallet = agent.wallet_agent
            return {
                "balance": wallet.balance,
                "audit_reserve": wallet.audit_reserve,
                "total": wallet.balance + wallet.audit_reserve,
                "currency": "FIL",
            }
        else:
            storage = agent.storage
            agent_node = await storage.get_node(agent.agent_id) if storage else None
            if agent_node:
                return {
                    "balance": agent_node.properties.get("initialBalance", 0),
                    "audit_reserve": 0,
                    "total": agent_node.properties.get("initialBalance", 0),
                    "currency": "FIL",
                }
            return {"balance": 0, "audit_reserve": 0, "total": 0, "currency": "FIL"}
    except Exception as e:
        logger.error(f"Error getting wallet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving wallet.")


@router.get("/api/keys")
async def get_keys(request: Request):
    """Get configured API keys (no secrets exposed)."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'db'):
            return {"keys": [], "count": 0}

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(storage.db, agent.agent_id)
        keys = await key_storage.list_keys()

        keys_data = []
        for k in keys:
            keys_data.append({
                "id": k.id,
                "provider": k.provider_id,
                "is_active": k.is_active,
                "quota_limit": k.quota_limit,
                "quota_used": k.quota_used,
                "created_at": k.created_at.isoformat() if k.created_at else None,
            })

        return {
            "keys": keys_data,
            "count": len(keys_data),
        }
    except Exception as e:
        logger.error(f"Error getting keys: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving keys.")


@router.post("/api/keys")
async def add_key(request: Request):
    """Add a new API key for a provider."""
    try:
        body = await request.json()
        provider = body.get("provider", "").lower().strip()
        api_key = body.get("api_key", "").strip()
        quota_limit = body.get("quota_limit")  # Optional

        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")

        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'db'):
            raise HTTPException(status_code=503, detail="Storage not available")

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(storage.db, agent.agent_id)

        # Check if key already exists for this provider
        existing_keys = await key_storage.list_keys()
        for k in existing_keys:
            if k.provider_id == provider:
                raise HTTPException(
                    status_code=409,
                    detail=f"Key already exists for provider '{provider}'. Delete it first to add a new one."
                )

        # Store the key
        key_id = await key_storage.store_key(
            provider_id=provider,
            api_key=api_key,
            quota_limit=quota_limit,
        )

        return {
            "success": True,
            "key_id": key_id,
            "provider": provider,
            "message": f"API key for {provider} added successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error adding key.")


@router.delete("/api/keys/{provider}")
async def delete_key(request: Request, provider: str):
    """Delete an API key for a provider."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'db'):
            raise HTTPException(status_code=503, detail="Storage not available")

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(storage.db, agent.agent_id)

        # Find and delete the key
        existing_keys = await key_storage.list_keys()
        key_to_delete = None
        for k in existing_keys:
            if k.provider_id == provider.lower():
                key_to_delete = k
                break

        if not key_to_delete:
            raise HTTPException(status_code=404, detail=f"No key found for provider '{provider}'")

        # Delete the key (soft delete by deactivating, or hard delete)
        await storage.db.execute(
            "DELETE FROM agent_service_keys WHERE id = ?",
            (key_to_delete.id,)
        )

        return {
            "success": True,
            "provider": provider,
            "message": f"API key for {provider} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting key.")


@router.patch("/api/keys/{provider}")
async def update_key(request: Request, provider: str):
    """Update an API key's settings (quota, active status)."""
    try:
        body = await request.json()
        quota_limit = body.get("quota_limit")
        is_active = body.get("is_active")

        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'db'):
            raise HTTPException(status_code=503, detail="Storage not available")

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(storage.db, agent.agent_id)

        # Find the key
        existing_keys = await key_storage.list_keys()
        key_to_update = None
        for k in existing_keys:
            if k.provider_id == provider.lower():
                key_to_update = k
                break

        if not key_to_update:
            raise HTTPException(status_code=404, detail=f"No key found for provider '{provider}'")

        # Build update query — validate column names for safe interpolation
        updates = []
        params = []
        if quota_limit is not None:
            updates.append(f"{safe_column_name('quota_limit')} = ?")
            params.append(quota_limit)
        if is_active is not None:
            updates.append(f"{safe_column_name('is_active')} = ?")
            params.append(1 if is_active else 0)

        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")

        params.append(key_to_update.id)
        await storage.db.execute(
            f"UPDATE agent_service_keys SET {', '.join(updates)} WHERE id = ?",
            tuple(params)
        )

        return {
            "success": True,
            "provider": provider,
            "message": f"API key for {provider} updated successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error updating key.")


@router.get("/api/keys/{provider}/usage")
async def get_key_usage(request: Request, provider: str, days: int = Query(30, ge=1, le=365)):
    """Get usage history for a specific API key."""
    try:
        agent = get_agent(request)
        storage = agent.storage

        if not storage or not hasattr(storage, 'db'):
            return {"usage": [], "count": 0, "provider": provider}

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(storage.db, agent.agent_id)
        usage_records = await key_storage.get_usage(provider, days=days)

        usage_data = []
        for u in usage_records:
            usage_data.append({
                "id": u.id,
                "operation": u.operation,
                "units_consumed": u.units_consumed,
                "cost_estimate_usd": u.cost_estimate_usd,
                "recorded_at": u.recorded_at.isoformat() if u.recorded_at else None,
            })

        return {
            "provider": provider,
            "usage": usage_data,
            "count": len(usage_data),
            "days": days,
        }
    except Exception as e:
        logger.error(f"Error getting key usage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving key usage.")


# ---------------------------------------------------------------------------
# Three-tier key panel endpoints (resources.js)
#
# Kestrel's key resolution is layered: (1) Agent key, (2) User BYOK,
# (3) Platform pool.  The console's Resources panel queries the backend to
# render each tier and to show which source is active.  Tiers 2 and 3 live
# in Postgres-backed platform deployments only; in a local-sovereign SQLite
# agent these routes return empty/"not available" responses so the panel
# renders cleanly instead of the browser spamming 405 Method Not Allowed.
# ---------------------------------------------------------------------------


def _get_postgres_pool(agent):
    """Return the asyncpg Pool if the agent runs on a Postgres backend, else None.

    Local-sovereign SQLite agents have no pool; the BYOK/platform storages
    require one.  Callers use this to decide between "real" three-tier
    behavior and the empty/disabled shape.
    """
    storage = getattr(agent, "storage", None)
    if not storage or not hasattr(storage, "db"):
        return None
    db = storage.db
    backend = getattr(db, "backend", None)
    if backend is None or getattr(db, "backend_type", None) != "postgres":
        return None
    return getattr(backend, "_pool", None)


@router.get("/api/keys/available-sources")
async def get_available_key_sources(
    request: Request,
    provider: str = Query(..., description="Service provider id (openrouter, openai, etc.)"),
):
    """Report which tiers (agent/user/platform) can serve a given provider.

    Always checks tier 1 (agent's own keys) via local ServiceKeyStorage.
    Tiers 2-3 are checked only when a Postgres pool is available; in local
    SQLite mode they're reported as False, which is accurate for that
    deployment.
    """
    agent = get_agent(request)
    provider_id = provider.lower().strip()

    sources = {"agent": False, "user": False, "platform": False}
    platform_margin = None

    storage = getattr(agent, "storage", None)
    if storage and hasattr(storage, "db"):
        try:
            from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
            key_storage = ServiceKeyStorage(storage.db, agent.agent_id)
            sources["agent"] = await key_storage.has_key(provider_id=provider_id)
        except Exception as e:
            logger.debug(f"Agent key source check failed for {provider_id}: {e}")

    pool = _get_postgres_pool(agent)
    if pool is not None:
        try:
            from kestrel_sovereign.services.layered_key_resolver import LayeredKeyResolver
            resolver = LayeredKeyResolver(pool)
            user_id = getattr(request.state, "user_id", None)
            agent_did = getattr(agent, "agent_id", None)
            platform_result = await resolver.get_available_sources(
                provider=provider_id,
                user_id=user_id,
                agent_did=agent_did,
            )
            sources["user"] = bool(platform_result.get("user"))
            sources["platform"] = bool(platform_result.get("platform"))
            margin = platform_result.get("platform_margin")
            if margin is not None:
                platform_margin = f"{margin}"
        except Exception as e:
            logger.debug(f"Platform/user source check failed for {provider_id}: {e}")

    return {
        "provider": provider_id,
        "sources": sources,
        "platform_margin": platform_margin,
    }


@router.get("/api/keys/user")
async def list_user_keys(request: Request):
    """List user BYOK keys (empty shape on local-sovereign SQLite deployments).

    In Postgres-backed platform deployments this returns the user's
    passphrase-encrypted keys.  In local mode (no pool, no multi-user
    concept) returns an empty list so the Resources panel renders the
    "no personal keys added" state cleanly.
    """
    agent = get_agent(request)
    pool = _get_postgres_pool(agent)
    user_id = getattr(request.state, "user_id", None)

    if pool is None or not user_id:
        return {"keys": [], "count": 0, "available": False}

    try:
        from kestrel_sovereign.security.user_key_storage import UserKeyStorage
        user_storage = UserKeyStorage(pool, user_id)
        keys = await user_storage.list_keys()
        return {
            "keys": [
                {
                    "id": k.id,
                    "provider": k.provider_id,
                    "display_name": k.display_name,
                    "is_active": k.is_active,
                    "quota_limit": k.quota_limit,
                    "quota_used": k.quota_used,
                    "created_at": k.created_at.isoformat() if k.created_at else None,
                }
                for k in keys
            ],
            "count": len(keys),
            "available": True,
        }
    except Exception as e:
        logger.error(f"Error listing user BYOK keys: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving user keys.")


@router.post("/api/keys/user")
async def add_user_key(request: Request):
    """Add a user BYOK key.  Only available in platform (Postgres) deployments."""
    agent = get_agent(request)
    pool = _get_postgres_pool(agent)
    user_id = getattr(request.state, "user_id", None)

    if pool is None or not user_id:
        raise HTTPException(
            status_code=503,
            detail="User BYOK is only available on platform deployments. "
                   "Use Agent Keys (Add Agent Key) on this local sovereign instance.",
        )

    body = await request.json()
    provider_id = (body.get("provider") or "").lower().strip()
    api_key = (body.get("api_key") or "").strip()
    passphrase = body.get("passphrase") or ""
    display_name = body.get("display_name")

    if not provider_id:
        raise HTTPException(status_code=400, detail="provider is required")
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    if len(passphrase) < 8:
        raise HTTPException(status_code=400, detail="passphrase must be at least 8 characters")

    try:
        from kestrel_sovereign.security.user_key_storage import UserKeyStorage
        user_storage = UserKeyStorage(pool, user_id)
        key_id = await user_storage.store_key(
            provider_id=provider_id,
            api_key=api_key,
            passphrase=passphrase,
            display_name=display_name,
        )
        return {
            "success": True,
            "key_id": key_id,
            "provider": provider_id,
            "message": f"Personal {provider_id} key stored successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding user BYOK key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error adding user key.")


@router.post("/api/keys/user/verify")
async def verify_user_passphrase(request: Request):
    """Verify the user's BYOK passphrase against ANY stored key (platform only)."""
    agent = get_agent(request)
    pool = _get_postgres_pool(agent)
    user_id = getattr(request.state, "user_id", None)

    if pool is None or not user_id:
        return {"valid": False, "available": False}

    body = await request.json()
    passphrase = body.get("passphrase") or ""
    if not passphrase:
        raise HTTPException(status_code=400, detail="passphrase is required")

    try:
        from kestrel_sovereign.security.user_key_storage import UserKeyStorage
        user_storage = UserKeyStorage(pool, user_id)
        keys = await user_storage.list_keys()
        if not keys:
            return {"valid": False, "available": True, "reason": "no_keys"}
        # Verify against the first active key — if the passphrase works for
        # any key it's correct (same passphrase is used for all of them).
        for k in keys:
            if await user_storage.verify_passphrase(k.provider_id, passphrase):
                return {"valid": True, "available": True}
        return {"valid": False, "available": True}
    except Exception as e:
        logger.error(f"Error verifying user passphrase: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error verifying passphrase.")


@router.delete("/api/keys/user/{provider}")
async def delete_user_key(request: Request, provider: str):
    """Delete a user BYOK key (platform deployments only)."""
    agent = get_agent(request)
    pool = _get_postgres_pool(agent)
    user_id = getattr(request.state, "user_id", None)

    if pool is None or not user_id:
        raise HTTPException(
            status_code=503,
            detail="User BYOK is only available on platform deployments.",
        )

    try:
        from kestrel_sovereign.security.user_key_storage import UserKeyStorage
        user_storage = UserKeyStorage(pool, user_id)
        deleted = await user_storage.delete_key(provider.lower())
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"No personal key configured for provider '{provider}'.",
            )
        return {
            "success": True,
            "provider": provider.lower(),
            "message": f"Personal {provider} key removed.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user BYOK key: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error deleting user key.")


@router.get("/api/keys/platform")
async def get_platform_access(request: Request):
    """List platform vending-machine providers (empty on local deployments)."""
    agent = get_agent(request)
    pool = _get_postgres_pool(agent)

    if pool is None:
        return {"providers": [], "available": False}

    try:
        from kestrel_sovereign.security.platform_key_storage import PlatformKeyStorage
        platform_storage = PlatformKeyStorage(pool)
        keys = await platform_storage.list_keys()
        providers = []
        for info in keys:
            providers.append({
                "provider_id": info.provider_id,
                "provider_name": getattr(info, "provider_name", info.provider_id),
                "is_available": info.is_active,
                "margin_pct": (f"{int(info.margin_pct * 100)}%"
                               if info.margin_pct is not None else None),
                "rate_limit_per_companion": getattr(info, "rate_limit_per_companion", None),
                "pricing_hint": getattr(info, "pricing_hint", None),
            })
        return {"providers": providers, "available": True}
    except Exception as e:
        logger.error(f"Error listing platform access: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving platform access.")


@router.get("/api/models")
async def list_agent_models(
    request: Request,
    featured_only: bool = Query(False, description="Only return featured models"),
    category: Optional[str] = Query(None, description="Filter by category (chat, embedding, image, audio)"),
    providers: Optional[str] = Query(None, description="Comma-separated provider names to filter"),
    use_cache: bool = Query(True, description="Use cached results if available"),
):
    """
    List available LLM models from all providers.

    Query Parameters:
        featured_only: Only return featured models (default: true)
        category: Filter by category (chat, embedding, image, audio)
        providers: Comma-separated list of providers to include
        use_cache: Use cached results (default: true)

    Returns:
        {
            "by_provider": {"openai": [...], "anthropic": [...]},
            "featured": [...],
            "all": [...],
            "default": "model-id"
        }
    """
    try:
        agent = get_agent(request)
        models = []

        # Parse category if provided
        model_category = None
        if category:
            try:
                model_category = ModelCategory(category.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid category: {category}. Must be one of: chat, embedding, image, audio"
                )

        # Parse providers if provided
        provider_list = None
        if providers:
            provider_list = [p.strip() for p in providers.split(",")]

        if hasattr(agent, 'llm_service') and agent.llm_service:
            models = await agent.llm_service.discover_all_models(
                use_cache=use_cache,
                featured_only=featured_only,
                category=model_category,
                providers=provider_list
            )

        # Convert to dicts for JSON response
        models_data = [m.to_dict() for m in models]

        # Group by vendor. `ModelInfo.provider` is the vendor field; the name
        # was kept for backward-file-compat but the semantic is vendor.
        by_vendor: Dict[str, List[Dict]] = {}
        for model in models:
            by_vendor.setdefault(model.provider, []).append(model.to_dict())

        # Featured models (computed per ModelCatalogService rules)
        featured = [m.to_dict() for m in models if m.is_featured]

        # Effective default model from the runtime routing source.
        default_model = None
        if hasattr(agent, 'llm_service') and agent.llm_service:
            default_model = agent.llm_service.get_active_model_id()

        # Surface the routes (vendor,route pairs) configured on this agent so
        # the UI can show a per-vendor route selector without extra round-trips.
        routes = []
        if hasattr(agent, 'llm_service') and agent.llm_service:
            for p in agent.llm_service.providers:
                routes.append({
                    "vendor": p.get("vendor"),
                    "route": p.get("route"),
                    "is_local": p.get("is_local"),
                    "model": p.get("model"),
                })

        return {
            "by_vendor": by_vendor,
            "featured": featured,
            "all": models_data,
            "default": default_model,
            "count": len(models_data),
            "routes": routes,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing models.")


@router.get("/api/model/current")
async def get_current_model(request: Request):
    """Return the currently active ``{vendor, model, route}`` selection."""
    try:
        agent = get_agent(request)
        selection = {"model": None, "vendor": None, "route": None, "model_name": None}

        if hasattr(agent, 'llm_service') and agent.llm_service:
            from kestrel_sovereign.llm.service import resolve_active_model_selection
            selection = resolve_active_model_selection(agent.llm_service)

        return {
            "model": selection["model"],
            "vendor": selection["vendor"],
            "route": selection["route"],
            "model_name": selection["model_name"],
        }
    except Exception as e:
        logger.error(f"Error getting current model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting current model.")


@router.post("/api/model/set")
async def set_current_model(request: Request):
    """Set the active ``{vendor, model, route?}`` for this session.

    Accepts JSON body: ``{"vendor": "anthropic", "model": "claude-sonnet-4-6", "route": "plan"}``.
    ``vendor`` and ``route`` are optional — omitting vendor lets routing scan
    all configured vendors for the model; omitting route uses the first
    configured route for that vendor. Also accepts combined forms in the
    ``model`` field: ``"anthropic/claude-sonnet-4-6"`` or
    ``"anthropic:plan/claude-sonnet-4-6"``.
    """
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="'model' field is required.")

        agent = get_agent(request)
        if not hasattr(agent, 'llm_service') or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")

        vendor = data.get("vendor")
        route = data.get("route")
        model_name = model

        # Combined forms in the model field.
        if "/" in model_name and not vendor:
            left, model_name = model_name.split("/", 1)
            if ":" in left:
                vendor, route = left.split(":", 1)
            else:
                vendor = left

        try:
            agent.llm_service.set_model_preference(model_name, vendor, route)
        except ValueError as ve:
            # set_model_preference raises ValueError for bare models that
            # can't be auto-resolved (unknown or ambiguous). Surface the
            # reason so the caller — especially an LLM using the set_model
            # tool — sees *why* and can try again with a vendor-qualified
            # form instead of silently failing.
            raise HTTPException(status_code=400, detail=str(ve))

        # Re-read the persisted selection so the response reflects any
        # vendor auto-resolution that happened inside set_model_preference.
        pref = agent.llm_service.get_model_preference()
        final_vendor = pref.get("vendor")
        final_route = pref.get("route")
        final_model = pref.get("model") or model_name

        if final_vendor and final_route:
            full = f"{final_vendor}:{final_route}/{final_model}"
        elif final_vendor:
            full = f"{final_vendor}/{final_model}"
        else:
            full = final_model

        return {
            "success": True,
            "vendor": final_vendor,
            "route": final_route,
            "model": final_model,
            "full_model": full,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting model.")


@router.get("/v1/models")
def list_models_v1(request: Request):
    """Return a minimal models list for OpenAI-compatible clients.

    Uses get_active_model_id() so the reported model reflects any
    mandate preference, not just the first provider's config default.
    """
    try:
        agent = get_agent(request)
        model_id = "kestrel-local"
        if hasattr(agent, 'llm_service') and agent.llm_service:
            active = agent.llm_service.get_active_model_id()
            if active and active != "auto":
                model_id = active
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"object": "list", "data": [{"id": "kestrel-local", "object": "model"}]}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat Completions-compatible endpoint.

    Respects the 'model' field from the request body. When a model is provided
    (e.g. "openai/gpt-5-mini"), it is passed as model_override to the agent for
    this request only.
    """
    try:
        data = await request.json()
        messages = data.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        user_input = last_user.get("content") if last_user else ""
        if not user_input:
            user_input = "\n".join([m.get("content", "") for m in messages])

        agent = get_agent(request)

        # Extract model from request and pass it through to the agent
        model_from_request = data.get("model")
        model_override = None
        # Ignore sentinel values that aren't real model names
        if model_from_request and model_from_request not in ("kestrel-local", "auto"):
            model_override = model_from_request

        assistant_text = await agent.process_input(
            user_input,
            model_override=model_override,
            caller=get_caller(request),
        )

        # Report the model that was actually routed, not just what the
        # client requested.  get_active_model_id() reflects mandate
        # preference and provider routing — the honest answer.
        resolved_model = "kestrel-local"
        if hasattr(agent, 'llm_service') and agent.llm_service:
            active = agent.llm_service.get_active_model_id()
            if active and active != "auto":
                resolved_model = active

        resp = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resolved_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return resp
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in chat completions.")
