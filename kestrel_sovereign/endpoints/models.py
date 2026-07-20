"""Model, wallet, and IPFS status endpoints."""
from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import aiohttp
import httpx
import asyncio
import os
import re
import time
import uuid
import logging

from kestrel_sovereign.kestrel_config.defaults import get_ipfs_api_url
from kestrel_sovereign.llm.model_metadata import ModelCategory
from kestrel_sovereign.sql_utils import safe_column_name
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.features.bootstrap.feature import rename_agent_core
from kestrel_sovereign.endpoints.agent_helpers import get_agent, get_caller
from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_database,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Validation: agent names must be alphanumeric + hyphens/underscores, 1-64 chars
_AGENT_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")


def _key_storage_privacy_detail() -> str:
    return "Service key storage is unavailable in the current privacy mode."


def _get_service_key_db(agent):
    if hides_persisted_user_content(agent):
        raise HTTPException(status_code=403, detail=_key_storage_privacy_detail())
    db = resolve_feature_database(agent)
    if db is None:
        raise HTTPException(status_code=503, detail="Storage not available")
    return db


def _service_key_storage_hidden(agent) -> bool:
    return hides_persisted_user_content(agent)


def _get_service_key_db_or_none(agent):
    if hides_persisted_user_content(agent):
        return None
    return resolve_feature_database(agent)


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
            # POST /api/agents works on this in-process manager. Keep the
            # capability explicit so older clients can safely treat absence as
            # false instead of attempting a route the host may not expose.
            "can_create_agents": True,
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
            "can_create_agents": False,
        }
    except HTTPException:
        raise
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

    # A name can be REGISTERED without being LOADED (remote agents,
    # autostart=false locals) — the manager's duplicate check only sees loaded
    # agents, so creation would silently REPLACE that registration in
    # multi_agent.toml (codex P2 round 7). Check against the CURRENT ON-DISK
    # config, not the startup-time snapshot: an operator can edit the file
    # while the server runs, and saving the stale snapshot back would discard
    # every external change (codex P1 round 10). Case-insensitive: the toml is
    # the routing namespace and case-folded collisions are operator error.
    def _current_config():
        """(config, path, reload_ok). For CONFIG-DRIVEN deployments a reload
        failure fails CLOSED (codex P2 round 12): writing the startup
        snapshot over a file we couldn't read would discard operator edits —
        or clobber a malformed file mid-repair."""
        cfg_path = getattr(request.app.state, 'multi_agent_config_path', None)
        if cfg_path:
            from kestrel_sovereign.multi_agent.config import MultiAgentConfig as _MAC
            try:
                return _MAC.from_file(cfg_path), cfg_path, True
            except Exception as reload_err:
                logger.error(
                    f"Could not reload {cfg_path} ({reload_err}); refusing to "
                    "persist over a file that can't be read (fail closed)."
                )
                return getattr(request.app.state, 'multi_agent_config', None), cfg_path, False
        return getattr(request.app.state, 'multi_agent_config', None), cfg_path, True

    ma_config_pre, _, _reload_ok_pre = _current_config()
    if ma_config_pre is not None:
        taken = {existing.lower() for existing in getattr(ma_config_pre, 'agents', {})}
        if name.lower() in taken:
            raise HTTPException(
                status_code=409,
                detail=f"An agent named '{name}' is already registered in the multi-agent config.",
            )
        # Reserve every port the CURRENT config knows about (codex P1 round
        # 11): an agent or host-port added to the file after startup isn't in
        # the manager's boot-time reservations, and allocating it would
        # persist a port conflict that fails validation on the next boot.
        from kestrel_sovereign.multi_agent.config import LocalAgentConfig as _LAC
        host_port = getattr(getattr(ma_config_pre, 'host', None), 'port', None)
        if isinstance(host_port, int):
            agent_manager._reserved_ports.add(host_port)
        for _cfg in getattr(ma_config_pre, 'agents', {}).values():
            if isinstance(_cfg, _LAC):
                agent_manager._reserved_ports.add(_cfg.port)

    try:
        agent = await agent_manager.create_agent(name)
        # Persist the registration when the deployment is config-file-driven
        # (codex P1 on #2358): startup loads multi_agent.toml whenever it
        # exists, so an unpersisted runtime creation silently vanishes on the
        # next restart. Auto-discovered deployments (no toml) re-discover from
        # agent_data/ and need no write. Best-effort: a persistence failure is
        # SURFACED in the response, never a rollback of the live agent.
        persisted = None
        # RE-read the on-disk config for the merge (codex P1 round 10): the
        # inception above awaited, so even our own pre-check snapshot may be
        # stale. The fresh object carries every external edit forward.
        ma_config, config_path, reload_ok = _current_config()
        if config_path and not reload_ok:
            # Agent is live but we refuse the stale rewrite — surfaced to the
            # dialog exactly like a write failure.
            persisted = False
        elif config_path and ma_config is not None:
            try:
                created_cfg = agent_manager._created_configs.get(name)
                if created_cfg is not None:
                    ma_config.agents[name] = created_cfg
                    # Mutating .agents doesn't rerun the model validator —
                    # re-validate the merged config so a conflict is surfaced
                    # HERE (persisted:false + intact file) instead of bricking
                    # the next boot (codex P1 round 11).
                    type(ma_config).model_validate(ma_config.model_dump())
                    ma_config.save(config_path)
                    # Keep the app-state snapshot current for the next reader.
                    request.app.state.multi_agent_config = ma_config
                    persisted = True
            except Exception as persist_err:
                logger.error(
                    f"Agent '{name}' created but NOT persisted to {config_path}: {persist_err}",
                    exc_info=True,
                )
                persisted = False
        return {
            "success": True,
            "agent": {
                "id": agent.agent_id,
                "name": name,
                "status": "online",
            },
            # None = auto-discovered deployment (no write needed);
            # True/False = toml-driven write outcome.
            "persisted": persisted,
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

    try:
        removed = await agent_manager.remove_agent(agent_name)
    except ValueError as e:
        # remove_agent refuses to delete an agent that still has budgeted child
        # agents (#2113) — that teardown must go through terminate_child. Surface
        # it as a controlled 409, not a 500.
        raise HTTPException(status_code=409, detail=str(e))
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
        # ``did`` stays the agent's canonical node id for backward compat with
        # everything already reading it (legacy did:pkh for classical/rotated
        # agents; the did:web URI for born-hybrid agents, #2397). Rotated
        # agents additionally expose the new did:web URI on ``signing_did``,
        # plus ``is_hybrid`` / chain depth / explicit ``legacy_did``.
        # ``legacy_did`` is null for born-hybrid agents — they never had a
        # classical identity, and aliasing the did:web there would fabricate
        # one.
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
        if identity_runtime is not None:
            legacy_did = identity_runtime.legacy_did
        elif agent.agent_id.startswith("did:web:"):
            legacy_did = None
        else:
            legacy_did = agent.agent_id

        return {
            "did": agent.agent_id,
            "legacy_did": legacy_did,
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
    except HTTPException:
        raise
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
async def update_identity(request: Request, response: Response, body: UpdateIdentityRequest):
    """Update agent name and/or description."""
    try:
        agent = get_agent(request)

        if body.name is None and body.description is None:
            raise HTTPException(status_code=422, detail="At least one of 'name' or 'description' required.")

        updated_fields = []
        rename_outcome = None
        partial_update = False

        if body.name is not None:
            try:
                rename_outcome = await rename_agent_core(agent, body.name)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            if not rename_outcome.any_written:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": "Error updating identity name.",
                        "rename_outcome": rename_outcome.to_dict(),
                    },
                )
            if rename_outcome.success:
                updated_fields.append("name")
            else:
                response.status_code = 207
                partial_update = True
                updated_fields.append("name_partial")

        if body.description is not None:
            from kestrel_sovereign.bootstrap.service import persist_agent_description
            # persist_agent_description does not swallow write failures, so a
            # failed metadata write or a failed update of an existing graph
            # node propagates here and becomes a 500 (handled below) — the
            # operator never sees a false success with stale data behind it.
            await persist_agent_description(
                agent._raw_storage.db,
                agent.storage,
                agent.agent_id,
                body.description,
            )
            updated_fields.append("description")

        # Return updated identity
        try:
            agent_node = await agent.storage.get_node(agent.agent_id)
        except Exception:
            if rename_outcome and rename_outcome.any_written and not rename_outcome.success:
                agent_node = None
            else:
                raise
        avatar_hash = agent_node.properties.get("avatar_hash") if agent_node else None

        payload = {
            "success": not partial_update,
            "updated_fields": updated_fields,
            "did": agent.agent_id,
            "name": agent_node.properties.get("name") if agent_node else getattr(agent, "_agent_name", None),
            "description": agent_node.properties.get("description") if agent_node else None,
            "avatar_hash": avatar_hash,
            "avatar_url": f"/api/files/{avatar_hash}" if avatar_hash else None,
        }
        if rename_outcome is not None:
            payload["rename_outcome"] = rename_outcome.to_dict()
        return payload

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
            # SSRF guard (#1727): reject URLs whose host resolves to a private /
            # loopback / link-local / metadata address before fetching.
            from kestrel_sovereign.security.ssrf import (
                SSRFError,
                assert_safe_url,
                pinned_httpx_async_transport,
            )
            try:
                validated_url = await assert_safe_url(url)
            except SSRFError as e:
                raise HTTPException(status_code=400, detail=f"Disallowed avatar URL: {e}")
            try:
                transport = pinned_httpx_async_transport(validated_url)
                async with httpx.AsyncClient(
                    timeout=30.0,
                    transport=transport,
                ) as client:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting constitution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving constitution.")


@router.get("/api/ipfs/status")
async def get_ipfs_status(request: Request):
    """Check IPFS node connectivity and status."""
    # Resolve the agent up front so a missing agent surfaces as the
    # contractual 503 before any network probing starts (#2495).
    agent = get_agent(request)
    status = {
        "local_node": {"available": False, "error": None, "peer_id": None, "version": None},
        "backup_tier": {},
        "gateways": [],
        "pinned_content": [],
    }

    try:
        from kestrel_sovereign.storage.sync.health import check_sovereign_ipfs_health

        status["backup_tier"] = (
            await check_sovereign_ipfs_health(
                api_url=os.environ.get("SOVEREIGN_IPFS_URL")
            )
        ).to_dict()
    except Exception as e:
        logger.error(f"Sovereign IPFS backup tier check failed: {e}")
        status["backup_tier"] = {
            "name": "sovereign_ipfs",
            "label": "sovereign-operated",
            "configured": False,
            "status": "decommissioned",
            "message": "Sovereign-operated IPFS tier state unavailable",
            "details": {},
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
        pass  # Filecoin adapter introspection is best-effort

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting wallet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving wallet.")


@router.get("/api/keys")
async def get_keys(request: Request):
    """Get configured API keys (no secrets exposed)."""
    try:
        agent = get_agent(request)
        if _service_key_storage_hidden(agent):
            raise HTTPException(status_code=403, detail=_key_storage_privacy_detail())
        db = _get_service_key_db_or_none(agent)
        if db is None:
            return {"keys": [], "count": 0}

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(db, agent.agent_id)
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
    except HTTPException:
        raise
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
        db = _get_service_key_db(agent)

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
        from kestrel_sovereign.security.exceptions import KeyStorageError

        key_storage = ServiceKeyStorage(db, agent.agent_id)

        # Store the key. Insert-only enforcement lives in storage (store_key
        # raises KeyStorageError over an existing key), so a duplicate — even
        # one that slips past a preflight under concurrency — surfaces as 409
        # here rather than a generic 500. Rotation is a separate approval-gated
        # path (replace=True); the plain add API never overwrites (F196).
        try:
            key_id = await key_storage.store_key(
                provider_id=provider,
                api_key=api_key,
                quota_limit=quota_limit,
            )
        except KeyStorageError as e:
            raise HTTPException(status_code=409, detail=str(e))

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
        db = _get_service_key_db(agent)

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(db, agent.agent_id)

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
        await db.execute(
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
        db = _get_service_key_db(agent)

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(db, agent.agent_id)

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
        await db.execute(
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
        if _service_key_storage_hidden(agent):
            raise HTTPException(status_code=403, detail=_key_storage_privacy_detail())
        db = _get_service_key_db_or_none(agent)
        if db is None:
            return {"usage": [], "count": 0, "provider": provider, "days": days}

        from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage

        key_storage = ServiceKeyStorage(db, agent.agent_id)
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
    except HTTPException:
        raise
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
    db = _get_service_key_db_or_none(agent)
    if db is None:
        return None
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

    db = _get_service_key_db_or_none(agent)
    if db is not None:
        try:
            from kestrel_sovereign.security.service_key_storage import ServiceKeyStorage
            key_storage = ServiceKeyStorage(db, agent.agent_id)
            sources["agent"] = await key_storage.has_key(provider_id=provider_id)
        except Exception as e:
            logger.debug(f"Agent key source check failed for {provider_id}: {e}")

    pool = _get_postgres_pool(agent)
    if pool is not None:
        try:
            # Frinz-side primitive (relocated 2026-05; see kestrel-sovereign#1156).
            # ImportError on a foundation-only deployment lands in the
            # outer except below and is logged at debug; sources stay False.
            from frinz.services.layered_key_resolver import LayeredKeyResolver
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
        # Frinz-side primitive (relocated 2026-05; see kestrel-sovereign#1156).
        from frinz.security.user_key_storage import UserKeyStorage
    except ImportError:
        # Foundation-only deployment (no Frinz installed). Same response
        # as the no-pool path — multi-user BYOK requires Frinz.
        return {"keys": [], "count": 0, "available": False}

    try:
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
        from frinz.security.user_key_storage import UserKeyStorage
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="User BYOK is only available on Frinz-platform deployments.",
        )

    try:
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
        from frinz.security.user_key_storage import UserKeyStorage
    except ImportError:
        return {"valid": False, "available": False}

    try:
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
        from frinz.security.user_key_storage import UserKeyStorage
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="User BYOK is only available on Frinz-platform deployments.",
        )

    try:
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
        from frinz.security.platform_key_storage import PlatformKeyStorage
    except ImportError:
        return {"providers": [], "available": False}

    try:
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
    vendor: Optional[str] = Query(None, description="Scope the list to a single vendor (pairs with 'route')"),
    route: Optional[str] = Query(None, description="Scope the list to a single route of 'vendor' (e.g. plan, api)"),
    use_cache: bool = Query(True, description="Use cached results if available"),
):
    """
    List available LLM models from all providers.

    Query Parameters:
        featured_only: Only return featured models (default: false — the
            selector fetches the full set once and filters featured/all
            client-side via the "Show all" expander)
        category: Filter by category (chat, embedding, image, audio)
        providers: Comma-separated list of providers to include
        vendor: Scope the returned list to a single vendor. When combined
            with ``route``, the list is drawn from THAT route's own serveable
            set (route-scoped discovery), so a plan route no longer offers
            api-only models it can't serve (#2262). Requires ``route``.
        route: The route of ``vendor`` to scope to (e.g. ``plan``, ``api``).
            Requires ``vendor``. A route change on the UI selector re-fetches
            with the new pair to repopulate the model combo.
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
        served_stale_catalog = False
        stale_embedding_routes: set = set()

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

        # ``route`` without ``vendor`` is ambiguous — a route name (plan/api)
        # only identifies a serveable set relative to its vendor.
        if route and not vendor:
            raise HTTPException(
                status_code=400,
                detail="'route' requires 'vendor' — a route is only meaningful within a vendor.",
            )

        if hasattr(agent, 'llm_service') and agent.llm_service:
            if vendor:
                # Route-scoped discovery: the returned set comes from THIS
                # (vendor, route)'s own adapter discovery, so a plan route's
                # list never cross-contaminates with api-only models (#2262).
                models = await agent.llm_service.discover_models_for_route(
                    vendor,
                    route,
                    use_cache=use_cache,
                    featured_only=featured_only,
                    category=model_category,
                )
            else:
                if use_cache:
                    from kestrel_sovereign.llm.model_cache import get_shared_model_cache

                    shared_catalog = get_shared_model_cache()
                    served_stale_catalog = (
                        shared_catalog.get() is None
                        and shared_catalog.get_any() is not None
                    )
                    if served_stale_catalog:
                        # Keep discovery-only embedding routes truthful without
                        # putting provider I/O back on the latency-sensitive
                        # stale-catalog response. A prior fresh discovery leaves
                        # its route-scoped embedding facet on this service; a
                        # cold process can still recover obvious embedding
                        # entries from the persisted general catalog.
                        cached_embeddings = getattr(
                            agent.llm_service,
                            "_embedding_discovery_cache",
                            None,
                        )
                        if cached_embeddings is not None:
                            stale_embedding_routes = {
                                m.route for m in cached_embeddings if m.route
                            }
                        else:
                            stale_embedding_routes = {
                                m.route
                                for m in (shared_catalog.get_any() or [])
                                if getattr(m, "route", None)
                                and getattr(m, "category", None)
                                == ModelCategory.EMBEDDING
                            }
                models = await agent.llm_service.discover_all_models(
                    use_cache=use_cache,
                    featured_only=featured_only,
                    category=model_category,
                    providers=provider_list,
                    # Model catalogs are descriptive, not routing state. Keep
                    # agent switching responsive when the five-minute cache is
                    # stale and coalesce provider discovery in the background.
                    stale_while_revalidate=use_cache,
                )

        # Convert to dicts for JSON response
        models_data = [m.to_dict() for m in models]

        # Group by vendor. `ModelInfo.provider` is the vendor field; the name
        # was kept for backward-file-compat but the semantic is vendor.
        by_vendor: Dict[str, List[Dict]] = {}
        for model, model_data in zip(models, models_data):
            by_vendor.setdefault(model.provider, []).append(model_data)

        # Rank each vendor bucket: featured first, then newest-first by the
        # provider-supplied ``created_at`` (naming-agnostic recency), then by
        # id. This makes the FIRST entry per vendor the best default — the UI
        # seeds to it, so an empty/cross-vendor server default no longer falls
        # through to an alphabetical accident like ``gpt-3.5-turbo``. (#2015)
        from kestrel_sovereign.llm.model_catalog import _created_key

        def _bucket_sort_key(m: Dict):
            return (
                not m.get("is_featured"),
                tuple(-n for n in _created_key(m.get("created_at"))),
                m.get("id", ""),
            )

        for bucket in by_vendor.values():
            bucket.sort(key=_bucket_sort_key)

        # Featured models (computed per ModelCatalogService rules)
        featured = [
            model_data
            for model, model_data in zip(models, models_data)
            if model.is_featured
        ]

        # Effective default model from the runtime routing source.
        default_model = None
        if hasattr(agent, 'llm_service') and agent.llm_service:
            default_model = agent.llm_service.get_active_model_id()

        # Surface the routes (vendor,route pairs) configured on this agent so
        # the UI can show a per-vendor route selector without extra round-trips.
        routes = []
        if hasattr(agent, 'llm_service') and agent.llm_service:
            # #2338: a route advertises embedding capability when discovery
            # finds ≥1 embedding model FOR THAT ROUTE — not only when a config
            # pin set ``supports_embeddings``, and NOT collapsed by vendor (an
            # embedding-capable openai:api must not flip capability on for a
            # non-embedding openai:plan/codex sibling). Key the discovered set by
            # the model's originating route.
            embedding_routes: set = set(stale_embedding_routes)
            if not served_stale_catalog:
                try:
                    discovered = await agent.llm_service.discover_embedding_models()
                    embedding_routes = {m.route for m in discovered if m.route}
                except Exception as e:  # pragma: no cover - never fail the list
                    logger.debug(f"embedding discovery skipped in /api/models: {e}")
            for p in agent.llm_service.providers:
                capabilities = p.get("capabilities") or {}
                route_name = p.get("name")
                supports_embeddings = bool(
                    capabilities.get("supports_embeddings")
                ) or (route_name in embedding_routes)
                routes.append({
                    "vendor": p.get("vendor"),
                    "route": p.get("route"),
                    "is_local": p.get("is_local"),
                    "model": p.get("model"),
                    # Whether this route can serve embeddings — lets the model
                    # settings popover's Embeddings section list only
                    # embedding-capable routes without a second round-trip (#2264).
                    # Discovery flips this on for routes with a discovered
                    # embedding model even without a TOML pin (#2338).
                    "supports_embeddings": supports_embeddings,
                    # The dim this route resolves to (declared capability), so the
                    # embeddings UI can mark a route whose dim can't write into the
                    # column ("needs migration") BEFORE selection (#2417).
                    "embedding_dim": capabilities.get("embedding_dim"),
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
        selection = {
            "model": None, "vendor": None, "route": None,
            "model_name": None, "is_auto": False,
        }

        if hasattr(agent, 'llm_service') and agent.llm_service:
            from kestrel_sovereign.llm.service import resolve_active_model_selection
            selection = resolve_active_model_selection(agent.llm_service)

        return {
            "model": selection["model"],
            "vendor": selection["vendor"],
            "route": selection["route"],
            "model_name": selection["model_name"],
            # #2419 — surface auto-resolution so the header button can render
            # "Auto — currently <model>" and make auto-drift observable.
            "is_auto": selection.get("is_auto", False),
        }
    except HTTPException:
        raise
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


@router.get("/api/embedding/models")
async def get_embedding_models(request: Request):
    """Return dynamically-discovered embedding models per vendor (#2338).

    Chat models are discovered dynamically; embedding models must be too. This
    reads the discovered catalog (OpenRouter's dedicated ``/embeddings/models``
    endpoint, Ollama's ``/api/show`` capability check, OpenAI's id-prefix
    filter) so the embeddings settings UI (#2337) populates its provider
    dropdown and model picker with NO TOML editing. Config pins are folded in
    as overrides (``is_pinned``), not prerequisites.

    ``shared_space_candidates`` are models discovered on BOTH a local and a
    cloud route (#2290/#2337 "Universal" option), computed by intersection
    rather than hardcoded to qwen3.
    """
    try:
        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")

        discovered = await agent.llm_service.discover_embedding_models()
        shared = await agent.llm_service.shared_embedding_space_candidates()
        # #2337 — the featured "Universal" options: shared models enriched with
        # their member routes (each carrying that route's own slug) so the UI can
        # render the pinned-at-top option and run guided setup across both members.
        universal = await agent.llm_service.universal_embedding_space_options()

        by_vendor: Dict[str, list] = {}
        for m in discovered:
            by_vendor.setdefault(m.provider, []).append(m.to_dict())

        return {
            "by_vendor": by_vendor,
            "all": [m.to_dict() for m in discovered],
            "shared_space_candidates": [m.to_dict() for m in shared],
            "universal": universal,
            "count": len(discovered),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing embedding models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing embedding models.")


@router.get("/api/embedding/settings")
async def get_embedding_settings(request: Request):
    """Return the resolved embedding-channel state for the active session (#2263).

    Surfaces enough for a UI to render an "Auto — follow chat" default and a
    dimension-mismatch warning: the configured ``embedding_route`` (or null =
    auto), the RESOLVED provider for the active session, its ``embedding_model``
    and ``embedding_dim``, and the deployment's ``KESTREL_EMBEDDING_DIM``.
    """
    try:
        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")
        # Fold live embedding discovery into route capabilities before reading
        # settings (#2366). ``get_embedding_settings`` surfaces the corpus
        # space-change warning from ``_embedding_space_change_warnings``, but that
        # record is only produced by ``reconcile_embedding_capabilities`` (the
        # POST/PUT paths reconcile; a fresh GET after startup would otherwise
        # report stale/empty capability state and no warning). Best-effort — a
        # discovery hiccup must not fail the read.
        if hasattr(agent.llm_service, "reconcile_embedding_capabilities"):
            try:
                await agent.llm_service.reconcile_embedding_capabilities(use_cache=True)
            except Exception as e:  # pragma: no cover - never fail the GET
                logger.debug(f"embedding capability reconcile skipped in GET: {e}")
        # Resolve the active route's model/dim through the single #2372 resolver
        # so a cleared pin falls through to corpus-match/catalog rather than
        # surfacing ``None`` (silent-off). Falls back to the sync read if the
        # service predates the async resolver.
        if hasattr(agent.llm_service, "aget_embedding_settings"):
            settings = await agent.llm_service.aget_embedding_settings()
        else:
            settings = agent.llm_service.get_embedding_settings()
        # #2289 — surface how many stored rows are on a different (or no)
        # embedding profile than the one currently resolved, so the UI can
        # render "Re-embed N memories". ``None`` when no profile resolves
        # (keyword-search fallback) or storage is unavailable. Only actionable
        # rows count toward the button; permanently-unembeddable rows are
        # surfaced separately as ``unembeddable_rows`` (#2426).
        await _apply_stale_embedding_counts(settings, agent)
        return settings
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedding settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting embedding settings.")


def _agent_raw_db(agent):
    """Return the raw ``AsyncDatabase`` handle behind the agent's storage.

    ``agent.storage`` is usually a PrivacyEnforcingStorage wrapper (no
    ``.db`` of its own) around the real AsyncStorage, which owns the
    ``.db`` handle. Reach through the wrapper's ``_storage`` when the outer
    object doesn't expose ``db`` directly. The handle is exposed only when
    the storage capability is bound to the same non-empty agent id as the
    request context; a cross-wired host must fail closed before corpus SQL.
    """
    storage = getattr(agent, "storage", None)
    agent_id = getattr(agent, "agent_id", None)
    storage_agent_id = getattr(storage, "agent_id", None) if storage else None
    db = getattr(storage, "db", None) if storage else None
    if db is None and storage is not None:
        inner = getattr(storage, "_storage", None)
        db = getattr(inner, "db", None) if inner else None
        if storage_agent_id is None and inner is not None:
            storage_agent_id = getattr(inner, "agent_id", None)
    if (
        not isinstance(agent_id, str)
        or not agent_id
        or not isinstance(storage_agent_id, str)
        or storage_agent_id != agent_id
    ):
        return None
    return db


async def _classify_stale_embedding_rows(agent, service=None) -> Optional[Dict[str, int]]:
    """Classify stored stale rows into actionable vs unembeddable (#2289/#2426).

    Scoped to the agent for every table; document chunks use their independent
    ownership ledger. Returns ``None`` when no
    embedding profile resolves or storage isn't available — the caller
    treats that as "re-embed action not applicable".

    A stale row is *unembeddable* when it has no recoverable source text
    (genuinely empty or undecryptable with the current keys) — the same rows
    a reindex run counts as ``skipped_empty`` and never rewrites. Those rows
    stay stale by the SQL predicate forever, so counting them toward the
    "Re-embed N memories" button keeps the UI advertising an action that can
    never clear them (#2426). The classifier reuses the reindexer's
    source-text extraction so the split matches the run outcome exactly.

    Returns ``{"stale", "actionable", "unembeddable"}``.

    ``service`` pins the count to a SPECIFIC route's embedding profile (#2372):
    the route-model echo passes the just-configured route's service so the
    stale-row count matches the echoed ``resolved_route`` instead of the globally
    active route's. Defaults to the active-route embedding service.
    """
    try:
        if service is None:
            service = agent.llm_service.get_embedding_service()
        if service is None:
            return None
        target = service.current_profile_id()
        if not target:
            return None
        db = _agent_raw_db(agent)
        if db is None:
            return None
        from kestrel_sovereign.storage.embedding_reindex import EmbeddingReindexer

        reindexer = EmbeddingReindexer(db, service, target)
        return await reindexer.classify_all_stale(
            agent_id=getattr(agent, "agent_id", None)
        )
    except Exception as e:  # pragma: no cover - defensive; never crash the GET
        logger.debug("stale_rows count failed: %s", e)
        return None


async def _apply_stale_embedding_counts(settings: Dict, agent, service=None) -> None:
    """Populate ``stale_rows`` (actionable) + ``unembeddable_rows`` on *settings*.

    Shared by the GET/POST settings and route-model echo paths so the button
    count everywhere reflects only rows a re-embed can actually clear, with the
    permanently-unembeddable rows surfaced separately (#2426). ``stale_rows`` is
    ``None`` (button hidden) when no profile resolves or storage is unavailable.
    """
    breakdown = await _classify_stale_embedding_rows(agent, service=service)
    if breakdown is None:
        settings["stale_rows"] = None
        settings["unembeddable_rows"] = 0
        return
    settings["stale_rows"] = breakdown["actionable"]
    settings["unembeddable_rows"] = breakdown["unembeddable"]


@router.api_route("/api/embedding/settings", methods=["POST", "PUT"])
async def set_embedding_settings(request: Request):
    """Set or clear the top-level ``embedding_route`` knob at runtime (#2263).

    Accepts JSON body ``{"embedding_route": "<vendor>:<route>"}`` to set, or
    ``{"embedding_route": null}`` (or ``""``/``"auto"``) to clear back to
    auto/follow-chat. Persists the value the same way runtime model selection
    persists (agent_metadata row), so it survives restart.
    """
    try:
        data = await request.json()
        if "embedding_route" not in data:
            raise HTTPException(
                status_code=400,
                detail="'embedding_route' field is required (use null to clear).",
            )
        route = data.get("embedding_route")
        # ``embedding_route`` must be a string selector or null (clear). A
        # non-string (e.g. 42, a list, a dict) would raise AttributeError deep
        # in set_embedding_route's ``route.strip()`` and surface as a 500 —
        # reject it here as plain bad input, consistent with the unknown-route
        # / no-embedding-support 400s below (#2286).
        if route is not None and not isinstance(route, str):
            raise HTTPException(
                status_code=400,
                detail="'embedding_route' must be a string or null.",
            )
        # Operator override for the dim-compatibility gate (#2417): allowed
        # mid-migration when the column is being re-sized + reindexed. Must be an
        # explicit JSON boolean — a truthy non-bool (``"false"``, ``0``, ``{}``,
        # …) must NOT silently bypass the gate.
        force = data.get("force", False)
        if not isinstance(force, bool):
            raise HTTPException(
                status_code=400,
                detail="'force' must be a JSON boolean (true/false).",
            )

        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")

        # Fold live embedding discovery into route capabilities BEFORE the sync
        # validator runs (#2338). ``set_embedding_route`` → ``_validate_embedding_route``
        # reads the static ``supports_embeddings`` capability; without this a
        # dynamically-discovered route (e.g. OpenRouter with no TOML pin) would
        # be rejected as "does not advertise embedding support" even though the
        # embeddings UI just listed it. Best-effort — a discovery hiccup must
        # not block clearing/setting a route that already has a static pin.
        if route not in (None, "", "auto", "none"):
            try:
                await agent.llm_service.reconcile_embedding_capabilities(use_cache=True)
            except Exception as e:  # pragma: no cover - never block the setter
                logger.debug(f"embedding capability reconcile skipped in setter: {e}")

        try:
            # Async set (#2326): after static validation, a cloud route is
            # live-probed with a canary embed so a listed-but-dead upstream
            # model (e.g. OpenRouter empty provider pool) is refused here rather
            # than silently 404'ing to keyword fallback on the next write.
            await agent.llm_service.aset_embedding_route(route, force=force)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        if hasattr(agent.llm_service, "aget_embedding_settings"):
            settings = await agent.llm_service.aget_embedding_settings()
        else:
            settings = agent.llm_service.get_embedding_settings()
        # Echo the authoritative stale-row count alongside the resolved settings
        # (#2338): changing the route can create stale memories, and the UI's
        # ``_renderReindex`` hides the "Re-embed N memories" button whenever
        # ``stale_rows`` is absent. Mirror the GET endpoint so the button state
        # is correct immediately after a route change without a second reload.
        await _apply_stale_embedding_counts(settings, agent)
        return {"success": True, **settings}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting embedding route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting embedding route.")


@router.api_route("/api/embedding/route-model", methods=["POST", "PUT"])
async def set_route_embedding_model(request: Request):
    """Pin (or clear) a route's embedding model at runtime (#2337).

    The embeddings UI's per-route model picker persists here instead of
    demanding a hand-edited ``embedding_model``/``embedding_dim`` under the route
    in kestrel.toml. Accepts JSON:

        {"route": "<vendor>:<route>",
         "embedding_model": "qwen/qwen3-embedding-8b",   # null/"" to clear
         "embedding_dim": 768}                            # optional

    Setting a cloud route's model runs a live canary probe (#2326) so a dead or
    misspelled upstream slug is refused with a 400 at configuration time rather
    than silently degrading to keyword search on the next write. Clearing
    restores the route's config/discovery default. The pin persists the same way
    the embedding_route knob does (agent_metadata) — not a new store. The
    response echoes the resolved embedding settings.
    """
    try:
        data = await request.json()
        route = data.get("route")
        if not route or not isinstance(route, str):
            raise HTTPException(
                status_code=400,
                detail="'route' field is required (a '<vendor>:<route>' selector).",
            )
        model = data.get("embedding_model")
        if model is not None and not isinstance(model, str):
            raise HTTPException(
                status_code=400,
                detail="'embedding_model' must be a string or null (to clear).",
            )
        dim = data.get("embedding_dim")
        if dim is not None:
            try:
                dim = int(dim)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="'embedding_dim' must be an integer or null.",
                )
        # Operator override for the dim-compatibility gate (#2417). Must be an
        # explicit JSON boolean — a truthy non-bool must NOT bypass the gate.
        force = data.get("force", False)
        if not isinstance(force, bool):
            raise HTTPException(
                status_code=400,
                detail="'force' must be a JSON boolean (true/false).",
            )

        from kestrel_sovereign.llm.service import EmbeddingSpaceConflictError

        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")
        if not hasattr(agent.llm_service, "aset_route_embedding_model"):
            raise HTTPException(
                status_code=501,
                detail="Per-route embedding model configuration is not supported.",
            )

        # Fold live embedding discovery into route capabilities first (#2338), so
        # a dynamically-discovered route (e.g. OpenRouter with no TOML pin) is
        # recognized when its model is being pinned. Best-effort.
        setting = model is not None and str(model).strip() != ""
        if setting:
            try:
                await agent.llm_service.reconcile_embedding_capabilities(use_cache=True)
            except Exception as e:  # pragma: no cover - never block the setter
                logger.debug(
                    f"embedding capability reconcile skipped in model setter: {e}"
                )

        try:
            # Live probe-on-save for cloud routes rejects a dead slug (#2337/#2326).
            await agent.llm_service.aset_route_embedding_model(
                route, model, dim, force=force
            )
        except EmbeddingSpaceConflictError as ce:
            # Pinning a model/dim that fragments a VERIFIED shared space is a
            # conflict, not a malformed request (#2440) — refuse at set time
            # instead of storing a phantom pin the space silently overrides.
            raise HTTPException(status_code=409, detail=str(ce))
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # #2418 — drain the persistence write scheduled by the pin/clear BEFORE
        # returning. Guided Universal setup issues these POSTs in sequence (pin a
        # member, then on a later failure clear it as rollback); each pin/clear
        # schedules an async persist of the full override map snapshotted at
        # schedule time. Without draining, an earlier "partial pin" write can
        # land AFTER the rollback "cleared" write and resurrect the partial state
        # on restart. Draining per request serializes the writes so the response
        # only returns once THIS request's override map is durably persisted.
        drain = getattr(agent.llm_service, "drain_preference_persistence", None)
        if callable(drain):
            try:
                await drain()
            except Exception as e:  # pragma: no cover - persistence is best-effort
                logger.debug(f"preference persistence drain failed: {e}")

        # Echo the settings for the PINNED route through the single #2372
        # resolver so the response reflects THIS route's own slug — not whatever
        # the globally-resolved embedding provider (embedding_route/chat) picks
        # (the cross-route echo the issue flagged).
        if hasattr(agent.llm_service, "aget_embedding_settings_for_route"):
            settings = await agent.llm_service.aget_embedding_settings_for_route(route)
        else:
            settings = agent.llm_service.get_embedding_settings()
        # Count stale rows against the ECHOED route's own embedding profile, not
        # the globally-active route's (#2372) — else a response that resolves
        # ``openrouter:api`` could report ``ollama:local``'s stale counts.
        route_service = None
        getter = getattr(agent.llm_service, "get_embedding_service_for_route", None)
        if callable(getter):
            try:
                route_service = getter(route)
            except Exception as e:  # pragma: no cover - defensive; fall back to active
                logger.debug(f"route embedding service build failed for {route}: {e}")
        await _apply_stale_embedding_counts(settings, agent, service=route_service)
        return {"success": True, **settings}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting route embedding model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error setting route embedding model."
        )


# In-memory registry of in-flight / recently-finished reindex jobs (#2336).
# Keyed by opaque ``job_id``. A UI POSTs to start a job and then polls
# ``GET /api/embedding/reindex/{job_id}`` until ``status`` is a terminal state
# (``done`` / ``partial`` / ``error``).
# Re-embedding is idempotent + resumable (see storage.embedding_reindex), so a
# lost job registry (process restart) loses no data — the operator just re-runs.
_REINDEX_JOBS: Dict[str, Dict] = {}
_REINDEX_JOBS_MAX = 32


def _reindex_job_public(job: Dict) -> Dict:
    """The subset of a job record that is safe to return to a client."""
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "target_profile": job["target_profile"],
        "embedding_dim": job["embedding_dim"],
        "tables": list(job["tables"]),
        "total_stale": job["total_stale"],
        "reembedded": dict(job["reembedded"]),
        "scanned": dict(job["scanned"]),
        # Full per-table stats so failed / skipped_empty / skipped_dim_mismatch
        # are visible instead of vanishing (#2360).
        "stats": {t: dict(s) for t, s in job.get("stats", {}).items()},
        "total_reembedded": job["total_reembedded"],
        "total_failed": job.get("total_failed", 0),
        "total_skipped_empty": job.get("total_skipped_empty", 0),
        "total_skipped_dim_mismatch": job.get("total_skipped_dim_mismatch", 0),
        # Rows that can never be embedded (no recoverable text). Lets the UI say
        # "N rows have no embeddable text" instead of surfacing a false error
        # for a corpus whose only stale rows are unembeddable (#2426).
        "unembeddable_rows": job.get("unembeddable_rows", 0),
        "error": job["error"],
        "updated_at": job["updated_at"],
    }


def _finalize_reindex_status(job: Dict) -> None:
    """Set the terminal ``status``/``error`` from the accumulated stats (#2360).

    ``done`` with ``total_reembedded == 0`` while rows were stale — or any
    ``failed`` rows — is NOT success: a dead/mis-resolved embedding service
    returns empty vectors for every row with no exception, which the old code
    reported as ``status: done, error: null``. Classify honestly:

    - ``error``   — stale rows existed but nothing was re-embedded for a reason
      the stats can't explain (total wipe-out; almost always a dead embedding
      service), or every attempt failed.
    - ``partial`` — some rows re-embedded but some failed / were skipped for a
      dimension mismatch.
    - ``done``    — every stale row is accounted for: re-embedded, or genuinely
      unembeddable (``skipped_empty`` — no recoverable text), with no failures.

    A corpus whose only stale rows are ``skipped_empty`` (no recoverable
    text — genuinely empty or undecryptable with the current keys) is ``done``,
    not ``error``: the stats themselves prove there is nothing embeddable
    (``failed == 0`` and every non-reembedded stale row is skipped_empty). Those
    rows are surfaced as ``unembeddable_rows`` so the UI can render an
    explanation instead of a scary error (#2426).
    """
    total_reembedded = job["total_reembedded"]
    total_failed = job.get("total_failed", 0)
    total_dim_mismatch = job.get("total_skipped_dim_mismatch", 0)
    total_skipped_empty = job.get("total_skipped_empty", 0)
    total_stale = job.get("total_stale", 0)

    # Rows with no recoverable text will never be embeddable — a permanent,
    # non-actionable condition, not a run failure. Surface the count so the UI
    # can explain them rather than reading them as an error (#2426).
    job["unembeddable_rows"] = total_skipped_empty

    # The part of the stale set left unexplained once re-embedded rows and the
    # permanently-unembeddable rows are set aside. A dead/mis-resolved embedding
    # service leaves this positive (rows had text but produced no usable vector).
    unexplained = total_stale - total_reembedded - total_skipped_empty

    if total_failed == 0 and total_dim_mismatch == 0 and unexplained <= 0:
        # Nothing failed and every stale row is accounted for. This covers the
        # all-skipped_empty case (total_reembedded == 0 but every stale row is
        # genuinely unembeddable) — success, not the #2360 silent-failure
        # signature.
        job["status"] = "done"
        job["error"] = None
        return

    if total_stale > 0 and total_reembedded == 0 and unexplained > 0:
        job["status"] = "error"
        job["error"] = (
            f"re-embedded 0 of {total_stale} stale row(s) "
            f"({total_failed} failed, {total_dim_mismatch} dim-mismatch, "
            f"{total_skipped_empty} unembeddable). "
            "The resolved embedding service returned no usable vectors — it is "
            "likely dead or mis-configured; check the active embedding route."
        )
        return

    job["status"] = "partial"
    job["error"] = (
        f"re-embedded {total_reembedded} row(s) but {total_failed} failed "
        f"and {total_dim_mismatch} were skipped for a dimension mismatch."
    )


def _accumulate_reindex_stats(job: Dict) -> None:
    """Recompute the job's roll-up totals from its per-table ``stats``."""
    stats = job.get("stats", {})
    job["total_reembedded"] = sum(job["reembedded"].values())
    job["total_failed"] = sum(int(s.get("failed", 0)) for s in stats.values())
    job["total_skipped_empty"] = sum(
        int(s.get("skipped_empty", 0)) for s in stats.values()
    )
    job["total_skipped_dim_mismatch"] = sum(
        int(s.get("skipped_dim_mismatch", 0)) for s in stats.values()
    )


async def _run_reindex_job(job: Dict, reindexer, tables, agent_id) -> None:
    """Background worker: re-embed each table, streaming progress into *job*."""
    try:
        for tname in tables:
            def _progress(stats, _tname=tname):
                job["reembedded"][_tname] = stats.reembedded
                job["scanned"][_tname] = stats.scanned
                job["stats"][_tname] = stats.as_dict()
                _accumulate_reindex_stats(job)
                job["updated_at"] = time.time()

            stats = await reindexer.reindex_table(
                tname, agent_id=agent_id, progress=_progress
            )
            job["reembedded"][tname] = stats.reembedded
            job["scanned"][tname] = stats.scanned
            job["stats"][tname] = stats.as_dict()
        _accumulate_reindex_stats(job)
        _finalize_reindex_status(job)
    except Exception as e:  # pragma: no cover - defensive; surfaced via status
        logger.error("reindex job %s failed: %s", job["job_id"], e, exc_info=True)
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        job["updated_at"] = time.time()


def _prune_reindex_jobs() -> None:
    """Bound the registry: drop the oldest finished jobs past the cap."""
    if len(_REINDEX_JOBS) <= _REINDEX_JOBS_MAX:
        return
    finished = sorted(
        (
            j
            for j in _REINDEX_JOBS.values()
            if j["status"] in ("done", "error", "partial")
        ),
        key=lambda j: j["updated_at"],
    )
    while len(_REINDEX_JOBS) > _REINDEX_JOBS_MAX and finished:
        _REINDEX_JOBS.pop(finished.pop(0)["job_id"], None)


async def _resolve_reindex_target(agent, db=None, agent_id=None):
    """Resolve ``(embedding_service, target_profile_id, target_dim, column_dim)``.

    Resolves the embedding service **fresh** from the currently-resolved route
    so a route change made after the agent booted (or via the settings API) is
    honoured — the #2360 live-dogfood failure was a stale/cached embedding
    service that no longer matched the persisted route, producing scanned-N /
    reembedded-0 with no error. Two things guarantee freshness, mirroring the
    CLI's ``_reindex`` path:

    1. Apply the agent's persisted runtime ``embedding_route`` (#2263) to the
       live ``llm_service`` before resolving, so the endpoint targets the same
       profile the live agent (and the CLI) resolve — not the config default.
    2. Invalidate the per-instance embedding discovery cache so a re-pinned /
       changed model is reflected instead of a stale resolution.

    Raises :class:`HTTPException` (409) with the same refusal reasons the CLI
    prints when reindexing can't proceed: no embedding provider resolves,
    ``embedding_route = "none"``, the service can't describe itself, or the
    resolved embedding dimension doesn't match the vector-column width.
    """
    from kestrel_sovereign import cli_embeddings

    llm_service = agent.llm_service

    # (1) Load the persisted embedding_route so we resolve the profile the
    # live agent actually uses (route changes since boot included). Applying it
    # is deferred until AFTER capabilities are reconciled (step 3): a cleared
    # per-route pin drops ``supports_embeddings``, and ``set_embedding_route``'s
    # sync validator (``_validate_embedding_route``) would reject the route as
    # "does not advertise embedding support" if it ran against the stale flag —
    # 409'ing before the fix could re-advertise capability (#2372).
    found, route = False, None
    if db is not None:
        try:
            found, route = await cli_embeddings._load_persisted_embedding_route(
                db, agent_id
            )
        except Exception as exc:
            logger.debug("could not load persisted embedding_route: %s", exc)
            found, route = False, None

    # (2) Bypass whatever the embedding resolver cached so a changed route /
    # re-pinned model is re-discovered instead of served stale.
    if hasattr(llm_service, "_embedding_discovery_cache"):
        try:
            llm_service._embedding_discovery_cache = None
        except Exception:  # pragma: no cover - defensive
            pass

    # (2b) Re-apply the persisted per-route ``embedding_model`` pins (#2337)
    # BEFORE reconcile/resolve, exactly as the agent boot path
    # (``ModelPreferenceMixin._load_route_embedding_models``) and the CLI's
    # ``_apply_persisted_embedding_config`` do. This is the #2423 fix: a
    # multi-agent host holds a per-agent ``LLMService`` instance, and the
    # instance the reindex resolves from is not guaranteed to be the one whose
    # in-memory pin the route-model POST mutated. Without re-seeding the pin from
    # the DB (the authoritative persisted state the settings GET honours),
    # step (3)'s cache-invalidated reconcile re-resolves the route to its
    # discovery/config default (e.g. ``nomic-embed-text``) and the reindex
    # stamps rows with a profile that DISAGREES with the settings surface's
    # pinned model (``qwen3-embedding:8b``) — the exact divergence #2423 hit
    # live. Each pin re-advertises embedding support for its exact route and is
    # folded in as ``is_pinned`` so it wins the resolver order verbatim.
    if db is not None and hasattr(llm_service, "set_route_embedding_model"):
        try:
            overrides = await cli_embeddings._load_persisted_route_embedding_models(
                db, agent_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not load persisted embedding_model pins: %s", exc)
            overrides = {}
        # Sync runtime overrides to the DB state FIRST (#2423): the DB is the
        # authoritative persisted state the settings GET honours. A pure additive
        # re-seed would leave a stale in-memory pin active on a per-agent
        # ``LLMService`` instance after the operator CLEARED the pin (DB now
        # ``{}``) — the resolver would keep stamping rows with the old pinned
        # profile instead of the cleared/default model. Drop every in-memory
        # override no longer present in the DB map before applying the survivors.
        if hasattr(llm_service, "get_route_embedding_model_overrides"):
            try:
                runtime_overrides = llm_service.get_route_embedding_model_overrides()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("could not read runtime embedding_model pins: %s", exc)
                runtime_overrides = {}
            for stale_route in runtime_overrides:
                if stale_route in overrides:
                    continue
                try:
                    llm_service.set_route_embedding_model(
                        stale_route, None, persist=False
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        "could not clear stale embedding_model pin for %s: %s",
                        stale_route,
                        exc,
                    )
        for pin_route, spec in overrides.items():
            if not isinstance(spec, dict):
                continue
            model = spec.get("model")
            if not model:
                continue
            try:
                llm_service.set_route_embedding_model(
                    pin_route, model, spec.get("dim"), persist=False
                )
            except Exception as exc:  # pragma: no cover - defensive; skip bad pin
                logger.debug(
                    "skipping persisted embedding_model pin for %s: %s",
                    pin_route,
                    exc,
                )

    # (3) Re-advertise capability across every discovering route BEFORE applying
    # the persisted route: a cleared per-route pin drops ``supports_embeddings``,
    # and both ``set_embedding_route``'s validator and
    # ``resolve_embedding_provider`` gate on that sync flag — so without this the
    # cleared-pin route is rejected (409) or resolves to None here (the wrong /
    # stale profile the settings GET now avoids) rather than the corpus/catalog
    # default. Mirrors the settings POST + ``aget_embedding_settings`` ordering so
    # the reindex target and the settings GET agree in the cleared-pin state
    # (#2372).
    if hasattr(llm_service, "reconcile_embedding_capabilities"):
        try:
            await llm_service.reconcile_embedding_capabilities(use_cache=True)
        except Exception as exc:  # pragma: no cover - defensive; keep prior behaviour
            logger.debug("reindex embedding capability reconcile skipped: %s", exc)

    # (4) Now apply the persisted route — capability has been re-advertised, so a
    # cleared-pin route that legitimately discovers embedding models validates
    # instead of 409'ing on the stale flag. If it still can't be applied, REFUSE:
    # continuing would silently reindex production rows into whatever route was
    # already active (the wrong embedding profile). Mirrors the CLI, which exits
    # non-zero here (cli_embeddings._reindex ~line 566) rather than falling
    # through to the previously-active route.
    if found:
        try:
            llm_service.set_embedding_route(route, persist=False)
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"the persisted embedding_route {route!r} is no longer "
                    f"valid ({exc}); fix the embedding route (via the "
                    "settings API/UI or config) before reindexing."
                ),
            )

    # (5) Resolve the active route's (model, dim) through the SINGLE #2372
    # resolver so the reindex target matches the settings GET and the route-model
    # echo — the #2372 failure was three surfaces resolving three different
    # profiles (including a stale one) for the same cleared-pin state.
    if hasattr(llm_service, "resolve_route_embedding_model"):
        try:
            active = llm_service.resolve_embedding_provider()
            if active is not None:
                await llm_service.resolve_route_embedding_model(active)
        except Exception as exc:  # pragma: no cover - defensive; keep prior behaviour
            logger.debug("reindex active-route embedding resolve skipped: %s", exc)

    err, embedding_service, target = cli_embeddings._resolve_target(llm_service)
    if err is not None:
        raise HTTPException(status_code=409, detail=err)

    target_dim = getattr(embedding_service, "embedding_dim", None)
    column_dim = cli_embeddings._resolve_column_dim()
    if target_dim and column_dim and int(target_dim) != int(column_dim):
        raise HTTPException(
            status_code=409,
            detail=(
                f"resolved embedding dimension ({target_dim}) does not match the "
                f"vector-column width ({column_dim}); re-embedding at the new "
                "dimension requires a column migration first (set "
                f"KESTREL_EMBEDDING_DIM={target_dim}, drop + recreate the "
                "embedding_vec columns, restart, then re-embed)."
            ),
        )
    return embedding_service, target, target_dim, column_dim


@router.post("/api/embedding/reindex")
async def reindex_embeddings(request: Request):
    """Re-embed stored vectors to the resolved embedding profile from the UI (#2336).

    Wraps the same checkpointed/resumable core the CLI's ``kestrel embeddings
    reindex`` uses. JSON body:

    - ``dry_run`` (default ``true``): report per-table stale counts only — no
      writes. Mirrors the CLI's dry-run scope.
    - ``dry_run: false``: execute. Small/empty corpora complete inline; a
      non-empty corpus is re-embedded in a background job and the response
      carries a ``job_id`` the caller polls via
      ``GET /api/embedding/reindex/{job_id}`` so a big corpus never blocks the
      request for minutes.
    - ``tables``: optional list restricting the sweep (default: all three
      embedding-bearing tables).

    Agent scoping comes from a storage capability bound to the request agent;
    document chunks are filtered through their ownership ledger, so there is
    no ``KESTREL_DB_PATH`` ambiguity (#2327). Refuses with 409 when no embedding
    provider resolves, ``embedding_route = "none"``, or on a dimension mismatch.
    """
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        dry_run = bool(data.get("dry_run", True))
        tables_req = data.get("tables")

        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")

        from kestrel_sovereign.storage.embedding_reindex import (
            REINDEX_TABLES,
            EmbeddingReindexer,
        )

        if tables_req is not None:
            if not isinstance(tables_req, list) or not all(
                isinstance(t, str) for t in tables_req
            ):
                raise HTTPException(
                    status_code=400,
                    detail="'tables' must be a list of table names.",
                )
            unknown = [t for t in tables_req if t not in REINDEX_TABLES]
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"unknown table(s): {', '.join(unknown)}. "
                        f"Valid: {', '.join(REINDEX_TABLES)}."
                    ),
                )
            tables = tuple(tables_req) or REINDEX_TABLES
        else:
            tables = REINDEX_TABLES

        db = _agent_raw_db(agent)
        if db is None:
            raise HTTPException(status_code=503, detail="Storage not available.")

        agent_id = getattr(agent, "agent_id", None)

        embedding_service, target, target_dim, column_dim = (
            await _resolve_reindex_target(agent, db=db, agent_id=agent_id)
        )

        reindexer = EmbeddingReindexer(
            db, embedding_service, target, column_dim=column_dim
        )

        counts = await reindexer.count_all_stale(agent_id=agent_id, tables=tables)
        total_stale = sum(counts.values())

        if dry_run:
            # Split the stale set so the confirm dialog counts only rows a
            # re-embed can actually clear (#2426). ``total_stale`` stays the full
            # count — the executed job's finalizer classifies against it — but the
            # UI advertises ``actionable_stale`` to match the settings button.
            breakdown = await reindexer.classify_all_stale(
                agent_id=agent_id, tables=tables
            )
            return {
                "dry_run": True,
                "target_profile": target,
                "embedding_dim": target_dim,
                "stale_rows": counts,
                "total_stale": total_stale,
                "actionable_stale": breakdown["actionable"],
                "unembeddable_rows": breakdown["unembeddable"],
            }

        # Execute. Nothing stale → report done inline (no job needed).
        if not total_stale:
            return {
                "dry_run": False,
                "status": "done",
                "target_profile": target,
                "embedding_dim": target_dim,
                "stale_rows": counts,
                "total_stale": 0,
                "reembedded": {t: 0 for t in tables},
                "total_reembedded": 0,
            }

        # Non-empty corpus: run in the background so the request never blocks
        # for minutes. The caller polls GET /api/embedding/reindex/{job_id}.
        job_id = uuid.uuid4().hex[:16]
        job = {
            "job_id": job_id,
            # Owner scoping (#2336): the job is readable only by the agent
            # context that created it. Never surfaced via _reindex_job_public.
            "owner_agent_id": agent_id,
            "status": "running",
            "target_profile": target,
            "embedding_dim": target_dim,
            "tables": list(tables),
            "total_stale": total_stale,
            "reembedded": {t: 0 for t in tables},
            "scanned": {t: 0 for t in tables},
            # Full per-table ReindexStats (#2360): failed / skipped_empty /
            # skipped_dim_mismatch were previously computed but never surfaced,
            # so 63 rows could vanish into invisible buckets while the job
            # reported "done, error: null".
            "stats": {t: {} for t in tables},
            "total_reembedded": 0,
            "total_failed": 0,
            "total_skipped_empty": 0,
            "total_skipped_dim_mismatch": 0,
            "unembeddable_rows": 0,
            "error": None,
            "updated_at": time.time(),
        }
        _REINDEX_JOBS[job_id] = job
        _prune_reindex_jobs()
        asyncio.create_task(_run_reindex_job(job, reindexer, tables, agent_id))
        return {"dry_run": False, **_reindex_job_public(job)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting embedding reindex: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error starting embedding reindex.")


@router.get("/api/embedding/reindex/{job_id}")
async def get_reindex_job(job_id: str, request: Request):
    """Return progress for a background reindex job started via POST (#2336).

    Scoped to the creating agent context (#2336): a job is readable only by the
    agent that started it. Any other routed agent gets 404 (not 403 — the job's
    existence is not disclosed across agents).
    """
    try:
        agent = get_agent(request)  # auth / agent-context gate
        job = _REINDEX_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="reindex job not found.")
        requester_agent_id = getattr(agent, "agent_id", None)
        if job.get("owner_agent_id") != requester_agent_id:
            # Do not leak existence of another agent's job.
            raise HTTPException(status_code=404, detail="reindex job not found.")
        return _reindex_job_public(job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading reindex job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error reading reindex job.")


@router.api_route("/api/embedding/space/verify", methods=["POST"])
async def verify_embedding_space(request: Request):
    """Run the shared-space parity probe and apply passing pins (#2290).

    Embeds K canary texts through each declared shared space's member routes and
    requires pairwise cosine ``>= parity_threshold`` before the pin's
    model-identity ``space_id`` is applied. Optional JSON body
    ``{"name": "<pin>"}`` limits to one pin. Measured drift is recorded on the
    pinned space's ``embedding_profiles`` row when storage is available.
    """
    try:
        agent = get_agent(request)
        if not hasattr(agent, "llm_service") or not agent.llm_service:
            raise HTTPException(status_code=503, detail="LLM service not available.")
        try:
            data = await request.json()
        except Exception:
            data = {}
        pin_name = data.get("name") if isinstance(data, dict) else None

        # Reach the raw DB handle (through the privacy wrapper) so measured
        # drift can be recorded; best-effort, verification runs regardless.
        db = _agent_raw_db(agent)

        results = await agent.llm_service.verify_embedding_space_parity(
            pin_name, record_to=db
        )
        return {
            "success": True,
            "results": {name: r.to_dict() for name, r in results.items()},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying embedding space: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error verifying embedding space.")


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
    except HTTPException:
        raise
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

        # Extract user_passphrase for USER_BYOK agents
        user_passphrase = data.get("user_passphrase")

        assistant_text = await agent.process_input(
            user_input,
            model_override=model_override,
            caller=get_caller(request),
            user_passphrase=user_passphrase,
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
    except HTTPException:
        # Preserve the original status code (notably 503 from get_agent
        # when no agent is bound — multi-agent mode requires the
        # /api/agents/<name>/v1/chat/completions form). The pre-fix
        # ``except Exception`` branch swallowed this and emitted a
        # misleading 500 with "Internal error in chat completions",
        # which read like a server bug to clients (Open WebUI in
        # particular) when it was actually a routing problem.
        raise
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error in chat completions.")
