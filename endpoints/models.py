"""Model, wallet, and IPFS status endpoints."""
from fastapi import APIRouter, HTTPException, Request, Query
from typing import Optional, List
import aiohttp
import time
import logging

from kestrel_sovereign.llm.model_metadata import ModelCategory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/api/identity")
async def get_identity(request: Request):
    """Get agent identity information including avatar."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        storage = agent.storage

        agent_node = await storage.get_node(agent.agent_id) if storage else None

        # Get avatar_hash from agent node properties (like constitution_hash)
        avatar_hash = agent_node.properties.get("avatar_hash") if agent_node else None
        avatar_url = f"/api/files/{avatar_hash}" if avatar_hash else None

        return {
            "did": agent.agent_id,
            "node_type": agent_node.node_type if agent_node else "agent",
            "created_at": agent_node.properties.get("created_at") if agent_node else None,
            "constitution_hash": agent_node.properties.get("constitution_hash") if agent_node else None,
            "avatar_hash": avatar_hash,
            "avatar_url": avatar_url,
        }
    except Exception as e:
        logger.error(f"Error getting identity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving identity.")


@router.get("/api/constitution")
async def get_constitution(request: Request):
    """Get the agent's constitution."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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

    local_api_url = "http://localhost:8889/api/v0"
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
                        data = await resp.json()
                        pins = data.get("Keys", {})
                        status["pinned_content"] = [
                            {"cid": cid, "type": info.get("Type")}
                            for cid, info in list(pins.items())[:20]
                        ]
    except Exception as e:
        status["local_node"]["error"] = str(e)

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
                gw_status["error"] = str(e)
            status["gateways"].append(gw_status)

    if hasattr(request.app.state, 'agent') and request.app.state.agent:
        agent = request.app.state.agent
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

    return status


@router.get("/api/wallet")
async def get_wallet(request: Request):
    """Get wallet balance and transaction history."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent

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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        body = await request.json()
        provider = body.get("provider", "").lower().strip()
        api_key = body.get("api_key", "").strip()
        quota_limit = body.get("quota_limit")  # Optional

        if not provider:
            raise HTTPException(status_code=400, detail="Provider is required")
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")

        agent = request.app.state.agent
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
        raise HTTPException(status_code=500, detail=f"Error adding key: {str(e)}")


@router.delete("/api/keys/{provider}")
async def delete_key(request: Request, provider: str):
    """Delete an API key for a provider."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
        raise HTTPException(status_code=500, detail=f"Error deleting key: {str(e)}")


@router.patch("/api/keys/{provider}")
async def update_key(request: Request, provider: str):
    """Update an API key's settings (quota, active status)."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        body = await request.json()
        quota_limit = body.get("quota_limit")
        is_active = body.get("is_active")

        agent = request.app.state.agent
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

        # Build update query
        updates = []
        params = []
        if quota_limit is not None:
            updates.append("quota_limit = ?")
            params.append(quota_limit)
        if is_active is not None:
            updates.append("is_active = ?")
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
        raise HTTPException(status_code=500, detail=f"Error updating key: {str(e)}")


@router.get("/api/keys/{provider}/usage")
async def get_key_usage(request: Request, provider: str, days: int = Query(30)):
    """Get usage history for a specific API key."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
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

        # Group by provider
        by_provider = {}
        for model in models:
            if model.provider not in by_provider:
                by_provider[model.provider] = []
            by_provider[model.provider].append(model.to_dict())

        # Get featured models
        featured = [m.to_dict() for m in models if m.is_featured]

        # Get default model
        default_model = None
        if hasattr(agent, 'llm_service') and agent.llm_service:
            default_model = agent.llm_service.default_model

        return {
            "by_provider": by_provider,
            "featured": featured,
            "all": models_data,
            "default": default_model,
            "count": len(models_data),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing models.")


@router.get("/api/model/current")
async def get_current_model(request: Request):
    """Get the currently active model for UI sync.
    
    Returns the model/provider from mandate preference if set,
    otherwise falls back to the first provider's default model.
    """
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")

    try:
        agent = request.app.state.agent
        provider = None
        model_name = None

        if hasattr(agent, 'llm_service') and agent.llm_service:
            llm_service = agent.llm_service
            
            # First check mandate preference (set via !model-set or UI)
            pref = llm_service.get_model_preference()
            model_name = pref.get('model')
            provider = pref.get('provider')

            # If no mandate preference, use the first provider (what actually gets used)
            if not model_name and llm_service.providers:
                first_provider = llm_service.providers[0]
                provider = first_provider.get('name')
                model_name = first_provider.get('model')

        if model_name:
            full_model = f"{provider}/{model_name}" if provider else model_name
        else:
            full_model = None

        return {
            "model": full_model,
            "provider": provider,
            "model_name": model_name
        }
    except Exception as e:
        logger.error(f"Error getting current model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error getting current model.")


@router.get("/v1/models")
def list_models_v1(request: Request):
    """Return a minimal models list for OpenAI-compatible clients."""
    try:
        agent = request.app.state.agent if hasattr(request.app.state, 'agent') else None
        model_id = "kestrel-local"
        if agent and hasattr(agent, 'llm_service') and agent.llm_service.providers:
            prov = agent.llm_service.providers[0]
            model_id = prov.get('model') or model_id
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"object": "list", "data": [{"id": "kestrel-local", "object": "model"}]}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Minimal Chat Completions-compatible endpoint."""
    if not hasattr(request.app.state, 'agent') or not request.app.state.agent:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    try:
        data = await request.json()
        messages = data.get("messages", [])
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        user_input = last_user.get("content") if last_user else ""
        if not user_input:
            user_input = "\n".join([m.get("content", "") for m in messages])

        agent = request.app.state.agent
        assistant_text = await agent.process_input(user_input)

        resp = {
            "id": f"chatcmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", "kestrel-local"),
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
