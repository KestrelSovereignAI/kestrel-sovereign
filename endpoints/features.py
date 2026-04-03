"""Feature Store API — catalog, install, enable, disable, configure features."""

import asyncio
import logging
import subprocess
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from endpoints.agent_helpers import get_agent
from kestrel_sovereign.feature_registry import (
    FeaturePackageInfo,
    FeatureStatus,
    get_all_skills,
    get_package_for_feature,
    get_registry,
    get_skills_for_package,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["features"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ConfigUpdateRequest(BaseModel):
    """Partial config update for a feature."""

    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feature_package_to_dict(info: FeaturePackageInfo) -> Dict[str, Any]:
    """Serialize a FeaturePackageInfo to a JSON-safe dict."""
    d = asdict(info)
    d["status"] = info.status.value
    d["skills"] = [asdict(s) for s in info.skills]
    return d


def _get_enabled_class_names(agent) -> set:
    """Return the set of Feature class names currently enabled on *agent*."""
    return set(agent.features.keys()) if hasattr(agent, "features") else set()


def _get_feature_or_404(agent, name: str):
    """Look up a loaded Feature by class name, raising 404 if not found."""
    features = getattr(agent, "features", {})
    feature = features.get(name)
    if feature is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not loaded on this agent")
    return feature


def _tool_to_dict(tool) -> Dict[str, Any]:
    """Serialize an AgentTool to a JSON-safe dict."""
    schema = tool.schema
    return {
        "name": schema.name,
        "description": schema.description,
        "category": schema.category.value if hasattr(schema.category, "value") else str(schema.category),
        "parameters": schema.parameters,
        "command_prefix": getattr(schema, "command_prefix", None),
    }


# ---------------------------------------------------------------------------
# Feature catalog endpoints
# ---------------------------------------------------------------------------


@router.get("/api/features")
async def list_features(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status: available, installed, enabled, disabled"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
) -> Dict[str, Any]:
    """
    Full feature catalog with status per feature.

    Returns all known features from the registry with their runtime status
    (available / installed / enabled / disabled).
    """
    agent = get_agent(request)
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    results = []
    for info in registry.values():
        if status and info.status.value != status:
            continue
        if tag and tag not in info.tags:
            continue
        results.append(_feature_package_to_dict(info))

    return {"features": results, "count": len(results)}


@router.get("/api/features/installed")
async def list_installed_features(request: Request) -> Dict[str, Any]:
    """
    Only installed/enabled features with their tools.

    Returns features that are currently loaded on the agent, along with the
    tools each feature exposes.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    results = []
    for name, feature in features.items():
        tools = feature.get_tools()
        pkg = get_package_for_feature(name)
        entry: Dict[str, Any] = {
            "name": name,
            "tool_name": feature.tool_name,
            "description": feature.tool_description,
            "tools": [_tool_to_dict(t) for t in tools],
        }
        if pkg:
            entry["package"] = pkg.package
            entry["tags"] = pkg.tags
            entry["icon"] = pkg.icon
            entry["core"] = pkg.core
        results.append(entry)

    return {"features": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Single-feature detail & lifecycle
# ---------------------------------------------------------------------------


@router.get("/api/features/{name}")
async def get_feature_detail(request: Request, name: str) -> Dict[str, Any]:
    """
    Detail view for a feature.

    Returns description, tools provided, hooks registered, config schema,
    package info, and install instructions.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # Try loaded feature first
    feature = features.get(name)
    if feature is not None:
        tools = feature.get_tools()
        hooks = feature.get_hooks()
        pkg = get_package_for_feature(name)

        detail: Dict[str, Any] = {
            "name": name,
            "tool_name": feature.tool_name,
            "description": feature.tool_description,
            "status": "enabled",
            "tools": [_tool_to_dict(t) for t in tools],
            "hooks": [{"name": h.name, "event": h.event} for h in hooks] if hooks else [],
            "config_schema": feature.config_schema,
        }
        if pkg:
            detail["package"] = pkg.package
            detail["git"] = pkg.git
            detail["tags"] = pkg.tags
            detail["icon"] = pkg.icon
            detail["core"] = pkg.core
            detail["skills"] = [asdict(s) for s in pkg.skills]
            detail["install_instructions"] = f"pip install {pkg.package}" if not pkg.core else None
        return detail

    # Not loaded — look up in registry
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    for info in registry.values():
        if name in info.features or info.name == name:
            d = _feature_package_to_dict(info)
            d["install_instructions"] = f"pip install {info.package}" if not info.core else None
            return d

    raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry or loaded features")


@router.post("/api/features/{name}/install")
async def install_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Install a feature package via pip.

    Requires a sovereign agent — governed agents cannot install packages.
    """
    agent = get_agent(request)

    # Look up package info from registry
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    pkg_info = None
    for info in registry.values():
        if name in info.features or info.name == name:
            pkg_info = info
            break

    if pkg_info is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry")

    if pkg_info.core:
        raise HTTPException(status_code=400, detail=f"Feature '{name}' is a core feature and is already installed")

    if pkg_info.status == FeatureStatus.ENABLED:
        raise HTTPException(status_code=400, detail=f"Feature '{name}' is already enabled")

    # Install via pip in a subprocess
    package_spec = pkg_info.package
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "install", package_spec],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error(f"pip install failed for {package_spec}: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Installation failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Installation timed out")

    return {
        "status": "installed",
        "package": package_spec,
        "message": f"Package '{package_spec}' installed. Restart the agent to load the feature.",
    }


@router.post("/api/features/{name}/enable")
async def enable_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Enable a loaded feature.

    Calls the feature's on_enable() lifecycle method.
    """
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    await feature.on_enable()
    return {"name": name, "status": "enabled"}


@router.post("/api/features/{name}/disable")
async def disable_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Disable a loaded feature.

    Calls the feature's on_disable() lifecycle method.
    """
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    await feature.on_disable()
    return {"name": name, "status": "disabled"}


@router.post("/api/features/{name}/remove")
async def remove_feature(request: Request, name: str) -> Dict[str, Any]:
    """
    Uninstall a feature package.

    Calls on_remove() for cleanup, then pip uninstalls the package.
    Requires a sovereign agent — governed agents cannot remove packages.
    """
    agent = get_agent(request)

    # Check if feature is loaded — call on_remove if so
    features = getattr(agent, "features", {})
    feature = features.get(name)
    if feature is not None:
        await feature.on_remove()

    # Look up package info
    pkg_info = get_package_for_feature(name)
    if pkg_info is None:
        raise HTTPException(status_code=404, detail=f"Feature '{name}' not found in registry")

    if pkg_info.core:
        raise HTTPException(status_code=400, detail="Cannot remove a core feature")

    package_spec = pkg_info.package
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "uninstall", "-y", package_spec],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"pip uninstall failed for {package_spec}: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Removal failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Removal timed out")

    return {
        "status": "removed",
        "package": package_spec,
        "message": f"Package '{package_spec}' uninstalled. Restart the agent to fully unload.",
    }


# ---------------------------------------------------------------------------
# Feature configuration endpoints
# ---------------------------------------------------------------------------


@router.get("/api/features/{name}/config")
async def get_feature_config(request: Request, name: str) -> Dict[str, Any]:
    """Current configuration for a loaded feature."""
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    config = await feature.get_config()
    return {
        "name": name,
        "config": config,
        "config_schema": feature.config_schema,
    }


@router.patch("/api/features/{name}/config")
async def update_feature_config(
    request: Request,
    name: str,
    body: ConfigUpdateRequest,
) -> Dict[str, Any]:
    """
    Update feature configuration.

    Validates against the feature's config_schema if available.
    """
    agent = get_agent(request)
    feature = _get_feature_or_404(agent, name)

    schema = feature.config_schema
    if schema is not None:
        _validate_config(body.config, schema)

    await feature.set_config(body.config)
    updated = await feature.get_config()

    return {
        "name": name,
        "config": updated,
        "message": "Configuration updated",
    }


def _validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """
    Basic JSON Schema validation for feature config.

    Checks required fields and type constraints from the schema.
    Raises HTTPException(422) on validation failure.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name in required:
        if field_name not in config:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required config field: '{field_name}'",
            )

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for key, value in config.items():
        if key in properties:
            prop_schema = properties[key]
            expected_type_name = prop_schema.get("type")
            if expected_type_name and expected_type_name in type_map:
                expected = type_map[expected_type_name]
                if not isinstance(value, expected):
                    raise HTTPException(
                        status_code=422,
                        detail=f"Config field '{key}' must be {expected_type_name}, got {type(value).__name__}",
                    )


# ---------------------------------------------------------------------------
# Skill discovery endpoints
# ---------------------------------------------------------------------------


@router.get("/api/features/{name}/skills")
async def get_feature_skills(request: Request, name: str) -> Dict[str, Any]:
    """
    Skills provided by a specific feature.

    Returns tools from the loaded feature instance if available, otherwise
    falls back to static skill declarations from the registry.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # If feature is loaded, return live tools
    feature = features.get(name)
    if feature is not None:
        tools = feature.get_tools()
        return {
            "feature": name,
            "skills": [_tool_to_dict(t) for t in tools],
            "count": len(tools),
            "source": "live",
        }

    # Fall back to registry declarations
    # Try by class name first, then by package short name
    pkg = get_package_for_feature(name)
    if pkg is not None:
        skills = [asdict(s) for s in pkg.skills]
        return {
            "feature": name,
            "skills": skills,
            "count": len(skills),
            "source": "registry",
        }

    # Try as package short name
    skills = get_skills_for_package(name)
    if skills:
        return {
            "feature": name,
            "skills": [asdict(s) for s in skills],
            "count": len(skills),
            "source": "registry",
        }

    raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")


@router.get("/api/skills")
async def list_all_skills(
    request: Request,
    tag: Optional[str] = Query(None, description="Filter by skill tag"),
    category: Optional[str] = Query(None, description="Filter by skill category"),
) -> Dict[str, Any]:
    """
    All skills across all features, searchable by tag/category.

    Merges live tools from loaded features with static registry declarations
    for unloaded features.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})
    seen_skill_names: set = set()
    results: List[Dict[str, Any]] = []

    # Live skills from loaded features
    for feature_name, feature in features.items():
        for tool in feature.get_tools():
            skill = _tool_to_dict(tool)
            skill["feature"] = feature_name
            skill["source"] = "live"

            if tag and tag not in (skill.get("category", ""),):
                # Check tool parameters or skip — tags aren't on live tools
                pass
            if category and skill.get("category", "") != category:
                continue

            results.append(skill)
            seen_skill_names.add(skill["name"])

    # Static registry skills for unloaded features
    enabled = _get_enabled_class_names(agent)
    registry = get_registry(enabled_class_names=enabled)

    for info in registry.values():
        # Skip features already represented by live tools
        if set(info.features) & enabled:
            continue
        for skill_info in info.skills:
            if skill_info.name in seen_skill_names:
                continue
            if tag and tag not in skill_info.tags:
                continue
            if category and skill_info.category != category:
                continue
            skill_dict = asdict(skill_info)
            skill_dict["feature"] = info.name
            skill_dict["source"] = "registry"
            results.append(skill_dict)
            seen_skill_names.add(skill_info.name)

    return {"skills": results, "count": len(results)}


@router.get("/api/skills/{skill_id}/schema")
async def get_skill_schema(request: Request, skill_id: str) -> Dict[str, Any]:
    """
    OpenAI function-calling schema for a specific skill.

    Returns the skill in the format expected by LLM function-calling APIs.
    """
    agent = get_agent(request)
    features = getattr(agent, "features", {})

    # Search loaded features for this skill
    for feature_name, feature in features.items():
        for tool in feature.get_tools():
            if tool.schema.name == skill_id:
                schema = tool.schema
                return {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.parameters,
                    },
                    "feature": feature_name,
                }

    # Check registry for static declaration
    all_skills = get_all_skills()
    for skill in all_skills:
        if skill.name == skill_id:
            return {
                "type": "function",
                "function": {
                    "name": skill.name,
                    "description": skill.description,
                    "parameters": {},
                },
                "source": "registry",
            }

    raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
