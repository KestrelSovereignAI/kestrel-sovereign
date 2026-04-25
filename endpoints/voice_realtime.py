"""
Voice Realtime ephemeral-session mint endpoint.

``POST /voice/realtime/session`` composes a per-agent Realtime session
config server-side (voice, agent system prompt + voice tag snippet,
privacy-gated tools, turn detection) and mints a short-lived client token
via the OpenAI Realtime API. The browser then uses that token to open a
WebRTC peer connection directly to OpenAI — the long-lived
``OPENAI_API_KEY`` never leaves the server.

Privacy gate: this endpoint consults the voice path resolver (#723). When
the resolver decides a path other than ``realtime`` (local-only, anonymous
pipeline, or cloud pipeline), the endpoint returns HTTP 409 with the
resolver's reason so the frontend can fall back to the Pipeline WebSocket
path without a round trip.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from endpoints.agent_helpers import get_agent
from kestrel_sdk.voice import AudioFormat, ToolDef, TurnDetectionConfig
from kestrel_sovereign.features.voice.feature import VoiceFeature
from kestrel_sovereign.voice import tags as voice_tags
from kestrel_sovereign.voice.routing import (
    InstalledProviders,
    UserVoicePreferences,
    VoiceRoute,
    VoiceRoutingContext,
    resolve as resolve_voice_route,
)

logger = logging.getLogger(__name__)

# NOTE: prefix is `/realtime`, not `/voice/realtime`. VoiceFeature.get_router
# mounts this router via ``parent.include_router(realtime_router)`` where the
# parent already has ``prefix="/voice"`` — FastAPI concatenates the two, so a
# `/voice/realtime` here would land at `/voice/voice/realtime/...` in
# production (and 404 cleanly on every rookery host). Tests fixture-wrap with
# the parent router so the same nesting is exercised.
router = APIRouter(prefix="/realtime", tags=["voice"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RealtimeSessionRequest(BaseModel):
    """Parameters the browser sends when requesting a new Realtime session."""

    voice: str = ""
    # Optional free-form override appended to the composed instructions. Use
    # for ad-hoc directives ("speak like a 1920s newscaster") not captured in
    # the agent's persistent system prompt.
    user_instructions: str = ""
    # Turn-detection knobs with sensible defaults — frontend may override.
    turn_detection_mode: str = "server_vad"  # "server_vad" | "semantic_vad" | "none"
    silence_ms: int = 500


class RealtimeSessionResponse(BaseModel):
    """Minted ephemeral-session bundle the browser uses for its WebRTC connection."""

    path: str  # Always "realtime" in the success case.
    session_id: str
    client_secret: dict  # {"value": "ek_...", "expires_at": <unix-epoch>}
    model: str
    voice: str


class RealtimeUnavailableResponse(BaseModel):
    """Returned as HTTP 409 when the resolver refuses the Realtime path."""

    path: Optional[str]  # "local" | "pipeline" | None
    reason: str
    fallback_tts: Optional[str] = None
    fallback_stt: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/session")
async def create_realtime_session(body: RealtimeSessionRequest, request: Request):
    """Mint an ephemeral OpenAI Realtime session for the browser.

    Flow:

    1. Get the current agent from the request.
    2. Ask the voice path resolver (#723) for the active route given the
       agent's LLM vendor, privacy config, and installed providers. If the
       resolver picks anything other than ``realtime``, return 409 with
       fallback provider names so the frontend switches to Pipeline.
    3. Compose the session's ``instructions`` = agent system prompt + voice
       tag snippet (#724) + user override.
    4. Look up the conversation provider (``openai_realtime``) in the
       registry and call its ``mint_ephemeral`` helper.
    5. Return the ephemeral bundle.
    """
    agent = get_agent(request)

    feature = _get_voice_feature(agent)
    if feature is None:
        raise HTTPException(
            status_code=503,
            detail="Voice feature not enabled on this agent.",
        )

    route = await feature._resolve_route()
    if route.path != "realtime" or not route.conversation_provider:
        return _unavailable_response(route)

    registry = await feature._ensure_registry()
    provider = registry.get_conversation(route.conversation_provider)
    if provider is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Resolver chose conversation provider "
                f"'{route.conversation_provider}' but it is not registered."
            ),
        )

    mint = getattr(provider, "mint_ephemeral", None)
    if mint is None:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Conversation provider '{route.conversation_provider}' "
                f"does not support ephemeral-token minting."
            ),
        )

    voice = body.voice or _default_voice(feature)
    instructions = _compose_instructions(agent, body.user_instructions)
    tools = _collect_tools(agent)
    turn_detection = TurnDetectionConfig(
        mode=_clamp_turn_mode(body.turn_detection_mode),
        silence_ms=max(0, body.silence_ms),
    )

    try:
        session = await mint(
            voice=voice,
            instructions=instructions,
            tools=tools,
            turn_detection=turn_detection,
            audio_format=AudioFormat(),
        )
    except Exception as exc:  # noqa: BLE001 — surface to client as 502
        logger.exception("Realtime session mint failed")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to mint Realtime session: {exc}",
        ) from exc

    logger.info(
        "voice_realtime.mint agent=%s path=realtime model=%s voice=%s "
        "privacy=%s tools=%d",
        getattr(agent, "agent_id", "?"),
        session.model,
        voice,
        feature._get_privacy_mode_name(),
        len(tools),
    )

    return RealtimeSessionResponse(
        path="realtime",
        session_id=session.session_id,
        client_secret={
            "value": session.client_secret,
            "expires_at": session.expires_at,
        },
        model=session.model,
        voice=session.voice,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_voice_feature(agent: Any) -> Optional[VoiceFeature]:
    # The agent's features dict is keyed by class name (e.g. "VoiceFeature"),
    # not by tool name — see endpoints/voice.py which uses the same lookup.
    # Keep "voice" as a secondary key for any future agent that uses
    # tool-name keying so the endpoint stays robust either way.
    features = getattr(agent, "features", {}) or {}
    feature = features.get("VoiceFeature") or features.get("voice")
    if isinstance(feature, VoiceFeature):
        return feature
    return None


def _unavailable_response(route: VoiceRoute):
    """Translate a non-realtime VoiceRoute into a 409 payload.

    409 Conflict signals "the requested flow is not available given current
    state" — semantically cleaner than 400 (caller did nothing wrong) or 503
    (service is there but the chosen route isn't). The frontend uses the
    ``fallback_*`` fields to open the Pipeline WebSocket instead.
    """
    from fastapi.responses import JSONResponse

    body = RealtimeUnavailableResponse(
        path=route.path,
        reason=route.reason,
        fallback_tts=route.tts_provider,
        fallback_stt=route.stt_provider,
    )
    return JSONResponse(status_code=409, content=body.model_dump())


def _default_voice(feature: VoiceFeature) -> str:
    """Pick a sensible default voice when the caller didn't supply one."""
    configured = getattr(feature._voice_config, "tts_voice_id", "")
    if configured:
        return configured
    # Fall back to the first voice the active TTS provider reports, if any.
    # Providers are expected to ship a small recommended-voices list.
    return "cedar"  # OpenAI's recommended flagship voice per their docs.


def _compose_instructions(agent: Any, user_override: str) -> str:
    """Build the session's ``instructions`` field.

    Order:
    1. Agent's system prompt (from ``agent.identity.system_prompt`` if set).
    2. Voice tag snippet (#724) teaching the agent to emit ``[excited]`` /
       ``[whispering]`` / etc. inline when voice is active.
    3. User's per-session override (passed in the request body).

    Blank sections are omitted so a minimal agent with no system prompt
    still gets a working session with just the tag snippet.
    """
    parts: list[str] = []
    identity = getattr(agent, "identity", None)
    system_prompt = ""
    if identity is not None:
        system_prompt = getattr(identity, "system_prompt", "") or ""
    if system_prompt.strip():
        parts.append(system_prompt.strip())

    snippet = voice_tags.get_voice_prompt_snippet()
    if snippet.strip():
        parts.append(snippet.strip())

    if user_override.strip():
        parts.append(user_override.strip())

    return "\n\n".join(parts)


def _collect_tools(agent: Any) -> list[ToolDef]:
    """Gather ToolDef entries for all currently-enabled agent tools.

    Translates the agent's Feature-level tools into the ``ToolDef`` shape the
    Realtime API expects. ``schema.parameters`` in the live agent is a
    ``list[ToolParameter]`` (Kestrel SDK shape), not a JSON Schema dict — we
    translate it here. Tools without parseable schemas get a permissive
    ``{"type": "object"}`` placeholder so OpenAI still accepts zero-arg calls.
    """
    tools: list[ToolDef] = []
    features = getattr(agent, "features", {}) or {}
    for feature in features.values():
        get_tools = getattr(feature, "get_tools", None)
        if get_tools is None:
            continue
        try:
            feature_tools = get_tools()
        except Exception:  # noqa: BLE001 — a single feature misbehaving mustn't break mint
            continue
        for t in feature_tools or []:
            schema = getattr(t, "schema", None)
            if schema is None:
                continue
            name = getattr(schema, "name", "") or getattr(t, "name", "")
            if not name:
                continue
            description = getattr(schema, "description", "") or ""
            parameters_schema = _build_parameters_schema(getattr(schema, "parameters", None))
            tools.append(ToolDef(name=name, description=description, parameters_schema=parameters_schema))
    return tools


def _build_parameters_schema(parameters: Any) -> dict:
    """Convert a tool's parameters spec into a JSON Schema dict.

    Accepts:

    * ``list[ToolParameter]`` (the canonical Kestrel SDK shape) — converts
      each ``ToolParameter`` into a JSON Schema property entry, collecting
      ``required: True`` entries into the top-level ``required`` array.
    * Plain ``dict`` already shaped as JSON Schema — passed through.
    * Anything else (None, malformed) — falls back to the permissive
      ``{"type": "object"}`` so OpenAI still accepts the tool definition
      and zero-arg invocations work.

    The output is guaranteed JSON-serializable; the openai SDK's session
    create call ``json.dumps``s the whole tool list and would fail on raw
    ``ToolParameter`` instances (the original bug, surfaced as
    ``Object of type ToolParameter is not JSON serializable``).
    """
    if isinstance(parameters, dict):
        return parameters
    if not isinstance(parameters, (list, tuple)):
        return {"type": "object"}

    properties: dict[str, dict] = {}
    required: list[str] = []
    for param in parameters:
        param_name = getattr(param, "name", None)
        if not param_name:
            continue
        prop: dict[str, Any] = {
            "type": getattr(param, "type", None) or "string",
        }
        desc = getattr(param, "description", None)
        if desc:
            prop["description"] = desc
        enum = getattr(param, "enum", None)
        if enum:
            prop["enum"] = list(enum)
        items = getattr(param, "items", None)
        if items:
            prop["items"] = items
        properties[param_name] = prop
        if getattr(param, "required", False):
            required.append(param_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _clamp_turn_mode(mode: str) -> str:
    """Guard against arbitrary strings in the request body."""
    allowed = {"server_vad", "semantic_vad", "none"}
    return mode if mode in allowed else "server_vad"
