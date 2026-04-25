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
    # Per-call routing overrides — the voice picker sends these so the user
    # can force Pipeline (e.g. to keep their selected chat LLM as the brain)
    # or pin a specific TTS provider for this session without persisting the
    # change. Empty string / true = use persisted defaults.
    prefer_realtime: bool = True
    preferred_tts: str = ""
    preferred_stt: str = ""


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

    overrides = UserVoicePreferences(
        preferred_tts=body.preferred_tts or None,
        preferred_stt=body.preferred_stt or None,
        prefer_realtime=body.prefer_realtime,
    )
    route = await feature._resolve_route(overrides=overrides)
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
    * Plain ``dict`` already shaped as JSON Schema — passed through (then
      sanitized; see :func:`_sanitize_schema_for_openai`).
    * Anything else (None, malformed) — falls back to the permissive
      ``{"type": "object"}``.

    The output is guaranteed JSON-serializable AND structurally valid for
    the OpenAI Realtime API's strict tool-schema validator (every array has
    ``items``; every object has ``properties``; ``required`` only references
    declared properties). A single misshapen tool used to fail the entire
    mint with ``Invalid schema for function 'foo': In context=('properties',
    'steps'), array schema missing items.``
    """
    if isinstance(parameters, dict):
        return _sanitize_schema_for_openai(parameters)
    if not isinstance(parameters, (list, tuple)):
        return {"type": "object", "properties": {}}

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
    return _sanitize_schema_for_openai(schema)


def _sanitize_schema_for_openai(schema: Any) -> Any:
    """Recursively patch a JSON Schema fragment to satisfy OpenAI's validator.

    OpenAI's tool-schema validator is stricter than vanilla JSON Schema:

    * Every node with ``type: "array"`` must have an ``items`` value.
    * Every node with ``type: "object"`` must have a ``properties`` value.
    * Every entry in ``required`` must exist in ``properties``.

    A single tool failing any of these blows up the entire ``sessions.create``
    call (it rejects the whole batch). This walk patches the most common
    shortcomings in place rather than dropping the offending tool, so
    Realtime stays usable even when an upstream feature ships a sloppy
    schema. Returns a new dict; doesn't mutate the input.
    """
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = dict(schema)
    type_ = out.get("type")

    if type_ == "array":
        # Default array element schema = anything; OpenAI accepts {} ↔ "any".
        items = out.get("items")
        if not isinstance(items, dict):
            out["items"] = {}
        else:
            out["items"] = _sanitize_schema_for_openai(items)

    if type_ == "object":
        props = out.get("properties")
        if not isinstance(props, dict):
            props = {}
        sanitized_props = {
            name: _sanitize_schema_for_openai(value)
            for name, value in props.items()
        }
        out["properties"] = sanitized_props
        # Drop required entries that don't have a matching property —
        # OpenAI rejects mismatched required arrays.
        required = out.get("required")
        if isinstance(required, list):
            cleaned = [r for r in required if r in sanitized_props]
            if cleaned:
                out["required"] = cleaned
            else:
                out.pop("required", None)

    return out


def _clamp_turn_mode(mode: str) -> str:
    """Guard against arbitrary strings in the request body."""
    allowed = {"server_vad", "semantic_vad", "none"}
    return mode if mode in allowed else "server_vad"


# ---------------------------------------------------------------------------
# Route introspection — lets the UI show which voice path + which model is
# actually answering, so users aren't surprised that "voice" can mean
# gpt-realtime-1.5 (Realtime path) instead of their selected chat model.
# ---------------------------------------------------------------------------


class RouteIntrospectionResponse(BaseModel):
    """What the UI needs to display the active voice setup at a glance."""

    path: Optional[str]                     # "realtime" | "pipeline" | "local" | None
    reason: str
    llm_vendor: str                         # the agent's current chat-LLM vendor
    conversation_provider: Optional[str]    # set on Realtime path
    tts_provider: Optional[str]             # set on Pipeline / local path
    stt_provider: Optional[str]
    # Voice-side model + voice IDs the resolver would actually mint with
    # right now. None if the provider isn't installed or model discovery
    # hasn't completed (the UI should show "discovering..." then).
    voice_model: Optional[str]
    voice_id: Optional[str]
    available_conversation_providers: list[str]
    available_tts_providers: list[str]
    available_stt_providers: list[str]


@router.get("/route")
async def introspect_voice_route(
    request: Request,
    prefer_realtime: bool = True,
    preferred_tts: str = "",
    preferred_stt: str = "",
) -> RouteIntrospectionResponse:
    """Return the current resolved voice route + the model that would answer.

    Pure-introspection: never mints a session, never opens a connection.
    The UI calls this on load and whenever the user changes the chat-LLM
    selection or voice-picker overrides so the model display can preview
    "if I started voice now, what would actually answer?".

    Query params let the picker preview the result of forcing a path
    ("would Pipeline pick ElevenLabs?") without committing to a session.
    """
    agent = get_agent(request)
    feature = _get_voice_feature(agent)
    if feature is None:
        raise HTTPException(status_code=503, detail="Voice feature not enabled on this agent.")

    overrides = UserVoicePreferences(
        preferred_tts=preferred_tts or None,
        preferred_stt=preferred_stt or None,
        prefer_realtime=prefer_realtime,
    )
    route = await feature._resolve_route(overrides=overrides)
    registry = await feature._ensure_registry()
    voice_model = None
    voice_id = None

    # Conversation provider: ask it which model + voice it would use.
    if route.conversation_provider:
        provider = registry.get_conversation(route.conversation_provider)
        if provider is not None:
            try:
                models = await provider.discover_models()
                if models:
                    voice_model = models[0]
            except Exception:  # noqa: BLE001 — discovery best-effort
                pass
        voice_id = getattr(feature._voice_config, "tts_voice_id", "") or _default_voice(feature)

    # Pipeline path: model name comes from the TTS provider's discovery.
    elif route.tts_provider:
        tts = registry.get_tts(route.tts_provider)
        if tts is not None:
            try:
                # Providers expose `discover_models()` per the upgraded
                # contract (#1 in each cross-repo). Fall back to whatever
                # `_resolve_model` would pick.
                discover = getattr(tts, "discover_models", None)
                if discover:
                    models = await discover()
                    if models:
                        voice_model = models[0]
            except Exception:  # noqa: BLE001
                pass
        voice_id = getattr(feature._voice_config, "tts_voice_id", "")

    return RouteIntrospectionResponse(
        path=route.path,
        reason=route.reason,
        llm_vendor=feature._get_llm_vendor(),
        conversation_provider=route.conversation_provider,
        tts_provider=route.tts_provider,
        stt_provider=route.stt_provider,
        voice_model=voice_model,
        voice_id=voice_id or None,
        available_conversation_providers=sorted(
            getattr(registry, "list_conversation_providers", lambda: [])()
        ),
        available_tts_providers=sorted(registry.list_tts_providers()),
        available_stt_providers=sorted(registry.list_stt_providers()),
    )


# ---------------------------------------------------------------------------
# Tool dispatch — the Realtime model invokes tools server-side via OpenAI's
# function-calling protocol. The browser receives the call (data channel),
# POSTs here so we can run the tool against the agent's enabled features,
# and forwards the result back to OpenAI to unblock the model.
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    call_id: str
    name: str
    arguments: dict = {}


class ToolCallResponse(BaseModel):
    call_id: str
    result: Any


@router.post("/tools/{session_id}")
async def dispatch_tool_call(
    session_id: str,
    body: ToolCallRequest,
    request: Request,
):
    """Run a tool the Realtime model requested, return the result.

    The frontend is responsible for committing the result back to the
    Realtime data channel via ``client.commitToolResult``. This endpoint
    just executes the tool against the agent's enabled features and
    returns the raw result (or an error payload — the frontend commits
    SOMETHING either way; silence wedges the model).

    ``session_id`` is part of the path for audit logging and future
    per-session ACLs (e.g. only allow tools registered with the session
    that minted that ID). Today it's logged but not enforced.
    """
    agent = get_agent(request)
    feature = _get_voice_feature(agent)
    if feature is None:
        raise HTTPException(status_code=503, detail="Voice feature not enabled on this agent.")

    tool = _resolve_tool(agent, body.name)
    if tool is None:
        logger.warning(
            "voice_realtime.tool unknown agent=%s session=%s name=%s",
            getattr(agent, "agent_id", "?"), session_id, body.name,
        )
        return ToolCallResponse(
            call_id=body.call_id,
            result={"error": f"Tool not found: {body.name!r}"},
        )

    args = body.arguments or {}
    try:
        result = await tool.execute(**args)
    except TypeError as exc:
        # Bad arguments shape — return as a normal error result so the
        # model can self-correct rather than wedging the session.
        logger.warning(
            "voice_realtime.tool bad args agent=%s session=%s name=%s args=%s exc=%s",
            getattr(agent, "agent_id", "?"), session_id, body.name, args, exc,
        )
        return ToolCallResponse(
            call_id=body.call_id,
            result={"error": f"Bad arguments for {body.name}: {exc}"},
        )
    except Exception as exc:  # noqa: BLE001 — surface to the model as a string
        logger.exception("voice_realtime.tool execution failed name=%s", body.name)
        return ToolCallResponse(
            call_id=body.call_id,
            result={"error": f"Tool {body.name} raised: {exc}"},
        )

    logger.info(
        "voice_realtime.tool agent=%s session=%s name=%s args_keys=%s",
        getattr(agent, "agent_id", "?"),
        session_id,
        body.name,
        sorted(args.keys()),
    )
    return ToolCallResponse(call_id=body.call_id, result=result)


def _resolve_tool(agent: Any, name: str):
    """Find an AgentTool by name across all enabled features.

    Mirrors how the existing chat path looks up tools — features are
    keyed by class name in ``agent.features``; each feature exposes
    ``get_tools()`` returning ``AgentTool`` instances with a ``.name``
    attribute. First match wins (names should be globally unique within
    an agent's enabled-feature set).
    """
    features = getattr(agent, "features", {}) or {}
    for feature in features.values():
        get_tools = getattr(feature, "get_tools", None)
        if get_tools is None:
            continue
        try:
            for tool in get_tools() or []:
                if getattr(tool, "name", None) == name:
                    return tool
        except Exception:  # noqa: BLE001 — one broken feature mustn't block tool lookup
            continue
    return None
