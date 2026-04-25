"""
Unit tests for the ephemeral-token mint endpoint ``POST /voice/realtime/session``.

Covers:

* Success path: resolver picks realtime, conversation provider mints a
  token, response body carries the client_secret + expires_at.
* 409 fallback: resolver picks pipeline/local/None — frontend gets
  structured payload to switch paths.
* 503 when VoiceFeature isn't enabled.
* 500 when the resolver chose a provider that isn't registered.
* 502 when the provider's mint call raises.
* ``_compose_instructions`` stitches system prompt + tag snippet + user
  override with blank-section omission.
* ``_collect_tools`` gracefully skips features without parseable schemas.

All tests use a fake agent + fake provider — no network, no FastAPI app
startup.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from endpoints.voice_realtime import (
    _clamp_turn_mode,
    _collect_tools,
    _compose_instructions,
    _default_voice,
    router as realtime_router,
)
from kestrel_sdk.voice import ToolDef
from kestrel_sovereign.voice.routing import VoiceRoute


# ---------------------------------------------------------------------------
# Fake agent + fake provider
# ---------------------------------------------------------------------------


class _FakeEphemeralSession:
    def __init__(self) -> None:
        self.session_id = "sess_123"
        self.client_secret = "ek_abc"
        self.expires_at = 1700000060
        self.model = "gpt-realtime-1.5"
        self.voice = "cedar"


class _FakeConversationProvider:
    name = "openai_realtime"
    is_local = False

    def __init__(self, *, raises: Exception | None = None) -> None:
        self._raises = raises
        self.mint_calls: list[dict] = []

    async def mint_ephemeral(self, **kwargs) -> _FakeEphemeralSession:
        if self._raises is not None:
            raise self._raises
        self.mint_calls.append(kwargs)
        return _FakeEphemeralSession()


def _make_agent(
    *,
    route: VoiceRoute,
    provider: _FakeConversationProvider | None,
    include_voice_feature: bool = True,
    tools: list[Any] | None = None,
    system_prompt: str = "",
) -> Any:
    registry = MagicMock()
    registry.get_conversation = MagicMock(
        return_value=provider if provider is not None else None
    )

    voice_feature = MagicMock()
    voice_feature._resolve_route = AsyncMock(return_value=route)
    voice_feature._ensure_registry = AsyncMock(return_value=registry)
    voice_feature._get_privacy_mode_name = MagicMock(return_value="normal")
    voice_feature._voice_config = SimpleNamespace(tts_voice_id="cedar")
    # Required for isinstance(feature, VoiceFeature) to pass — monkey-patch
    # a class marker instead.
    from kestrel_sovereign.features.voice.feature import VoiceFeature

    voice_feature.__class__ = VoiceFeature

    # Mirror production keying — agent.features is keyed by class name
    # ("VoiceFeature"), not tool name ("voice"). Bug-fix backstop: the
    # endpoint must find the feature under the class-name key.
    features = {"VoiceFeature": voice_feature} if include_voice_feature else {}
    # Add a fake tool-bearing feature so _collect_tools exercises.
    if tools is not None:
        features["tool_owner"] = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name=t.name,
                    schema=SimpleNamespace(
                        name=t.name,
                        description=t.description,
                        parameters=t.parameters_schema,
                    ),
                )
                for t in tools
            ]
        )

    identity = SimpleNamespace(system_prompt=system_prompt)
    return SimpleNamespace(
        agent_id="test-agent",
        features=features,
        identity=identity,
    )


@pytest.fixture
def client() -> TestClient:
    # Mirror the production nesting: VoiceFeature.get_router() includes the
    # realtime router into the parent /voice router, so the final path is
    # /voice/realtime/session. Without this wrapper the test would call
    # /realtime/session and miss any future regression where the parent's
    # prefix changes.
    app = FastAPI()
    voice_parent = APIRouter(prefix="/voice", tags=["voice"])
    voice_parent.include_router(realtime_router)
    app.include_router(voice_parent)
    return TestClient(app)


def _inject_agent(monkeypatch: pytest.MonkeyPatch, agent: Any) -> None:
    """Stub get_agent to return the fake so the endpoint can run sans middleware."""
    import endpoints.voice_realtime as vr

    monkeypatch.setattr(vr, "get_agent", lambda _request: agent)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestRealtimeSessionSuccess:
    def test_mint_returns_client_secret(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _FakeConversationProvider()
        agent = _make_agent(
            route=VoiceRoute(
                path="realtime",
                conversation_provider="openai_realtime",
                reason="Realtime path",
            ),
            provider=provider,
        )
        _inject_agent(monkeypatch, agent)

        resp = client.post("/voice/realtime/session", json={"voice": "cedar"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["path"] == "realtime"
        assert body["session_id"] == "sess_123"
        assert body["client_secret"] == {"value": "ek_abc", "expires_at": 1700000060}
        assert body["model"] == "gpt-realtime-1.5"
        assert body["voice"] == "cedar"

    def test_mint_forwards_voice_instructions_and_tools(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _FakeConversationProvider()
        tools = [ToolDef(name="echo", description="desc", parameters_schema={"type": "object"})]
        agent = _make_agent(
            route=VoiceRoute(
                path="realtime",
                conversation_provider="openai_realtime",
                reason="Realtime path",
            ),
            provider=provider,
            tools=tools,
            system_prompt="You are warm.",
        )
        _inject_agent(monkeypatch, agent)

        resp = client.post(
            "/voice/realtime/session",
            json={
                "voice": "marin",
                "user_instructions": "Like a 1920s newscaster.",
                "turn_detection_mode": "semantic_vad",
                "silence_ms": 800,
            },
        )
        assert resp.status_code == 200
        assert provider.mint_calls, "mint was not invoked"
        kwargs = provider.mint_calls[0]
        assert kwargs["voice"] == "marin"
        assert "You are warm." in kwargs["instructions"]
        assert "Like a 1920s newscaster." in kwargs["instructions"]
        assert kwargs["tools"][0].name == "echo"
        assert kwargs["turn_detection"].mode == "semantic_vad"
        assert kwargs["turn_detection"].silence_ms == 800


# ---------------------------------------------------------------------------
# 409 fallback paths
# ---------------------------------------------------------------------------


class TestRealtimeUnavailable:
    def test_pipeline_route_returns_409_with_fallbacks(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(
                path="pipeline",
                tts_provider="elevenlabs",
                stt_provider="openai",
                reason="Pipeline path: non-OpenAI LLM.",
            ),
            provider=None,
        )
        _inject_agent(monkeypatch, agent)

        resp = client.post("/voice/realtime/session", json={"voice": "cedar"})
        assert resp.status_code == 409
        body = resp.json()
        assert body["path"] == "pipeline"
        assert body["fallback_tts"] == "elevenlabs"
        assert body["fallback_stt"] == "openai"
        assert "Pipeline" in body["reason"]

    def test_local_route_returns_409(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(
                path="local",
                tts_provider="piper",
                stt_provider="faster_whisper",
                reason="Local-only pipeline",
            ),
            provider=None,
        )
        _inject_agent(monkeypatch, agent)

        resp = client.post("/voice/realtime/session", json={"voice": "cedar"})
        assert resp.status_code == 409
        body = resp.json()
        assert body["path"] == "local"
        assert body["fallback_tts"] == "piper"

    def test_none_route_returns_409(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(path=None, reason="Nothing installed."),
            provider=None,
        )
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={"voice": "cedar"})
        assert resp.status_code == 409
        assert resp.json()["path"] is None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRealtimeErrors:
    def test_503_when_no_voice_feature(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(path="realtime", conversation_provider="openai_realtime", reason=""),
            provider=None,
            include_voice_feature=False,
        )
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 503

    def test_500_when_resolver_chose_unregistered_provider(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(
                path="realtime",
                conversation_provider="openai_realtime",
                reason="Realtime path",
            ),
            provider=None,  # resolver said realtime but registry has nothing
        )
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 500
        assert "not registered" in resp.json()["detail"].lower()

    def test_502_when_mint_raises(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _FakeConversationProvider(raises=RuntimeError("OpenAI 503"))
        agent = _make_agent(
            route=VoiceRoute(
                path="realtime",
                conversation_provider="openai_realtime",
                reason="",
            ),
            provider=provider,
        )
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestComposeInstructions:
    def test_omits_blank_sections(self) -> None:
        agent = SimpleNamespace(identity=SimpleNamespace(system_prompt=""))
        out = _compose_instructions(agent, "")
        # Only the tag snippet should remain.
        assert "tag" in out.lower() or out == ""

    def test_order_system_then_snippet_then_override(self) -> None:
        agent = SimpleNamespace(
            identity=SimpleNamespace(system_prompt="SYS_PROMPT_MARKER")
        )
        out = _compose_instructions(agent, "OVERRIDE_MARKER")
        sys_idx = out.find("SYS_PROMPT_MARKER")
        override_idx = out.find("OVERRIDE_MARKER")
        assert sys_idx != -1 and override_idx != -1
        assert sys_idx < override_idx  # system prompt precedes override

    def test_no_identity_still_works(self) -> None:
        agent = SimpleNamespace()  # no identity
        out = _compose_instructions(agent, "hello")
        assert "hello" in out


class TestCollectTools:
    def test_extracts_tooldef_from_features(self) -> None:
        f1 = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="search",
                    schema=SimpleNamespace(
                        name="search",
                        description="d",
                        parameters={"type": "object"},
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"search": f1})
        tools = _collect_tools(agent)
        assert len(tools) == 1
        assert tools[0].name == "search"

    def test_skips_features_that_raise(self) -> None:
        broken = SimpleNamespace(get_tools=MagicMock(side_effect=RuntimeError))
        good = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="ok",
                    schema=SimpleNamespace(
                        name="ok", description="", parameters={"type": "object"}
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"broken": broken, "good": good})
        tools = _collect_tools(agent)
        # Broken feature skipped; good one still collected.
        assert len(tools) == 1
        assert tools[0].name == "ok"

    def test_skips_tools_without_schema(self) -> None:
        f = SimpleNamespace(
            get_tools=lambda: [SimpleNamespace(name="noschema", schema=None)]
        )
        agent = SimpleNamespace(features={"f": f})
        tools = _collect_tools(agent)
        assert tools == []

    def test_agent_without_features_returns_empty(self) -> None:
        agent = SimpleNamespace()
        assert _collect_tools(agent) == []

    def test_real_tool_parameters_serialize_to_json(self) -> None:
        """Regression: live agents expose ``schema.parameters`` as a list of
        ``ToolParameter`` dataclasses (Kestrel SDK shape), not a JSON Schema
        dict. The previous _collect_tools shoved the raw list through to
        ``ToolDef.parameters_schema`` and the OpenAI SDK exploded with::

            Object of type ToolParameter is not JSON serializable

        when it tried to ``json.dumps`` the session config. This test would
        have caught it: build a tool whose schema has real ToolParameter
        instances, run _collect_tools, and assert the result is JSON-encodable.
        """
        import json

        from kestrel_sdk.tools.base import ToolParameter

        param_required = ToolParameter(
            name="city",
            type="string",
            description="City name to look up",
            required=True,
        )
        param_optional = ToolParameter(
            name="units",
            type="string",
            description="Temperature units",
            required=False,
            enum=["celsius", "fahrenheit"],
        )
        feature = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="weather",
                    schema=SimpleNamespace(
                        name="weather",
                        description="Look up the weather.",
                        parameters=[param_required, param_optional],
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"f": feature})

        tools = _collect_tools(agent)
        assert len(tools) == 1
        # The critical assertion: parameters_schema round-trips through
        # json.dumps without raising. If anything in there is still a
        # ToolParameter (or other non-JSON type), this fails — same way
        # the OpenAI SDK fails when minting a session.
        encoded = json.dumps(tools[0].parameters_schema)
        decoded = json.loads(encoded)
        assert decoded["type"] == "object"
        assert decoded["properties"]["city"]["type"] == "string"
        assert decoded["properties"]["city"]["description"] == "City name to look up"
        assert decoded["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]
        assert decoded["required"] == ["city"]
        # Optional params should NOT appear in `required`.
        assert "units" not in decoded["required"]

    def test_dict_parameters_pass_through_unchanged(self) -> None:
        """Tools that already declare parameters as a JSON Schema dict (rare,
        but possible) should not be re-wrapped — pass through identity."""
        explicit = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }
        feature = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="search",
                    schema=SimpleNamespace(
                        name="search", description="d", parameters=explicit
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"f": feature})
        tools = _collect_tools(agent)
        assert tools[0].parameters_schema == explicit

    def test_array_param_without_items_gets_default_items(self) -> None:
        """Regression: live `run_workflow` tool declares `steps: array` with
        no `items`. OpenAI's strict validator rejects with::

            Invalid schema for function 'run_workflow':
            In context=('properties', 'steps'), array schema missing items.

        The converter must default `items: {}` (any-type) when the
        ToolParameter doesn't declare item shape.
        """
        from kestrel_sdk.tools.base import ToolParameter

        param = ToolParameter(
            name="steps",
            type="array",
            description="Workflow steps to run",
            required=True,
            # Critically: no `items` set — exactly the bug shape.
        )
        feature = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="run_workflow",
                    schema=SimpleNamespace(
                        name="run_workflow",
                        description="Run a workflow.",
                        parameters=[param],
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"f": feature})
        tools = _collect_tools(agent)
        steps_schema = tools[0].parameters_schema["properties"]["steps"]
        assert steps_schema["type"] == "array"
        assert "items" in steps_schema, (
            "OpenAI rejects array schemas without `items`; converter must "
            "default to `items: {}` when the ToolParameter doesn't declare it"
        )

    def test_dict_passthrough_runs_sanitizer_on_nested_arrays(self) -> None:
        """Tools that pass an explicit dict schema with a nested broken
        array also need patching — the dict-passthrough path must run the
        same sanitizer."""
        explicit_with_bug = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"},  # missing items
                "config": {
                    "type": "object",
                    "properties": {
                        "modes": {"type": "array"},  # missing items, nested
                    },
                },
            },
            "required": ["tags", "missing_prop"],  # references nonexistent prop
        }
        feature = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="x",
                    schema=SimpleNamespace(
                        name="x", description="d", parameters=explicit_with_bug
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"f": feature})
        tools = _collect_tools(agent)
        s = tools[0].parameters_schema
        assert "items" in s["properties"]["tags"]
        assert "items" in s["properties"]["config"]["properties"]["modes"]
        # `missing_prop` should be filtered from `required` since it has no
        # corresponding property entry — OpenAI rejects mismatched required.
        assert s["required"] == ["tags"]

    def test_object_without_properties_gets_empty_properties(self) -> None:
        """Object nodes with no `properties` are also rejected by OpenAI."""
        explicit = {"type": "object"}  # no properties at all
        feature = SimpleNamespace(
            get_tools=lambda: [
                SimpleNamespace(
                    name="empty",
                    schema=SimpleNamespace(
                        name="empty", description="d", parameters=explicit
                    ),
                )
            ]
        )
        agent = SimpleNamespace(features={"f": feature})
        tools = _collect_tools(agent)
        assert tools[0].parameters_schema["properties"] == {}

    def test_collect_tools_output_is_fully_json_serializable(self) -> None:
        """The strongest invariant: whatever _collect_tools returns can be
        passed straight to ``json.dumps``, which is what the OpenAI SDK does
        internally on the session-create call. Use a mix of all parameter
        shapes the live runtime can produce.
        """
        import json

        from kestrel_sdk.tools.base import ToolParameter

        feature = SimpleNamespace(
            get_tools=lambda: [
                # Tool with ToolParameter list (canonical shape)
                SimpleNamespace(
                    name="t1",
                    schema=SimpleNamespace(
                        name="t1",
                        description="d1",
                        parameters=[
                            ToolParameter(name="x", type="integer", description="x", required=True),
                            ToolParameter(
                                name="tags", type="array", description="tags",
                                items={"type": "string"},
                            ),
                        ],
                    ),
                ),
                # Tool with empty parameter list
                SimpleNamespace(
                    name="t2",
                    schema=SimpleNamespace(name="t2", description="d2", parameters=[]),
                ),
                # Tool with no parameters attribute
                SimpleNamespace(
                    name="t3",
                    schema=SimpleNamespace(name="t3", description="d3", parameters=None),
                ),
            ]
        )
        agent = SimpleNamespace(features={"f": feature})
        tools = _collect_tools(agent)
        # Build the kwargs the OpenAI SDK gets — with `tools` as a list of
        # the same tool dicts ephemeral_session._tool_to_openai produces.
        payload = [
            {"type": "function", "name": t.name, "description": t.description,
             "parameters": t.parameters_schema}
            for t in tools
        ]
        # If any element is non-serializable, this raises.
        json.dumps(payload)


class TestClampTurnMode:
    def test_allows_known_modes(self) -> None:
        assert _clamp_turn_mode("server_vad") == "server_vad"
        assert _clamp_turn_mode("semantic_vad") == "semantic_vad"
        assert _clamp_turn_mode("none") == "none"

    def test_falls_back_for_unknown(self) -> None:
        assert _clamp_turn_mode("some_attack_string") == "server_vad"
        assert _clamp_turn_mode("") == "server_vad"


class TestDefaultVoice:
    def test_uses_configured_voice_id(self) -> None:
        feature = SimpleNamespace(_voice_config=SimpleNamespace(tts_voice_id="marin"))
        assert _default_voice(feature) == "marin"

    def test_falls_back_to_cedar(self) -> None:
        feature = SimpleNamespace(_voice_config=SimpleNamespace(tts_voice_id=""))
        assert _default_voice(feature) == "cedar"


# ---------------------------------------------------------------------------
# Router-prefix regression — guards against the doubled `/voice/voice/` bug
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature-key regression — guards against the "voice" vs "VoiceFeature" bug
# ---------------------------------------------------------------------------


class TestFeatureKeyLookup:
    """The agent.features dict is keyed by class name in the live runtime —
    confirmed by the existing endpoints/voice.py which looks up
    ``features.get("VoiceFeature")``. The endpoint we ship in #726 must use
    the same key or every call returns 503 against a real agent.
    """

    def test_class_name_key_is_found(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _make_agent already uses the class-name key; this test is
        # explicit so the intent is documented.
        provider = _FakeConversationProvider()
        agent = _make_agent(
            route=VoiceRoute(path="realtime", conversation_provider="openai_realtime", reason=""),
            provider=provider,
        )
        # Confirm the key the fixture uses really is "VoiceFeature".
        assert "VoiceFeature" in agent.features
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 200, resp.text

    def test_wrong_key_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-key the same feature under "voice" (the old buggy key) and
        # show that a fallback key in _get_voice_feature still finds it —
        # the endpoint stays robust either way.
        provider = _FakeConversationProvider()
        agent = _make_agent(
            route=VoiceRoute(path="realtime", conversation_provider="openai_realtime", reason=""),
            provider=provider,
        )
        agent.features = {"voice": agent.features["VoiceFeature"]}
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 200, resp.text

    def test_neither_key_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = _make_agent(
            route=VoiceRoute(path="realtime", conversation_provider="openai_realtime", reason=""),
            provider=_FakeConversationProvider(),
        )
        agent.features = {}  # voice feature genuinely missing
        _inject_agent(monkeypatch, agent)
        resp = client.post("/voice/realtime/session", json={})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tool dispatch endpoint
# ---------------------------------------------------------------------------


class TestToolDispatch:
    """POST /voice/realtime/tools/{session_id} runs the requested tool against
    the agent's enabled features and returns the result. Critical
    invariants:

    1. Always returns a 200 with a result payload — even on error — so
       the frontend can commit *something* back to the Realtime data
       channel. Silence wedges the model.
    2. Unknown tool names → returns ``{result: {error: ...}}``, not 404.
    3. Tool execution failures → caught, returned as ``{result: {error: ...}}``.
    """

    def _agent_with_tool(self, tool_obj: Any) -> Any:
        feature = SimpleNamespace(get_tools=lambda: [tool_obj])
        from kestrel_sovereign.features.voice.feature import VoiceFeature

        voice_marker = MagicMock()
        voice_marker.__class__ = VoiceFeature
        return SimpleNamespace(
            agent_id="test",
            features={"VoiceFeature": voice_marker, "tool_owner": feature},
            identity=SimpleNamespace(system_prompt=""),
        )

    def test_dispatches_known_tool_and_returns_result(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _execute(**kwargs):
            return {"echoed": kwargs}

        tool = SimpleNamespace(name="echo", execute=_execute)
        agent = self._agent_with_tool(tool)
        _inject_agent(monkeypatch, agent)

        resp = client.post(
            "/voice/realtime/tools/sess_abc",
            json={"call_id": "call_1", "name": "echo", "arguments": {"text": "hi"}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["call_id"] == "call_1"
        assert body["result"] == {"echoed": {"text": "hi"}}

    def test_unknown_tool_returns_error_result_not_404(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = self._agent_with_tool(SimpleNamespace(name="other", execute=AsyncMock()))
        _inject_agent(monkeypatch, agent)
        resp = client.post(
            "/voice/realtime/tools/sess",
            json={"call_id": "c", "name": "ghost", "arguments": {}},
        )
        # 200 + error payload — silence would wedge the model.
        assert resp.status_code == 200
        body = resp.json()
        assert body["call_id"] == "c"
        assert "error" in body["result"]
        assert "ghost" in body["result"]["error"]

    def test_tool_exception_returns_error_result(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _explode(**_):
            raise RuntimeError("kaboom")

        agent = self._agent_with_tool(SimpleNamespace(name="boom", execute=_explode))
        _inject_agent(monkeypatch, agent)
        resp = client.post(
            "/voice/realtime/tools/sess",
            json={"call_id": "c", "name": "boom", "arguments": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "kaboom" in body["result"]["error"]

    def test_bad_arguments_returns_error_not_500(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _strict(*, required_arg):
            return {"ok": required_arg}

        agent = self._agent_with_tool(SimpleNamespace(name="strict", execute=_strict))
        _inject_agent(monkeypatch, agent)
        resp = client.post(
            "/voice/realtime/tools/sess",
            json={"call_id": "c", "name": "strict", "arguments": {"wrong_kwarg": 1}},
        )
        # Still 200 so the frontend commits the error result.
        assert resp.status_code == 200
        body = resp.json()
        assert "Bad arguments" in body["result"]["error"]

    def test_503_when_voice_feature_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = SimpleNamespace(features={}, identity=None)
        _inject_agent(monkeypatch, agent)
        resp = client.post(
            "/voice/realtime/tools/sess",
            json={"call_id": "c", "name": "anything", "arguments": {}},
        )
        assert resp.status_code == 503

    def test_router_registers_tools_path(self) -> None:
        voice_parent = APIRouter(prefix="/voice", tags=["voice"])
        voice_parent.include_router(realtime_router)
        paths = {route.path for route in voice_parent.routes}
        assert "/voice/realtime/tools/{session_id}" in paths


def test_router_registers_session_at_voice_realtime_session() -> None:
    """The realtime router's prefix must be `/realtime` so that nesting it
    inside the parent `/voice` router yields exactly `/voice/realtime/session`.

    Setting the realtime router's prefix to `/voice/realtime` would land the
    final route at `/voice/voice/realtime/session` and 404 against every
    Kestrel deployment that mounts via VoiceFeature.get_router (which is all
    of them in production).
    """
    voice_parent = APIRouter(prefix="/voice", tags=["voice"])
    voice_parent.include_router(realtime_router)
    paths = {route.path for route in voice_parent.routes}
    assert "/voice/realtime/session" in paths
    assert "/voice/voice/realtime/session" not in paths
