"""Regression guards for #2117 (findings F045 + F046).

F045 — the Vertex AI service-account route built its client with a bogus
``vendorai=True`` kwarg (``google.genai.Client`` has no such keyword); the
correct keyword is ``vertexai=True``. Any agent routed to ``google:vertex``
via service account therefore raised ``TypeError`` on first client build.

F046 — the direct Gemini adapter called ``client.generate_content_async``,
a method the maintained ``google-genai`` client does not expose, and ignored
the routed model. Generation is now ported onto
``client.aio.models.generate_content(model=..., contents=..., config=...)``.

Both tests stub ``google.genai`` with a ``Client`` that enforces the real
keyword-only signature, so a future vendor-rename sweep can't silently
rebreak the wire contract.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.provider_registry import ProviderRegistry
from kestrel_sovereign.llm.google_adapter import GoogleAdapter
from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter


def _empty_registry() -> ProviderRegistry:
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg.config = {}
    return reg


class _FakeGenaiClient:
    """Enforces the real keyword-only signature of ``google.genai.Client``.

    The real constructor accepts ``vertexai``, ``project``, ``location`` and
    ``api_key`` as keyword-only arguments. A stray positional or an unknown
    keyword (e.g. the old ``vendorai``) raises ``TypeError`` — exactly as the
    genuine SDK would.
    """

    def __init__(self, *, vertexai: bool = False, project=None,
                 location=None, api_key=None):
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self.api_key = api_key


def _install_fake_genai(monkeypatch):
    """Point ``from google import genai`` at ``_FakeGenaiClient``.

    When the real ``google-genai`` SDK is importable we swap only its
    ``Client`` (robust against import caching / namespace-package attribute
    resolution). Otherwise we inject a stub module so the test still runs in a
    minimal environment.
    """
    try:
        import google.genai as real_genai  # noqa: F401
    except ImportError:
        fake = types.ModuleType("google.genai")
        fake.Client = _FakeGenaiClient
        monkeypatch.setitem(sys.modules, "google.genai", fake)
        google_pkg = types.ModuleType("google")
        google_pkg.genai = fake
        monkeypatch.setitem(sys.modules, "google", google_pkg)
        return fake
    monkeypatch.setattr("google.genai.Client", _FakeGenaiClient)
    return real_genai


# --- F045: Vertex service-account route builds a client without error ---

def test_vertex_service_account_route_builds_client(monkeypatch):
    _install_fake_genai(monkeypatch)
    # No GOOGLE_API_KEY → force the service-account (project/location) branch.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    reg = _empty_registry()
    route_cfg = {"project_id": "test-project", "location": "us-central1"}
    client, adapter = reg._build_client_and_adapter(
        vendor="google", route="vertex",
        adapter_cls=VertexAIAdapter,
        vendor_cfg={}, route_cfg=route_cfg,
    )

    assert isinstance(client, _FakeGenaiClient)
    # The client was built with the REAL keyword (vertexai), not the bogus one.
    assert client.vertexai is True
    assert client.project == "test-project"
    assert client.location == "us-central1"
    assert isinstance(adapter, VertexAIAdapter)


def test_vertex_service_account_route_rejects_bogus_kwarg(monkeypatch):
    """If the client build ever regresses to ``vendorai=True``, the real
    keyword-only signature must make it fail loudly — this locks that in."""
    with pytest.raises(TypeError):
        _FakeGenaiClient(vendorai=True, project="p", location="l")


# --- F046: direct Gemini adapter honors the routed model via google-genai ---

def test_google_route_builds_genai_client(monkeypatch):
    _install_fake_genai(monkeypatch)
    reg = _empty_registry()
    route_cfg = {"api_key": "test-key"}
    client, adapter = reg._build_client_and_adapter(
        vendor="google", route="api",
        adapter_cls=GoogleAdapter,
        vendor_cfg={}, route_cfg=route_cfg,
    )
    assert isinstance(client, _FakeGenaiClient)
    assert client.api_key == "test-key"
    assert isinstance(adapter, GoogleAdapter)


@pytest.mark.asyncio
async def test_google_adapter_get_response_uses_routed_model():
    """GoogleAdapter must call the maintained async API with the routed
    model, not the deprecated ``generate_content_async``."""
    captured = {}

    class _Part:
        text = "hello world"

    class _Content:
        parts = [_Part()]

    class _Candidate:
        content = _Content()

    class _Response:
        candidates = [_Candidate()]

    async def _generate_content(*, model, contents, config):
        captured["model"] = model
        captured["contents"] = contents
        captured["config"] = config
        return _Response()

    class _Models:
        generate_content = staticmethod(_generate_content)

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    adapter = GoogleAdapter()
    messages = adapter.create_messages(user_prompt="hi")
    result = await adapter.get_response(
        client=_Client(),
        model="gemini-2.5-flash",
        messages=messages,
    )

    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == messages
    assert result.content == "hello world"
    # The deprecated module-level path must not be used.
    assert not hasattr(_Client(), "generate_content_async")
