"""CodexAdapter OAuth token-refresh recovery tests (#887).

When the ChatGPT-backend Responses API returns 401 with ``token_expired``,
the adapter must:

1. First re-read ``~/.codex/auth.json`` — the token may have been refreshed
   externally by the codex CLI; we just need to pick it up.
2. If the file is also stale, exchange the ``refresh_token`` for a fresh
   access_token at ``https://auth.openai.com/oauth/token``, persist the
   result, and retry the request once.
3. Only if refresh itself fails or returns no usable token do we propagate
   the 401 (which then drives ``_maybe_disable_route`` for truly revoked
   credentials).

The pre-#887 behavior — propagating ``token_expired`` 401 immediately —
caused the service-layer auto-disable to mark the route dead until server
restart, even though the standard OAuth refresh flow could have recovered.
"""
import base64
import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.continuation_store import InMemoryContinuationStore


_TOKEN_NONCE = 0


def _fake_token(account_id: str = "acct-test") -> str:
    """Build a minimal JWT carrying a chatgpt_account_id claim. Each call
    embeds a unique ``nonce`` so back-to-back tokens are byte-distinct, the
    way real refreshed access tokens are."""
    global _TOKEN_NONCE
    _TOKEN_NONCE += 1
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
            "nonce": _TOKEN_NONCE,
        }).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(f"sig-{_TOKEN_NONCE}".encode()).rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


# ---------------------------------------------------------------------------
# Fake httpx client that scripts a 401 → success sequence and captures the
# Authorization header used on each attempt so we can assert the retry used
# a different token.
# ---------------------------------------------------------------------------


class _ScriptedResponse:
    def __init__(self, status_code: int, sse_lines: List[str], text: str = ""):
        self.status_code = status_code
        self._sse_lines = sse_lines
        self.text = text

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line

    async def aread(self) -> bytes:
        return self.text.encode()


class _ScriptedStreamCtx:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *a):
        return None


class _ScriptedAsyncClient:
    """One client per test; ``responses`` is consumed in order."""

    def __init__(self, responses: List[_ScriptedResponse], captured_headers: List[Dict[str, str]]):
        self._responses = list(responses)
        self._captured = captured_headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    def stream(self, method: str, url: str, *, headers, json):
        self._captured.append(dict(headers))
        return _ScriptedStreamCtx(self._responses.pop(0))


def _sse(events: List[Dict[str, Any]]) -> List[str]:
    return [f"data: {json.dumps(e)}" for e in events]


def _completed(response_id: str) -> Dict[str, Any]:
    return {
        "type": "response.completed",
        "response": {"id": response_id, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
    }


def _patch_httpx(responses: List[_ScriptedResponse], captured: List[Dict[str, str]]):
    fake = _ScriptedAsyncClient(responses, captured)

    class _Factory:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return fake

        async def __aexit__(self, *a):
            return None

    return patch("kestrel_sovereign.llm.codex_adapter.httpx.AsyncClient", _Factory)


@pytest.mark.asyncio
class TestExternalRefreshAdoption:
    """When the codex CLI (or another peer) refreshes auth.json out of band,
    the adapter must adopt the newer token on the next 401 instead of
    propagating the error."""

    async def test_401_then_file_has_new_token_retries_with_new_token(self, tmp_path):
        original_token = _fake_token("acct-test")
        new_file_token = _fake_token("acct-test")  # ← different bytes (different JWT)
        # Sanity check the two tokens differ so the retry path is meaningful.
        assert original_token != new_file_token

        # auth.json on disk has been refreshed externally to ``new_file_token``.
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "tokens": {"access_token": new_file_token, "refresh_token": "rt-irrelevant"},
        }))

        captured: List[Dict[str, str]] = []
        responses = [
            _ScriptedResponse(401, [], text='{"error":{"code":"token_expired"}}'),
            _ScriptedResponse(200, _sse([
                {"type": "response.output_text.delta", "delta": "ok"},
                _completed("resp_after_refresh"),
            ])),
        ]
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        with _patch_httpx(responses, captured), \
             patch("kestrel_sovereign.llm.codex_adapter.CODEX_AUTH_FILE", auth_path):
            resp = await adapter.get_response(
                client=original_token,
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
            )

        assert resp.content == "ok"
        # First attempt used the original token; second used the refreshed.
        assert len(captured) == 2
        assert captured[0]["Authorization"] == f"Bearer {original_token}"
        assert captured[1]["Authorization"] == f"Bearer {new_file_token}"
        # In-memory cache now holds the refreshed token for subsequent calls.
        assert adapter._refreshed_token == new_file_token


@pytest.mark.asyncio
class TestOAuthRefreshFlow:
    """When the file is also stale, the adapter calls the OAuth endpoint,
    persists the new tokens, and retries."""

    async def test_401_then_oauth_refresh_succeeds(self, tmp_path):
        original_token = _fake_token()
        post_refresh_token = _fake_token()
        assert original_token != post_refresh_token

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "tokens": {
                "access_token": original_token,  # same as what we got 401 with → file is stale
                "refresh_token": "rt-original",
            },
        }))

        captured: List[Dict[str, str]] = []
        responses = [
            _ScriptedResponse(401, [], text='{"error":{"code":"token_expired"}}'),
            _ScriptedResponse(200, _sse([
                {"type": "response.output_text.delta", "delta": "ok"},
                _completed("resp_after_refresh"),
            ])),
        ]
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        async def fake_refresh(refresh_token: str) -> Dict[str, Any]:
            assert refresh_token == "rt-original"
            return {
                "access_token": post_refresh_token,
                "refresh_token": "rt-rotated",
                "id_token": "id-new",
            }

        with _patch_httpx(responses, captured), \
             patch("kestrel_sovereign.llm.codex_adapter.CODEX_AUTH_FILE", auth_path), \
             patch("kestrel_sovereign.llm.codex_adapter._refresh_codex_oauth_token", fake_refresh):
            resp = await adapter.get_response(
                client=original_token,
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
            )

        assert resp.content == "ok"
        # Retry used the refreshed token.
        assert captured[1]["Authorization"] == f"Bearer {post_refresh_token}"
        # auth.json was updated with the new tokens.
        persisted = json.loads(auth_path.read_text())
        assert persisted["tokens"]["access_token"] == post_refresh_token
        assert persisted["tokens"]["refresh_token"] == "rt-rotated"
        assert persisted["tokens"]["id_token"] == "id-new"
        assert "last_refresh" in persisted


@pytest.mark.asyncio
class TestRefreshFailurePropagates:
    """When refresh itself fails (revoked credentials, no refresh_token),
    the original 401 propagates so service.py's auto-disable still works."""

    async def test_401_with_no_refresh_token_propagates(self, tmp_path):
        original_token = _fake_token()
        auth_path = tmp_path / "auth.json"
        # File has the same access_token AND no refresh_token.
        auth_path.write_text(json.dumps({
            "tokens": {"access_token": original_token},
        }))

        captured: List[Dict[str, str]] = []
        responses = [
            _ScriptedResponse(401, [], text='{"error":{"code":"token_expired"}}'),
        ]
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        with _patch_httpx(responses, captured), \
             patch("kestrel_sovereign.llm.codex_adapter.CODEX_AUTH_FILE", auth_path):
            with pytest.raises(RuntimeError, match="Codex API returned 401"):
                await adapter.get_response(
                    client=original_token,
                    model="gpt-5.5",
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hi"},
                    ],
                )

        # Only one HTTP attempt — refresh wasn't possible, no retry.
        assert len(captured) == 1

    async def test_401_then_oauth_refresh_4xx_propagates(self, tmp_path):
        original_token = _fake_token()
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "tokens": {
                "access_token": original_token,
                "refresh_token": "rt-revoked",
            },
        }))

        captured: List[Dict[str, str]] = []
        responses = [
            _ScriptedResponse(401, [], text='{"error":{"code":"token_expired"}}'),
        ]
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        async def failing_refresh(refresh_token: str) -> Dict[str, Any]:
            raise RuntimeError("OAuth refresh failed with 400: invalid_grant")

        with _patch_httpx(responses, captured), \
             patch("kestrel_sovereign.llm.codex_adapter.CODEX_AUTH_FILE", auth_path), \
             patch("kestrel_sovereign.llm.codex_adapter._refresh_codex_oauth_token", failing_refresh):
            with pytest.raises(RuntimeError, match="Codex API returned 401"):
                await adapter.get_response(
                    client=original_token,
                    model="gpt-5.5",
                    messages=[
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "hi"},
                    ],
                )

        assert len(captured) == 1


@pytest.mark.asyncio
class TestRefreshCacheReuse:
    """After one call refreshes the token, subsequent calls use the cached
    refreshed token instead of replaying the file-read + refresh dance."""

    async def test_subsequent_call_uses_cached_refreshed_token(self, tmp_path):
        original_token = _fake_token()
        new_token = _fake_token()
        assert original_token != new_token

        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps({
            "tokens": {"access_token": new_token, "refresh_token": "rt"},
        }))

        captured: List[Dict[str, str]] = []
        responses = [
            _ScriptedResponse(401, [], text='{"error":{"code":"token_expired"}}'),
            _ScriptedResponse(200, _sse([_completed("resp_first")])),
            _ScriptedResponse(200, _sse([_completed("resp_second")])),
        ]
        adapter = CodexAdapter(continuation_store=InMemoryContinuationStore())

        with _patch_httpx(responses, captured), \
             patch("kestrel_sovereign.llm.codex_adapter.CODEX_AUTH_FILE", auth_path):
            await adapter.get_response(
                client=original_token,
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
            )
            # Second call: provider_registry would still pass the OLD client
            # token, but the adapter must use its in-memory refreshed override
            # so we don't trigger another 401.
            await adapter.get_response(
                client=original_token,
                model="gpt-5.5",
                messages=[{"role": "user", "content": "hi again"}],
            )

        # 3 HTTP attempts total: T1 first 401, T1 retry (success), T2 first
        # try (success — used cached refreshed token, no 401).
        assert len(captured) == 3
        assert captured[0]["Authorization"] == f"Bearer {original_token}"
        assert captured[1]["Authorization"] == f"Bearer {new_token}"
        assert captured[2]["Authorization"] == f"Bearer {new_token}"
