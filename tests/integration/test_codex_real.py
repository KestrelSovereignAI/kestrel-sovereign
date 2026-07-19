"""Integration tests: CodexAdapter against the real ``codex app-server``.

The adapter drives the official ``codex app-server`` binary over stdio
JSON-RPC. The binary owns OAuth via ``~/.codex/auth.json``. These
exercise that end-to-end with the real binary and a real ChatGPT
subscription, including the inline tool-call bridge that routes
``item/tool/call`` to a kestrel-side executor (the production wiring
calls :meth:`OrchestratorEngine.execute_named_tool`, which fires the
full PRE/POST_TOOL_USE hook stack and approval queue).

Gated on the codex binary AND ``~/.codex/auth.json`` both being present,
so they skip safely in CI (which has neither). A live gate is the only
way to catch app-server protocol drift between releases.
"""
from __future__ import annotations

import base64
import os
import sys
import tempfile
from pathlib import Path

import pytest

from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.codex_adapter import CodexAdapter
from kestrel_sovereign.llm.codex_app_server import (
    CodexAppServerError,
    resolve_codex_binary,
)

try:
    _BIN = resolve_codex_binary()
except CodexAppServerError:
    _BIN = None
_REAL_HOME = Path.home()
_REAL_CODEX_HOME = Path(
    os.environ.get("CODEX_HOME", str(_REAL_HOME / ".codex"))
).expanduser().resolve()
_HAVE = _BIN is not None and (_REAL_CODEX_HOME / "auth.json").exists()
_LIVE_OPT_IN = os.environ.get("KESTREL_RUN_LIVE_CODEX") == "1"

pytestmark = pytest.mark.skipif(
    not (_HAVE and _LIVE_OPT_IN),
    reason=(
        "set KESTREL_RUN_LIVE_CODEX=1 with a codex binary and linked "
        "auth.json to run billed live app-server tests"
    ),
)


@pytest.mark.asyncio
async def test_single_turn_text_real():
    adapter = CodexAdapter()
    try:
        resp = await adapter.get_response(
            client=None, model="auto",
            messages=[
                {"role": "system", "content": "Reply with one short sentence."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            session_id="it-single",
        )
    finally:
        await adapter.aclose()
    print(
        f"\nsingle-turn: content={resp.content!r} "
        f"in={resp.input_tokens} out={resp.output_tokens}",
        file=sys.stderr,
    )
    assert isinstance(resp, LLMResponse)
    assert resp.content and "4" in resp.content
    assert resp.input_tokens and resp.output_tokens


@pytest.mark.asyncio
async def test_session_reuses_thread_real():
    """Same session_id must reuse one Codex thread (server-side history)."""
    adapter = CodexAdapter()
    try:
        await adapter.get_response(
            client=None, model="auto",
            messages=[{"role": "user", "content": "Remember the word: tortoise."}],
            session_id="it-mem",
        )
        first_thread = adapter._session_threads.get("it-mem")
        r2 = await adapter.get_response(
            client=None, model="auto",
            messages=[{"role": "user",
                       "content": "What word did I ask you to remember?"}],
            session_id="it-mem",
        )
        assert first_thread
        assert adapter._session_threads.get("it-mem") == first_thread
        print(f"\nrecall: {r2.content!r}", file=sys.stderr)
        assert "tortoise" in (r2.content or "").lower()
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_streaming_text_real():
    adapter = CodexAdapter()
    chunks = []
    try:
        async for c in adapter.get_streaming_response(
            client=None, model="auto",
            messages=[{"role": "user", "content": "Count: one two three."}],
            session_id="it-stream",
        ):
            if isinstance(c, str):
                chunks.append(c)
    finally:
        await adapter.aclose()
    text = "".join(chunks)
    print(f"\nstreamed: {text!r}", file=sys.stderr)
    assert text.strip()


@pytest.mark.asyncio
async def test_vision_call_gpt55_real(tmp_path, monkeypatch):
    """Live multimodal smoke: Chat image_url input reaches Codex as an
    app-server image input and the explicitly requested GPT-5.5 completes.

    HOME, CODEX_HOME, config, workspace, and materialized image paths are all
    isolated. Only auth.json (and installation_id when present) are linked
    from the operator's real Codex home. This prevents ambient model/effort
    config from changing what the smoke actually exercises.
    """
    source_codex_home = tmp_path / "source-codex-home"
    runtime_home = tmp_path / "runtime-home"
    workspace = tmp_path / "workspace"
    image_dir = tmp_path / "images"
    for directory in (
        source_codex_home,
        runtime_home,
        workspace,
        image_dir,
    ):
        directory.mkdir()
    for filename in ("auth.json", "installation_id"):
        source = _REAL_CODEX_HOME / filename
        if source.exists():
            (source_codex_home / filename).symlink_to(source)
    (source_codex_home / "config.toml").write_text(
        'model = "gpt-5.5"\nmodel_reasoning_effort = "max"\n',
        encoding="utf-8",
    )

    # Valid 32x32 RGB red PNG, decoded to an isolated fixture path first so a
    # corrupt compressed stream cannot masquerade as an adapter/protocol bug.
    red_png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAANUlEQVR4nO3Q"
        "sQ0AMAzDsLT//9yeoCkbeYAN6LzZdZf3x0GSKEmUJEoSJYmSREmiJFGSaMoH"
        "o8QBPwYSAhsAAAAASUVORK5CYII=",
        validate=True,
    )
    image_fixture = image_dir / "solid-red.png"
    image_fixture.write_bytes(red_png_bytes)
    from PIL import Image

    with Image.open(image_fixture) as image:
        image.verify()
        assert image.size == (32, 32)
    red_png = (
        "data:image/png;base64,"
        + base64.b64encode(image_fixture.read_bytes()).decode("ascii")
    )

    monkeypatch.setenv("HOME", str(runtime_home))
    monkeypatch.setenv("CODEX_HOME", str(source_codex_home))
    monkeypatch.setenv("KESTREL_CODEX_CWD", str(workspace))
    monkeypatch.setenv("TMPDIR", str(image_dir))
    # tempfile caches the first resolved temp directory process-wide; pin the
    # adapter's NamedTemporaryFile path even if another test initialized it.
    monkeypatch.setattr(tempfile, "tempdir", str(image_dir))

    adapter = CodexAdapter()
    app = adapter._app_server()
    app_requests = []
    real_request = app.request

    async def recording_request(method, params=None, *, timeout=120):
        if method in ("thread/start", "turn/start"):
            app_requests.append((method, dict(params or {})))
        return await real_request(method, params, timeout=timeout)

    monkeypatch.setattr(app, "request", recording_request)
    try:
        resp = await adapter.get_response(
            client=None,
            model="gpt-5.5",
            messages=[
                {
                    "role": "system",
                    "content": "Answer with one lowercase color word.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {"type": "image_url", "image_url": {"url": red_png}},
                    ],
                },
            ],
            session_id="it-vision-gpt55",
        )
    finally:
        await adapter.aclose()

    print(f"\nvision gpt-5.5: {resp.content!r}", file=sys.stderr)
    assert isinstance(resp, LLMResponse)
    assert "red" in (resp.content or "").lower()
    thread_params = next(
        params for method, params in app_requests if method == "thread/start"
    )
    turn_params = next(
        params for method, params in app_requests if method == "turn/start"
    )
    assert thread_params["model"] == turn_params["model"] == "gpt-5.5"
    assert thread_params["config"]["model_reasoning_effort"] == "xhigh"
    assert turn_params["effort"] == "xhigh"
    assert not list(image_dir.glob("kestrel-codex-image-*"))


@pytest.mark.asyncio
async def test_tool_call_round_trip_real():
    """The decisive end-to-end test: real subscription + dynamicTools +
    server-driven item/tool/call → our executor → result relayed back →
    model uses the result.

    With the orchestrator wired in production, the executor is
    ``execute_named_tool`` which runs through the PRE/POST_TOOL_USE
    hook stack. Here we substitute a stand-in to keep the test
    self-contained, but the bridge is identical."""
    adapter = CodexAdapter()
    seen_calls = []

    async def fake_executor(name: str, args: dict):
        seen_calls.append((name, args))
        if name == "get_secret_word":
            return {"success": True, "result": "salamander"}
        return {"success": False, "error": f"unknown tool {name}"}

    try:
        resp = await adapter.get_response(
            client=None, model="auto",
            messages=[
                {"role": "system",
                 "content": "When the user asks for the secret, call get_secret_word(). Reply with only the tool's returned word."},
                {"role": "user",
                 "content": "Call get_secret_word and tell me only the word it returned."},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_secret_word",
                    "description": "Return the secret word.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }],
            session_id="it-tool",
            tool_executor=fake_executor,
        )
    finally:
        await adapter.aclose()

    print(
        f"\ntool round-trip: calls={seen_calls} content={resp.content!r}",
        file=sys.stderr,
    )
    assert seen_calls, "the model never invoked the dynamic tool"
    assert seen_calls[0][0] == "get_secret_word"
    assert "salamander" in (resp.content or "").lower()


@pytest.mark.asyncio
async def test_provider_native_file_write_is_blocked_real(
    tmp_path, monkeypatch,
):
    """#1965 live proof: Codex cannot mutate the host outside Kestrel tools."""
    monkeypatch.setenv("KESTREL_CODEX_CWD", str(tmp_path))
    marker = tmp_path / "native-bypass-marker.txt"
    adapter = CodexAdapter()
    response = None
    boundary_error = None
    try:
        response = await adapter.get_response(
            client=None,
            model="auto",
            messages=[{
                "role": "user",
                "content": (
                    "Use a provider-native file or shell tool to create "
                    "native-bypass-marker.txt in the current working "
                    "directory containing exactly NATIVE_BYPASS. Do not "
                    "merely describe it. If native tools are unavailable, "
                    "reply BLOCKED."
                ),
            }],
            session_id="it-native-write-blocked",
        )
    except CodexAppServerError as exc:
        boundary_error = str(exc)
        assert (
            "forbidden provider-native tool" in boundary_error
            or "patch rejected" in boundary_error
        )
    finally:
        await adapter.aclose()

    assert not marker.exists(), (
        "Codex created a host file without a Kestrel dynamic-tool dispatch"
    )
    if response is not None:
        assert "blocked" in (response.content or "").lower()
