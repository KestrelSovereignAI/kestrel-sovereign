"""#3740 acceptance: a real turn reads a 50,000-character file in full.

A tool result whose serialized form passes the orchestrator's cap reaches the
model as a head-and-tail preview, and the middle is gone. ``git_show_file`` now
returns the file in ranged chunks that each fit the cap. These turns run the
real orchestrator tool loop at its default settings, on a real agent, against
a real git repository; only the model is scripted, and it does what the tool
result tells it to: call again with ``offset=next_offset`` until that is null,
or, batching, call once for each of ``batch_offsets`` with
``limit=batch_limit`` in one response.

The scripted model is one the orchestrator cannot identify, so unless a test
says otherwise it prunes the turn against its 32,768-token fallback window,
the smallest it uses. Every call's arguments and every pruned result's stub
stay in the turn, so a long enough read fills that window and a batch's first
chunk is pruned before the model reads it. A read does not size itself to the
room its turn has left; that is #3851.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Callable, List, Optional

import pytest

from kestrel_sovereign import kestrel_agent
from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.streaming import is_only_sentinels
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall
from tests.shared.booted_agent import booted_agent

#: The orchestrator's defaults: an 8,000-character result cap and a 50-call
#: tool loop that stops at 90% of its budget or after five low-output
#: reasoning-only calls.
DEFAULT_RESULT_CAP = 8000
DEFAULT_LOOP_SETTINGS = {
    "MAX_TOOL_RESULT_CHARS": DEFAULT_RESULT_CAP,
    "MAX_TOOL_ITERATIONS": 50,
    "KESTREL_BUDGET_STOP_PCT": 90,
    "KESTREL_MAX_LOW_DELTA": 5,
    "KESTREL_DIMINISHING_THRESHOLD": 500,
}

DOC_CHARS = 50_000
DONE = "I have read the whole file."
STOPPED = "I stopped reading:"


class PrunedBeforeRead(AssertionError):
    """The orchestrator pruned a result of this read before the model read it."""


def _document() -> str:
    """A 50,000-character markdown file with the text that inflates under
    JSON escaping: accents, dashes, curly quotes, CJK, emoji outside the BMP,
    backslashes, double quotes, tabs and newlines."""
    sections: List[str] = []
    n = 0
    while sum(len(s) for s in sections) < DOC_CHARS:
        n += 1
        sections.append(
            f"## §{n} — Zürich café notes\n\n"
            f"Paragraph {n}: the naïve reading of the two-brain design is "
            f"wrong; “fast” and “slow” paths share one ledger. 漢字テスト {n}. "
            f"The falcon 🦅 returns to the glove at step {n}.\n\n"
            f"```python\npath = \"C:\\\\data\\\\{n}\"\n\tprint('done', {n})\n```\n\n"
        )
    return "".join(sections)[:DOC_CHARS]


def _astral_document() -> str:
    """The worst case for the cap: every character is outside the BMP and
    serializes to twelve, so 50,000 of them are 600,000 characters."""
    return "🦅" * DOC_CHARS


def _git(repo, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
        capture_output=True,
        check=True,
    )


def _commit_document(tmp_path, document: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs").mkdir()
    (repo / "docs" / "TWO_BRAIN.md").write_bytes(document.encode("utf-8"))
    _git(repo, "init", "-b", "main")
    _git(repo, "add", "docs/TWO_BRAIN.md")
    _git(repo, "commit", "-m", "add the document")
    return repo


@pytest.fixture
def default_loop_settings(monkeypatch):
    """Pin the cap and the loop's stop rules at their defaults, wherever a
    host environment set them."""
    oe._init_constants()
    for module in (kestrel_agent, oe):
        for name, value in DEFAULT_LOOP_SETTINGS.items():
            monkeypatch.setattr(module, name, value)


class _PagingModel:
    """A model that reads one file to its end as the tool results direct.

    Every call sees the turn's real message list: the tool results it parses
    are the exact text the orchestrator put in front of the model. Sequential,
    it follows ``next_offset``; batched, it makes one call per entry of
    ``batch_offsets`` with ``limit=batch_limit``, all in one response, and
    follows ``next_offset`` when a chunk offers no batch.
    """

    def __init__(self, repo_path: str, path: str, *, batched: bool) -> None:
        self.repo_path = repo_path
        self.path = path
        self.batched = batched
        self.tool_messages: List[str] = []
        self.requests: List[dict] = []
        self.offered: List[set] = []
        self.calls = 0
        self._pending = 0

    def _read(self, pages: List[dict]) -> LLMResponse:
        tool_calls = []
        for page in pages:
            arguments: dict[str, Any] = {
                "ref": "HEAD",
                "path": self.path,
                "repo_path": self.repo_path,
                **page,
            }
            self.requests.append(arguments)
            tool_calls.append(
                ToolCall(
                    id=f"read-{len(self.requests)}",
                    name="git_show_file",
                    arguments=arguments,
                )
            )
        self._pending = len(tool_calls)
        return LLMResponse(content="", tool_calls=tool_calls)

    def _respond(self, kwargs: dict) -> LLMResponse | str:
        self.calls += 1
        self.offered.append(
            {tool["function"]["name"] for tool in kwargs.get("tools") or []}
        )
        messages = kwargs.get("messages") or [{}]
        if messages[-1].get("role") != "tool":
            # The first call: read from the start, passing no offset at all.
            return self._read([{}])
        # This response's results: the tool messages since the model last
        # spoke, in the order it made the calls.
        results: List[str] = []
        for message in reversed(messages):
            if message.get("role") != "tool":
                break
            results.insert(0, message["content"])
        assert len(results) == self._pending, (len(results), self._pending)
        for content in results:
            assert not content.startswith("[Large tool output"), (
                "the orchestrator replaced the tool result with its preview"
            )
            if content.startswith("[Result truncated"):
                raise PrunedBeforeRead(
                    "the orchestrator pruned a result before the model read it"
                )
        self.tool_messages.extend(results)
        last = json.loads(results[-1])
        if last["status"] != "ok":
            return f"{STOPPED} {last['error']}"
        data = last["data"]
        if data["next_offset"] is None:
            return DONE
        if self.batched and "batch_offsets" in data:
            return self._read(
                [
                    {"offset": offset, "limit": data["batch_limit"]}
                    for offset in data["batch_offsets"]
                ]
            )
        return self._read([{"offset": data["next_offset"]}])

    async def generate_with_messages(self, **kwargs):
        reply = self._respond(kwargs)
        return LLMResponse(content=reply) if isinstance(reply, str) else reply

    async def stream_with_tool_detection(self, **kwargs):
        yield self._respond(kwargs)


async def _non_streaming_turn(agent, text: str) -> str:
    return await agent.process_input(text, session_id="paged-read")


async def _streaming_turn(agent, text: str) -> str:
    chunks = [
        chunk
        async for chunk in agent.process_input_streaming(text, session_id="paged-read")
    ]
    return "".join(
        chunk for chunk in chunks
        if isinstance(chunk, str) and not is_only_sentinels(chunk)
    )


async def _run_read_turn(
    tmp_path,
    monkeypatch,
    turn: Callable,
    document: str,
    *,
    batched: bool,
    context_tokens: Optional[int] = None,
):
    """Run one real turn whose model reads the committed ``document``.

    ``context_tokens`` is the model's context window, which the orchestrator
    prunes old tool results against; left unset, the scripted model is one it
    cannot identify and it takes its 32,768-token fallback.

    Returns the turn's reply, the model, and the repository. The model has
    already checked that every result it saw was whole.
    """
    repo = _commit_document(tmp_path, document)
    monkeypatch.setenv("KESTREL_CLI_ALLOWED_REPO_ROOTS", str(repo))
    assert any(ord(ch) > 0xFFFF for ch in document)

    async with booted_agent(
        tmp_path / "agent", name="PagedRead",
        provenance="test:git_show_file_paged_turn", features={"CliFeature"},
    ) as agent:
        # The state after the turn's first ``cli`` dispatch: the feature's
        # own tools are mounted, so the model is offered git_show_file.
        cli = next(
            feature for feature in agent.features.values()
            if type(feature).__name__ == "CliFeature"
        )
        agent._register_explored_feature_tools(cli)
        model = _PagingModel(str(repo), "docs/TWO_BRAIN.md", batched=batched)
        if context_tokens is not None:
            # The window the orchestrator honours ahead of any lookup.
            monkeypatch.setattr(
                agent.llm_service, "_context_limit", context_tokens, raising=False
            )
        monkeypatch.setattr(
            agent.llm_service, "generate_with_messages", model.generate_with_messages
        )
        monkeypatch.setattr(
            agent.llm_service,
            "stream_with_tool_detection",
            model.stream_with_tool_detection,
        )

        reply = await turn(agent, "Read docs/TWO_BRAIN.md at HEAD, all of it.")

    # The loop ran to the model's own answer, not to a budget stop. (A
    # streamed reply puts a separator between tool rounds and its answer.)
    assert "diminishing returns" not in reply
    assert "Maximum tool call iterations" not in reply
    assert model.offered and all(
        "git_show_file" in offered for offered in model.offered
    ), model.offered
    assert len(model.tool_messages) == len(model.requests)
    return reply, model, repo


def _chunks_read(model: _PagingModel, total: int, cap: int) -> List[str]:
    """The chunks the model read, in order, checked as one contiguous read
    from the start of a ``total``-character file: each call returned what it
    asked for, where it asked."""
    chunks = []
    expected_offset = 0
    for request, message in zip(model.requests, model.tool_messages):
        # The model saw each result whole: under the cap, and JSON rather
        # than the orchestrator's head-and-tail preview or pruning stub.
        assert len(message) <= cap
        result = json.loads(message)
        if result["status"] != "ok":
            break
        data = result["data"]
        assert data["offset"] == request.get("offset", 0) == expected_offset
        assert data["total_chars"] == total
        assert data["returned_chars"] == len(data["content"]) > 0
        if "limit" in request:
            # A batched call holds exactly what it asked for.
            assert data["returned_chars"] == min(
                request["limit"], total - expected_offset
            )
        expected_offset += data["returned_chars"]
        assert data["next_offset"] == (
            expected_offset if expected_offset < total else None
        )
        chunks.append(data["content"])
    return chunks


async def _read_in_a_turn(
    tmp_path,
    monkeypatch,
    turn: Callable,
    document: str,
    *,
    cap: int = DEFAULT_RESULT_CAP,
    **kwargs,
):
    """Run one real turn whose model reads the whole committed ``document``,
    under the tool-result ``cap`` the test configured.

    Returns the turn's reply, the model, and the tool results it saw in order.
    """
    reply, model, repo = await _run_read_turn(
        tmp_path, monkeypatch, turn, document, **kwargs
    )

    assert reply.strip().endswith(DONE), reply
    chunks = _chunks_read(model, len(document), cap)
    assert len(chunks) == len(model.tool_messages)
    assert "".join(chunks) == document
    assert "".join(chunks).encode("utf-8") == (
        repo / "docs" / "TWO_BRAIN.md"
    ).read_bytes()
    return reply, model, model.tool_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", [_non_streaming_turn, _streaming_turn])
async def test_a_turn_reads_a_50k_file_in_full_through_ranged_chunks(
    tmp_path, monkeypatch, default_loop_settings, turn,
):
    _, _, messages = await _read_in_a_turn(
        tmp_path, monkeypatch, turn, _document(), batched=False
    )

    for index, message in enumerate(messages[:-1]):
        # As large as fits: one more character (at most 12 escaped, plus
        # a digit in each offset the result names) would pass the cap.
        assert len(message) > DEFAULT_RESULT_CAP - 20, (index, len(message))
    # Escaped, this text is 1.4 times its length, and still takes 10 calls.
    assert len(messages) <= 10, len(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", [_non_streaming_turn, _streaming_turn])
async def test_a_turn_reads_a_50k_file_in_full_through_batched_chunks(
    tmp_path, monkeypatch, default_loop_settings, turn,
):
    """Batched, three chunks to a response, the same file takes four reads
    and the model's answer, inside the fallback window."""
    _, model, messages = await _read_in_a_turn(
        tmp_path, monkeypatch, turn, _document(), batched=True
    )

    assert model.calls <= 5, model.calls
    assert len(messages) >= 10


#: A configured cap below 1,000 (KESTREL_MAX_TOOL_RESULT_CHARS is settable).
#: The orchestrator previews any result over the cap it is given, with no
#: floor. At this cap every page carries little of the file beside its
#: envelope, so the file here is short enough to read in one turn.
SMALL_RESULT_CAP = 900
SMALL_DOC_CHARS = 800


@pytest.fixture
def small_cap_loop_settings(monkeypatch, default_loop_settings):
    """The loop's defaults, with the cap configured below 1,000."""
    for module in (kestrel_agent, oe):
        monkeypatch.setattr(module, "MAX_TOOL_RESULT_CHARS", SMALL_RESULT_CAP)


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True], ids=["sequential", "batched"])
@pytest.mark.parametrize("turn", [_non_streaming_turn, _streaming_turn])
async def test_a_turn_reads_a_file_whole_under_a_cap_below_1000(
    tmp_path, monkeypatch, small_cap_loop_settings, turn, batched,
):
    """#3740: pages are sized to the cap the orchestrator applies.

    Paging once sized against a floor of 1,000 that the orchestrator does not
    have: at a cap of 900 the first page of this file was 997 characters, and
    the model was given the orchestrator's preview of it instead.
    """
    document = _document()[:SMALL_DOC_CHARS]

    _, _, messages = await _read_in_a_turn(
        tmp_path, monkeypatch, turn, document, batched=batched, cap=SMALL_RESULT_CAP
    )

    assert len(messages) > 1


#: A model's own context window, such as the orchestrator resolves for a model
#: it can identify, in place of its 32,768-token fallback.
MODEL_WINDOW_TOKENS = 200_000


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", [_non_streaming_turn, _streaming_turn])
async def test_a_turn_reads_50k_characters_outside_the_bmp_through_batched_chunks(
    tmp_path, monkeypatch, default_loop_settings, turn,
):
    """#3740: batching keeps a heavily escaped file inside the turn's budget.

    Serialized, these 50,000 characters are 600,000: about eighty chunks, more
    than the loop's 45 responses allow one chunk at a time (a turn following
    only ``next_offset`` stops with "diminishing returns" before the end).
    Three to a response they take about 28 batches.

    The model's window is set here: in the fallback window the batches lose a
    chunk to the prune before the end (see the next test).
    """
    _, model, messages = await _read_in_a_turn(
        tmp_path,
        monkeypatch,
        turn,
        _astral_document(),
        batched=True,
        context_tokens=MODEL_WINDOW_TOKENS,
    )

    assert len(messages) > 45
    # The reads plus the model's answer, inside the loop's 45. (The count
    # moves with the length of repo_path, which every chunk carries.)
    assert model.calls < 35, model.calls


@pytest.mark.asyncio
@pytest.mark.xfail(
    raises=PrunedBeforeRead,
    strict=True,
    reason=(
        "#3851: a batch is sized to the tool-result cap, not to the room the "
        "turn has left, so the fallback window prunes a chunk unread"
    ),
)
@pytest.mark.parametrize("turn", [_non_streaming_turn, _streaming_turn])
async def test_a_turn_reads_50k_characters_outside_the_bmp_in_the_fallback_window(
    tmp_path, monkeypatch, default_loop_settings, turn,
):
    """The same read in the 32,768-token fallback window, which the
    orchestrator prunes against for a model it cannot identify.

    Eighty calls' own arguments and pruned-result stubs, about 20,000
    characters, stay in the turn beside this agent's 39,000 characters of
    system prompt and request. Measured: three chunks to a response, the
    batch reading from offset 29,500 has its first chunk pruned before the
    model reads it; two to a response, the same happens near offset 46,400.
    Sizing a read to the room its turn has left is #3851.
    """
    await _read_in_a_turn(
        tmp_path, monkeypatch, turn, _astral_document(), batched=True
    )
