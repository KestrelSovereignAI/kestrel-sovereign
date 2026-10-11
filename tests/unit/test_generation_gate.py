"""#3552: one gate decides whether a generation becomes an answer.

``judge_generation`` accepts a generation only on positive evidence that it
finished: the provider's terminal stop reason is a natural end. A stop at an
output cap, an unknown or missing stop reason, a provider timeout, the
orchestrator's inactivity bound and a Stop are all failed attempts.

The first part of this file pins the decision itself. The second reads the
code of every registered answer path (``ANSWER_PATHS``) and checks it reaches
the gate, and that no model call the package makes through an LLM service
reference sits on an unregistered one.
The third is the table the ruling asked for: every path a turn's answer can
take x every way a generation can end. Each case drives a real agent on real
storage through the real LLM service and real Ollama adapters, whose daemon
is an in-process HTTP handler; only what the daemon answers is scripted.
Whatever the path, a generation that did not end naturally is never recorded
as (part of) an answer, a tool batch that already ran is recorded as its fixed
checkpoint, and the turn lock is released. A natural end is the answer.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import json
import os
import pathlib
import re
import textwrap
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from kestrel_sovereign.agent import orchestrator_engine as oe
from kestrel_sovereign.agent.streaming import (
    INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT,
    STRICT_AUDIT_CANCELLED_TOOL_BATCH_CHECKPOINT,
)
from kestrel_sovereign.llm.adapter import LLMResponse
from kestrel_sovereign.llm.generation_gate import (
    ANSWER_PATHS,
    NATURAL_STOP_REASONS,
    AnswerPath,
    GenerationCancelledError,
    NoCompletionEvidenceError,
    UnfinishedStopError,
    failure_summary,
    first_unfinished_attempt,
    judge_generation,
)
from kestrel_sovereign.llm.output_ceiling import (
    IncompleteGenerationError,
    OutputCapReachedError,
    attach_stop_reason,
)


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


def _ended(stop_reason: Optional[str]) -> LLMResponse:
    return attach_stop_reason(LLMResponse(content="text"), stop_reason)


@pytest.mark.parametrize("stop_reason", sorted(NATURAL_STOP_REASONS))
def test_a_natural_end_is_accepted(stop_reason):
    assert judge_generation(_ended(stop_reason), provider="p", model="m") is None


@pytest.mark.parametrize("stop_reason", [
    "max_tokens", "length", "MAX_TOKENS", "model_context_window_exceeded",
])
def test_a_stop_at_an_output_limit_is_a_failed_attempt(stop_reason):
    failure = judge_generation(_ended(stop_reason), provider="p", model="m")
    assert isinstance(failure, OutputCapReachedError)
    assert failure.stop_reason == stop_reason
    assert failure_summary(failure) == "stopped at its output limit"


@pytest.mark.parametrize("stop_reason", [
    "refusal", "pause_turn", "SAFETY", "content_filter", "interrupted",
    "unload", "a-reason-nobody-has-invented-yet",
])
def test_any_other_stop_reason_is_a_failed_attempt(stop_reason):
    failure = judge_generation(_ended(stop_reason), provider="p", model="m")
    assert isinstance(failure, UnfinishedStopError)
    assert failure.stop_reason == stop_reason


@pytest.mark.parametrize("response", [
    None, "a bare string", LLMResponse(content="no stop reason"), _ended(""),
])
def test_no_evidence_is_a_failed_attempt(response):
    assert isinstance(
        judge_generation(response, provider="p", model="m"),
        NoCompletionEvidenceError,
    )


def test_a_stop_is_a_failed_attempt_even_after_a_natural_end():
    failure = judge_generation(_ended("stop"), cancelled=True)
    assert isinstance(failure, GenerationCancelledError)


def test_an_error_is_a_failed_attempt():
    plain = RuntimeError("HTTP 500 from the provider")
    assert judge_generation(error=plain) is plain
    assert failure_summary(plain) == "failed"

    cap = OutputCapReachedError(provider="ollama", model="m", cap=64)
    wrapper = RuntimeError("route failed")
    wrapper.__cause__ = cap
    assert judge_generation(error=wrapper) is cap
    assert failure_summary(wrapper) == "stopped at the output cap of 64 tokens"


def test_every_failure_is_an_incomplete_generation_but_a_plain_error():
    for failure in (
        judge_generation(_ended("length")),
        judge_generation(_ended("unload")),
        judge_generation(_ended(None)),
        judge_generation(cancelled=True),
    ):
        assert isinstance(failure, IncompleteGenerationError)


def test_an_aggregate_verdict_is_the_first_unfinished_attempt_whichever_came_last():
    cap = OutputCapReachedError(provider="ollama", model="m", cap=64)
    wrapped_cap = RuntimeError("route failed")
    wrapped_cap.__cause__ = cap
    routes = [RuntimeError("HTTP 500"), wrapped_cap, RuntimeError("HTTP 404")]

    assert first_unfinished_attempt(routes) is cap
    assert first_unfinished_attempt([RuntimeError("a"), RuntimeError("b")]) is None


# --------------------------------------------------------------------------
# The registry: every answer path reaches the gate
# --------------------------------------------------------------------------
#
# ``ANSWER_PATHS`` names every path by which a generation becomes a turn's
# answer or a completed-action checkpoint. These tests read the code itself:
# each registered path makes the calls it says reach the gate, each of those
# leads to ``judge_generation``, and every model call the package makes
# through an LLM service reference sits on a registered path, as one of that
# path's declared gates, or is declared here not to produce an answer.

#: Model calls that produce no turn answer and no completed-action
#: checkpoint, by ``module:Qualified.function``, and why. The answer gate does
#: not govern them (#3573 reviews their own completion handling). A new model
#: call that does produce an answer registers in ``ANSWER_PATHS`` instead.
NON_ANSWER_MODEL_CALLS = {
    "kestrel_sovereign.agent.context_manager:"
    "ContextManager.start_salvage_worker._llm_completion":
        "summarizes already-pruned history for durable salvage",
    "kestrel_sovereign.agent.conversation_manager:ConversationManager.compact_session":
        "a compaction summary of earlier turns",
    "kestrel_sovereign.agent.conversation_manager:ConversationManager.summarize_messages":
        "a summary of earlier turns",
    "kestrel_sovereign.agent.memory_manager:MemoryManager._summarize_chunk":
        "a memory chunk's summary",
    "kestrel_sovereign.bootstrap.service:BootstrapService.generate_soul_md":
        "writes SOUL.md once discovery completes; it is not a reply",
    "kestrel_sovereign.features.base:Feature.execute_as_subagent":
        "a feature subagent's result is a tool result the turn's next model "
        "call reads, not the turn's answer",
    "kestrel_sovereign.features.base:Feature._handle_feature_tool_calls":
        "a feature subagent's tool loop (see execute_as_subagent)",
    "kestrel_sovereign.features.base:Feature._repair_subagent_premature_yield":
        "a feature subagent's repair (see execute_as_subagent)",
    "kestrel_sovereign.features.consent.feature:ConsentFeature._generate_consent":
        "the agent's view recorded on a consent request",
    "kestrel_sovereign.features.context.feature:ContextFeature.recursive_query":
        "a context tool's result",
    "kestrel_sovereign.features.memory.reflection_hook:"
    "ReflectionSleepHook._attest_application":
        "a sleep-time attestation over retrieved memories",
    "kestrel_sovereign.storage.memory_answerability:LLMAnswerabilityGate.filter":
        "a retrieval filter's decision",
}

#: The LLM service's generation calls.
_MODEL_CALLS = frozenset({
    "generate", "generate_stream", "generate_with_messages", "get_response",
    "get_response_with_model", "stream_with_messages",
    "stream_with_tool_detection",
})

_PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "kestrel_sovereign"


def _qualname(path: AnswerPath) -> str:
    return path.function.split(":", 1)[1]


def _short_name(path: AnswerPath) -> str:
    return _qualname(path).rsplit(".", 1)[-1]


def _resolve(path: AnswerPath) -> Callable[..., Any]:
    module_name, qualname = path.function.split(":", 1)
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return inspect.unwrap(target)


def _called_name(call: ast.Call) -> Optional[str]:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _parse_gate(gate: str) -> tuple:
    """``(name, keyword or None, number of calls)`` of a declared gate."""
    match = re.fullmatch(r"(\w+)(?:\((\w+)=True\))?(?:\*(\d+))?", gate)
    assert match, f"malformed gate {gate!r}"
    return match.group(1), match.group(2), int(match.group(3) or 1)


def _function_calls(path: AnswerPath) -> tuple:
    """Every call in ``path``'s function, and the names it calls anywhere.

    Each call comes as ``(call, discarded, nested)``: ``discarded`` when the
    call is a statement of its own, so its result is dropped, and ``nested``
    the names of the functions defined inside ``path``'s that it sits in.
    """
    root = ast.parse(textwrap.dedent(inspect.getsource(_resolve(path)))).body[0]
    calls: List[tuple] = []

    def visit(node: ast.AST, nested: tuple) -> None:
        for child in ast.iter_child_nodes(node):
            inner = nested
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = nested + (child.name,)
            if isinstance(child, ast.Call):
                calls.append((child, isinstance(node, ast.Expr), nested))
            visit(child, inner)

    visit(root, ())
    called = {_called_name(call) for call, _, _ in calls}
    return calls, called


def test_every_answer_path_resolves_and_is_registered_once():
    names = [path.name for path in ANSWER_PATHS]
    functions = [path.function for path in ANSWER_PATHS]
    short_names = [_short_name(path) for path in ANSWER_PATHS]
    assert len(set(names)) == len(names)
    assert len(set(functions)) == len(functions)
    # A gate names another path's function by its name alone.
    assert len(set(short_names)) == len(short_names)
    for path in ANSWER_PATHS:
        assert callable(_resolve(path)), path
        assert path.gates, path
        gate_names = [_parse_gate(gate)[0] for gate in path.gates]
        assert len(set(gate_names)) == len(gate_names), path


@pytest.mark.parametrize("path", ANSWER_PATHS, ids=lambda path: path.name)
def test_every_answer_path_makes_the_calls_that_reach_the_gate(path):
    calls, called = _function_calls(path)
    registered = {_short_name(other) for other in ANSWER_PATHS}
    where = f"{path.name} ({path.function})"
    for gate in path.gates:
        name, keyword, count = _parse_gate(gate)
        assert name == "judge_generation" or name in registered, (
            f"{where}: {name} is neither the gate nor a registered path"
        )
        made = [entry for entry in calls if _called_name(entry[0]) == name]
        # One call per place the function's answer can come from: a place
        # added without its gate shows up as a count that no longer matches.
        assert len(made) == count, (
            f"{where} calls {name} {len(made)} times; its registration "
            f"declares {count}"
        )
        for call, discarded, nested in made:
            line = f"{where}, line {call.lineno}"
            # A gate call inside a function that is never called gates nothing.
            assert set(nested) <= called, f"{line}: {name} sits in an uncalled function"
            if keyword is not None:
                passed = {kw.arg: kw.value for kw in call.keywords}
                value = passed.get(keyword)
                assert isinstance(value, ast.Constant) and value.value is True, (
                    f"{line}: a call to {name} does not pass {keyword}=True"
                )
            if name == "judge_generation":
                # The gate returns its verdict rather than raising it.
                assert not discarded, f"{line}: judge_generation's verdict is dropped"


def test_every_answer_path_leads_to_the_gate():
    by_name = {_short_name(path): path for path in ANSWER_PATHS}

    def reaches(name: str, seen: frozenset) -> bool:
        if name == "judge_generation":
            return True
        if name in seen or name not in by_name:
            return False
        return any(
            reaches(_parse_gate(gate)[0], seen | {name})
            for gate in by_name[name].gates
        )

    unreached = [path.name for path in ANSWER_PATHS if not reaches(_short_name(path), frozenset())]
    assert not unreached, unreached


def _is_llm_service(expr: ast.AST, aliases: frozenset) -> bool:
    """Whether ``expr`` is an LLM service reference: an ``llm_service`` name
    or attribute (``self._llm_service`` too), or a local alias of one."""
    if isinstance(expr, ast.Name):
        return expr.id.endswith("llm_service") or expr.id in aliases
    if isinstance(expr, ast.Attribute):
        return expr.attr.endswith("llm_service")
    return False


def _model_call_sites() -> List[tuple]:
    """``(module:qualname of the enclosing function, generation call, line)``
    for every generation call the package makes through an LLM service
    reference."""
    sites = []
    for source in sorted(_PACKAGE.rglob("*.py")):
        module = ".".join(source.relative_to(_PACKAGE.parent).with_suffix("").parts)
        tree = ast.parse(source.read_text(encoding="utf-8"))

        def visit(node: ast.AST, scope: List[str], aliases: frozenset) -> None:
            for child in ast.iter_child_nodes(node):
                inner, inner_aliases = scope, aliases
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inner = scope + [child.name]
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # ``llm = self.llm_service`` and the like.
                    inner_aliases = aliases | {
                        target.id
                        for assign in ast.walk(child)
                        if isinstance(assign, ast.Assign)
                        and _is_llm_service(assign.value, aliases)
                        for target in assign.targets
                        if isinstance(target, ast.Name)
                    }
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in _MODEL_CALLS
                    and _is_llm_service(child.func.value, aliases)
                ):
                    sites.append(
                        (f"{module}:{'.'.join(scope)}", child.func.attr, child.lineno)
                    )
                visit(child, inner, inner_aliases)

        visit(tree, [], frozenset())
    return sites


def test_every_model_call_is_on_an_answer_path_or_declared_not_to_be_one():
    gates_of = {
        path.function: {_parse_gate(gate)[0] for gate in path.gates}
        for path in ANSWER_PATHS
    }

    def on_answer_path(site: str, api: str) -> bool:
        # The registered function the call sits in (or in a function nested
        # in it) must declare that very call as a gate: an extra, ungated
        # model call on an answer path is as unaccounted as one off it.
        module, qualname = site.split(":", 1)
        parts = qualname.split(".")
        for end in range(1, len(parts) + 1):
            gates = gates_of.get(f"{module}:{'.'.join(parts[:end])}")
            if gates is not None:
                return api in gates
        return False

    sites = _model_call_sites()
    assert sites, "the scan found no model calls at all"
    unaccounted = [
        f"{site} -> {api} (line {line})" for site, api, line in sites
        if not on_answer_path(site, api) and site not in NON_ANSWER_MODEL_CALLS
    ]
    assert not unaccounted, (
        "model calls on no registered answer path: register the path in "
        "kestrel_sovereign.llm.generation_gate.ANSWER_PATHS, or declare in "
        f"NON_ANSWER_MODEL_CALLS why it produces no answer: {unaccounted}"
    )
    stale = set(NON_ANSWER_MODEL_CALLS) - {site for site, _, _ in sites}
    assert not stale, f"declared non-answer model calls that no longer exist: {stale}"


# --------------------------------------------------------------------------
# The table: every path x every way a generation ends
# --------------------------------------------------------------------------

#: The inactivity bound, shrunk so a silent provider trips it quickly. Only
#: the cases that need it run under it (:func:`_bound_for`): a short bound also
#: trips on a provider that answers at once whenever a loaded test host stalls
#: the event loop for longer, so every other case keeps a bound it cannot reach.
TURN_BOUND = 1.0
#: The inactivity bound of every case that does not need the short one.
UNREACHED_BOUND = 60.0
#: A model no discovery has described, so the service passes the turn's tools.
MODEL = "kestrel-gate-local:1b"
#: The generation under test, in two streamed pieces.
PIECES = ("PARTIAL ", "ANSWER")
TEXT = "".join(PIECES)
#: What another route answers, complete, when the service falls back to it.
FALLBACK = "FALLBACK REPLY"
#: What ``/api/agent/invoke`` answers for a stopped turn.
STOPPED = "Request stopped during execution."

CAUSES = ("natural", "cap", "unknown", "missing", "timeout", "watchdog", "cancel")

#: How the generation under test ends, as the daemon reports it.
_DONE_REASON = {"natural": "stop", "cap": "length", "unknown": "unload"}

PATHS = (
    # A streamed turn (the chat UI): its first call is the generation.
    "stream_first",
    # A streamed turn whose first call ran a tool: the follow-up is.
    "stream_follow_up",
    # A streamed turn on two routes: the first streams part of the
    # generation, the second would answer if the service fell back.
    "stream_fallback",
    # ``POST /api/agent/invoke`` (non-streaming): its first call.
    "invoke_first",
    # ``/api/agent/invoke`` after a tool ran: the follow-up.
    "invoke_follow_up",
    # ``/api/agent/invoke`` on two routes, the second answering completely.
    "invoke_fallback",
    # ``/api/agent/invoke`` after a tool ran, on two routes that both fail:
    # the turn sees the service's aggregate of their errors.
    "invoke_aggregate",
    # ``/api/agent/invoke`` whose first call writes a tool call as text: the
    # premature-yield repair is.
    "invoke_repair",
    # A streamed turn whose follow-up, after a tool ran, writes a tool call as
    # text: the repair of that follow-up is.
    "stream_repair",
)

#: Paths on which a tool runs before the generation under test.
TOOL_RAN = ("stream_follow_up", "invoke_follow_up", "invoke_aggregate", "stream_repair")
#: Paths on which the generation under test is a premature-yield repair.
REPAIRED = ("invoke_repair", "stream_repair")
#: A model reply that writes a tool call as text: it executed nothing, so the
#: turn asks the model once more (``_repair_premature_turn_yield``).
MARKUP = '<function_calls><invoke name="state_of_mind"></invoke></function_calls>'


def _bound_for(path: str, cause: str) -> float:
    """The inactivity bound a table case runs under.

    ``watchdog`` tests the bound. A Stop on a streamed turn whose provider is
    silent also ends only when the bound trips: the Ollama adapter does not
    honour the cancel token, and the streamed turn checks Stop between items
    (#3572). The streamed fallback case is not silent: it streams its pieces.
    """
    if cause == "watchdog":
        return TURN_BOUND
    if cause == "cancel" and path in ("stream_first", "stream_follow_up", "stream_repair"):
        return TURN_BOUND
    return UNREACHED_BOUND


class _Body(httpx.AsyncByteStream):
    """A daemon response body: ``chunks``, then how it ends (``tail``)."""

    def __init__(self, chunks: List[bytes], tail: Any = None) -> None:
        self._chunks = chunks
        self._tail = tail

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        if self._tail == "hang":
            await asyncio.Event().wait()
        elif isinstance(self._tail, BaseException):
            raise self._tail

    async def aclose(self) -> None:
        return None


def _final(done_reason: Optional[str], content: str = "") -> Dict[str, Any]:
    reply: Dict[str, Any] = {
        "model": MODEL,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 11,
        "eval_count": 7,
    }
    if done_reason is not None:
        reply["done_reason"] = done_reason
    return reply


def _line(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _respond(body: _Body) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "application/x-ndjson"}, stream=body,
    )


def _complete(content: str, *, streamed: bool) -> httpx.Response:
    if streamed:
        return _respond(_Body([
            _line({"model": MODEL, "message": {"role": "assistant", "content": content},
                   "done": False}),
            _line(_final("stop")),
        ]))
    return _respond(_Body([_line(_final("stop", content))]))


def _tool_call() -> httpx.Response:
    reply = _final("stop")
    reply["message"]["tool_calls"] = [
        {"function": {"name": "state_of_mind", "arguments": {}}}
    ]
    return _respond(_Body([_line(reply)]))


def _ending(cause: str, *, streamed: bool, stop: Callable[[], None]) -> httpx.Response:
    """The generation under test, ending by ``cause``."""
    if cause == "cancel":
        # Stop lands while the response is arriving.
        stop()
    if streamed:
        chunks = [
            _line({"model": MODEL, "message": {"role": "assistant", "content": piece},
                   "done": False})
            for piece in PIECES
        ]
        content = ""
    else:
        chunks = []
        content = TEXT
    if cause == "timeout":
        return _respond(_Body(chunks, httpx.ReadTimeout("provider read timed out")))
    if cause == "watchdog":
        return _respond(_Body(chunks, "hang"))
    if cause == "cancel":
        # A non-streaming call is cancelled while it waits; a stream delivers
        # its pieces and a natural end, which the Stop must still overrule.
        if not streamed:
            return _respond(_Body(chunks, "hang"))
        return _respond(_Body(chunks + [_line(_final("stop"))]))
    return _respond(_Body(chunks + [_line(_final(_DONE_REASON.get(cause), content))]))


class _Daemon:
    """One route's Ollama daemon. ``answer(body)`` decides each chat reply."""

    def __init__(self, answer: Callable[[Dict[str, Any]], httpx.Response]) -> None:
        self.answer = answer
        self.chats: List[Dict[str, Any]] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/chat":
            # Retrieval embeds the turn on the same route; this daemon is
            # chat-only, as a local install without an embedding model is.
            return httpx.Response(404, json={"error": "model not found"})
        body = json.loads(request.content)
        self.chats.append(body)
        return self.answer(body)


def _route(agent, name: str, daemon: _Daemon) -> dict:
    """A real Ollama route, built as the registry builds one."""
    import ollama

    from kestrel_sovereign.llm.provider_registry import ProviderRegistry

    info = ProviderRegistry({})._build_route(
        "ollama", name, {"is_cloud": False},
        {"adapter": "OllamaAdapter", "host": "http://ollama.test:11434",
         "model": MODEL, "max_output_tokens": 64},
    )
    [provider] = agent.llm_service._convert_providers_format([info])
    provider["client"] = ollama.AsyncClient(
        host="http://ollama.test:11434",
        transport=httpx.MockTransport(daemon.handle),
    )
    return provider


@asynccontextmanager
async def _booted_agent(tmp_path):
    """A real agent on real storage; only the daemons are scripted."""
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.inception_service import create_kestrel_identity_async
    from kestrel_sovereign.kestrel_agent import KestrelAgent
    from kestrel_sovereign.llm.service import LLMService
    from tests.shared.genesis_audit import complete_deterministic_genesis_audit

    credentials = await create_kestrel_identity_async(
        output_dir=str(tmp_path), is_test_instance=True, agent_name="Gate"
    )
    llm_service = LLMService()
    agent = KestrelAgent(
        did=credentials.agent_did,
        storage_path=os.path.join(str(tmp_path), "kestrel_prime.db"),
        llm_service=llm_service,
    )
    try:
        await agent.initialize()
        await complete_deterministic_genesis_audit(agent, provenance="test:generation_gate")
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.COMPLETE)
        yield agent
    finally:
        await agent.shutdown()
        await llm_service.close()


@asynccontextmanager
async def _invoke_client(agent):
    """``/api/agent/invoke`` served for ``agent`` on the test's own loop."""
    from fastapi import FastAPI

    from kestrel_sovereign.api_errors import register_api_error_handlers
    from kestrel_sovereign.endpoints.agent import router
    from kestrel_sovereign.rate_limit import limiter

    app = FastAPI()
    app.state.limiter = limiter
    app.state.agent = agent
    app.include_router(router)
    register_api_error_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        yield client


async def _assistant_rows(agent, session_id: str) -> List[str]:
    history = await agent.privacy_agent.get_conversation_history(
        limit=50, session_id=session_id,
    )
    return [row["content"] for row in history if row.get("role") == "assistant"]


def _conversation_holder(agent):
    from kestrel_sdk.signals import ResourceLock

    return agent._get_lock_manager().holder(ResourceLock.CONVERSATION)


def _is_repair(body: Dict[str, Any]) -> bool:
    """Whether a chat is the turn's premature-yield repair."""
    return any(
        message.get("role") == "assistant" and message.get("content") == MARKUP
        for message in body["messages"]
    )


def _answers(path: str, cause: str, stop: Callable[[], None]):
    """The daemon answers for route A (and B, on two-route paths)."""
    streamed = path == "stream_fallback"

    def route_a(body: Dict[str, Any]) -> httpx.Response:
        if not _is_repair(body):
            if path in TOOL_RAN and body["messages"][-1]["role"] != "tool":
                return _tool_call()
            if path in REPAIRED:
                return _respond(_Body([_line(_final("stop", MARKUP))]))
        return _ending(cause, streamed=streamed, stop=stop)

    def route_b(body: Dict[str, Any]) -> httpx.Response:
        if path == "invoke_aggregate":
            return httpx.Response(404, json={"error": f"model '{MODEL}' not found"})
        return _complete(FALLBACK, streamed=streamed)

    return route_a, route_b


async def _run_turn(agent, path: str, session_id: str) -> Dict[str, Any]:
    """Drive one turn down ``path``; what the caller saw."""
    if path.startswith("stream"):
        chunks: List[Any] = []
        error: Optional[BaseException] = None
        try:
            async for chunk in agent.process_input_streaming(
                "Where do I work?", session_id=session_id, request_id=session_id,
            ):
                chunks.append(chunk)
        except Exception as exc:  # noqa: BLE001 - the outcome under test
            error = exc
        return {
            "ok": error is None,
            "text": "".join(c for c in chunks if isinstance(c, str)),
            "error": error,
        }
    async with _invoke_client(agent) as client:
        response = await client.post(
            "/api/agent/invoke",
            json={"input": "Where do I work?", "session_id": session_id},
        )
    body = response.json()
    return {
        "ok": response.status_code == 200,
        "text": body.get("response") or "" if response.status_code == 200 else "",
        "status": response.status_code,
        "body": body,
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("path", PATHS)
async def test_only_a_natural_end_becomes_the_answer(tmp_path, monkeypatch, path, cause):
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", _bound_for(path, cause))
    # Retries of a transient provider error are not under test; without them
    # a provider timeout fails its attempt at once.
    from kestrel_sovereign.llm import ollama_adapter

    async def no_retry(func, *args, **kwargs):
        return await func(*args, **kwargs)

    monkeypatch.setattr(ollama_adapter, "with_retry", no_retry)

    session_id = f"gate-{path}-{cause}"
    async with _booted_agent(tmp_path) as agent:
        route_a, route_b = _answers(
            path, cause, stop=lambda: agent.cancel_current_request(),
        )
        daemon_a, daemon_b = _Daemon(route_a), _Daemon(route_b)
        service = agent.llm_service
        providers = [_route(agent, "local", daemon_a)]
        if path in ("stream_fallback", "invoke_fallback", "invoke_aggregate"):
            providers.append(_route(agent, "spare", daemon_b))
        monkeypatch.setattr(service, "providers", providers)
        monkeypatch.setattr(service, "config", {
            **(service.config or {}),
            "route_priority": [p["name"] for p in providers],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})
        if path == "stream_fallback":
            # A model without tool support: the service streams its text.
            monkeypatch.setattr(
                service, "_check_model_tool_support",
                lambda providers, tools, model_override: None,
            )

        loop = asyncio.get_running_loop()
        started = loop.time()
        seen = await asyncio.wait_for(_run_turn(agent, path, session_id), 30)
        elapsed = loop.time() - started
        rows = await _assistant_rows(agent, session_id)

        # The turn released its lock, and no provider held it longer than a
        # turn's own work (past the bound, for a silent one).
        assert _conversation_holder(agent) is None
        assert elapsed < TURN_BOUND + 20
        if path in REPAIRED:
            # The generation under test was the repair's.
            assert [_is_repair(chat) for chat in daemon_a.chats][-1:] == [True]

        if cause == "natural":
            assert seen["ok"], seen
            assert TEXT in seen["text"]
            assert rows and TEXT in rows[-1]
            assert not daemon_b.chats, "a route that answered was not the answer"
            return

        # Never accepted: no recorded row carries any of the generation. A
        # streamed turn showed its pieces live; a non-streaming caller is not
        # handed them at all.
        marker = PIECES[0].strip()
        assert not any(marker in row for row in rows), rows
        if path.startswith("invoke"):
            assert marker not in seen["text"], seen
        if path == "invoke_fallback" and cause in ("cap", "unknown", "missing", "timeout"):
            # The failed route contributed nothing; the other route's
            # complete answer stands alone.
            assert seen["ok"] and seen["text"] == FALLBACK, seen
            assert rows == [FALLBACK]
            return
        if cause == "cancel":
            # A Stop ends the turn as stopped rather than as an error: a
            # stream just ends; the invoke door answers its stop notice.
            if path.startswith("invoke"):
                assert seen["text"] == STOPPED, seen
        else:
            assert not seen["ok"], seen
        if path == "stream_fallback":
            # Fallback cannot glue another route's answer onto the part
            # already streamed.
            assert not daemon_b.chats
            assert FALLBACK not in seen["text"]

        if path in TOOL_RAN:
            # The tool ran, so the turn records the fixed checkpoint of it.
            expected = (
                STRICT_AUDIT_CANCELLED_TOOL_BATCH_CHECKPOINT
                if cause == "cancel"
                else INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT
            )
            assert rows == [expected], rows
        else:
            assert all(row == "" for row in rows), rows


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
async def test_a_route_that_fails_before_sending_anything_still_falls_back(
    tmp_path, monkeypatch,
):
    """Falling back is allowed while the failing route has sent nothing: the
    other route's answer then stands alone (#3552)."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", UNREACHED_BOUND)
    async with _booted_agent(tmp_path) as agent:
        failing = _Daemon(lambda body: httpx.Response(
            404, json={"error": f"model '{MODEL}' not found"},
        ))
        answering = _Daemon(lambda body: _complete(FALLBACK, streamed=True))
        service = agent.llm_service
        providers = [_route(agent, "local", failing), _route(agent, "spare", answering)]
        monkeypatch.setattr(service, "providers", providers)
        monkeypatch.setattr(service, "config", {
            **(service.config or {}),
            "route_priority": ["ollama:local", "ollama:spare"],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})
        monkeypatch.setattr(
            service, "_check_model_tool_support",
            lambda providers, tools, model_override: None,
        )

        seen = await asyncio.wait_for(_run_turn(agent, "stream_fallback", "fb"), 30)

        assert seen["ok"], seen
        assert seen["text"] == FALLBACK
        assert await _assistant_rows(agent, "fb") == [FALLBACK]
        assert failing.chats and answering.chats


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("cause", ("natural", "cap", "unknown", "missing", "timeout"))
async def test_a_discovery_reply_becomes_the_answer_only_when_it_finished(
    tmp_path, monkeypatch, cause,
):
    """``bootstrap.discovery``: a new agent's discovery reply is the turn's
    answer only when it ended naturally. One that did not contributes nothing:
    discovery steps aside and the turn answers on its own."""
    from kestrel_sovereign.bootstrap import BootstrapState
    from kestrel_sovereign.llm import ollama_adapter

    async def no_retry(func, *args, **kwargs):
        return await func(*args, **kwargs)

    monkeypatch.setattr(ollama_adapter, "with_retry", no_retry)
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", UNREACHED_BOUND)
    async with _booted_agent(tmp_path) as agent:
        await agent.bootstrap_service.set_bootstrap_state(BootstrapState.DISCOVERY)

        def answer(body: Dict[str, Any]) -> httpx.Response:
            if len(daemon.chats) == 1:
                return _ending(cause, streamed=False, stop=lambda: None)
            return _complete(FALLBACK, streamed=False)

        daemon = _Daemon(answer)
        service = agent.llm_service
        monkeypatch.setattr(service, "providers", [_route(agent, "local", daemon)])
        monkeypatch.setattr(service, "config", {
            **(service.config or {}), "route_priority": ["ollama:local"],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})

        seen = await asyncio.wait_for(_run_turn(agent, "invoke_first", "disc"), 30)
        rows = await _assistant_rows(agent, "disc")

        assert seen["ok"], seen
        if cause == "natural":
            assert seen["text"] == TEXT
            assert rows[-1] == TEXT
            assert len(daemon.chats) == 1
            return
        assert seen["text"] == FALLBACK, seen
        assert not any(PIECES[0].strip() in row for row in rows), rows
        assert len(daemon.chats) == 2
        assert await agent.bootstrap_service.get_bootstrap_state() == BootstrapState.COMPLETE


# --------------------------------------------------------------------------
# A rejected response whose route ran a tool inside its call does not fall back
# --------------------------------------------------------------------------
#
# Some routes (the Codex app-server) run a turn's tools inside one model call,
# through the ``tool_executor`` the service hands the adapter, and report them
# as ``executed_tool_calls`` on the response. When the gate rejects such a
# response the tools may already have acted, so the call ends there rather
# than ask another route, which could act again. (A route that raises after
# its tools ran, and so returns no response, is #3574.)

#: The tool a route's call runs inline before its response is judged.
INLINE_TOOL = ("send_message", {"to": "operator"})


class _InlineToolAdapter:
    """One route's adapter. Its call first hands ``tool`` (name, args) to the
    executor when ``act``, and reports it on its response as Codex does; the
    response then ends by ``ending``: ``"answers"`` (a finished
    :data:`FALLBACK`), ``"unfinished"`` (a stop that is not a natural end) or
    ``"no_evidence"`` (no stop reason at all). Streamed, an unfinished call
    sends nothing but its terminal response."""

    def __init__(self, *, act: bool, ending: str, tool: tuple = INLINE_TOOL):
        self.act = act
        self.tool = tool
        self.ending = ending
        self.calls = 0
        self.tool_results: List[Any] = []

    def create_messages(self, **kwargs):
        return [{"role": "user", "content": kwargs.get("user_prompt", "")}]

    async def _run(self, tool_executor) -> LLMResponse:
        self.calls += 1
        executed = []
        if self.act:
            name, args = self.tool
            result = await tool_executor(name, args)
            self.tool_results.append(result)
            executed.append({"id": "call-1", "name": name, "arguments": args,
                             "result": result})
        if self.ending == "answers":
            response = attach_stop_reason(LLMResponse(content=FALLBACK), "stop")
        elif self.ending == "unfinished":
            response = attach_stop_reason(LLMResponse(content=""), "length")
        else:
            assert self.ending == "no_evidence"
            response = LLMResponse(content="")
        if executed:
            response.executed_tool_calls = executed
        return response

    async def get_response(self, *, tool_executor=None, **kwargs):
        return await self._run(tool_executor)

    async def get_streaming_response_with_tools(self, *, tool_executor=None, **kwargs):
        response = await self._run(tool_executor)
        if response.content:
            yield response.content
        yield response


def _scripted_route(name: str, adapter: Any) -> dict:
    from unittest.mock import AsyncMock

    return {
        "name": name, "vendor": "ollama", "route": name.split(":", 1)[1],
        "client": AsyncMock(), "adapter": adapter, "model": MODEL,
        "is_cloud": False, "is_local": True, "base_url": None,
        "selection_hints": [],
    }


@pytest.fixture
def two_route_service(monkeypatch):
    from unittest.mock import AsyncMock

    from kestrel_sovereign.llm.service import LLMService

    service = LLMService()
    monkeypatch.setattr(service, "_ensure_models_discovered", AsyncMock())
    monkeypatch.setattr(service, "_mandate_preference", {})
    monkeypatch.setattr(service, "config", {
        "route_priority": ["ollama:local", "ollama:spare"],
    })

    def install(first: _InlineToolAdapter) -> _InlineToolAdapter:
        spare = _InlineToolAdapter(act=False, ending="answers")
        monkeypatch.setattr(service, "providers", [
            _scripted_route("ollama:local", first),
            _scripted_route("ollama:spare", spare),
        ])
        return spare

    return service, install


async def _call(service, entry: str, executor) -> Any:
    """``entry`` with ``executor``; its answer, or the text it streamed."""
    messages = [{"role": "user", "content": "send it"}]
    tools = [{"name": INLINE_TOOL[0], "description": "send",
              "parameters": {"type": "object"}}]
    if entry == "generate_with_messages":
        answer = await service.generate_with_messages(
            messages=messages, tools=tools, tool_executor=executor,
        )
        return answer.content
    text = ""
    async for item in service.stream_with_tool_detection(
        messages=messages, tools=tools, tool_executor=executor,
    ):
        if isinstance(item, str):
            text += item
    return text


#: Both route walks a turn's answer comes from, x every way the gate rejects
#: a response.
_REJECTED_AFTER_A_TOOL = [
    (entry, ending)
    for entry in ("generate_with_messages", "stream_with_tool_detection")
    for ending in ("unfinished", "no_evidence")
]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry,ending", _REJECTED_AFTER_A_TOOL)
async def test_a_rejected_response_whose_route_ran_a_tool_ends_the_call(
    two_route_service, entry, ending,
):
    """Another route would be asked again and could act again: the call
    fails, with the gate's verdict as its cause, and no fallback runs."""
    from kestrel_sovereign.llm.generation_gate import (
        UnfinishedAfterInlineToolsError,
    )
    from kestrel_sovereign.llm.streaming import LLMStreamingError

    service, install = two_route_service
    first = _InlineToolAdapter(act=True, ending=ending)
    spare = install(first)
    ran = []

    async def executor(name, args):
        ran.append(name)
        return args, {"sent": True}

    with pytest.raises(Exception) as raised:
        await _call(service, entry, executor)

    failure = raised.value
    if entry == "stream_with_tool_detection":
        assert isinstance(failure, LLMStreamingError), failure
        failure = failure.underlying
    assert isinstance(failure, UnfinishedAfterInlineToolsError), failure
    expected = (
        NoCompletionEvidenceError if ending == "no_evidence"
        else OutputCapReachedError
    )
    assert isinstance(failure.__cause__, expected), failure.__cause__
    assert ran == [INLINE_TOOL[0]], "the tool ran once"
    assert spare.calls == 0, "no other route was asked"


@pytest.mark.asyncio
@pytest.mark.parametrize("entry,ending", _REJECTED_AFTER_A_TOOL)
async def test_a_rejected_response_whose_route_ran_no_tool_falls_back(
    two_route_service, entry, ending,
):
    """The same rejection, with no tool run: the other route's answer stands
    alone, and the rejected response contributes nothing."""
    service, install = two_route_service
    first = _InlineToolAdapter(act=False, ending=ending)
    spare = install(first)

    async def executor(name, args):
        raise AssertionError("no tool runs")

    assert await _call(service, entry, executor) == FALLBACK
    assert first.calls == 1 and spare.calls == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("isolated_process_rate_limiter")
@pytest.mark.parametrize("path", ["stream", "invoke"])
async def test_a_turn_whose_route_ran_a_tool_and_did_not_finish_is_not_answered_elsewhere(
    tmp_path, monkeypatch, path,
):
    """A real turn. Its first route runs a real tool through the turn's own
    inline executor and returns a response the gate rejects; a second route
    would answer. The turn fails, records the tool batch's fixed checkpoint
    and no answer, and the second route is never asked."""
    monkeypatch.setattr(oe, "ORCHESTRATOR_TURN_TIMEOUT_SECS", UNREACHED_BOUND)
    async with _booted_agent(tmp_path) as agent:
        unused = _Daemon(lambda body: httpx.Response(
            404, json={"error": f"model '{MODEL}' not found"},
        ))
        answering = _Daemon(lambda body: _complete(FALLBACK, streamed=False))
        first = _route(agent, "local", unused)
        acting = _InlineToolAdapter(
            act=True, ending="unfinished", tool=("state_of_mind", {}),
        )
        # The route's real adapter, with its call replaced by one that runs
        # the turn's tool inline (instance attributes, as a third-party
        # adapter could define them).
        first["adapter"].get_response = acting.get_response
        first["adapter"].get_streaming_response_with_tools = (
            acting.get_streaming_response_with_tools
        )
        service = agent.llm_service
        providers = [first, _route(agent, "spare", answering)]
        monkeypatch.setattr(service, "providers", providers)
        monkeypatch.setattr(service, "config", {
            **(service.config or {}),
            "route_priority": ["ollama:local", "ollama:spare"],
        })
        monkeypatch.setattr(service, "_mandate_preference", {})

        session_id = f"inline-{path}"
        seen = await asyncio.wait_for(
            _run_turn(agent, f"{path}_first", session_id), 30,
        )

        assert acting.calls == 1
        assert len(acting.tool_results) == 1, "the tool ran once"
        assert not answering.chats, "no other route was asked"
        assert not unused.chats
        assert not seen["ok"], seen
        assert FALLBACK not in seen["text"]
        assert await _assistant_rows(agent, session_id) == [
            INCOMPLETE_GENERATION_TOOL_BATCH_CHECKPOINT
        ]
        assert _conversation_holder(agent) is None

# --------------------------------------------------------------------------
# Adapters report the evidence
# --------------------------------------------------------------------------


class _CodexAppServer:
    """The app-server side of one Codex turn that ends with ``status``."""

    def __init__(self, status: Optional[str]) -> None:
        self.status = status

    async def ensure_started(self) -> None:
        return None

    async def request(self, method, params=None, *, timeout=120):
        if method == "config/read":
            return {"config": {}, "origins": {}}
        if method == "model/list":
            return {"data": [], "nextCursor": None}
        if method == "thread/start":
            return {"thread": {"id": "thr-1"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-1"}}
        raise AssertionError(f"unexpected request: {method}")

    def register_server_request_handler(self, method, handler, *, thread_id=None):
        return lambda: None

    def open_turn_sink(self, thread_id):
        return thread_id

    def close_turn_sink(self, thread_id):
        return None

    async def iter_turn_events(self, sink, *, idle_timeout=120, thread_id=None,
                               cancel_token=None):
        turn = {} if self.status is None else {"status": self.status}
        for event in (
            {"method": "item/agentMessage/delta", "params": {"delta": "ok"}},
            {"method": "item/completed",
             "params": {"item": {"type": "agentMessage", "text": "ok"}}},
            {"method": "turn/completed", "params": {"turn": turn}},
        ):
            yield event


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("status,accepted", [
    ("completed", True), ("interrupted", False), (None, False),
])
async def test_a_codex_turn_reports_its_status_as_the_stop_reason(
    streamed, status, accepted,
):
    from unittest.mock import patch

    from kestrel_sovereign.llm.codex_adapter import CodexAdapter

    adapter = CodexAdapter()
    adapter._client = _CodexAppServer(status)
    kwargs = dict(
        client=None, model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}], session_id=f"s-{streamed}",
    )
    with patch.object(adapter, "_read_codex_models_cache", return_value=[]):
        if streamed:
            items = [i async for i in adapter.get_streaming_response_with_tools(**kwargs)]
            response = items[-1]
        else:
            response = await adapter.get_response(**kwargs)

    assert isinstance(response, LLMResponse) and response.content == "ok"
    assert getattr(response, "stop_reason", None) == status
    assert (judge_generation(response) is None) is accepted


def _vertex_candidate(text: str, finish_reason: str):
    from types import SimpleNamespace

    from google.genai import types

    return SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
        finish_reason=types.FinishReason[finish_reason],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason,accepted", [
    ("STOP", True), ("MAX_TOKENS", False), ("SAFETY", False),
])
async def test_vertex_reports_its_finish_reason_as_the_stop_reason(
    finish_reason, accepted,
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter

    candidate = _vertex_candidate("hi", finish_reason)

    async def stream(**_kwargs):
        async def chunks():
            yield SimpleNamespace(text="hi", candidates=None, usage_metadata=None)
            yield SimpleNamespace(text=None, candidates=[candidate], usage_metadata=None)
        return chunks()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=SimpleNamespace(
            candidates=[candidate], text="hi", usage_metadata=None,
        )),
        generate_content_stream=stream,
    )))
    adapter = VertexAIAdapter()
    messages = [{"role": "user", "content": "hi"}]

    whole = await adapter.get_response(client=client, model="gemini", messages=messages)
    streamed = [
        item async for item in adapter.get_streaming_response_with_tools(
            client=client, model="gemini", messages=messages,
        )
    ][-1]

    for response in (whole, streamed):
        assert getattr(response, "stop_reason", None) == finish_reason
        assert (judge_generation(response) is None) is accepted
