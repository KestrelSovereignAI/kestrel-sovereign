import asyncio
import contextlib
import inspect
import json
import logging
import os
import re
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Type, Union, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from kestrel_sdk.hooks.base import Hook
from abc import ABC, abstractmethod
from kestrel_sdk.tools.base import ToolSchema, ToolParameter, ToolCategory, AgentTool
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus
# Import the A2A types from the SDK directly rather than the
# ``kestrel_sovereign.a2a`` re-export package. Importing the sovereign a2a
# package runs its ``__init__`` which eagerly pulls in task_manager/task_worker
# → the A2A stores → the storage SQLA models, and those models size a vector
# column at import time via ``get_provider_embedding_service()``. When
# ``features.base`` is still being initialized (it is imported very early), that
# chain re-enters ``from kestrel_sovereign.features.base import Feature`` against
# the half-built module and raises a circular ImportError that silently disables
# provider embeddings (#1792). The sovereign modules are pure re-exports of
# these same SDK symbols, so importing them from the SDK is equivalent.
from kestrel_sdk.a2a.agent_card import AgentCard, AgentSkill, AgentCapabilities
from kestrel_sdk.a2a.types import Task, TaskState, TaskStatus, Artifact, DataPart, Message, TextPart

# The SDK Feature is the canonical base class for feature packages.
# Sovereign's richer Feature inherits from it so extracted packages that
# subclass kestrel_sdk.features.base.Feature are also recognized as
# kestrel_sovereign.features.base.Feature at runtime (issubclass passes).
from kestrel_sdk.features.base import Feature as _SdkFeature
from kestrel_sdk.features.ui import UIContributions
# One source of truth for tool-schema generation: the @tool decorator and its
# docstring parser live in the SDK. Sovereign re-exports them (not a second
# copy) so both in-tree and external features share the SDK's annotation
# resolution — Optional[X] / PEP 604 unions / PEP 563 string annotations map to
# real JSON types instead of silently degrading to "string" (review finding
# F003). The two former in-tree copies were verified behaviourally identical to
# these across every feature docstring in the tree before removal.
from kestrel_sdk.features.base import tool, parse_docstring_params

logger = logging.getLogger(__name__)

# Maximum tool call iterations (configurable via environment variable)
# Increased to 50 for long-running tasks like code analysis and multi-step operations
MAX_TOOL_ITERATIONS = int(os.environ.get("KESTREL_MAX_TOOL_ITERATIONS", "50"))

CONTINUATION_INTENT_RE = re.compile(
    r"\b("
    r"let me|i(?:'ll| will| am going to)|"
    r"one moment|hang on|checking|calling|running|searching|looking up"
    r")\b.{0,80}\b("
    r"check|look|search|call|run|fetch|inspect|open|query|verify|use|try|"
    r"github|tool|cli|browser|file|repo|issue|database|db|workflow|job|"
    r"dispatch|provider"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)

TURN_COMPLETION_REPAIR_PROMPT = """You just wrote text that indicates this task is still in progress, but you did not emit a tool call.

Continue the same task now:
- If the work requires an available tool, emit the tool call now.
- If no tool is needed or available, provide the final answer now.
- Do not describe a future tool call without making it."""


def is_flat_toolresult_envelope(value: Any) -> bool:
    """True if ``value`` is a serialized (flat) ToolResult envelope (#F025).

    Discriminates a real ``ToolResult.to_dict()`` from an arbitrary dict that
    merely has a ``status`` domain field. A genuine envelope satisfies the
    ToolResult invariants (enforced in ``ToolResult.__post_init__``): OK carries
    a ``confirmation``, ERROR carries an ``error``, PARTIAL carries both. A
    legacy service payload like ``{"status": "ok", "items": [...]}`` therefore
    does NOT match and keeps its raw shape.
    """
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if status == "ok":
        return "confirmation" in value
    if status == "error":
        return "error" in value
    if status == "partial":
        return "confirmation" in value and "error" in value
    return False


def _serialize_tool_result(result: Any) -> Any:
    """Convert a tool result to a JSON-serializable format.

    Handles dataclasses with to_dict(), lists, enums, and nested structures.
    """
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, dict):
        return {k: _serialize_tool_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_serialize_tool_result(item) for item in result]
    if hasattr(result, 'to_dict'):
        return result.to_dict()
    if hasattr(result, 'value'):  # Enum
        return result.value
    return str(result)


def _attach_envelope_parts(
    response: Dict[str, Any],
    explicit_parts: Any,
    buffered_parts: Optional[List[dict]],
) -> None:
    """Attach first-class typed parts to a tool-result envelope (#2641).

    ``buffered_parts`` holds ``emit_part`` calls that found no turn collector
    (buffered on the "tool result under construction" contextvar);
    ``explicit_parts`` is an optional ``ToolResult.parts`` list a tool returned
    directly. Both are sanitized with the same type/size rules ``emit_part``
    applies, then written to ``response["parts"]`` — the envelope-carried form
    subagent dispatch passes through to the parent turn's collector. The key is
    only present when at least one valid part exists, so envelopes without
    parts are byte-identical to the pre-#2641 shape.
    """
    entries: List[Any] = []
    if buffered_parts:
        entries.extend(buffered_parts)
    if isinstance(explicit_parts, (list, tuple)):
        entries.extend(explicit_parts)
    if not entries:
        return
    # Lazy import: features.base is imported very early and importing
    # ``kestrel_sovereign.agent`` at module scope would re-enter this module
    # through the agent package's __init__ chain (same class of cycle as #1792).
    from kestrel_sovereign.agent.parts import sanitize_part
    clean = [p for p in (sanitize_part(e) for e in entries) if p is not None]
    if len(clean) != len(entries):
        logger.warning(
            "Dropped %d invalid typed part entries from %s tool-result envelope",
            len(entries) - len(clean), response.get("tool"),
        )
    # Overwrite rather than merge: ``explicit_parts`` IS the authoritative
    # source for anything a serializer may already have copied into the dict.
    response.pop("parts", None)
    if clean:
        response["parts"] = clean


def _subagent_turn_identity(session_id: Optional[str]):
    """Name a subagent LLM call's turn session WITHOUT joining its LLM thread.

    ``generate_with_messages(session_id=...)`` carries two different facts on
    one parameter: *which Timeline band this span belongs to* and *which
    provider conversation to continue*. For the orchestrator's own turn they
    are the same session. For a feature subagent they are not — it runs inside
    the user's turn (so its spans belong in that band, #2940) but it is a
    separate LLM conversation with its own system prompt and its own tool
    palette.

    Passing the turn's session on the wire would therefore make continuation-
    backed providers treat the subagent as the user's conversation. Concretely
    on ``openai:plan``: ``CodexAdapter._ensure_thread`` keys its thread cache on
    ``session_id`` and fingerprints ``(model, instructions, tools)``, so the
    subagent's differing prompt/tools would evict the user conversation's
    thread ("starting fresh thread" — losing its server-side history), cache
    the subagent's thread under the user's session id, and get evicted right
    back by the next user-facing turn. That is the same corruption the
    ``per-turn-reflection::<session>`` namespacing exists to prevent, and the
    #2841/#2845 continuation-loss class.

    So the session rides :class:`LLMInvocationContext` instead — the carrier
    the span and per-session metering already read (``_llm_request_span`` is
    handed ``invocation_context.session_id``), and which nothing provider-
    facing consumes. Returns ``None`` for a sessionless call so the resolver
    keeps using ambient identity untouched; a returned context sets only the
    session, and ``resolve_invocation_context`` merges the ambient
    companion/user/correlation fields and OR-s ``redact_content`` over it.
    """
    if not session_id:
        return None
    # Function-local like every other ``kestrel_sovereign`` import in this
    # module: features.base is imported very early in the discovery chain, so
    # its module-level surface is kept to ``kestrel_sdk`` (the #1792 class of
    # re-entrancy). This particular target is cycle-free on its own —
    # ``kestrel_sovereign.llm`` is a namespace package and invocation_context
    # imports only logging_config — so the local import is convention, not a
    # workaround.
    from kestrel_sovereign.llm.invocation_context import LLMInvocationContext

    return LLMInvocationContext(session_id=session_id)


@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for A2A task handling. Features implement this."""
    async def handle_task(self, task: Task) -> Task:
        """Handle an A2A task and return the updated task."""
        ...


class Feature(_SdkFeature):
    """
    Base class for Kestrel Features - each Feature IS a subagent.

    Extends the SDK's minimal Feature interface with sovereign-specific runtime
    behavior (LLM calls, subagent execution, hook enforcement). Packages that
    subclass kestrel_sdk.features.base.Feature are ALSO recognized as sovereign
    Features at discovery time because of this inheritance chain.

    A Feature encapsulates a specific domain of functionality (e.g., Sovereignty, MCP, Models).
    It can expose methods as Tools to the agent, and can be called AS a tool by the orchestrator
    with its own LLM context (A2A pattern).
    """

    # Node type used for persisting feature config in the knowledge graph.
    _CONFIG_NODE_TYPE = "feature_config"

    #: The closed vocabulary of ``data["reason_code"]`` values each of this
    #: feature's tools may return in a failed result, keyed by tool name.
    #: When a scheduled tool fails, the scheduler names the cause in the
    #: dispatch failure — text that leaves the redaction/cap boundary for
    #: ``signal_log.error`` — only by membership here, never by shape (#3184).
    #: A code that is not declared is dropped and logged as undeclared. Read
    #: by attribute name, so an out-of-tree feature (subclassing the SDK
    #: Feature) declares it the same way.
    tool_reason_codes: ClassVar[Mapping[str, frozenset[str]]] = {}

    def __init__(self, agent):
        self.agent = agent
        self.name = self.__class__.__name__
        self.disabled_skills: set = set()

    @staticmethod
    def _signals_unfinished_tool_work(content: Optional[str]) -> bool:
        """Return True when assistant text promises more tool-backed work."""
        if not content:
            return False
        return bool(CONTINUATION_INTENT_RE.search(content))

    @staticmethod
    def _append_missing_tool_call_repair(messages: list, content: str) -> list:
        """Return a repaired message list for one no-tool continuation retry."""
        repaired = list(messages)
        repaired.append({"role": "assistant", "content": content or ""})
        repaired.append({"role": "user", "content": TURN_COMPLETION_REPAIR_PROMPT})
        return repaired

    @staticmethod
    def _extract_response_reasoning_content(response: Any) -> Optional[str]:
        """Return provider reasoning that must be replayed with tool history."""
        raw = getattr(response, "raw", None)
        if isinstance(raw, dict):
            reasoning = raw.get("reasoning_content")
            return reasoning if isinstance(reasoning, str) and reasoning else None

        try:
            message = raw.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return None

        reasoning = getattr(message, "reasoning_content", None)
        return reasoning if isinstance(reasoning, str) and reasoning else None

    def _build_subagent_assistant_tool_history_msg(self, response: Any) -> dict:
        """Build assistant tool-call history for feature subagent loops."""
        message = {
            "role": "assistant",
            "content": getattr(response, "content", None) or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            tc.arguments if isinstance(tc.arguments, dict)
                            else json.loads(tc.arguments) if tc.arguments else {}
                        ),
                    },
                }
                for tc in response.tool_calls
            ],
        }

        reasoning_content = self._extract_response_reasoning_content(response)
        if reasoning_content and getattr(response, "tool_calls", None):
            message["reasoning_content"] = reasoning_content

        return message

    async def _repair_subagent_premature_yield(
        self,
        response: Any,
        messages: list,
        tools: List[Dict[str, Any]],
        tool_executor: Optional[Any] = None,
        model_override: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Give a feature subagent one more step when it narrates but emits no tool.

        ``tool_executor`` is threaded through to ``generate_with_messages``
        so codex-routed repair turns don't hit the same "requires a
        tool_executor callback" error the initial subagent call was
        wired around (codex round 1 P2 on #1461 follow-up).

        ``session_id`` is the session of the turn that dispatched the subagent,
        resolved once in :meth:`execute_as_subagent` and handed down. A repair
        turn is part of the same turn as the call it repairs, so it belongs in
        that turn's Timeline band; without it the repair span exported
        sessionless and became a band of its own (#2940). It travels as the
        invocation *identity*, never as the wire ``session_id`` — see
        :func:`_subagent_turn_identity` for why that distinction is load-bearing.
        """
        content = getattr(response, "content", "") or ""
        if not tools or not self._signals_unfinished_tool_work(content):
            return response

        logger.warning(
            "[SUBAGENT %s] Model signaled continuation without tool_calls; issuing one repair turn",
            self.name,
        )
        return await self.agent.llm_service.generate_with_messages(
            messages=self._append_missing_tool_call_repair(messages, content),
            tools=tools if tools else None,
            tool_executor=tool_executor,
            model_override=model_override,
            invocation_context=_subagent_turn_identity(session_id),
        )

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    @abstractmethod
    async def initialize(self):
        """Initialize the feature."""
        pass

    async def shutdown(self):
        """Cleanup resources.

        Reverses **every** post-load registration a feature makes on the agent
        so runtime disable, soft-disable, and boot rollback all leave nothing
        stranded (kestrel-sovereign#2522). Boot rollback
        (``KestrelAgent._boot_teardown_features``) and runtime disable both call
        ``shutdown()`` (via ``_unregister_feature_runtime``), so a feature that
        registered any of the following and then hit a later boot-phase failure
        must not leave feature-bound handlers / closures / loops stranded:

        * dispatcher **signal sources** and ``task:`` / other provider-defined
          **wait providers** (P2);
        * long-lived **background tasks** it started via
          :meth:`_track_owned_background_task` — e.g. Peers' hourly expiry sweep,
          which the agent's global reap only cancels at FULL shutdown, so
          without feature ownership a disabled feature's loop keeps running (P1);
        * **sleep hooks** it appended to ``agent.sleep_hooks`` via
          :meth:`_register_sleep_hook` — e.g. Memory's ReflectionSleepHook (P1).

        Each teardown is best-effort and idempotent so repeated shutdowns are
        safe. Overrides that do their own teardown should call
        ``await super().shutdown()``.
        """
        await self._unregister_owned_signal_sources()
        await self._unregister_owned_wait_providers()
        await self._cancel_owned_background_tasks()
        await self._unregister_owned_sleep_hooks()

    # ------------------------------------------------------------------
    # Signal-source ownership (#2522 P2)
    #
    # A feature that self-registers dispatcher sources must be able to remove
    # *exactly the sources it created* on shutdown / boot rollback — never a
    # host's pre-existing source. It records the names it newly registered here
    # and tears them down in ``shutdown()``.
    # ------------------------------------------------------------------
    @staticmethod
    def _registry_holds_claims(registry) -> bool:
        """Can *registry* record who holds a source, in a stated role?

        The ONE capability question asked about a signal registry, and it asks
        about the WHOLE contract a claim needs: state the owner and the role at
        registration, and release by owner and role at teardown. It used to be
        asked twice with two different markers — ``adopt`` when registering,
        ``release_all`` when tearing down — and two questions about one
        capability is how the ledgers this replaced came to disagree (#3053).

        Both halves are checked because they arrived at different times: a
        registry written against the ledger as it first shipped has
        ``release_all`` and takes ``owner=`` but not ``role=``, and calling it
        with the keyword is a ``TypeError`` mid-``initialize()``. Such a
        registry takes the name-tracking path instead — no claims, but correct
        teardown, which is the half that cannot be skipped.

        Asked about ``register_with_policy``, the call every registration can
        be expressed as. ``register_batch`` is asked about SEPARATELY, at the
        point of use (:meth:`_register_signal_sources`), because it is an
        optimisation rather than a capability: a registry that cannot take the
        role in its batch — or has no batch at all — can still hold claims
        perfectly well one source at a time. Requiring both here refused a
        registry over a method it did not need, and pushed it onto the name
        path, whose teardown removes by name and cannot see a peer's claim.

        A method taking ``**kwargs`` counts. That is the forwarding-proxy
        shape, and refusing it would push a proxy onto the name path, whose
        teardown removes by name and cannot see a peer's claim in the registry
        behind it. The other way round — a registry that accepts the keyword
        and ignores it — leaks a claim its teardown never releases. A leak is
        the better failure than deleting a source something else is dispatching
        on, so that is the way this errs.

        No compatibility path for the shape in between, deliberately. The claims
        ledger's first form — ``adopt`` / ``release_all`` / ``owner=`` without
        ``role=`` — exists only in unreleased main (#3053 landed after v0.53.4
        and ships in the same release as this), so nothing was ever built
        against it. Keeping a path for it would mean keeping ``adopt``'s
        ``created=`` flag, which is the guess this change exists to delete.
        """
        if not hasattr(registry, "release_all"):
            return False
        return Feature._takes_claim_role(
            getattr(registry, "register_with_policy", None)
        )

    @staticmethod
    def _takes_claim_role(method) -> bool:
        """Can *method* be told which role a claim is held in?

        ``**kwargs`` counts — see :meth:`_registry_holds_claims` for why that
        is the direction to err in.
        """
        if not callable(method):
            return False
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):  # a C callable, or an odd wrapper
            return False
        if "role" in parameters:
            return True
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

    def _register_signal_sources(self, registrations, policy):
        """Register signal sources AS THIS FEATURE; return the outcomes.

        The one door imperative registration goes through. Ownership is stated
        at the moment of registration, so ``owner=None`` means the host and
        nothing has to be inferred from the policy — the guess that three
        consecutive review rounds each found a different edge of (#3074).

        A registry that predates the claims ledger cannot take ``owner=``. For
        that case ONLY, the names this feature created are tracked on the
        feature, because nothing else can know them and its sources would
        otherwise never be torn down. That is not the second ledger #3053
        removed: there is exactly one record per registry, and which one is
        decided by what the registry can actually do.

        The registry is always the agent's. It used to be overridable, for a
        feature that had resolved one itself — but teardown reads the agent's,
        so an override could only ever be a way for the two halves to disagree
        about which registry holds the claim. Every call site was passing the
        agent's registry anyway.

        A caller that decides it cannot use what it registered must hand the
        claim back (:meth:`_disown_signal_sources`) — taking it at registration
        and declining afterwards are two halves of one door.
        """
        registry = getattr(getattr(self, "agent", None), "signal_registry", None)
        if registry is None:
            return []
        items = (
            list(registrations)
            if isinstance(registrations, (list, tuple, set))
            else [registrations]
        )
        if not items:
            return []
        claims = self._registry_holds_claims(registry)
        owner_kw = {}
        if claims:
            # CLAIM_IMPERATIVE, because this is the feature registering a source
            # ITSELF. Its declared contributions claim the other role, and
            # `shutdown()` releases only this one — the two are torn down by
            # different paths and either can fail alone (#3053).
            from kestrel_sovereign.signals import CLAIM_IMPERATIVE

            owner_kw = {"owner": self, "role": CLAIM_IMPERATIVE}
        batch = getattr(registry, "register_batch", None)
        register = getattr(registry, "register_with_policy", None)
        # The batch is asked about HERE, not in the capability question, because
        # it is an optimisation and not a capability: a registry with no batch,
        # or one that cannot take the role in it, still holds claims perfectly
        # well one source at a time. What it costs is atomicity, which matters
        # only under the raising policies — every imperative site registers
        # OPTIONAL, where each source is independent by definition.
        usable_batch = callable(batch) and (
            not claims or self._takes_claim_role(batch)
        )
        if len(items) > 1 and usable_batch:
            # Several sources go through `register_batch` because it is ATOMIC
            # under the raising policies: registering them one at a time would
            # leave a half-registered set behind on the failure the batch exists
            # to roll back.
            outcomes = list(batch(items, policy, **owner_kw))
        elif callable(register):
            outcomes = [register(item, policy, **owner_kw) for item in items]
        else:
            return []
        if not claims:
            self._track_unowned_signal_sources(outcomes)
        return outcomes

    def _signal_source_names(self, outcomes, *, created_only: bool):
        """Names out of whatever a ``register_*`` call returned.

        Accepts a :class:`RegistrationOutcome`, a list of them, a list of
        names, or a single name, because the helpers this reads from return all
        four shapes. *created_only* excludes a source this feature merely rode
        as an equivalent incumbent — the distinction a name-tracking registry
        has no way to express, so it must not remove what it did not create.
        """
        try:
            from kestrel_sovereign.signals import (
                RegistrationOutcome,
                RegistrationState,
            )
        except Exception:  # pragma: no cover - signals always importable in-tree
            RegistrationOutcome = None  # type: ignore[assignment]
            RegistrationState = None  # type: ignore[assignment]

        items = outcomes if isinstance(outcomes, (list, tuple, set)) else [outcomes]
        names = []
        for item in items:
            if isinstance(item, str):
                # A name-list helper already excluded sources a host owned, so
                # anything reaching here by name was created by this feature.
                names.append(item)
            elif (
                RegistrationOutcome is not None
                and isinstance(item, RegistrationOutcome)
            ):
                if item.state is RegistrationState.REGISTERED:
                    names.append(item.name)
                elif (
                    not created_only
                    and item.state is RegistrationState.ALREADY_EQUIVALENT
                ):
                    # Rode an incumbent: a real dependency, but not this
                    # feature's to remove — it claims ALONGSIDE the holder.
                    names.append(item.name)
        return names

    def _track_unowned_signal_sources(self, outcomes) -> None:
        """Record names for a registry that cannot hold claims."""
        owned = getattr(self, "_owned_signal_source_names", None)
        if owned is None:
            owned = []
            self._owned_signal_source_names = owned
        for name in self._signal_source_names(outcomes, created_only=True):
            if name not in owned:
                owned.append(name)

    def _drop_tracked_signal_sources(self, registry, names) -> None:
        """Unregister *names* this feature created, and stop tracking them.

        The whole of the name-tracking path's removal, in one place: teardown
        and declining a source do the same thing to it, and having them differ
        is how declining left a source registered AND untracked — dispatchable
        forever, and no longer removable by shutdown.

        Best-effort and idempotent: unregistering an already-absent source is a
        benign no-op, so repeated shutdowns are safe.
        """
        owned = getattr(self, "_owned_signal_source_names", None) or []
        removing = [name for name in owned if name in names]
        self._owned_signal_source_names = [n for n in owned if n not in removing]
        for name in removing:
            try:
                registry.unregister(name)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logger.warning(
                    "feature '%s': could not unregister signal source '%s': %s",
                    getattr(self, "name", type(self).__name__),
                    name,
                    exc,
                )

    def _disown_signal_sources(self, outcomes) -> None:
        """Hand back claims :meth:`_register_signal_sources` took.

        For a feature that registers a source and then finds it cannot use it.
        The claim is taken at registration now, so declining has to give it
        back — and giving back the LAST claim on a source removes it, which is
        the right answer: nothing needs a source its only holder refused.

        Reads the agent's registry, the same one
        :meth:`_register_signal_sources` claimed through — releasing from a
        different instance than the one holding the claim is the divergence
        that removing the override closed.
        """
        registry = getattr(getattr(self, "agent", None), "signal_registry", None)
        if registry is None:
            return
        names = self._signal_source_names(outcomes, created_only=False)
        if not self._registry_holds_claims(registry):
            # Only what this feature CREATED is removed — a ridden incumbent is
            # a peer's, and this path has no claims to express shared use. But
            # what it created IS removed: dropping only the bookkeeping left a
            # refused source dispatchable and beyond the reach of shutdown.
            self._drop_tracked_signal_sources(registry, names)
            return
        from kestrel_sovereign.signals import CLAIM_IMPERATIVE

        for name in names:
            try:
                registry.release(name, self, CLAIM_IMPERATIVE)
            except Exception as exc:  # noqa: BLE001 - best-effort
                logger.warning(
                    "feature '%s': could not release signal source '%s': %s",
                    getattr(self, "name", type(self).__name__),
                    name,
                    exc,
                )

    async def _unregister_owned_signal_sources(self) -> None:
        """Unregister the signal sources this feature registered (#2522 P2).

        Best-effort and idempotent: unregistering an already-absent source is a
        benign no-op, so repeated shutdowns are safe.
        """
        registry = getattr(getattr(self, "agent", None), "signal_registry", None)
        if registry is None:
            return
        if not self._registry_holds_claims(registry):
            # Registry without the ownership API: fall back to removing exactly
            # the names recorded for it.
            self._drop_tracked_signal_sources(
                registry, list(getattr(self, "_owned_signal_source_names", None) or ()),
            )
            return
        try:
            # ONLY the sources this feature registered itself. Its declared
            # contributions are released by the contribution runtime, which is a
            # different teardown that can fail on its own — and
            # `_unregister_feature_runtime` deliberately continues to here after
            # a rejected `deactivate()`. Releasing both roles together dropped a
            # still-active contribution's claim (#3053).
            from kestrel_sovereign.signals import CLAIM_IMPERATIVE

            registry.release_all(self, CLAIM_IMPERATIVE)
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning(
                "feature '%s': could not release its signal sources: %s",
                getattr(self, "name", type(self).__name__),
                exc,
            )

    # ------------------------------------------------------------------
    # Wait-provider ownership (#2522, identity-aware stack in P3)
    #
    # Features register a ``Waitable`` provider (for example ``task:`` or a
    # workflow-owned kind) with the agent's WaitRegistry in
    # ``post_all_features_loaded``. The registry keeps a
    # per-kind ownership STACK, so a feature only needs to record the exact
    # provider it pushed; on ``shutdown()`` (runtime disable AND boot rollback)
    # it removes that provider from the stack by IDENTITY, and the registry lets
    # the nearest still-live predecessor become effective. This restores the
    # host provider a feature displaced — but only while the feature's own
    # provider is still current — and never a newer provider some other owner
    # installed after us, nor an already-removed (disabled) predecessor. The
    # feature no longer saves a "previous" snapshot: that snapshot could name a
    # provider a later teardown already removed, which is exactly how a
    # ``host → A → B`` chain used to resurrect a disabled A on ``disable B``.
    # ------------------------------------------------------------------
    def _register_wait_provider(self, registry, provider, *, replace: bool = True):
        """Register ``provider`` on ``registry`` and record it for teardown.

        The single call features use in ``post_all_features_loaded``. It records
        the exact provider this feature pushed so :meth:`shutdown` can remove it
        from the registry's per-kind stack by identity (#2522 P3). This is the
        wait-provider analogue of :meth:`_register_signal_sources`.
        """
        kind = getattr(provider, "kind", None)
        owned = getattr(self, "_owned_wait_providers", None)
        if owned is None:
            owned = []
            self._owned_wait_providers = owned
        if kind:
            for index, (owned_kind, owned_provider) in enumerate(owned):
                if owned_kind == kind:
                    # Re-registering this kind in one live cycle with a NEW
                    # provider object: drop our previous provider from the stack
                    # first so we don't leave a self-owned entry buried beneath
                    # the new one (the displaced host/other provider is kept —
                    # it lives further down the stack, untouched).
                    if owned_provider is not provider and hasattr(
                        registry, "deregister"
                    ):
                        registry.deregister(kind, owned_provider)
                    owned[index] = (kind, provider)
                    break
            else:
                owned.append((kind, provider))
        registry.register(provider, replace=replace)

    async def _unregister_owned_wait_providers(self) -> None:
        """Remove the wait providers this feature pushed (#2522 P3).

        For each kind this feature registered, remove its own provider from the
        registry's per-kind stack by identity. The registry then lets the
        nearest still-live predecessor become effective again — never a newer
        provider some other owner installed after us, and never a predecessor an
        earlier teardown already removed. Best-effort and idempotent: a provider
        already gone from the stack is a benign no-op, so repeated shutdowns are
        safe.
        """
        owned = getattr(self, "_owned_wait_providers", None)
        if not owned:
            return
        registry = getattr(getattr(self, "agent", None), "wait_registry", None)
        if registry is not None:
            for kind, provider in owned:
                try:
                    if hasattr(registry, "deregister"):
                        registry.deregister(kind, provider)
                    elif hasattr(registry, "restore_if_current"):
                        # Registry predating the stack API: identity-aware
                        # single-slot teardown still removes our provider.
                        registry.restore_if_current(kind, provider)
                    elif hasattr(registry, "unregister"):
                        # Oldest fallback: a plain unregister clears the slot.
                        registry.unregister(kind)
                except Exception as exc:  # noqa: BLE001 - best-effort teardown
                    logger.warning(
                        "feature '%s': could not remove wait provider "
                        "kind '%s': %s",
                        getattr(self, "name", type(self).__name__),
                        kind,
                        exc,
                    )
        self._owned_wait_providers = []

    # ------------------------------------------------------------------
    # Background-task ownership (#2522 P1)
    #
    # A feature that starts a long-lived agent-owned background task in
    # ``post_all_features_loaded`` (Peers' hourly expiry sweep) must be able to
    # cancel EXACTLY the tasks it started on shutdown / boot rollback / soft
    # disable. The agent's global ``_shutdown_background_tasks`` only reaps at
    # FULL agent shutdown, so without feature ownership a disabled feature's
    # loop keeps running against a torn-down feature.
    # ------------------------------------------------------------------
    def _track_owned_background_task(self, coro, *, name: str) -> asyncio.Task:
        """Start an agent-owned background task AND record it for feature
        teardown (#2522 P1).

        Delegates to ``agent._track_background_task`` so the task still lives in
        the agent's global reap set (the full-shutdown safety net), but also
        records it here so :meth:`shutdown` cancels exactly this feature's tasks
        on runtime disable / boot rollback / soft disable. Returns the created
        ``asyncio.Task``. This is the background-task analogue of
        :meth:`_register_signal_sources` / :meth:`_register_wait_provider`.
        """
        agent = getattr(self, "agent", None)
        track = getattr(agent, "_track_background_task", None)
        if not callable(track):
            # The agent owns background-task lifecycle; a feature can't safely
            # start an unreaped task. Fail loudly rather than leak the coroutine.
            coro.close()
            raise RuntimeError(
                f"feature '{getattr(self, 'name', type(self).__name__)}': agent "
                "has no _track_background_task; cannot own background task "
                f"'{name}'"
            )
        task = track(coro, name=name)
        owned = getattr(self, "_owned_background_tasks", None)
        if owned is None:
            owned = []
            self._owned_background_tasks = owned
        owned.append(task)
        # Self-clean on completion so repeated / unbounded spawners (Peers'
        # per-question supervisors, RestartCoordinator's per-restart ack) don't
        # accumulate finished-task refs here — the mirror of the agent's global
        # set auto-discard. ``_owned`` binds THIS list, so a late callback firing
        # after :meth:`_cancel_owned_background_tasks` rebinds the attribute is a
        # harmless no-op rather than touching the fresh list (#2522 P1/P2).
        task.add_done_callback(
            lambda finished, _owned=owned: (
                _owned.remove(finished) if finished in _owned else None
            )
        )
        return task

    async def _cancel_owned_background_tasks(self) -> None:
        """Cancel exactly the background tasks this feature started (#2522 P1).

        Best-effort and idempotent: a task already finished / cancelled is a
        benign no-op, so repeated shutdowns are safe. Only tasks this feature
        started via :meth:`_track_owned_background_task` are cancelled — never a
        host's or another feature's. The agent's ``done_callback`` drops the
        cancelled task from its global set, so this does not leave a dangling
        reference there.
        """
        tasks = getattr(self, "_owned_background_tasks", None)
        if not tasks:
            return
        pending = [task for task in tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._owned_background_tasks = []

    # ------------------------------------------------------------------
    # Sleep-hook ownership (#2522 P1)
    #
    # A feature that appends a ``*SleepHook`` to ``agent.sleep_hooks`` in
    # ``post_all_features_loaded`` (Memory's ReflectionSleepHook) must remove
    # exactly its own hook on shutdown / boot rollback / soft disable — never
    # another feature's or a host's. Ownership is tracked by object IDENTITY.
    # ------------------------------------------------------------------
    def _register_sleep_hook(self, agent, hook) -> None:
        """Append ``hook`` to ``agent.sleep_hooks`` and record it for teardown.

        Records the exact instance so :meth:`shutdown` removes ONLY this
        feature's hook (by identity) on runtime disable / boot rollback / soft
        disable. Idempotent per instance: re-registering the same hook object
        neither double-appends to ``agent.sleep_hooks`` nor double-tracks it.
        """
        hooks = getattr(agent, "sleep_hooks", None)
        if hooks is None:
            hooks = []
            agent.sleep_hooks = hooks
        owned = getattr(self, "_owned_sleep_hooks", None)
        if owned is None:
            owned = []
            self._owned_sleep_hooks = owned
        if not any(existing is hook for existing in hooks):
            hooks.append(hook)
        if not any(existing is hook for existing in owned):
            owned.append(hook)

    async def _unregister_owned_sleep_hooks(self) -> None:
        """Remove exactly the sleep hooks this feature registered (#2522 P1).

        Removes by object identity so a hook another feature/host appended is
        never touched. Best-effort and idempotent: a hook already gone is a
        benign no-op, so repeated shutdowns are safe.
        """
        owned = getattr(self, "_owned_sleep_hooks", None)
        if not owned:
            return
        hooks = getattr(getattr(self, "agent", None), "sleep_hooks", None)
        if isinstance(hooks, list):
            for hook in owned:
                for index in range(len(hooks) - 1, -1, -1):
                    if hooks[index] is hook:
                        del hooks[index]
        self._owned_sleep_hooks = []

    async def on_enable(self):
        """Called when feature is enabled.

        Register hooks, start background tasks. Hooks returned by
        ``get_hooks()`` are auto-registered before this method is called,
        so only use this for additional setup beyond hook registration.
        """
        pass

    async def on_disable(self):
        """Called when feature is disabled.

        Unregister hooks, stop background tasks. Hooks returned by
        ``get_hooks()`` are auto-unregistered after this method is called,
        so only use this for additional teardown beyond hook unregistration.
        """
        pass

    async def on_remove(self):
        """Called before feature package is uninstalled. Clean up stored data."""
        pass

    def get_hooks(self) -> List["Hook"]:
        """Return hooks this feature wants registered.

        Hooks are auto-registered with the agent's HooksManager when the
        feature is enabled, and auto-unregistered when disabled. Features
        that need hooks should override this instead of manually calling
        ``hooks_manager.register()``.

        Returns:
            List of Hook instances to register.
        """
        return []

    def get_router(self):
        """Return a FastAPI APIRouter to mount, or None.

        Features that expose HTTP endpoints can override this to return
        an APIRouter instance. The agent will include the router in the
        FastAPI app after all features are loaded.

        Returns:
            Optional APIRouter instance, or None.
        """
        return None

    def get_ui_contributions(self) -> Optional["UIContributions"]:
        """Static assets + entry modules this feature contributes to the web UI.

        Returns a ``UIContributions`` descriptor or ``None``. Mirrors
        ``get_router()``: the server discovers it after all features load,
        mounts any declared ``static_dir`` at ``/features/{name}/static/``, and
        merges the manifest into ``GET /api/ui/contributions`` (enabled-only).
        The frontend boot loader dynamically ``import()``s the declared modules
        in order; each module registers its slot contributions via
        ``UI.register(...)``.

        Returns:
            Optional ``UIContributions``, or None.
        """
        return None

    @property
    def promote_tools_on_startup(self) -> bool:
        """Whether this feature's individual tools should be direct at startup.

        Most features start as a single dispatcher tool and promote their
        individual tools after first use. Features with meta-orchestration or
        agent-management tools can opt in here so startup remains generic.
        """
        return False

    async def post_all_features_loaded(self, agent):
        """Called after ALL features are discovered and initialized.

        Use this for cross-feature wiring that depends on other features
        being available. The ``agent.features`` dict is fully populated
        when this method is called.

        Args:
            agent: The KestrelAgent instance with all features loaded.
        """
        pass

    @property
    def config_schema(self) -> Optional[Dict]:
        """JSON Schema for feature configuration.

        UI can render a form from this schema. Return None if the feature
        has no user-configurable settings.
        """
        return None

    async def get_config(self) -> Dict:
        """Return the feature's current configuration."""
        return {}

    async def set_config(self, config: Dict) -> None:
        """Update the feature's configuration.

        Args:
            config: New configuration values (validated against config_schema).
        """
        pass

    # ------------------------------------------------------------------
    # Config persistence helpers
    # ------------------------------------------------------------------

    def _config_node_id(self) -> str:
        """Return the graph node ID used to persist this feature's config."""
        return f"feature_config:{self.name}"

    async def load_persisted_config(
        self, *, raise_on_error: bool = False
    ) -> Optional[Dict]:
        """Load persisted config from agent storage (graph store).

        Returns the stored config dict, or None if nothing is persisted.  The
        default remains best-effort for existing feature implementations;
        callers that must not confuse a failed durable read with an absent
        config can request the original storage exception.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return None
        # The graph-store contract is asynchronous. A non-async ``get_node``
        # surface is not a configured graph store, so retain the historical
        # absent-storage result rather than treating it as a durable read.
        get_node = getattr(storage, "get_node", None)
        if not inspect.iscoroutinefunction(get_node):
            return None
        try:
            node = await get_node(self._config_node_id())
            if node is not None:
                config = node.properties.get("config")
                if isinstance(config, str):
                    config = json.loads(config)
                # Restore disabled_skills from persisted config
                disabled = config.get("disabled_skills") if config else None
                if isinstance(disabled, list):
                    self.disabled_skills = set(disabled)
                return config
        except Exception:
            if raise_on_error:
                raise
            logger.warning("Failed to load persisted config for %s", self.name)
        return None

    async def persist_config(self, config: Dict) -> None:
        """Save config to agent storage (graph store).

        Stores the config as a graph node so it survives restarts.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            logger.debug(f"No storage available to persist config for {self.name}")
            return
        # Privacy boundary (#2672): the feature ``config`` is an arbitrary,
        # operator-influenced settings dict (it can hold API keys and free-form
        # values), so it is gated at its single source of truth rather than by
        # blanket-trusting a ``feature_config`` node type. Persisting config is
        # purely a survive-restarts cache; a volatile privacy mode's contract is
        # "don't persist", and the feature still boots and re-derives its config
        # from kestrel.toml, so skip the durable write entirely while volatile.
        allows = getattr(storage, "allows_persistent_writes", None)
        if callable(allows):
            try:
                permitted = bool(allows())
            except Exception:  # noqa: BLE001 - never let a policy probe block boot
                permitted = False
            if not permitted:
                logger.debug(
                    "Skipping durable config persist for %s — persistent writes "
                    "are disabled in the current privacy mode (#2672)",
                    self.name,
                )
                return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            node = GraphNode(
                node_id=self._config_node_id(),
                node_type=self._CONFIG_NODE_TYPE,
                label=f"{self.name} config",
                properties={"config": config},
            )
            await storage.add_node(node)
        except Exception as e:
            logger.warning(f"Failed to persist config for {self.name}: {e}")

    # =========================================================================
    # Feature-as-Subagent Interface (A2A Pattern)
    # =========================================================================

    @property
    def tool_name(self) -> str:
        """
        Name used when this feature is called as a tool by the orchestrator.
        Converts class name to snake_case (e.g., ModelAgent -> model_agent).
        """
        # Convert CamelCase to snake_case
        name = self.name
        # Insert underscore before uppercase letters and lowercase everything
        import re
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return snake

    @property
    @abstractmethod
    def tool_description(self) -> str:
        """
        Description of what this feature/subagent can do.
        This is shown to the orchestrator LLM when selecting which feature to call.

        Example:
            "Manage LLM models - list available models, change active model, pull new models"
        """
        pass

    # =========================================================================
    # A2A Protocol Implementation
    # =========================================================================

    def get_agent_card(self) -> AgentCard:
        """
        Generate an AgentCard for this Feature.

        This allows the Feature to be discovered and called as an A2A agent.
        The AgentCard describes the Feature's capabilities (skills) to other agents.

        Uses the canonical AgentSkill attached by the @tool decorator — same
        metadata source as get_tools(), no parallel construction.
        Skills in ``self.disabled_skills`` are excluded (get_tools() already
        filters them, so the card stays in sync).
        """
        skills = []
        for tool in self.get_tools():
            if hasattr(tool, 'agent_skill') and tool.agent_skill is not None:
                skills.append(tool.agent_skill)
            else:
                # Fallback for tools without a decorator-attached AgentSkill
                schema = tool.schema
                skills.append(AgentSkill(
                    id=schema.name,
                    name=schema.name,
                    description=schema.description,
                    tags=[schema.category.value] if schema.category else None,
                    inputModes=["application/json"],
                    outputModes=["application/json"],
                    category=schema.category.value if schema.category else None,
                ))

        return AgentCard(
            name=self.tool_name,
            description=self.tool_description,
            url=f"/agents/{self.tool_name}",
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False,
                pushNotifications=False,
                stateTransitionHistory=False,
            ),
            skills=skills,
        )

    async def handle_task(self, task: Task) -> Task:
        """
        Handle an A2A task by routing to the appropriate skill/tool.

        This is the A2A TaskHandler implementation. When TaskManager routes
        a task to this Feature, this method:
        1. Extracts the skill name from task metadata
        2. Finds the corresponding @tool method
        3. Executes it with the provided arguments
        4. Returns the updated task with results

        Args:
            task: The A2A Task to handle

        Returns:
            Updated Task with status and artifacts
        """
        try:
            # Update task to WORKING state
            task.status = TaskStatus(state=TaskState.WORKING)

            # Extract skill and args from task metadata
            metadata = task.metadata or {}
            skill_name = metadata.get("skill")
            args = metadata.get("args", {})

            if not skill_name:
                # If no skill specified, try to infer from message
                if task.history and task.history[-1].parts:
                    for part in task.history[-1].parts:
                        if hasattr(part, 'text'):
                            # Could parse command from text here
                            pass

                # Default to first skill if only one exists
                tools = self.get_tools()
                if len(tools) == 1:
                    skill_name = tools[0].name
                else:
                    raise ValueError(f"No skill specified. Available: {[t.name for t in tools]}")

            # Find and execute the tool
            tool = self._get_tool_by_name(skill_name)
            if not tool:
                raise ValueError(f"Unknown skill: {skill_name}")

            result = await tool.execute(**args)

            # Create artifact with result
            artifact = Artifact(
                name=f"{skill_name}_result",
                parts=[DataPart(data=result)],
            )
            task.artifacts = [artifact]

            # Mark completed
            task.status = TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=f"Completed {skill_name}")]
                )
            )

            return task

        except Exception as e:
            logger.error(f"Feature {self.name} task handling failed: {e}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text=f"Error: {str(e)}")]
                )
            )
            return task

    def _get_tool_by_name(self, name: str) -> Optional[AgentTool]:
        """Get a tool by its name."""
        for tool in self.get_tools():
            if tool.name == name:
                return tool
        return None

    def get_skill_for_command(self, command: str) -> Optional[str]:
        """
        Find the skill that handles a given command prefix.

        Args:
            command: Command string like "!list-models"

        Returns:
            Skill name if found, None otherwise
        """
        for tool in self.get_tools():
            if tool.schema.command_prefix and command.startswith(tool.schema.command_prefix):
                return tool.name
        return None

    def to_orchestrator_tool(self) -> Dict[str, Any]:
        """
        Convert this feature to an orchestrator-level tool definition.

        The orchestrator sees features as high-level tools, not individual
        tool methods. Each feature gets a 'task' parameter describing what
        the orchestrator wants it to do.

        Returns:
            OpenAI function calling format tool definition
        """
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What you want this agent to do"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context from the conversation"
                        }
                    },
                    "required": ["task"]
                }
            }
        }

    def _turn_session_id(self) -> Optional[str]:
        """The chat session of the turn this feature call is running for.

        Feature work is dispatched from inside a turn but is not handed the
        turn's session, so anything that must name it — a wake routed back to
        the originating window, the Timeline band an ``llm.*`` span belongs to
        (#2940) — has to ask the agent. Goes through the turn lifecycle's
        :meth:`~kestrel_sovereign.agent.turn_lifecycle.TurnLifecycleMixin.get_turn_bound_session_id`
        rather than reading ``agent._active_session_id`` directly: that
        attribute is agent-global, so unattended work (a cron tick, a task
        detached from a turn that has since exited) would otherwise read
        whichever *concurrent* chat turn happens to be in flight. The lifecycle
        pairs it with the task-local turn id and answers None unless the caller
        owns the live turn.

        None outside a turn, for a session-less turn, and for any agent double
        that does not implement the accessor — all of which mean "no chat
        window", which callers treat as system-initiated.
        """
        agent = getattr(self, "agent", None)
        if agent is None:
            return None
        resolve = getattr(agent, "get_turn_bound_session_id", None)
        if not callable(resolve):
            # Compatibility for older agents while the public SDK-facing
            # lifecycle accessor rolls out.
            resolve = getattr(agent, "_get_turn_bound_session_id", None)
        if not callable(resolve):
            return None
        try:
            session_id = resolve()
        except Exception:  # pragma: no cover - defensive; stub agents
            return None
        if isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
        return None

    async def execute_as_subagent(
        self,
        task: str,
        context: Optional[str] = None,
        max_iterations: Optional[int] = None,
        denied_tools: Optional[set] = None,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute this feature as a subagent with its own LLM context.

        This is the A2A (Agent-to-Agent) pattern. When the orchestrator calls
        a feature as a tool, the feature:
        1. Gets its own system prompt (feature-specific)
        2. Receives the task from the orchestrator
        3. Has access to ITS OWN tools only (minus any denied by security)
        4. Makes its own LLM call(s) to decide what to do
        5. Returns results to the orchestrator

        Args:
            task: What the orchestrator wants this feature to do
            context: Optional conversation context from the orchestrator
            denied_tools: Tool names denied by security policy (stripped from palette)
            model_override: The route/model the orchestrator resolved for THIS
                turn (``vendor:route/model``). Threaded in so the subagent's own
                reasoning call targets the exact route the main turn used, instead
                of falling back to default route_priority resolution — which set
                ``provider=unknown`` and rejected valid model+route combinations
                on the subagent path (#2352). ``None`` preserves the legacy
                default-resolution behaviour for callers that don't have a
                resolved model (e.g. some external transports).

        Returns:
            Dict with success status and result. When any tool in the subagent
            run produced first-class typed parts (#2641) — via ``emit_part`` or
            an envelope-carried ``parts`` field — they are returned under a
            ``parts`` key so the orchestrator's dispatch site can re-emit them
            into the parent turn's collector. Callers that don't handle parts
            can ignore the key.
        """
        # Subagent-local buffer for typed parts (#2641). Tools executed inside
        # this subagent — via the post-LLM loop OR the codex inline executor,
        # which runs on a reader-spawned task with a frozen context — deposit
        # their parts here, and the envelope carries them back to the
        # dispatching orchestrator by contract instead of ContextVar
        # happenstance.
        subagent_parts: List[dict] = []
        try:
            # Get feature's own tools, excluding any denied by security policy
            available_tools = self.get_tools()
            if denied_tools:
                available_tools = [t for t in available_tools if t.name not in denied_tools]
                logger.info(f"Feature {self.name}: stripped {len(denied_tools)} denied tools, {len(available_tools)} remaining")

            # If ALL tools are denied, return immediately with denial
            if not available_tools and denied_tools:
                denied_list = ", ".join(sorted(denied_tools))
                return {
                    "success": False,
                    "error": f"All tools in {self.name} are blocked by security policy (denied: {denied_list}). "
                             f"The requested operation cannot be performed.",
                }

            feature_tools = [
                tool.schema.to_openai_format()
                for tool in available_tools
            ]
            logger.debug(f"Feature {self.name} has {len(feature_tools)} tools available")

            # Feature-specific system prompt
            system_prompt = self._get_subagent_prompt()

            # Build user prompt with task and context
            user_prompt = f"Task: {task}"
            if context:
                user_prompt += f"\n\nConversation context: {context}"

            logger.info(f"Feature {self.name} executing subagent task: {task[:100]}...")

            # ``llm_service.generate`` accepts ``tool_executor`` and
            # delegates to ``get_response`` which calls
            # ``messages_for(adapter)`` per-provider so the message
            # shape gets translated to each route's native format
            # (Gemini's ``parts`` + ``_system`` vs OpenAI's
            # ``role``/``content``). Using ``generate_with_messages``
            # with a hand-built OpenAI-style list would bypass that
            # translation and break Gemini/Vertex routes — codex
            # round 3 P2 on #1461 follow-up.
            tool_executor = (
                self._make_feature_inline_tool_executor(parts_sink=subagent_parts)
                if feature_tools else None
            )
            # The subagent reasons on behalf of the turn that dispatched it, so
            # every LLM call it makes — this one AND the continuation/repair
            # turns the tool loop drives — joins that turn's session band
            # (#2940). Resolved once here: ``_turn_session_id`` answers from the
            # turn lifecycle, and a multi-step subagent can outlive the turn
            # binding, so re-resolving deeper in the loop would let a long run
            # start stamping None halfway through and split the band anyway.
            turn_session_id = self._turn_session_id()
            response = await self.agent.llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=feature_tools if feature_tools else None,
                tool_executor=tool_executor,
                model_override=model_override,
                # Safe as the plain parameter here, unlike on the continuation
                # calls below: ``generate`` never forwards ``session_id`` to an
                # adapter (it feeds only the invocation context and the span),
                # so it carries no provider-continuation meaning to collide with.
                session_id=turn_session_id,
            )

            # Log what we got back
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_names = [tc.name for tc in response.tool_calls]
                logger.info(f"Feature {self.name} LLM called tools: {tool_names}")
            else:
                content_preview = str(response)[:100] if response else "None"
                logger.debug(f"Feature {self.name} LLM returned text (no tool calls): {content_preview}...")

            # Handle tool calls within this feature's context. Thread
            # ``tool_executor`` through so every nested generate_with_
            # messages call (continuation, repair) on the codex-routed
            # path has the executor it needs to satisfy inline tool
            # calls (codex round 1 P2 on #1461 follow-up).
            result = await self._handle_feature_tool_calls(
                response,
                feature_tools,
                system_prompt,
                max_iterations=max_iterations,
                user_prompt=user_prompt,
                tool_executor=tool_executor,
                model_override=model_override,
                parts_sink=subagent_parts,
                session_id=turn_session_id,
            )

            # Debug: Log what we're returning to the orchestrator
            result_preview = str(result)[:500] if result else "None"
            logger.info(f"[SUBAGENT-RESULT] Feature {self.name} returning to orchestrator: {result_preview}")

            envelope: Dict[str, Any] = {"success": True, "result": result}
            if subagent_parts:
                envelope["parts"] = subagent_parts
            return envelope

        except Exception as e:
            logger.error(f"Feature {self.name} subagent execution failed: {e}")
            err_envelope: Dict[str, Any] = {"success": False, "error": str(e)}
            # Parts emitted before the failure (e.g. a *_pending card) still
            # travel — matching the direct path, where emit_part delivers
            # immediately regardless of how the tool call ends.
            if subagent_parts:
                err_envelope["parts"] = subagent_parts
            return err_envelope

    def _make_feature_inline_tool_executor(self, parts_sink: Optional[List[dict]] = None):
        """Build an inline ``(name, args) -> result_dict`` async callable
        bound to this feature's OWN tool palette, gated by the same
        ``PRE_TOOL_USE`` hooks the non-inline ``_handle_feature_tool_calls``
        path enforces.

        ``parts_sink`` (#2641) is the subagent-local typed-parts buffer:
        threaded into ``_execute_subagent_tool`` so parts emitted by inline
        tool calls — which the codex app-server dispatches on reader-spawned
        tasks whose frozen context has NO turn collector — land on the sink
        and ride the subagent's result envelope instead of being dropped.
        This is the subagent-path equivalent of the parent-turn
        ``bind_part_collector`` fix in
        ``OrchestratorEngineMixin._make_inline_tool_executor``.

        Required by adapters that execute tool calls INSIDE the LLM
        turn and block until the result arrives — the codex app-server
        (openai:plan, gpt-5.5) is the live case today. Without an
        executor, codex-routed subagent LLM calls fail at the provider
        layer with "requires a tool_executor callback when tools are
        provided", which is what hid Emma's memory_feature failures
        until the observability fix (#1461) made the error visible.

        Scoped to this feature's tools rather than the agent's global
        palette — a subagent shouldn't be able to reach for tools
        outside its own feature mid-turn. A name not in this feature's
        palette returns a structured error envelope; a PRE_TOOL_USE
        deny returns the same PERMISSION DENIED envelope the
        non-inline path produces (codex round 1 P1 on #1461 follow-up
        — without this, hook-gated policies were bypassed by the
        inline-execution path).

        NESTED cross-task bindings (#2672 review P1 follow-up, #2928). This
        executor is BUILT while ``execute_as_subagent`` runs on the PARENT inline
        executor's reader task, INSIDE that executor's
        ``bind_transition_lock_reentry`` scope, so the owning turn's
        transition-lock reentry token is visible in the ContextVar here. But the
        codex app-server dispatches THIS subagent's OWN inline tools on a
        SEPARATE, freshly-spawned reader task that does NOT inherit that binding
        — so a nested durable-identity write (rename / description / discovery
        history / user name / SOUL) invoked by the subagent would
        re-acquire the transition lock from a token-less foreign task and DEADLOCK
        against the turn that holds it (the turn is blocked awaiting the app-server
        result; the write is blocked acquiring the lock the turn holds). Capture the
        bound token here and re-present it around the subagent tool call — the exact
        cross-task seam ``OrchestratorEngineMixin._make_inline_tool_executor``
        installs for the parent turn — so that one write re-enters the owning turn's
        span. The parent executor also carries the lifecycle-authorized turn/session
        binding; capture and re-present that binding here so a nested lifecycle tool
        (notably ``request_restart``) can name the originating window after this
        second reader-task boundary. An executor built outside a live turn captures
        an explicit unbound value, so neither binding grants authority to unrelated
        background work.
        """
        from kestrel_sovereign.agent.turn_lifecycle import (
            bind_turn_session,
            capture_turn_session_binding,
        )
        from kestrel_sovereign.storage.privacy_wrapper import (
            bind_transition_lock_reentry,
            current_bound_reentry_token,
        )
        transition_reentry_token = current_bound_reentry_token()
        turn_session_binding = capture_turn_session_binding(self.agent)
        from kestrel_sovereign.auth import capture_caller_context_binding

        turn_caller_binding = capture_caller_context_binding()

        async def _exec(name: str, args: Dict[str, Any]):
            from kestrel_sovereign.auth import caller_context_binding_scope

            with (
                bind_transition_lock_reentry(transition_reentry_token),
                bind_turn_session(turn_session_binding),
                caller_context_binding_scope(turn_caller_binding),
            ):
                return await self._execute_subagent_tool(
                    tool_name=name,
                    args=args or {},
                    tools_by_name={t.name: t for t in self.get_tools()},
                    return_with_effective_args=True,
                    parts_sink=parts_sink,
                )
        return _exec

    async def _execute_subagent_tool(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        tools_by_name: Dict[str, Any],
        return_with_effective_args: bool = False,
        parts_sink: Optional[List[dict]] = None,
    ) -> Any:
        """Execute one of this feature's tools with PRE_TOOL_USE hook
        enforcement. Shared between the inline-executor path (used by
        codex app-server) and the post-LLM tool loop in
        ``_handle_feature_tool_calls`` so security policies, approval
        prompts, and argument-redaction hooks apply uniformly
        regardless of which transport ran the tool call.

        ``return_with_effective_args`` controls the return shape:

          - ``False`` (default, used by the post-LLM loop): returns
            just the ``result`` dict. The loop already knows the args.
          - ``True`` (used by the inline executor): returns the
            ``(effective_args, result)`` tuple the codex adapter
            expects. Codex round 4 P1 on #1461 follow-up — without
            this, audit / observability paths (``executed_tool_calls``,
            ``a2a_tool_dispatches.args_redacted``, persisted
            ``tool_results``) record the PRE-redaction args even
            when a hook rewrote them, leaking PII into log storage."""
        def _shape(effective_args_value: Dict[str, Any], result_value: Any) -> Any:
            """Return either the raw result or the
            ``(effective_args, result)`` tuple per the
            ``return_with_effective_args`` flag — this is what tells
            the codex adapter which args to log into
            ``executed_tool_calls`` / ``a2a_tool_dispatches``."""
            if return_with_effective_args:
                return (effective_args_value, result_value)
            return result_value

        selected_tool = tools_by_name.get(tool_name)
        if selected_tool is None:
            return _shape(args, {
                "success": False,
                "error": (
                    f"Tool {tool_name!r} is not in subagent "
                    f"{self.name!r}'s palette; available: "
                    f"{sorted(tools_by_name)}"
                ),
            })

        # #2641: bind the subagent-local parts sink as the active collector
        # for the WHOLE gated call — PRE_TOOL_USE hooks included, not just the
        # tool body — so ``emit_part`` lands deterministically on THIS
        # subagent's buffer no matter which task/context the transport
        # dispatched the call on (the codex app-server runs inline calls on
        # reader-spawned tasks whose frozen context has no live collector —
        # the exact gap parts.py documents for the parent-turn executor).
        # Covering the hooks matters for ordering: a part emitted by a PRE
        # hook must land on the same buffer as the tool's own parts, in
        # emission order (hook part first). This binding is the ONLY
        # ContextVar-side capture the subagent path performs — the ambient
        # collector outside this scope is the PARENT turn's (the dispatch
        # loop runs on the parent's task, and orchestrator+feature share one
        # hooks manager in production), so draining it from subagent code
        # steals the outer PRE_TOOL_USE gate's parts and reorders them
        # behind tool-body parts (#2641 review P1, both rounds).
        if parts_sink is None:
            sink_scope = contextlib.nullcontext()
        else:
            from kestrel_sovereign.agent.parts import bind_part_collector
            sink_scope = bind_part_collector(parts_sink)

        with sink_scope:
            hooks_manager = getattr(self.agent, "hooks_manager", None)
            effective_args = args
            if hooks_manager is not None:
                from kestrel_sdk.hooks.base import (
                    HookEvent, HookInput, PermissionDecision,
                )
                hook_input = HookInput(
                    session_id="subagent",
                    hook_event_name=HookEvent.PRE_TOOL_USE.value,
                    tool_name=tool_name,
                    tool_input=args,
                    feature_name=type(self).__name__,
                )
                hook_output = await hooks_manager.execute_hooks(
                    HookEvent.PRE_TOOL_USE, hook_input,
                )
                # Compute the effective args from the post-hook state FIRST,
                # before the block check. A hook chain may redact via an
                # early MODIFY hook (in-place mutation of
                # ``hook_input.tool_input``) and then DENY via a later
                # PermissionHook; the blocking branch must surface the
                # REDACTED args to the codex audit path or PII the
                # redaction hook removed will leak straight into
                # ``a2a_tool_dispatches.args_redacted`` /
                # ``executed_tool_calls``. Codex round 5 P1 on #1461
                # follow-up.
                mutated_input = getattr(hook_input, "tool_input", None)
                if isinstance(mutated_input, dict):
                    effective_args = mutated_input
                updated = getattr(hook_output, "updated_input", None)
                if isinstance(updated, dict):
                    effective_args = updated

                # Both DENY and ASK must short-circuit. ASK means "human
                # approval required" — the orchestrator-driven path's
                # ``execute_named_tool`` blocks both, and the codex inline
                # subagent path must match that contract or approval-gated
                # tools silently run without approval (codex round 2 P1
                # on #1461 follow-up).
                if hook_output.permission_decision in (
                    PermissionDecision.DENY,
                    PermissionDecision.ASK,
                ):
                    reason = (
                        hook_output.permission_reason
                        or "Blocked by security policy"
                    )
                    decision_label = (
                        "PERMISSION DENIED"
                        if hook_output.permission_decision == PermissionDecision.DENY
                        else "APPROVAL REQUIRED"
                    )
                    logger.info(
                        "[SUBAGENT-TOOL] %s blocked (%s): %s",
                        tool_name, decision_label, reason,
                    )
                    # Surface the POST-hook args even on the block path —
                    # an upstream redaction hook may have run before the
                    # downstream permission hook denied, and the codex
                    # audit row should record the redacted form, not the
                    # raw PII (codex round 5 P1 on #1461 follow-up).
                    return _shape(effective_args, {
                        "success": False,
                        "error": (
                            f"{decision_label}: {reason}. The tool was "
                            f"NOT executed. Do NOT tell the user this "
                            f"action succeeded — inform them it was "
                            f"blocked by security policy."
                        ),
                    })
                # ``effective_args`` was already resolved above to the
                # post-hook state (mutated ``tool_input`` first, then any
                # ``updated_input`` override) — see the comment block
                # before the DENY/ASK branch.

            try:
                result = await selected_tool.execute(**effective_args)
            except Exception as e:
                logger.warning(
                    "[SUBAGENT-TOOL] %s raised %s",
                    tool_name, e,
                )
                return _shape(
                    effective_args,
                    {"success": False, "error": f"{type(e).__name__}: {e}"},
                )
            serialized = _serialize_tool_result(result)
            # Harvest envelope-carried parts (``ToolResult.parts`` / a result
            # dict's ``parts`` field) into the sink too — the explicit,
            # ContextVar-free half of the #2641 contract. The envelope keeps its
            # copy; delivery to the outbound stream happens once, at the
            # orchestrator's dispatch site.
            if parts_sink is not None and isinstance(serialized, dict):
                envelope_parts = serialized.get("parts")
                if isinstance(envelope_parts, list) and envelope_parts:
                    parts_sink.extend(envelope_parts)
            return _shape(effective_args, serialized)

    def _get_subagent_prompt(self) -> str:
        """
        Get the system prompt for this feature's subagent context.

        Override this in subclasses for more specialized prompts.
        """
        tool_names = [t.name for t in self.get_tools()]
        tools_list = ", ".join(tool_names) if tool_names else "None"

        return f"""EXECUTION MODE: You are now executing as the {self.name} subagent.

You have been invoked to perform a specific task. DO NOT engage in conversation.
DO NOT ask clarifying questions. DO NOT respond with greetings or pleasantries.
DO NOT say you are "awaiting task input" - you already have a task.
EXECUTE THE TASK IMMEDIATELY using your tools.

Your capabilities: {self.tool_description}
Available tools: {tools_list}

EXECUTION PROTOCOL:
1. The task is in the next message - execute it immediately
2. Call the appropriate tool(s) to complete the task
3. If a tool fails or returns no results, report what you tried and what happened
4. Do NOT ask for more input - summarize results and complete your response
5. Use function calling to invoke tools - do not describe actions, DO THEM
6. If multiple tools are needed, call them in sequence
7. After getting tool results (success or failure), provide a brief summary

CRITICAL: You have ONE task. Execute it now. Do not wait for more input.

ABSOLUTE PROHIBITION - NEVER FABRICATE:
- NEVER invent a CID, hash, transaction ID, wallet address, or any cryptographic value
- NEVER generate a plausible-looking result without actually calling a tool
- If a tool call fails or is not available, say so explicitly - do not fill in fake values
- A fabricated cryptographic value is a lie and a constitutional violation"""

    async def _handle_feature_tool_calls(
        self,
        response: Union[str, Any],
        tools: List[Dict[str, Any]],
        system_prompt: str,
        max_iterations: int = None,
        user_prompt: Optional[str] = None,
        tool_executor: Optional[Any] = None,
        model_override: Optional[str] = None,
        parts_sink: Optional[List[dict]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Handle tool calls within this feature's context.

        This is similar to the orchestrator's tool handling but scoped to
        this feature's tools only.

        Args:
            response: Initial LLM response (string or message with tool_calls)
            tools: This feature's tools in OpenAI format
            system_prompt: The feature's system prompt for continuation
            max_iterations: Maximum tool call iterations to prevent infinite loops.
                           Defaults to KESTREL_MAX_TOOL_ITERATIONS env var (default: 5)
            parts_sink: Subagent-local buffer for first-class typed parts
                        (#2641). Threaded into ``_execute_subagent_tool``,
                        which binds it as the active collector for the whole
                        gated call and harvests envelope-carried ``parts``
                        into it, so ``execute_as_subagent`` can return them
                        by contract. The loop itself never drains the ambient
                        collector — that is the parent turn's buffer.
            session_id: Session of the turn that dispatched the subagent,
                        resolved once by ``execute_as_subagent``. Every
                        continuation and repair call this loop makes is part of
                        that same turn, so each stamps it and lands in one
                        Timeline band; a multi-step subagent otherwise split its
                        turn across a band per step (#2940). Applied as the
                        invocation identity — NOT the wire ``session_id``, which
                        is a provider continuation key the subagent must not
                        share (:func:`_subagent_turn_identity`).

        Returns:
            Final text response after all tool calls are processed
        """
        # Use module constant if not explicitly specified
        if max_iterations is None:
            max_iterations = MAX_TOOL_ITERATIONS

        # If response is just a string, return it directly
        if isinstance(response, str):
            return response

        # Build message history for multi-turn tool calling
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        # Check if response has tool_calls attribute
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            response = await self._repair_subagent_premature_yield(
                response,
                messages,
                tools,
                tool_executor=tool_executor,
                model_override=model_override,
                session_id=session_id,
            )
            if isinstance(response, str):
                return response
            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                return response.content or ""

        messages.append(self._build_subagent_assistant_tool_history_msg(response))

        # Get tools by name for execution
        tools_by_name = {agent_tool.name: agent_tool for agent_tool in self.get_tools()}

        for iteration in range(max_iterations):
            # Warn when approaching iteration limit
            if iteration >= max_iterations * 0.8:  # 80% threshold
                logger.warning(f"[SUBAGENT {self.name}] Approaching max iterations: {iteration + 1}/{max_iterations}")
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call.name

                # arguments is already a dict from our ToolCall dataclass
                if isinstance(tool_call.arguments, dict):
                    args = tool_call.arguments
                else:
                    try:
                        args = json.loads(tool_call.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}

                # Execute through the shared hook-enforced helper so
                # this loop AND the inline-executor path apply
                # identical PRE_TOOL_USE policy. Without this the
                # codex app-server inline path would bypass security
                # hooks the orchestrator-driven path enforces (codex
                # round 1 P1 on #1461 follow-up).
                result = await self._execute_subagent_tool(
                    tool_name=tool_name,
                    args=args,
                    tools_by_name=tools_by_name,
                    parts_sink=parts_sink,
                )
                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                logger.info(
                    f"[SUBAGENT-TOOL] {tool_name} result ({len(result_str)} chars): {result_str[:300]}..."
                )

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # Continue conversation with tool results — thread the
            # ``tool_executor`` through so codex-routed continuation
            # turns don't hit the same "requires a tool_executor"
            # provider error the initial subagent call avoided
            # (codex round 1 P2 on #1461 follow-up).
            response = await self.agent.llm_service.generate_with_messages(
                messages=messages,
                tools=tools if tools else None,
                tool_executor=tool_executor,
                model_override=model_override,
                # Turn session as invocation identity, not as the wire session:
                # the subagent shares the turn's Timeline band but must NOT
                # share its provider thread (:func:`_subagent_turn_identity`).
                invocation_context=_subagent_turn_identity(session_id),
            )

            # If response is string or has no more tool calls, we're done
            if isinstance(response, str):
                return response

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                response = await self._repair_subagent_premature_yield(
                    response,
                    messages,
                    tools,
                    tool_executor=tool_executor,
                    model_override=model_override,
                    session_id=session_id,
                )
                if isinstance(response, str):
                    return response
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    messages.append(self._build_subagent_assistant_tool_history_msg(response))
                    continue
                return response.content or ""

            # Add assistant response with new tool calls to messages
            messages.append(self._build_subagent_assistant_tool_history_msg(response))

        return "Error: Maximum tool call iterations exceeded"

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    def get_tools(self) -> List[AgentTool]:
        """
        Auto-discover methods decorated with @tool and return them as AgentTool instances.

        Each returned tool carries the canonical AgentSkill created by the @tool
        decorator, so get_agent_card() can reuse it without rebuilding metadata.

        Tools whose names appear in ``self.disabled_skills`` are excluded.
        """
        tools = []
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_tool_schema"):
                schema_data = method._tool_schema

                # Skip disabled skills
                if schema_data["name"] in self.disabled_skills:
                    continue

                agent_skill = getattr(method, "_agent_skill", None)

                # Create a dynamic AgentTool wrapper
                class DynamicTool(AgentTool):
                    def __init__(self, func, schema_data, agent_skill):
                        self.func = func
                        self._schema_data = schema_data
                        self.agent_skill = agent_skill

                    @property
                    def name(self) -> str:
                        return self._schema_data["name"]

                    @property
                    def schema(self) -> ToolSchema:
                        return ToolSchema(
                            name=self._schema_data["name"],
                            description=self._schema_data["description"],
                            category=self._schema_data["category"],
                            parameters=self._schema_data["parameters"],
                            command_prefix=self._schema_data.get("command_prefix")
                        )

                    async def execute(self, **kwargs) -> Dict[str, Any]:
                        # #2641: bind a "tool result under construction"
                        # buffer for the duration of the wrapped call.
                        # ``emit_part`` calls that find no turn collector
                        # (foreign-task transports, subagent loops) land here
                        # and are attached to the returned envelope's
                        # ``parts`` field, making delivery an envelope
                        # contract rather than ContextVar happenstance. Lazy
                        # import for the same #1792-class cycle reason as
                        # ``_attach_envelope_parts``.
                        from kestrel_sovereign.agent.parts import (
                            tool_result_parts_buffer,
                        )
                        with tool_result_parts_buffer() as pending_parts:
                            try:
                                result = await self.func(**kwargs)
                            except Exception as e:
                                logger.error(f"Error executing tool {self.name}: {e}")
                                response: Dict[str, Any] = {
                                    "success": False,
                                    "error": str(e),
                                    "tool": self.name,
                                }
                                # Parts emitted before the failure (e.g. a
                                # *_pending card) still travel, matching the
                                # collector path where emit_part delivers
                                # immediately.
                                _attach_envelope_parts(response, None, pending_parts)
                                return response

                        # ToolResult-returning @tool methods (#1042
                        # layer 4 contract) get serialized at the wrap
                        # site so downstream readers never see the raw
                        # frozen-dataclass instance. Without this,
                        # in-process callers (run_workflow,
                        # check_task_status) end up with a non-JSON-
                        # serializable object embedded in the wire
                        # payload — the workaround was per-callsite
                        # ``_serialize_step_payload`` helpers in
                        # PR-E pilot (#1066) which this fix
                        # supersedes. See #1070.
                        #
                        # Honesty: the wrapper's ``success`` flag
                        # historically meant "the call did not raise."
                        # That conflated transport with semantic
                        # outcome — a migrated tool returning
                        # ``ToolResult.failed`` would still surface
                        # ``success: True`` to callers like
                        # ``command_handler`` that branch on it.
                        # We now derive ``success`` from the
                        # ToolResult status:
                        #   - OK → success=True
                        #   - PARTIAL → success=True (it succeeded
                        #     enough to produce a confirmation; the
                        #     ``error`` field is also populated so
                        #     callers that surface both still get the
                        #     full picture)
                        #   - ERROR → success=False, error copied
                        #     into the wrapper's top-level error
                        if isinstance(result, ToolResult):
                            # Unified wire shape (#F025): spread the ToolResult
                            # envelope at the TOP level — matching the SDK
                            # wrapper (kestrel_sdk.features.base.DynamicTool) so
                            # in-tree and external features serialize
                            # identically. ``status``/``confirmation``/
                            # ``error``/``data`` now sit top-level, so the
                            # honesty layer (summarize_tool_result_for_audit
                            # reads top-level ``status``) sees a PARTIAL instead
                            # of it being hidden under a nested ``result`` where
                            # only a derived ``success`` was visible (#F001), and
                            # command_handler renders every feature's
                            # ``!command`` from the same shape (#F002).
                            #
                            # ``success`` is retained (derived from status:
                            # OK/PARTIAL → True, ERROR → False) purely as a
                            # back-compat courtesy for the many existing readers
                            # that branch on it; ``status`` is the canonical
                            # signal going forward.
                            response = result.to_dict()
                            response["tool"] = self.name
                            response["success"] = (
                                result.status is not ToolResultStatus.ERROR
                            )
                            # ``getattr`` because the pinned SDK's ToolResult
                            # predates the ``parts`` field (#2641) — this
                            # honors it as soon as the SDK ships it, and is a
                            # no-op until then.
                            _attach_envelope_parts(
                                response, getattr(result, "parts", None), pending_parts,
                            )
                            return response

                        # Pre-migration return shape (Dict[str, Any]
                        # or other) — keep the original wrapper. The
                        # ``success: True`` here remains transport-
                        # level for un-migrated tools; the #1061
                        # bulk waves migrate them away one by one.
                        response = {
                            "success": True,
                            "result": result,
                            "tool": self.name,
                        }
                        _attach_envelope_parts(response, None, pending_parts)
                        return response

                tools.append(DynamicTool(method, schema_data, agent_skill))
        return tools
