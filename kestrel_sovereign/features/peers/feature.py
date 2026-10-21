"""Peer discovery and native A2A communication for hosted agents.

Native message, task and blocking ask tools share one official SDK producer
and one sender-owned dispatch ledger. Standard Message Parts carry all input
handoffs. Published skill hints are advisory and never grant execution rights.
The question supervisor is being migrated before the draft protocol is retired.
"""

import json
import logging
import os
from collections.abc import Sequence as SequenceABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult, ToolResultStatus

from kestrel_sovereign._async_ownership import (
    await_owned_task,
    create_owned_task,
    owned_task_error,
    raise_owned_outcome,
)
from kestrel_sovereign.a2a.transport_auth import ensure_a2a_transport_key
from kestrel_sovereign.execution_custody import (
    execution_terminal_error,
    execution_work_operation,
    is_execution_control_error,
    owned_execution_stream,
)
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.features.peers.directory import (
    LocalHostPeerDirectory,
    PeerAccessDeniedError,
    PeerDirectoryConfigurationError,
    PeerDirectoryError,
    PeerDirectoryRouter,
    PeerIdentity,
    PeerNotFoundError,
    PeerProtocolError,
    PeerRequester,
    PeerSelfTargetError,
    PeerSubscriptionUnavailableError,
    PeerTaskConflictError,
    PeerTransportError,
    PeerUnavailableError,
    iter_sse_events,
)
from kestrel_sovereign.paths import AGENT_DB_PATH_ENV, runtime_path_env

logger = logging.getLogger(__name__)

def _discover_host_url() -> Optional[str]:
    """Discover the multi_agent host URL.

    Checks in order:
    1. KESTREL_HOST_URL env var (set by ProcessManager or manually)
    2. multi_agent.toml in project directory (read host port)
    3. None if not in a multi_agent environment
    """
    # Explicit env var (most reliable)
    host_url = os.environ.get("KESTREL_HOST_URL")
    if host_url:
        return host_url.rstrip("/")

    # Try reading multi_agent.toml to get host port. Resolve via the
    # paths module so pip-installed users land on their KESTREL_HOME /
    # ~/.kestrel project root rather than the package's site-packages
    # parent (which would never have a multi_agent.toml).
    from kestrel_sovereign.paths import project_dir
    for candidate in [
        Path.cwd() / "multi_agent.toml",
        project_dir() / "multi_agent.toml",
    ]:
        if candidate.exists():
            try:
                import toml
                data = toml.load(candidate)
                port = data.get("host", {}).get("port", 8888)
                return f"http://localhost:{port}"
            except Exception as e:
                logger.debug(f"Could not read {candidate}: {e}")

    return None


# Canonical artifact group name for send-side references (durable
# pointers to saved-memory / recall items, URIs, etc.). Kept distinct
# from the responder-side ``reply_body`` convention so a recipient can
# tell sender-attached handoff payload apart from a responder's reply.
REFERENCES_ARTIFACT_NAME = "references"
MAX_OUTBOUND_ARTIFACT_ITEMS = 32
MAX_OUTBOUND_ARTIFACT_BYTES = 64 * 1024


class OutboundArtifactValidationError(ValueError):
    """Typed send-side validation failure for outbound A2A handoff payloads."""

    def __init__(self, field: str, code: str, message: str):
        super().__init__(message)
        self.field = field
        self.code = code


# A2A task lifecycle states after which the recipient does no further work.
A2A_TERMINAL_TASK_STATES = ("completed", "failed", "canceled")

# ``error_type`` of a creator cancellation that failed in transport or found
# the recipient unavailable (#3586). No decision came back, so whether the
# recipient acted on it is unknown.
RECIPIENT_UNREACHABLE_ERROR_TYPE = "recipient_unreachable"


class ReceiverSettlementOutcome(Enum):
    """Where a recipient's task stands once its asker's deadline passed."""

    # This sender's cancellation holds the task: it is terminal and will not
    # be answered or re-delivered.
    WITHDRAWN = "withdrawn"
    # The recipient reached a terminal state of its own; the asker is owed it.
    ANSWERED = "answered"
    # Neither could be confirmed (the recipient was unreachable).
    UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class ReceiverSettlement:
    """Result of settling an expired question with its recipient (#3440)."""

    outcome: ReceiverSettlementOutcome
    state: str = "expired"
    reply_text: str = ""
    # The recipient did not answer within the settlement bound, or both
    # requests to it failed in transport, so another attempt on the same
    # recipient in the same pass would only wait or fail again.
    recipient_unresponsive: bool = False

    @property
    def receiver_terminal(self) -> bool:
        return self.outcome is not ReceiverSettlementOutcome.UNCONFIRMED


@dataclass(frozen=True)
class _QuestionClaim:
    """One pass's claim on a pending question's row, with the store that
    issued it (#3440). The claim is what lets the pass wake the asker for
    the row, or settle it with the recipient. Its renewals, the pre-relay
    ownership check, the confirmation that retires a delivered wake, and the
    restore of an undelivered one all go to that store, never to whichever
    one the agent holds at the time."""

    store: Any
    token: str


class _ClaimLeaseRenewingHandle:
    """A wake handle that keeps its question's claim lease while the wake's
    cognition turn runs (#3440).

    Renewal lives inside ``wait`` so it lasts exactly as long as someone is
    waiting for the wake, inline or in a delivery supervisor, and ends with
    that waiter whether the wake lands, fails, or the waiter is cancelled. It
    renews once as the wait begins, then every ``interval``. ``renew`` returns
    False once the row has nothing left to hold it for, or its claim was
    replaced."""

    def __init__(self, handle: Any, *, interval: float, renew) -> None:
        self._handle = handle
        self._interval = interval
        self._renew = renew

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    async def wait(self) -> Any:
        import asyncio

        waiter = create_owned_task(self._handle.wait())
        primary = None
        try:
            renewing = await self._renew()
            while renewing:
                done, _ = await asyncio.wait({waiter}, timeout=self._interval)
                if done:
                    break
                renewing = await self._renew()
        except BaseException as error:
            primary = error
        if primary is not None:
            # An abandoned wait interrupts its original dispatch, but owns
            # physical retirement and any stronger native receipt it returns.
            if not waiter.done():
                waiter.cancel()
        outcome = await await_owned_task(
            waiter, cancel_on_caller_cancellation=primary is None,
        )
        # The SDK facade may await an already-cancelled dispatch whose first
        # original carrier was consumed by its owning completion callback.
        # Lease renewal must not hide that exact dispatch behind a new facade.
        from kestrel_sdk.signals import SignalHandle
        dispatch = self._handle.task if isinstance(self._handle, SignalHandle) else None
        dispatch_error = (
            owned_task_error(dispatch)
            if isinstance(dispatch, asyncio.Task) and dispatch.done() else None
        )
        terminal = execution_terminal_error(
            dispatch_error, primary, outcome.error, outcome.cancellation,
        )
        if terminal is not None:
            raise terminal
        return outcome.result


class OutboundSigningError(RuntimeError):
    """A loaded hybrid identity could not authenticate an A2A envelope.

    The public ``code`` is deliberately coarse: callers and the outbound audit
    need an honest failure reason, but neither surface should receive provider
    exception text that could disclose signing implementation details.
    """

    def __init__(self, code: str):
        super().__init__("hybrid A2A envelope signing failed")
        self.code = code


def _normalize_outbound_artifact(item: Any, default_index: int) -> Dict[str, Any]:
    """Normalize one sender-supplied artifact into an A2A artifact wire
    dict (the shape the recipient's ``/tasks/send`` endpoint validates
    into an ``Artifact``).

    Accepts a dict with any of: ``name``, ``description``, ``metadata``,
    ``index``, ``last_chunk``/``lastChunk``, and a body given as
    ``parts`` (already wire-shaped), ``text`` (→ TextPart), or ``data``
    (→ DataPart for structured payloads). Supporting ``data`` is what
    lets a handoff carry structured metadata rather than only raw text.
    """
    if not isinstance(item, dict):
        raise OutboundArtifactValidationError(
            "artifacts",
            "invalid_artifact_item",
            "artifacts items must be structured dicts, not strings or scalars",
        )

    parts = item.get("parts")
    if not parts:
        if item.get("text") is not None:
            parts = [{"type": "text", "text": str(item["text"])}]
        elif item.get("data") is not None:
            parts = [{"type": "data", "data": item["data"]}]
        else:
            parts = []

    artifact: Dict[str, Any] = {
        "name": item.get("name") or "attachment",
        "parts": parts,
        "index": item.get("index", default_index),
    }
    if item.get("description") is not None:
        artifact["description"] = item["description"]
    if item.get("metadata") is not None:
        artifact["metadata"] = item["metadata"]
    last_chunk = item.get("last_chunk", item.get("lastChunk"))
    if last_chunk is not None:
        artifact["lastChunk"] = bool(last_chunk)
    return artifact


def _normalize_outbound_reference(ref: Any, index: int) -> Dict[str, Any]:
    """Normalize one durable reference into a structured-data artifact
    in the ``references`` group. A reference is a pointer (saved-memory
    or recall item id, URI, etc.); we carry it as a ``DataPart`` so the
    recipient gets the structured descriptor intact rather than a
    stringified blob."""
    if not isinstance(ref, dict):
        raise OutboundArtifactValidationError(
            "references",
            "invalid_reference_item",
            "references items must be structured dicts, not strings or scalars",
        )
    data = ref
    return {
        "name": REFERENCES_ARTIFACT_NAME,
        "parts": [{"type": "data", "data": data}],
        "index": index,
        "metadata": {"kind": "reference"},
    }


def _coerce_structured_sequence(value: Any, field: str) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (str, bytes, bytearray)):
        raise OutboundArtifactValidationError(
            field,
            f"{field}_must_be_structured",
            f"{field} must be a structured dict or list of dicts; got string",
        )
    if not isinstance(value, (list, tuple)):
        raise OutboundArtifactValidationError(
            field,
            f"{field}_must_be_structured",
            f"{field} must be a structured dict or list of dicts",
        )
    return list(value)


def _coerce_outbound_artifacts(
    artifacts: Optional[Any],
    references: Optional[Any],
) -> List[Dict[str, Any]]:
    """Build the outbound ``artifacts`` wire list from sender-supplied
    ``artifacts`` and ``references``. Artifacts keep their own ordering;
    references are appended as a separate ``references`` group with
    monotonic indices so the recipient can reassemble them in order."""
    wire: List[Dict[str, Any]] = []
    artifact_items = _coerce_structured_sequence(artifacts, "artifacts")
    reference_items = _coerce_structured_sequence(references, "references")
    if len(artifact_items) + len(reference_items) > MAX_OUTBOUND_ARTIFACT_ITEMS:
        raise OutboundArtifactValidationError(
            "artifacts",
            "too_many_items",
            "outbound artifacts and references are limited to "
            f"{MAX_OUTBOUND_ARTIFACT_ITEMS} total items",
        )
    for i, item in enumerate(artifact_items):
        wire.append(_normalize_outbound_artifact(item, i))
    for i, ref in enumerate(reference_items):
        wire.append(_normalize_outbound_reference(ref, i))
    # Encode with the SAME settings httpx will use on the wire so the
    # size check reflects what's actually about to be sent (compact
    # separators) and rejects payloads httpx will refuse later
    # (allow_nan=False). Without matching settings the validation
    # over-estimates by ~30% on dict-heavy payloads and a NaN value
    # passes here only to die later as a generic send error (codex
    # round 1 P2).
    try:
        size = len(
            json.dumps(
                wire,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise OutboundArtifactValidationError(
            "artifacts",
            "not_json_serializable",
            f"outbound artifacts/references must be JSON-serializable: {exc}",
        ) from exc
    if size > MAX_OUTBOUND_ARTIFACT_BYTES:
        field = "references" if reference_items and not artifact_items else "artifacts"
        raise OutboundArtifactValidationError(
            field,
            "payload_too_large",
            "outbound artifacts/references exceed "
            f"{MAX_OUTBOUND_ARTIFACT_BYTES} bytes when serialized",
        )
    return wire


class PeersFeature(Feature):
    """Inter-agent communication — ask questions to sibling agents in the multi_agent."""

    @property
    def tool_description(self) -> str:
        return (
            "Communicate with other agents in the multi_agent — "
            "send messages to peer agents and list available peers"
        )

    @property
    def promote_tools_on_startup(self) -> bool:
        return True

    @execution_work_operation
    async def initialize(self):
        self._host_url = _discover_host_url()
        self._transport_key = ensure_a2a_transport_key()
        # ProcessManager children boot through the single-agent server branch
        # and therefore never receive AgentManager's in-memory DID resolver.
        # Their launcher supplies an authenticated public-material snapshot;
        # install it before accepting signed peer actions.
        from kestrel_sovereign.a2a.did_registry import (
            install_process_a2a_did_resolver,
        )

        process_resolver = install_process_a2a_did_resolver(self.agent)
        if process_resolver is None:
            process_resolver = getattr(
                self.agent,
                "_a2a_process_did_resolver",
                None,
            )
        from kestrel_sovereign.a2a.transport_auth import (
            is_a2a_transport_only_process,
        )

        managed_subprocess = is_a2a_transport_only_process()
        if managed_subprocess:
            from kestrel_sovereign.a2a.inbound_authorization import (
                mark_a2a_inbound_scoped_policy,
            )

            # This launch marker is authoritative even if registry or host
            # wiring is absent. A broken managed child must deny A2A rather
            # than silently regain standalone unsigned compatibility.
            mark_a2a_inbound_scoped_policy(self.agent, required=True)
        self._own_name = self._get_own_name()
        # A hosted runtime injects both objects at agent construction.  The
        # requester scope is host-authenticated, opaque to this feature, and
        # never sourced from a tool argument or user-provided metadata.  When
        # neither is supplied, retain the local host HTTP adapter as the
        # backwards-compatible default.
        self._peer_router = getattr(self.agent, "peer_directory_router", None)
        self._peer_requester = getattr(self.agent, "peer_requester", None)
        if (self._peer_router is None) != (self._peer_requester is None):
            raise PeerDirectoryConfigurationError(
                "Injected peer router and trusted requester identity must "
                "be supplied together"
            )
        if self._peer_router is not None:
            if not isinstance(self._peer_requester, PeerRequester):
                raise PeerDirectoryConfigurationError(
                    "Injected peer router requires a trusted requester "
                    "identity and authorization scope"
                )
        elif self._host_url:
            self._install_local_host_router()

        if managed_subprocess:
            from kestrel_sovereign.a2a.inbound_authorization import (
                install_a2a_inbound_sender_authorizer,
            )

            # The endpoint reads its immutable policy from the agent, while
            # the subprocess compatibility router normally belongs to this
            # feature. Publish the exact pair only for a launcher-marked child.
            self.agent.peer_directory_router = self._peer_router
            self.agent.peer_requester = self._peer_requester
            sender_id_resolver = (
                process_resolver.directory_agent_id
                if process_resolver is not None
                else lambda _signing_did: None
            )
            install_a2a_inbound_sender_authorizer(
                None,
                recipient=self.agent,
                sender_id_resolver=sender_id_resolver,
            )

        # #1576: every outbound A2A dispatch writes a sender-side audit
        # row. The receiver-side ``a2a_tasks`` row tells us what the
        # peer saw; the outbound row tells US what we sent, when, to
        # whom, via which tool, and (after a later
        # ``get_peer_task_result`` fetch) what state it settled in.
        # Without this, the sender has no introspection surface for
        # "what did I dispatch and to whom?" beyond per-task_id round
        # trips.
        from kestrel_sovereign.a2a.outbound_store import (
            ensure_a2a_outbound_tasks_table,
        )
        from kestrel_sovereign.features.storage_access import (
            resolve_feature_database,
        )
        self._db = resolve_feature_database(self.agent)
        # A hosted retained route is security-sensitive.  Keep a separate
        # readiness bit because a partially-created table without its canonical
        # route index is not an acceptable substitute for durable ownership.
        self._outbound_route_store_ready = False
        if self._db is not None:
            try:
                await ensure_a2a_outbound_tasks_table(self._db)
                self._outbound_route_store_ready = True
            except Exception as exc:  # noqa: BLE001
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "PeersFeature: failed to ensure "
                    "a2a_outbound_tasks table: %s", exc,
                )

        if self._peer_router is not None and not isinstance(
            self._peer_router, LocalHostPeerDirectory,
        ):
            logger.info(
                "PeersFeature initialized with injected scoped peer router "
                "(self=%s)", self._own_name,
            )
        elif self._host_url:
            logger.info(f"PeersFeature initialized: host={self._host_url}, self={self._own_name}")
        else:
            logger.info("PeersFeature initialized but no multi_agent host found (standalone mode)")

    def _get_own_name(self) -> str:
        """Get this agent's name (from KESTREL_DB_PATH basename or agent node)."""
        # Try agent node name first
        if hasattr(self.agent, '_agent_name') and self.agent._agent_name:
            return self.agent._agent_name

        # Fall back to data dir basename
        db_path = runtime_path_env(AGENT_DB_PATH_ENV, "")
        if db_path:
            return Path(db_path).name

        return "unknown"

    def _current_legacy_outbound_sender(self) -> str:
        """Return the current public display name for an unsigned envelope.

        ``_own_name`` is intentionally retained as this feature's startup
        identity for logs and legacy fallbacks, but it is stale after a
        volatile rename. The hosted receiver authorizes unsigned legacy A2A
        against the live name it publishes on its agent card, so this sender
        metadata must resolve through the same live source. Hybrid envelopes
        subsequently overwrite this value with their signing DID.
        """

        resolver = getattr(self.agent, "resolve_effective_name", None)
        if callable(resolver):
            resolved = resolver(default=None)
            if isinstance(resolved, str) and resolved.strip():
                return resolved
        live_name = getattr(self.agent, "_agent_name", None)
        if isinstance(live_name, str) and live_name.strip():
            return live_name
        return self._own_name

    def _install_local_host_router(self) -> None:
        """Install the legacy local-host adapter with a private local scope."""
        host_url = getattr(self, "_host_url", None)
        if not host_url:
            return
        local_identity = str(getattr(self.agent, "did", None) or self._own_name)
        self._peer_requester = PeerRequester(
            identity=local_identity,
            authorization_scope=object(),
        )
        # Late-bind the factory so a host's transport instrumentation (and the
        # long-standing local test seam) applies to every operation, not only
        # the first operation that installed this adapter.
        self._peer_router = LocalHostPeerDirectory(
            host_url,
            execution_custody=getattr(self.agent, "_execution_custody", None),
            transport_key=getattr(self, "_transport_key", ""),
            client_factory=lambda *args, **kwargs: httpx.AsyncClient(
                *args, **kwargs,
            ),
            local_cancel=getattr(self, "_local_host_cancel", None),
            local_stop=getattr(self, "_local_host_stop", None),
            local_get=getattr(self, "_local_host_get", None),
            local_subscribe=getattr(self, "_local_host_subscribe", None),
            principal_payload_factory=self._build_principal_action_payload,
            native_session_factory=getattr(self, "_native_session_factory", None),
        )

    def _peer_directory_context(
        self,
    ) -> Optional[Tuple[PeerDirectoryRouter, PeerRequester]]:
        """Return the injected scoped router, lazily restoring local tests.

        Tests and embedding code that construct a feature directly historically
        set ``_host_url`` without running ``initialize``.  Lazily installing
        the local adapter preserves that supported local behavior; an injected
        router never falls back to host discovery when its mandatory requester
        context is missing.
        """
        router = getattr(self, "_peer_router", None)
        requester = getattr(self, "_peer_requester", None)
        if (router is None) != (requester is None):
            raise PeerDirectoryConfigurationError(
                "Injected peer router and trusted requester identity must "
                "be supplied together"
            )
        if router is not None:
            if not isinstance(requester, PeerRequester):
                raise PeerDirectoryConfigurationError(
                    "Injected peer router requires a trusted requester "
                    "identity and authorization scope"
                )
            return router, requester
        if getattr(self, "_host_url", None):
            self._install_local_host_router()
            return self._peer_directory_context()
        return None

    def hosted_peer_directory_context(
        self,
    ) -> Optional[Tuple[PeerDirectoryRouter, PeerRequester]]:
        """Return this feature's live trusted directory pair for host policy.

        A normal ``KestrelAgent`` keeps constructor-injected directory fields
        on the agent object, but the compatibility local-host adapter belongs
        to this feature instance. The multi-agent host must install its
        immutable inbound A2A policy from the effective feature context,
        rather than assuming those public construction fields still describe
        the route in use.
        """

        return self._peer_directory_context()

    def refresh_local_host_peer_directory(
        self,
        *,
        host_url: str,
        transport_key: str,
        local_cancel=None,
        local_stop=None,
        local_get=None,
        local_subscribe=None,
        native_session_factory=None,
    ) -> Optional[Tuple[PeerDirectoryRouter, PeerRequester]]:
        """Refresh only the local compatibility adapter for hosted policy.

        Host registration can occur after platform port resolution or peer-key
        generation. Those are host-owned runtime facts, so the local adapter
        must be reconstructed from them before the host freezes its inbound
        policy. An injected scoped router is intentionally never replaced.
        """

        router = getattr(self, "_peer_router", None)
        requester = getattr(self, "_peer_requester", None)
        if (router is None) != (requester is None):
            raise PeerDirectoryConfigurationError(
                "Injected peer router and trusted requester identity must "
                "be supplied together"
            )
        if router is not None and not isinstance(router, LocalHostPeerDirectory):
            return self._peer_directory_context()
        self._host_url = host_url.rstrip("/")
        self._transport_key = transport_key
        self._local_host_cancel = local_cancel
        self._local_host_stop = local_stop
        self._local_host_get = local_get
        self._local_host_subscribe = local_subscribe
        self._native_session_factory = native_session_factory
        self._peer_router = None
        self._peer_requester = None
        self._install_local_host_router()
        return self._peer_directory_context()

    def _requires_durable_peer_binding(self) -> bool:
        """Whether retained routes must have a durable stable identity.

        The local host adapter is a compatibility transport for one
        operator's fleet, where old rows without a stable identity may still
        use the historical task-id/name route.  An injected router represents
        a hosted, scoped directory: permitting a retained operation to fall
        back to a mutable name there could retarget a task to a replacement
        peer after restart.  Both transports reserve a newly written route
        before delivery and activate it only after accepting the peer's task
        id; hosted sends additionally require that reservation to exist.
        """
        router = getattr(self, "_peer_router", None)
        return router is not None and not isinstance(
            router, LocalHostPeerDirectory,
        )

    @execution_work_operation
    async def _resolve_automatic_peer(
        self, recipient: str,
    ) -> Tuple[PeerDirectoryRouter, PeerRequester, PeerIdentity]:
        """Resolve only within the current automatic peer directory.

        The route receives the stable ``PeerIdentity`` returned by the scoped
        provider, never the caller-provided name.  This is the critical guard
        against cross-scope DID/name probing and recipient substitution.
        """
        # Automatic peers are deliberately addressed only by their directory
        # name or slug.  A DID is a stable identity returned *by* a directory,
        # not an alternate automatic address.  Reject it before calling a
        # provider so an overly-permissive implementation cannot turn the
        # automatic shortcut into a cross-scope identity probe.
        if (
            not isinstance(recipient, str)
            or recipient.strip().casefold().startswith("did:")
        ):
            raise PeerNotFoundError("Peer is not in the automatic directory")
        context = self._peer_directory_context()
        if context is None:
            raise PeerDirectoryConfigurationError(
                "Not running in a multi_agent environment — no peer router"
            )
        router, requester = context
        try:
            peer = await router.resolve_peer(requester, recipient)
        except PeerDirectoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(exc):
                raise
            logger.exception("Peer router resolution raised unexpectedly")
            raise PeerProtocolError("Peer directory resolution failed") from exc
        if peer is None:
            raise PeerNotFoundError("Peer is not in the automatic directory")
        if not isinstance(peer, PeerIdentity):
            raise PeerProtocolError(
                "Peer directory returned an invalid peer identity"
            )
        if peer.agent_id == requester.identity:
            raise PeerSelfTargetError("Cannot route to the requesting agent")
        return router, requester, peer

    @execution_work_operation
    async def _resolve_retained_automatic_peer(
        self,
        recipient: str,
        recipient_agent_id: Optional[str],
    ) -> Tuple[PeerDirectoryRouter, PeerRequester, PeerIdentity]:
        """Resolve a persisted peer identity under the current scope.

        ``recipient_agent_id`` is written only after a successful automatic
        directory resolution.  Retained question/outbound-task state therefore
        survives display-name or slug changes without promoting a DID to a
        caller-addressable automatic-peer input.  The router must reauthorize
        the stable identity and return a current route before it is used.

        Legacy local-host records predate the stable-id column and retain the
        historical name/slug resolution behavior until they settle.  Injected
        hosted routers never get that fallback: a missing binding must deny
        the retained route rather than let a replacement peer claim the old
        display name.
        """
        if not recipient_agent_id:
            if self._requires_durable_peer_binding():
                raise PeerNotFoundError(
                    "No durable stable identity exists for this peer task"
                )
            return await self._resolve_automatic_peer(recipient)
        if not isinstance(recipient_agent_id, str) or not recipient_agent_id.strip():
            raise PeerProtocolError("Persisted peer identity is invalid")

        context = self._peer_directory_context()
        if context is None:
            raise PeerDirectoryConfigurationError(
                "Not running in a multi_agent environment — no peer router"
            )
        router, requester = context
        try:
            peer = await router.resolve_peer_by_agent_id(
                requester, recipient_agent_id,
            )
        except PeerDirectoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(exc):
                raise
            logger.exception("Peer router stable-identity resolution raised unexpectedly")
            raise PeerProtocolError("Peer directory resolution failed") from exc
        if peer is None:
            raise PeerNotFoundError("Peer is not in the automatic directory")
        if not isinstance(peer, PeerIdentity):
            raise PeerProtocolError(
                "Peer directory returned an invalid peer identity"
            )
        if peer.agent_id != recipient_agent_id:
            raise PeerProtocolError(
                "Peer directory returned a different stable peer identity"
            )
        if peer.agent_id == requester.identity:
            raise PeerSelfTargetError("Cannot route to the requesting agent")
        return router, requester, peer

    async def _outbound_recipient_agent_id(self, task_id: str) -> Optional[str]:
        """Read the stable recipient retained for one sender-owned task.

        The outbound audit is optional only for true no-store legacy
        local-host agents, so an unavailable record there is represented as
        ``None`` and uses the historical name/slug route.  A configured store
        that failed initialization is not equivalent to no store: it may have
        lost the stable binding for a delivered task, so every retained route
        must fail closed.  Hosted routes likewise fail closed when the durable
        binding is absent or unreadable, so a replacement peer can never
        receive a retained result fetch.
        """
        db = getattr(self, "_db", None)
        if db is None:
            if self._requires_durable_peer_binding():
                raise PeerNotFoundError(
                    "No durable stable identity exists for this peer task"
                )
            return None
        # A database was supplied, but its outbound route store could not be
        # initialized.  This is authoritative failure, not legacy no-store
        # compatibility: resolving ``recipient`` here could route an old task
        # to a same-name replacement peer after a restart.
        if not getattr(self, "_outbound_route_store_ready", False):
            raise PeerNotFoundError(
                "No durable stable identity exists for this peer task"
            )
        try:
            from kestrel_sovereign.a2a.outbound_store import (
                ROUTE_STATE_ROUTABLE,
                OutboundTaskRouteAmbiguousError,
                get_outbound_task,
            )

            outbound = await get_outbound_task(
                db,
                agent_id=str(getattr(self.agent, "did", None) or self._own_name),
                task_id=task_id,
            )
        except OutboundTaskRouteAmbiguousError as exc:
            # ``None`` remains the narrow legacy-local compatibility signal:
            # no outbound row was ever recorded.  A duplicate historical key
            # is instead affirmative evidence that the retained route is
            # unsafe, and must never fall through to mutable name resolution.
            raise PeerNotFoundError(
                "No durable stable identity exists for this peer task"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - database backend boundary
            if is_execution_control_error(exc):
                raise
            logger.debug(
                "outbound_store: retained recipient lookup failed for %s: %s",
                task_id, exc,
            )
            # Once initialization has established the route store, an
            # unreadable lookup is not distinguishable from a missing or
            # unsafe binding.  In either case, falling back to ``recipient``
            # would let a same-name replacement peer receive a retained
            # result fetch or subscription.  ``None`` is reserved exclusively
            # for the true legacy no-store path.
            if getattr(self, "_outbound_route_store_ready", False):
                raise PeerNotFoundError(
                    "No durable stable identity exists for this peer task"
                ) from exc
            return None
        if outbound is None:
            # ``None`` means two very different things depending on whether a
            # route store was ready.  Without a store, this is the narrow
            # historical local-host compatibility path: there was nowhere to
            # retain an outbound binding.  With a ready store, however, it is
            # affirmative evidence that this task has no durable owner (for
            # example, every reservation retry failed before and after
            # delivery).  Falling back to ``recipient`` in the latter case
            # would let a same-name replacement peer receive a retained fetch
            # or subscription after restart.
            if getattr(self, "_outbound_route_store_ready", False):
                raise PeerNotFoundError(
                    "No durable stable identity exists for this peer task"
                )
            return None

        recipient_agent_id = outbound.recipient_agent_id
        if outbound.route_state != ROUTE_STATE_ROUTABLE:
            # A reservation becomes routable only when its exact owner safely
            # accepts the peer's task id.  This includes local-host sends:
            # when an older peer returns a colliding id, keeping the sender's
            # provisional id routable would make a later fetch/subscription
            # use a task route that was never accepted.  Historical ambiguous
            # rows and failed reservations remain audit evidence only.
            raise PeerNotFoundError(
                "No durable stable identity exists for this peer task"
            )
        if (
            self._requires_durable_peer_binding()
            and (
                outbound.route_state != ROUTE_STATE_ROUTABLE
                or not recipient_agent_id
            )
        ):
            # A hosted reservation starts non-routable and becomes routable
            # only when one atomic rekey/activation accepted the peer task id.
            # This is intentionally independent of the best-effort terminal
            # audit marker: a marker write can fail without making a rejected
            # reservation eligible for retained routing after restart.
            raise PeerNotFoundError(
                "No durable stable identity exists for this peer task"
            )
        return recipient_agent_id

    def _maybe_sign_outbound(
        self,
        payload: Dict[str, Any],
        *,
        task_id: str,
        sess_id: str,
        message: str,
    ) -> None:
        """Sign the outbound A2A envelope for a loaded hybrid identity.

        Sets ``metadata["sender"]`` to the signing DID — the *verified*
        identifier — and attaches ``metadata["signature"]`` (hybrid Ed25519 +
        ML-DSA-65 over the canonical view: sender, task_id, session_id, message,
        timestamp). The kids are derived from the agent's published verification
        methods so the recipient's verifier can match them. Non-hybrid
        (pre-ceremony) agents send unsigned — the recipient allows that under
        the same-host boundary (back-compat). Once a hybrid identity is loaded,
        missing material or any signer error raises ``OutboundSigningError`` so
        the caller can abort before constructing an HTTP client (#2475).
        """
        identity = getattr(self.agent, "identity", None)
        if identity is None or not getattr(identity, "is_hybrid", False):
            return
        keypair = getattr(identity, "hybrid_keypair", None)
        signing_did = getattr(identity, "signing_did", None)
        vms = getattr(identity, "new_verification_methods", None)
        if not keypair or not signing_did or not vms:
            raise OutboundSigningError("missing_hybrid_signing_material")
        try:
            from datetime import datetime, timezone

            from kestrel_sovereign.a2a.envelope_signing import (
                bound_envelope_fields,
                canonical_message,
                kids_from_verification_methods,
                sign_envelope,
            )

            md = payload.setdefault("metadata", {})
            classical_kid, pq_kid = kids_from_verification_methods(vms)
            # Bind the behaviour-steering fields (skill/verb/reply/causation
            # chain) and the top-level artifacts so they can't be rewritten on
            # an otherwise-valid envelope (#1721).
            bound = bound_envelope_fields(md, artifacts=payload.get("artifacts"))
            block = sign_envelope(
                keypair,
                sender=signing_did,
                task_id=task_id,
                message=canonical_message([message]),
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=sess_id,
                bound=bound,
                classical_kid=classical_kid,
                pq_kid=pq_kid,
            )
            # The signed DID is the verified identifier the recipient binds to.
            md["sender"] = signing_did
            md["signature"] = block
        except Exception as exc:  # noqa: BLE001 - fail closed at trust boundary
            if is_execution_control_error(exc):
                raise
            logger.error(
                "A2A sign-on-send failed for loaded hybrid identity (%s)",
                type(exc).__name__,
            )
            raise OutboundSigningError("hybrid_signer_error") from exc

    def _build_principal_action_payload(
        self,
        task_id: str,
        verb: str,
    ) -> Dict[str, Any]:
        """Build a signed creator-principal envelope for HTTP read/SSE.

        Process-isolated peers cannot carry the host manager's in-memory
        capability. Binding the action verb, task id, session id, and message
        into the ordinary replay-protected A2A signature gives the recipient
        the same durable creator principal without trusting the shared API key.
        """

        if verb not in {"read_task", "subscribe_task"}:
            raise OutboundSigningError("unsupported_principal_action")
        if not isinstance(task_id, str) or not task_id:
            raise OutboundSigningError("invalid_principal_task_id")
        session_id = f"a2a-{verb}:{task_id}"
        message = f"{verb}:{task_id}"
        payload: Dict[str, Any] = {
            "id": task_id,
            "sessionId": session_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
            "metadata": {
                "sender": self._current_legacy_outbound_sender(),
                "a2a_verb": verb,
            },
        }
        self._maybe_sign_outbound(
            payload,
            task_id=task_id,
            sess_id=session_id,
            message=message,
        )
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping) or not metadata.get("signature"):
            raise OutboundSigningError("principal_signature_required")
        return payload

    @tool(
        name="stop_peer",
        description=(
            "Cooperatively stop a peer agent's current in-flight work through "
            "the authenticated signal rail. Any authorized peer may pull this "
            "bounded, rate-limited andon cord; it does not Hold, terminate, "
            "cascade to the peer's descendants, or grant hierarchy. Use "
            "scope='agent' for the peer's current work, or scope='turn' with "
            "that exact observable turn id."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peer stop",
    )
    @execution_work_operation
    async def stop_peer(
        self,
        recipient: str,
        reason: str = "Peer requested cooperative Stop",
        scope: str = "agent",
        target: str = "",
    ) -> ToolResult:
        """Request a peer Stop without bypassing dispatcher policy."""

        from uuid import uuid4

        from kestrel_sdk.signals import Status

        from kestrel_sovereign.a2a.envelope_signing import (
            A2A_AUDIENCE_METADATA_KEY,
        )
        from kestrel_sovereign.signals.sources.peer_stop import (
            MAX_PEER_STOP_REASON_CHARS,
            PEER_STOP_A2A_VERB,
            encode_peer_stop_intent,
            peer_stop_delivery_id,
        )
        from kestrel_sovereign.stop import StopDisposition, StopOutcome, StopScope

        try:
            stop_scope = StopScope(scope)
        except (TypeError, ValueError):
            return ToolResult.failed(
                "Peer Stop scope must be agent or turn",
                data={"stopped": False, "recipient": recipient},
            )
        if stop_scope is StopScope.HOST:
            return ToolResult.failed(
                "Peer Stop cannot target host scope",
                data={"stopped": False, "recipient": recipient},
            )
        if stop_scope is StopScope.TOOL_CALL:
            return ToolResult.failed(
                "Peer Stop cannot target tool_call scope until the peer exposes "
                "a live tool-call cancellation address",
                data={"stopped": False, "recipient": recipient},
            )
        work_target = target or None
        if stop_scope is StopScope.AGENT:
            if work_target is not None:
                return ToolResult.failed(
                    "Agent-scope peer Stop is routed by the peer directory and "
                    "cannot carry a second target",
                    data={"stopped": False, "recipient": recipient},
                )
        elif work_target is None:
            return ToolResult.failed(
                f"Peer {stop_scope.value} Stop requires an exact work target",
                data={"stopped": False, "recipient": recipient},
            )
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > MAX_PEER_STOP_REASON_CHARS
        ):
            return ToolResult.failed(
                "Peer Stop reason must be non-empty and at most "
                f"{MAX_PEER_STOP_REASON_CHARS} characters",
                data={"stopped": False, "recipient": recipient},
            )

        try:
            router, requester, peer = await self._resolve_automatic_peer(recipient)
        except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError) as control_error:
            if is_execution_control_error(control_error):
                raise
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"stopped": False, "recipient": recipient},
            )
        except (PeerTransportError, PeerUnavailableError) as control_error:
            if is_execution_control_error(control_error):
                raise
            return ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"stopped": False, "recipient": recipient},
            )
        except PeerDirectoryError as control_error:
            if is_execution_control_error(control_error):
                raise
            logger.exception("Peer Stop route resolution failed for %r", recipient)
            return ToolResult.failed(
                "Peer routing is unavailable",
                data={"stopped": False, "recipient": recipient},
            )

        correlation_id = uuid4().hex
        session_id = f"peer-stop-{correlation_id}"
        try:
            message = encode_peer_stop_intent(
                scope=stop_scope,
                target=work_target,
                reason=reason,
                # Cascade is sovereign-only (#3143); a peer never asks for it.
                cascade=False,
                correlation_id=correlation_id,
            )
        except (TypeError, ValueError) as error:
            return ToolResult.failed(
                f"Invalid peer Stop request: {error}",
                data={"stopped": False, "recipient": recipient},
            )

        metadata: Dict[str, Any] = {
            "sender": self._current_legacy_outbound_sender(),
            "a2a_verb": PEER_STOP_A2A_VERB,
            # Signed on remote routes and manager-bound on same-host routes.
            # Receivers compare it to their trusted route identity; it never
            # supplies Signal.target_agent.
            A2A_AUDIENCE_METADATA_KEY: peer.agent_id,
        }
        chain_provider = getattr(self.agent, "_provide_causation_chain", None)
        if callable(chain_provider):
            try:
                chain = chain_provider()
            except Exception as error:  # noqa: BLE001 - agent context provider
                if is_execution_control_error(error):
                    raise
                logger.warning(
                    "Could not read peer Stop causation chain: %s",
                    type(error).__name__,
                )
                return ToolResult.failed(
                    "Peer Stop could not preserve its causation chain",
                    data={"stopped": False, "recipient": recipient},
                )
            if chain:
                metadata["causation_chain"] = chain
        route_stop = getattr(router, "stop_peer", None)
        if not callable(route_stop):
            return ToolResult.failed(
                "Peer router does not support cooperative Stop",
                data={"stopped": False, "recipient": recipient},
            )

        try:
            raw_response = await self._deliver_peer_stop(
                route_stop,
                requester,
                peer,
                correlation_id=correlation_id,
                session_id=session_id,
                message=message,
                metadata=metadata,
                recipient=recipient,
            )
        except OutboundSigningError as error:
            if is_execution_control_error(error):
                raise
            return ToolResult.failed(
                "Peer Stop was not sent because sender authentication failed",
                data={
                    "stopped": False,
                    "recipient": recipient,
                    "error_type": "a2a_signing_failed",
                    "error_code": error.code,
                },
            )
        except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError) as control_error:
            if is_execution_control_error(control_error):
                raise
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"stopped": False, "recipient": recipient},
            )
        except (PeerTransportError, PeerUnavailableError) as control_error:
            if is_execution_control_error(control_error):
                raise
            return ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"stopped": False, "recipient": recipient},
            )
        except PeerDirectoryError as control_error:
            if is_execution_control_error(control_error):
                raise
            logger.exception("Peer Stop dispatch failed for %r", recipient)
            return ToolResult.failed(
                "Peer Stop dispatch failed",
                data={"stopped": False, "recipient": recipient},
            )
        except Exception as control_error:  # noqa: BLE001 - peer-router provider boundary
            if is_execution_control_error(control_error):
                raise
            logger.exception("Peer Stop dispatch failed for %r", recipient)
            return ToolResult.failed(
                "Peer Stop dispatch failed",
                data={"stopped": False, "recipient": recipient},
            )
        if not isinstance(raw_response, Mapping):
            return ToolResult.failed(
                "Peer returned a malformed Stop receipt",
                data={"stopped": False, "recipient": recipient},
            )
        response = dict(raw_response)
        raw_outcomes = response.get("stop_outcomes")
        signal_receipt = response.get("signal_receipt")
        if not isinstance(raw_outcomes, list) or not isinstance(
            signal_receipt, Mapping
        ):
            return ToolResult.failed(
                "Peer returned a malformed Stop receipt",
                data={"stopped": False, "recipient": recipient},
            )
        receipt_signal_id = signal_receipt.get("signal_id")
        receipt_status_value = signal_receipt.get("status")
        receipt_detail = signal_receipt.get("detail")
        receipt_kind = response.get("receipt_kind")
        if (
            not isinstance(receipt_signal_id, str)
            or not receipt_signal_id.strip()
            or receipt_kind not in {"operation", "delivery"}
            # Only a delivery can be refused before any dispatcher decision.
            or not (
                isinstance(receipt_status_value, str)
                or (receipt_status_value is None and receipt_kind == "delivery")
            )
            or "status" not in signal_receipt
            or "detail" not in signal_receipt
            or (receipt_detail is not None and not isinstance(receipt_detail, str))
        ):
            return ToolResult.failed(
                "Peer returned a malformed Stop receipt",
                data={"stopped": False, "recipient": recipient},
            )
        if receipt_status_value is not None:
            try:
                Status(receipt_status_value)
            except ValueError:
                return ToolResult.failed(
                    "Peer returned a malformed Stop receipt",
                    data={"stopped": False, "recipient": recipient},
                )
        try:
            outcomes = [StopOutcome.from_dict(item) for item in raw_outcomes]
        except (KeyError, TypeError, ValueError):
            return ToolResult.failed(
                "Peer returned a malformed Stop outcome",
                data={"stopped": False, "recipient": recipient},
            )
        stop_correlation_id = response.get("stop_correlation_id")
        # The response names the one record its outcomes are (#3169): the
        # Stop *operation*'s receipt (the handler's effect, or its replay), or
        # a decision about this *delivery* alone (cycle/depth, rate limit,
        # validation, identity reuse).  A delivery record is keyed by the
        # signal it answers and can never report an effect.
        delivery_correlation_id = peer_stop_delivery_id(receipt_signal_id)
        if receipt_kind == "delivery":
            record_matches = stop_correlation_id == delivery_correlation_id and all(
                outcome.disposition
                in {StopDisposition.REFUSED, StopDisposition.UNREACHABLE}
                for outcome in outcomes
            )
        else:
            record_matches = (
                isinstance(stop_correlation_id, str)
                and bool(stop_correlation_id.strip())
                and stop_correlation_id != delivery_correlation_id
            )
        if (
            response.get("correlation_id") != correlation_id
            or not record_matches
            or len(outcomes) != 1
            or any(
                outcome.agent_id != peer.agent_id
                or outcome.correlation_id != stop_correlation_id
                or outcome.scope is not stop_scope
                or outcome.requested_target
                != (
                    peer.agent_id
                    if stop_scope is StopScope.AGENT
                    else work_target
                )
                for outcome in outcomes
            )
        ):
            return ToolResult.failed(
                "Peer Stop receipt did not match the routed peer",
                data={"stopped": False, "recipient": recipient},
            )

        response["recipient"] = recipient
        response["recipient_agent_id"] = peer.agent_id
        # A retry is answered from the original receipt (or completes an
        # interrupted Stop), so the outcomes, not the dispatcher status, say
        # what happened.
        response["stopped"] = any(
            outcome.disposition is StopDisposition.STOPPED
            for outcome in outcomes
        )
        failed = any(
            outcome.disposition
            in {StopDisposition.REFUSED, StopDisposition.UNREACHABLE}
            for outcome in outcomes
        )
        if failed:
            return ToolResult.failed(
                "Peer Stop was refused or could not be confirmed",
                data=response,
            )
        return ToolResult.ok(
            confirmation=(
                f"Peer '{recipient}' stopped cooperative work"
                if response["stopped"]
                else f"Peer '{recipient}' had no matching active work"
            ),
            data=response,
        )

    async def _deliver_peer_stop(
        self,
        route_stop,
        requester: PeerRequester,
        peer: PeerIdentity,
        *,
        correlation_id: str,
        session_id: str,
        message: str,
        metadata: Mapping[str, Any],
        recipient: str,
    ) -> object:
        """Deliver one peer Stop, re-signing each attempt of the same request.

        A lost request or response is ambiguous: the peer may already have
        dispatched this Stop.  The recipient refuses a verbatim resend by its
        replay nonce, so every attempt signs the SAME intent, ``id`` and
        ``sessionId`` with a fresh nonce.  The recipient keys durable
        idempotency on (verified sender, correlation id), so a retry of an
        already-dispatched Stop is answered ``COALESCED`` from the original
        receipt and never stops the peer a second time (#3169).  Only
        undelivered/unconfirmed transport outcomes are retried; any
        authorization or protocol decision is final.
        """

        from kestrel_sovereign.signals.sources.peer_stop import (
            PEER_STOP_DELIVERY_ATTEMPTS,
        )

        for attempt in range(1, PEER_STOP_DELIVERY_ATTEMPTS + 1):
            payload: Dict[str, Any] = {
                "id": correlation_id,
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "metadata": dict(metadata),
            }
            self._maybe_sign_outbound(
                payload,
                task_id=correlation_id,
                sess_id=session_id,
                message=message,
            )
            try:
                return await route_stop(requester, peer, payload)
            except (PeerTransportError, PeerUnavailableError) as control_error:
                if is_execution_control_error(control_error):
                    raise
                if attempt >= PEER_STOP_DELIVERY_ATTEMPTS:
                    raise
                logger.warning(
                    "Peer Stop delivery to %r was not confirmed (attempt %d/%d); "
                    "re-signing the same request",
                    recipient,
                    attempt,
                    PEER_STOP_DELIVERY_ATTEMPTS,
                )
        raise AssertionError("unreachable: delivery loop always returns or raises")

    @tool(
        name="list_peers",
        description="List all available peer agents in the multi_agent.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peers"
    )
    @execution_work_operation
    async def list_peers(self) -> ToolResult:
        """
        Discover available peer agents via the scoped peer directory.
        Returns their names, status, and capabilities.
        """
        try:
            context = self._peer_directory_context()
        except PeerDirectoryConfigurationError as exc:
            if is_execution_control_error(exc):
                raise
            return ToolResult.failed(
                "Peer routing is not configured safely",
                data={"peers": [], "error": str(exc)},
            )
        if context is None:
            # Honesty: standalone mode is not a failure (the listing
            # WAS performed and returned the truthful "0 peers"), but
            # the agent must speak that no host is configured rather
            # than narrate "found 0 peers" as if peers really were
            # absent. PARTIAL with the diagnostic in the caveat.
            return ToolResult.partial(
                confirmation="No peers (standalone mode)",
                error="Not running in a multi_agent environment — no host to query",
                data={"peers": [], "note": "Not running in a multi_agent environment"},
            )

        router, requester = context
        try:
            directory = await router.list_peers(requester)
        except PeerAccessDeniedError as control_error:
            if is_execution_control_error(control_error):
                raise
            # Do not distinguish a denied scope from an empty/unknown peer
            # directory.  In hosted mode either distinction can be used to
            # probe another tenant's namespace.
            return ToolResult.failed(
                "Could not list peers in the current authorization scope",
                data={"peers": [], "error": "Peer directory unavailable"},
            )
        except PeerTransportError as control_error:
            if is_execution_control_error(control_error):
                raise
            return ToolResult.failed(
                "Could not connect to multi_agent host",
                data={"peers": [], "error": "Could not connect to multi_agent host"},
            )
        except PeerDirectoryError as exc:
            if is_execution_control_error(exc):
                raise
            logger.error("Failed to list peers: %s", exc)
            return ToolResult.failed(
                "Could not list peers",
                data={"peers": [], "error": "Could not list peers"},
            )
        except Exception as control_error:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(control_error):
                raise
            logger.exception("Peer router raised unexpectedly while listing peers")
            return ToolResult.failed(
                "Could not list peers",
                data={"peers": [], "error": "Could not list peers"},
            )

        if not isinstance(directory, SequenceABC) or isinstance(
            directory, (str, bytes, bytearray),
        ):
            logger.warning("Peer directory returned a non-sequence listing")
            return ToolResult.failed(
                "Could not list peers",
                data={"peers": [], "error": "Could not list peers"},
            )

        peers = []
        for peer in directory:
            if not isinstance(peer, PeerIdentity):
                logger.warning("Peer directory returned an invalid listing entry")
                continue
            if peer.agent_id != requester.identity:
                peers.append({
                    "name": peer.name or peer.slug,
                    "slug": peer.slug,
                    "status": peer.status,
                    "description": peer.description,
                })

        return ToolResult.ok(
            confirmation=f"Found {len(peers)} peer(s) (self={self._own_name})",
            data={"peers": peers, "self": self._own_name},
        )

    @tool(
        name="ask_agent",
        description="Send a native A2A message to a peer and wait for its terminal or interrupted response. The durable dispatch remains available if delivery becomes uncertain.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!ask",
    )
    @execution_work_operation
    async def ask_agent(self, agent_name: str, message: str) -> ToolResult:
        from a2a.types import Message
        from google.protobuf.json_format import MessageToDict

        from kestrel_sovereign.a2a.projections import message_text

        receipt = await self._dispatch_native_peer(
            agent_name,
            message,
            context_id="",
            parts=None,
            skill_id="",
            verb="message",
            dispatch_tool="ask_agent",
            return_immediately=False,
        )
        if isinstance(receipt, ToolResult):
            return receipt
        response = receipt.response
        text = (
            message_text(response)
            if isinstance(response, Message)
            else (
                message_text(response.status.message)
                if response.status.HasField("message")
                else ""
            )
        )
        data = self._native_dispatch_result(receipt, agent_name).data
        data["response"] = text
        data["native_response"] = MessageToDict(response)
        return ToolResult.ok(
            confirmation=f"Peer response received from {agent_name}", data=data
        )

    # ------------------------------------------------------------------
    # A2A: send a task to a peer agent (with inbound-wake semantics).
    # This is the supersedes-mesh direction (#645): peer-addressed
    # tasks land in the recipient's task_store AND trigger an
    # ``a2a.task_submitted`` signal that wakes their cognition loop,
    # so the recipient autonomously acts on the task rather than
    # waiting for a human-driven chat turn to notice it.
    # ------------------------------------------------------------------

    async def _post_a2a_task(
        self,
        recipient: str,
        message: str,
        skill_id: str = "",
        session_id: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Any]] = None,
        references: Optional[List[Any]] = None,
        dispatch_tool: str = "_post_a2a_task",
    ) -> Tuple[
        Optional[Dict[str, Any]],
        Optional[list],
        Optional[str],
        Optional[ToolResult],
    ]:
        """Shared POST helper for all three a2a verbs.

        Returns ``(task_data, chain, recipient_agent_id, error_result)``. On success
        ``task_data`` is the Task envelope from the recipient (with
        ``id``, ``status``, etc.) and ``chain`` is the serialized
        causation chain we attached to outbound metadata (or None when
        no chain was active). ``recipient_agent_id`` is the scoped stable
        identity to persist with any durable follow-up state; on failure
        ``error_result`` is a populated ToolResult.failed envelope the caller
        returns directly. The chain is returned so question-supervisor wiring
        can rehydrate it into the resumption signal without a second ContextVar
        read after the spawn (#1444).

        Centralizing this means the three verbs (send_a2a_message,
        send_a2a_question, send_a2a_task) share identical wire
        semantics, causation-chain attachment, and error handling —
        the difference between them is only what the caller does with
        the result (fire-and-forget vs fire-and-resume vs return
        task_id).
        """
        from uuid import uuid4

        task_id = uuid4().hex
        sess_id = session_id or uuid4().hex
        # An unsigned legacy envelope is authorized by the receiver against
        # its current published display name. Do not reuse the feature's
        # startup-cached name after a volatile rename. Hybrid signing below
        # replaces this with the authenticated DID before any routing occurs.
        outbound_metadata: Dict[str, Any] = {
            "sender": self._current_legacy_outbound_sender()
        }
        if skill_id:
            outbound_metadata["skill"] = skill_id
        if extra_metadata:
            outbound_metadata.update(extra_metadata)

        # #1576: capture the audit-row write so it fires before every
        # post-task-id return path (success or transport failure).  The local
        # host adapter treats a missing audit store as best-effort state; when
        # a store is available, both local and hosted routers reserve the
        # stable recipient before delivery.  A peer that returns a different,
        # already-claimed task id must never leave the provisional local row
        # routable.
        verb = str((extra_metadata or {}).get("a2a_verb") or "task")
        resolved_peer_agent_id: Optional[str] = None

        async def _persist_outbound(
            error: Optional[str] = None,
            effective_task_id: Optional[str] = None,
            route_state: str = "routable",
        ) -> Optional[Any]:
            """Persist the audit row.

            ``effective_task_id`` lets the success path pass the
            peer-echoed id from the response (which in production
            equals our local ``task_id`` — kestrel-claw protocol
            echoes the id back — but may diverge in tests with
            artificial mocks). Failure paths omit it and the local
            ``task_id`` is recorded; that's the id the agent would
            need to reference the attempted dispatch.
            """
            db = getattr(self, "_db", None)
            if db is None:
                return None
            if (
                self._requires_durable_peer_binding()
                and not getattr(self, "_outbound_route_store_ready", False)
            ):
                return None
            audit_id = effective_task_id or task_id
            # Scope the audit row to THIS agent (DID preferred, name
            # fallback) so a shared-backend Postgres deployment can't
            # leak rows across agents (codex review #1576 round 3 P1).
            audit_agent = (
                getattr(self.agent, "did", None) or self._own_name
            )
            try:
                from kestrel_sovereign.a2a.outbound_store import (
                    record_outbound_dispatch,
                )
                return await record_outbound_dispatch(
                    db,
                    agent_id=str(audit_agent),
                    task_id=audit_id,
                    recipient=recipient,
                    recipient_agent_id=resolved_peer_agent_id,
                    verb=verb,
                    session_id=sess_id,
                    skill_id=skill_id or None,
                    dispatch_tool=dispatch_tool,
                    message=message,
                    error=error,
                    route_state=route_state,
                )
            except Exception as exc:  # noqa: BLE001
                if is_execution_control_error(exc):
                    raise
                logger.debug(
                    "outbound_store: record failed for task %s → %s: %s",
                    audit_id, recipient, exc,
                )
                return None
        # Attach the in-flight signal-driven turn's causation chain so
        # the receiving agent's a2a.task_submitted signal carries the
        # lineage. Without this, A→B→A ping-pong loops bypass the
        # dispatcher's cycle detection (every inbound task starts
        # fresh at depth 1). Codex P1 on PR #1366.
        chain: Optional[list] = None
        chain_provider = getattr(self.agent, "_provide_causation_chain", None)
        if callable(chain_provider):
            try:
                chain = chain_provider()
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.debug(
                    "Failed to read causation chain for outbound A2A task: %s",
                    e,
                )
                chain = None
            if chain:
                outbound_metadata["causation_chain"] = chain
        payload = {
            "id": task_id,
            "sessionId": sess_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            },
            "metadata": outbound_metadata,
        }
        # Send-side artifacts/references: durable handoff payload the
        # sender attaches at creation time. Only put the key on the wire
        # when there's something to attach so legacy recipients that
        # ignore unknown keys see no change.
        try:
            outbound_artifacts = _coerce_outbound_artifacts(
                artifacts, references,
            )
        except OutboundArtifactValidationError as exc:
            return None, None, None, ToolResult.failed(
                f"Invalid A2A {exc.field}: {exc}",
                data={
                    "sent": False,
                    "recipient": recipient,
                    "error_type": f"invalid_a2a_{exc.field}",
                    "error_code": exc.code,
                    "field": exc.field,
                },
            )
        if outbound_artifacts:
            payload["artifacts"] = outbound_artifacts

        # Cryptographic sender authentication (#1706): if this agent has a
        # hybrid identity, sign the envelope so the recipient can verify it
        # (#1673). Non-hybrid agents send unsigned — back-compat.
        try:
            self._maybe_sign_outbound(
                payload, task_id=task_id, sess_id=sess_id, message=message,
            )
        except OutboundSigningError as exc:
            if is_execution_control_error(exc):
                raise
            # A loaded hybrid agent is never permitted to shed authentication.
            # Record only a stable, non-secret code, and return before an HTTP
            # client exists so retries cannot reuse an unsigned payload (#2475).
            await _persist_outbound(error=f"signing_failed:{exc.code}")
            return None, None, None, ToolResult.failed(
                "A2A dispatch aborted because hybrid envelope signing failed; "
                "no network request was sent",
                data={
                    "sent": False,
                    "recipient": recipient,
                    "task_id": task_id,
                    "error_type": "a2a_signing_failed",
                    "error_code": exc.code,
                },
            )

        # Resolve only after local payload validation and required signing.
        # A hybrid signing failure must make no network request at all; the
        # resulting signed envelope is still routed only through a scoped
        # resolution and never by interpolating ``recipient`` into an address.
        try:
            router, requester, peer = await self._resolve_automatic_peer(
                recipient,
            )
        except PeerDirectoryConfigurationError as control_error:
            if is_execution_control_error(control_error):
                raise
            return None, None, None, ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"sent": False, "recipient": recipient},
            )
        except PeerSelfTargetError as control_error:
            if is_execution_control_error(control_error):
                raise
            return None, None, None, ToolResult.failed(
                "Cannot send an A2A task to yourself",
                data={"sent": False, "recipient": recipient},
            )
        except PeerAccessDeniedError as control_error:
            if is_execution_control_error(control_error):
                raise
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient},
            )
        except PeerNotFoundError as control_error:
            if is_execution_control_error(control_error):
                raise
            # Use the same response for absent, cross-scope, and ambiguous
            # names so the automatic shortcut is not a namespace oracle.
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient},
            )
        except PeerTransportError as control_error:
            if is_execution_control_error(control_error):
                raise
            return None, None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}' — multi_agent host unreachable",
                data={"sent": False, "recipient": recipient},
            )
        except PeerDirectoryError as exc:
            if is_execution_control_error(exc):
                raise
            logger.error("Could not resolve A2A recipient %r: %s", recipient, exc)
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient},
            )

        # This value came from the scoped router, not from a tool argument.
        # Persist it with every later sender-side record so display-name/slug
        # changes cannot retarget a pending question or task-result fetch.
        resolved_peer_agent_id = peer.agent_id

        # Reserve the sender-owned task id and the router-issued stable peer
        # identity *before* delivery.  A hosted router is multi-tenant: if
        # this write fails, sending first and trying to reconstruct the route
        # from ``recipient`` after restart could hand a result/subscription to
        # a same-name replacement peer.  Local-host audit persistence remains
        # deliberately best-effort for backwards compatibility.
        require_durable_binding = self._requires_durable_peer_binding()
        from kestrel_sovereign.a2a.outbound_store import (
            ROUTE_STATE_RESERVED,
        )

        outbound_store_absent = getattr(self, "_db", None) is None
        reserved_outbound = await _persist_outbound(
            route_state=ROUTE_STATE_RESERVED,
        )
        reservation_write_failed = (
            not outbound_store_absent and reserved_outbound is None
        )
        if require_durable_binding and reserved_outbound is None:
            logger.error(
                "Refusing hosted A2A dispatch for task=%s: stable peer "
                "identity could not be persisted before delivery",
                task_id,
            )
            return None, None, None, ToolResult.failed(
                "Could not safely send A2A task because the peer identity "
                "could not be persisted",
                data={
                    "sent": False,
                    "recipient": recipient,
                    "task_id": task_id,
                    "error_type": "peer_identity_persistence_failed",
                },
            )

        async def _mark_dispatch_failed(error: str) -> None:
            """Record a failed dispatch without duplicating its audit row.

            The reservation remains ``route_state='reserved'`` regardless of
            whether this audit stamp lands.  That non-routable state, rather
            than this best-effort lifecycle annotation, is the retained-route
            denial invariant.
            """
            if reserved_outbound is None:
                # A present store already failed to create the original
                # reservation.  A later retry must not accidentally create a
                # directly-routable row merely because it is writing a
                # lifecycle error rather than the reservation itself.
                await _persist_outbound(
                    error=error,
                    route_state=ROUTE_STATE_RESERVED,
                )
                return
            db = getattr(self, "_db", None)
            if db is None:
                return
            try:
                from kestrel_sovereign.a2a.outbound_store import (
                    update_outbound_terminal_state,
                )

                await update_outbound_terminal_state(
                    db,
                    agent_id=str(
                        getattr(self.agent, "did", None) or self._own_name
                    ),
                    task_id=task_id,
                    terminal_state="dispatch_failed",
                    error=error,
                )
            except Exception as exc:  # noqa: BLE001 - audit failure must not mask route error
                if is_execution_control_error(exc):
                    raise
                logger.debug(
                    "outbound_store: failed to close dispatch %s: %s",
                    task_id,
                    exc,
                )

        try:
            routed_task = await router.send_a2a_task(requester, peer, payload)
            if not isinstance(routed_task, Mapping):
                raise PeerProtocolError("Peer router returned an invalid task envelope")
            task_data = dict(routed_task)
        except (PeerNotFoundError, PeerAccessDeniedError) as control_error:
            if is_execution_control_error(control_error):
                raise
            await _mark_dispatch_failed("peer_not_in_automatic_directory")
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerUnavailableError as control_error:
            if is_execution_control_error(control_error):
                raise
            await _mark_dispatch_failed(f"peer_unavailable:{recipient}")
            return None, None, None, ToolResult.failed(
                f"Agent '{recipient}' is offline or TaskManager unavailable",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerTransportError as control_error:
            if is_execution_control_error(control_error):
                raise
            await _mark_dispatch_failed(f"connect_error:{recipient}")
            return None, None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerDirectoryError as exc:
            if is_execution_control_error(exc):
                raise
            logger.error("A2A send to %r failed: %s", recipient, exc)
            await _mark_dispatch_failed(f"peer_router_error:{type(exc).__name__}")
            return None, None, None, ToolResult.failed(
                f"Could not send A2A task to '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(exc):
                raise
            logger.exception("A2A peer router raised unexpectedly for %r", recipient)
            await _mark_dispatch_failed(f"peer_router_error:{type(exc).__name__}")
            return None, None, None, ToolResult.failed(
                f"Could not send A2A task to '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        # Ensure id/sessionId always populate (older recipients might
        # echo only one or the other).
        task_data.setdefault("id", task_id)
        task_data.setdefault("sessionId", sess_id)
        # A compliant recipient echoes our envelope id.  Older recipients can
        # return a different id; move the already-reserved binding to that id
        # atomically before exposing it to the caller.  Every persisted route,
        # including the local-host compatibility route, runs this transition
        # even for an echoed id.  If it fails, the recipient may have accepted
        # work but no retained route is allowed to degrade into a display-name
        # lookup or claim another task's stable-recipient binding.
        effective_task_id = str(task_data.get("id") or task_id)
        if reservation_write_failed:
            # There is a local sender store, so this was not the historical
            # no-store compatibility path: its reservation write failed.
            # Do not accept or expose the peer's task id without first
            # durably binding it to the resolved recipient.  We make one
            # post-delivery attempt to persist a *non-routable* provisional
            # audit row, which lets a transient write failure survive restart
            # as an explicit denied route.  Either way the caller receives
            # only our provisional id, never a possibly colliding peer id.
            reserved_outbound = await _persist_outbound(
                route_state=ROUTE_STATE_RESERVED,
            )
            await _mark_dispatch_failed("peer_identity_reservation_failed")
            return None, None, None, ToolResult.failed(
                "A2A task may have been delivered, but its peer identity "
                "could not be persisted safely for retained routing",
                data={
                    "sent": True,
                    "recipient": recipient,
                    "task_id": task_id,
                    "error_type": "peer_identity_persistence_failed",
                },
            )
        if reserved_outbound is not None:
            try:
                from kestrel_sovereign.a2a.outbound_store import (
                    rekey_outbound_task,
                )

                rekeyed = await rekey_outbound_task(
                    getattr(self, "_db", None),
                    record_id=reserved_outbound.id,
                    agent_id=str(
                        getattr(self.agent, "did", None) or self._own_name
                    ),
                    old_task_id=task_id,
                    new_task_id=effective_task_id,
                    recipient_agent_id=resolved_peer_agent_id,
                    activate=True,
                )
            except Exception as exc:  # noqa: BLE001 - storage boundary
                if is_execution_control_error(exc):
                    raise
                logger.error(
                    "Failed to persist peer task-id binding %s -> %s: %s",
                    task_id,
                    effective_task_id,
                    exc,
                )
                rekeyed = 0
            if rekeyed != 1:
                # The reservation is still non-routable.  This is crucial for
                # local hosts too: a legacy peer can return a task id already
                # owned by a different peer, and reporting that id would send
                # later fetches/subscriptions to the existing owner.  Return
                # an honest delivered-but-untrackable failure without exposing
                # the peer-returned id; the provisional id is audit-only and
                # cannot route because the reservation never activated.
                await _mark_dispatch_failed("peer_identity_rekey_failed")
                return None, None, None, ToolResult.failed(
                    "A2A task may have been delivered, but its peer identity "
                    "could not be persisted safely for retained routing",
                    data={
                        "sent": True,
                        "recipient": recipient,
                        "task_id": task_id,
                        "error_type": "peer_identity_persistence_failed",
                    },
                )
        # A present sender store has activated its exact recipient binding
        # above.  With no sender store, retain local-host early-init
        # compatibility: preserve delivery and let a later legacy fetch
        # resolve by name. A present-but-failed store returned above.
        return task_data, chain, resolved_peer_agent_id, None

    async def _dispatch_native_peer(
        self,
        recipient: str,
        message: str,
        *,
        context_id: str,
        parts: list | None,
        skill_id: str,
        verb: str,
        dispatch_tool: str,
        return_immediately: bool = True,
    ):
        """The common native request, reservation and send path for peer tools."""
        from uuid import uuid4

        from a2a.types import Message, Part, Role, SendMessageRequest
        from a2a.utils.errors import A2AError
        from google.protobuf.json_format import ParseDict, ParseError

        from kestrel_sovereign.a2a.extensions import (
            causation_metadata,
            skill_hint_metadata,
        )
        from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
        from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            DispatchState,
            OutboundDispatchConflict,
        )

        if not isinstance(message, str) or not message.strip():
            return ToolResult.failed("A2A message must contain text")
        if not isinstance(context_id, str) or not isinstance(skill_id, str):
            return ToolResult.failed("A2A context_id and skill_id must be strings")
        if parts is not None and not isinstance(parts, list):
            return ToolResult.failed(
                "A2A parts must be an array of standard Part objects"
            )
        if any(not isinstance(part, dict) for part in (parts or [])):
            return ToolResult.failed("Each A2A part must be a standard Part object")
        try:
            content = [Part(text=message)] + [
                ParseDict(part, Part()) for part in (parts or [])
            ]
        except ParseError as error:
            return ToolResult.failed(f"Invalid A2A part: {error}")
        provider = getattr(self.agent, "_provide_causation_chain", None)
        chain = provider() if callable(provider) else None
        metadata = causation_metadata(chain) if chain else {}
        if skill_id:
            metadata.update(skill_hint_metadata(skill_id))
        dispatch = None
        store = None
        try:
            store = self._native_outbound_store()
            router, requester, peer = await self._resolve_automatic_peer(recipient)
            async with router.open_a2a_session(requester, peer) as session:
                request = SendMessageRequest(
                    tenant=session.recipient_scope.tenant,
                    message=Message(
                        message_id=uuid4().hex,
                        context_id=context_id,
                        role=Role.ROLE_USER,
                        parts=content,
                    ),
                    metadata=metadata,
                    configuration={"return_immediately": return_immediately},
                )
                dispatch, wire = await session.reserve(
                    request,
                    dispatch_tool=dispatch_tool,
                    verb=verb,
                )
                return await session.send(dispatch, wire)
        except (
            PeerDirectoryError,
            A2AError,
            OutboundDispatchConflict,
            NativePeerProtocolError,
            NativeSenderProofError,
        ) as error:
            if is_execution_control_error(error):
                raise
            current = (
                None if dispatch is None else await store.get(dispatch.dispatch_id)
            )
            committed = current is not None and current.state in {
                DispatchState.ACKNOWLEDGED,
                DispatchState.COMPLETE,
            }
            uncertain = dispatch is not None and (
                current is None
                or current.state
                in {
                    DispatchState.SUBMITTING,
                    DispatchState.UNKNOWN,
                }
            )
            return ToolResult.failed(
                "A2A delivery was acknowledged; inspect the retained dispatch before further action"
                if committed
                else "A2A delivery is uncertain; reconcile the dispatch before sending again"
                if uncertain
                else f"Could not send A2A {verb}: {error}",
                data={
                    "sent": True if committed else None if uncertain else False,
                    "recipient": recipient,
                    "remote_task_id": current.remote_task_id if committed else None,
                    "dispatch_id": dispatch.dispatch_id
                    if dispatch is not None
                    else None,
                    "delivery_state": current.state.value
                    if current is not None
                    else "not_submitted",
                },
            )

    @staticmethod
    def _native_dispatch_result(receipt, recipient: str) -> ToolResult:
        from a2a.types import Message
        from google.protobuf.json_format import MessageToDict

        data = {
            "sent": True,
            "recipient": recipient,
            "dispatch_id": receipt.dispatch.dispatch_id,
            "remote_task_id": receipt.dispatch.remote_task_id,
            "context_id": receipt.dispatch.remote_context_id,
            "state": receipt.dispatch.remote_task_state,
            "response_kind": receipt.dispatch.response_kind,
        }
        if isinstance(receipt.response, Message):
            data["response"] = MessageToDict(receipt.response)
        return ToolResult.ok(
            confirmation=f"A2A {receipt.dispatch.verb} sent to {recipient}", data=data
        )

    @tool(
        name="send_a2a_message",
        description=(
            "Send an async message to another agent — fire-and-forget, "
            "no reply expected. Persists in the recipient's TaskStore "
            "and fires the a2a.task_submitted signal so they wake and "
            "see it on their next cognition turn, but the caller does "
            "NOT track lifecycle. Use this for notifications, FYIs, "
            "status updates ('I just shipped PR 42'). For a tracked "
            "work assignment use send_a2a_task; for a synchronous "
            "Q&A use send_a2a_question."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a tell",
    )
    @execution_work_operation
    async def send_a2a_message(
        self, recipient: str, message: str, context_id: str = ""
    ) -> ToolResult:
        """Send one native message; its immutable dispatch ID is sender-owned."""
        receipt = await self._dispatch_native_peer(
            recipient,
            message,
            context_id=context_id,
            parts=None,
            skill_id="",
            verb="message",
            dispatch_tool="send_a2a_message",
        )
        if isinstance(receipt, ToolResult):
            return receipt
        return self._native_dispatch_result(receipt, recipient)

    @tool(
        name="send_a2a_question",
        description=(
            "Ask another agent a question. Fire-and-resume: this tool "
            "POSTs the question, spawns a background SSE subscription "
            "on the recipient's task, and returns IMMEDIATELY with "
            "``awaiting_reply=True``. Your current turn ends here. "
            "When the recipient's task reaches a terminal state, the "
            "``a2a.question_answered`` signal fires a fresh COGNITION "
            "turn on your dispatcher with the reply text inline — "
            "respond there. If ``timeout_seconds`` passes first, you "
            "are woken with ``state='expired'`` and the question is "
            "withdrawn (the recipient's task is canceled); the wake's "
            "``withdrawn`` flag says whether that was confirmed. Do NOT block "
            "your turn waiting for the answer; the supervisor will wake "
            "you. For fire-and-forget "
            "use send_a2a_message; for tracked work you'll check on "
            "later use send_a2a_task.\n\n"
            "SEND-SIDE ARTIFACTS: pass ``artifacts`` and/or "
            "``references`` to attach durable payload (planning docs, "
            "evidence, saved-memory/recall references) to the question "
            "so the recipient can retrieve it from the task store while "
            "answering. This is the SEND side — distinct from the "
            "RESPONDER-side attach_artifact_to_a2a_task tool a recipient "
            "uses to attach output onto an incoming task before "
            "responding."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a ask",
    )
    @execution_work_operation
    async def send_a2a_question(
        self,
        recipient: str,
        message: str,
        session_id: str = "",
        timeout_seconds: int = 300,
        # NOTE on the annotations: dropping ``Optional[...]`` is
        # deliberate — the @tool decorator's schema generator
        # (kestrel_sdk.features.base) reads ``get_origin``, which
        # returns ``Union`` for ``Optional[List[Any]]`` and falls
        # through to ``"string"`` in its type_map. That makes the
        # LLM-facing schema advertise these params as strings, so
        # the LLM passes JSON-encoded blobs that the strict
        # validator in ``_coerce_outbound_artifacts`` now rejects.
        # ``List[Any] = None`` works at runtime (Python doesn't
        # enforce defaults against annotations) and the schema
        # correctly renders ``array`` of ``object``. Codex round 2
        # P2 on PR #1628.
        artifacts: List[Any] = None,
        references: List[Any] = None,
    ) -> ToolResult:
        """
        Submit an A2A question to a peer agent under the fire-and-resume
        contract (#1444). The POST happens synchronously (so transport
        failures are surfaced to the caller immediately), but the wait
        for the answer does NOT block the current turn — it's handled
        by a background subscription supervisor that:

        1. Records a ``pending_a2a_questions`` row keyed by ``task_id``
        2. Opens an SSE stream to the recipient's
           ``/api/agent/tasks/{task_id}/subscribe`` endpoint
        3. On terminal status frame (completed/failed/canceled),
           extracts the reply text, marks the pending row RESOLVED,
           and enqueues a local ``a2a.question_answered`` signal so a
           fresh COGNITION turn fires on the sender with the reply
           inline.
        4. Auto-reconnects with backoff (1s/2s/5s/10s capped) on
           transient httpx failures until terminal or deadline.
        5. At the deadline, withdraws the question from the recipient
           (#3440) — see :meth:`_expire_pending_question`.

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The question / prompt.
            session_id: Optional A2A session id.
            timeout_seconds: Wall-clock cap on the supervisor. Past this
                deadline the question is withdrawn from the recipient
                (its task is canceled, so it is neither answered nor
                re-delivered) and the asker is woken by a synthetic
                ``a2a.question_answered`` signal with
                ``state='expired'`` — or with the recipient's own answer
                if it arrived first. A recipient that cannot be reached
                is not confirmed withdrawn (the wake says so) and is
                withdrawn later, or its late answer relayed. The hourly
                expiry sweep is the backstop when the supervisor is gone.
                Default 300s.
            artifacts: Optional send-side handoff payload attached to the
                question. Each item is a dict with ``name`` and a body
                (``text`` for raw text, ``data`` for a structured dict, or
                pre-shaped ``parts``), plus optional ``description``,
                ``metadata``, ``index``, ``last_chunk``. Persisted on the
                recipient's task so it can retrieve them while answering.
            references: Optional durable references (pointers to
                saved-memory / recall items, URIs). Each item is a dict
                descriptor; carried as structured-data artifacts in the
                ``references`` group.
        """
        task_data, chain, recipient_agent_id, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
            extra_metadata={
                "a2a_verb": "question",
                "reply_expected": True,
            },
            artifacts=artifacts, references=references,
            dispatch_tool="send_a2a_question",
        )
        if err is not None:
            return err

        task_id = task_data["id"]
        sess_id = task_data["sessionId"]
        # Compute UTC deadline once — same value lands in the pending
        # row and in the supervisor's monotonic loop cap. We store ISO
        # for cross-backend portability, then convert back to monotonic
        # inside the supervisor.
        from datetime import datetime, timedelta, timezone
        deadline_utc = (
            datetime.now(timezone.utc)
            + timedelta(seconds=max(int(timeout_seconds), 1))
        )

        # Hard requirement: store + dispatcher must be wired. Skip with
        # a clear error message if the agent didn't initialize them
        # (e.g. mid-boot tool call) rather than silent fallback.
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is None:
            return ToolResult.failed(
                "send_a2a_question is unavailable: agent has no "
                "pending_a2a_questions store wired. This indicates a "
                "boot-order bug — file an issue.",
                data={
                    "sent": True,
                    "awaiting_reply": False,
                    "task_id": task_id,
                    "recipient": recipient,
                },
            )

        # Record the in-flight correlation row before spawning the
        # supervisor so a process crash between POST and supervisor
        # start is recoverable via the startup-replay sweep.
        try:
            await store.insert(
                task_id=task_id,
                recipient=recipient,
                recipient_agent_id=recipient_agent_id,
                original_question=message,
                origin_turn_id=self._safe_get_current_turn_id(),
                origin_session_id=sess_id,
                deadline=deadline_utc,
            )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            # Codex round 3 P2d on PR #1453: without a pending row the
            # supervisor's mark_resolved would return False on the
            # terminal frame and silently drop the resumption signal as
            # a duplicate — the asking lineage would never resume even
            # though the task was sent and the receiver answered.
            # Surface this as a failure so the caller knows fire-and-
            # resume is NOT in play: the task was POSTed (receiver will
            # still act), but resumption is broken.
            logger.error(
                "Failed to record pending_a2a_question for task=%s "
                "recipient=%s: %s. Failing the tool call rather than "
                "silently losing the resumption signal.",
                task_id, recipient, e, exc_info=True,
            )
            return ToolResult.failed(
                f"Question was POSTed to {recipient} (task_id={task_id}) "
                f"but the local pending-questions store rejected the "
                f"correlation row ({type(e).__name__}: {e}). Without "
                f"that row, the a2a.question_answered signal cannot "
                f"fire — your turn will NOT be resumed when "
                f"{recipient} answers. The receiver will still process "
                f"the task; you can fetch the result manually with "
                f"get_peer_task_result.",
                data={
                    "sent": True,
                    "awaiting_reply": False,
                    "task_id": task_id,
                    "session_id": sess_id,
                    "recipient": recipient,
                    "store_error": f"{type(e).__name__}: {e}",
                },
            )

        # The POST and durable correlation insert completed under the caller's
        # admission. Publish only its registered subscription as a resident
        # source, not a continuation of that ending turn. The supervisor still
        # revalidates the persisted outbound route and current peer directory;
        # correlation alone never authorizes a route or a cognition turn.
        # Spawn the supervisor as a FEATURE-owned resident task. It runs the
        # SSE loop, fires the a2a.question_answered signal on terminal frame,
        # and exits. Still agent-tracked (auto-cancelled at full agent shutdown
        # by ``_shutdown_background_tasks``), but also owned by this feature so
        # runtime disable / boot rollback / soft disable cancel it via
        # ``Feature.shutdown()`` — an agent-only task would keep supervising
        # (and could still fire a resumption signal) after this feature is torn
        # down (kestrel-sovereign#2522 P1). Same ownership as the startup-replay
        # supervisor and the hourly expiry sweep.
        self._track_owned_runtime_task(
            self._supervise_a2a_question(
                task_id=task_id,
                recipient=recipient,
                recipient_agent_id=recipient_agent_id,
                original_question=message,
                sess_id=sess_id,
                deadline_utc=deadline_utc,
                causation_chain=chain,
            ),
            name=f"a2a_question_supervisor:{recipient}:{task_id}",
        )

        return ToolResult.ok(
            confirmation=(
                f"Question sent to {recipient} (task_id={task_id}). "
                f"Your turn ends now — the a2a.question_answered "
                f"signal will fire a fresh cognition turn with the "
                f"reply when {recipient} reaches a terminal state "
                f"(or {timeout_seconds}s elapses)."
            ),
            data={
                "sent": True,
                "awaiting_reply": True,
                "task_id": task_id,
                "session_id": sess_id,
                "recipient": recipient,
                "expires_at": deadline_utc.isoformat(),
                "resume_via": "a2a.question_answered",
            },
        )

    @tool(
        name="get_peer_task_result",
        description=(
            "Read the full standard Task for your immutable outbound dispatch_id. "
            "Reconciles uncertain original delivery by peer reads without resending. "
            "The retained ledger supplies the stable peer, tenant and remote task ID."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a result",
    )
    @execution_work_operation
    async def get_peer_task_result(self, dispatch_id: str) -> ToolResult:
        from a2a.utils.errors import A2AError
        from google.protobuf.json_format import MessageToDict

        from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
        from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
        from kestrel_sovereign.a2a.projections import message_text
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            DispatchState,
            OutboundDispatchConflict,
        )

        try:
            async with self._open_native_dispatch(dispatch_id) as (session, dispatch):
                if dispatch.state in {DispatchState.SUBMITTING, DispatchState.UNKNOWN}:
                    reconciled = await session.reconcile(dispatch)
                    if reconciled is None:
                        return ToolResult.failed(
                            "Delivery is still uncertain; no matching authorized peer Task was found",
                            data={
                                "dispatch_id": dispatch_id,
                                "delivery_state": "unknown",
                            },
                        )
                    dispatch = reconciled
                receipt = await session.get_task(dispatch)
        except (
            PeerDirectoryError,
            A2AError,
            OutboundDispatchConflict,
            NativePeerProtocolError,
            NativeSenderProofError,
        ) as error:
            if is_execution_control_error(error):
                raise
            return ToolResult.failed(
                f"Could not read native peer task: {error}",
                data={"dispatch_id": dispatch_id},
            )
        task = receipt.response
        reply = (
            message_text(task.status.message) if task.status.HasField("message") else ""
        )
        return ToolResult.ok(
            confirmation="Read native peer task",
            data={
                "dispatch_id": dispatch_id,
                "remote_task_id": task.id,
                "state": receipt.dispatch.remote_task_state,
                "reply_text": reply,
                "task": MessageToDict(task),
            },
        )

    @execution_work_operation
    async def cancel_outbound_task(
        self,
        task_id: str,
        reason: Optional[str] = None,
        *,
        local_recipient_match: bool = False,
    ) -> Optional[ToolResult]:
        """Route a local dispatch handle to its reauthorized native recipient.

        TaskFeature owns the public cancellation tool. An exact absent outbound
        handle permits its independently proven local inbox path; unreadable or
        ambiguous evidence always fails closed. The protocol carries no reason
        field, so an optional operator reason remains local presentation.
        """
        from a2a.utils.errors import A2AError

        from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
        from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            DispatchState,
            OutboundDispatchConflict,
        )

        try:
            store = self._native_outbound_store()
            dispatch = await store.get(task_id)
            if dispatch is None:
                return (
                    None
                    if local_recipient_match
                    else ToolResult.failed(
                        "No sender-owned outbound dispatch exists",
                        data={"dispatch_id": task_id},
                    )
                )
            if local_recipient_match:
                return ToolResult.failed(
                    "Handle is ambiguous between inbound and outbound work",
                    data={"dispatch_id": task_id, "error_type": "ambiguous_direction"},
                )
            async with self._open_native_dispatch(task_id) as (session, dispatch):
                if dispatch.state in {DispatchState.SUBMITTING, DispatchState.UNKNOWN}:
                    reconciled = await session.reconcile(dispatch)
                    if reconciled is None:
                        return ToolResult.failed(
                            "Delivery remains uncertain; cancellation has no acknowledged remote task",
                            data={"dispatch_id": task_id, "delivery_state": "unknown"},
                        )
                    dispatch = reconciled
                if dispatch.cancel_state in {
                    DispatchState.SUBMITTING,
                    DispatchState.UNKNOWN,
                }:
                    receipt = await session.get_task(dispatch)
                    if receipt.dispatch.state is not DispatchState.COMPLETE:
                        return ToolResult.failed(
                            "Cancellation remains uncertain; the peer task is still active",
                            data={
                                "dispatch_id": task_id,
                                "remote_task_id": dispatch.remote_task_id,
                                "cancel_state": receipt.dispatch.cancel_state.value,
                            },
                        )
                elif dispatch.state is DispatchState.COMPLETE:
                    receipt = await session.get_task(dispatch)
                else:
                    receipt = await session.cancel_task(dispatch)
        except (
            PeerDirectoryError,
            A2AError,
            OutboundDispatchConflict,
            NativePeerProtocolError,
            NativeSenderProofError,
        ) as error:
            if is_execution_control_error(error):
                raise
            return ToolResult.failed(
                f"Could not confirm native task cancellation: {error}",
                data={"dispatch_id": task_id, "cancellation_confirmed": False},
            )
        state = receipt.dispatch.remote_task_state
        if state != "canceled":
            return ToolResult.failed(
                f"Peer task is already terminal: {state}",
                data={
                    "dispatch_id": task_id,
                    "remote_task_id": receipt.dispatch.remote_task_id,
                    "state": state,
                },
            )
        return ToolResult.ok(
            confirmation="Peer task cancellation confirmed",
            data={
                "dispatch_id": task_id,
                "remote_task_id": receipt.dispatch.remote_task_id,
                "state": state,
                "cancellation_confirmed": True,
                "local_reason": reason,
            },
        )

    @asynccontextmanager
    async def _open_native_dispatch(self, dispatch_id: str):
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            OutboundDispatchConflict,
        )

        if not isinstance(dispatch_id, str) or not dispatch_id.strip():
            raise OutboundDispatchConflict("An immutable local dispatch_id is required")
        store = self._native_outbound_store()
        dispatch = await store.get(dispatch_id)
        if dispatch is None:
            raise OutboundDispatchConflict("No sender-owned native dispatch exists")
        router, requester, peer = await self._resolve_retained_automatic_peer(
            "", dispatch.recipient_id
        )
        async with router.open_a2a_session(requester, peer) as session:
            yield session, dispatch

    def _native_outbound_store(self):
        from kestrel_sovereign.a2a.task_manager import (
            OutboundActorScopeError,
            TaskManager,
        )
        from kestrel_sovereign.features.storage_access import (
            AgentIdentityUnavailable,
            resolve_scoped_agent_did,
        )

        manager = getattr(self.agent, "task_manager", None)
        if not isinstance(manager, TaskManager):
            raise PeerDirectoryConfigurationError(
                "Native outbound audit requires the agent's canonical task manager"
            )
        try:
            owner = resolve_scoped_agent_did(self.agent)
            return manager.outbound_for_actor(owner)
        except AgentIdentityUnavailable as error:
            raise PeerAccessDeniedError(
                "Outbound audit has no authenticated local owner"
            ) from error
        except OutboundActorScopeError as error:
            raise PeerAccessDeniedError(str(error)) from error

    @tool(
        name="list_outbound_a2a_tasks",
        description=(
            "List your native outbound A2A dispatch audit. Each row has an immutable "
            "dispatch_id, stable recipient, selected tenant and interface, delivery state, "
            "and the nullable remote_task_id and last observed task state. "
            "Message bodies are not stored in this audit."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a outbound",
    )
    @execution_work_operation
    async def list_outbound_a2a_tasks(
        self, limit: int = 50, recipient: str = ""
    ) -> ToolResult:
        """List this sender's native dispatch audit without fetching peer data."""
        from dataclasses import asdict

        try:
            store = self._native_outbound_store()
            recipient_id = None
            if recipient:
                _, _, peer = await self._resolve_automatic_peer(recipient)
                recipient_id = peer.agent_id
            rows = await store.list_recent(limit=limit, recipient_id=recipient_id)
        except (PeerDirectoryError, ValueError) as error:
            if is_execution_control_error(error):
                raise
            return ToolResult.failed(str(error), data={"rows": [], "count": 0})
        return ToolResult.ok(
            confirmation=f"Found {len(rows)} outbound A2A dispatches",
            data={"rows": [asdict(row) for row in rows], "count": len(rows)},
        )

    # ------------------------------------------------------------------
    # Subscription supervisor (#1444)
    #
    # When ``send_a2a_question`` POSTs an outbound task, this helper is
    # spawned as a tracked background coroutine. It opens an SSE stream
    # to ``GET /tasks/{task_id}/subscribe`` on the recipient, parses
    # ``status`` events, and when the terminal frame arrives:
    #
    #   - marks the pending row RESOLVED
    #   - extracts reply text from the terminal status.message.parts
    #   - builds an ``a2a.question_answered`` signal
    #   - enqueues it on the local dispatcher so a fresh COGNITION turn
    #     fires with the reply inline
    #
    # Reconnect: transient httpx failures back off 1s/2s/5s/10s capped,
    # restarting the stream until either (a) we see a terminal frame,
    # or (b) the wall-clock deadline passes. The hourly expiry sweep
    # is the deadline backstop — even if this supervisor goes silent
    # after a process crash, the resumption rail still fires.
    # ------------------------------------------------------------------

    def _safe_get_current_turn_id(self) -> Optional[str]:
        """Best-effort read of the in-flight turn id. The agent may not
        expose ``_get_current_turn_id`` in every embed (e.g. tests with
        a partial agent stub) — fall back to None rather than raising."""
        fn = getattr(self.agent, "_get_current_turn_id", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception as control_error:
            if is_execution_control_error(control_error):
                raise
            return None

    @execution_work_operation
    async def _supervise_a2a_question(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        deadline_utc: Any,
        causation_chain: Optional[list],
        recipient_agent_id: Optional[str] = None,
    ) -> None:
        """Background coroutine: SSE-subscribe → fire signal on terminal.

        Runs until the recipient's task is terminal or ``deadline_utc``
        passes. Reconnects on transient httpx failure with exponential
        backoff. Errors are logged not re-raised — supervisor death
        must not surface as an unhandled task exception."""
        import asyncio
        from datetime import datetime, timezone

        terminal_states = A2A_TERMINAL_TASK_STATES
        backoffs = [1.0, 2.0, 5.0, 10.0]
        backoff_idx = 0
        state: Optional[str] = None
        reply_text = ""

        try:
            recipient_agent_id = await self._question_recipient_agent_id(
                task_id, recipient_agent_id,
            )
        except (
            PeerNotFoundError,
            PeerAccessDeniedError,
            PeerSelfTargetError,
        ) as control_error:
            if is_execution_control_error(control_error):
                raise
            state = "failed"
            reply_text = "Peer task subscription is no longer authorized."
        except PeerDirectoryConfigurationError as control_error:
            if is_execution_control_error(control_error):
                raise
            state = "failed"
            reply_text = "Peer routing is no longer configured safely."

        def _remaining() -> float:
            return max(
                0.0,
                (deadline_utc - datetime.now(timezone.utc)).total_seconds(),
            )

        while _remaining() > 0 and state not in terminal_states:
            # Pass the remaining wall-clock to the router so its transport
            # cannot block a stalled stream past the promised deadline.  The
            # local HTTP adapter uses it for connect/read/pool timeouts;
            # hosted adapters receive the same bounded contract.
            remaining = _remaining()
            if remaining < 0.5:
                break
            try:
                router, requester, peer = await self._resolve_retained_automatic_peer(
                    recipient, recipient_agent_id,
                )
                async with owned_execution_stream(self, router.subscribe_a2a_task(
                    requester,
                    peer,
                    task_id,
                    timeout_seconds=remaining,
                )) as subscription:
                    async for subscription_event in subscription:
                        # Resolution alone does not prove the subscription path
                        # is healthy: a local host can fail immediately while
                        # opening the stream.  Reset only after the provider has
                        # yielded an event, so repeated transport failures retain
                        # the 1/2/5/10-second progression.
                        backoff_idx = 0
                        # Codex round 3 P2c on PR #1453: enforce the deadline
                        # INSIDE the stream loop.  A provider can keep a healthy
                        # stream open indefinitely, so its transport timeout alone
                        # is not the deadline guarantee.
                        if _remaining() <= 0:
                            break
                        event_name = subscription_event.event or "message"
                        data_str = subscription_event.data or ""
                        if event_name in ("keepalive", "ping"):
                            continue
                        if event_name != "status":
                            continue
                        parsed = self._parse_sse_status_data(data_str)
                        if not parsed:
                            continue
                        event_state, event_reply = parsed
                        if event_state in terminal_states:
                            state = event_state
                            reply_text = event_reply
                            break
                # A cleanly exhausted stream also proves that subscription
                # setup succeeded, even when the peer emitted no event.
                backoff_idx = 0
                # Stream ended cleanly — if we saw a terminal, exit the outer
                # loop; otherwise reconnect (or exit at the deadline).
                if state in terminal_states:
                    break
            except PeerSubscriptionUnavailableError as control_error:
                if is_execution_control_error(control_error):
                    raise
                # Hard cut: recipient lacks the subscription surface.  Don't
                # burn the whole deadline reconnecting to a legacy peer.
                logger.error(
                    "A2A question supervisor for task=%s recipient=%s: "
                    "subscription unavailable. Marking pending row failed.",
                    task_id, recipient,
                )
                state = "failed"
                reply_text = (
                    f"Recipient '{recipient}' does not expose "
                    f"/tasks/{{id}}/subscribe — upgrade them to the build "
                    f"that ships the fire-and-resume A2A question protocol "
                    f"(#1444)."
                )
                break
            except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError) as control_error:
                if is_execution_control_error(control_error):
                    raise
                # Scope changes and cross-scope probes must not reveal whether
                # the recipient or task exists.  This sender had a prior task,
                # so fail its resumption safely rather than retrying a route it
                # is no longer authorized to observe.
                state = "failed"
                reply_text = "Peer task subscription is no longer authorized."
                break
            except PeerDirectoryConfigurationError as control_error:
                if is_execution_control_error(control_error):
                    raise
                state = "failed"
                reply_text = "Peer routing is no longer configured safely."
                break
            except PeerTransportError as exc:
                if is_execution_control_error(exc):
                    raise
                logger.debug(
                    "A2A subscription stream for task=%s recipient=%s "
                    "dropped (%s); backing off",
                    task_id, recipient, exc,
                )
            except PeerDirectoryError as exc:
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "A2A subscription supervisor for task=%s "
                    "recipient=%s router error: %s",
                    task_id, recipient, exc,
                )
            except Exception as exc:  # noqa: BLE001 - provider extension boundary
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "A2A subscription supervisor for task=%s "
                    "recipient=%s unexpected router error: %s",
                    task_id, recipient, type(exc).__name__,
                )

            if state in terminal_states:
                break
            # Backoff + retry, but only if we have remaining wall-clock.
            backoff = backoffs[min(backoff_idx, len(backoffs) - 1)]
            backoff_idx += 1
            await asyncio.sleep(min(backoff, _remaining()))

        if state not in terminal_states:
            # Deadline passed without terminal. Expire NOW
            # (deadline-accurate) rather than letting the caller wait up to
            # an hour for the hourly sweep — promised wake-by-deadline must
            # actually happen at the deadline (codex round 2 P2a on PR
            # #1453).
            logger.info(
                "A2A subscription supervisor for task=%s recipient=%s "
                "exited at deadline without terminal frame. Expiring the "
                "question.",
                task_id, recipient,
            )
            await self._expire_pending_question(
                getattr(self.agent, "pending_a2a_questions", None),
                task_id=task_id,
                recipient=recipient,
                recipient_agent_id=recipient_agent_id,
                original_question=original_question,
                sess_id=sess_id,
                causation_chain=causation_chain,
            )
            return

        # Terminal: claim the row (resolving it) + fire local signal.
        # Resolve-first so the startup-replay sweep doesn't double-fire if it
        # raced this supervisor to the same terminal event. The answer stays
        # owed on the row until the fire confirms its delivery (#3440).
        store = getattr(self.agent, "pending_a2a_questions", None)
        claim: Optional[_QuestionClaim] = None
        if store is not None:
            claim = self._new_question_claim(store)
            try:
                was_waiting = await self._claim_question_wake(
                    claim, task_id, state=state, reply_text=reply_text,
                )
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(
                    "Failed to mark pending_a2a_question task=%s "
                    "resolved: %s. Firing signal anyway — the resumed "
                    "turn should not be lost to a write failure.",
                    task_id, e,
                )
                was_waiting = True
                # Whether the claim took is unknown, so it cannot vouch for
                # this pass: the wake fires unfenced, and the row (if the
                # claim did take) is redelivered once its lease lapses.
                claim = None
            if not was_waiting:
                # Someone else (startup-replay sweep, hourly expiry) got
                # there first. They own the signal fire; drop ours.
                logger.debug(
                    "A2A pending row for task=%s already terminal — "
                    "dropping duplicate signal from supervisor.",
                    task_id,
                )
                return

        fired = await self._fire_question_answered_signal(
            task_id=task_id,
            recipient=recipient,
            original_question=original_question,
            sess_id=sess_id,
            state=state,
            reply_text=reply_text,
            causation_chain=causation_chain,
            claim=claim,
        )
        if not fired:
            await self._restore_pending_question_waiting(
                task_id,
                claim=claim,
                state=state,
                reply_text=reply_text,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                causation_chain=causation_chain,
            )

    async def _question_recipient_agent_id(
        self, task_id: str, recipient_agent_id: Optional[str],
    ) -> Optional[str]:
        """Return the stable recipient identity a question may be routed to.

        Pending-question rows are correlation/audit state, not route
        authority.  Re-read the outbound route whenever it exists rather than
        trusting a value passed by the immediate-send path or a restart replay
        row.  Hosted sends require this route; local sends with a persisted
        reservation use the same check so a failed divergent-id rekey cannot
        reach a replacement peer after restart.  A no-store legacy local send
        remains compatible by falling back to its already-recorded recipient
        identity/name.
        """
        durable_recipient_agent_id = await self._outbound_recipient_agent_id(
            task_id,
        )
        if durable_recipient_agent_id is None:
            return recipient_agent_id
        if (
            recipient_agent_id is not None
            and recipient_agent_id != durable_recipient_agent_id
        ):
            raise PeerNotFoundError(
                "Pending question recipient does not match outbound route"
            )
        return durable_recipient_agent_id

    # ------------------------------------------------------------------
    # Expiry is propagated to the recipient (#3440)
    #
    # A question whose deadline passes is withdrawn from the recipient with
    # the creator's signed cancellation before the asker is told ``expired``.
    # The recipient's store then agrees with the sender's: its queued wake is
    # suppressed, a stale re-delivery is refused at execution, and a late
    # ``respond_to_a2a_task`` is refused as a terminal-state transition. If
    # the recipient answered first, the cancellation conflicts and the asker
    # is woken with that answer instead of ``expired``. A recipient that
    # cannot be reached is settled later by the sweep, which relays its late
    # answer if one arrives. Both stores must agree before a row is retired,
    # however long the recipient is away: it keeps an undelivered submission
    # wake for as long as that wake stays undelivered. Whichever wake the
    # asker is owed stays on the sender's row until its delivery is
    # confirmed, so a crash in between delays it but cannot lose it.
    # ------------------------------------------------------------------

    # Bound on settling one question with its recipient. The deadline wake
    # waits for it, so it must stay short; an unreached recipient is retried
    # by the sweep.
    QUESTION_SETTLEMENT_TIMEOUT_SECONDS = 15.0
    # Seconds between renewals of a question's claim lease while a wake for
    # it is in flight. ``None`` renews three times per lease.
    QUESTION_CLAIM_LEASE_RENEW_SECONDS: Optional[float] = None
    QUESTION_EXPIRY_CANCEL_REASON = (
        "The asking agent's deadline for this question passed before an "
        "answer arrived, so it withdrew the question."
    )

    async def _settle_question_with_receiver(
        self,
        *,
        task_id: str,
        recipient: str,
        recipient_agent_id: Optional[str] = None,
    ) -> ReceiverSettlement:
        """Make the recipient's task terminal, or learn the terminal state it
        already reached. Bounded by ``QUESTION_SETTLEMENT_TIMEOUT_SECONDS``."""
        import asyncio

        try:
            waiter = create_owned_task(
                self._settle_question_with_receiver_unbounded(
                    task_id=task_id,
                    recipient=recipient,
                    recipient_agent_id=recipient_agent_id,
                ),
            )
            primary = None
            timed_out = False
            try:
                done, _ = await asyncio.wait(
                    {waiter}, timeout=self.QUESTION_SETTLEMENT_TIMEOUT_SECONDS,
                )
                if not done:
                    timed_out = True
                    primary = TimeoutError("A2A receiver settlement deadline")
            except BaseException as error:
                primary = error
            if primary is not None and not waiter.done():
                waiter.cancel()
            outcome = await await_owned_task(waiter)
            child_error = outcome.error
            if (timed_out and isinstance(child_error, asyncio.CancelledError)
                    and not is_execution_control_error(child_error)):
                child_error = None
            terminal = execution_terminal_error(
                primary, child_error, outcome.cancellation,
            )
            if terminal is not None:
                raise terminal
            return outcome.result
        except TimeoutError as error:
            if is_execution_control_error(error):
                raise
            logger.warning(
                "A2A question task=%s: could not settle with recipient=%s "
                "within %.0fs; leaving the recipient's task unconfirmed.",
                task_id, recipient, self.QUESTION_SETTLEMENT_TIMEOUT_SECONDS,
            )
            return ReceiverSettlement(
                ReceiverSettlementOutcome.UNCONFIRMED,
                recipient_unresponsive=True,
            )

    async def _settle_question_with_receiver_unbounded(
        self,
        *,
        task_id: str,
        recipient: str,
        recipient_agent_id: Optional[str],
    ) -> ReceiverSettlement:
        withdrawal = await self.cancel_outbound_task(
            task_id, reason=self.QUESTION_EXPIRY_CANCEL_REASON,
        )
        if withdrawal is not None and withdrawal.status in (
            ToolResultStatus.OK, ToolResultStatus.PARTIAL,
        ):
            # PARTIAL is a committed cancellation whose sender-side audit
            # stamp failed; the recipient's task is canceled either way.
            return ReceiverSettlement(ReceiverSettlementOutcome.WITHDRAWN)
        # The cancellation did not take. Either the recipient already reached
        # a terminal state (a lifecycle conflict) or it could not be reached;
        # one read tells the two apart and carries a real answer to the asker.
        observed = await self._read_settled_peer_task(
            task_id=task_id,
            recipient=recipient,
            recipient_agent_id=recipient_agent_id,
        )
        if observed.receiver_terminal:
            return observed
        logger.info(
            "A2A question task=%s: recipient=%s task not confirmed terminal "
            "(withdrawal: %s).",
            task_id, recipient,
            withdrawal.error if withdrawal is not None else "no route",
        )
        # When both requests failed in transport (or found the recipient
        # unavailable), its other questions would fail the same way, so the
        # sweep leaves them for its next pass. A recipient that gave either
        # request a decision keeps its classification (#3586).
        unreachable = (
            observed.recipient_unresponsive
            and withdrawal is not None
            and isinstance(withdrawal.data, Mapping)
            and withdrawal.data.get("error_type")
            == RECIPIENT_UNREACHABLE_ERROR_TYPE
        )
        return ReceiverSettlement(
            ReceiverSettlementOutcome.UNCONFIRMED,
            recipient_unresponsive=unreachable,
        )

    async def _read_settled_peer_task(
        self,
        *,
        task_id: str,
        recipient: str,
        recipient_agent_id: Optional[str],
    ) -> ReceiverSettlement:
        """Read the recipient's task: ``UNCONFIRMED`` unless it is terminal,
        flagged ``recipient_unresponsive`` when the read failed in transport
        or found the recipient unavailable."""
        try:
            agent_id = await self._question_recipient_agent_id(
                task_id, recipient_agent_id,
            )
            router, requester, peer = await self._resolve_retained_automatic_peer(
                recipient, agent_id,
            )
            data = await router.get_a2a_task(requester, peer, task_id)
        except PeerDirectoryError as exc:
            if is_execution_control_error(exc):
                raise
            logger.debug(
                "A2A question task=%s: recipient=%s task unreadable: %s",
                task_id, recipient, type(exc).__name__,
            )
            return ReceiverSettlement(
                ReceiverSettlementOutcome.UNCONFIRMED,
                recipient_unresponsive=isinstance(
                    exc, (PeerTransportError, PeerUnavailableError),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(exc):
                raise
            logger.exception(
                "Peer router raised unexpectedly reading task %s from %r",
                task_id, recipient,
            )
            return ReceiverSettlement(ReceiverSettlementOutcome.UNCONFIRMED)
        if not isinstance(data, Mapping):
            return ReceiverSettlement(ReceiverSettlementOutcome.UNCONFIRMED)
        parsed = self._parse_task_status_payload(data)
        if parsed is None or parsed[0] not in A2A_TERMINAL_TASK_STATES:
            return ReceiverSettlement(ReceiverSettlementOutcome.UNCONFIRMED)
        state, reply_text = parsed
        if state == "canceled" and self._canceled_by_this_agent(data):
            # Our own withdrawal committed even though its response was lost.
            return ReceiverSettlement(ReceiverSettlementOutcome.WITHDRAWN)
        return ReceiverSettlement(
            ReceiverSettlementOutcome.ANSWERED,
            state=state,
            reply_text=reply_text,
        )

    def _canceled_by_this_agent(self, task_data: Mapping[str, Any]) -> bool:
        metadata = task_data.get("metadata")
        receipt = (
            metadata.get("cancellation_receipt")
            if isinstance(metadata, Mapping)
            else None
        )
        own_id = getattr(self.agent, "did", None)
        return (
            isinstance(receipt, Mapping)
            and isinstance(own_id, str)
            and receipt.get("actor_agent_id") == own_id
        )

    async def _expire_pending_question(
        self,
        store,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        causation_chain: Optional[list],
        recipient_agent_id: Optional[str] = None,
    ) -> None:
        """Settle a question whose deadline passed, then wake the asker once.

        The asker is woken with the recipient's own terminal state when the
        recipient answered first, and with ``expired`` otherwise, flagged
        ``withdrawn`` only when the recipient's task is confirmed canceled: an
        unreached recipient may still answer (#3440). Settlement
        runs BEFORE the WAITING row is claimed: a crash in between leaves the
        row for replay, and a racing expiry that settles too only repeats an
        idempotent cancellation before losing the claim. The claim then gates
        the wake, so a racing hourly sweep or supervisor drops its duplicate.

        The wake is owed on the claimed row until its delivery is confirmed,
        whichever way the recipient settled: a process that dies before then
        leaves it for :meth:`_redeliver_abandoned_wakes`, and settlement does
        not start on the row while an ``expired`` wake is still owed (#3440).
        """
        settlement = await self._settle_question_with_receiver(
            task_id=task_id,
            recipient=recipient,
            recipient_agent_id=recipient_agent_id,
        )
        answered = settlement.outcome is ReceiverSettlementOutcome.ANSWERED
        withdrawn = settlement.outcome is ReceiverSettlementOutcome.WITHDRAWN
        state = settlement.state if answered else "expired"
        reply_text = settlement.reply_text if answered else ""
        claim: Optional[_QuestionClaim] = None
        if store is not None:
            claim = self._new_question_claim(store)
            try:
                was_waiting = await self._claim_question_wake(
                    claim,
                    task_id,
                    state=state,
                    reply_text=reply_text,
                    receiver_settled=settlement.receiver_terminal,
                )
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(
                    "Failed to mark pending_a2a_question task=%s %s: %s. "
                    "Firing signal anyway — better a possible duplicate "
                    "than a missed resumption.",
                    task_id, state, e,
                )
                was_waiting = True
                # As in the supervisor: an unknown claim fires unfenced.
                claim = None
            if not was_waiting:
                # Someone else (the hourly sweep or the supervisor that beat
                # us by a tick) got there first — drop our duplicate signal.
                return
        fired = await self._fire_question_answered_signal(
            task_id=task_id,
            recipient=recipient,
            original_question=original_question,
            sess_id=sess_id,
            state=state,
            reply_text=reply_text,
            causation_chain=causation_chain,
            claim=claim,
            withdrawn=withdrawn,
        )
        if not fired:
            await self._restore_pending_question_waiting(
                task_id,
                claim=claim,
                state=state,
                reply_text=reply_text,
                withdrawn=withdrawn,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                causation_chain=causation_chain,
            )

    def _new_question_claim(self, store) -> _QuestionClaim:
        """A fresh claim to take on one of ``store``'s rows."""
        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            new_question_claim,
        )

        return _QuestionClaim(store, new_question_claim())

    async def _claim_question_wake(
        self,
        claim: _QuestionClaim,
        task_id: str,
        *,
        state: str,
        reply_text: str,
        receiver_settled: bool = False,
    ) -> bool:
        """Claim a WAITING question's row under ``claim`` to wake the asker
        with ``state`` and ``reply_text``; whether the row was WAITING.

        The row leaves WAITING with that wake owed on it. The claim keeps a
        racing supervisor, sweep or replay from waking the asker a second
        time, and the wake is retired only when
        :meth:`_fire_question_answered_signal` confirms its delivery (#3440).
        An ``expired`` wake keeps the receiver settlement the row already
        recorded, unless ``receiver_settled`` confirms the recipient's task
        terminal."""
        if state == "expired":
            was_waiting = await claim.store.mark_expired(
                task_id, claim=claim.token, receiver_settled=receiver_settled,
            )
        else:
            was_waiting = await claim.store.mark_resolved(
                task_id, state=state, reply_text=reply_text, claim=claim.token,
            )
        return bool(was_waiting)

    async def _settle_expired_questions_with_receivers(
        self, store, *, now: Optional[datetime] = None,
    ) -> None:
        """Settle every expired question its recipient does not yet agree on.

        A row is retried on every pass, with no age cutoff, until the
        recipient's task is withdrawn or the recipient's own answer has woken
        the asker (see ``list_expired_awaiting_receiver``). A recipient that
        does not answer within the settlement bound, or whose cancellation
        and read both fail in transport, costs one attempt per pass, not one
        per question: its other unsettled rows wait for the next pass.
        ``now`` is the pass's clock for the claim lease."""
        import asyncio

        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            RECEIVER_SETTLEMENT_LATE_ANSWER,
        )

        rows = await store.list_expired_awaiting_receiver()
        unresponsive: set[str] = set()
        for row in rows:
            recipient_key = row.recipient_agent_id or row.recipient
            owes_answer = (
                row.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER
            )
            if recipient_key in unresponsive and not owes_answer:
                continue
            try:
                settlement = await self._settle_expired_row_with_receiver(
                    store, row, now=now,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(
                    "a2a receiver settlement: failed for task=%s: %s",
                    row.task_id, e,
                )
                continue
            if settlement is not None and settlement.recipient_unresponsive:
                unresponsive.add(recipient_key)

    async def _settle_expired_row_with_receiver(
        self, store, row, *, now: Optional[datetime] = None,
    ) -> Optional[ReceiverSettlement]:
        """Settle one unsettled expired question under this pass's claim.

        The asker already holds an ``expired`` wake for this question (no
        pass can claim the row for settlement while that wake is still
        owed). A withdrawal only confirms the row. A terminal state the
        recipient reached on its own is first recorded on the row as the
        answer owed to the asker, then relayed to it flagged ``late``; the
        row is confirmed only once that wake is delivered. A pass that dies
        anywhere in between leaves the row claimed and unconfirmed, and the
        first pass after the lease lapses claims it and picks it up where it
        stopped: it re-settles a PENDING row and re-relays a held answer.

        Every step names this pass's claim. Once a later claim replaces it
        (this pass stalled past its lease and another took the row over), the
        store refuses the step and the relay refuses to enqueue, so a stale
        pass can neither confirm the row nor wake the asker a second time.

        Returns the settlement attempted with the recipient, or ``None`` when
        this pass made no attempt (the row was claimed elsewhere, or only owed
        a held answer)."""
        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            RECEIVER_SETTLEMENT_LATE_ANSWER,
        )

        settlement_claim = await store.claim_receiver_settlement(
            row.task_id, now=now,
        )
        if settlement_claim is None:
            return None
        claimed = settlement_claim.question
        claim = _QuestionClaim(store, settlement_claim.token)
        settlement: Optional[ReceiverSettlement] = None
        if claimed.receiver_settlement == RECEIVER_SETTLEMENT_LATE_ANSWER:
            state = claimed.receiver_late_state
            reply_text = claimed.receiver_late_reply_text or ""
        else:
            settlement = await self._settle_question_with_receiver(
                task_id=claimed.task_id,
                recipient=claimed.recipient,
                recipient_agent_id=claimed.recipient_agent_id,
            )
            if settlement.outcome is ReceiverSettlementOutcome.UNCONFIRMED:
                return settlement
            if settlement.outcome is ReceiverSettlementOutcome.WITHDRAWN:
                await store.confirm_receiver_withdrawn(
                    claimed.task_id, claim=claim.token,
                )
                return settlement
            if not await store.record_late_answer(
                claimed.task_id,
                claim=claim.token,
                state=settlement.state,
                reply_text=settlement.reply_text,
            ):
                return settlement
            state, reply_text = settlement.state, settlement.reply_text
        if state not in A2A_TERMINAL_TASK_STATES:
            logger.error(
                "pending_a2a_question task=%s holds a late answer with "
                "non-terminal state %r; it cannot be relayed.",
                claimed.task_id, state,
            )
            return settlement
        # Delivered, the relay confirms the row under this claim. Undelivered
        # (or no longer this pass's to relay), the answer stays held on the
        # row and the lease lapses, so a later pass relays it.
        await self._fire_question_answered_signal(
            task_id=claimed.task_id,
            recipient=claimed.recipient,
            original_question=claimed.original_question,
            sess_id=claimed.origin_session_id or "",
            state=state,
            reply_text=reply_text,
            causation_chain=None,
            late=True,
            claim=claim,
        )
        return settlement

    async def _holds_question_claim(
        self, task_id: str, claim: _QuestionClaim,
    ) -> bool:
        """Whether ``claim`` still holds its question's row, renewing its
        lease if so: the check every claimed wake makes immediately before it
        is enqueued (#3440). A failed check is treated as lost ownership: the
        wake stays owed on the row, under whichever claim holds it."""
        try:
            return bool(
                await claim.store.renew_claim_lease(task_id, claim=claim.token)
            )
        except Exception as exc:  # noqa: BLE001 - the row stays owed
            if is_execution_control_error(exc):
                raise
            logger.warning(
                "Could not confirm the claim on pending_a2a_question "
                "task=%s; not waking the asker from this pass: %s",
                task_id, exc,
            )
            return False

    def _renewing_claim_lease(
        self, handle: Any, task_id: str, claim: _QuestionClaim,
    ) -> Any:
        """Wrap ``handle`` so ``claim``'s lease is renewed while the wake is
        awaited.

        A wake waits for a whole cognition turn, which has no fixed length.
        Without renewal the lease would lapse mid-turn, and another pass
        could take the row over and wake the asker again, or relay a late
        answer ahead of the ``expired`` wake. A handle with nothing to wait
        on is returned as it is, so delivery still reports it as
        unobservable."""
        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            QUESTION_CLAIM_LEASE,
        )

        renew = getattr(claim.store, "renew_claim_lease", None)
        if not callable(renew) or not callable(getattr(handle, "wait", None)):
            return handle
        interval = self.QUESTION_CLAIM_LEASE_RENEW_SECONDS
        if interval is None:
            interval = QUESTION_CLAIM_LEASE.total_seconds() / 3

        async def _renew() -> bool:
            try:
                return bool(await renew(task_id, claim=claim.token))
            except Exception as exc:  # noqa: BLE001 - retried next interval
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "Could not renew the claim lease of pending_a2a_question "
                    "task=%s: %s",
                    task_id, exc,
                )
                return True

        return _ClaimLeaseRenewingHandle(
            handle, interval=interval, renew=_renew,
        )

    async def _retire_delivered_wake(
        self, task_id: str, claim: Optional[_QuestionClaim], *, late: bool,
    ) -> None:
        """Retire the wake ``claim`` delivered from its row (#3440).

        Until this runs the row still owes the asker that wake, so a failure
        here costs a duplicate wake once the claim lapses, never a lost one.
        """
        if claim is None:
            return
        try:
            if late:
                retired = await claim.store.confirm_late_answer_delivered(
                    task_id, claim=claim.token,
                )
            else:
                retired = await claim.store.confirm_wake_delivered(
                    task_id, claim=claim.token,
                )
        except Exception as exc:  # noqa: BLE001 - the row stays owed
            if is_execution_control_error(exc):
                raise
            logger.warning(
                "a2a.question_answered for task=%s was delivered, but its "
                "pending row could not be retired: %s. The asker will be "
                "woken again once the row's claim lapses.",
                task_id, exc,
            )
            return
        if not retired:
            logger.warning(
                "a2a.question_answered for task=%s was delivered after this "
                "pass lost its claim on the row; the pass now holding it "
                "may wake the asker again.",
                task_id,
            )

    async def _fire_question_answered_signal(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        state: str,
        reply_text: str,
        causation_chain: Optional[list],
        await_delivery: bool = True,
        schedule_retry: bool = True,
        late: bool = False,
        claim: Optional[_QuestionClaim] = None,
        withdrawn: bool = False,
    ) -> bool:
        """Build, enqueue, and DELIVER the local ``a2a.question_answered``
        signal, returning whether the asker was actually woken.

        Factored out so the supervisor AND the startup-replay /
        hourly-expiry sweeps share one fire path. Errors here are
        logged not raised — losing a resumption signal is bad but
        crashing the dispatcher hop is worse.

        The caller has already claimed the durable ``WAITING`` row
        (claim-first, so a racing sweep can't double-fire) under ``claim``,
        which left this wake owed on the row. A delivered wake is retired from
        the row here, under that claim, so ``True`` here is what retires it:
        a process that dies before then leaves the wake owed, to be delivered
        again once the claim lapses (#3440). ``enqueue_signal`` only reports
        *acceptance onto the queue*, which is not delivery — returning
        ``True`` on it left the row terminal for a wake the dispatcher then
        failed or dropped, and the asker was never resumed (#2532). So
        ``True`` now means terminal ``Status.OK``.

        Ownership boundary (#2532): ``await_delivery=True`` waits for that
        terminal result inline and is correct for every FEATURE-OWNED
        background caller — the SSE supervisor, the hourly sweep loop, the
        restore-retry loop — which can block for the length of the resumed
        cognition turn. ``await_delivery=False`` is for the one caller that
        cannot: ``_replay_pending_a2a_questions`` runs inline on the boot path,
        where awaiting a cognition turn would deadlock startup against the very
        agent that is still initializing. That path hands the wait to a
        feature-owned supervisor which restores the ``WAITING`` row itself on
        any non-delivery, and returns ``True`` for "accepted, delivery
        supervised" so the caller does not also restore it; that supervisor
        retires the delivered wake itself.

        ``late`` relays a recipient's answer after the asker was already woken
        ``expired`` (#3440). Its row stays EXPIRED and keeps holding that
        answer until its delivery confirms the row, so an undelivered late
        wake has nothing to restore.

        ``claim`` is the claim on the row this wake is for: the one that took
        it out of WAITING, or the settlement pass's (a late relay). Its lease
        is renewed for as long as the wake is in flight. ``None`` means there
        is no row to retire (no pending-question store, or a claim whose
        write failed). A wake that carries one is enqueued only if that claim
        still holds the row at that moment: a pass that stalled and lost the
        row to another wakes the asker with nothing, and the row's wake stays
        with the pass now holding it. A late relay must carry one.

        ``withdrawn`` flags an ``expired`` wake whose question is confirmed
        withdrawn (the recipient's task canceled), so a resumed turn is never
        told that of a task that may still be live (#3440).
        """
        import asyncio

        from kestrel_sovereign.signals.delivery import (
            STATUS_SUPERVISOR_CANCELLED,
            await_terminal_delivery,
            supervise_terminal_delivery,
        )
        from kestrel_sovereign.signals.sources.a2a_question_answered import (
            build_signal_for_question_answered,
        )

        # #1576 codex round 2 P1: every terminal-state observation for
        # a sent question must stamp the outbound audit row. Supervisor
        # SSE terminal, supervisor deadline expiry, hourly sweep, and
        # startup replay all funnel through here — so this is the one
        # place to ensure the audit closes. Without this, questions
        # would complete via SSE / expire / get swept and the audit
        # row would still show ``terminal_state = NULL`` even though
        # the sender knew the state.
        _audit_db = getattr(self, "_db", None)
        if _audit_db is not None:
            try:
                from kestrel_sovereign.a2a.outbound_store import (
                    update_outbound_terminal_state,
                )
                await update_outbound_terminal_state(
                    _audit_db,
                    agent_id=str(
                        getattr(self.agent, "did", None) or self._own_name
                    ),
                    task_id=task_id,
                    terminal_state=state,
                )
            except Exception as exc:  # noqa: BLE001
                if is_execution_control_error(exc):
                    raise
                logger.debug(
                    "outbound_store: question-answered terminal stamp "
                    "failed for task=%s state=%s: %s",
                    task_id, state, exc,
                )

        dispatcher = getattr(self.agent, "dispatcher", None)
        if dispatcher is None:
            logger.error(
                "Cannot fire a2a.question_answered for task=%s — "
                "agent has no dispatcher.",
                task_id,
            )
            return False

        try:
            target_agent = getattr(self.agent, "did", None) or self._own_name
            signal = build_signal_for_question_answered(
                task_id=task_id,
                recipient=recipient,
                original_question=original_question,
                reply_text=reply_text or "",
                state=state,
                target_agent=target_agent,
                origin_session_id=sess_id,
                causation_chain=causation_chain,
                late=late,
                withdrawn=withdrawn,
            )
            if late and claim is None:
                logger.error(
                    "Not relaying the late answer for task=%s recipient=%s: "
                    "a late relay needs the claim on its row.",
                    task_id, recipient,
                )
                return False
            if claim is not None and not await self._holds_question_claim(
                task_id, claim,
            ):
                logger.info(
                    "Not waking the asker for task=%s recipient=%s: this "
                    "pass no longer holds its row.",
                    task_id, recipient,
                )
                return False
            enq = dispatcher.enqueue_signal(signal)
            handle = await enq if hasattr(enq, "__await__") else enq
            if claim is not None:
                # Hold the row's claim for as long as this wake is in flight.
                handle = self._renewing_claim_lease(handle, task_id, claim)
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.error(
                "Failed to enqueue a2a.question_answered for task=%s "
                "recipient=%s: %s",
                task_id, recipient, e,
                exc_info=True,
            )
            return False

        label = f"a2a.question_answered[{recipient}:{task_id}]"

        if not await_delivery:
            # Boot path — supervise the wait instead of blocking startup.
            async def _restore_after_failure(outcome) -> None:
                # Restoring the row is always right. Scheduling a retry is
                # not: when this supervisor was cancelled the feature is
                # tearing down, and `_cancel_owned_background_tasks` has
                # already captured its task list — so a retry started here
                # escapes teardown entirely and can emit an A2A wake after
                # Peers is disabled. The restored WAITING row is what makes
                # the next boot pick this up; that is the whole retry.
                if late:
                    return
                teardown = outcome.status == STATUS_SUPERVISOR_CANCELLED
                await self._restore_pending_question_waiting(
                    task_id,
                    claim=claim,
                    state=state,
                    reply_text=reply_text,
                    withdrawn=withdrawn,
                    recipient=recipient,
                    original_question=original_question,
                    sess_id=sess_id,
                    causation_chain=causation_chain,
                    schedule_retry=schedule_retry and not teardown,
                )

            async def _retire_after_delivery() -> None:
                # As on the inline path: once the asker was woken, a teardown
                # must not leave the wake owed.
                await self._await_question_checkpoint(
                    self._retire_delivered_wake(task_id, claim, late=late)
                )

            try:
                supervise_terminal_delivery(
                    self,
                    handle,
                    label=label,
                    task_name=f"a2a_question_answered_delivery:{task_id}",
                    on_delivered=_retire_after_delivery,
                    on_undelivered=_restore_after_failure,
                )
            except Exception as e:  # noqa: BLE001
                if is_execution_control_error(e):
                    raise
                # Nothing would ever observe this delivery, so report it as
                # unfired and let the caller restore the row synchronously.
                logger.error(
                    "Could not supervise a2a.question_answered delivery for "
                    "task=%s: %s", task_id, e, exc_info=True,
                )
                return False
            return True

        try:
            outcome = await await_terminal_delivery(handle, label=label)
        except asyncio.CancelledError as cancellation:
            if is_execution_control_error(cancellation):
                raise
            # We are being torn down mid-delivery. The caller's `if not fired`
            # restore never runs because this propagates past it, so the
            # obligation is ours: shield the restore so the row does not stay
            # terminal for a wake nobody observed. "Better a possible duplicate
            # than a missed resumption" — the same posture this file already
            # takes for a failed mark_resolved. A late answer is still held on
            # its row, so it has nothing to restore.
            if late:
                raise
            try:
                await self._await_question_checkpoint(
                    self._restore_pending_question_waiting(
                        task_id,
                        claim=claim,
                        state=state,
                        reply_text=reply_text,
                        recipient=recipient,
                        original_question=original_question,
                        sess_id=sess_id,
                        causation_chain=causation_chain,
                        schedule_retry=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "Could not restore pending_a2a_question task=%s to WAITING "
                    "while cancelled mid-delivery: %s", task_id, exc,
                )
            raise

        if not outcome.delivered:
            logger.warning(
                "a2a.question_answered for task=%s recipient=%s was accepted "
                "but never delivered (%s); %s",
                task_id, recipient, outcome.describe(),
                "the late answer stays held on its row for the next "
                "settlement pass" if late else
                "the pending row is restored to WAITING so the asker is "
                "still resumable",
            )
            return False
        # The asker was woken: a teardown arriving now must not leave the
        # wake owed, or it would be delivered again once the claim lapses.
        await self._await_question_checkpoint(
            self._retire_delivered_wake(task_id, claim, late=late)
        )
        return True

    async def _await_question_checkpoint(self, operation) -> None:
        """Join the original native row write even through caller cancellation."""
        outcome = await await_owned_task(create_owned_task(operation))
        raise_owned_outcome(outcome, operation="A2A question checkpoint")

    async def _restore_pending_question_waiting(
        self,
        task_id: str,
        *,
        claim: Optional[_QuestionClaim],
        state: Optional[str] = None,
        reply_text: Optional[str] = None,
        withdrawn: bool = False,
        recipient: Optional[str] = None,
        original_question: Optional[str] = None,
        sess_id: Optional[str] = None,
        causation_chain: Optional[list] = None,
        schedule_retry: bool = True,
    ) -> None:
        """Return a pending question to WAITING so replay can retry wakeup.

        Only under the ``claim`` that took the row out of WAITING, and in
        the store that issued it: a pass that lost its row to another (its
        lease lapsed and the wake was taken over) restores nothing (#3440).
        ``None`` means there is no row to restore."""
        if claim is None:
            return
        store = claim.store
        if not hasattr(store, "mark_waiting_for_retry"):
            return
        try:
            restored = await store.mark_waiting_for_retry(
                task_id,
                claim=claim.token,
                state=state,
                reply_text=reply_text,
            )
        except Exception as exc:  # noqa: BLE001
            if is_execution_control_error(exc):
                raise
            logger.error(
                "Failed to restore pending_a2a_question task=%s to WAITING "
                "after signal enqueue failure: %s",
                task_id, exc, exc_info=True,
            )
            return
        if restored:
            logger.warning(
                "Restored pending_a2a_question task=%s to WAITING after "
                "a2a.question_answered signal enqueue failure.",
                task_id,
            )
            if (
                schedule_retry
                and state
                and recipient is not None
                and original_question is not None
            ):
                self._schedule_question_answered_retry(
                    task_id=task_id,
                    recipient=recipient,
                    original_question=original_question,
                    sess_id=sess_id or "",
                    state=state,
                    reply_text=reply_text or "",
                    causation_chain=causation_chain,
                    withdrawn=withdrawn,
                )

    def _schedule_question_answered_retry(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        state: str,
        reply_text: str,
        causation_chain: Optional[list],
        withdrawn: bool = False,
    ) -> None:
        """Schedule near-term retries for a restored terminal wake payload.

        The retry loop is a FEATURE-owned background task (#2522 P1): it is
        reachable during boot startup-replay (a restored WAITING row whose
        wakeup signal failed to enqueue), so an agent-only task could keep
        retrying — and eventually fire a resumption signal — after boot
        rollback / soft disable tore this feature down. Owning it means
        ``Feature.shutdown()`` cancels it, while ``_track_owned_background_task``
        still registers it in the agent's global reap set for full shutdown.
        """
        coro = self._retry_restored_question_answered_signal(
            task_id=task_id,
            recipient=recipient,
            original_question=original_question,
            sess_id=sess_id,
            state=state,
            reply_text=reply_text,
            causation_chain=causation_chain,
            withdrawn=withdrawn,
        )
        self._track_owned_background_task(
            coro,
            name=f"a2a_question_answered_retry:{recipient}:{task_id}",
        )

    async def _retry_restored_question_answered_signal(
        self,
        *,
        task_id: str,
        recipient: str,
        original_question: str,
        sess_id: str,
        state: str,
        reply_text: str,
        causation_chain: Optional[list],
        withdrawn: bool = False,
    ) -> None:
        """Retry a restored terminal payload before waiting for restart/sweep."""
        import asyncio

        for delay_seconds in (1, 5, 15):
            await asyncio.sleep(delay_seconds)
            store = getattr(self.agent, "pending_a2a_questions", None)
            if store is None:
                return
            claim = self._new_question_claim(store)
            try:
                was_waiting = await self._claim_question_wake(
                    claim, task_id, state=state, reply_text=reply_text,
                )
            except Exception as exc:  # noqa: BLE001
                if is_execution_control_error(exc):
                    raise
                logger.warning(
                    "Failed to claim restored pending_a2a_question task=%s "
                    "for retry: %s",
                    task_id, exc, exc_info=True,
                )
                continue
            if not was_waiting:
                return

            fired = await self._fire_question_answered_signal(
                task_id=task_id,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                state=state,
                reply_text=reply_text,
                causation_chain=causation_chain,
                claim=claim,
                withdrawn=withdrawn,
            )
            if fired:
                return

            await self._restore_pending_question_waiting(
                task_id,
                claim=claim,
                state=state,
                reply_text=reply_text,
                withdrawn=withdrawn,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                causation_chain=causation_chain,
                schedule_retry=False,
            )

    async def _iter_sse_events(self, response):
        """Parse Server-Sent Events from an httpx streaming response.

        Yields ``{event, data}`` dicts per SSE frame (terminated by a
        blank line). httpx exposes ``aiter_lines()`` which already
        strips trailing newlines, so we accumulate ``event:`` and
        ``data:`` field values until the blank-line separator. Comment
        lines (``:`` prefix) are dropped silently."""
        async for event in iter_sse_events(response):
            # Keep the historical test/support method's dict shape while the
            # protocol-neutral parser remains the single implementation.
            yield {"event": event.event, "data": event.data}

    def _parse_sse_status_data(
        self, data_str: str,
    ) -> Optional[Tuple[str, str]]:
        """Parse a ``status`` SSE frame's data field.

        Returns ``(state, reply_text)`` on success or None if the
        frame is malformed / pre-terminal. Reply text is extracted
        from the same three locations the legacy polling code
        checked — ``status.message.parts``, top-level ``message``
        string, and ``artifacts[].parts`` — so the supervisor handles
        both the canonical A2A spec shape and kestrel's flattened
        endpoint shape (#1444 carries the same dual-shape logic
        forward from the legacy polling path)."""
        if not data_str:
            return None
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return self._parse_task_status_payload(data)

    def _parse_task_status_payload(
        self, data: Mapping[str, Any],
    ) -> Optional[Tuple[str, str]]:
        """Extract ``(state, reply_text)`` from one task envelope.

        Shared by the SSE ``status`` frame and a direct task read, which carry
        the same canonical and flattened shapes."""
        raw_status = data.get("status")
        if isinstance(raw_status, dict):
            current_state = raw_status.get("state")
        else:
            current_state = raw_status
        if not isinstance(current_state, str):
            return None
        # A task model dump carries its ``TaskState`` member, a ``str`` enum.
        current_state = str(getattr(current_state, "value", current_state))

        reply_text = ""
        if isinstance(raw_status, dict):
            msg = raw_status.get("message") or {}
            for part in (msg.get("parts") or []):
                if isinstance(part, dict) and "text" in part:
                    reply_text = part["text"] or ""
                    break
        if not reply_text:
            top_msg = data.get("message")
            if isinstance(top_msg, str) and top_msg:
                reply_text = top_msg
        if not reply_text:
            for artifact in (data.get("artifacts") or []):
                if isinstance(artifact, dict):
                    for part in (artifact.get("parts") or []):
                        if isinstance(part, dict) and "text" in part:
                            reply_text = part["text"] or ""
                            break
                if reply_text:
                    break

        return current_state, reply_text

    # ------------------------------------------------------------------
    # Startup-replay + hourly expiry sweep (#1444 step 6)
    #
    # On boot, walk ``pending_a2a_questions WHERE status='WAITING'``:
    #   - past-deadline rows → expire in a feature-owned task: settle with
    #     the recipient, then fire ``a2a.question_answered`` (the
    #     recipient's answer, or state='expired') so the asking lineage
    #     resumes with a clean branch in the prompt template
    #   - within-deadline rows → spawn a fresh subscription supervisor
    #     so the SSE wait survives process restarts
    # and resume receiver settlement for expired rows still unsettled: a
    # recipient not yet confirmed terminal, or a late answer not yet
    # delivered (#3440). A wake an earlier process claimed and never
    # confirmed delivered goes back to WAITING (once its claim lapsed)
    # carrying that wake, and is replayed with them.
    #
    # An hourly background task runs the same expired-row scan so rows
    # whose deadline lapses without a supervisor (e.g. supervisor
    # crashed) still get a synthetic terminal signal, plus the same
    # abandoned-wake recovery and receiver-settlement retry.
    # ------------------------------------------------------------------

    # Hourly cron interval — overridable via constructor for test
    # injection. 3600s is the Sovereign-decided default.
    EXPIRY_SWEEP_INTERVAL_SECONDS = 3600

    @execution_work_operation
    async def post_all_features_loaded(self, agent):
        """Run startup-replay and start the hourly expiry sweep.

        Called once after every feature has initialized — by that
        point the dispatcher and the ``pending_a2a_questions`` store
        are both wired on the agent. Skips silently when either is
        absent (standalone mode, no DB) or when no peer router is
        configured."""
        # Register the ``a2a:`` Waitable provider so an outbound A2A task can
        # be durably watched/re-armed with the CORRECT provider (#2729). This
        # is independent of the pending-question replay below, so it runs even
        # in standalone mode where no peer router is configured — a mismatched
        # ``task:<outbound-id>`` watch is still rejected because the ``a2a``
        # kind is available for the ownership cross-check.
        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            from kestrel_sovereign.features.peers.wait_provider import A2AWaitable

            # Record ownership so base shutdown()/boot rollback unregisters it.
            self._register_wait_provider(registry, A2AWaitable(self), replace=True)

        store = getattr(agent, "pending_a2a_questions", None)
        if store is None:
            logger.debug(
                "Skipping a2a question startup-replay — no "
                "pending_a2a_questions store wired."
            )
            return
        try:
            context = self._peer_directory_context()
        except PeerDirectoryConfigurationError as control_error:
            if is_execution_control_error(control_error):
                raise
            logger.error(
                "Skipping a2a question startup-replay — peer router is "
                "missing trusted requester context."
            )
            return
        if context is None:
            logger.debug(
                "Skipping a2a question startup-replay — no peer router."
            )
            return

        # Restored waits are resident work, not children of the cold scheduler
        # occurrence that happened to boot this runtime. Publish one feature-
        # owned driver under the original runtime's READY barrier; replay's
        # supervisors, deferred signal joins and retries inherit that resident
        # root through the ordinary child tracker. A newly committed outbound
        # question publishes its registered subscription through the same
        # explicit resident contract; arbitrary effect children never do.
        if callable(getattr(type(agent), "_track_runtime_task", None)):
            self._track_owned_runtime_task(
                self._replay_and_sweep_questions(store),
                name="a2a_question_expiry_sweep",
            )
        else:
            # Legacy standalone hosts have no resident/READY contract. Keep
            # their synchronous replay and unchanged inherited child custody.
            await self._replay_questions_safely(store)
            self._track_owned_background_task(
                self._hourly_expiry_sweep_loop(store),
                name="a2a_question_expiry_sweep",
            )

    async def _replay_and_sweep_questions(self, store) -> None:
        """Run restored replay and its backstop as one resident service."""
        await self._replay_questions_safely(store)
        await self._hourly_expiry_sweep_loop(store)

    async def _replay_questions_safely(self, store) -> None:
        """Keep ordinary replay failures recoverable without hiding control."""
        try:
            await self._replay_pending_a2a_questions(store)
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.warning(
                "a2a question startup-replay failed: %s. The hourly "
                "sweep is still the backstop — operators can still "
                "resume in-flight questions.",
                e, exc_info=True,
            )

        # Settlement makes network calls to recipients, so it never runs
        # inline on the boot path. FEATURE-owned for the same teardown
        # reason as the sweep below.
        self._track_owned_background_task(
            self._settle_expired_questions_with_receivers(store),
            name="a2a_question_receiver_settlement:replay",
        )

        # A wake the previous process claimed and never confirmed delivered
        # is usually still leased to it at boot, so replay could not take it
        # back. Every such lease lapses within one lease of this boot:
        # deliver those wakes then, not at the first hourly tick (#3440).
        self._track_owned_background_task(
            self._redeliver_abandoned_wakes_after_boot(store),
            name="a2a_question_wake_recovery:replay",
        )

        # The caller owns exactly one sweep: the resident service continues
        # into it; the legacy host explicitly tracks it after replay.

    @execution_work_operation
    async def _replay_pending_a2a_questions(self, store) -> None:
        """Walk every WAITING row at boot. Past-deadline rows are expired
        in a feature-owned task; within-deadline rows get a fresh
        subscription supervisor. The chain is NOT persisted across
        restarts — a restart erases the asking turn's context, so the
        resumed signal carries an empty chain and the dispatcher applies
        its normal depth-bounded cycle check from scratch."""
        from datetime import datetime, timezone

        # A wake an earlier process claimed and never confirmed delivered is
        # owed again once its claim lapsed: back to WAITING, and replayed
        # below with the rest (#3440). One whose claim still holds may still
        # be in flight; it is taken back a lease after boot, see
        # :meth:`_redeliver_abandoned_wakes_after_boot`.
        try:
            await store.reclaim_abandoned_wakes()
        except Exception as e:  # noqa: BLE001 - the hourly sweep retries it
            if is_execution_control_error(e):
                raise
            logger.warning(
                "a2a question startup-replay: could not reclaim abandoned "
                "wakes: %s. The hourly sweep retries.", e,
            )

        waiting = await store.list_waiting()
        if not waiting:
            logger.debug("a2a question startup-replay: no WAITING rows.")
            return

        now = datetime.now(timezone.utc)
        replayed = 0
        expired = 0
        for row in waiting:
            # Boot path: these run inline inside feature startup, so they must
            # NOT block on a cognition turn that cannot run until the agent
            # finishes initializing. Delivery is supervised instead (#2532).
            if getattr(row, "retry_state", None):
                await self._handle_retry_payload_row(
                    store, row, await_delivery=False,
                )
                replayed += 1
                continue
            try:
                deadline = datetime.fromisoformat(row.deadline)
            except (TypeError, ValueError):
                # Unparseable deadline → safer to expire than to spawn
                # a supervisor that might run forever.
                logger.warning(
                    "a2a startup-replay: unparseable deadline %r for "
                    "task=%s — treating as expired.",
                    row.deadline, row.task_id,
                )
                self._expire_replayed_row(store, row)
                expired += 1
                continue
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                self._expire_replayed_row(store, row)
                expired += 1
                continue
            # Within deadline — spawn supervisor. Chain is not
            # persisted across restarts; the resumed signal fires with
            # an empty chain and the dispatcher applies its normal
            # depth-bounded cycle check from scratch. FEATURE-owned (like
            # the hourly sweep) so boot rollback / soft disable cancel it
            # via ``Feature.shutdown()`` — an agent-only task would keep
            # supervising (and could still fire a resumption signal) after
            # this feature is torn down (kestrel-sovereign#2522 P1).
            self._track_owned_background_task(
                self._supervise_a2a_question(
                    task_id=row.task_id,
                    recipient=row.recipient,
                    recipient_agent_id=getattr(row, "recipient_agent_id", None),
                    original_question=row.original_question,
                    sess_id=row.origin_session_id or "",
                    deadline_utc=deadline,
                    causation_chain=None,
                ),
                name=(
                    f"a2a_question_supervisor:replay:"
                    f"{row.recipient}:{row.task_id}"
                ),
            )
            replayed += 1

        logger.info(
            "a2a question startup-replay: replayed=%d expired=%d "
            "total_waiting=%d",
            replayed, expired, len(waiting),
        )

    def _expire_replayed_row(self, store, row) -> None:
        """Expire a past-deadline row found at boot, off the boot path.

        Expiry settles with the recipient over the network and then waits for
        the resumed turn, neither of which boot can block on (#2532, #3440).
        FEATURE-owned so boot rollback / soft disable cancel it; the row stays
        WAITING until its claim, so a cancelled expiry is replayed next boot.
        """
        self._track_owned_background_task(
            self._handle_expired_row(store, row),
            name=f"a2a_question_expiry:replay:{row.recipient}:{row.task_id}",
        )

    @execution_work_operation
    async def _hourly_expiry_sweep_loop(self, store) -> None:
        """Sweep ``list_waiting_past_deadline`` every hour. Each row is
        expired via :meth:`_handle_expired_row` so the asking lineage
        resumes with a clean branch in the prompt template, wakes whose
        claim lapsed undelivered are delivered again first, and expired
        rows whose recipient is not yet settled are retried. Logs and
        continues on transient failures — this loop is the deadline
        backstop and must not die silently."""
        import asyncio

        while True:
            try:
                await asyncio.sleep(self.EXPIRY_SWEEP_INTERVAL_SECONDS)
                try:
                    await self._redeliver_abandoned_wakes(store)
                except Exception as e:
                    if is_execution_control_error(e):
                        raise
                    logger.warning(
                        "a2a question wake recovery failed: %s", e,
                    )
                # Guarded like the steps around it, so a store that cannot
                # list this tick's expiries still lets settlement run.
                try:
                    expired = await store.list_waiting_past_deadline()
                except Exception as e:
                    if is_execution_control_error(e):
                        raise
                    logger.warning(
                        "a2a expiry sweep: could not list past-deadline "
                        "questions: %s", e,
                    )
                    expired = []
                for row in expired:
                    try:
                        await self._handle_expired_row(store, row)
                    except Exception as e:
                        if is_execution_control_error(e):
                            raise
                        logger.warning(
                            "a2a expiry sweep: failed to expire row "
                            "task=%s: %s",
                            row.task_id, e,
                        )
                if expired:
                    logger.info(
                        "a2a expiry sweep: expired=%d", len(expired),
                    )
                # After this tick's expiries, so they never wait on this
                # tick's late-answer turns or stop for a settlement failure.
                # (Like an expiry's own wake, a relayed late answer is awaited
                # inline, so the next tick starts after it.) A row expired
                # just now was settled inline and is leased to its expiry.
                try:
                    await self._settle_expired_questions_with_receivers(store)
                except Exception as e:
                    if is_execution_control_error(e):
                        raise
                    logger.warning(
                        "a2a receiver settlement pass failed: %s", e,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(
                    "a2a expiry sweep iteration failed: %s. "
                    "Continuing.", e,
                )
                # Don't tight-loop on persistent failure — back off
                # then resume the normal cadence.
                await asyncio.sleep(60)

    async def _redeliver_abandoned_wakes(
        self, store, *, now: Optional[datetime] = None,
    ) -> None:
        """Deliver every wake whose claim lapsed before it was confirmed
        delivered (#3440).

        Its claiming pass died, or lost the row, between taking the row out
        of WAITING and confirming the wake: a supervisor's answer, an
        expiry's ``expired`` or answer, or a retried wake. Each goes back to
        WAITING carrying that wake and is delivered under a fresh claim, so a
        crash costs the asker a delay (at most a duplicate), never the wake.
        ``now`` is the pass's clock for the lapse."""
        import asyncio

        for row in await store.reclaim_abandoned_wakes(now=now):
            try:
                await self._handle_retry_payload_row(store, row)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if is_execution_control_error(e):
                    raise
                logger.warning(
                    "a2a question wake recovery: failed for task=%s: %s",
                    row.task_id, e,
                )

    async def _redeliver_abandoned_wakes_after_boot(self, store) -> None:
        """Deliver the wakes the previous process left claimed, once their
        claims have lapsed (#3440).

        That process renewed its claims no later than this boot, so every one
        of them has lapsed a lease after it. FEATURE-owned, so teardown
        cancels the wait; the hourly sweep remains the backstop."""
        import asyncio

        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            QUESTION_CLAIM_LEASE,
        )

        await asyncio.sleep(QUESTION_CLAIM_LEASE.total_seconds())
        try:
            await self._redeliver_abandoned_wakes(store)
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logger.warning(
                "a2a question wake recovery after boot failed: %s. The "
                "hourly sweep retries.", e,
            )

    async def _handle_expired_row(self, store, row) -> None:
        """Expire a single past-deadline WAITING row via
        :meth:`_expire_pending_question`. Idempotent: if the row was
        already terminal (raced the supervisor), drop silently instead of
        double-firing.

        Both callers are feature-owned background tasks (the hourly sweep and
        a boot-replay expiry), so terminal delivery is awaited inline (#2532).
        A row carrying an already-observed terminal payload is re-fired as
        captured instead."""
        if getattr(row, "retry_state", None):
            await self._handle_retry_payload_row(store, row)
            return

        await self._expire_pending_question(
            store,
            task_id=row.task_id,
            recipient=row.recipient,
            recipient_agent_id=getattr(row, "recipient_agent_id", None),
            original_question=row.original_question,
            sess_id=row.origin_session_id or "",
            causation_chain=None,
        )

    async def _handle_retry_payload_row(
        self, store, row, *, await_delivery: bool = True,
    ) -> None:
        """Re-fire a previously observed terminal answer after enqueue failure.

        The row is claimed again first, so the wake is owed on it until its
        delivery is confirmed, exactly as on its first attempt. An ``expired``
        wake keeps the receiver settlement the row already recorded, and says
        the question was withdrawn only when that settlement is confirmed:
        settlement cannot move while the wake is owed, so it is the one the
        first attempt recorded.

        ``await_delivery`` carries the #2532 ownership boundary through from
        the caller — see :meth:`_fire_question_answered_signal`."""
        from kestrel_sovereign.storage.async_pending_a2a_question_store import (
            RECEIVER_SETTLEMENT_CONFIRMED,
        )

        retry_state = getattr(row, "retry_state", None)
        if not retry_state:
            return
        reply_text = getattr(row, "retry_reply_text", None) or ""
        withdrawn = retry_state == "expired" and (
            getattr(row, "receiver_settlement", None)
            == RECEIVER_SETTLEMENT_CONFIRMED
        )

        claim = self._new_question_claim(store)
        if not await self._claim_question_wake(
            claim, row.task_id, state=retry_state, reply_text=reply_text,
        ):
            return

        fired = await self._fire_question_answered_signal(
            task_id=row.task_id,
            recipient=row.recipient,
            original_question=row.original_question,
            sess_id=row.origin_session_id or "",
            state=retry_state,
            reply_text=reply_text,
            causation_chain=None,
            await_delivery=await_delivery,
            claim=claim,
            withdrawn=withdrawn,
        )
        if not fired:
            await self._restore_pending_question_waiting(
                row.task_id,
                claim=claim,
                state=retry_state,
                reply_text=reply_text,
                withdrawn=withdrawn,
                recipient=row.recipient,
                original_question=row.original_question,
                sess_id=row.origin_session_id or "",
                causation_chain=None,
            )

    @tool(
        name="send_a2a_task",
        description=(
            "Delegate tracked work to a peer using native A2A. The dispatch_id is your "
            "immutable local handle; remote_task_id is assigned by the peer and can be null. "
            "Supply standard message parts for input attachments, such as {'text':'plan', "
            "'filename':'plan.txt'} or {'data':{'ref_type':'memory','id':'...'}}. "
            "Artifacts are peer outputs. skill_id is an optional advisory hint from the "
            "peer's signed Agent Card; it grants no direct tool invocation. "
            "Use context_id to continue a conversation context."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a send",
    )
    @execution_work_operation
    async def send_a2a_task(
        self,
        recipient: str,
        message: str,
        skill_id: str = "",
        context_id: str = "",
        parts: List[Any] = None,
    ) -> ToolResult:
        """Delegate work using standard input Parts and an optional advisory skill."""
        receipt = await self._dispatch_native_peer(
            recipient,
            message,
            context_id=context_id,
            parts=parts,
            skill_id=skill_id,
            verb="task",
            dispatch_tool="send_a2a_task",
        )
        if isinstance(receipt, ToolResult):
            return receipt
        return self._native_dispatch_result(receipt, recipient)
