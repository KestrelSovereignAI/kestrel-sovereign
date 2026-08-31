"""
Peers Feature — Inter-agent communication for multi_agent environments.

Allows agents to discover sibling agents and send messages to them
through the multi_agent host proxy. Works in both local and cloud
deployments. Two transports:

* ``ask_agent`` — synchronous Q&A via ``/api/agent/invoke``. Legacy
  surface kept until Epic #1367 reroutes it onto the A2A path as
  ``send_a2a_question``.

* ``send_a2a_task`` — asynchronous A2A task submission via
  ``/api/agent/tasks/send``. Persists to the recipient's TaskStore,
  fires the ``a2a.task_submitted`` signal so the recipient wakes,
  carries causation chain for cycle detection.

The legacy Mesh Protocol (send_mesh_message / mesh_inbox / receive_mesh_message
+ /agent/mesh endpoint + features/peers/mesh.py) was retired in #1367.
All Falconer workflow events (assign, review_needed, complete, etc.)
now go through send_a2a_task with ``metadata["skill"]`` set to the
corresponding ``workflow.*`` skill id.
"""

import json
import logging
import os
from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.a2a.transport_auth import ensure_a2a_transport_key
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

    async def initialize(self):
        self._host_url = _discover_host_url()
        self._transport_key = ensure_a2a_transport_key()
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

        # #1576: every outbound A2A dispatch writes a sender-side audit
        # row. The receiver-side ``a2a_tasks`` row tells us what the
        # peer saw; the outbound row tells US what we sent, when, to
        # whom, via which tool, and (after a later
        # ``get_peer_task_result`` fetch) what state it settled in.
        # Without this, the sender has no introspection surface for
        # "what did I dispatch and to whom?" beyond per-task_id round
        # trips.
        from kestrel_sovereign.features.storage_access import (
            resolve_feature_database,
        )
        from kestrel_sovereign.a2a.outbound_store import (
            ensure_a2a_outbound_tasks_table,
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
        db_path = os.environ.get("KESTREL_DB_PATH", "")
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
            transport_key=getattr(self, "_transport_key", ""),
            client_factory=lambda *args, **kwargs: httpx.AsyncClient(
                *args, **kwargs,
            ),
            local_cancel=getattr(self, "_local_host_cancel", None),
            local_get=getattr(self, "_local_host_get", None),
            local_subscribe=getattr(self, "_local_host_subscribe", None),
            principal_payload_factory=self._build_principal_action_payload,
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
        local_get=None,
        local_subscribe=None,
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
        self._local_host_get = local_get
        self._local_host_subscribe = local_subscribe
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
                OutboundTaskRouteAmbiguousError,
                ROUTE_STATE_ROUTABLE,
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
        name="list_peers",
        description="List all available peer agents in the multi_agent.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peers"
    )
    async def list_peers(self) -> ToolResult:
        """
        Discover available peer agents via the scoped peer directory.
        Returns their names, status, and capabilities.
        """
        try:
            context = self._peer_directory_context()
        except PeerDirectoryConfigurationError as exc:
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
        except PeerAccessDeniedError:
            # Do not distinguish a denied scope from an empty/unknown peer
            # directory.  In hosted mode either distinction can be used to
            # probe another tenant's namespace.
            return ToolResult.failed(
                "Could not list peers in the current authorization scope",
                data={"peers": [], "error": "Peer directory unavailable"},
            )
        except PeerTransportError:
            return ToolResult.failed(
                "Could not connect to multi_agent host",
                data={"peers": [], "error": "Could not connect to multi_agent host"},
            )
        except PeerDirectoryError as exc:
            logger.error("Failed to list peers: %s", exc)
            return ToolResult.failed(
                "Could not list peers",
                data={"peers": [], "error": "Could not list peers"},
            )
        except Exception:  # noqa: BLE001 - provider extension boundary
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
        description="Send a message to another agent in the multi_agent and get their response. Use this to collaborate, ask questions, or delegate tasks to peer agents.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!ask"
    )
    async def ask_agent(self, agent_name: str, message: str) -> ToolResult:
        """
        Send a message to a peer agent and return their response.

        Args:
            agent_name: Name of the agent to message (e.g. "emma", "claw")
            message: The message or question to send
        """
        try:
            router, requester, peer = await self._resolve_automatic_peer(
                agent_name,
            )
        except PeerDirectoryConfigurationError:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"response": None, "agent": agent_name},
            )
        except PeerSelfTargetError:
            return ToolResult.failed(
                "Cannot send a message to yourself",
                data={"response": None, "agent": agent_name},
            )
        except PeerAccessDeniedError:
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"response": None, "agent": agent_name},
            )
        except PeerNotFoundError:
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"response": None, "agent": agent_name},
            )
        except PeerTransportError:
            return ToolResult.failed(
                f"Could not reach agent '{agent_name}' — multi_agent host unreachable",
                data={"response": None, "agent": agent_name},
            )
        except PeerDirectoryError as exc:
            logger.error("Could not resolve peer %r: %s", agent_name, exc)
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"response": None, "agent": agent_name},
            )

        try:
            data = await router.invoke(requester, peer, message)
        except PeerNotFoundError:
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"response": None, "agent": agent_name},
            )
        except PeerAccessDeniedError:
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"response": None, "agent": agent_name},
            )
        except PeerUnavailableError:
            return ToolResult.failed(
                f"Agent '{agent_name}' is offline",
                data={"response": None, "agent": agent_name},
            )
        except PeerTransportError:
            return ToolResult.failed(
                f"Could not reach agent '{agent_name}' — multi_agent host unreachable",
                data={"response": None, "agent": agent_name},
            )
        except PeerDirectoryError as exc:
            logger.error("Failed to message peer %r: %s", agent_name, exc)
            return ToolResult.failed(
                f"Could not message agent '{agent_name}'",
                data={"response": None, "agent": agent_name},
            )
        except Exception:  # noqa: BLE001 - provider extension boundary
            logger.exception("Peer router raised unexpectedly while invoking %r", agent_name)
            return ToolResult.failed(
                f"Could not message agent '{agent_name}'",
                data={"response": None, "agent": agent_name},
            )

        if not isinstance(data, Mapping):
            logger.warning("Peer router returned a non-object invoke result")
            return ToolResult.failed(
                f"Could not message agent '{agent_name}'",
                data={"response": None, "agent": agent_name},
            )

        response_text = data.get("response", data.get("output", str(data)))
        return ToolResult.ok(
            confirmation=f"Got response from {agent_name}",
            data={"agent": agent_name, "response": response_text},
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
        except PeerDirectoryConfigurationError:
            return None, None, None, ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"sent": False, "recipient": recipient},
            )
        except PeerSelfTargetError:
            return None, None, None, ToolResult.failed(
                "Cannot send an A2A task to yourself",
                data={"sent": False, "recipient": recipient},
            )
        except PeerAccessDeniedError:
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient},
            )
        except PeerNotFoundError:
            # Use the same response for absent, cross-scope, and ambiguous
            # names so the automatic shortcut is not a namespace oracle.
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient},
            )
        except PeerTransportError:
            return None, None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}' — multi_agent host unreachable",
                data={"sent": False, "recipient": recipient},
            )
        except PeerDirectoryError as exc:
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
        except (PeerNotFoundError, PeerAccessDeniedError):
            await _mark_dispatch_failed("peer_not_in_automatic_directory")
            return None, None, None, ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerUnavailableError:
            await _mark_dispatch_failed(f"peer_unavailable:{recipient}")
            return None, None, None, ToolResult.failed(
                f"Agent '{recipient}' is offline or TaskManager unavailable",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerTransportError:
            await _mark_dispatch_failed(f"connect_error:{recipient}")
            return None, None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except PeerDirectoryError as exc:
            logger.error("A2A send to %r failed: %s", recipient, exc)
            await _mark_dispatch_failed(f"peer_router_error:{type(exc).__name__}")
            return None, None, None, ToolResult.failed(
                f"Could not send A2A task to '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
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
    async def send_a2a_message(
        self,
        recipient: str,
        message: str,
        session_id: str = "",
    ) -> ToolResult:
        """
        Send an async fire-and-forget A2A message. The recipient's
        cognition loop wakes (a2a.task_submitted), they see the
        message, they decide whether to act — but the caller doesn't
        wait or track. Same wire as send_a2a_task but no skill_id is
        attached (signals "informational, not work assignment").
        """
        task_data, _chain, _recipient_agent_id, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id="", session_id=session_id,
            extra_metadata={"a2a_verb": "message"},
            dispatch_tool="send_a2a_message",
        )
        if err is not None:
            return err
        return ToolResult.ok(
            confirmation=(
                f"A2A message sent to {recipient} "
                f"(task_id={task_data['id']}). Recipient has been signaled."
            ),
            data={
                "sent": True,
                "task_id": task_data["id"],
                "session_id": task_data["sessionId"],
                "recipient": recipient,
            },
        )

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
            "respond there. Do NOT block your turn waiting for the "
            "answer; the supervisor will wake you. For fire-and-forget "
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

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The question / prompt.
            session_id: Optional A2A session id.
            timeout_seconds: Wall-clock cap on the supervisor. The
                hourly expiry sweep marks any WAITING row past this
                deadline as EXPIRED and fires a synthetic
                ``a2a.question_answered`` signal with
                ``state='expired'`` so the asking lineage still
                resumes cleanly. Default 300s.
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

        # Spawn the supervisor as a FEATURE-owned background task. It runs the
        # SSE loop, fires the a2a.question_answered signal on terminal frame,
        # and exits. Still agent-tracked (auto-cancelled at full agent shutdown
        # by ``_shutdown_background_tasks``), but also owned by this feature so
        # runtime disable / boot rollback / soft disable cancel it via
        # ``Feature.shutdown()`` — an agent-only task would keep supervising
        # (and could still fire a resumption signal) after this feature is torn
        # down (kestrel-sovereign#2522 P1). Same ownership as the startup-replay
        # supervisor and the hourly expiry sweep.
        self._track_owned_background_task(
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
            "Fetch the current state + full reply text of an A2A "
            "task you previously sent to a peer agent. Use this when "
            "an `a2a.question_answered` signal arrived with "
            "`truncated=true` (the inline reply was clipped at 8 "
            "KiB) — this tool fetches the FULL untruncated body from "
            "the peer's task store. Returns the same envelope shape "
            "a local `get_task_result` would, but routed through the "
            "host proxy to the peer (#1444 truncation recovery path)."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a result",
    )
    async def get_peer_task_result(
        self,
        recipient: str,
        task_id: str,
    ) -> ToolResult:
        """Fetch a peer's task envelope and return the full reply
        text. Mirrors ``get_task_result`` but for tasks the caller
        SENT to a peer (not tasks in the caller's own store).

        Args:
            recipient: The peer agent name the task was sent to.
            task_id: The task id returned from
                ``send_a2a_question`` / ``send_a2a_task``.
        """
        try:
            retained_agent_id = await self._outbound_recipient_agent_id(task_id)
            router, requester, peer = await self._resolve_retained_automatic_peer(
                recipient, retained_agent_id,
            )
        except PeerDirectoryConfigurationError:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host "
                "to proxy through",
                data={"recipient": recipient, "task_id": task_id},
            )
        except (PeerNotFoundError, PeerAccessDeniedError):
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"recipient": recipient, "task_id": task_id},
            )
        except PeerSelfTargetError:
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"recipient": recipient, "task_id": task_id},
            )
        except PeerTransportError:
            return ToolResult.failed(
                f"Could not reach peer '{recipient}' for task {task_id}",
                data={"recipient": recipient, "task_id": task_id},
            )
        except PeerDirectoryError as exc:
            logger.error(
                "Could not resolve peer task recipient %r: %s", recipient, exc,
            )
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"recipient": recipient, "task_id": task_id},
            )

        try:
            data = await router.get_a2a_task(requester, peer, task_id)
        except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError):
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"recipient": recipient, "task_id": task_id},
            )
        except PeerTransportError:
            return ToolResult.failed(
                f"Could not reach peer '{recipient}' for task {task_id}",
                data={"recipient": recipient, "task_id": task_id},
            )
        except PeerDirectoryError as exc:
            logger.error(
                "Error fetching peer task %s from %r: %s",
                task_id, recipient, exc,
            )
            return ToolResult.failed(
                f"Error fetching peer task {task_id} from {recipient}",
                data={"recipient": recipient, "task_id": task_id},
            )
        except Exception:  # noqa: BLE001 - provider extension boundary
            logger.exception(
                "Peer router raised unexpectedly fetching task %s from %r",
                task_id, recipient,
            )
            return ToolResult.failed(
                f"Error fetching peer task {task_id} from {recipient}",
                data={"recipient": recipient, "task_id": task_id},
            )

        if not isinstance(data, Mapping):
            return ToolResult.failed(
                f"Peer '{recipient}' returned an invalid task result",
                data={"recipient": recipient, "task_id": task_id},
            )

        # Reuse the supervisor's dual-shape parser to extract the
        # reply text — handles both canonical A2A and kestrel's
        # flattened endpoint shape consistently with how the
        # ``a2a.question_answered`` signal got built in the first
        # place.
        raw_status = data.get("status")
        if isinstance(raw_status, dict):
            current_state = raw_status.get("state", "unknown")
        elif isinstance(raw_status, str):
            current_state = raw_status
        else:
            current_state = "unknown"
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
        # Group artifacts by ``name`` first, then reassemble each
        # group in INDEX order. The A2A artifact model allows a task
        # to carry multiple unrelated artifact groups simultaneously
        # (e.g. ``reply_body`` chunks + a separate ``debug_log``);
        # concatenating ALL text parts globally would interleave
        # unrelated groups and pollute the answer. The chunking
        # convention in the receiver-side
        # ``attach_artifact_to_a2a_task`` tool documents
        # ``reply_body`` as the canonical group name for chunked Q&A
        # replies — we look there first, falling back to whatever
        # single group exists. Codex round 2 P2 on the artifact PR.
        artifacts_raw = data.get("artifacts") or []
        terminal_states = ("completed", "failed", "canceled")
        groups: dict[str, list[dict]] = {}
        for art in artifacts_raw:
            if not isinstance(art, dict):
                continue
            group_name = art.get("name") or ""
            groups.setdefault(group_name, []).append(art)

        artifact_bodies: dict[str, str] = {}
        artifact_group_complete: dict[str, bool] = {}
        for group_name, group_arts in groups.items():
            arts_sorted = sorted(
                group_arts,
                key=lambda a: (
                    a.get("index")
                    if isinstance(a.get("index"), int)
                    else 0
                ),
            )
            body = "".join(
                part["text"] or ""
                for art in arts_sorted
                for part in (art.get("parts") or [])
                if isinstance(part, dict) and "text" in part
            )
            last_chunk_seen = any(
                a.get("lastChunk") is True for a in arts_sorted
            )
            complete = (
                current_state in terminal_states or last_chunk_seen
            )
            artifact_bodies[group_name] = body
            artifact_group_complete[group_name] = complete

        # Primary body: ``reply_body`` is the documented convention;
        # fall back to whichever single group exists (preserves
        # backwards-compat with legacy senders that don't follow the
        # naming convention) or empty.
        if "reply_body" in artifact_bodies:
            primary_name = "reply_body"
        elif len(artifact_bodies) == 1:
            primary_name = next(iter(artifact_bodies))
        else:
            primary_name = None
        artifact_body = (
            artifact_bodies.get(primary_name, "") if primary_name else ""
        )
        # If the inline reply was empty but the primary artifact group
        # carries text, the asking lineage's answer IS the artifact
        # body — surface it as reply_text so the resumed turn doesn't
        # have to special-case the chunked path.
        if not reply_text and artifact_body:
            reply_text = artifact_body

        # Completeness:
        #   - No artifacts → inline message IS the body → complete.
        #   - Primary group has its completeness flag (terminal state
        #     OR lastChunk=True).
        #   - No primary group identifiable (multiple unnamed groups,
        #     none labeled ``reply_body``) → fall back to overall
        #     completeness: complete iff EVERY group is complete OR
        #     task is terminal.
        if not artifact_bodies:
            artifact_body_complete = True
        elif primary_name is not None:
            artifact_body_complete = artifact_group_complete[primary_name]
        else:
            artifact_body_complete = current_state in terminal_states or all(
                artifact_group_complete.values()
            )

        artifact_segment_count = sum(len(g) for g in groups.values())

        # #1576: close the loop on the sender-side outbound row. When
        # the peer reports a terminal state, stamp it on our local
        # audit row so a later ``list_outbound_a2a_tasks`` shows
        # ``terminal_state`` populated. Non-terminal interim states
        # are intentionally NOT stamped — the row stays NULL until a
        # terminal fetch lands, matching Emma's pinned acceptance
        # ("terminal/error state when known").
        _audit_db = getattr(self, "_db", None)
        if (
            _audit_db is not None
            and current_state in terminal_states
        ):
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
                    terminal_state=current_state,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "outbound_store: terminal stamp failed for %s: %s",
                    task_id, exc,
                )

        return ToolResult.ok(
            confirmation=(
                f"Fetched peer task {task_id[:8]} from {recipient} "
                f"(state={current_state}, {len(reply_text)} chars, "
                f"{artifact_segment_count} artifact segment(s) across "
                f"{len(groups)} group(s), complete={artifact_body_complete})"
            ),
            data={
                "recipient": recipient,
                "task_id": task_id,
                "state": current_state,
                "reply_text": reply_text,
                "artifacts": artifacts_raw,
                # The primary group body (the documented ``reply_body``
                # convention) reassembled in index order.
                "artifact_body": artifact_body,
                "artifact_body_complete": artifact_body_complete,
                "artifact_segment_count": artifact_segment_count,
                # All groups, keyed by name, for callers that need to
                # inspect non-reply artifacts (logs, side-channel
                # results, etc.).
                "artifact_bodies": artifact_bodies,
                "artifact_group_complete": artifact_group_complete,
            },
        )

    async def cancel_outbound_task(
        self,
        task_id: str,
        reason: Optional[str] = None,
        *,
        local_recipient_match: bool = False,
    ) -> Optional[ToolResult]:
        """Route a creator's cancellation to the durable task recipient.

        This is intentionally not a second public tool. ``TaskFeature`` owns
        the user-facing cancellation surface and checks this seam before its
        local TaskStore because a shared PostgreSQL backend also exposes the
        recipient's row to the sender. ``None`` means the exact sender-owned
        outbound route is absent; every unreadable or unsafe route returns a
        fail-closed result. The durable route supplies the recipient; neither
        a display name nor possession of the task id is routing authority.
        """
        db = getattr(self, "_db", None)
        if db is None:
            if local_recipient_match:
                return None
            return ToolResult.failed(
                "The durable outbound route store is unavailable",
                data={"task_id": task_id},
            )
        if not getattr(self, "_outbound_route_store_ready", False):
            return ToolResult.failed(
                "The durable outbound route store is unavailable",
                data={"task_id": task_id},
            )

        try:
            from kestrel_sovereign.a2a.outbound_store import (
                ROUTE_STATE_ROUTABLE,
                get_outbound_task,
            )

            audit_agent_id = str(
                getattr(self.agent, "did", None) or self._own_name
            )
            outbound = await get_outbound_task(
                db,
                agent_id=audit_agent_id,
                task_id=task_id,
            )
            if outbound is None:
                if local_recipient_match:
                    return None
                return ToolResult.failed(
                    "No unambiguous local or outbound task route exists",
                    data={"task_id": task_id},
                )
            if local_recipient_match:
                return ToolResult.failed(
                    "Task ID is ambiguous between inbound and outbound work",
                    data={"task_id": task_id, "error_type": "ambiguous_direction"},
                )
            recipient_agent_id = outbound.recipient_agent_id
            if (
                outbound.route_state != ROUTE_STATE_ROUTABLE
                or not isinstance(recipient_agent_id, str)
                or not recipient_agent_id
            ):
                raise PeerNotFoundError(
                    "No durable stable identity exists for this peer task"
                )
            router, requester, peer = await self._resolve_retained_automatic_peer(
                outbound.recipient,
                recipient_agent_id,
            )
        except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError):
            return ToolResult.failed(
                "Peer is not available in the automatic directory",
                data={"task_id": task_id},
            )
        except PeerDirectoryConfigurationError:
            return ToolResult.failed(
                "Peer routing is not configured safely",
                data={"task_id": task_id},
            )
        except PeerDirectoryError:
            return ToolResult.failed(
                "Could not resolve the durable outbound task recipient",
                data={"task_id": task_id},
            )
        except Exception:  # noqa: BLE001 - durable route backend boundary
            logger.exception(
                "Could not read durable outbound route for task %s", task_id
            )
            return ToolResult.failed(
                "Could not read the durable outbound task route",
                data={"task_id": task_id},
            )

        reason_text = reason or "Task canceled by creator"
        session_id = f"a2a-cancel:{task_id}"
        payload: Dict[str, Any] = {
            "reason": reason_text,
            "sessionId": session_id,
            "metadata": {
                "sender": self._current_legacy_outbound_sender(),
                "a2a_verb": "cancel_task",
            },
        }
        try:
            self._maybe_sign_outbound(
                payload,
                task_id=task_id,
                sess_id=session_id,
                message=reason_text,
            )
            response = await router.cancel_a2a_task(
                requester,
                peer,
                task_id,
                payload,
            )
        except OutboundSigningError as exc:
            return ToolResult.failed(
                "Could not authenticate task cancellation",
                data={"task_id": task_id, "error_type": exc.code},
            )
        except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError):
            return ToolResult.failed(
                "Task cancellation is not authorized",
                data={"task_id": task_id},
            )
        except PeerTaskConflictError:
            return ToolResult.failed(
                "Task cancellation conflicts with the recipient's terminal state",
                data={"task_id": task_id, "error_type": "lifecycle_conflict"},
            )
        except PeerTransportError:
            return ToolResult.failed(
                "Could not reach the task recipient",
                data={"task_id": task_id},
            )
        except PeerDirectoryError:
            return ToolResult.failed(
                "Peer task cancellation failed",
                data={"task_id": task_id},
            )
        except Exception:  # noqa: BLE001 - provider extension boundary
            logger.exception(
                "Peer router raised unexpectedly canceling task %s", task_id
            )
            return ToolResult.failed(
                "Peer task cancellation failed",
                data={"task_id": task_id},
            )
        if not isinstance(response, Mapping):
            return ToolResult.failed(
                "Peer returned an invalid cancellation receipt",
                data={"task_id": task_id},
            )
        receipt = response.get("cancellation_receipt")
        if (
            response.get("id") != task_id
            or response.get("status") != "canceled"
            or not isinstance(receipt, Mapping)
            or not isinstance(receipt.get("status_before"), str)
        ):
            return ToolResult.failed(
                "Peer returned an invalid cancellation receipt",
                data={"task_id": task_id},
            )
        durable_reason = receipt.get("reason", reason_text)

        try:
            from kestrel_sovereign.a2a.outbound_store import (
                update_outbound_terminal_state,
            )

            stamped = await update_outbound_terminal_state(
                db,
                agent_id=audit_agent_id,
                task_id=task_id,
                terminal_state="canceled",
            )
        except Exception:  # noqa: BLE001 - audit backend boundary
            logger.exception(
                "Cancellation succeeded but its outbound audit stamp failed: %s",
                task_id,
            )
            stamped = 0
        audit_confirmed = stamped == 1
        if not audit_confirmed:
            # A question's terminal SSE can win the race and stamp this exact
            # sender-owned row before the cancellation response returns. A
            # zero-row CAS (or uncertain post-commit backend exception) is
            # successful only when a fresh read proves the desired state.
            try:
                refreshed = await get_outbound_task(
                    db,
                    agent_id=audit_agent_id,
                    task_id=task_id,
                )
            except Exception:  # noqa: BLE001 - audit reconciliation boundary
                logger.exception(
                    "Could not reconcile outbound cancellation audit row: %s",
                    task_id,
                )
                refreshed = None
            audit_confirmed = (
                refreshed is not None
                and refreshed.terminal_state == "canceled"
            )
        if not audit_confirmed:
            return ToolResult.partial(
                confirmation=f"Cancelled outbound task {task_id[:8]}",
                error="Cancellation succeeded but its outbound audit row was not updated",
                data={
                    "task_id": task_id,
                    "status": "canceled",
                    "status_before": receipt["status_before"],
                    "reason": durable_reason,
                },
            )
        return ToolResult.ok(
            confirmation=f"Cancelled outbound task {task_id[:8]}",
            data={
                "task_id": task_id,
                "status": "canceled",
                "status_before": receipt["status_before"],
                "reason": durable_reason,
            },
        )

    @tool(
        name="list_outbound_a2a_tasks",
        description=(
            "List the A2A tasks you SENT to peer agents — your local "
            "audit log of outbound dispatches (#1576). Each row carries "
            "task_id, recipient, verb (message/question/task), "
            "dispatch_tool, created_at, and terminal_state (populated "
            "after a get_peer_task_result fetch confirms the peer's "
            "final state). Use this when you need to enumerate "
            "'what did I send and to whom?' without per-id round trips."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a outbound",
    )
    async def list_outbound_a2a_tasks(
        self,
        limit: int = 50,
        recipient: str = "",
    ) -> ToolResult:
        """Return the most recent outbound A2A dispatches, newest first.

        Args:
            limit: Maximum rows to return (clamped to [1, 1000]).
                Default 50.
            recipient: Optional peer name to filter by; empty returns
                rows for every recipient.
        """
        db = getattr(self, "_db", None)
        if db is None:
            return ToolResult.failed(
                "Outbound audit store unavailable (no DB attached)",
                data={"rows": [], "count": 0},
            )
        try:
            from kestrel_sovereign.a2a.outbound_store import (
                list_outbound_tasks,
            )
            rows = await list_outbound_tasks(
                db,
                agent_id=str(
                    getattr(self.agent, "did", None) or self._own_name
                ),
                limit=limit,
                recipient=recipient or None,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(
                f"Outbound audit query failed: {exc}",
                data={"rows": [], "count": 0},
            )
        public = [r.to_public_dict() for r in rows]
        return ToolResult.ok(
            confirmation=(
                f"Outbound A2A audit: {len(public)} row(s)"
                + (f" to {recipient}" if recipient else "")
            ),
            data={"rows": public, "count": len(public)},
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
        except Exception:
            return None

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

        terminal_states = ("completed", "failed", "canceled")
        backoffs = [1.0, 2.0, 5.0, 10.0]
        backoff_idx = 0
        state: Optional[str] = None
        reply_text = ""

        # Pending-question rows are correlation/audit state, not route
        # authority.  Re-read the outbound route whenever it exists rather
        # than trusting a value passed by the immediate-send path or a restart
        # replay row.  Hosted sends require this route; local sends with a
        # persisted reservation use the same check so a failed divergent-id
        # rekey cannot subscribe to a replacement peer after restart.  A
        # no-store legacy local send remains compatible by falling back to its
        # already-recorded recipient identity/name below.
        try:
            durable_recipient_agent_id = (
                await self._outbound_recipient_agent_id(task_id)
            )
            if durable_recipient_agent_id is not None:
                if (
                    recipient_agent_id is not None
                    and recipient_agent_id != durable_recipient_agent_id
                ):
                    raise PeerNotFoundError(
                        "Pending question recipient does not match outbound route"
                    )
                recipient_agent_id = durable_recipient_agent_id
        except (
            PeerNotFoundError,
            PeerAccessDeniedError,
            PeerSelfTargetError,
        ):
            state = "failed"
            reply_text = "Peer task subscription is no longer authorized."
        except PeerDirectoryConfigurationError:
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
                async for subscription_event in router.subscribe_a2a_task(
                    requester,
                    peer,
                    task_id,
                    timeout_seconds=remaining,
                ):
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
            except PeerSubscriptionUnavailableError:
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
            except (PeerNotFoundError, PeerAccessDeniedError, PeerSelfTargetError):
                # Scope changes and cross-scope probes must not reveal whether
                # the recipient or task exists.  This sender had a prior task,
                # so fail its resumption safely rather than retrying a route it
                # is no longer authorized to observe.
                state = "failed"
                reply_text = "Peer task subscription is no longer authorized."
                break
            except PeerDirectoryConfigurationError:
                state = "failed"
                reply_text = "Peer routing is no longer configured safely."
                break
            except PeerTransportError as exc:
                logger.debug(
                    "A2A subscription stream for task=%s recipient=%s "
                    "dropped (%s); backing off",
                    task_id, recipient, exc,
                )
            except PeerDirectoryError as exc:
                logger.warning(
                    "A2A subscription supervisor for task=%s "
                    "recipient=%s router error: %s",
                    task_id, recipient, exc,
                )
            except Exception as exc:  # noqa: BLE001 - provider extension boundary
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
            # Deadline passed without terminal. Fire the synthetic
            # ``state='expired'`` signal NOW (deadline-accurate) rather
            # than letting the caller wait up to an hour for the hourly
            # sweep — promised wake-by-deadline must actually happen at
            # the deadline (codex round 2 P2a on PR #1453). Mark-expired
            # FIRST so a racing hourly sweep that's also walking this row
            # gets a False return and drops its duplicate signal.
            logger.info(
                "A2A subscription supervisor for task=%s recipient=%s "
                "exited at deadline without terminal frame. Firing "
                "deadline-accurate expired signal.",
                task_id, recipient,
            )
            store = getattr(self.agent, "pending_a2a_questions", None)
            if store is not None:
                try:
                    was_waiting = await store.mark_expired(task_id)
                except Exception as e:
                    logger.warning(
                        "Failed to mark pending_a2a_question task=%s "
                        "expired: %s. Firing signal anyway — better a "
                        "possible duplicate than a missed resumption.",
                        task_id, e,
                    )
                    was_waiting = True
                if not was_waiting:
                    # Someone else (hourly sweep that beat us by a tick)
                    # got there first — drop our duplicate signal.
                    return
            fired = await self._fire_question_answered_signal(
                task_id=task_id,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                state="expired",
                reply_text="",
                causation_chain=causation_chain,
            )
            if not fired and store is not None and was_waiting:
                await self._restore_pending_question_waiting(
                    task_id,
                    state="expired",
                    reply_text="",
                    recipient=recipient,
                    original_question=original_question,
                    sess_id=sess_id,
                    causation_chain=causation_chain,
                )
            return

        # Terminal: mark resolved + fire local signal. Resolve-first so
        # the startup-replay sweep doesn't double-fire if it raced this
        # supervisor to the same terminal event.
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is not None:
            try:
                was_waiting = await store.mark_resolved(task_id)
            except Exception as e:
                logger.warning(
                    "Failed to mark pending_a2a_question task=%s "
                    "resolved: %s. Firing signal anyway — the resumed "
                    "turn should not be lost to a write failure.",
                    task_id, e,
                )
                was_waiting = True
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
        )
        if not fired and store is not None and was_waiting:
            await self._restore_pending_question_waiting(
                task_id,
                state=state,
                reply_text=reply_text,
                recipient=recipient,
                original_question=original_question,
                sess_id=sess_id,
                causation_chain=causation_chain,
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
    ) -> bool:
        """Build, enqueue, and DELIVER the local ``a2a.question_answered``
        signal, returning whether the asker was actually woken.

        Factored out so the supervisor AND the startup-replay /
        hourly-expiry sweeps share one fire path. Errors here are
        logged not raised — losing a resumption signal is bad but
        crashing the dispatcher hop is worse.

        The caller has already retired the durable ``WAITING`` row (claim-first,
        so a racing sweep can't double-fire), so ``True`` here is what keeps
        that row retired. ``enqueue_signal`` only reports *acceptance onto the
        queue*, which is not delivery — returning ``True`` on it left the row
        terminal for a wake the dispatcher then failed or dropped, and the
        asker was never resumed (#2532). So ``True`` now means terminal
        ``Status.OK``.

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
        supervised" so the caller does not also restore it.
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
            )
            enq = dispatcher.enqueue_signal(signal)
            handle = await enq if hasattr(enq, "__await__") else enq
        except Exception as e:
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
                teardown = outcome.status == STATUS_SUPERVISOR_CANCELLED
                await self._restore_pending_question_waiting(
                    task_id,
                    state=state,
                    reply_text=reply_text,
                    recipient=recipient,
                    original_question=original_question,
                    sess_id=sess_id,
                    causation_chain=causation_chain,
                    schedule_retry=schedule_retry and not teardown,
                )

            try:
                supervise_terminal_delivery(
                    self,
                    handle,
                    label=label,
                    task_name=f"a2a_question_answered_delivery:{task_id}",
                    on_undelivered=_restore_after_failure,
                )
            except Exception as e:  # noqa: BLE001
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
        except asyncio.CancelledError:
            # We are being torn down mid-delivery. The caller's `if not fired`
            # restore never runs because this propagates past it, so the
            # obligation is ours: shield the restore so the row does not stay
            # terminal for a wake nobody observed. "Better a possible duplicate
            # than a missed resumption" — the same posture this file already
            # takes for a failed mark_resolved.
            try:
                await asyncio.shield(
                    self._restore_pending_question_waiting(
                        task_id,
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
                logger.warning(
                    "Could not restore pending_a2a_question task=%s to WAITING "
                    "while cancelled mid-delivery: %s", task_id, exc,
                )
            raise

        if not outcome.delivered:
            logger.warning(
                "a2a.question_answered for task=%s recipient=%s was accepted "
                "but never delivered (%s); the pending row is restored to "
                "WAITING so the asker is still resumable",
                task_id, recipient, outcome.describe(),
            )
            return False
        return True

    async def _restore_pending_question_waiting(
        self,
        task_id: str,
        *,
        state: Optional[str] = None,
        reply_text: Optional[str] = None,
        recipient: Optional[str] = None,
        original_question: Optional[str] = None,
        sess_id: Optional[str] = None,
        causation_chain: Optional[list] = None,
        schedule_retry: bool = True,
    ) -> None:
        """Return a pending question to WAITING so replay can retry wakeup."""
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is None or not hasattr(store, "mark_waiting_for_retry"):
            return
        try:
            restored = await store.mark_waiting_for_retry(
                task_id,
                state=state,
                reply_text=reply_text,
            )
        except Exception as exc:  # noqa: BLE001
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
    ) -> None:
        """Retry a restored terminal payload before waiting for restart/sweep."""
        import asyncio

        for delay_seconds in (1, 5, 15):
            await asyncio.sleep(delay_seconds)
            store = getattr(self.agent, "pending_a2a_questions", None)
            if store is None:
                return
            try:
                if state == "expired":
                    was_waiting = await store.mark_expired(task_id)
                else:
                    was_waiting = await store.mark_resolved(task_id)
            except Exception as exc:  # noqa: BLE001
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
            )
            if fired:
                return

            await self._restore_pending_question_waiting(
                task_id,
                state=state,
                reply_text=reply_text,
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

        raw_status = data.get("status")
        if isinstance(raw_status, dict):
            current_state = raw_status.get("state")
        else:
            current_state = raw_status
        if not isinstance(current_state, str):
            return None

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
    #   - past-deadline rows → mark EXPIRED + fire synthetic
    #     ``a2a.question_answered`` with state='expired' so the asking
    #     lineage resumes with a clean branch in the prompt template
    #   - within-deadline rows → spawn a fresh subscription supervisor
    #     so the SSE wait survives process restarts
    #
    # An hourly background task runs the same expired-row scan so rows
    # whose deadline lapses without a supervisor (e.g. supervisor
    # crashed) still get a synthetic terminal signal.
    # ------------------------------------------------------------------

    # Hourly cron interval — overridable via constructor for test
    # injection. 3600s is the Sovereign-decided default.
    EXPIRY_SWEEP_INTERVAL_SECONDS = 3600

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
        except PeerDirectoryConfigurationError:
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

        try:
            await self._replay_pending_a2a_questions(store)
        except Exception as e:
            logger.warning(
                "a2a question startup-replay failed: %s. The hourly "
                "sweep is still the backstop — operators can still "
                "resume in-flight questions.",
                e, exc_info=True,
            )

        # Hourly sweep as a FEATURE-owned background task. Still agent-tracked
        # (auto-cancelled at full agent shutdown by
        # ``_shutdown_background_tasks``), but also owned by this feature so
        # runtime disable / boot rollback / soft disable cancel it via
        # ``Feature.shutdown()`` — the agent's global reap only fires at full
        # shutdown, so an agent-only task would keep sweeping after this feature
        # is torn down (kestrel-sovereign#2522 P1).
        self._track_owned_background_task(
            self._hourly_expiry_sweep_loop(store),
            name="a2a_question_expiry_sweep",
        )

    async def _replay_pending_a2a_questions(self, store) -> None:
        """Walk every WAITING row at boot. Past-deadline rows get a
        synthetic ``state='expired'`` signal; within-deadline rows get
        a fresh subscription supervisor. The chain is NOT persisted
        across restarts — a restart erases the asking turn's
        context, so the resumed signal carries an empty chain and
        the dispatcher applies its normal depth-bounded cycle check
        from scratch."""
        from datetime import datetime, timezone

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
                await self._handle_expired_row(store, row, await_delivery=False)
                expired += 1
                continue
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                await self._handle_expired_row(store, row, await_delivery=False)
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

    async def _hourly_expiry_sweep_loop(self, store) -> None:
        """Sweep ``list_waiting_past_deadline`` every hour. For each
        row mark EXPIRED + fire a synthetic ``a2a.question_answered``
        signal with ``state='expired'`` so the asking lineage
        resumes with a clean branch in the prompt template. Logs
        and continues on transient failures — this loop is the
        deadline backstop and must not die silently."""
        import asyncio

        while True:
            try:
                await asyncio.sleep(self.EXPIRY_SWEEP_INTERVAL_SECONDS)
                expired = await store.list_waiting_past_deadline()
                for row in expired:
                    try:
                        await self._handle_expired_row(store, row)
                    except Exception as e:
                        logger.warning(
                            "a2a expiry sweep: failed to expire row "
                            "task=%s: %s",
                            row.task_id, e,
                        )
                if expired:
                    logger.info(
                        "a2a expiry sweep: expired=%d", len(expired),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    "a2a expiry sweep iteration failed: %s. "
                    "Continuing.", e,
                )
                # Don't tight-loop on persistent failure — back off
                # then resume the normal cadence.
                await asyncio.sleep(60)

    async def _handle_expired_row(
        self, store, row, *, await_delivery: bool = True,
    ) -> None:
        """Mark a single WAITING row EXPIRED + fire the synthetic
        ``state='expired'`` signal. Idempotent: if the row was
        already terminal (raced the supervisor), drop silently
        instead of double-firing.

        ``await_delivery`` selects the #2532 ownership boundary and is passed
        straight through to :meth:`_fire_question_answered_signal`: the hourly
        sweep (a feature-owned background loop) waits for terminal delivery
        inline, while startup replay — which runs on the boot path, where the
        cognition turn being waited on cannot start until boot finishes —
        hands the wait to a supervisor that restores the row itself."""
        if getattr(row, "retry_state", None):
            await self._handle_retry_payload_row(
                store, row, await_delivery=await_delivery,
            )
            return

        was_waiting = await store.mark_expired(row.task_id)
        if not was_waiting:
            return
        fired = await self._fire_question_answered_signal(
            task_id=row.task_id,
            recipient=row.recipient,
            original_question=row.original_question,
            sess_id=row.origin_session_id or "",
            state="expired",
            reply_text="",
            causation_chain=None,
            await_delivery=await_delivery,
        )
        if not fired:
            await self._restore_pending_question_waiting(
                row.task_id,
                state="expired",
                reply_text="",
                recipient=row.recipient,
                original_question=row.original_question,
                sess_id=row.origin_session_id or "",
                causation_chain=None,
            )

    async def _handle_retry_payload_row(
        self, store, row, *, await_delivery: bool = True,
    ) -> None:
        """Re-fire a previously observed terminal answer after enqueue failure.

        ``await_delivery`` carries the #2532 ownership boundary through from
        the caller — see :meth:`_fire_question_answered_signal`."""
        retry_state = getattr(row, "retry_state", None)
        if not retry_state:
            return

        if retry_state == "expired":
            was_waiting = await store.mark_expired(row.task_id)
        else:
            was_waiting = await store.mark_resolved(row.task_id)
        if not was_waiting:
            return

        fired = await self._fire_question_answered_signal(
            task_id=row.task_id,
            recipient=row.recipient,
            original_question=row.original_question,
            sess_id=row.origin_session_id or "",
            state=retry_state,
            reply_text=getattr(row, "retry_reply_text", None) or "",
            causation_chain=None,
            await_delivery=await_delivery,
        )
        if not fired:
            await self._restore_pending_question_waiting(
                row.task_id,
                state=retry_state,
                reply_text=getattr(row, "retry_reply_text", None) or "",
                recipient=row.recipient,
                original_question=row.original_question,
                sess_id=row.origin_session_id or "",
                causation_chain=None,
            )

    @tool(
        name="send_a2a_task",
        description=(
            "Submit a tracked A2A task to another agent. Persists in "
            "the recipient's TaskStore, fires the a2a.task_submitted "
            "signal so they wake and process it, returns the task_id "
            "for tracking. Caller can poll status via get_a2a_task "
            "(or receive the a2a.task_complete signal). Use this for "
            "delegated work you'll check on later. For an answer "
            "now use send_a2a_question; for a fire-and-forget "
            "notification use send_a2a_message.\n\n"
            "SEND-SIDE ARTIFACTS: pass ``artifacts`` and/or "
            "``references`` to hand off durable payload (planning docs, "
            "evidence bundles, saved-memory/recall references, logs, "
            "diffs) WITH the task — the recipient retrieves them from "
            "the task store via get_task_result/check_task_status. "
            "This is the SEND side; it is distinct from the "
            "RESPONDER-side attach_artifact_to_a2a_task tool, which a "
            "RECIPIENT uses to attach output onto an INCOMING task "
            "before responding. Each artifact is a dict like "
            "{'name': 'plan', 'text': '...'} (or 'data': {...} for "
            "structured metadata, optional 'index'/'last_chunk' for "
            "chunked bodies). Each reference is a dict descriptor like "
            "{'ref_type': 'memory', 'id': '...', 'label': '...'}."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a send",
    )
    async def send_a2a_task(
        self,
        recipient: str,
        message: str,
        skill_id: str = "",
        session_id: str = "",
        # See send_a2a_question for why these are ``List[Any]``
        # rather than ``Optional[List[Any]]``: kestrel_sdk's @tool
        # schema generator maps Union (the Optional unwrap) to
        # ``string``, which is incompatible with the strict
        # validator. Codex round 2 P2 on PR #1628.
        artifacts: List[Any] = None,
        references: List[Any] = None,
    ) -> ToolResult:
        """
        Submit an A2A task to a peer agent and wake their cognition loop.

        Args:
            recipient: Peer agent name (e.g. "Meridian").
            message: The task description / prompt for the recipient.
            skill_id: Optional A2A skill id from the receiver's
                AgentCard (e.g. ``"workflow.assign"``). The valid set is
                whatever that specific recipient advertises in its
                AgentCard — call ``list_peers`` first to discover each
                peer's advertised capabilities/skills. Defaults to
                empty — the receiver routes via their default handler.
            session_id: Optional A2A session id; auto-generated when
                empty so multiple sends are independent sessions.
            artifacts: Optional send-side handoff payload. Each item is
                a dict with ``name`` and a body (``text`` for raw text,
                ``data`` for a structured dict, or pre-shaped
                ``parts``), plus optional ``description``, ``metadata``,
                ``index``, ``last_chunk``. Persisted on the recipient's
                task at SUBMITTED so the recipient can retrieve them.
            references: Optional durable references (pointers to
                saved-memory / recall items, URIs). Each item is a dict
                descriptor; carried as structured-data artifacts in the
                ``references`` group.
        """
        task_data, _chain, _recipient_agent_id, err = await self._post_a2a_task(
            recipient=recipient, message=message,
            skill_id=skill_id, session_id=session_id,
            extra_metadata={"a2a_verb": "task"},
            artifacts=artifacts, references=references,
            dispatch_tool="send_a2a_task",
        )
        if err is not None:
            return err
        attached = len(_coerce_outbound_artifacts(artifacts, references))
        return ToolResult.ok(
            confirmation=(
                f"A2A task {task_data['id']} submitted to {recipient} "
                f"(state={(task_data.get('status') or {}).get('state','?')}, "
                f"{attached} artifact(s) attached). "
                f"Recipient's dispatcher has been signaled."
            ),
            data={
                "sent": True,
                "task_id": task_data["id"],
                "session_id": task_data["sessionId"],
                "state": (task_data.get("status") or {}).get("state"),
                "recipient": recipient,
                "artifacts_attached": attached,
            },
        )

    # Agent Mesh Protocol retired in #1367. The send_mesh_message /
    # mesh_inbox / receive_mesh_message tools and the /agent/mesh
    # endpoint were replaced by send_a2a_task above (and the wider
    # send_a2a_* family in the follow-up epic). All inter-agent
    # communication now goes through /api/agent/tasks/send so it gets
    # persistence (TaskStore), lifecycle (SUBMITTED→WORKING→COMPLETED),
    # and dispatcher-driven inbound wake (a2a.task_submitted signal).
