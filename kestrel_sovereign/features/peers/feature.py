"""Peer discovery and native A2A communication for hosted agents.

Native message, task and blocking ask tools share one official SDK producer
and one sender-owned dispatch ledger. Standard Message Parts carry all input
handoffs. Published skill hints are advisory and never grant execution rights.
Questions use the primary native correlation ledger and durable signal dispatcher.
"""

import logging
import os
from collections.abc import Sequence as SequenceABC
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from kestrel_sovereign.a2a.transport_auth import ensure_a2a_transport_key
from kestrel_sovereign.execution_custody import (
    execution_work_operation,
    is_execution_control_error,
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
    PeerTransportError,
    PeerUnavailableError,
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


class OutboundSigningError(RuntimeError):
    """A loaded hybrid identity could not authenticate an A2A envelope.

    The public ``code`` is deliberately coarse: callers and the outbound audit
    need an honest failure reason, but neither surface should receive provider
    exception text that could disclose signing implementation details.
    """

    def __init__(self, code: str):
        super().__init__("hybrid A2A envelope signing failed")
        self.code = code


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

        if self._peer_router is not None and not isinstance(
            self._peer_router,
            LocalHostPeerDirectory,
        ):
            logger.info(
                "PeersFeature initialized with injected scoped peer router (self=%s)",
                self._own_name,
            )
        elif self._host_url:
            logger.info(
                f"PeersFeature initialized: host={self._host_url}, self={self._own_name}"
            )
        else:
            logger.info(
                "PeersFeature initialized but no multi_agent host found (standalone mode)"
            )

    def _get_own_name(self) -> str:
        """Get this agent's name (from KESTREL_DB_PATH basename or agent node)."""
        # Try agent node name first
        if hasattr(self.agent, "_agent_name") and self.agent._agent_name:
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
                *args,
                **kwargs,
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

    @execution_work_operation
    async def _resolve_automatic_peer(
        self,
        recipient: str,
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
        if not isinstance(recipient, str) or recipient.strip().casefold().startswith(
            "did:"
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
            raise PeerProtocolError("Peer directory returned an invalid peer identity")
        if peer.agent_id == requester.identity:
            raise PeerSelfTargetError("Cannot route to the requesting agent")
        return router, requester, peer

    @execution_work_operation
    async def _resolve_retained_automatic_peer(
        self,
        recipient_agent_id: str,
    ) -> Tuple[PeerDirectoryRouter, PeerRequester, PeerIdentity]:
        """Resolve only the stable identity retained by the native ledger."""
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
                requester,
                recipient_agent_id,
            )
        except PeerDirectoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider extension boundary
            if is_execution_control_error(exc):
                raise
            logger.exception(
                "Peer router stable-identity resolution raised unexpectedly"
            )
            raise PeerProtocolError("Peer directory resolution failed") from exc
        if peer is None:
            raise PeerNotFoundError("Peer is not in the automatic directory")
        if not isinstance(peer, PeerIdentity):
            raise PeerProtocolError("Peer directory returned an invalid peer identity")
        if peer.agent_id != recipient_agent_id:
            raise PeerProtocolError(
                "Peer directory returned a different stable peer identity"
            )
        if peer.agent_id == requester.identity:
            raise PeerSelfTargetError("Cannot route to the requesting agent")
        return router, requester, peer

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
        except (
            PeerNotFoundError,
            PeerAccessDeniedError,
            PeerSelfTargetError,
        ) as control_error:
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
        except (
            PeerNotFoundError,
            PeerAccessDeniedError,
            PeerSelfTargetError,
        ) as control_error:
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
                != (peer.agent_id if stop_scope is StopScope.AGENT else work_target)
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
            outcome.disposition is StopDisposition.STOPPED for outcome in outcomes
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
        command_prefix="!peers",
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
            directory,
            (str, bytes, bytearray),
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
                peers.append(
                    {
                        "name": peer.name or peer.slug,
                        "slug": peer.slug,
                        "status": peer.status,
                        "description": peer.description,
                    }
                )

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
            verb="task",
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
        question_deadline: datetime | None = None,
        continue_dispatch_id: str = "",
    ):
        """The common native request, reservation and send path for peer tools."""
        from uuid import uuid4

        from a2a.types import Message, Part, Role, SendMessageRequest
        from a2a.utils.errors import A2AError
        from google.protobuf.json_format import ParseDict, ParseError

        from kestrel_sovereign.a2a.extensions import (
            causation_metadata,
            delivery_intent_metadata,
            skill_hint_metadata,
            question_correlation_metadata,
        )
        from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
        from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
        from kestrel_sovereign.a2a.stores.unified.question_store import QuestionConflict
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            DispatchState,
            OutboundDispatchConflict,
        )

        if not isinstance(message, str) or not message.strip():
            return ToolResult.failed("A2A message must contain text")
        if not isinstance(context_id, str) or not isinstance(skill_id, str):
            return ToolResult.failed("A2A context_id and skill_id must be strings")
        if not isinstance(continue_dispatch_id, str):
            return ToolResult.failed("A2A continue_dispatch_id must be a string")
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
        if verb == "question":
            if question_deadline is None:
                raise ValueError("Native questions require a pre-dispatch deadline")
            metadata.update(question_correlation_metadata("question", uuid4().hex))
        elif not continue_dispatch_id:
            metadata.update(
                delivery_intent_metadata(
                    "notification" if verb == "message" else "assignment"
                )
            )
        if skill_id:
            metadata.update(skill_hint_metadata(skill_id))
        dispatch = None
        store = None
        try:
            store = self._native_outbound_store()
            parent = None
            if continue_dispatch_id:
                parent = await store.get(continue_dispatch_id)
                if parent is None:
                    raise OutboundDispatchConflict(
                        "No sender-owned continuation dispatch exists"
                    )
                if context_id and context_id != parent.remote_context_id:
                    raise OutboundDispatchConflict(
                        "Continuation cannot change the retained task context"
                    )
                context_id = parent.remote_context_id or ""
            router, requester, peer = await self._resolve_automatic_peer(recipient)
            async with router.open_a2a_session(requester, peer) as session:
                request = SendMessageRequest(
                    tenant=session.recipient_scope.tenant,
                    message=Message(
                        message_id=uuid4().hex,
                        context_id=context_id,
                        task_id=(parent.remote_task_id or "") if parent else "",
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
                    parent=parent,
                )
                if verb == "question":
                    await self._native_question_store().reserve(
                        dispatch,
                        wire,
                        deadline=question_deadline,
                        origin_turn_id=self.agent.get_current_turn_id() or "",
                        origin_session_id=self._turn_session_id() or "",
                    )
                return await session.send(dispatch, wire)
        except (
            PeerDirectoryError,
            A2AError,
            OutboundDispatchConflict,
            QuestionConflict,
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
            "root_dispatch_id": receipt.dispatch.root_dispatch_id,
            "parent_dispatch_id": receipt.dispatch.parent_dispatch_id,
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
            "Send a native A2A question and end this turn. Its durable local "
            "question correlation resumes you through a2a.question_answered. "
            "Use standard Message Parts for handoff data. Never resend an uncertain dispatch."
        ),
        category=ToolCategory.COMMUNICATION,
        command_prefix="!a2a ask",
    )
    @execution_work_operation
    async def send_a2a_question(
        self,
        recipient: str,
        message: str,
        context_id: str = "",
        timeout_seconds: int = 300,
        parts: list = None,
    ) -> ToolResult:
        from datetime import UTC, datetime, timedelta
        from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor

        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds <= 0
        ):
            return ToolResult.failed(
                "Question timeout_seconds must be a positive integer"
            )
        import asyncio

        supervisor = vars(self).get("_question_supervisor")
        owner = vars(self).get("_question_supervisor_task")
        if (
            not isinstance(supervisor, NativeQuestionSupervisor)
            or not isinstance(owner, asyncio.Task)
            or owner.done()
        ):
            return ToolResult.failed(
                "Native question recovery is not running for this feature"
            )
        if (
            supervisor.store is not self._native_question_store()
            or supervisor.dispatcher is not self.agent.dispatcher
        ):
            raise PeerDirectoryConfigurationError(
                "Native question recovery belongs to a replaced runtime"
            )
        deadline = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
        receipt = await self._dispatch_native_peer(
            recipient,
            message,
            context_id=context_id,
            parts=parts,
            skill_id="",
            verb="question",
            dispatch_tool="send_a2a_question",
            question_deadline=deadline,
        )
        supervisor.notify()
        if isinstance(receipt, ToolResult):
            return receipt
        result = self._native_dispatch_result(receipt, recipient)
        question = await supervisor.store.for_dispatch(receipt.dispatch.dispatch_id)
        result.data.update(
            question_id=question.question_id,
            awaiting_reply=True,
            expires_at=question.deadline,
            resume_via="a2a.question_answered",
        )
        return result

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

        from kestrel_sovereign.a2a.native_client import NativePeerProtocolError
        from kestrel_sovereign.a2a.native_sender import NativeSenderProofError
        from kestrel_sovereign.a2a.stores.unified.outbound_store import (
            DispatchState,
            OutboundDispatchConflict,
        )

        try:
            local = await self._native_outbound_store().get(dispatch_id)
            if local is not None and local.verb == "question":
                retained = await self._native_question_store().retained_response(local)
                if retained is not None:
                    return self._native_peer_result(local, retained)
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
        return self._native_peer_result(receipt.dispatch, receipt.response)

    @staticmethod
    def _native_peer_result(dispatch, response):
        from a2a.types import Message
        from google.protobuf.json_format import MessageToDict
        from kestrel_sovereign.a2a.projections import message_text

        taskless = isinstance(response, Message)
        message = (
            response
            if taskless
            else response.status.message
            if response.status.HasField("message")
            else None
        )
        return ToolResult.ok(
            confirmation="Read native peer response",
            data={
                "dispatch_id": dispatch.dispatch_id,
                "remote_task_id": dispatch.remote_task_id,
                "state": "completed" if taskless else dispatch.remote_task_state,
                "reply_text": (message_text(message) or "")
                if message is not None
                else "",
                "response_kind": dispatch.response_kind,
                "message" if taskless else "task": MessageToDict(response),
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

        store = None
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
            current = await store.get(task_id) if store is not None else None
            if (
                current is not None
                and current.state is DispatchState.COMPLETE
                and current.remote_task_state == "canceled"
                and current.cancel_state is DispatchState.COMPLETE
            ):
                return self._native_cancellation_result(current, reason)
            return ToolResult.failed(
                f"Could not confirm native task cancellation: {error}",
                data={"dispatch_id": task_id, "cancellation_confirmed": False},
            )
        return self._native_cancellation_result(receipt.dispatch, reason)

    @staticmethod
    def _native_cancellation_result(dispatch, reason: str | None) -> ToolResult:
        state = dispatch.remote_task_state
        if state != "canceled":
            return ToolResult.failed(
                f"Peer task is already terminal: {state}",
                data={
                    "dispatch_id": dispatch.dispatch_id,
                    "remote_task_id": dispatch.remote_task_id,
                    "state": state,
                },
            )
        return ToolResult.ok(
            confirmation="Peer task cancellation confirmed",
            data={
                "dispatch_id": dispatch.dispatch_id,
                "remote_task_id": dispatch.remote_task_id,
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
            dispatch.recipient_id
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

    # The caller owns exactly one sweep: the resident service continues
    # into it; the legacy host explicitly tracks it after replay.

    @tool(
        name="send_a2a_task",
        description=(
            "Delegate tracked work to a peer using native A2A. The dispatch_id is your "
            "immutable local handle; remote_task_id is assigned by the peer and can be null. "
            "Supply standard message parts for input attachments, such as {'text':'plan', "
            "'filename':'plan.txt'} or {'data':{'ref_type':'memory','id':'...'}}. "
            "Artifacts are peer outputs. skill_id is an optional advisory hint from the "
            "peer's signed Agent Card; it grants no direct tool invocation. "
            "Use context_id for a conversation context. To send follow-up input to an "
            "existing task, set continue_dispatch_id to its latest acknowledged local "
            "dispatch. The retained task, context and verified peer binding must match."
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
        continue_dispatch_id: str = "",
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
            continue_dispatch_id=continue_dispatch_id,
        )
        if isinstance(receipt, ToolResult):
            return receipt
        return self._native_dispatch_result(receipt, recipient)

    def _native_question_store(self):
        outbound = self._native_outbound_store()
        return self.agent.task_manager.questions_for_actor(
            outbound.sender_scope.recipient_id
        )

    @execution_work_operation
    async def post_all_features_loaded(self, agent):
        from kestrel_sovereign.a2a.question_supervisor import NativeQuestionSupervisor
        from kestrel_sovereign.features.peers.wait_provider import A2AWaitable
        from kestrel_sovereign.signals.sources.a2a_question_answered import (
            DURABLE_COGNITION_CONSUMER_ID,
        )

        registry = getattr(agent, "wait_registry", None)
        if registry is not None:
            self._register_wait_provider(registry, A2AWaitable(self), replace=True)
        if self._peer_directory_context() is None:
            return
        store = self._native_question_store()
        if not await agent.dispatcher.has_durable_consumer(
            DURABLE_COGNITION_CONSUMER_ID
        ):
            raise PeerDirectoryConfigurationError(
                "Native question durable consumer is not registered"
            )
        interval = float(os.environ.get("KESTREL_A2A_QUESTION_POLL_SECONDS", "1"))
        supervisor = NativeQuestionSupervisor(
            store=store,
            open_dispatch=self._open_native_dispatch,
            dispatcher=agent.dispatcher,
            interval=interval,
            max_interval=float(os.environ.get("KESTREL_A2A_QUESTION_MAX_POLL_SECONDS", "60")),
            is_current=lambda: (
                vars(self).get("_question_supervisor") is supervisor
                and self._native_question_store() is store
                and self.agent.dispatcher is supervisor.dispatcher
            ),
        )
        if vars(self).get("_question_supervisor") is not None:
            raise RuntimeError("Native question recovery was already started")
        self._question_supervisor = supervisor
        try:
            self._question_supervisor_task = self._track_owned_runtime_task(
                supervisor.run(), name="native_a2a_question_recovery"
            )
        except BaseException:
            self._question_supervisor = None
            self._question_supervisor_task = None
            raise

    async def shutdown(self):
        try:
            await super().shutdown()
        finally:
            self._question_supervisor = None
            self._question_supervisor_task = None
