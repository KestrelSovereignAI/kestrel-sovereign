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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool

logger = logging.getLogger(__name__)

# Timeout for inter-agent calls (seconds)
PEER_CONNECT_TIMEOUT = 5.0
PEER_READ_TIMEOUT = 300.0  # Local LLM responses (e.g. Kimi K2.5) can be very slow


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
        self._api_key = os.environ.get("KESTREL_API_KEY", "")
        self._own_name = self._get_own_name()

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
        if self._db is not None:
            try:
                await ensure_a2a_outbound_tasks_table(self._db)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PeersFeature: failed to ensure "
                    "a2a_outbound_tasks table: %s", exc,
                )

        if self._host_url:
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

    def _build_headers(self) -> dict:
        """Build headers for inter-agent HTTP calls."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        return headers

    def _maybe_sign_outbound(
        self,
        payload: Dict[str, Any],
        *,
        task_id: str,
        sess_id: str,
        message: str,
    ) -> None:
        """Sign the outbound A2A envelope if this agent has a hybrid identity (#1706).

        Sets ``metadata["sender"]`` to the signing DID — the *verified*
        identifier — and attaches ``metadata["signature"]`` (hybrid Ed25519 +
        ML-DSA-65 over the canonical view: sender, task_id, session_id, message,
        timestamp). The kids are derived from the agent's published verification
        methods so the recipient's verifier can match them. Non-hybrid
        (pre-ceremony) agents send unsigned — the recipient allows that under
        the same-host boundary (back-compat). Best-effort: a signing failure
        falls back to sending unsigned rather than breaking dispatch.
        """
        identity = getattr(self.agent, "identity", None)
        if identity is None or not getattr(identity, "is_hybrid", False):
            return
        keypair = getattr(identity, "hybrid_keypair", None)
        signing_did = getattr(identity, "signing_did", None)
        vms = getattr(identity, "new_verification_methods", None)
        if not keypair or not signing_did or not vms:
            return
        try:
            from datetime import datetime, timezone
            from kestrel_sovereign.a2a.envelope_signing import (
                canonical_message,
                kids_from_verification_methods,
                sign_envelope,
            )

            classical_kid, pq_kid = kids_from_verification_methods(vms)
            block = sign_envelope(
                keypair,
                sender=signing_did,
                task_id=task_id,
                message=canonical_message([message]),
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=sess_id,
                classical_kid=classical_kid,
                pq_kid=pq_kid,
            )
            md = payload.setdefault("metadata", {})
            # The signed DID is the verified identifier the recipient binds to.
            md["sender"] = signing_did
            md["signature"] = block
        except Exception as exc:  # noqa: BLE001 - never break dispatch on signing
            logger.warning("A2A sign-on-send failed; sending unsigned: %s", exc)

    @tool(
        name="list_peers",
        description="List all available peer agents in the multi_agent.",
        category=ToolCategory.COMMUNICATION,
        command_prefix="!peers"
    )
    async def list_peers(self) -> ToolResult:
        """
        Discover available peer agents via the multi_agent host.
        Returns their names, status, and capabilities.
        """
        if not self._host_url:
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

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._host_url}/api/agents",
                    headers=self._build_headers(),
                    timeout=PEER_CONNECT_TIMEOUT,
                )
                resp.raise_for_status()
                agents_data = resp.json()
        except httpx.ConnectError:
            return ToolResult.failed(
                "Could not connect to multi_agent host",
                data={"peers": [], "error": "Could not connect to multi_agent host"},
            )
        except Exception as e:
            logger.error(f"Failed to list peers: {e}")
            return ToolResult.failed(
                str(e),
                data={"peers": [], "error": str(e)},
            )

        peers = []
        for agent in agents_data if isinstance(agents_data, list) else agents_data.get("agents", []):
            name = agent.get("name", agent.get("id", ""))
            if name.lower() != self._own_name.lower():
                peers.append({
                    "name": name,
                    "status": agent.get("status", "unknown"),
                    "description": agent.get("description", ""),
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
        if not self._host_url:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"response": None, "agent": agent_name},
            )

        if agent_name.lower() == self._own_name.lower():
            return ToolResult.failed(
                "Cannot send a message to yourself",
                data={"response": None, "agent": agent_name},
            )

        url = f"{self._host_url}/api/agents/{agent_name}/api/agent/invoke"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json={"input": message},
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except httpx.ConnectError:
            return ToolResult.failed(
                f"Could not reach agent '{agent_name}' — multi_agent host unreachable",
                data={"response": None, "agent": agent_name},
            )
        except httpx.TimeoutException:
            return ToolResult.failed(
                f"Agent '{agent_name}' took too long to respond",
                data={"response": None, "agent": agent_name},
            )
        except Exception as e:
            logger.error(f"Failed to message agent '{agent_name}': {e}")
            return ToolResult.failed(
                str(e),
                data={"response": None, "agent": agent_name},
            )

        if resp.status_code == 404:
            return ToolResult.failed(
                f"Agent '{agent_name}' not found in the multi_agent",
                data={"response": None, "agent": agent_name},
            )
        if resp.status_code == 503:
            return ToolResult.failed(
                f"Agent '{agent_name}' is offline",
                data={"response": None, "agent": agent_name},
            )

        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return ToolResult.failed(
                str(e),
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
        Optional[ToolResult],
    ]:
        """Shared POST helper for all three a2a verbs.

        Returns ``(task_data, chain, error_result)``. On success
        ``task_data`` is the Task envelope from the recipient (with
        ``id``, ``status``, etc.) and ``chain`` is the serialized
        causation chain we attached to outbound metadata (or None when
        no chain was active); on failure ``error_result`` is a
        populated ToolResult.failed envelope the caller returns
        directly. The chain is returned so question-supervisor wiring
        can rehydrate it into the resumption signal without a second
        ContextVar read after the spawn (#1444).

        Centralizing this means the three verbs (send_a2a_message,
        send_a2a_question, send_a2a_task) share identical wire
        semantics, causation-chain attachment, and error handling —
        the difference between them is only what the caller does with
        the result (fire-and-forget vs fire-and-resume vs return
        task_id).
        """
        from uuid import uuid4

        if not self._host_url:
            return None, None, ToolResult.failed(
                "Not running in a multi_agent environment — no host to proxy through",
                data={"sent": False, "recipient": recipient},
            )

        if recipient.lower() == self._own_name.lower():
            return None, None, ToolResult.failed(
                "Cannot send an A2A task to yourself",
                data={"sent": False, "recipient": recipient},
            )

        task_id = uuid4().hex
        sess_id = session_id or uuid4().hex
        url = f"{self._host_url}/api/agents/{recipient}/api/agent/tasks/send"
        outbound_metadata: Dict[str, Any] = {"sender": self._own_name}
        if skill_id:
            outbound_metadata["skill"] = skill_id
        if extra_metadata:
            outbound_metadata.update(extra_metadata)

        # #1576: capture the audit-row write so it fires before EVERY
        # post-task_id return path (success or transport failure). The
        # helper swallows audit-store errors so dispatch can't be
        # broken by a DB hiccup.
        verb = str((extra_metadata or {}).get("a2a_verb") or "task")

        async def _persist_outbound(
            error: Optional[str] = None,
            effective_task_id: Optional[str] = None,
        ) -> None:
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
                return
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
                await record_outbound_dispatch(
                    db,
                    agent_id=str(audit_agent),
                    task_id=audit_id,
                    recipient=recipient,
                    verb=verb,
                    session_id=sess_id,
                    skill_id=skill_id or None,
                    dispatch_tool=dispatch_tool,
                    message=message,
                    error=error,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "outbound_store: record failed for task %s → %s: %s",
                    audit_id, recipient, exc,
                )
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
            return None, None, ToolResult.failed(
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
        self._maybe_sign_outbound(
            payload, task_id=task_id, sess_id=sess_id, message=message,
        )

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_READ_TIMEOUT,
                        write=PEER_READ_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except httpx.ConnectError:
            await _persist_outbound(error=f"connect_error:{recipient}")
            return None, None, ToolResult.failed(
                f"Could not reach agent '{recipient}'",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except httpx.TimeoutException:
            await _persist_outbound(error=f"timeout:{recipient}")
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' timed out",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        except Exception as e:
            logger.error(f"A2A send to '{recipient}' failed: {e}")
            await _persist_outbound(error=str(e))
            return None, None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        if resp.status_code == 404:
            await _persist_outbound(error=f"http_404:{recipient}")
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' not found or A2A endpoint missing",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )
        if resp.status_code == 503:
            await _persist_outbound(error=f"http_503:{recipient}")
            return None, None, ToolResult.failed(
                f"Agent '{recipient}' is offline or TaskManager unavailable",
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        try:
            resp.raise_for_status()
            task_data = resp.json()
        except Exception as e:
            await _persist_outbound(error=str(e))
            return None, None, ToolResult.failed(
                str(e),
                data={"sent": False, "recipient": recipient, "task_id": task_id},
            )

        # Ensure id/sessionId always populate (older recipients might
        # echo only one or the other).
        task_data.setdefault("id", task_id)
        task_data.setdefault("sessionId", sess_id)
        # Audit success — terminal_state stays NULL until a later
        # ``get_peer_task_result`` fetch learns the peer's final state.
        # Use the surfaced (peer-echoed) id so the audit row matches
        # what the caller sees in ``result.data["task_id"]``.
        await _persist_outbound(
            effective_task_id=str(task_data.get("id") or task_id),
        )
        return task_data, chain, None

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
        task_data, _chain, err = await self._post_a2a_task(
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
        """
        task_data, chain, err = await self._post_a2a_task(
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

        # Spawn the supervisor as an agent-owned background task. It
        # runs the SSE loop, fires the a2a.question_answered signal on
        # terminal frame, and exits.
        self.agent._track_background_task(
            self._supervise_a2a_question(
                task_id=task_id,
                recipient=recipient,
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
        if not self._host_url:
            return ToolResult.failed(
                "Not running in a multi_agent environment — no host "
                "to proxy through",
                data={"recipient": recipient, "task_id": task_id},
            )
        url = (
            f"{self._host_url}/api/agents/{recipient}"
            f"/api/agent/tasks/{task_id}"
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    headers=self._build_headers(),
                    timeout=httpx.Timeout(
                        connect=PEER_CONNECT_TIMEOUT,
                        read=PEER_CONNECT_TIMEOUT,
                        write=PEER_CONNECT_TIMEOUT,
                        pool=PEER_CONNECT_TIMEOUT,
                    ),
                )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            return ToolResult.failed(
                f"Could not reach peer '{recipient}' for task "
                f"{task_id}: {e}",
                data={"recipient": recipient, "task_id": task_id},
            )
        except Exception as e:
            return ToolResult.failed(
                f"Error fetching peer task {task_id} from "
                f"{recipient}: {e}",
                data={"recipient": recipient, "task_id": task_id},
            )

        if resp.status_code == 404:
            return ToolResult.failed(
                f"Task {task_id} not found on peer '{recipient}' "
                f"(either the peer evicted it or task_id is wrong)",
                data={"recipient": recipient, "task_id": task_id},
            )
        if resp.status_code != 200:
            return ToolResult.failed(
                f"Peer '{recipient}' returned HTTP {resp.status_code} "
                f"for task {task_id}",
                data={
                    "recipient": recipient,
                    "task_id": task_id,
                    "status_code": resp.status_code,
                },
            )
        try:
            data = resp.json()
        except ValueError as e:
            return ToolResult.failed(
                f"Peer '{recipient}' returned malformed JSON for "
                f"task {task_id}: {e}",
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
    ) -> None:
        """Background coroutine: SSE-subscribe → fire signal on terminal.

        Runs until the recipient's task is terminal or ``deadline_utc``
        passes. Reconnects on transient httpx failure with exponential
        backoff. Errors are logged not re-raised — supervisor death
        must not surface as an unhandled task exception."""
        import asyncio
        from datetime import datetime, timezone

        subscribe_url = (
            f"{self._host_url}/api/agents/{recipient}"
            f"/api/agent/tasks/{task_id}/subscribe"
        )
        terminal_states = ("completed", "failed", "canceled")
        backoffs = [1.0, 2.0, 5.0, 10.0]
        backoff_idx = 0
        state: Optional[str] = None
        reply_text = ""

        def _remaining() -> float:
            return max(
                0.0,
                (deadline_utc - datetime.now(timezone.utc)).total_seconds(),
            )

        while _remaining() > 0 and state not in terminal_states:
            # Bound this connect attempt's timeouts by the remaining
            # wall-clock so a peer/proxy that accepts the connection
            # then stalls without yielding any frames cannot block
            # ``aiter_lines()`` past the deadline (codex round 5 P2
            # on PR #1453). Without these caps the ``async for sse_event``
            # loop never wakes to see ``_remaining() <= 0`` and the
            # deadline-accurate expired signal never fires for stalled
            # streams. Allow a small floor so a fast deadline doesn't
            # immediately raise on connect — anything below 0.5s, we
            # just exit at the outer ``while`` check.
            remaining = _remaining()
            if remaining < 0.5:
                break
            iter_timeout = httpx.Timeout(
                connect=min(PEER_CONNECT_TIMEOUT, remaining),
                read=remaining,
                write=min(PEER_CONNECT_TIMEOUT, remaining),
                pool=min(PEER_CONNECT_TIMEOUT, remaining),
            )
            try:
                async with httpx.AsyncClient(timeout=iter_timeout) as client:
                    async with client.stream(
                        "GET",
                        subscribe_url,
                        headers=self._build_headers(),
                    ) as resp:
                        if resp.status_code == 404:
                            # Hard cut: recipient lacks the /subscribe
                            # endpoint (legacy build). Don't burn the
                            # whole deadline reconnecting.
                            logger.error(
                                "A2A question supervisor for task=%s "
                                "recipient=%s: /subscribe returned 404. "
                                "Recipient does not expose the async "
                                "question protocol. Marking the pending "
                                "row resolved with state=failed.",
                                task_id, recipient,
                            )
                            state = "failed"
                            reply_text = (
                                f"Recipient '{recipient}' does not "
                                f"expose /tasks/{{id}}/subscribe — "
                                f"upgrade them to the build that ships "
                                f"the fire-and-resume A2A question "
                                f"protocol (#1444)."
                            )
                            break
                        if resp.status_code != 200:
                            raise httpx.RequestError(
                                f"subscribe HTTP {resp.status_code}"
                            )
                        # Successful connect — reset backoff.
                        backoff_idx = 0
                        async for sse_event in self._iter_sse_events(resp):
                            # Codex round 3 P2c on PR #1453: enforce
                            # the deadline INSIDE the stream loop. On
                            # a healthy long-running task the receiver
                            # keeps the connection open emitting
                            # status/keepalive frames; without this
                            # check the supervisor blows past
                            # ``timeout_seconds`` without firing the
                            # deadline-accurate expired signal.
                            if _remaining() <= 0:
                                break
                            event_name = sse_event.get("event") or "message"
                            data_str = sse_event.get("data") or ""
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
                        # Stream ended cleanly — if we saw a terminal,
                        # exit the outer loop; otherwise reconnect (or
                        # the outer ``while`` will exit if the deadline
                        # passed during the stream read).
                        if state in terminal_states:
                            break
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.debug(
                    "A2A subscription stream for task=%s recipient=%s "
                    "dropped (%s); backing off",
                    task_id, recipient, e,
                )
            except Exception as e:
                logger.warning(
                    "A2A subscription supervisor for task=%s "
                    "recipient=%s unexpected error: %s",
                    task_id, recipient, e,
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
                await self._restore_pending_question_waiting(task_id)
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
            await self._restore_pending_question_waiting(task_id)

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
    ) -> bool:
        """Build and enqueue the local ``a2a.question_answered`` signal.

        Factored out so the supervisor AND the future startup-replay /
        hourly-expiry sweeps share one fire path. Errors here are
        logged not raised — losing a resumption signal is bad but
        crashing the dispatcher hop is worse."""
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
            await dispatcher.enqueue_signal(signal)
            return True
        except Exception as e:
            logger.error(
                "Failed to enqueue a2a.question_answered for task=%s "
                "recipient=%s: %s",
                task_id, recipient, e,
                exc_info=True,
            )
            return False

    async def _restore_pending_question_waiting(self, task_id: str) -> None:
        """Return a pending question to WAITING so replay can retry wakeup."""
        store = getattr(self.agent, "pending_a2a_questions", None)
        if store is None or not hasattr(store, "mark_waiting_for_retry"):
            return
        try:
            restored = await store.mark_waiting_for_retry(task_id)
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

    async def _iter_sse_events(self, response):
        """Parse Server-Sent Events from an httpx streaming response.

        Yields ``{event, data}`` dicts per SSE frame (terminated by a
        blank line). httpx exposes ``aiter_lines()`` which already
        strips trailing newlines, so we accumulate ``event:`` and
        ``data:`` field values until the blank-line separator. Comment
        lines (``:`` prefix) are dropped silently."""
        event_name = None
        data_lines: List[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if event_name is not None or data_lines:
                    yield {
                        "event": event_name,
                        "data": "\n".join(data_lines),
                    }
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                # Comment / heartbeat — ignore.
                continue
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip())
            # Other SSE fields (id:, retry:) are not used by our
            # producer; ignore them.

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
        absent (non-multi-agent mode, no DB) or when no host URL is
        configured (no peers to subscribe to)."""
        store = getattr(agent, "pending_a2a_questions", None)
        if store is None:
            logger.debug(
                "Skipping a2a question startup-replay — no "
                "pending_a2a_questions store wired."
            )
            return
        if self._host_url is None:
            logger.debug(
                "Skipping a2a question startup-replay — no host URL."
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

        # Hourly sweep as agent-owned background task. Auto-cancelled
        # on agent shutdown by ``_shutdown_background_tasks``.
        agent._track_background_task(
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
                await self._handle_expired_row(store, row)
                expired += 1
                continue
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if deadline <= now:
                await self._handle_expired_row(store, row)
                expired += 1
                continue
            # Within deadline — spawn supervisor. Chain is not
            # persisted across restarts; the resumed signal fires with
            # an empty chain and the dispatcher applies its normal
            # depth-bounded cycle check from scratch.
            self.agent._track_background_task(
                self._supervise_a2a_question(
                    task_id=row.task_id,
                    recipient=row.recipient,
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

    async def _handle_expired_row(self, store, row) -> None:
        """Mark a single WAITING row EXPIRED + fire the synthetic
        ``state='expired'`` signal. Idempotent: if the row was
        already terminal (raced the supervisor), drop silently
        instead of double-firing."""
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
        )
        if not fired:
            await self._restore_pending_question_waiting(row.task_id)

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
                AgentCard (e.g. ``"workflow.assign"``). Defaults to
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
        task_data, _chain, err = await self._post_a2a_task(
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
