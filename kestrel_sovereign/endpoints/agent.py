"""Agent invoke and streaming endpoints."""
from collections import defaultdict
from dataclasses import dataclass
from fastapi import APIRouter, Depends, HTTPException, Request, Query, Response, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional
import asyncio
import inspect
import json
import logging
import os
import re
import time

from kestrel_sovereign.streams.tap import AgentStreamTap

from kestrel_sovereign.kestrel_config.constants import (
    MAX_SSE_CONNECTIONS_PER_CLIENT,
    SSE_PING_INTERVAL_SECONDS,
)
from kestrel_sovereign.rate_limit import limiter
from kestrel_sovereign.security.demo_isolation import enforce_destructive_op
from kestrel_sovereign.endpoints.agent_helpers import (
    get_agent,
    request_invocation_provenance,
    resolve_request_invocation_id,
    validate_request_invocation_id,
)
from kestrel_sovereign.api_errors import ApiHTTPException
from kestrel_sovereign.a2a.stores.unified.task_store import TaskAlreadyExistsError
from kestrel_sovereign.agent.invocation import (
    InvocationCancelledError,
    invocation_log_correlation,
    invocation_id_response_header,
    new_stream_delivery_id,
    validate_invocation_id,
)
from kestrel_sovereign.agent.request_lifecycle import (
    RequestCompletionDisposition,
)
from kestrel_sovereign._async_ownership import OwnedAsyncIterator
from kestrel_sovereign.storage.privacy_wrapper import (
    PRIVACY_TRANSITION_RETRY_MESSAGE,
    PrivacyViolationError,
)
from kestrel_sovereign.stop import (
    CancellationAuthority,
    CooperativeStopTarget,
    StopDisposition,
    StopCleanupRegistry,
    StopRequest,
    StopScope,
)

logger = logging.getLogger(__name__)

# SSE connection tracking: maps (client_ip, agent_id) -> active connection count
# In multi-agent mode each agent gets its own connection pool per client.
_sse_connections: dict[tuple[str, str], int] = defaultdict(int)
_sse_lock = asyncio.Lock()

# #871 — every Kestrel HTTP route lives under /api/* now. The deprecated
# /agent/* prefix is kept working by a thin path-rewrite middleware in
# server.py for one release.
router = APIRouter(prefix="/api/agent", tags=["agent"])

# Regex strips invalid JSON escape sequences (e.g. \! from zsh shells)
_INVALID_JSON_ESCAPE = re.compile(rb'\\([^"\\/bfnrtu])')

LEGACY_CONTEXT_MODEL = "legacy/unknown"
_KITE_EVIDENCE_CONTRACT = "kite-http-evidence-v1"
_KITE_EVIDENCE_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_KITE_EVIDENCE_VALUE_RE = re.compile(r"^kite-evidence-[A-Za-z0-9_-]{20,128}$")


def _kite_release_evidence_allowed(agent: Any) -> bool:
    """Expose the fixed evidence seam only on an opted-in test agent."""
    return bool(getattr(agent, "is_test_instance", False)) and os.environ.get(
        "KESTREL_KITE_RELEASE_EVIDENCE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _kite_evidence_error(message: str) -> ApiHTTPException:
    return ApiHTTPException(status_code=400, code="invalid_kite_evidence_request", message=message)


def _kite_evidence_signature(payload: dict[str, object]) -> str:
    """Sign only the fixed, content-free typed envelope."""
    from kestrel_sovereign.knowledge.kite_evidence_signing import sign_kite_evidence

    return sign_kite_evidence(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _kite_verified_runtime_semantic_selection(agent: Any) -> tuple[Any, Any]:
    """Return the agent-bound semantic contract for the fixed recall probe.

    The typed Kite route has no caller-provided semantic inputs.  Re-check the
    already selected local profile and runtime pins at the operation boundary:
    a test-only route must not turn an incomplete or stale draft selection
    into a stable fallback.
    """
    from kestrel_sovereign.knowledge.capabilities import SemanticRuntimeCapabilities
    from kestrel_sovereign.knowledge.inference import (
        InferenceProfile,
        validate_inference_profile,
    )

    profile = getattr(agent, "semantic_inference_profile", None)
    capabilities = getattr(agent, "semantic_capabilities", None)
    if not isinstance(profile, InferenceProfile) or not isinstance(
        capabilities, SemanticRuntimeCapabilities
    ):
        raise RuntimeError(
            "Kite paraphrase recall requires a locally verified runtime semantic selection"
        )
    try:
        validate_inference_profile(profile)
        capabilities.validate()
        runtime_report = agent.storage.semantic_rdf_capability_report()
    except (AttributeError, ValueError) as error:
        raise RuntimeError(
            "Kite paraphrase recall runtime semantic selection is unavailable"
        ) from error
    if not capabilities.rdf_runtime_matches(runtime_report):
        raise RuntimeError(
            "Kite paraphrase recall runtime semantic selection is unverified"
        )
    return profile, capabilities


async def _kite_runtime_observation(
    agent: Any, *, request_id: str, provenance: Any, request: object,
) -> tuple[str, dict[str, object]]:
    """Run one allowlisted test operation, never an assistant prompt.

    The endpoint cannot select a storage method, query, artifact, or identity:
    operation names are a closed set and the server owns the underlying call.
    """
    if not _kite_release_evidence_allowed(agent):
        raise ApiHTTPException(status_code=404, code="not_found", message="Not found.")
    if not isinstance(request, dict) or set(request).difference({"operation", "nonce", "value"}):
        raise _kite_evidence_error("Invalid Kite evidence request.")
    operation, nonce, value = request.get("operation"), request.get("nonce"), request.get("value")
    if not isinstance(operation, str) or not _KITE_EVIDENCE_NONCE_RE.fullmatch(nonce if isinstance(nonce, str) else ""):
        raise _kite_evidence_error("Invalid Kite evidence request.")
    if operation == "save":
        if not isinstance(value, str) or not _KITE_EVIDENCE_VALUE_RE.fullmatch(value):
            raise _kite_evidence_error("Invalid Kite evidence value.")
        command: str | None = f"!memory-save-fact user preferred_deploy_region {value}"
    elif operation == "delete":
        if value is not None:
            raise _kite_evidence_error("Invalid Kite evidence request.")
        command = "!memory-forget-fact user preferred_deploy_region"
    elif operation == "diagnostics":
        if value is not None:
            raise _kite_evidence_error("Invalid Kite evidence request.")
        command = None
    elif operation == "quarantine":
        if value is not None:
            raise _kite_evidence_error("Invalid Kite evidence request.")
        command = None
    elif operation in {"sleep", "sleep_changed", "sleep_unchanged", "paraphrase_recall", "erasure_core_snapshot"}:
        if value is not None:
            raise _kite_evidence_error("Invalid Kite evidence request.")
        command = None
    else:
        raise _kite_evidence_error("Invalid Kite evidence operation.")

    from kestrel_sovereign.agent.invocation import invocation_scope
    from kestrel_sovereign.knowledge.kite_evidence_signing import (
        KiteEvidenceNonceReplay, KiteEvidenceSigningError, consume_kite_evidence_nonce,
    )
    try:
        nonce_receipt = consume_kite_evidence_nonce(
            nonce, issue_receipt=operation == "erasure_core_snapshot",
        )
    except KiteEvidenceNonceReplay as error:
        raise ApiHTTPException(status_code=409, code="kite_evidence_nonce_replayed", message="Kite evidence nonce was already consumed.") from error
    except KiteEvidenceSigningError as error:
        raise RuntimeError("Kite evidence nonce ledger is unavailable") from error

    with invocation_scope(request_id, provenance=provenance):
        if operation in {"sleep", "sleep_changed", "sleep_unchanged"}:
            report = await agent.sleep(tier="local", skip_export=True)
            if getattr(report, "success", None) is not True:
                raise RuntimeError("Kite sleep operation did not complete")
            return operation, {"sleep_success_count": 1}
        if operation == "paraphrase_recall":
            profile, capabilities = _kite_verified_runtime_semantic_selection(agent)
            maintenance = await agent.storage.run_semantic_maintenance(
                profile,
                inference_limits=getattr(agent, "semantic_inference_limits", None),
                maintenance_limits=getattr(agent, "semantic_maintenance_limits", None),
                semantic_capabilities=capabilities,
            )
            if getattr(getattr(maintenance, "status", None), "value", None) not in {"complete", "no_op"}:
                raise RuntimeError("Kite paraphrase recall maintenance did not reach a terminal checkpoint")
            recall = await agent.storage.semantic_recall_candidates(
                query="Which region should the deployment use?", candidate_scan_limit=10,
                inference_profile=profile,
                inference_limits=getattr(agent, "semantic_inference_limits", None),
                maintenance_limits=getattr(agent, "semantic_maintenance_limits", None),
            )
            hydrated = await agent.storage.hydrate_semantic_recall_candidates(
                tuple(candidate.assertion.assertion_id for candidate in getattr(recall, "candidates", ())),
                expected_checkpoint_generation=recall.checkpoint_generation,
                inference_profile=profile,
                inference_limits=getattr(agent, "semantic_inference_limits", None),
                maintenance_limits=getattr(agent, "semantic_maintenance_limits", None),
            )
            if not hydrated or any(not candidate.source_occurrences for candidate in hydrated):
                raise RuntimeError("Kite paraphrase recall did not hydrate provenance")
            return operation, {"retrieval_count": len(hydrated), "provenance_check_count": len(hydrated)}
        if operation == "erasure_core_snapshot":
            from kestrel_sovereign.knowledge.kite_erasure_authority import (
                KiteErasureDrillAuthorityError,
                _issue_kite_erasure_drill_capability,
                _typed_kite_erasure_endpoint_issuance_scope,
            )

            try:
                with _typed_kite_erasure_endpoint_issuance_scope():
                    capability = _issue_kite_erasure_drill_capability(
                        nonce_receipt, operation=operation,
                    )
                    observation = await agent.storage.semantic_release_erasure_drill(
                        capability=capability,
                    )
            except (KiteErasureDrillAuthorityError, KiteEvidenceSigningError) as error:
                raise RuntimeError("Kite core erasure authority is unavailable") from error
            if not isinstance(observation, dict):
                raise RuntimeError("Kite core erasure drill is unavailable")
            return operation, observation
        if operation == "diagnostics":
            observation = await agent.storage.semantic_release_kite_diagnostics(
                operation_id=f"kite-diagnostics-{nonce}"
            )
            if not isinstance(observation, dict):
                raise RuntimeError("Kite semantic diagnostics are unavailable")
            return operation, observation
        if operation == "quarantine":
            observation = await agent.storage.semantic_release_kite_invalid_import_quarantine(
                operation_id=f"kite-import-quarantine-{nonce}"
            )
            if observation != {"invalid_import_quarantine_count": 1}:
                raise RuntimeError("Kite invalid import probe did not complete")
            return operation, observation

        task_manager = getattr(agent, "task_manager", None)
        if task_manager is None:
            raise RuntimeError("Kite runtime command dispatcher is unavailable")
        raw = await task_manager.execute_command(command)
    from kestrel_sovereign.features.base import is_flat_toolresult_envelope
    if not is_flat_toolresult_envelope(raw) or raw.get("status") != "ok" or not isinstance(raw.get("data"), dict):
        raise RuntimeError("Kite runtime command did not return a successful typed result")
    data = raw["data"]
    if operation == "save":
        if data.get("saved") is not True:
            raise RuntimeError("Kite runtime fact write did not complete")
        return operation, {"fact_write_count": 1}
    if operation == "delete":
        if data.get("deleted") is not True:
            raise RuntimeError("Kite runtime fact deletion did not complete")
        return operation, {"fact_delete_count": 1}
    raise RuntimeError("Kite runtime command operation was not recognized")


def _privacy_transition_conflict() -> HTTPException:
    """Return the content-safe retry contract for an active fact lease."""
    return HTTPException(
        status_code=409,
        detail=PRIVACY_TRANSITION_RETRY_MESSAGE,
        headers={"Retry-After": "1"},
    )


def _invalid_json_message(error: ValueError) -> str:
    """Describe malformed JSON without echoing submitted body content."""
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON at line {error.lineno}, column {error.colno}."
    return "Invalid JSON request body."


def _require_json_object(value: Any) -> dict:
    """Reject valid JSON scalars/arrays before endpoint code calls ``.get``."""
    if isinstance(value, dict):
        return value
    raise ApiHTTPException(
        status_code=400,
        code="invalid_request_body",
        message="JSON request body must be an object.",
    )


def _latest_assistant_model_identity(
    history: list[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """Return the latest assistant turn's stamped model/provider identity."""
    for row in reversed(history):
        if (row.get("role") or "").lower() != "assistant":
            continue
        model = row.get("model") or None
        provider = row.get("provider") or None
        if model and provider:
            context_model = f"{provider}/{model}"
        elif model:
            context_model = model
        else:
            context_model = LEGACY_CONTEXT_MODEL
        return {
            "model": model,
            "provider": provider,
            "context_model": context_model,
            "model_source": "assistant_turn" if model else "legacy_assistant_turn",
        }
    return {
        "model": None,
        "provider": None,
        "context_model": LEGACY_CONTEXT_MODEL,
        "model_source": "no_assistant_turn",
    }


async def _parse_json_body(request: Request) -> dict:
    """Parse JSON body, recovering from common shell-escaping issues."""
    try:
        return _require_json_object(await request.json())
    except (json.JSONDecodeError, ValueError) as orig_err:
        raw = await request.body()
        cleaned = _INVALID_JSON_ESCAPE.sub(lambda m: m.group(1), raw)
        if cleaned != raw:
            try:
                return _require_json_object(json.loads(cleaned))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        raise ApiHTTPException(
            status_code=400,
            code="invalid_json",
            message=_invalid_json_message(orig_err),
        )


async def _parse_optional_json_body(request: Request) -> dict:
    """Parse JSON when present, returning an empty dict for an empty body."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        return _require_json_object(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError) as orig_err:
        cleaned = _INVALID_JSON_ESCAPE.sub(lambda m: m.group(1), raw)
        if cleaned != raw:
            try:
                return _require_json_object(json.loads(cleaned))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        raise ApiHTTPException(
            status_code=400,
            code="invalid_json",
            message=_invalid_json_message(orig_err),
        )


@router.post("/invoke")
@limiter.limit("60/minute")
async def invoke_agent(request: Request, http_response: Response):
    """
    Main endpoint to interact with the Kestrel Agent.
    It takes user input and returns the agent's response.
    Optionally accepts:
      - 'model' parameter to override the default model
      - 'session_id' to load context from a specific conversation session
    """
    try:
        data = await _parse_json_body(request)
        user_input = data.get("input")
        model_override = data.get("model")
        provider_override = data.get("provider")
        session_id = data.get("session_id")
        user_passphrase = data.get("user_passphrase")
        kite_evidence_request = data.get("kite_evidence")

        if user_input is None and not isinstance(kite_evidence_request, dict):
            raise ApiHTTPException(
                status_code=400,
                code="input_required",
                message="Input not provided.",
            )

        # Combine provider and model for proper routing
        if provider_override and model_override:
            model_override = f"{provider_override}/{model_override}"

        agent = get_agent(request)
        caller = getattr(request.state, "caller", None)

        # A client may repeat the same opaque request id after a transport
        # failure. Tool provenance derives its operation identity from this
        # task-local id, so an exact retry reaches the canonical store's own
        # idempotency ledger instead of being mistaken for a new invocation.
        request_id = resolve_request_invocation_id(request, data)
        invocation_provenance = request_invocation_provenance(
            request,
            source_locator="POST:/api/agent/invoke",
        )
        if hasattr(agent, "register_active_request"):
            agent.register_active_request(request_id)
        else:
            agent._current_request_id = request_id

        request_cancelled = getattr(agent, "is_request_cancelled", None)
        if callable(request_cancelled) and request_cancelled(request_id) is True:
            try:
                http_response.headers["X-Request-ID"] = (
                    invocation_id_response_header(request_id)
                )
                return {
                    "response": "Request stopped before execution.",
                    "session_id": session_id,
                    "model": None,
                    "provider": None,
                }
            finally:
                agent._cleanup_cancelled_request(request_id)

        if isinstance(kite_evidence_request, dict):
            if user_input not in (None, ""):
                raise _kite_evidence_error("Kite evidence requests cannot include input.")
            owner_task = asyncio.current_task()
            owner_cancellation_baseline = (
                owner_task.cancelling() if owner_task is not None else 0
            )
            try:
                evidence_task = asyncio.create_task(
                    _kite_runtime_observation(
                        agent,
                        request_id=request_id,
                        provenance=request_invocation_provenance(
                            request,
                            source_locator=(
                                "POST:/api/agent/invoke#kite-release-evidence"
                            ),
                        ),
                        request=kite_evidence_request,
                    ),
                    name=(
                        "kite-evidence:"
                        f"{invocation_log_correlation(request_id)}"
                    ),
                )
                bind_operation = getattr(
                    type(agent), "bind_request_operation", None
                )
                if callable(bind_operation):
                    bind_operation(agent, request_id, evidence_task)
                try:
                    operation, observation = await evidence_task
                except asyncio.CancelledError:
                    if (
                        owner_task is not None
                        and owner_task.cancelling()
                        > owner_cancellation_baseline
                    ):
                        raise
                    if not (
                        callable(request_cancelled)
                        and request_cancelled(request_id) is True
                    ):
                        raise
                    http_response.headers["X-Request-ID"] = (
                        invocation_id_response_header(request_id)
                    )
                    return {
                        "response": "Request stopped during execution.",
                        "session_id": session_id,
                        "model": None,
                        "provider": None,
                    }
                # Stop can linearize after the evidence task has produced its
                # observation but before this owner publishes signed success.
                # Re-read the exact generation with no following await.
                if (
                    callable(request_cancelled)
                    and request_cancelled(request_id) is True
                ):
                    http_response.headers["X-Request-ID"] = (
                        invocation_id_response_header(request_id)
                    )
                    return {
                        "response": "Request stopped during execution.",
                        "session_id": session_id,
                        "model": None,
                        "provider": None,
                    }
            finally:
                agent._cleanup_cancelled_request(request_id)
            nonce = kite_evidence_request.get("nonce")
            assert isinstance(nonce, str)
            signed = {
                "contract": _KITE_EVIDENCE_CONTRACT,
                "nonce": nonce,
                "operation": operation,
                "observation": observation,
            }
            http_response.headers["X-Request-ID"] = invocation_id_response_header(request_id)
            return {
                "response": "Kite runtime evidence operation complete.",
                "session_id": None,
                "model": None,
                "provider": None,
                "kite_evidence": {**signed, "signature": _kite_evidence_signature(signed)},
            }

        # Pre-resolve the effective session_id so it can be returned to
        # the client. Without this, the frontend pane never learns the
        # implicit UUID derived inside add_conversation and stays
        # anchored on `null`, causing later auto-load + context-status
        # paths to lose continuity. Reviewer flagged at chat.js:520.
        try:
            effective_session_id = await agent.storage.resolve_session_id(session_id)
        except Exception:
            effective_session_id = session_id  # fall back; never block the request

        try:
            response = await agent.process_input(
                user_input,
                model_override=model_override,
                session_id=effective_session_id,
                caller=caller,
                user_passphrase=user_passphrase,
                invocation_id=request_id,
                invocation_provenance=invocation_provenance,
            )
            if callable(request_cancelled) and request_cancelled(request_id) is True:
                http_response.headers["X-Request-ID"] = (
                    invocation_id_response_header(request_id)
                )
                return {
                    "response": "Request stopped during execution.",
                    "session_id": effective_session_id,
                    "model": None,
                    "provider": None,
                }
        except (asyncio.CancelledError, InvocationCancelledError):
            if callable(request_cancelled) and request_cancelled(request_id) is True:
                http_response.headers["X-Request-ID"] = (
                    invocation_id_response_header(request_id)
                )
                return {
                    "response": "Request stopped during execution.",
                    "session_id": effective_session_id,
                    "model": None,
                    "provider": None,
                }
            raise
        finally:
            agent._cleanup_cancelled_request(request_id)
        # Extract model/provider identity for frontend footer rendering (#1373)
        identity = agent._conversation_response_identity(use_last_identity=True)
        http_response.headers["X-Request-ID"] = invocation_id_response_header(request_id)
        return {
            "response": response,
            "session_id": effective_session_id,
            "model": identity.get("model"),
            "provider": identity.get("provider"),
        }
    except HTTPException:
        raise
    except Exception:
        # Invocation failures can wrap caller content, provider errors, or a
        # client-controlled retry id.  Keep the operator event useful without
        # recording any of those values outside the governed request path.
        logger.error("Agent invocation failed")
        raise ApiHTTPException(
            status_code=500,
            code="invoke_failed",
            message="An internal error occurred.",
        )


# Chat attachments (#1662). Images can be sent to the model as vision input
# (eager) or read on demand (lazy); documents are read on demand. Stored
# content-addressed + encrypted via the agent file store; the privacy wrapper
# enforces the mode (EPHEMERAL rejects, ISOLATED buffers) so an attachment
# inherits the message's privacy automatically.
_ATTACHMENT_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
_ATTACHMENT_DOC_TYPES = {"application/pdf", "text/plain", "text/markdown"}
_ATTACHMENT_TYPES = _ATTACHMENT_IMAGE_TYPES | _ATTACHMENT_DOC_TYPES
_ATTACHMENT_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
# SHA-256 hex (the file store's content hash) — the only id we accept back.
_ATTACHMENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _sanitize_attachments(raw) -> list:
    """Coerce client-supplied attachment refs to a safe, fixed shape before
    they touch persisted metadata. Drops anything malformed; bounds the count.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:10]:  # a turn won't reference more than a handful
        if not isinstance(item, dict):
            continue
        h = item.get("hash")
        if not isinstance(h, str) or not _ATTACHMENT_HASH_RE.match(h):
            continue
        kind = item.get("kind")
        if kind not in ("image", "document"):
            kind = "document"
        mime = item.get("mime")
        safe_mime = mime if (isinstance(mime, str) and mime in _ATTACHMENT_TYPES) else None
        out.append({
            "hash": h,
            "kind": kind,
            "mime": safe_mime,
            "name": (str(item.get("name") or "attachment"))[:255],
            # #1662 eager vision: inline (sent as vision input this turn)
            # requires an image kind AND a trusted image MIME. This is a cheap
            # first filter — the bytes are re-validated by magic number before
            # they're sent, so a tampered ref can't smuggle a document through.
            "inline": (
                bool(item.get("inline"))
                and kind == "image"
                and bool(safe_mime)
                and safe_mime.startswith("image/")
            ),
        })
    return out


@router.post("/attachments")
@limiter.limit("30/minute")
async def upload_attachment(request: Request, file: UploadFile = File(...)):
    """Upload a chat attachment (image or document); returns a reference the
    composer attaches to the next message."""
    agent = get_agent(request)
    # agent.storage is the PrivacyEnforcingStorage facade — its store_file
    # checks write permission (EPHEMERAL -> PrivacyViolationError) and session-
    # buffers in ISOLATED. Going through storage.files would bypass all that.
    storage = getattr(agent, "storage", None)
    if not storage or not hasattr(storage, "store_file"):
        raise HTTPException(status_code=503, detail="File storage not available.")

    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in _ATTACHMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported attachment type '{ctype}'. Allowed: images "
                "(jpeg/png/webp/gif) and documents (pdf/text/markdown)."
            ),
        )
    # Read at most cap+1 bytes so an oversized upload (allowed content-type)
    # can't force materializing an arbitrarily large body in memory.
    data = await file.read(_ATTACHMENT_MAX_SIZE + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > _ATTACHMENT_MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum {_ATTACHMENT_MAX_SIZE // (1024 * 1024)} MB.",
        )

    name = (file.filename or "attachment")[:255]
    kind = "image" if ctype in _ATTACHMENT_IMAGE_TYPES else "document"
    try:
        content_hash = await storage.store_file(
            data,
            name,
            metadata={
                "type": "attachment",
                "kind": kind,
                "mime_type": ctype,
                "agent_id": getattr(agent, "agent_id", ""),
            },
        )
    except ValueError as e:
        # store_file raises ValueError when the content exceeds the store cap.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # PrivacyEnforcingStorage rejects writes in EPHEMERAL mode.
        if type(e).__name__ == "PrivacyViolationError":
            raise HTTPException(
                status_code=403,
                detail="Attachments are disabled in this privacy mode.",
            )
        logger.error(f"Attachment store failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store attachment.")

    return {
        "success": True,
        "hash": content_hash,
        "mime": ctype,
        "name": name,
        "kind": kind,
        "size": len(data),
        "url": f"/api/files/{content_hash}",
    }


@router.post("/stream")
@limiter.limit("60/minute")
async def stream_agent_response(request: Request):
    """
    Streaming endpoint for chat responses.
    Returns text chunks as they are generated.
    Optionally accepts 'session_id' to load context from a specific conversation.
    """
    agent = None
    request_id = None
    stream_tap = None
    stream_delivery_id = None
    request_lifecycle_registered = False
    stream_tap_registered = False

    def cleanup_unstarted_stream() -> None:
        """Undo setup if constructing the response fails before generation."""
        if stream_tap_registered and stream_tap is not None and stream_delivery_id is not None:
            stream_tap.unregister(stream_delivery_id)
        if request_lifecycle_registered and agent is not None and request_id is not None:
            agent._cleanup_cancelled_request(request_id)

    try:
        data = await _parse_json_body(request)
        user_input = data.get("input")
        model_override = data.get("model")
        provider_override = data.get("provider")
        session_id = data.get("session_id")
        audit_before_streaming = data.get("audit_before_streaming", False)
        # #1662: attachment references the composer uploaded for this turn.
        # Sanitize to the known shape — never trust client JSON verbatim into
        # persisted metadata.
        attachments = _sanitize_attachments(data.get("attachments"))

        if user_input is None:
            raise ApiHTTPException(
                status_code=400,
                code="input_required",
                message="Input not provided.",
            )

        agent = get_agent(request)
        caller = getattr(request.state, "caller", None)

        # Combine provider and model into provider/model format for routing.
        # The streaming path uses "/" in model_override to identify the provider
        # and filter to only that provider (avoids trying all providers).
        if provider_override and model_override:
            model_override = f"{provider_override}/{model_override}"

        # The client may supply the same opaque id for a transport retry. It
        # is both the cancellation key and the task-local provenance identity.
        request_id = resolve_request_invocation_id(request, data)
        invocation_provenance = request_invocation_provenance(
            request,
            source_locator="POST:/api/agent/stream",
        )
        if hasattr(agent, "register_active_request"):
            agent.register_active_request(request_id)
        else:
            agent._current_request_id = request_id
        request_lifecycle_registered = True

        # Register the stream tap so TTS consumers can subscribe
        stream_tap = AgentStreamTap.get_instance()
        # A retry may deliberately reuse ``request_id`` to reach the canonical
        # assertion idempotency ledger.  TTS delivery is independent: a fresh,
        # server-owned id prevents concurrent response streams from publishing
        # into or closing each other's tap queue.
        stream_delivery_id = new_stream_delivery_id()
        stream_tap.register(stream_delivery_id)
        stream_tap_registered = True

        # Pre-resolve the effective session_id and surface it via a
        # response header. Resolved BEFORE StreamingResponse is created
        # because headers are immutable once the body starts streaming.
        # The frontend pane uses this to learn its durable conversation
        # id on first send (replacing the prior pane.sessionId=null
        # heuristic that left auto-load + context-status fragile).
        try:
            effective_session_id = await agent.storage.resolve_session_id(session_id)
        except Exception:
            effective_session_id = session_id  # fall back; never block the stream

        async def generate():
            # Shared stop notice for the in-loop cancel check AND the post-loop
            # fallback (#2674). A strict (fail-closed) response audit that is
            # stopped before dispatch WITHHOLDS every chunk and returns cleanly,
            # so the generator is empty and the in-loop check below never runs —
            # without the post-loop emit the client would get a silent, empty 200
            # instead of the standard "Request stopped" body.
            stop_notice = (
                "\n\n---\n⏹️ **Request stopped**\n\n"
                "Type `!continue` to resume from where I left off, or start a new message."
            )
            stop_notice_emitted = False
            # #2674 P2: track whether the turn ever surfaced a user-visible
            # response chunk. The post-loop fallback below must fire ONLY for a
            # genuinely empty stream (strict cancel-before-dispatch withholds
            # every chunk). If any chunk reached the client, the response is
            # complete — a cancellation that becomes visible AFTER the final
            # chunk but BEFORE this async generator exits must not retroactively
            # append "Request stopped" to an already-delivered answer.
            response_chunk_yielded = False
            agent_stream = None
            try:
                if agent.is_request_cancelled(request_id) is True:
                    yield stop_notice
                    stop_notice_emitted = True
                    return
                from kestrel_sovereign.agent.streaming import strip_revise_sentinels
                agent_stream = OwnedAsyncIterator(
                    lambda: agent.process_input_streaming(
                        user_input,
                        model_override=model_override,
                        session_id=effective_session_id,
                        audit_before_streaming=audit_before_streaming,
                        caller=caller,
                        request_id=request_id,
                        invocation_provenance=invocation_provenance,
                        attachments=attachments,
                    ),
                    operation="agent stream cleanup",
                )
                async for chunk in agent_stream:
                    # Check if request was cancelled
                    if agent.is_request_cancelled(request_id):
                        yield stop_notice
                        stop_notice_emitted = True
                        break
                    # Wave 5E: strip the in-band revise sentinel before
                    # publishing to TTS subscribers — voice/TTS speaks
                    # raw chunks aloud, so leaking ``\\x1eKESTREL:REVISE...``
                    # into the audio path is a regression. The chat
                    # client receives the sentinel-bearing chunk on
                    # the yield below and strips it client-side.
                    tts_chunk = strip_revise_sentinels(chunk)
                    if tts_chunk:
                        await stream_tap.publish(stream_delivery_id, tts_chunk)
                    response_chunk_yielded = True
                    yield chunk
                # #2674: a strict-audit turn cancelled before dispatch withholds
                # every chunk and returns cleanly, so the loop body above never
                # ran and the in-loop notice never fired. Surface the standard
                # stop notice once here when the request was cancelled and the
                # stream produced NO output. Gating on ``response_chunk_yielded``
                # closes the P2 race: a normal/incremental or strict-approved
                # turn that already delivered its final chunk must not be labeled
                # stopped just because cancellation landed after that chunk but
                # before this generator exited. A break already set
                # ``stop_notice_emitted``, so a normal in-loop cancel won't
                # double-emit either.
                if (
                    not stop_notice_emitted
                    and not response_chunk_yielded
                    and agent.is_request_cancelled(request_id)
                ):
                    yield stop_notice
                    stop_notice_emitted = True
            except Exception as e:
                # A request id and exception text can be client-controlled or
                # contain withheld content.  Keep only a one-way correlation
                # in the operator log; the client receives the shared safe
                # error boundary below.
                from kestrel_sovereign.agent.invocation import (
                    invocation_log_correlation,
                )
                logger.error(
                    "Streaming request failed (correlation=%s)",
                    invocation_log_correlation(request_id),
                )
                # #2674 findings 3 & 4: emit the user-visible error through the
                # ONE shared safe boundary used by /api/bridge/stream too, so the
                # two transports cannot drift. It NEVER reflects ``str(e)``,
                # ``underlying``, or ``provider`` — an adapter that raises after
                # yielding partial prose can carry withheld response content or an
                # injected marker, and ``LLMStreamingError.provider`` is an
                # unvalidated free string (finding 4: it leaked
                # ROUTE_FIELD_UNBOUNDED_MARKER__WITHHELD_TEXT). A route failure
                # still gets the no-blind-fallback / recovery guidance via a
                # CONSTANT "your selected model route" label; the failing route
                # and full error remain unavailable to this transport.
                from kestrel_sovereign.llm.streaming_errors import (
                    agent_stream_error_block,
                )
                yield agent_stream_error_block(e)
            finally:
                agent_stream_cleanup_failed = False
                try:
                    # One producer task owns construction, iteration, and close
                    # of the nested generator. Its join is cancellation-safe
                    # without moving ContextVar token reset into a copied task
                    # context.
                    if agent_stream is not None:
                        await agent_stream.aclose()
                except BaseException:
                    agent_stream_cleanup_failed = (
                        agent_stream is not None
                        and agent_stream.cleanup_error is not None
                    )
                    raise
                else:
                    agent_stream_cleanup_failed = (
                        agent_stream is not None
                        and agent_stream.cleanup_error is not None
                    )
                finally:
                    try:
                        # Signal stream completion for TTS consumers
                        await stream_tap.finish(stream_delivery_id)
                    finally:
                        # A failed nested close is an abandoned lifecycle,
                        # never proof that Stop succeeded.
                        if agent_stream_cleanup_failed:
                            agent._cleanup_cancelled_request(
                                request_id,
                                disposition=(
                                    RequestCompletionDisposition.ABANDONED
                                ),
                            )
                        else:
                            agent._cleanup_cancelled_request(request_id)

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": invocation_id_response_header(request_id),
            "X-Stream-Delivery-ID": stream_delivery_id,
        }
        if effective_session_id:
            headers["X-Session-Id"] = effective_session_id
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers=headers,
        )
    except HTTPException:
        cleanup_unstarted_stream()
        raise
    except Exception:
        cleanup_unstarted_stream()
        logger.error("Error setting up stream")
        raise ApiHTTPException(
            status_code=500,
            code="stream_setup_failed",
            message="Error setting up stream.",
        )


@router.post("/stop")
async def stop_agent_request(request: Request):
    """
    Stop the current agent request/streaming.
    Used by the stop button in the UI.
    """
    try:
        data = await _parse_optional_json_body(request)
        # The body and query forms predate the shared retry-header contract and
        # remain literal values.  Only X-Request-ID is a percent-encoded wire
        # form, so a client can copy an invoke/stream response header here
        # verbatim without forking the cancellation key.
        body_has_request_id = "request_id" in data
        query_has_request_id = "request_id" in request.query_params
        explicit_request_id = (
            data["request_id"]
            if body_has_request_id
            else request.query_params.get("request_id")
        )
        explicit_request_id_present = (
            body_has_request_id or query_has_request_id
        )
        body_has_turn_id = "turn_id" in data
        query_has_turn_id = "turn_id" in request.query_params
        explicit_turn_id_present = body_has_turn_id or query_has_turn_id
        explicit_turn_id = (
            data["turn_id"]
            if body_has_turn_id
            else request.query_params.get("turn_id")
        )
        if explicit_turn_id_present and (
            explicit_request_id_present
            or request.headers.get("X-Request-ID") is not None
        ):
            raise ApiHTTPException(
                status_code=400,
                code="ambiguous_stop_target",
                message="Pass either request_id or turn_id, not both.",
            )
        if explicit_turn_id_present:
            try:
                turn_id = validate_invocation_id(explicit_turn_id)
            except ValueError as error:
                raise ApiHTTPException(
                    status_code=400,
                    code="invalid_turn_id",
                    message=f"Invalid turn_id: {error}",
                ) from error
        else:
            turn_id = None
        if explicit_request_id_present:
            request_id = validate_request_invocation_id(explicit_request_id)
        elif request.headers.get("X-Request-ID") is not None:
            request_id = resolve_request_invocation_id(request, {})
        else:
            request_id = None
        agent = get_agent(request)
        agent_id = getattr(agent, "agent_id", None)
        if not isinstance(agent_id, str) or not agent_id.strip():
            # Compatibility for pre-inception/test agents. This is an address,
            # not a grant; HTTP caller authorization remains at the route.
            agent_id = "local-agent"
        caller = getattr(request.state, "caller", None)
        actor_id = getattr(caller, "identity", None)
        if not isinstance(actor_id, str) or not actor_id.strip():
            actor_id = f"local-operator:{agent_id}"

        active_request_ids = set(
            getattr(agent, "_active_request_ids", set()) or set()
        )
        abandoned_turns = getattr(agent, "_abandoned_request_generations", None)
        if isinstance(abandoned_turns, dict):
            active_request_ids.update(abandoned_turns)
        current_turn = getattr(agent, "_current_request_id", None)
        if isinstance(current_turn, str) and current_turn:
            active_request_ids.add(current_turn)
        if request_id is not None:
            active_request_ids.add(request_id)
        instance_binding_accessor = vars(agent).get(
            "active_turn_request_bindings"
        )
        if callable(instance_binding_accessor):
            has_binding_accessor = True
            raw_turn_bindings = instance_binding_accessor()
        else:
            class_binding_accessor = getattr(
                type(agent),
                "active_turn_request_bindings",
                None,
            )
            has_binding_accessor = callable(class_binding_accessor)
            raw_turn_bindings = (
                class_binding_accessor(agent) if has_binding_accessor else None
            )
        if has_binding_accessor:
            if not isinstance(raw_turn_bindings, dict):
                raise TypeError("agent turn binding inventory has an invalid type")
            turn_request_ids = {}
            turn_request_generations = {}
            for indexed_turn_id, binding in raw_turn_bindings.items():
                if (
                    not isinstance(indexed_turn_id, str)
                    or not isinstance(binding, tuple)
                    or len(binding) != 2
                    or not isinstance(binding[0], str)
                ):
                    raise TypeError("agent turn binding inventory is malformed")
                try:
                    validate_invocation_id(indexed_turn_id)
                    validate_invocation_id(binding[0])
                except ValueError as error:
                    raise TypeError(
                        "agent turn binding inventory is malformed"
                    ) from error
                turn_request_ids[indexed_turn_id] = binding[0]
                if binding[1] is not None:
                    if (
                        not isinstance(binding[1], int)
                        or isinstance(binding[1], bool)
                        or binding[1] <= 0
                    ):
                        raise TypeError("agent turn generation is malformed")
                    turn_request_generations[indexed_turn_id] = binding[1]
        else:
            turn_index_accessor = vars(agent).get("active_turn_request_ids")
            if not callable(turn_index_accessor):
                turn_index_accessor = getattr(
                    type(agent),
                    "active_turn_request_ids",
                    None,
                )
                if callable(turn_index_accessor):
                    turn_request_ids = turn_index_accessor(agent)
                else:
                    turn_request_ids = {}
            else:
                turn_request_ids = turn_index_accessor()
            turn_request_generations = {}
        if not isinstance(turn_request_ids, dict):
            raise TypeError("agent turn request inventory has an invalid type")
        turn_addresses = active_request_ids.union(turn_request_ids)

        async def cancel_request(stop_request: StopRequest) -> StopDisposition:
            cancelled_request_ids: list[Optional[str]] = []
            if stop_request.scope is StopScope.TURN:
                cancel_kwargs = {"request_id": stop_request.target}
                if stop_request.request_generation is not None:
                    cancel_kwargs["generation"] = stop_request.request_generation
                canceled = agent.cancel_current_request(**cancel_kwargs)
                if canceled:
                    cancelled_request_ids.append(stop_request.target)
                else:
                    # The matching invoke/stream may have been dispatched by
                    # the client but not yet reached lifecycle registration.
                    # Fence that exact ID briefly; registration consumes the
                    # tombstone before cognition can begin.  Unknown IDs keep
                    # the historical ALREADY_COMPLETE result.
                    reserve = getattr(
                        type(agent),
                        "reserve_request_cancellation",
                        None,
                    )
                    if (
                        stop_request.request_generation is None
                        and callable(reserve)
                    ):
                        reserve(agent, stop_request.target)
            else:
                canceled = False
                for active_request_id in sorted(active_request_ids):
                    request_cancelled = agent.cancel_current_request(
                        request_id=active_request_id
                    )
                    if request_cancelled:
                        cancelled_request_ids.append(active_request_id)
                    canceled = request_cancelled or canceled
                if not active_request_ids:
                    canceled = agent.cancel_current_request(request_id=None)
                    if canceled:
                        cancelled_request_ids.append(None)
            if canceled:
                wait_for_completion = getattr(
                    agent,
                    "wait_for_request_completion",
                    None,
                )
                if not callable(wait_for_completion):
                    raise RuntimeError(
                        "agent cannot confirm request lifecycle completion"
                    )
                # Every cancellation marker is installed before the first
                # await, so agent-wide Stop reaches all snapshotted turns at
                # once. STOPPED is returned only after each one has run its
                # endpoint cleanup; CancellationAuthority bounds this wait.
                abandoned = False
                for cancelled_request_id in cancelled_request_ids:
                    wait_kwargs = {}
                    if (
                        stop_request.scope is StopScope.TURN
                        and stop_request.request_generation is not None
                    ):
                        wait_kwargs["generation"] = (
                            stop_request.request_generation
                        )
                    completion_disposition = await wait_for_completion(
                        cancelled_request_id,
                        **wait_kwargs,
                    )
                    abandoned = abandoned or (
                        completion_disposition
                        is RequestCompletionDisposition.ABANDONED
                    )
                if abandoned:
                    return StopDisposition.UNREACHABLE
            return (
                StopDisposition.STOPPED
                if canceled
                else StopDisposition.ALREADY_COMPLETE
            )

        cleanup_registry = getattr(
            request.app.state,
            "stop_cleanup_registry",
            None,
        )
        if cleanup_registry is None:
            cleanup_registry = StopCleanupRegistry()
            request.app.state.stop_cleanup_registry = cleanup_registry
        elif not isinstance(cleanup_registry, StopCleanupRegistry):
            raise TypeError("app Stop cleanup registry has an invalid type")

        authority = CancellationAuthority(
            lambda: (
                CooperativeStopTarget(
                    target_id=agent_id,
                    agent_id=agent_id,
                    cancel=cancel_request,
                    turn_ids=frozenset(turn_addresses),
                    turn_request_ids=turn_request_ids,
                    turn_request_generations=turn_request_generations,
                ),
            ),
            cleanup_registry=cleanup_registry,
        )
        stop_request = StopRequest(
            scope=(
                StopScope.TURN
                if request_id is not None or turn_id is not None
                else StopScope.AGENT
            ),
            actor_id=actor_id,
            target=(
                turn_id
                if turn_id is not None
                else request_id if request_id is not None else agent_id
            ),
            target_agent_id=(
                agent_id
                if request_id is not None or turn_id is not None
                else None
            ),
            target_is_turn_id=turn_id is not None,
        )
        outcomes = await authority.stop(stop_request)
        failed_outcomes = tuple(
            outcome
            for outcome in outcomes
            if outcome.disposition
            in {StopDisposition.REFUSED, StopDisposition.UNREACHABLE}
        )
        if failed_outcomes:
            raise ApiHTTPException(
                status_code=503,
                code="stop_not_confirmed",
                message="Cooperative Stop could not be confirmed.",
                details=[outcome.to_dict() for outcome in failed_outcomes],
            )
        cancelled = any(
            outcome.disposition is StopDisposition.STOPPED for outcome in outcomes
        )
        return {
            "success": True,
            "cancelled": cancelled,
            "request_id": request_id,
            "turn_id": turn_id,
            "message": "Request cancelled" if cancelled else "No active request to cancel",
            "stop_outcomes": [outcome.to_dict() for outcome in outcomes],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error stopping agent.")


def _audit_status(agent) -> dict:
    """Derive response-audit state from the real ResponseAuditFeature hook.

    The previous ``audit_enabled`` boolean was a dead flag the audit hook
    never consulted (#2034). Report the live hook's mode/registration so
    the status API and the on/off switch share one state machine.
    """
    features = getattr(agent, "features", None) or {}
    feature = features.get("ResponseAuditFeature")
    if feature is None:
        return {"available": False, "mode": "skip", "hook_registered": False}
    hook = getattr(feature, "_hook", None)
    hook_registered = hook is not None and getattr(hook, "enabled", False)
    return {
        "available": True,
        "mode": getattr(feature, "_mode", "skip"),
        "hook_registered": hook_registered,
    }


@router.get("/info")
async def get_agent_info(request: Request):
    """Get agent information including DID and privacy mode."""
    try:
        agent = get_agent(request)
        return {
            "agent_id": agent.agent_id,
            "privacy_mode": agent.privacy_mode.value if hasattr(agent.privacy_mode, 'value') else str(agent.privacy_mode),
            "features": list(agent.features.keys()) if hasattr(agent, 'features') else [],
            "audit": _audit_status(agent),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving agent info.")


@router.get("/privacy-mode")
async def get_privacy_mode(request: Request):
    """Get current privacy mode."""
    try:
        agent = get_agent(request)
        mode = agent.privacy_mode
        privacy_agent = getattr(agent, "privacy_agent", None)
        return {
            "privacy_mode": mode.value if hasattr(mode, 'value') else str(mode),
            "allows_cloud_llm": privacy_agent.privacy_config.allows_cloud_llm() if privacy_agent else True,
            "allows_storage": privacy_agent.can_store("conversation") if privacy_agent else True,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving privacy mode.")


@router.post(
    "/privacy-mode",
    dependencies=[Depends(enforce_destructive_op)],
)
async def set_privacy_mode(request: Request):
    """Set privacy mode.

    Gated by the demo-isolation rail (#766 / #867).  A privacy-mode flip is
    destructive in practice — flipping a live agent into EPHEMERAL means the
    next exit triggers the leak-purge, which can hard-DELETE rows the agent
    didn't author during the session if the leak-purge isn't scoped (see
    #867 for the wipe that prompted this gate).  On a live agent the rail
    therefore requires the ``X-Kestrel-Allow-Destructive`` header so a
    stray script can't change a live agent's privacy contract by accident.
    """
    try:
        from kestrel_sovereign.privacy import PrivacyMode, privacy_mode_to_config

        data = await request.json()
        mode_str = data.get("mode", "").upper()

        try:
            new_mode = PrivacyMode[mode_str]
        except KeyError:
            valid_modes = [m.name for m in PrivacyMode]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid privacy mode '{mode_str}'. Valid modes: {valid_modes}"
            )

        agent = get_agent(request)
        transition = None
        if getattr(type(agent), "set_privacy_mode_with_effects", None):
            transition = await agent.set_privacy_mode_with_effects(new_mode)
        else:
            await agent.set_privacy_mode(new_mode)

        # Data-destructive downgrade staged pending confirmation: nothing changed,
        # so report the pending state (NOT success) with the warning. The client
        # confirms via POST /privacy-mode/confirm (or the !confirm-privacy-mode
        # command). Reporting success here is exactly the split-state bug.
        if getattr(transition, "requires_confirmation", False):
            return {
                "success": False,
                "requires_confirmation": True,
                "pending_mode": transition.pending_mode,
                "mode": agent.privacy_mode.value,
                "message": transition.message,
            }

        if getattr(transition, "retryable_conflict", False):
            raise _privacy_transition_conflict()

        # An EPHEMERAL exit was REFUSED because a required no-trace purge sweep
        # failed (#2673). Nothing flipped — the agent stayed in EPHEMERAL — so we
        # must report the ACTUAL (unchanged) mode and failure, never success.
        # Reporting success here would let the agent claim a transition that did
        # not happen.
        if transition is not None and not getattr(transition, "applied", True):
            return {
                "success": False,
                "purge_failed": getattr(transition, "purge_failed", False),
                "mode": agent.privacy_mode.value,
                "message": transition.message,
            }

        # If switching to a local-only mode, auto-switch model to a local provider
        # If switching back to cloud-allowed mode, restore the previous model
        config = privacy_mode_to_config(new_mode)
        model_switched = getattr(transition, "model_switched", None)
        if transition is None and hasattr(agent, 'llm_service') and agent.llm_service:
            llm = agent.llm_service
            if not config.allows_cloud_llm():
                # Save the resolved active cloud selection before overriding to local,
                # so we can restore it when privacy allows cloud again.
                current_pref = llm.get_model_preference() or {}
                current_vendor = current_pref.get("vendor")
                current_model = current_pref.get("model")
                current_route = current_pref.get("route")
                if not current_model and getattr(llm, "providers", None):
                    first = llm.providers[0]
                    current_vendor = first.get("vendor")
                    current_model = first.get("model")
                    current_route = first.get("route")
                if current_model and not (
                    next((p for p in llm.providers if p.get("vendor") == current_vendor and p.get("is_local")), None)
                ):
                    llm._pre_ephemeral_preference = {
                        "vendor": current_vendor,
                        "model": current_model,
                        "route": current_route,
                    }

                local_routes = [p for p in llm.providers if p.get("is_local")]
                # Prefer ollama over llama_cpp — ollama is more universally available
                local_route = next(
                    (p for p in local_routes if p.get("vendor") == "ollama"),
                    local_routes[0] if local_routes else None,
                )
                if local_route:
                    llm.set_model_preference(
                        local_route["model"], local_route.get("vendor"), local_route.get("route")
                    )
                    model_switched = {
                        "vendor": local_route.get("vendor"),
                        "route": local_route.get("route"),
                        "model": local_route["model"],
                    }
            elif config.allows_cloud_llm():
                # Restore previous cloud preference if we saved one
                saved = getattr(llm, '_pre_ephemeral_preference', None)
                if saved:
                    llm.set_model_preference(
                        saved.get("model", ""),
                        saved.get("vendor"),
                        saved.get("route"),
                    )
                    model_switched = saved
                    llm._pre_ephemeral_preference = None

        # Auto-switch voice providers if VoiceFeature is active
        voice_switched = getattr(transition, "voice_switched", None)
        biometric_warning = getattr(transition, "biometric_warning", None)
        features = getattr(agent, "features", {})
        vf = features.get("VoiceFeature") if features else None
        if transition is None and vf and hasattr(vf, "on_privacy_mode_changed"):
            try:
                voice_switched = await vf.on_privacy_mode_changed()
            except Exception as ve:
                logger.warning("Voice auto-switch failed: %s", ve)

            # Biometric warning when switching TO a mode that allows cloud voice
            if config.allows_cloud_llm() and hasattr(vf, "biometric_warning"):
                # Only warn if there are cloud voice providers configured
                vc = getattr(vf, "_voice_config", None)
                if vc and (vc.tts_provider or vc.stt_provider):
                    biometric_warning = vf.biometric_warning()

        return {
            "success": True,
            "mode": new_mode.value,
            "message": f"Privacy mode set to {new_mode.value}",
            "allows_cloud_llm": config.allows_cloud_llm(),
            "model_switched": model_switched,
            "voice_switched": voice_switched,
            "biometric_warning": biometric_warning,
        }
    except HTTPException:
        raise
    except PrivacyViolationError:
        # Never interpolate the exception: storage/provider details are not
        # part of the public response or operator log contract.
        logger.info("Privacy mode change deferred by an active fact operation")
        raise _privacy_transition_conflict()
    except Exception as e:
        logger.error(f"Error setting privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error setting privacy mode.")


@router.post(
    "/privacy-mode/confirm",
    dependencies=[Depends(enforce_destructive_op)],
)
async def confirm_privacy_mode(request: Request):
    """Confirm a privacy-mode change that was staged pending confirmation.

    The counterpart to a ``requires_confirmation`` response from
    ``POST /privacy-mode``. Applies the staged (data-destructive) transition
    atomically; a no-op with an explanatory message if nothing is pending.
    Behind the same demo-isolation rail as the change it confirms.
    """
    try:
        agent = get_agent(request)
        if not getattr(type(agent), "confirm_privacy_transition", None):
            raise HTTPException(status_code=400, detail="Agent does not support staged privacy transitions.")
        result = await agent.confirm_privacy_transition()
        if getattr(result, "retryable_conflict", False):
            raise _privacy_transition_conflict()
        # applied is False for a no-op confirm (nothing was pending) as well as
        # for a staged result — so a stale/double-click confirm reports success
        # False instead of masquerading as an applied transition.
        applied = getattr(result, "applied", False)
        return {
            "success": applied,
            "applied": applied,
            "mode": agent.privacy_mode.value,
            "message": result.message,
            "allows_cloud_llm": result.allows_cloud_llm,
            "model_switched": result.model_switched,
            "voice_switched": result.voice_switched,
            "biometric_warning": result.biometric_warning,
        }
    except HTTPException:
        raise
    except PrivacyViolationError:
        logger.info(
            "Privacy mode confirmation deferred by an active fact operation"
        )
        raise _privacy_transition_conflict()
    except Exception as e:
        logger.error(f"Error confirming privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error confirming privacy mode.")


@router.post("/privacy-mode/cancel")
async def cancel_privacy_mode(request: Request):
    """Discard a privacy-mode change staged pending confirmation.

    The counterpart to declining a ``requires_confirmation`` response from
    ``POST /privacy-mode``: drops the staged (data-destructive) transition so a
    later confirm — from another tab or the ``!confirm-privacy-mode`` command —
    can't apply a change the user declined. A no-op if nothing is pending. Not
    destructive (it only clears a pending intent), so no demo-isolation gate.
    """
    try:
        agent = get_agent(request)
        if not getattr(type(agent), "cancel_privacy_transition", None):
            # Nothing to cancel on an agent without staged transitions.
            return {"success": True, "mode": agent.privacy_mode.value,
                    "message": "No pending privacy-mode change to cancel."}
        result = await agent.cancel_privacy_transition()
        return {
            "success": True,
            "mode": agent.privacy_mode.value,
            "message": result.message,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling privacy mode: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error cancelling privacy mode.")


@router.get("/notifications")
async def get_notifications(request: Request):
    """
    Get and clear pending task completion notifications.

    This is a polling endpoint - call it periodically to check for
    notifications about completed background tasks.
    """
    try:
        agent = get_agent(request)
        notifications = agent.get_pending_notifications()
        return {
            "notifications": notifications,
            "count": len(notifications)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting notifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving notifications.")


@router.get("/notifications/sse")
@limiter.limit("30/minute")
async def notifications_sse(request: Request):
    """
    Server-Sent Events endpoint for real-time task notifications.

    Clients connect and receive events as background tasks complete.
    Events are formatted as:
        event: task_notification
        data: {"message": "...", "type": "completed|failed|canceled"}

    Also sends periodic keepalive pings every 15 seconds.

    Connection limits: max MAX_SSE_CONNECTIONS_PER_CLIENT concurrent connections
    per client IP to prevent resource exhaustion.
    """
    import json

    # Validate agent is available before starting SSE stream
    agent = get_agent(request)

    # Enforce per-client, per-agent SSE connection limit
    client_ip = request.client.host if request.client else "unknown"
    agent_id = getattr(agent, 'agent_id', 'default')
    conn_key = (client_ip, agent_id)
    async with _sse_lock:
        if _sse_connections[conn_key] >= MAX_SSE_CONNECTIONS_PER_CLIENT:
            raise HTTPException(
                status_code=429,
                detail=f"Too many SSE connections (limit: {MAX_SSE_CONNECTIONS_PER_CLIENT})"
            )
        _sse_connections[conn_key] += 1

    async def event_generator():
        """Generate SSE events for task notifications and agent event bus."""
        agent = get_agent(request)

        # Forward events from agent.emit_event (e.g. approval_request) to this
        # stream. Without this listener, SecurityFeature._emit_approval_request
        # fires into an empty _event_listeners list and approval popups never
        # reach the browser. See #748.
        event_queue: asyncio.Queue = asyncio.Queue()

        async def _forward(event_type: str, data):
            await event_queue.put((event_type, data))

        listener_registered = False
        if hasattr(agent, "add_event_listener"):
            agent.add_event_listener(_forward)
            listener_registered = True

        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'status': 'connected'})}\n\n"

            # Replay any events buffered while no listener was connected —
            # e.g. the restart `completed` status emitted from
            # feature.initialize() during host startup, before this
            # reconnect landed (#1551). Drained once; registering the
            # listener above first means any event emitted from here on
            # goes to the queue instead, so nothing is dropped or doubled.
            if hasattr(agent, "get_pending_events"):
                for ev_type, ev_data in agent.get_pending_events():
                    yield f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n"

            # Replay sticky current-state aux events (e.g. a channel pairing QR,
            # #1825) so EVERY new chat session shows them, not only the client
            # connected when they were produced. Not drained — they persist
            # until cleared (e.g. on link success).
            if hasattr(agent, "get_sticky_events"):
                for ev_type, ev_data in agent.get_sticky_events():
                    yield f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n"

            ping_interval = SSE_PING_INTERVAL_SECONDS
            # Wave 5C: revising events are time-sensitive — the chat
            # UI uses them to retract pre-tool prose BEFORE post-tool
            # synthesis chunks land on the parallel /api/agent/stream
            # channel. The previous polling interval of 500ms could
            # cause SSE delivery to race the chat-stream chunks. We
            # block on event_queue.get() with a short timeout instead
            # so emit_event-sourced events deliver sub-frame; the
            # timeout still drives the legacy task-notification poll
            # and the keepalive ping. Codex P2 of #1084.
            task_poll_interval = 0.5
            last_ping = time.monotonic()
            last_task_poll = 0.0

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected")
                    break

                # Block waiting for an event, but bound the wait so
                # task notifications + ping still fire on schedule.
                try:
                    event_type, data = await asyncio.wait_for(
                        event_queue.get(), timeout=0.1,
                    )
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    # Drain any other events queued behind the first.
                    while True:
                        try:
                            event_type, data = event_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    pass

                current_time = time.monotonic()

                # Task-completion notifications come from a polling
                # source (agent.get_pending_notifications), not the
                # event queue. Keep them on the slower interval.
                if current_time - last_task_poll >= task_poll_interval:
                    notifications = agent.get_pending_notifications()
                    for notification in notifications:
                        if notification.startswith("✅"):
                            notif_type = "completed"
                        elif notification.startswith("❌"):
                            notif_type = "failed"
                        elif notification.startswith("⚠️"):
                            notif_type = "canceled"
                        else:
                            notif_type = "info"

                        event_data = json.dumps({
                            "message": notification,
                            "type": notif_type,
                        })
                        yield f"event: task_notification\ndata: {event_data}\n\n"
                    last_task_poll = current_time

                # Send keepalive ping
                if current_time - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'time': current_time})}\n\n"
                    last_ping = current_time

        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'Internal server error'})}\n\n"
        finally:
            if listener_registered:
                agent.remove_event_listener(_forward)
            async with _sse_lock:
                _sse_connections[conn_key] -= 1
                if _sse_connections[conn_key] <= 0:
                    del _sse_connections[conn_key]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def _codex_thread_occupancy(
    agent: Any, session_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Best-effort: codex's TRUE server-side thread occupancy for the active
    openai:plan session, or ``None``.

    On openai:plan, codex holds the conversation thread server-side while
    Kestrel sends only incremental turns — so the per-turn payload the
    context monitor measures is NOT the number that fills the window and
    trips codex's lossy auto-compaction (#1844). Read the real occupancy off
    the CodexAdapter (``get_thread_occupancy``) so the monitor can surface
    it and a low per-turn reading can't masquerade as a low context.

    Gated to the RESOLVED active provider — NOT the display model string.
    Using ``resolve_provider_routing`` (the single source of truth every
    turn funnels through) means a route-less ``openai/<model>`` preference
    that actually executes on openai:plan still surfaces occupancy (codex
    review round 3), while a switch AWAY from plan drops the CodexAdapter
    from the resolved primary so a stale snapshot can't leak (round 2).
    """
    if not session_id:
        return None
    llm = getattr(agent, "llm_service", None)
    if llm is None:
        return None
    # The primary (first) resolved provider is the one that will actually
    # serve the next turn; fallbacks behind it don't define current context.
    resolver = getattr(llm, "resolve_provider_routing", None)
    providers: list = []
    if callable(resolver):
        try:
            providers, _ = resolver()
        except Exception:
            providers = []
    primary = providers[0] if providers else None
    adapter = primary.get("adapter") if isinstance(primary, dict) else None
    getter = getattr(adapter, "get_thread_occupancy", None)
    if not callable(getter):
        # Active route isn't the CodexAdapter (only it exposes this) — no
        # server-side occupancy to report.
        return None
    try:
        return getter(session_id) or None
    except Exception:
        return None


@router.get("/context-status")
async def get_context_status(
    request: Request,
    session_id: Optional[str] = Query(None, description="Session ID to get context for"),
    full: bool = Query(
        False,
        description=(
            "When True, run the full breakdown including RAG retrieval. "
            "The frequent footer poll passes False (cheap path); the "
            "breakdown popup (#1310) passes True on open."
        ),
    ),
):
    """Thin HTTP wrapper over :func:`compute_context_status`.

    Shared with the agent ``context`` tool so the chat-footer pill and the
    agent's own self-report cannot drift (#1969).
    """
    return await compute_context_status(get_agent(request), session_id, full=full)


@dataclass
class _ContextStatusMeasurement:
    """Data acquired for context-status before UI/route-cap shaping."""

    history: List[Dict[str, Any]]
    model_identity: Dict[str, Optional[str]]
    current_model: str
    context_limit: int
    breakdown: Dict[str, Any]


async def _acquire_context_status_measurement(
    agent: Any,
    session_id: str,
    *,
    full: bool,
) -> _ContextStatusMeasurement:
    """Acquire one dry-run production plan for a session."""

    from kestrel_sovereign.agent.token_counter import get_token_counter
    from kestrel_sovereign.agent.context_manager import CONTEXT_HISTORY_LIMIT

    privacy_agent = getattr(agent, "privacy_agent", None)
    history_reader = getattr(
        privacy_agent, "get_conversation_history", None
    )
    if (
        callable(history_reader)
        and not type(privacy_agent).__module__.startswith("unittest.mock")
    ):
        # This is the live turn's acquisition path. It is load-bearing for
        # ISOLATED/EPHEMERAL buffers, which do not live in persistent storage.
        history = await history_reader(
            limit=CONTEXT_HISTORY_LIMIT,
            session_id=session_id,
        )
    else:
        # Compatibility for partial endpoint fixtures and older hosts.
        history = await agent.storage.get_conversation_history(
            limit=CONTEXT_HISTORY_LIMIT,
            session_id=session_id,
        )
    model_identity = _latest_assistant_model_identity(history)
    current_model = model_identity["context_model"] or LEGACY_CONTEXT_MODEL
    counter = get_token_counter(current_model)
    context_limit = counter.get_context_limit()

    constitution_text = ""
    get_const = getattr(agent, "_get_governing_constitution", None)
    is_real_governing_getter = (
        callable(get_const)
        and not type(get_const).__module__.startswith("unittest.mock")
    )
    if not is_real_governing_getter:
        # Compatibility for partial endpoint fixtures and pre-mixin hosts.
        get_const = getattr(agent, "get_constitution", None)
    if callable(get_const):
        try:
            got = (
                get_const(allow_lazy_anchor=False)
                if is_real_governing_getter
                else get_const()
            )
            constitution_text = await got if hasattr(got, "__await__") else got
            constitution_text = constitution_text or ""
        except Exception as exc:
            logger.debug("constitution fetch failed for breakdown: %s", exc)
            if is_real_governing_getter:
                raise RuntimeError(
                    "governing constitution is unavailable for context "
                    "measurement"
                ) from exc
    if (
        isinstance(constitution_text, str)
        and constitution_text.lstrip().startswith("Error:")
    ):
        raise RuntimeError(
            "governing constitution is unavailable for context measurement"
        )
    if is_real_governing_getter and not str(constitution_text).strip():
        raise RuntimeError(
            "governing constitution is unavailable for context measurement"
        )

    tool_schemas: Optional[List[Dict[str, Any]]] = None
    build_tools = getattr(agent, "_build_all_tools", None)
    if not callable(build_tools):
        registry = getattr(agent, "tool_registry", None)
        build_tools = getattr(registry, "_build_all_tools", None)
    if callable(build_tools):
        try:
            tool_schemas = list(build_tools())
        except Exception as exc:
            logger.debug("tool schema fetch failed for breakdown: %s", exc)

    query = ""
    try:
        from kestrel_sovereign.agent.context_builder import (
            extract_raw_user_content,
        )

        for row in reversed(history):
            if (row.get("role") or "").lower() == "user":
                query = extract_raw_user_content(row.get("content", "") or "")
                break
    except Exception as exc:
        logger.debug("last-user-query lookup failed for breakdown: %s", exc)

    context_manager = getattr(agent, "context_manager", None)
    plan_builder = getattr(context_manager, "build_context_plan", None)
    is_real_plan_builder = (
        callable(plan_builder)
        and not type(context_manager).__module__.startswith("unittest.mock")
    )
    if is_real_plan_builder:
        from kestrel_sovereign.agent.context_stages import ContextBuildMode

        privacy_mode = getattr(agent, "_privacy_mode", None)
        if privacy_mode is None:
            privacy_agent = getattr(agent, "privacy_agent", None)
            privacy_mode = getattr(privacy_agent, "privacy_mode", "NORMAL")
        privacy_mode = getattr(
            privacy_mode, "name", getattr(privacy_mode, "value", privacy_mode)
        )

        reflection_guidance = None
        features = getattr(agent, "features", None)
        reflection_feature = (
            features.get("ReflectionFeature")
            if isinstance(features, dict)
            else None
        )
        if reflection_feature is not None:
            getter = getattr(reflection_feature, "get_active_guidance", None)
            if callable(getter):
                try:
                    reflection_guidance = await getter()
                except Exception as exc:
                    logger.debug(
                        "reflection guidance fetch failed for breakdown: %s",
                        exc,
                    )

        plan = await plan_builder(
            query=query,
            constitution=constitution_text,
            include_briefing=not bool(
                getattr(agent, "_session_briefed", False)
            ),
            include_memories=True,
            include_rag=True,
            privacy_mode=str(privacy_mode or "NORMAL").upper(),
            conversation_history=history,
            reflection_guidance=reflection_guidance,
            tools=tool_schemas,
            mode=ContextBuildMode.DRY_RUN,
            measure_expensive_sections=full,
        )
        breakdown = plan.to_breakdown()
        context_limit = int(breakdown["context_limit"])
        current_model = str(breakdown.get("model") or current_model)
    else:
        # Compatibility for partial/legacy agent test doubles. Production
        # KestrelAgent always owns ContextManager.build_context_plan.
        from kestrel_sovereign.agent.context_builder import ContextBuilder

        agent_builder = getattr(agent, "context_builder", None)
        builder = ContextBuilder(
            storage=agent.storage,
            model=current_model,
            consolidator=getattr(agent_builder, "consolidator", None),
            agent_data_path=getattr(agent_builder, "agent_data_path", None),
        )
        memory_retriever = None
        if full:
            from kestrel_sovereign.agent.context_manager import _retrieval_config

            memory_min_score = _retrieval_config().get("memory_min_score")
            memory_manager = getattr(context_manager, "memory_manager", None)
            retrieve_memories = getattr(memory_manager, "retrieve_memories", None)
            if callable(retrieve_memories):
                async def memory_retriever(
                    query: str, max_tokens: int
                ) -> Optional[str]:
                    return await retrieve_memories(
                        query=query,
                        max_tokens=max_tokens,
                        counter=counter,
                        read_only=True,
                        min_score=memory_min_score,
                    )

        breakdown = await builder.measure_context_breakdown(
            query=query,
            history=history,
            constitution=constitution_text,
            include_briefing=True,
            message_count=len(history),
            tools=tool_schemas,
            include_rag=full,
            memory_retriever=memory_retriever,
        )
        breakdown.pop("_artifacts", None)
        for section_name in ("memories", "rag"):
            row = breakdown.get("sections", {}).get(section_name)
            if isinstance(row, dict) and (
                row.get("status") in {"unknown", "skipped"}
                or row.get("measured") is False
            ):
                row["tokens"] = None

    if full and not query:
        rag_section = breakdown.get("sections", {}).get("rag")
        if isinstance(rag_section, dict):
            rag_section["query_used_label"] = (
                "estimated against latest stored chunks — no recent user "
                "turn available for query-specific retrieval"
            )

    return _ContextStatusMeasurement(
        history=history,
        model_identity=model_identity,
        current_model=current_model,
        context_limit=context_limit,
        breakdown=breakdown,
    )


async def compute_context_status(
    agent,
    session_id: Optional[str] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Render whole-window status from ContextManager's read-only plan.

    Single source of truth for BOTH the chat-footer pill (via the HTTP route
    above) AND the agent ``context_status`` tool (#1969). Before this, the tool
    used a separate cross-session, raw-content token count and drifted from the
    production context path.

    The pill in the chat footer (chat.js) reads ``utilization_percent``
    and renders the ● N msgs · X% indicator. The popup (#1310) reads
    ``breakdown`` for the layered taxonomy. Both render the typed,
    side-effect-free plan from ``ContextManager.build_context_plan``; live
    turns commit that same plan before rendering it. The ``breakdown`` field
    contains the canonical sections (system, tools,
    history, episodes, memories, rag, dynamic_context_overhead) plus
    the elastic-budget snapshot from #1309.

    Two modes:

    - ``full=False`` (default): cheap path for the frequent footer
      poll. Memory/RAG acquisition is omitted and those sections are explicitly
      ``unknown``/``skipped`` with no token value.
    - ``full=True``: invoked once when the popup opens. The production
      relevance gates and retrieval path execute read-only.
    """
    try:
        from kestrel_sovereign.agent.token_counter import get_token_counter
        from kestrel_sovereign.agent.token_budget import RESPONSE_RESERVE

        # 1. No active session → return an idle shape.  Previously passing
        # session_id=None into get_conversation_history leaked the agent's
        # cross-session aggregate count and falsely rolled utilization to
        # 100%, which surfaced in the chat footer as "472 msgs · 100%
        # Compact" on an empty pane.  See #713.  "Context window status"
        # is only meaningful for an active conversation; with none, there's
        # nothing to report.
        if not session_id:
            current_model = agent.get_current_model()
            counter = get_token_counter(current_model)
            context_limit = counter.get_context_limit()
            return {
                "model": current_model,
                "provider": None,
                "context_model": current_model,
                "model_source": "current_preference",
                "message_count": 0,
                "total_tokens": 0,
                "context_limit": context_limit,
                "response_reserve": RESPONSE_RESERVE,
                "total_budget": context_limit - RESPONSE_RESERVE,
                "utilization_percent": 0.0,
                "compaction_recommended": False,
                "status": "idle",
                "warnings": [],
                "breakdown": None,
                "route_cap": None,
                "silently_pruned_path_active": False,
            }

        measurement = await _acquire_context_status_measurement(
            agent,
            session_id,
            full=full,
        )
        history = measurement.history
        message_count = len(history)
        model_identity = measurement.model_identity
        current_model = measurement.current_model
        context_limit = measurement.context_limit
        breakdown = measurement.breakdown

        # C / #1311: attach salvage-state counts so the popup can
        # render the layered taxonomy (pointer-only / pending-fold /
        # folded / failed-fold) and surface back-pressure warnings.
        # Best-effort — failure to load counts must not break the
        # endpoint, just degrade the popup's salvage row to zeros.
        try:
            from kestrel_sovereign.agent.salvage import (
                DEFAULT_PENDING_WARN_THRESHOLD,
                get_salvage_state_counts,
            )
            conv_store_for_counts = (
                getattr(agent.conversation_manager, "_get_conversation_store", lambda: None)()
                if hasattr(agent, "conversation_manager")
                else None
            )
            if conv_store_for_counts is not None and "sections" in breakdown:
                salvage_counts = await get_salvage_state_counts(
                    conv_store_for_counts, session_id=session_id
                )
                hist_section = breakdown["sections"].get("history")
                if isinstance(hist_section, dict):
                    hist_section["salvages"] = salvage_counts
                    hist_section["salvages"]["warn_threshold"] = (
                        DEFAULT_PENDING_WARN_THRESHOLD
                    )
        except Exception as e:
            logger.debug(f"salvage counts fetch failed for breakdown: {e}")

        # 5. Pill % = projected whole-window utilization (the design's
        # core correctness fix: previously the pill reported history
        # slice utilization, which was misleading whenever other
        # sections dominated). Greenfield — no compat constraint
        # (Emma's 2026-05-20 review: "make the number correct").
        utilization_percent = float(breakdown["utilization_percent"])
        total_measured = int(breakdown["total_measured"])
        total_budget = int(breakdown["total_budget"])

        # 6. Status + warnings keyed off the whole-window figure.
        warnings: List[str] = []
        if utilization_percent < 50:
            status_str = "healthy"
        elif utilization_percent < 70:
            status_str = "normal"
        elif utilization_percent < 85:
            status_str = "warning"
            warnings.append(
                f"Context window {utilization_percent:.0f}% full - "
                "consider !compact to save older turns into a durable summary"
            )
        else:
            status_str = "critical"
            warnings.append(
                f"Context window {utilization_percent:.0f}% full - "
                "compaction strongly recommended"
            )

        # 7. Surface the plan's declared salvage disposition. This is true when
        # automatic salvage is disabled or when the projected pruned span
        # contains id-less/in-memory rows that no durable marker can link.
        salvage_projection = breakdown.get("salvage", {})
        silently_pruned_path_active = bool(
            salvage_projection.get("silent_prune_possible", False)
        )

        # #1503: route per-turn cap visibility. Some subscription tiers
        # (notably ChatGPT-Plus on ``openai:plan``) enforce a per-turn
        # payload cap well below the model's full context window. Pure
        # whole-window utilization is misleading on those routes — a
        # session at 3 % on a 256K model can still bust a 32768-token
        # route cap. Surface the cap so the UI can show binding
        # headroom before the turn fires (catches the over-cap failure
        # mode handled reactively by #1395 / #1410).
        route_cap_block: Optional[Dict[str, Any]] = None
        try:
            from kestrel_sovereign.llm.model_catalog import get_catalog_service
            catalog = get_catalog_service()
            cap_tokens = catalog.get_route_context_cap(current_model)
            if isinstance(cap_tokens, int) and cap_tokens > 0:
                projected = max(0, int(total_measured))
                cap_util_percent = (
                    (projected / cap_tokens) * 100.0 if cap_tokens else 0.0
                )
                # Resolve the route key the cap applied to so the UI
                # can show its name. Use the catalog's matched-route
                # helper, which spans ALL precedence layers (env var,
                # discovered, file) — the previous local loop only
                # scanned the file layer, so env-only / discovered-only
                # deployments showed ``route: null`` and the knob hint
                # vanished (codex round 2 P3 on the dynamic-cap PR).
                route_id: Optional[str] = None
                helper = getattr(catalog, "get_matched_route_cap_key", None)
                if callable(helper):
                    try:
                        candidate = helper(current_model)
                        # Validate the return is a real string — a mocked
                        # catalog (MagicMock) auto-returns a child mock
                        # that's truthy but not a route name, so guard
                        # explicitly rather than smuggling the mock
                        # through into the response shape.
                        if isinstance(candidate, str) and candidate:
                            route_id = candidate
                    except Exception:
                        route_id = None
                if route_id is None:
                    # Fall back to the legacy file-layer scan when the
                    # helper is absent or returned no real match. This
                    # also catches mocked-catalog test paths cleanly.
                    for known_route in getattr(catalog, "_route_context_caps", {}):
                        if (
                            current_model.lower() == known_route.lower()
                            or current_model.lower().startswith(
                                known_route.lower() + "/"
                            )
                        ) and (
                            route_id is None or len(known_route) > len(route_id)
                        ):
                            route_id = known_route
                # Operator knob hint per route (best-effort). ``openai:plan``
                # uses ``KESTREL_OPENAI_PLAN_CONTEXT_CAP`` (#1395 wiring);
                # other routes leave the knob hint empty.
                knob = (
                    "KESTREL_OPENAI_PLAN_CONTEXT_CAP"
                    if route_id == "openai:plan"
                    else None
                )
                # IMPORTANT: ``TokenCounter.get_context_limit()`` already
                # returns the route cap on capped routes (see
                # ``agent/token_counter.py:241-246``), so ``context_limit``
                # above is typically equal to ``cap_tokens`` — the
                # existing whole-window pill is therefore already
                # measuring against the route cap (modulo the response
                # reserve). The route_cap block exists so the UI can
                # NAME that cap, show the actionable knob, and report
                # raw headroom — not to provide a separate percentage
                # that would be redundant with the existing pill
                # (codex round 2 P2 on #1503).
                route_cap_block = {
                    "route": route_id,
                    "cap_tokens": cap_tokens,
                    "projected_turn_payload": projected,
                    "utilization_percent": round(cap_util_percent, 1),
                    "headroom_tokens": max(0, cap_tokens - projected),
                    "knob": knob,
                    # On the cheap footer poll (``full=False``) the
                    # breakdown was measured without RAG, so the
                    # projection is a FLOOR — the real turn payload may
                    # be higher. The popup (``full=True``) runs RAG but
                    # remains a projection. The UI uses this flag
                    # to label the pill / popup honestly (codex round 1
                    # P2 on #1503).
                    "includes_rag": bool(full) and (
                        breakdown.get("sections", {})
                        .get("rag", {})
                        .get("status")
                        not in {"unknown", "skipped"}
                    ),
                }
        except Exception as e:
            # Catalog probe must never break the endpoint — degrade to
            # "no route cap surface" rather than 500ing the footer poll.
            logger.debug(f"route_cap probe failed for breakdown: {e}")

        # #1844: codex's TRUE server-side thread occupancy on openai:plan.
        # ``total_measured`` above is Kestrel's incremental per-turn payload;
        # on this route codex accumulates the full thread server-side and
        # auto-compacts (lossily) when IT fills — which the per-turn number
        # can't see. Surface the real occupancy so a low reading can't
        # masquerade as a healthy context. ``None`` off-route / when unknown.
        codex_thread_block: Optional[Dict[str, Any]] = None
        try:
            codex_thread_block = _codex_thread_occupancy(agent, session_id)
        except Exception as e:  # never break the footer poll
            logger.debug(f"codex thread occupancy probe failed: {e}")

        return {
            "model": model_identity["model"],
            "provider": model_identity["provider"],
            "context_model": current_model,
            "model_source": model_identity["model_source"],
            "message_count": message_count,
            "total_tokens": total_measured,  # projected whole-window total
            "context_limit": context_limit,
            "response_reserve": breakdown["response_reserve"],
            "total_budget": total_budget,
            "utilization_percent": utilization_percent,
            "compaction_recommended": utilization_percent >= 70,
            "status": status_str,
            "warnings": warnings,
            # Layered breakdown the popup renders. Sections include
            # system (with subsections), tools, history (with
            # messages_kept_after_pruning + raw_tokens), episodes,
            # memories, rag, and dynamic_context_overhead.
            "breakdown": breakdown,
            # Route-level per-turn cap (#1503). ``None`` when the active
            # route declares no cap or the catalog probe failed.
            "route_cap": route_cap_block,
            # #1844: codex's TRUE server-side thread occupancy on openai:plan
            # ({used_tokens, window_tokens, occupancy_percent}), or ``None``
            # off-route / when no turn has reported usage yet. The chat
            # footer/popup should prefer this over ``utilization_percent`` on
            # openai:plan, since that figure only measures Kestrel's per-turn
            # payload — not the server-side thread that actually compacts.
            "codex_thread": codex_thread_block,
            # Plan-derived projection, not per-turn salvage commit evidence.
            "silently_pruned_path_active": silently_pruned_path_active,
        }
    except Exception as e:
        logger.error(f"Error getting context status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving context status.")


@router.get("/reflection/status")
async def reflection_status(request: Request):
    """Get reflection and self-improvement status.

    Returns scheduled reflection tasks, recent execution history,
    and health trend from training cycles.
    """
    agent = get_agent(request)

    result = {
        "sleep_hooks_active": bool(getattr(agent, "sleep_hooks", None)),
        "scheduled_tasks": [],
        "recent_executions": [],
    }

    # Get scheduled reflection tasks. SchedulerFeature.schedule_list now
    # returns a ToolResult envelope (#1061 wave 8); the .data dict still
    # carries the legacy {"tasks": [...]} shape.
    scheduler = agent.features.get("SchedulerFeature") if hasattr(agent, "features") else None
    if scheduler:
        try:
            envelope = await scheduler.schedule_list()
            scheduled = (envelope.data or {}).get("tasks", []) if envelope.data else []
            result["scheduled_tasks"] = [
                t for t in scheduled
                if t["task_name"] in ("reflect", "training_cycle")
            ]
        except Exception as e:
            logger.warning(f"Failed to get scheduled tasks: {e}")

    # Get recent reflection execution history from task_execution_log
    db = None
    if hasattr(agent, "_raw_storage") and hasattr(agent._raw_storage, "db"):
        db = agent._raw_storage.db
    if db:
        try:
            agent_id = getattr(agent, "agent_id", "") or getattr(agent, "did", "")
            rows = await db.fetchall(
                """
                SELECT tel.task_id, st.task_name, tel.status, tel.duration_ms, tel.executed_at,
                       SUBSTR(tel.result_text, 1, 500) as result_preview
                FROM task_execution_log tel
                JOIN scheduled_tasks st ON tel.task_id = st.id
                WHERE tel.agent_id = ? AND st.task_name IN ('reflect', 'training_cycle')
                ORDER BY tel.executed_at DESC LIMIT 10
                """,
                (agent_id,),
            )
            result["recent_executions"] = [
                {
                    "task_id": r[0], "task_name": r[1], "status": r[2],
                    "duration_ms": r[3], "executed_at": r[4], "result_preview": r[5],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to get execution history: {e}")

    return result


@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None, description="Filter by status: working, completed, failed, submitted, canceled"),
    limit: int = Query(50, le=100, description="Max results")
):
    """
    List background A2A tasks.

    Returns tasks managed by the agent's TaskManager.
    """
    agent = get_agent(request)

    # Check if agent has a task_manager
    if not hasattr(agent, 'task_manager') or not agent.task_manager:
        return {"tasks": [], "total": 0, "message": "TaskManager not available"}

    try:
        from kestrel_sovereign.a2a.types import TaskState

        # Parse status filter
        task_state = None
        if status:
            try:
                task_state = TaskState(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: working, completed, failed, submitted, canceled"
                )

        # Get tasks from task store
        tasks = await agent.task_manager.task_store.list_tasks(limit=limit)

        # Filter by status if provided
        if task_state:
            tasks = [t for t in tasks if t.status.state == task_state]

        # Convert to response format
        task_list = []
        for task in tasks:
            # Extract message text
            message_text = None
            if task.status.message and task.status.message.parts:
                for part in task.status.message.parts:
                    if hasattr(part, 'text'):
                        message_text = part.text
                        break

            # Extract metadata
            metadata = task.metadata or {}

            task_list.append({
                "id": task.id,
                "status": task.status.state.value,
                "message": message_text,
                "agent_id": metadata.get("agent_id"),
                "skill": metadata.get("skill"),
                "artifacts_count": len(task.artifacts) if task.artifacts else 0,
            })

        return {
            "tasks": task_list,
            "total": len(task_list)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error listing tasks.")


def _a2a_did_resolver(agent):
    """Return the DID resolver used to verify A2A sender signatures (#1673).

    Resolution is injectable so the verification *policy* stays in the
    envelope-signing module while the *topology* is the host's choice: a
    resolver attached as ``agent.a2a_did_resolver`` maps a peer DID to its DID
    document. The intended default is local same-host resolution (read peer
    agents' on-disk DID documents — no network), with federated ``did:web`` as
    an optional fetcher; neither is ever required. Until a host wires one,
    ``None`` means signatures can't be resolved, so a signed envelope is
    rejected. Unsigned envelopes still pass by default under the same-host
    shared-API-key boundary unless ``KESTREL_A2A_REQUIRE_SIGNED`` is set.
    """
    return getattr(agent, "a2a_did_resolver", None)


def _a2a_inbound_sender_authorizer(agent):
    """Return the recipient's post-verification A2A authorization seam."""
    return getattr(agent, "a2a_inbound_sender_authorizer", None)


def _a2a_inbound_scope_snapshot(agent):
    """Capture seam identities around asynchronous trust decisions."""
    from kestrel_sovereign.a2a.inbound_authorization import (
        has_a2a_inbound_scoped_policy,
    )

    return (
        _a2a_inbound_sender_authorizer(agent),
        _a2a_did_resolver(agent),
        getattr(agent, "peer_directory_router", None),
        getattr(agent, "peer_requester", None),
        has_a2a_inbound_scoped_policy(agent),
    )


def _a2a_inbound_scope_unchanged(agent, snapshot) -> bool:
    """Require the exact authorizer/router/requester objects to remain live."""
    current = _a2a_inbound_scope_snapshot(agent)
    return (
        current[0] is snapshot[0]
        and current[1] is snapshot[1]
        and current[2] is snapshot[2]
        and current[3] is snapshot[3]
        and current[4] is snapshot[4]
    )


def _a2a_sender_witness_unchanged(before, after) -> bool:
    """Compare manager-owned sender witnesses without invoking object equality."""
    return (
        before[0] == after[0]
        and before[1] is after[1]
        and before[2] is after[2]
        and before[3] == after[3]
    )


def _a2a_inbound_requires_verified_sender(agent, authorizer) -> bool:
    """Fail closed when hosted scope is present or its seam is malformed."""
    from kestrel_sovereign.a2a.inbound_authorization import (
        has_a2a_inbound_scoped_policy,
    )

    if has_a2a_inbound_scoped_policy(agent):
        return True
    if authorizer is not None:
        try:
            required = authorizer.requires_verified_sender
        except Exception:  # noqa: BLE001 - injected host policy boundary
            return True
        return (
            required is not False
            or getattr(agent, "peer_directory_router", None) is not None
            or getattr(agent, "peer_requester", None) is not None
        )
    return (
        getattr(agent, "peer_directory_router", None) is not None
        or getattr(agent, "peer_requester", None) is not None
    )


def _a2a_inbound_current_scope_is_valid(
    authorizer,
    hosted_policy=None,
) -> bool:
    """Validate the installed scoped seam without another provider await."""
    if hosted_policy is not None:
        validator = getattr(authorizer, "has_valid_policy_scope", None)
        arguments = (hosted_policy.router, hosted_policy.requester)
    else:
        validator = getattr(authorizer, "has_valid_current_scope", None)
        arguments = ()
    if not callable(validator):
        return False
    try:
        return validator(*arguments) is True
    except Exception:  # noqa: BLE001 - injected host policy boundary
        return False


async def _authorize_verified_a2a_sender(
    authorizer,
    sender_did: str,
    hosted_policy=None,
) -> bool:
    """Invoke the explicit inbound seam, requiring a literal True verdict."""
    if authorizer is None or not callable(
        getattr(authorizer, "authorize", None)
    ):
        return False
    try:
        if hosted_policy is not None:
            policy_authorize = getattr(
                authorizer,
                "authorize_with_policy",
                None,
            )
            if not callable(policy_authorize):
                return False
            result = policy_authorize(
                sender_did,
                router=hosted_policy.router,
                requester=hosted_policy.requester,
            )
        else:
            result = authorizer.authorize(sender_did)
        if inspect.isawaitable(result):
            result = await result
    except Exception:  # noqa: BLE001 - injected host policy boundary
        logger.warning(
            "Inbound A2A sender authorization provider failed",
            exc_info=True,
        )
        return False
    return result is True


def _a2a_replay_store(agent):
    """Return a cached shared replay-nonce store for signed A2A envelopes."""
    existing = getattr(agent, "_a2a_replay_nonce_store", None)
    if existing is not None:
        return existing

    raw_storage = getattr(agent, "_raw_storage", None)
    db = getattr(raw_storage, "db", None)
    if db is None:
        return None

    from kestrel_sovereign.a2a.replay_store import SharedReplayNonceStore

    store = SharedReplayNonceStore(db)
    try:
        setattr(agent, "_a2a_replay_nonce_store", store)
    except Exception:
        return store
    return store


async def _create_a2a_task_under_lifecycle_lease(
    agent,
    params,
    parts,
    raw_artifacts,
    sender_artifacts,
    manager,
    hosted_policy=None,
    commit=None,
):
    """Verify, authorize, and commit one A2A action under stable topology."""
    from kestrel_sovereign.a2a.envelope_signing import (
        canonical_message,
        verify_inbound_envelope,
    )

    if hosted_policy is not None:
        inbound_authorizer = hosted_policy.authorizer
        # Hosted recipients normally require a verified sender.  Keep the
        # envelope verifier open only long enough to identify a genuinely
        # absent signature: the manager-owned legacy path below then permits
        # one exact current same-host pre-ceremony sender.  A present malformed
        # or invalid signature remains a hard verification failure.
        scoped_sender_required = True
        verification_scope = None
        resolver = hosted_policy.resolver
    else:
        inbound_authorizer = _a2a_inbound_sender_authorizer(agent)
        scoped_sender_required = _a2a_inbound_requires_verified_sender(
            agent,
            inbound_authorizer,
        )
        # Capture after requires_verified_sender has monotonically observed any
        # newly attached scope, avoiding a self-induced marker transition.
        verification_scope = _a2a_inbound_scope_snapshot(agent)
        resolver = verification_scope[1]
    globally_required_signed = os.environ.get(
        "KESTREL_A2A_REQUIRE_SIGNED", ""
    ).lower() in (
        "1", "true", "yes",
    )
    require_signed = (
        globally_required_signed
        or (hosted_policy is None and scoped_sender_required)
    )
    claimed_sender = str(params.metadata.get("sender") or "")
    sender_witness = (
        manager.a2a_sender_identity_witness(claimed_sender)
        if manager is not None and claimed_sender
        else None
    )
    signed_message_text = canonical_message([part.text for part in parts])
    sender_verdict = await verify_inbound_envelope(
        params.metadata,
        task_id=params.id,
        message=signed_message_text,
        session_id=params.sessionId,
        artifacts=raw_artifacts,
        resolver=resolver,
        require_signed=require_signed,
        replay_store=_a2a_replay_store(agent),
    )
    if not sender_verdict.ok:
        raise HTTPException(
            status_code=403,
            detail=f"A2A sender verification failed: {sender_verdict.reason}",
        )
    if callable(commit) and not sender_verdict.verified:
        # Legacy unsigned envelopes are a narrow task-creation compatibility
        # lane. They cannot authorize a destructive lifecycle transition: the
        # shared host API key authenticates transport access, not the peer name
        # in caller-controlled metadata. Local Core callers use the separate
        # host-attested submission/cancellation contract instead.
        raise HTTPException(
            status_code=403,
            detail="A2A cancellation requires a verified sender signature",
        )
    if hosted_policy is not None:
        if manager.a2a_hosted_policy_for(agent) is not hosted_policy:
            raise HTTPException(
                status_code=403,
                detail="A2A hosted policy changed during verification",
            )
    elif not _a2a_inbound_scope_unchanged(agent, verification_scope):
        raise HTTPException(
            status_code=403,
            detail="A2A sender authorization context changed during verification",
        )
    if sender_verdict.verified and sender_witness is not None:
        current_witness = manager.a2a_sender_identity_witness(
            sender_verdict.sender
        )
        if (
            sender_witness[0] == "ambiguous"
            or not _a2a_sender_witness_unchanged(
                sender_witness,
                current_witness,
            )
            or (
                sender_witness[0] == "local"
                and sender_verdict.verification_document_fingerprint
                != sender_witness[3]
            )
        ):
            raise HTTPException(
                status_code=403,
                detail="A2A sender identity changed during verification",
            )

    if hosted_policy is not None:
        authorization_scope = None
        inbound_authorizer = hosted_policy.authorizer
        scoped_sender_required = True
    else:
        authorization_scope = _a2a_inbound_scope_snapshot(agent)
        inbound_authorizer = authorization_scope[0]
        scoped_sender_required = _a2a_inbound_requires_verified_sender(
            agent,
            inbound_authorizer,
        )
    authorized_legacy_sender_id = None
    if sender_verdict.verified:
        if inbound_authorizer is not None:
            if (
                scoped_sender_required
                and not _a2a_inbound_current_scope_is_valid(
                    inbound_authorizer,
                    hosted_policy,
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="A2A sender authorization context is invalid",
                )
            authorized = await _authorize_verified_a2a_sender(
                inbound_authorizer,
                sender_verdict.sender,
                hosted_policy,
            )
            if not authorized:
                raise HTTPException(
                    status_code=403,
                    detail="A2A sender authorization failed",
                )
            if (
                hosted_policy is not None
                and manager.a2a_hosted_policy_for(agent) is not hosted_policy
            ):
                raise HTTPException(
                    status_code=403,
                    detail="A2A hosted policy changed during authorization",
                )
            if (
                hosted_policy is None
                and not _a2a_inbound_scope_unchanged(
                    agent,
                    authorization_scope,
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "A2A sender authorization context changed "
                        "during authorization"
                    ),
                )
            if (
                scoped_sender_required
                and not _a2a_inbound_current_scope_is_valid(
                    inbound_authorizer,
                    hosted_policy,
                )
            ):
                raise HTTPException(
                    status_code=403,
                    detail="A2A sender authorization context is invalid",
                )
            if sender_witness is not None:
                current_witness = manager.a2a_sender_identity_witness(
                    sender_verdict.sender
                )
                if not _a2a_sender_witness_unchanged(
                    sender_witness,
                    current_witness,
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="A2A sender identity changed during authorization",
                    )
        elif scoped_sender_required:
            raise HTTPException(
                status_code=403,
                detail="A2A sender authorization unavailable",
            )
    elif hosted_policy is not None:
        # Hosted unsigned compatibility is deliberately narrower than the
        # historic same-host API-key fallback.  The manager, while holding its
        # lifecycle lease, proves the claimed name is an exact current local
        # non-hybrid sender and asks the immutable recipient directory policy
        # to authorize its stable DID. Unknown, cross-user, external, and
        # hybrid-downgrade claims fail closed.
        authorize_legacy = getattr(
            manager,
            "authorize_a2a_legacy_unsigned_sender",
            None,
        )
        authorized_sender_id = None
        if callable(authorize_legacy):
            try:
                authorized_sender_id = await authorize_legacy(
                    agent,
                    claimed_sender,
                    hosted_policy,
                )
            except Exception:  # noqa: BLE001 - manager policy boundary
                logger.warning(
                    "Hosted legacy unsigned A2A sender authorization failed",
                    exc_info=True,
                )
        if not isinstance(authorized_sender_id, str) or not authorized_sender_id:
            raise HTTPException(
                status_code=403,
                detail="A2A unsigned sender is not an authorized local legacy peer",
            )
        authorized_legacy_sender_id = authorized_sender_id
        if manager.a2a_hosted_policy_for(agent) is not hosted_policy:
            raise HTTPException(
                status_code=403,
                detail="A2A hosted policy changed during legacy authorization",
            )
    elif scoped_sender_required:
        raise HTTPException(
            status_code=403,
            detail="A2A scoped recipients require a verified sender",
        )

    params.metadata["sender_verified"] = sender_verdict.verified
    authorized_sender_id = (
        sender_verdict.sender
        if sender_verdict.verified
        else authorized_legacy_sender_id
    )
    # A same-host agent may create a task before its hybrid ceremony and
    # cancel it afterward with its successor signing DID.  The manager witness
    # cryptographically bound that successor DID to the exact live local agent;
    # persist and compare the manager's stable routing DID so the principal
    # does not change merely because its signing key advanced.  External
    # senders retain their verified signing DID because this host has no local
    # lifecycle witness with which to normalize them.
    if (
        sender_verdict.verified
        and sender_witness is not None
        and sender_witness[0] == "local"
    ):
        witnessed_agent = sender_witness[1]
        stable_sender_id = None
        for attribute in ("did", "agent_id"):
            candidate = getattr(witnessed_agent, attribute, None)
            if isinstance(candidate, str) and candidate:
                stable_sender_id = candidate
                break
        witnessed_identity = sender_witness[2]
        bound_dids = {
            candidate
            for candidate in (
                getattr(witnessed_identity, "legacy_did", None),
                getattr(witnessed_identity, "new_did", None),
            )
            if isinstance(candidate, str) and candidate
        }
        if (
            isinstance(stable_sender_id, str)
            and stable_sender_id in bound_dids
            and sender_verdict.sender in bound_dids
        ):
            authorized_sender_id = stable_sender_id
    if callable(commit):
        try:
            return await commit(authorized_sender_id)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to commit verified A2A action: %s",
                exc,
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail="Failed to commit A2A action"
            ) from exc

    local_name = (
        getattr(agent, "did", None)
        or getattr(agent, "_agent_name", None)
        or "unknown"
    )
    try:
        return await agent.task_manager.create_task(
            params=params,
            agent_name=local_name,
            artifacts=sender_artifacts or None,
            creator_agent_id=authorized_sender_id,
        )
    except TaskAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Failed to create A2A task from peer submission: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to create task")


async def _create_verified_a2a_task(
    agent,
    params,
    parts,
    raw_artifacts,
    sender_artifacts,
    commit=None,
):
    """Use a shared manager lease for hosted recipients; preserve standalone flow."""
    manager = getattr(agent, "_a2a_host_manager", None)
    # Current managers expose a reader lease so independent hosted recipients
    # can verify and commit concurrently. Retain the old lifecycle-lock seam
    # for compatibility hosts; it remains exclusive and therefore safe.
    lease_factory = getattr(manager, "a2a_execution_lease", None)
    if not callable(lease_factory):
        lease_factory = getattr(manager, "a2a_lifecycle_lease", None)
    if callable(lease_factory):
        async with lease_factory():
            hosted_policy = manager.a2a_hosted_policy_for(agent)
            if hosted_policy is None:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "A2A recipient is no longer published "
                        "with hosted policy"
                    ),
                )
            return await _create_a2a_task_under_lifecycle_lease(
                agent,
                params,
                parts,
                raw_artifacts,
                sender_artifacts,
                manager,
                hosted_policy,
                commit,
            )
    return await _create_a2a_task_under_lifecycle_lease(
        agent,
        params,
        parts,
        raw_artifacts,
        sender_artifacts,
        None,
        None,
        commit,
    )


@router.post("/tasks/send")
@limiter.limit("120/minute")
async def send_task(request: Request):
    """
    Receive an A2A task creation request from another agent.

    Inbound shape (matches A2A ``TaskSendParams``):

        {
          "id": "<uuid>",                # caller-assigned task id
          "sessionId": "<uuid>",
          "message": {
            "role": "user",
            "parts": [{"type":"text", "text": "..."}]
          },
          "metadata": {                  # optional
            "sender": "<agent name or did>",
            "skill": "<workflow.* skill id>",
            ...
          },
          "artifacts": [                 # optional — send-side handoff
            {                            # payload (docs, refs, evidence)
              "name": "plan",
              "parts": [{"type":"text", "text": "..."}],
              "index": 0,
              "lastChunk": true
            }
          ]
        }

    The endpoint calls ``task_manager.create_task`` which persists the
    task AND fires the ``on_task_submitted`` callback. That callback
    builds a ``a2a.task_submitted`` Signal and enqueues it via the
    dispatcher so this agent wakes up and acts on the new task. Without
    this endpoint, agents had no wire-level way to submit a task —
    only the local agent's own code could call create_task — which made
    inter-agent A2A submission impossible to surface from a tool.

    Cryptographic sender verification (#1673): if the envelope carries a
    ``metadata["signature"]`` block, it is verified against the sender's
    resolved DID document via ``verify_inbound_envelope`` (hybrid Ed25519 +
    ML-DSA-65, replay-windowed, DID-bound). A present-but-invalid signature is
    always rejected; an unsigned envelope is allowed for standalone/local
    shared-API-key compatibility unless ``KESTREL_A2A_REQUIRE_SIGNED`` is set.
    A scoped hosted recipient always requires a verified signature. The verdict
    is recorded as ``metadata["sender_verified"]`` for downstream governance
    tiering.

    DID-document resolution and recipient authorization are separate trust
    decisions. After successful cryptographic verification, a recipient-scoped
    ``agent.a2a_inbound_sender_authorizer`` must approve the verified sender
    before ``sender_verified`` is set or a task is created. Hosted scoped
    recipients require a signed envelope and fail closed if that seam or its
    live scope is missing/revoked. Standalone shared-API-key deployments retain
    unsigned compatibility.
    """
    agent = get_agent(request)
    body = await _parse_json_body(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(
            status_code=503,
            detail="TaskManager not available — agent cannot accept A2A tasks",
        )

    from kestrel_sovereign.a2a.types import (
        Artifact,
        Message,
        TextPart,
        TaskSendParams,
    )
    from kestrel_sovereign.a2a.local_submission import (
        HOST_ATTESTED_LOCAL_SUBMISSION_METADATA,
    )
    try:
        # Parse body into TaskSendParams. Sender-side already validated
        # the shape, but we re-validate here because this is the only
        # untrusted-input boundary.
        message_data = body.get("message") or {}
        parts_data = message_data.get("parts") or []
        parts = []
        for p in parts_data:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(TextPart(text=str(p.get("text", ""))))
        if not parts:
            raise HTTPException(
                status_code=400,
                detail="task message must contain at least one text part",
            )
        message = Message(
            role=str(message_data.get("role", "user")),
            parts=parts,
        )
        raw_metadata = body.get("metadata") or {}
        if (
            isinstance(raw_metadata, dict)
            and HOST_ATTESTED_LOCAL_SUBMISSION_METADATA in raw_metadata
        ):
            # This marker is Core-internal provenance for the explicit local
            # host-attested path.  A wire client can claim arbitrary metadata,
            # so accepting it here would let an external sender forge the task
            # ownership relationship used by local retrieval/subscription.
            raise HTTPException(
                status_code=400,
                detail="host-attested local A2A provenance is not accepted over the wire",
            )
        params = TaskSendParams(
            id=str(body.get("id") or ""),
            sessionId=str(body.get("sessionId") or ""),
            message=message,
            metadata=raw_metadata,
        )
        # Send-side artifacts/references: a sender may attach durable
        # handoff payload (planning docs, evidence bundles, saved-memory
        # references, logs, diffs) at task-creation time. This is the
        # send-side mirror of the responder-side attach flow — the
        # artifacts land on the task at SUBMITTED so the recipient can
        # retrieve them before doing any work. Validate here because
        # this is the untrusted-input boundary.
        raw_artifacts = body.get("artifacts") or []
        if not isinstance(raw_artifacts, list):
            raise HTTPException(
                status_code=400,
                detail="task 'artifacts' must be a list of artifact objects",
            )
        sender_artifacts = []
        for a in raw_artifacts:
            if not isinstance(a, dict):
                raise HTTPException(
                    status_code=400,
                    detail="each artifact must be an object",
                )
            sender_artifacts.append(Artifact.model_validate(a))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid TaskSendParams: {e}",
        )

    if not params.id or not params.sessionId:
        raise HTTPException(
            status_code=400,
            detail="TaskSendParams.id and TaskSendParams.sessionId are required",
        )

    task = await _create_verified_a2a_task(
        agent,
        params,
        parts,
        raw_artifacts,
        sender_artifacts,
    )

    # Return the canonical A2A Task envelope (model_dump produces the
    # standard JSON-RPC-friendly shape).
    return task.model_dump()


@router.get("/tasks/{task_id}")
async def get_task(request: Request, task_id: str):
    """
    Fetch a single background A2A task with its artifacts.

    Used by the Tasks panel to render "Load artifacts" for a given task.
    """
    agent = get_agent(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(status_code=404, detail="TaskManager not available")

    try:
        task = await agent.task_manager.task_store.get(task_id)
    except Exception as e:
        logger.error(f"Error loading task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error loading task.")

    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    message_text = None
    if task.status.message and task.status.message.parts:
        for part in task.status.message.parts:
            if hasattr(part, "text"):
                message_text = part.text
                break

    artifacts_payload = []
    for artifact in (task.artifacts or []):
        if hasattr(artifact, "model_dump"):
            artifacts_payload.append(artifact.model_dump())
        else:
            artifacts_payload.append(artifact)

    return {
        "id": task.id,
        "status": task.status.state.value,
        "message": message_text,
        "artifacts": artifacts_payload,
        "metadata": task.metadata or {},
    }


@router.post("/tasks/{task_id:path}/cancel")
@limiter.limit("120/minute")
async def cancel_task_from_peer(request: Request, task_id: str):
    """Cancel an A2A task through the recipient's authoritative store.

    The API key authenticates the host connection, not the peer actor. The
    cancellation therefore carries the same DID-signed, replay-protected
    envelope as task creation and repeats the recipient's live peer-scope
    authorization before the atomic creator/recipient predicate is applied.
    """
    agent = get_agent(request)
    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(status_code=404, detail="TaskManager not available")

    body = await _parse_json_body(request)
    reason = body.get("reason") or "Task canceled by creator"
    session_id = body.get("sessionId") or ""
    metadata = body.get("metadata") or {}
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4096:
        raise HTTPException(
            status_code=400,
            detail="Cancellation reason must be a non-empty string up to 4096 characters",
        )
    if not isinstance(session_id, str) or not session_id:
        raise HTTPException(status_code=400, detail="sessionId is required")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    if metadata.get("a2a_verb") != "cancel_task":
        raise HTTPException(
            status_code=400,
            detail="Cancellation envelope must bind a2a_verb=cancel_task",
        )

    from kestrel_sovereign.a2a.task_manager import (
        TaskCancellationAuthorizationError,
    )
    from kestrel_sovereign.a2a.types import Message, TaskSendParams, TextPart

    params = TaskSendParams(
        id=task_id,
        sessionId=session_id,
        message=Message(role="user", parts=[TextPart(text=reason)]),
        metadata=metadata,
    )
    recipient_agent_id = next(
        (
            candidate
            for candidate in (
                getattr(agent.task_manager, "host_agent_id", None),
                getattr(agent, "did", None),
                getattr(agent, "agent_id", None),
            )
            if isinstance(candidate, str) and candidate
        ),
        None,
    )
    if not isinstance(recipient_agent_id, str) or not recipient_agent_id:
        raise HTTPException(
            status_code=503,
            detail="A2A task cancellation requires a durable recipient identity",
        )

    async def _cancel(authorized_sender_id: str):
        if not isinstance(authorized_sender_id, str) or not authorized_sender_id:
            raise HTTPException(
                status_code=403,
                detail="A2A task cancellation requires an authenticated agent",
            )
        try:
            return await agent.task_manager.cancel_task(
                task_id,
                reason=reason,
                agent_name=authorized_sender_id,
                recipient_agent_id=recipient_agent_id,
            )
        except TaskCancellationAuthorizationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            status_code = 404 if str(exc) == f"Task not found: {task_id}" else 409
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    task = await _create_verified_a2a_task(
        agent,
        params,
        params.message.parts,
        [],
        [],
        commit=_cancel,
    )
    return {
        "id": task.id,
        "status": task.status.state.value,
        "cancellation_receipt": (task.metadata or {}).get(
            "cancellation_receipt"
        ),
    }


@router.get("/tasks/{task_id}/subscribe")
@limiter.limit("30/minute")
async def subscribe_task(request: Request, task_id: str):
    """
    SSE stream of status updates for a single A2A task.

    Subscribers receive:
        event: status
        data: {"id": "...", "status": {...}, "final": true|false}

    Plus periodic ``event: keepalive`` pings so HTTP intermediaries
    (reverse proxies, Castle towers, NAT idle-close) don't close the
    long-lived connection between updates. The stream closes after the
    first event whose ``final == true`` is delivered.

    This endpoint exists so peer agents that just POST'd a question via
    ``/tasks/send`` can wait for the answer with a push subscription
    instead of polling ``GET /tasks/{id}`` on an adaptive backoff
    (#1444). The sender's ``PeersFeature._post_a2a_task`` opens this
    stream in a background-tracked coroutine and turns the terminal
    event into a local ``a2a.question_answered`` cognition signal.

    Auth: same handshake as ``POST /tasks/send`` — the agent-routing
    middleware applies its API-key check before this handler runs.

    Connection limits: ``MAX_SSE_CONNECTIONS_PER_CLIENT`` per
    (client_ip, agent_id) pair, same posture as ``/notifications/sse``.

    Final-state snapshot on connect: ``TaskManager.subscribe`` already
    yields a "status" event with the current task state immediately
    after subscription so a late subscriber doesn't miss a terminal
    that already fired. We forward that snapshot as the first SSE
    frame.
    """
    agent = get_agent(request)

    if not hasattr(agent, "task_manager") or not agent.task_manager:
        raise HTTPException(
            status_code=404, detail="TaskManager not available",
        )

    # 404 on unknown task_id rather than holding an SSE connection
    # open against a non-existent subscription target — the sender
    # would otherwise idle forever waiting for terminal events that
    # can never fire.
    task = await agent.task_manager.task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found",
        )

    client_ip = request.client.host if request.client else "unknown"
    agent_id = getattr(agent, "agent_id", "default")
    conn_key = (client_ip, agent_id)
    async with _sse_lock:
        if _sse_connections[conn_key] >= MAX_SSE_CONNECTIONS_PER_CLIENT:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many SSE connections "
                    f"(limit: {MAX_SSE_CONNECTIONS_PER_CLIENT})"
                ),
            )
        _sse_connections[conn_key] += 1

    async def event_generator():
        import json
        last_ping = time.monotonic()
        ping_interval = 10.0  # 10s heartbeat for intermediaries
        try:
            # ``TaskManager.subscribe`` already yields the current state
            # first, then streams updates, then breaks on the first
            # final event. It also yields its own ``keepalive`` events
            # on its internal timeout — we forward those as SSE
            # comments-or-pings so the connection stays warm even when
            # the task sits in SUBMITTED for a while.
            async for ev in agent.task_manager.subscribe(task_id):
                if await request.is_disconnected():
                    logger.debug(
                        "task subscribe client disconnected (task=%s)",
                        task_id[:8],
                    )
                    break
                ev_name = ev.get("event") or "status"
                ev_data = ev.get("data") or ""
                yield f"event: {ev_name}\ndata: {ev_data}\n\n"
                last_ping = time.monotonic()
                if ev.get("final"):
                    # Terminal event delivered; subscribe()'s loop
                    # already breaks, but yielding a small comment line
                    # signals end-of-stream to the SSE client cleanly.
                    yield ": end-of-stream\n\n"
                    break
                # Top up the keepalive cadence if a long stretch of
                # quiet just ended.
                now = time.monotonic()
                if now - last_ping >= ping_interval:
                    yield f"event: ping\ndata: {json.dumps({'t': now})}\n\n"
                    last_ping = now
        except asyncio.CancelledError:
            logger.debug(
                "task subscribe cancelled (task=%s)", task_id[:8],
            )
        except Exception as e:
            logger.error(
                "task subscribe error (task=%s): %s",
                task_id[:8], e, exc_info=True,
            )
            yield (
                "event: error\ndata: "
                + json.dumps({"error": "Internal server error"})
                + "\n\n"
            )
        finally:
            async with _sse_lock:
                _sse_connections[conn_key] -= 1
                if _sse_connections[conn_key] <= 0:
                    del _sse_connections[conn_key]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Heartbeat Endpoints ---
#
# In the OpenClaw / kestrel-claw tradition, a "heartbeat" is a scheduled LLM
# turn that reads HEARTBEAT.md and replies HEARTBEAT_OK or an alert.  That
# surface is owned by HeartbeatRunner (kestrel_sovereign/heartbeat.py) and
# these endpoints route to it.  Liveness / readiness probes (structured
# subsystem checks, no LLM) live under /agent/health/* below.


@router.get("/heartbeat/status")
async def heartbeat_status(request: Request):
    """Get heartbeat system status and recent history."""
    agent = get_agent(request)
    runner = getattr(agent, 'heartbeat_runner', None)
    if not runner:
        return {"enabled": False, "message": "Heartbeat not configured"}

    return runner.get_status()


@router.post("/heartbeat/trigger")
async def heartbeat_trigger(request: Request):
    """Manually trigger a heartbeat check."""
    agent = get_agent(request)
    runner = getattr(agent, 'heartbeat_runner', None)
    if not runner:
        raise HTTPException(status_code=404, detail="Heartbeat not configured")

    try:
        result = await runner.run_once()
        return result.to_dict()
    except Exception as e:
        logger.error(f"Heartbeat trigger error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error triggering heartbeat.")


# --- Health (liveness probe) Endpoints ---


def _get_health_feature(agent):
    """Resolve the HealthFeature instance from an agent, or None."""
    features = getattr(agent, 'features', None) or {}
    if isinstance(features, dict):
        candidates = features.values()
    else:
        candidates = features
    for feat in candidates:
        if feat.__class__.__name__ == "HealthFeature":
            return feat
    return None


@router.get("/health/status")
async def agent_health_status(request: Request):
    """Return HealthFeature status (feature state, interval, last result).

    Separate from :func:`heartbeat_status` — heartbeat is an LLM-driven
    self-check while ``/agent/health/*`` is the structured liveness probe.
    """
    agent = get_agent(request)
    feature = _get_health_feature(agent)
    if not feature:
        return {"enabled": False, "message": "HealthFeature not available on this agent"}
    return feature.get_status()


@router.post("/health/trigger")
async def agent_health_trigger(request: Request):
    """Run a single liveness check synchronously and return the result."""
    agent = get_agent(request)
    feature = _get_health_feature(agent)
    if not feature:
        raise HTTPException(
            status_code=404,
            detail="HealthFeature not available on this agent",
        )
    try:
        return await feature.run_once()
    except Exception as e:
        logger.error(f"Health trigger error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error running liveness check.")


# =========================================================================
# Agent Mesh Protocol — RETIRED (#1367 phase 5).
#
# The legacy POST /agent/mesh and GET /agent/mesh/inbox endpoints have
# been removed. All inter-agent messaging now goes through the A2A
# task path (POST /api/agent/tasks/send + GET /api/agent/tasks/{id}),
# which provides persistence, lifecycle states, and signal-driven
# inbound wake (a2a.task_submitted). Falconer workflow events that
# used to be MeshMessage(type=ASSIGN|REVIEW_NEEDED|...) are now A2A
# tasks with metadata["skill"]="workflow.assign" (etc.).
# =========================================================================
