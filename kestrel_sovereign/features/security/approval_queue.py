"""
Kestrel Security - Queue-based Approval System.

This module provides a queue for pending approval requests, allowing
the agent to stack requests while waiting for user decisions.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

if TYPE_CHECKING:
    from kestrel_sovereign.security.auto_approve import AutoApprovePolicy

    from .permissions import PermissionStore

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """
    A pending approval request in the queue.

    When a tool requires user approval, a request is created and the
    execution waits on resume_event until the user makes a decision.
    """
    id: str
    feature_name: str
    tool_name: str
    tool_args: Dict
    created_at: datetime
    # Optional wall-clock cap. ``None`` means "wait indefinitely for
    # the user" — appropriate for interactive approvals where the
    # user owns the timing. A finite value is for batch/automation
    # callers that want a deterministic abandon point. Stale-request
    # cleanup is the operator's responsibility via
    # ``ApprovalQueue.sweep_stale``, not an implicit per-request
    # timer the user never sees. Earlier defaults (300s, then 3600s)
    # were both arbitrary and both produced the same disappear-modal
    # shape, just with different mean-time-to-bug.
    timeout_seconds: Optional[float] = None
    status: ApprovalStatus = ApprovalStatus.PENDING

    # For resumption after approval
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    user_decision: Optional[str] = None  # "once", "session", "always"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "feature_name": self.feature_name,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "created_at": self.created_at.isoformat(),
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "user_decision": self.user_decision,
        }


# Type for the SSE callbacks
OnRequestAddedCallback = Callable[[ApprovalRequest], Awaitable[None]]
# ``reason`` is one of "timeout" | "cancelled" — the request exited
# ``request_approval`` without a user submit, so any UI showing the modal
# must withdraw it. See #877.
OnRequestWithdrawnCallback = Callable[[ApprovalRequest, str], Awaitable[None]]


class ApprovalQueue:
    """
    Queue-based approval system for tool calls.

    When a tool requires user approval, a request is added to the queue
    and the execution pauses until the user makes a decision. The UI
    receives SSE events to display pending requests.

    Example:
        queue = ApprovalQueue(on_request_added=emit_sse_event)

        # In security hook
        approved, scope = await queue.request_approval(
            feature_name="WalletAgent",
            tool_name="send_payment",
            tool_args={"amount": 100, "recipient": "alice"}
        )

        if approved:
            # User approved - scope is "once", "session", or "always"
            logger.info(f"Approved with scope: {scope}")
        else:
            # User denied or timeout
            logger.info("Denied")
    """

    def __init__(
        self,
        on_request_added: Optional[OnRequestAddedCallback] = None,
        on_request_withdrawn: Optional[OnRequestWithdrawnCallback] = None,
        permission_store: Optional["PermissionStore"] = None,
        auto_approve_policy: Optional["AutoApprovePolicy"] = None,
        agent: Optional[object] = None,
    ):
        """
        Initialize the approval queue.

        Args:
            on_request_added: Optional async callback when a request is added.
                             Used to emit SSE events to the UI.
            on_request_withdrawn: Optional async callback when a request is
                             evicted without a user submit (timeout or task
                             cancellation). Used to emit SSE events so the UI
                             can withdraw any open modal — without this, the
                             user's modal would 404 on submit. See #877.
            permission_store: Optional store for persisting the user's scope
                             choice ("session"/"always") and writing audit
                             rows.  When set, every approval resolved through
                             :meth:`request_approval` is persisted/audited
                             centrally so that callers don't have to remember
                             to do it (#785).  When None, callers retain the
                             old responsibility of persisting scope themselves.
        """
        self._pending: Dict[str, ApprovalRequest] = {}
        self._resolved: Dict[str, ApprovalRequest] = {}
        self._on_request_added = on_request_added
        self._on_request_withdrawn = on_request_withdrawn
        self._permission_store = permission_store
        self._auto_approve_policy = auto_approve_policy
        self._agent = agent

    @property
    def pending_count(self) -> int:
        """Get the number of pending requests."""
        return len(self._pending)

    @property
    def pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending requests."""
        return list(self._pending.values())

    async def request_approval(
        self,
        feature_name: str,
        tool_name: str,
        tool_args: Dict,
        timeout: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Queue a request and wait for user decision.

        This method blocks until the user approves, denies, or the
        wall-clock ``timeout`` (if any) elapses.

        Args:
            feature_name: Name of the feature making the request
            tool_name: Name of the tool requesting approval
            tool_args: Arguments to the tool (shown to user)
            timeout: Wall-clock seconds to wait. ``None`` (default) =
                wait indefinitely; the user owns the timing.
                Operators can call ``sweep_stale`` to clean up
                requests that are clearly abandoned. Pass a finite
                value only for batch/automation callers that need a
                deterministic abandon point.

        Returns:
            Tuple of (approved: bool, scope: str)
            - approved: True if user approved, False if denied or timeout
            - scope: "once", "session", "always", or "timeout"
        """
        if self._permission_store is not None:
            try:
                from .permissions import PermissionLevel

                level = await self._permission_store.get_permission(
                    feature_name,
                    tool_name,
                )
                if level == PermissionLevel.DENY:
                    await self._permission_store.log_decision(
                        feature_name=feature_name,
                        tool_name=tool_name,
                        action="tool_execution",
                        decision="auto_denied",
                        args_summary=self._summarize_args(tool_args),
                    )
                    logger.info(
                        "ApprovalQueue denied %s.%s from explicit policy",
                        feature_name,
                        tool_name,
                    )
                    return (False, "denied")
                if (
                    self._permission_store.get_global_auto_mode()
                    and level == PermissionLevel.AUTO
                ):
                    await self._permission_store.log_decision(
                        feature_name=feature_name,
                        tool_name=tool_name,
                        action="tool_execution",
                        decision="auto_mode_allowed",
                        user_choice="constitutional_honesty_unflagged",
                        args_summary=self._summarize_args(tool_args),
                    )
                    logger.info(
                        "ApprovalQueue auto-mode approved %s.%s without prompting",
                        feature_name,
                        tool_name,
                    )
                    return (True, "auto")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "ApprovalQueue: failed to evaluate pre-approval policy for "
                    f"{feature_name}.{tool_name}: {e}",
                    exc_info=True,
                )

        # Scoped auto-approve allowlist. Runs AFTER the explicit DENY/AUTO
        # fast-path (so an operator DENY still hard-stops) and BEFORE a
        # human is queued. A match means the Sovereign pre-authorised this
        # exact pattern for this agent+repo; we write the full audit row
        # *before* returning so the invocation can never run silently.
        if self._auto_approve_policy is not None:
            try:
                agent_name = getattr(self._agent, "_agent_name", None)
                agent_did = getattr(self._agent, "did", None) or "anonymous"
                match = await self._auto_approve_policy.evaluate(
                    agent_name=agent_name,
                    feature_name=feature_name,
                    tool_name=tool_name,
                    tool_args=tool_args,
                )
                if match is not None and self._permission_store is not None:
                    audit_id = await self._permission_store.log_auto_approve(
                        agent_did=agent_did,
                        agent_name=agent_name,
                        feature_name=feature_name,
                        tool_name=tool_name,
                        command=match.command,
                        pattern=match.rule.pattern,
                        repo_scope=match.rule.repo_scope,
                        rule_source=match.rule.source,
                    )
                    await self._permission_store.log_decision(
                        feature_name=feature_name,
                        tool_name=tool_name,
                        action="tool_execution",
                        decision="auto_approved",
                        user_choice=f"auto_approve:{match.rule.source}",
                        args_summary=self._summarize_args(tool_args),
                    )
                    logger.info(
                        "ApprovalQueue auto-approved %s.%s for agent=%s "
                        "(audit_id=%s, source=%s)",
                        feature_name, tool_name, agent_name or "?",
                        audit_id, match.rule.source,
                    )
                    # The audit id rides the existing allowed_by
                    # "approval:<scope>" chain so the executing tool can
                    # finalize the real exit code once it returns.
                    return (True, f"auto_approve:{audit_id}")
            except Exception as e:  # noqa: BLE001 - never block on policy
                logger.warning(
                    "ApprovalQueue: auto-approve evaluation failed for "
                    f"{feature_name}.{tool_name}: {e}",
                    exc_info=True,
                )

        request = ApprovalRequest(
            id=str(uuid4()),
            feature_name=feature_name,
            tool_name=tool_name,
            tool_args=tool_args,
            created_at=datetime.now(timezone.utc),
            timeout_seconds=timeout,
        )

        self._pending[request.id] = request
        logger.info(
            f"Approval request queued: {request.id[:8]} "
            f"({feature_name}.{tool_name})"
        )

        # Notify UI via SSE
        if self._on_request_added:
            try:
                await self._on_request_added(request)
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Failed to notify UI of approval request (network error): {e}", exc_info=True)
            except (TypeError, AttributeError) as e:
                logger.warning(f"Failed to notify UI of approval request (callback error): {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Failed to notify UI of approval request: {e}", exc_info=True)

        # Wait for user decision or wall-clock timeout.
        #
        # Past behavior popped the request and emitted ``approval_withdrawn``
        # on every non-success exit path, including ``CancelledError`` —
        # which fires whenever the calling task dies (HTTP stream
        # dropped, agent loop torn down, user switched chat tabs in
        # the multi_agent). PR #877 reframed the user-facing message but
        # kept the underlying behavior: a slow user lost the chance
        # to decide. That was spackle.
        #
        # New invariant: only TIMEOUT removes the request. Cancellation
        # leaves it alive — the modal stays open, the user can decide
        # at their leisure, and ``submit_decision`` records the
        # outcome whenever the click finally lands. Stale entries are
        # reaped by ``sweep_stale`` (called from a periodic
        # background task or directly by tests).
        timed_out = False
        try:
            await asyncio.wait_for(
                request.resume_event.wait(),
                timeout=timeout
            )

            approved = request.status == ApprovalStatus.APPROVED
            scope = request.user_decision or "once"

            logger.info(
                f"Approval request {request.id[:8]} resolved: "
                f"{'approved' if approved else 'denied'} ({scope})"
            )

            await self._persist_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                tool_args=tool_args,
                approved=approved,
                scope=scope,
            )

            return (approved, scope)

        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMEOUT
            timed_out = True
            logger.warning(
                f"Approval request {request.id[:8]} timed out after {timeout}s"
            )
            await self._persist_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                tool_args=tool_args,
                approved=False,
                scope="timeout",
            )
            return (False, "timeout")

        except asyncio.CancelledError:
            # Calling task cancelled (HTTP stream dropped, browser
            # closed, user switched to a different agent in the
            # multi_agent). The user has not yet decided. Leave the
            # request in ``_pending`` so the modal stays interactive,
            # and re-raise without firing withdrawal — the modal must
            # NOT auto-close on us.
            logger.info(
                f"Approval request {request.id[:8]} await cancelled; "
                "request kept alive for user decision"
            )
            raise

        finally:
            # Only remove the request on a true success (resume_event
            # fired) or timeout. Cancellation leaves it for the user.
            if request.resume_event.is_set() or timed_out:
                popped = self._pending.pop(request.id, None)
            else:
                popped = None
            # Notify UI ONLY when we genuinely abandon the request via
            # timeout. Successful user-submit closes its own modal;
            # cancellation no longer triggers withdrawal at all.
            if popped is not None and timed_out and self._on_request_withdrawn:
                try:
                    await self._on_request_withdrawn(popped, "timeout")
                except (ConnectionError, TimeoutError) as e:
                    logger.warning(
                        f"Failed to notify UI of approval withdrawal (network error): {e}",
                        exc_info=True,
                    )
                except (TypeError, AttributeError) as e:
                    logger.warning(
                        f"Failed to notify UI of approval withdrawal (callback error): {e}",
                        exc_info=True,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to notify UI of approval withdrawal: {e}",
                        exc_info=True,
                    )

    async def _persist_decision(
        self,
        *,
        feature_name: str,
        tool_name: str,
        tool_args: Dict,
        approved: bool,
        scope: str,
    ) -> None:
        """Persist the user's scope choice and write an audit row.

        Idempotent for the "once" case (no persistence). When ``scope`` is
        ``"session"`` or ``"always"``, the corresponding ``set_permission``
        call is recorded so the next invocation of this tool skips the
        popup. When the request was denied or timed out, no permission is
        set but the audit row still records the decision.

        This is the single home for scope-aware persistence — see #785.
        Callers (the security hook AND every direct ``approval_queue``
        caller in features like ``code_edit``, ``compute``, ``keys``,
        ``reflection``) all benefit without having to repeat the logic.
        """
        if self._permission_store is None:
            return

        # Lazy import: PermissionLevel lives next door but we keep the
        # import out of module-load to avoid a circular reference.
        from .permissions import PermissionLevel

        try:
            if approved and scope == "always":
                await self._permission_store.set_permission(
                    feature_name,
                    tool_name,
                    PermissionLevel.ALLOW,
                    scope="always",
                    reason="User approved with 'always' scope",
                )
            elif approved and scope == "session":
                await self._permission_store.set_permission(
                    feature_name,
                    tool_name,
                    PermissionLevel.ALLOW,
                    scope="session",
                    reason="User approved for this session",
                )
            # "once" / "denied" / "timeout" / "cancelled" → no permission row.

            # Audit every decision so operators can see what fired even when
            # nothing was persisted.
            if approved:
                decision = "user_approved"
            elif scope == "timeout":
                decision = "timeout"
            elif scope in ("cancelled", "cancelled_all"):
                decision = "user_cancelled"
            else:
                decision = "user_denied"

            args_summary = self._summarize_args(tool_args)
            await self._permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action="tool_execution",
                decision=decision,
                user_choice=scope,
                args_summary=args_summary,
            )
        except Exception as e:  # noqa: BLE001
            # A persistence failure must not corrupt the user's decision.
            # Log loudly and let the caller proceed with `approved` as-is.
            logger.warning(
                "ApprovalQueue: failed to persist decision for "
                f"{feature_name}.{tool_name}: {e}",
                exc_info=True,
            )

    @staticmethod
    def _summarize_args(args: Optional[Dict], max_chars: int = 500) -> Optional[str]:
        """Truncate tool args for the audit log so secrets don't get logged
        in full.  Mirrors :meth:`SecurityHook._summarize_args` so the audit
        rows look the same regardless of which path produced them."""
        if not args:
            return None
        try:
            import json
            text = json.dumps(args, default=str)
        except (TypeError, ValueError):
            text = repr(args)
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def submit_decision(
        self,
        request_id: str,
        approved: bool,
        scope: str = "once",
    ) -> bool:
        """
        Submit a user's decision for a pending request.

        Called by the API when the user makes a decision in the UI.

        Args:
            request_id: ID of the pending request
            approved: Whether the user approved the request
            scope: Scope of approval - "once", "session", or "always"

        Returns:
            True if the request was found and decision submitted,
            False if the request was not found (expired or invalid)
        """
        request = self._pending.get(request_id)
        if not request:
            logger.warning(f"Decision submitted for unknown request: {request_id}")
            return False

        # Idempotency / CAS: a request that already has a decision must
        # not accept another one. Without this guard, callers that race
        # (UI double-click, polling responder ticking faster than the
        # awaiter's finally-block can pop _pending) silently overwrite
        # the user's first decision and inflate any per-call counters
        # downstream. The pop happens in request_approval()'s
        # finally-block on the awaiter's next scheduled tick — so the
        # request lingers in _pending for one or more event-loop
        # iterations after resume_event.set(), which is the exact race
        # window this guard closes.
        if request.status != ApprovalStatus.PENDING:
            logger.warning(
                f"Decision submitted for already-decided request "
                f"{request_id[:8]} (status={request.status.value}); ignored"
            )
            return False

        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.user_decision = scope
        self._resolved[request_id] = request
        if len(self._resolved) > 512:
            oldest = next(iter(self._resolved))
            self._resolved.pop(oldest, None)
        request.resume_event.set()  # Unblock the waiting coroutine

        logger.info(
            f"Decision submitted for {request_id[:8]}: "
            f"{'approved' if approved else 'denied'} ({scope})"
        )

        return True

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """
        Get a pending request by ID.

        Args:
            request_id: ID of the request

        Returns:
            ApprovalRequest if found, None otherwise
        """
        return self._pending.get(request_id) or self._resolved.get(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a pending request.

        Args:
            request_id: ID of the request to cancel

        Returns:
            True if the request was found and cancelled
        """
        request = self._pending.get(request_id)
        if not request:
            return False

        request.status = ApprovalStatus.DENIED
        request.user_decision = "cancelled"
        request.resume_event.set()

        logger.info(f"Request {request_id[:8]} cancelled")
        return True

    async def sweep_stale(
        self,
        older_than_seconds: float,
    ) -> int:
        """Remove pending requests older than ``older_than_seconds``.

        The cancellation-leaves-request-alive contract means
        ``_pending`` grows unboundedly when many agent tasks die
        before users decide. ``sweep_stale`` is the operator's
        cleanup primitive — call it on whatever cadence and cutoff
        makes sense for the deployment (e.g. hourly with a 24h
        cutoff). The cutoff is intentionally a required argument:
        there's no sensible default when individual requests carry
        no implicit deadline.

        Fires ``on_request_withdrawn(req, "timeout")`` for each
        reaped request so any still-mounted UI modal closes.

        Returns the number of requests removed.
        """
        now = datetime.now(timezone.utc)
        reaped: List[ApprovalRequest] = []
        for rid in list(self._pending.keys()):
            req = self._pending.get(rid)
            if req is None:
                continue
            if req.resume_event.is_set():
                # Decision already submitted — let the regular path
                # clean it up (we don't want to race with an awaiting
                # request_approval coroutine that's about to return).
                continue
            cutoff = older_than_seconds
            age = (now - req.created_at).total_seconds()
            if age >= cutoff:
                reaped.append(req)
                self._pending.pop(rid, None)
                req.status = ApprovalStatus.TIMEOUT

        for req in reaped:
            logger.info(
                f"Sweeping stale approval request {req.id[:8]} "
                f"(age > {older_than_seconds}s)"
            )
            if self._on_request_withdrawn:
                try:
                    await self._on_request_withdrawn(req, "timeout")
                except Exception as e:
                    logger.warning(
                        f"Sweep withdrawal callback failed for "
                        f"{req.id[:8]}: {e}",
                    )

        return len(reaped)

    def cancel_all(self) -> int:
        """
        Cancel all pending requests.

        Returns:
            Number of requests cancelled
        """
        count = 0
        for request in list(self._pending.values()):
            request.status = ApprovalStatus.DENIED
            request.user_decision = "cancelled_all"
            request.resume_event.set()
            count += 1

        logger.info(f"Cancelled {count} pending requests")
        return count

    def set_callback(self, callback: Optional[OnRequestAddedCallback]) -> None:
        """
        Set or update the SSE callback.

        Args:
            callback: New callback function or None to disable
        """
        self._on_request_added = callback

    def __repr__(self) -> str:
        return f"ApprovalQueue(pending={self.pending_count})"
