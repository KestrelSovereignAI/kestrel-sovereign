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

from .args_summary import summarize_args

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
    # The original tool-call awaiter was cancelled (Cloud Run request
    # timeout, SSE disconnect, client abort) before a decision landed.
    # Distinct from PENDING (still awaitable), APPROVED/DENIED (decision
    # landed while the awaiter was alive), and TIMEOUT (removed by the
    # awaiter itself). A late decision on an AWAITER_GONE request still
    # persists the scope rule for future calls, but the in-flight call is
    # orphaned — see submit_decision / request_approval. (#2558)
    AWAITER_GONE = "awaiter_gone"


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
    # Approval scope ("once"/"session"/"always") on approve, or the
    # denial provenance on deny: "user_denied" (a human pressed deny via
    # the deny tool) vs "cancelled"/"cancelled_all" (task torn down, not a
    # user decision). Distinct from the operator/auto policy "denied" that
    # ``request_approval`` early-returns without ever creating a request.
    user_decision: Optional[str] = None
    # Which kind of call this was: a tool execution, or a feature-as-subagent
    # DISPATCH (#3107). Carried on the request because the decision may be
    # persisted minutes later, on a different task, long after the hook that
    # knew which event fired has returned. Both writers must agree — a search
    # that excludes dispatch envelopes is only correct if EVERY door labels
    # them.
    audit_action: str = "tool_execution"

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


@dataclass(frozen=True)
class DecisionResult:
    """Result of applying a user's approval-queue decision."""

    in_memory: bool
    persisted: bool
    error: Optional[str] = None
    # True when the decision was accepted for a request whose original
    # tool-call awaiter had already been cancelled (AWAITER_GONE). The
    # scope rule is still persisted (future calls benefit), but the
    # in-flight call is orphaned — the frontend should prompt a retry.
    # (#2558)
    awaiter_gone: bool = False

    def __bool__(self) -> bool:
        """Back-compat truthiness for callers that only need acceptance."""
        return self.in_memory


@dataclass(frozen=True)
class DenialClassification:
    """Provenance of an ``approved=False`` result from ``request_approval``.

    ``request_approval`` returns ``(False, scope)`` for several distinct
    reasons, and only one of them — an explicit human "deny" — is a real
    user denial. Every consumer that reports the outcome (SecurityHook,
    ComputeFeature, KeysFeature, the CLI, the Talon verifier, …) must branch
    on the provenance so a policy block, a headless ``no_approver``
    short-circuit (#2029), a timeout, or a torn-down task is never mislabeled
    as "the user said no" (#1542). This is the single source of truth for that
    classification so the consumers can't drift apart.
    """

    scope: str
    is_user_denial: bool
    reason: str        # machine category, suitable for audit/metadata
    description: str   # human-readable phrase for messages


# Non-user-denial provenance scopes → (audit reason, human description). Any
# scope NOT in this map (notably "user_denied", or a user-chosen approve scope
# returned with approved=False) is treated as a genuine user denial.
_NON_USER_DENIAL_SCOPES: Dict[str, tuple] = {
    "denied": ("policy_denied", "blocked by an operator/auto policy"),
    "no_approver": (
        "no_approver",
        "no interactive approver is available (headless/test instance)",
    ),
    "timeout": ("timeout", "the approval request timed out"),
    "cancelled": ("cancelled", "the approval request was cancelled"),
    "cancelled_all": ("cancelled", "the approval request was cancelled"),
}


def classify_denial(scope: Optional[str]) -> DenialClassification:
    """Classify the ``scope`` of an ``approved=False`` approval result.

    Returns a :class:`DenialClassification` whose ``is_user_denial`` is True
    only for an explicit human deny. Consumers should report
    ``denied_by_user`` / "denied by user" strictly off ``is_user_denial`` and
    use ``reason`` / ``description`` for everything else.
    """
    key = scope or ""
    if key in _NON_USER_DENIAL_SCOPES:
        reason, description = _NON_USER_DENIAL_SCOPES[key]
        return DenialClassification(
            scope=key,
            is_user_denial=False,
            reason=reason,
            description=description,
        )
    return DenialClassification(
        scope=key or "user_denied",
        is_user_denial=True,
        reason="user_denied",
        description="denied by the user",
    )


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
                             rows.  When set, every submitted approval
                             decision is persisted/audited centrally so that
                             callers don't have to remember to do it (#785).
                             When None, persistence is treated as unnecessary.
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
        *,
        allow_blocking: bool = True,
        audit_action: str = "tool_execution",
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
            allow_blocking: When ``False``, a request that would otherwise
                queue-and-wait for an interactive approver instead returns a
                non-blocking ``(False, "no_approver")`` immediately. This is for
                background/non-interactive callers (e.g. a scheduler tick, #2111)
                that have no human attached to the approval queue and must not
                wedge their loop waiting forever — the same guarantee the #2029
                headless-test guard gives, generalized to any non-interactive
                caller regardless of ``is_test_instance``. The AUTO fast-path and
                policy ALLOW/DENY still resolve normally before this applies.

        Returns:
            Tuple of (approved: bool, scope: str)
            - approved: True if user approved, False if denied or timeout
            - scope: on approve, "once"/"session"/"always" (or "auto" /
              "auto_approve:<id>" for policy auto-approval). On deny, the
              scope carries provenance: "user_denied" for an explicit human
              denial, "denied" for an operator/auto policy DENY,
              "no_approver" when the agent is a headless test instance with
              no one to answer the queue (#2029), and
              "timeout"/"cancelled"/"cancelled_all" when no user ever
              decided. Consumers MUST NOT treat anything but "user_denied"
              (or a user-chosen approve scope returned with approved=False)
              as a user denial (#1542).
        """
        always_ask = False
        if self._permission_store is not None:
            try:
                from .permissions import PermissionLevel

                level = await self._permission_store.get_permission(
                    feature_name,
                    tool_name,
                )
                always_ask = level == PermissionLevel.ALWAYS_ASK
                if level == PermissionLevel.DENY:
                    await self._permission_store.log_decision(
                        feature_name=feature_name,
                        tool_name=tool_name,
                        action=audit_action,
                        decision="auto_denied",
                        args_summary=summarize_args(tool_args),
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
                        action=audit_action,
                        decision="auto_mode_allowed",
                        user_choice="constitutional_honesty_unflagged",
                        args_summary=summarize_args(tool_args),
                    )
                    logger.info(
                        "ApprovalQueue auto-mode approved %s.%s without prompting",
                        feature_name,
                        tool_name,
                    )
                    return (True, "auto")
            except Exception as e:  # noqa: BLE001
                # Fail CLOSED: a security gate must not auto-approve on a
                # store-read error. We cannot confirm the tool is NOT
                # ALWAYS_ASK, so assume it is — this skips the scoped
                # auto-approve path below and queues a human (#2056).
                always_ask = True
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
        if self._auto_approve_policy is not None and not always_ask:
            try:
                from .permissions import PermissionLevel

                # The internal computer_use gate keys this call as
                # "computer_use.shell", but the permissions UI registers /
                # denies it under the canonical class name
                # "ComputerUseFeature.shell". An operator DENY on the
                # canonical key MUST still hard-stop the auto-approve path,
                # otherwise revocation is ineffective for the exact
                # commands being auto-approved (codex review P1, #1290).
                _alias = {"computer_use": "ComputerUseFeature"}
                _deny_keys = {feature_name}
                if feature_name in _alias:
                    _deny_keys.add(_alias[feature_name])
                _denied = False
                if self._permission_store is not None:
                    for _fk in _deny_keys:
                        if await self._permission_store.get_permission(
                            _fk, tool_name
                        ) == PermissionLevel.DENY:
                            _denied = True
                            break
                if _denied:
                    await self._permission_store.log_decision(
                        feature_name=feature_name,
                        tool_name=tool_name,
                        action=audit_action,
                        decision="auto_denied",
                        user_choice="canonical_deny",
                        args_summary=summarize_args(tool_args),
                    )
                    logger.info(
                        "ApprovalQueue: canonical DENY blocks auto-approve "
                        "for %s.%s", feature_name, tool_name,
                    )
                    return (False, "denied")

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
                        action=audit_action,
                        decision="auto_approved",
                        user_choice=f"auto_approve:{match.rule.source}",
                        args_summary=summarize_args(tool_args),
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

        # Non-interactive guard (#2029). A tagged test instance is headless by
        # definition — driven via ``kestrel ask`` / the API, with no human
        # attached to answer the approval queue. If we reach this point the
        # request would block on ``resume_event`` indefinitely (the default
        # ``timeout`` is ``None`` = wait forever), wedging the agent's request
        # worker until restart. That is the exact #2029 hang: a single
        # ``spawn_agent`` call (ASK-gated) bricks the agent. Reaching here also
        # means global auto-mode is OFF — the AUTO fast-path above returns
        # before this — i.e. the operator did NOT opt this test instance into
        # non-interactive auto-approve (KESTREL_TEST_AUTO_APPROVE, #1936). So
        # there is genuinely no approver: return an honest, non-blocking
        # ``no_approver`` result instead of queuing forever. Production /
        # sovereign agents (``is_test_instance`` falsy) are unaffected — their
        # Sovereign answers asynchronously via the Mews approval panel, so a
        # pending request legitimately stays open for them — UNLESS the caller
        # is non-interactive (``allow_blocking=False``, e.g. a scheduler tick,
        # #2111): there is no request/response cycle a Sovereign would answer in
        # that context, and blocking would wedge the background loop forever.
        non_interactive = not allow_blocking
        headless_test = bool(getattr(self._agent, "is_test_instance", False))
        if non_interactive or headless_test:
            await self._persist_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                tool_args=tool_args,
                approved=False,
                scope="no_approver",
                audit_action=audit_action,
            )
            logger.warning(
                "ApprovalQueue: %s.%s requires approval but no interactive "
                "approver is available (%s); returning non-blocking "
                "'no_approver' instead of queuing.",
                feature_name,
                tool_name,
                "non-interactive caller" if non_interactive else
                "headless test instance — set KESTREL_TEST_AUTO_APPROVE=1 to "
                "auto-approve ASK-level tools",
            )
            return (False, "no_approver")

        request = ApprovalRequest(
            id=str(uuid4()),
            feature_name=feature_name,
            tool_name=tool_name,
            tool_args=tool_args,
            created_at=datetime.now(timezone.utc),
            timeout_seconds=timeout,
            audit_action=audit_action,
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
                audit_action=audit_action,
            )
            return (False, "timeout")

        except asyncio.CancelledError:
            # Calling task cancelled (HTTP stream dropped, browser
            # closed, user switched to a different agent in the
            # multi_agent). The user has not yet decided. Leave the
            # request in ``_pending`` so the modal stays interactive,
            # and re-raise without firing withdrawal — the modal must
            # NOT auto-close on us. Mark the request AWAITER_GONE so a
            # late ``submit_decision`` can tell the caller its click did
            # nothing for the orphaned in-flight call (#2558).
            # Clobber unconditionally: if a concurrent ``submit_decision``
            # already flipped to APPROVED/DENIED and is mid-persist, the
            # tool call itself is STILL dead — the persisted scope rule
            # benefits future calls, but this call is orphaned regardless
            # of who won the state-write race. ``submit_decision``'s
            # re-check-after-persist reads AWAITER_GONE and reports
            # ``awaiter_gone=True`` to the caller so the frontend prompts
            # a retry.
            request.status = ApprovalStatus.AWAITER_GONE
            logger.info(
                f"Approval request {request.id[:8]} await cancelled; "
                "request marked AWAITER_GONE — decision will still be "
                "recorded but the tool call is orphaned"
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
        audit_action: str = "tool_execution",
    ) -> Optional[str]:
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
            return None

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
            elif scope == "no_approver":
                # Headless/no-approver block (#2029) — emphatically NOT a
                # user denial. Audited distinctly so operators (and the Talon
                # verifier's classify_denial) never mislabel it as one.
                decision = "no_approver"
            else:
                decision = "user_denied"

            args_summary = summarize_args(tool_args)
            await self._permission_store.log_decision(
                feature_name=feature_name,
                tool_name=tool_name,
                action=audit_action,
                decision=decision,
                user_choice=scope,
                args_summary=args_summary,
            )
        except Exception as e:  # noqa: BLE001
            # A persistence failure must not corrupt the user's decision.
            # Log loudly and let the caller report the exact persistence
            # failure while preserving the in-memory decision.
            logger.warning(
                "ApprovalQueue: failed to persist decision for "
                f"{feature_name}.{tool_name}: {e}",
                exc_info=True,
            )
            return str(e)
        return None

    async def submit_decision(
        self,
        request_id: str,
        approved: bool,
        scope: str = "once",
    ) -> DecisionResult:
        """
        Submit a user's decision for a pending request.

        Called by the API when the user makes a decision in the UI.

        Args:
            request_id: ID of the pending request
            approved: Whether the user approved the request
            scope: Scope of approval - "once", "session", or "always"

        Returns:
            DecisionResult with ``in_memory=True`` when the pending request
            accepted the decision, and ``persisted=True`` only when the
            scope/audit persistence path completed or was unnecessary.
        """
        request = self._pending.get(request_id)
        if not request:
            logger.warning(f"Decision submitted for unknown request: {request_id}")
            return DecisionResult(
                in_memory=False,
                persisted=False,
                error="request not found or expired",
            )

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
        # AWAITER_GONE requests are still decidable: the awaiter died
        # (Cloud Run timeout / SSE disconnect / abort) but the request
        # was deliberately kept in ``_pending`` so a late click still
        # persists the scope rule for future calls. We accept it here,
        # but the in-flight call is orphaned — signal that via
        # ``awaiter_gone`` so the frontend can prompt a retry (#2558).
        awaiter_gone = request.status == ApprovalStatus.AWAITER_GONE
        if request.status != ApprovalStatus.PENDING and not awaiter_gone:
            logger.warning(
                f"Decision submitted for already-decided request "
                f"{request_id[:8]} (status={request.status.value}); ignored"
            )
            return DecisionResult(
                in_memory=False,
                persisted=False,
                error=f"request already {request.status.value}",
            )

        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.user_decision = scope
        self._resolved[request_id] = request
        if len(self._resolved) > 512:
            oldest = next(iter(self._resolved))
            self._resolved.pop(oldest, None)

        persist_error = await self._persist_decision(
            feature_name=request.feature_name,
            tool_name=request.tool_name,
            tool_args=request.tool_args,
            approved=approved,
            scope=scope,
            audit_action=request.audit_action,
        )

        # Re-read status after the persist await: a concurrent awaiter
        # cancellation may have flipped PENDING → AWAITER_GONE while we
        # were persisting. Codex round-1 P0: without this refresh, a
        # late-approve-then-cancel race left ``awaiter_gone=False`` for
        # the caller (frontend never prompts a retry) and the request
        # stranded in ``_pending`` because the AWAITER_GONE pop branch
        # was skipped.
        if not awaiter_gone and request.status == ApprovalStatus.AWAITER_GONE:
            awaiter_gone = True

        if not awaiter_gone:
            request.resume_event.set()  # Unblock the waiting coroutine

        # Always pop — if a live awaiter beat us to it via its finally
        # block, this is a no-op; if the awaiter is dead, this is the
        # only cleanup path. Cheaper than trying to detect awaiter
        # liveness.
        self._pending.pop(request_id, None)

        logger.info(
            f"Decision submitted for {request_id[:8]}: "
            f"{'approved' if approved else 'denied'} ({scope})"
            f"{' [awaiter gone — call orphaned]' if awaiter_gone else ''}"
        )

        return DecisionResult(
            in_memory=True,
            persisted=persist_error is None,
            error=persist_error,
            awaiter_gone=awaiter_gone,
        )

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

        # An AWAITER_GONE request has no live waiter — its awaiter died and
        # left the request in ``_pending`` deliberately. Setting
        # ``resume_event`` would unblock nobody, and ``sweep_stale`` skips
        # entries whose event is set, so the request would linger forever.
        # Pop it ourselves instead. (#2558)
        awaiter_gone = request.status == ApprovalStatus.AWAITER_GONE
        request.status = ApprovalStatus.DENIED
        request.user_decision = "cancelled"
        if awaiter_gone:
            self._pending.pop(request_id, None)
        else:
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
        for request_id, request in list(self._pending.items()):
            # AWAITER_GONE requests have no live waiter to pop them — see
            # cancel_request. Pop them directly so they don't linger. (#2558)
            awaiter_gone = request.status == ApprovalStatus.AWAITER_GONE
            request.status = ApprovalStatus.DENIED
            request.user_decision = "cancelled_all"
            if awaiter_gone:
                self._pending.pop(request_id, None)
            else:
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
