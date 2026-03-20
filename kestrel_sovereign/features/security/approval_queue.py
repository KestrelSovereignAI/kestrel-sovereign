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
from typing import Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

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
    timeout_seconds: float = 300.0  # 5 minute default
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


# Type for the SSE callback
OnRequestAddedCallback = Callable[[ApprovalRequest], Awaitable[None]]


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
    ):
        """
        Initialize the approval queue.

        Args:
            on_request_added: Optional async callback when a request is added.
                             Used to emit SSE events to the UI.
        """
        self._pending: Dict[str, ApprovalRequest] = {}
        self._on_request_added = on_request_added

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
        timeout: float = 300.0,
    ) -> tuple[bool, str]:
        """
        Queue a request and wait for user decision.

        This method blocks until the user approves, denies, or the
        request times out.

        Args:
            feature_name: Name of the feature making the request
            tool_name: Name of the tool requesting approval
            tool_args: Arguments to the tool (shown to user)
            timeout: Timeout in seconds (default 5 minutes)

        Returns:
            Tuple of (approved: bool, scope: str)
            - approved: True if user approved, False if denied or timeout
            - scope: "once", "session", "always", or "timeout"
        """
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

        # Wait for user decision or timeout
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
            logger.warning(
                f"Approval request {request.id[:8]} timed out after {timeout}s"
            )
            return (False, "timeout")

        finally:
            self._pending.pop(request.id, None)

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

        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.user_decision = scope
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
        return self._pending.get(request_id)

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
