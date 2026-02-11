"""
Stripe Webhook Handler for Crypto On-Ramp.

Handles webhook events from Stripe when on-ramp sessions
complete, fail, or require action.

Webhook Events:
- crypto.onramp_session.updated - Session status changed
- crypto.onramp_session.completed - Purchase completed successfully

Security:
- Validates webhook signatures using STRIPE_WEBHOOK_SECRET
- Logs all webhook events for audit trail
- Idempotent processing (safe to receive duplicate events)
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Callable, Awaitable

from .stripe_onramp import StripeOnRamp, OnRampSession, OnRampStatus

logger = logging.getLogger(__name__)


@dataclass
class WebhookResult:
    """Result of processing a webhook."""

    success: bool
    session_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None


class StripeWebhookHandler:
    """
    Handles Stripe webhook events for crypto on-ramp.

    Usage:
        handler = StripeWebhookHandler(onramp)
        handler.on_deposit_complete = my_callback

        # In FastAPI endpoint:
        @app.post("/webhooks/stripe/crypto")
        async def stripe_webhook(request: Request):
            result = await handler.handle_webhook(
                await request.body(),
                request.headers.get("Stripe-Signature")
            )
            return {"status": "received" if result.success else "error"}
    """

    def __init__(self, onramp: StripeOnRamp):
        """
        Initialize webhook handler.

        Args:
            onramp: StripeOnRamp instance for session updates

        Raises:
            ValueError: If STRIPE_WEBHOOK_SECRET is not configured
        """
        self.onramp = onramp
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

        if not self.webhook_secret:
            raise ValueError(
                "STRIPE_WEBHOOK_SECRET must be set. Webhook handler cannot "
                "process events without signature verification. Set the "
                "STRIPE_WEBHOOK_SECRET environment variable to enable secure "
                "webhook processing."
            )

        # Callback for when a deposit completes successfully
        # Signature: async def callback(session: OnRampSession) -> None
        self.on_deposit_complete: Optional[
            Callable[[OnRampSession], Awaitable[None]]
        ] = None

        # Callback for when a deposit fails
        self.on_deposit_failed: Optional[
            Callable[[OnRampSession], Awaitable[None]]
        ] = None

        logger.info("StripeWebhookHandler initialized")

    async def handle_webhook(
        self, payload: bytes, signature: Optional[str]
    ) -> WebhookResult:
        """
        Handle an incoming Stripe webhook.

        Args:
            payload: Raw request body
            signature: Stripe-Signature header value

        Returns:
            WebhookResult with success/failure status
        """
        try:
            event = self._verify_and_parse_event(payload, signature)
            if not event:
                return WebhookResult(
                    success=False, error="Failed to verify webhook signature"
                )

            event_type = event.get("type", "")
            logger.info(f"Received Stripe webhook: {event_type}")

            # Route to appropriate handler
            if event_type == "crypto.onramp_session.updated":
                return await self._handle_session_updated(event)
            elif event_type == "crypto.onramp_session.completed":
                return await self._handle_session_completed(event)
            else:
                logger.debug(f"Ignoring unhandled event type: {event_type}")
                return WebhookResult(
                    success=True, message=f"Event type {event_type} ignored"
                )

        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return WebhookResult(success=False, error=str(e))

    def _verify_and_parse_event(
        self, payload: bytes, signature: Optional[str]
    ) -> Optional[dict]:
        """
        Verify webhook signature and parse event.

        Args:
            payload: Raw request body
            signature: Stripe-Signature header

        Returns:
            Parsed event dict or None if verification fails
        """
        try:
            import stripe

            if not signature:
                logger.error("Missing Stripe-Signature header")
                return None

            # Verify signature (mandatory for security)
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return event

        except ImportError as e:
            logger.error(f"stripe package not installed: {e}")
            return None

        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return None

        except Exception as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            return None

    async def _handle_session_updated(self, event: dict) -> WebhookResult:
        """
        Handle crypto.onramp_session.updated event.

        This is called when the session status changes (e.g., processing started).
        """
        session_data = event.get("data", {}).get("object", {})
        stripe_session_id = session_data.get("id")

        if not stripe_session_id:
            return WebhookResult(success=False, error="Missing session ID in event")

        # Find our session record
        session = self.onramp.get_session_by_stripe_id(stripe_session_id)
        if not session:
            logger.warning(f"Unknown Stripe session: {stripe_session_id}")
            return WebhookResult(
                success=True, message="Session not found (may be from different app)"
            )

        # Map Stripe status to our status
        stripe_status = session_data.get("status", "").lower()
        status_map = {
            "pending": OnRampStatus.PENDING,
            "requires_action": OnRampStatus.REQUIRES_ACTION,
            "processing": OnRampStatus.PROCESSING,
            "succeeded": OnRampStatus.SUCCEEDED,
            "failed": OnRampStatus.FAILED,
            "expired": OnRampStatus.EXPIRED,
        }

        new_status = status_map.get(stripe_status)
        if not new_status:
            logger.warning(f"Unknown Stripe status: {stripe_status}")
            return WebhookResult(success=True, message=f"Unknown status: {stripe_status}")

        # Extract amounts if present
        crypto_amount = None
        fiat_amount = None

        transaction = session_data.get("transaction_details", {})
        if transaction.get("destination_amount"):
            crypto_amount = Decimal(transaction["destination_amount"])
        if transaction.get("source_amount"):
            fiat_amount = Decimal(transaction["source_amount"])

        # Update session
        updated_session = await self.onramp.update_session_status(
            session.session_id,
            new_status,
            crypto_amount=crypto_amount,
            fiat_amount=fiat_amount,
        )

        logger.info(
            f"Updated session {session.session_id}: {stripe_status} "
            f"(crypto: {crypto_amount})"
        )

        return WebhookResult(
            success=True,
            session_id=session.session_id,
            message=f"Session updated to {new_status.value}",
        )

    async def _handle_session_completed(self, event: dict) -> WebhookResult:
        """
        Handle crypto.onramp_session.completed event.

        This is called when the purchase is successfully completed
        and crypto has been deposited to the wallet.
        """
        session_data = event.get("data", {}).get("object", {})
        stripe_session_id = session_data.get("id")

        if not stripe_session_id:
            return WebhookResult(success=False, error="Missing session ID in event")

        # Find our session record
        session = self.onramp.get_session_by_stripe_id(stripe_session_id)
        if not session:
            logger.warning(f"Unknown Stripe session: {stripe_session_id}")
            return WebhookResult(
                success=True, message="Session not found (may be from different app)"
            )

        # Extract amounts
        transaction = session_data.get("transaction_details", {})
        crypto_amount = None
        fiat_amount = None

        if transaction.get("destination_amount"):
            crypto_amount = Decimal(transaction["destination_amount"])
        if transaction.get("source_amount"):
            fiat_amount = Decimal(transaction["source_amount"])

        # Update session to succeeded
        updated_session = await self.onramp.update_session_status(
            session.session_id,
            OnRampStatus.SUCCEEDED,
            crypto_amount=crypto_amount,
            fiat_amount=fiat_amount,
        )

        logger.info(
            f"On-ramp completed for session {session.session_id}: "
            f"{crypto_amount} {session.destination_currency} deposited to "
            f"{session.wallet_address}"
        )

        # Call success callback if registered
        if self.on_deposit_complete and updated_session:
            try:
                await self.on_deposit_complete(updated_session)
            except Exception as e:
                logger.error(f"Deposit callback failed: {e}")

        return WebhookResult(
            success=True,
            session_id=session.session_id,
            message=f"Deposit completed: {crypto_amount} {session.destination_currency}",
        )

    async def handle_failed_session(
        self, stripe_session_id: str, reason: str
    ) -> WebhookResult:
        """
        Handle a failed on-ramp session.

        Args:
            stripe_session_id: Stripe session that failed
            reason: Failure reason

        Returns:
            WebhookResult with status
        """
        session = self.onramp.get_session_by_stripe_id(stripe_session_id)
        if not session:
            return WebhookResult(success=True, message="Session not found")

        updated_session = await self.onramp.update_session_status(
            session.session_id, OnRampStatus.FAILED
        )

        logger.warning(
            f"On-ramp failed for session {session.session_id}: {reason}"
        )

        # Call failure callback if registered
        if self.on_deposit_failed and updated_session:
            try:
                await self.on_deposit_failed(updated_session)
            except Exception as e:
                logger.error(f"Failure callback failed: {e}")

        return WebhookResult(
            success=True,
            session_id=session.session_id,
            message=f"Session marked as failed: {reason}",
        )

    def __repr__(self) -> str:
        return f"StripeWebhookHandler(onramp={self.onramp})"
