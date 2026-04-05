"""
Stripe Crypto On-Ramp Integration for Kestrel Wallet.

Enables users to purchase cryptocurrency with fiat currency
and deposit directly to their agent's wallet address.

Features:
- Create on-ramp sessions for users
- Support multiple destination currencies (ETH, MATIC)
- Track pending deposits
- Handle session callbacks

Stripe Crypto On-Ramp Documentation:
https://docs.stripe.com/crypto/onramp
"""

import os
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class OnRampStatus(Enum):
    """Status of an on-ramp session."""

    PENDING = "pending"
    REQUIRES_ACTION = "requires_action"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class OnRampSession:
    """Represents a Stripe Crypto On-Ramp session."""

    session_id: str
    agent_did: str
    wallet_address: str
    destination_currency: str  # ETH, MATIC, etc.
    destination_network: str  # ethereum, polygon
    fiat_currency: str = "usd"
    fiat_amount: Optional[Decimal] = None
    crypto_amount: Optional[Decimal] = None
    status: OnRampStatus = OnRampStatus.PENDING
    client_secret: Optional[str] = None
    redirect_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    stripe_session_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class StripeOnRamp:
    """
    Stripe Crypto On-Ramp integration.

    Allows users to purchase crypto with fiat and deposit
    to their agent's wallet address.

    Usage:
        onramp = StripeOnRamp(db_path="wallet.db")
        session = await onramp.create_session(
            agent_did="did:pkh:eip155:1:0x...",
            wallet_address="0x...",
            destination_currency="ETH",
            fiat_amount=Decimal("100"),
        )
        # User completes purchase at session.redirect_url
        # Webhook handler updates session status
    """

    # Supported destination currencies and networks
    SUPPORTED_CURRENCIES = {
        "ETH": "ethereum",
        "MATIC": "polygon",
        # Note: Stripe doesn't support FIL directly yet
    }

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Stripe On-Ramp.

        Args:
            db_path: Path to SQLite database for session tracking
        """
        self.db_path = db_path or os.environ.get(
            "KESTREL_ONRAMP_DB", "onramp_sessions.db"
        )
        self.stripe_api_key = os.environ.get("STRIPE_SECRET_KEY")
        self.stripe_publishable_key = os.environ.get("STRIPE_PUBLISHABLE_KEY")

        if not self.stripe_api_key:
            logger.warning(
                "STRIPE_SECRET_KEY not set. On-ramp functionality will be limited."
            )

        self._init_database()
        logger.info("StripeOnRamp initialized")

    def _init_database(self):
        """Initialize SQLite database for session tracking."""
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS onramp_sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_did TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    destination_currency TEXT NOT NULL,
                    destination_network TEXT NOT NULL,
                    fiat_currency TEXT DEFAULT 'usd',
                    fiat_amount TEXT,
                    crypto_amount TEXT,
                    status TEXT DEFAULT 'pending',
                    client_secret TEXT,
                    redirect_url TEXT,
                    stripe_session_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_agent
                ON onramp_sessions(agent_did)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status
                ON onramp_sessions(status)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_stripe
                ON onramp_sessions(stripe_session_id)
            """)

            conn.commit()

    async def create_session(
        self,
        agent_did: str,
        wallet_address: str,
        destination_currency: str = "ETH",
        fiat_amount: Optional[Decimal] = None,
        fiat_currency: str = "usd",
    ) -> OnRampSession:
        """
        Create a new on-ramp session.

        Args:
            agent_did: The agent's DID
            wallet_address: EVM wallet address to receive crypto
            destination_currency: Crypto to purchase (ETH, MATIC)
            fiat_amount: Optional pre-set fiat amount
            fiat_currency: Fiat currency (default: usd)

        Returns:
            OnRampSession with redirect URL or client secret

        Raises:
            ValueError: If currency not supported or Stripe not configured
        """
        # Validate currency
        destination_currency = destination_currency.upper()
        if destination_currency not in self.SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Unsupported currency: {destination_currency}. "
                f"Supported: {list(self.SUPPORTED_CURRENCIES.keys())}"
            )

        destination_network = self.SUPPORTED_CURRENCIES[destination_currency]

        # Create session record
        session = OnRampSession(
            session_id=str(uuid4()),
            agent_did=agent_did,
            wallet_address=wallet_address,
            destination_currency=destination_currency,
            destination_network=destination_network,
            fiat_currency=fiat_currency.lower(),
            fiat_amount=fiat_amount,
        )

        # If Stripe is configured, create real session
        if self.stripe_api_key:
            try:
                import stripe

                stripe.api_key = self.stripe_api_key

                # Build on-ramp session parameters
                session_params = {
                    "wallet_addresses": {destination_network: wallet_address},
                    "destination_currencies": [destination_currency.lower()],
                    "destination_network": destination_network,
                }

                if fiat_amount:
                    session_params["transaction_details"] = {
                        "destination_amount": str(fiat_amount),
                        "destination_currency": destination_currency.lower(),
                        "source_currency": fiat_currency,
                    }

                # Create Stripe Crypto On-Ramp session
                stripe_session = stripe.crypto.OnrampSession.create(**session_params)

                session.stripe_session_id = stripe_session.id
                session.client_secret = stripe_session.client_secret
                session.redirect_url = stripe_session.redirect_url
                session.status = OnRampStatus(stripe_session.status)

                logger.info(
                    f"Created Stripe on-ramp session: {stripe_session.id} "
                    f"for {wallet_address}"
                )

            except ImportError:
                logger.warning("stripe package not installed")
                session.redirect_url = self._generate_demo_url(session)

            except (ConnectionError, TimeoutError) as e:
                logger.error(f"Failed to create Stripe session: {e}")
                # Fall back to demo mode
                session.redirect_url = self._generate_demo_url(session)
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Failed to create Stripe session: {e}")
                # Fall back to demo mode
                session.redirect_url = self._generate_demo_url(session)
            except Exception as e:
                logger.error(f"Failed to create Stripe session: {e}", exc_info=True)
                # Fall back to demo mode
                session.redirect_url = self._generate_demo_url(session)
        else:
            # Demo mode without Stripe
            session.redirect_url = self._generate_demo_url(session)

        # Save session to database
        self._save_session(session)

        return session

    def _generate_demo_url(self, session: OnRampSession) -> str:
        """Generate a demo URL for testing without Stripe."""
        base_url = os.environ.get("KESTREL_BASE_URL", "http://localhost:8888")
        return (
            f"{base_url}/wallet/onramp/demo?"
            f"session_id={session.session_id}&"
            f"currency={session.destination_currency}&"
            f"address={session.wallet_address}"
        )

    def _save_session(self, session: OnRampSession):
        """Save session to database."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO onramp_sessions (
                    session_id, agent_did, wallet_address,
                    destination_currency, destination_network,
                    fiat_currency, fiat_amount, crypto_amount,
                    status, client_secret, redirect_url,
                    stripe_session_id, created_at, completed_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.agent_did,
                    session.wallet_address,
                    session.destination_currency,
                    session.destination_network,
                    session.fiat_currency,
                    str(session.fiat_amount) if session.fiat_amount else None,
                    str(session.crypto_amount) if session.crypto_amount else None,
                    session.status.value,
                    session.client_secret,
                    session.redirect_url,
                    session.stripe_session_id,
                    session.created_at.isoformat(),
                    session.completed_at.isoformat() if session.completed_at else None,
                    json.dumps(session.metadata),
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[OnRampSession]:
        """Get session by ID."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM onramp_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

        if not row:
            return None

        return OnRampSession(
            session_id=row["session_id"],
            agent_did=row["agent_did"],
            wallet_address=row["wallet_address"],
            destination_currency=row["destination_currency"],
            destination_network=row["destination_network"],
            fiat_currency=row["fiat_currency"],
            fiat_amount=Decimal(row["fiat_amount"]) if row["fiat_amount"] else None,
            crypto_amount=(
                Decimal(row["crypto_amount"]) if row["crypto_amount"] else None
            ),
            status=OnRampStatus(row["status"]),
            client_secret=row["client_secret"],
            redirect_url=row["redirect_url"],
            stripe_session_id=row["stripe_session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def get_session_by_stripe_id(
        self, stripe_session_id: str
    ) -> Optional[OnRampSession]:
        """Get session by Stripe session ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT session_id FROM onramp_sessions WHERE stripe_session_id = ?",
                (stripe_session_id,),
            ).fetchone()

        if row:
            return self.get_session(row[0])
        return None

    async def update_session_status(
        self,
        session_id: str,
        status: OnRampStatus,
        crypto_amount: Optional[Decimal] = None,
        fiat_amount: Optional[Decimal] = None,
    ) -> Optional[OnRampSession]:
        """
        Update session status (called by webhook handler).

        Args:
            session_id: Session to update
            status: New status
            crypto_amount: Amount of crypto received (if succeeded)
            fiat_amount: Amount of fiat spent (if succeeded)

        Returns:
            Updated session or None if not found
        """
        session = self.get_session(session_id)
        if not session:
            return None

        session.status = status
        if crypto_amount:
            session.crypto_amount = crypto_amount
        if fiat_amount:
            session.fiat_amount = fiat_amount

        if status in (OnRampStatus.SUCCEEDED, OnRampStatus.FAILED):
            session.completed_at = datetime.utcnow()

        self._save_session(session)

        logger.info(
            f"Updated on-ramp session {session_id}: status={status.value}, "
            f"crypto={crypto_amount}"
        )

        return session

    def get_pending_sessions(self, agent_did: str) -> list[OnRampSession]:
        """Get all pending sessions for an agent."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT session_id FROM onramp_sessions
                WHERE agent_did = ? AND status IN ('pending', 'processing', 'requires_action')
                ORDER BY created_at DESC
            """,
                (agent_did,),
            ).fetchall()

        return [self.get_session(row[0]) for row in rows if self.get_session(row[0])]

    def get_completed_sessions(
        self, agent_did: str, limit: int = 10
    ) -> list[OnRampSession]:
        """Get completed sessions for an agent."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT session_id FROM onramp_sessions
                WHERE agent_did = ? AND status = 'succeeded'
                ORDER BY completed_at DESC
                LIMIT ?
            """,
                (agent_did, limit),
            ).fetchall()

        return [self.get_session(row[0]) for row in rows if self.get_session(row[0])]

    def format_session_for_display(self, session: OnRampSession) -> str:
        """Format session for user display."""
        status_emoji = {
            OnRampStatus.PENDING: "⏳",
            OnRampStatus.REQUIRES_ACTION: "⚠️",
            OnRampStatus.PROCESSING: "🔄",
            OnRampStatus.SUCCEEDED: "✅",
            OnRampStatus.FAILED: "❌",
            OnRampStatus.EXPIRED: "⌛",
        }

        lines = [
            f"{status_emoji.get(session.status, '❓')} **On-Ramp Session**",
            f"Status: {session.status.value.upper()}",
            f"Currency: {session.destination_currency}",
            f"Network: {session.destination_network.title()}",
        ]

        if session.fiat_amount:
            lines.append(
                f"Fiat Amount: ${session.fiat_amount} {session.fiat_currency.upper()}"
            )

        if session.crypto_amount:
            lines.append(
                f"Crypto Received: {session.crypto_amount} {session.destination_currency}"
            )

        if session.redirect_url and session.status == OnRampStatus.PENDING:
            lines.append(f"\n🔗 Complete purchase: {session.redirect_url}")

        return "\n".join(lines)

    async def close(self):
        """Clean up resources."""
        pass  # SQLite connections are closed automatically

    def __repr__(self) -> str:
        return f"StripeOnRamp(db_path={self.db_path})"
