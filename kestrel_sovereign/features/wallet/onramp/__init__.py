"""
Fiat On-Ramp Package for Kestrel Wallet.

Provides integrations for users to fund their agent's wallet
with fiat currency (credit card, bank transfer, etc.).

Currently supports:
- Stripe Crypto On-Ramp
"""

from .stripe_onramp import StripeOnRamp, OnRampSession, OnRampStatus
from .webhook_handler import StripeWebhookHandler

__all__ = [
    "StripeOnRamp",
    "OnRampSession",
    "OnRampStatus",
    "StripeWebhookHandler",
]
