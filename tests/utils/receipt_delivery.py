"""Weak-referenceable sparse delivery fixture for native receipt-owner tests."""
from types import SimpleNamespace


class ReceiptDelivery(SimpleNamespace):
    """Native DurableDelivery snapshots support the same weak ownership seam."""
