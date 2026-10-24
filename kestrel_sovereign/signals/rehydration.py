"""Explicit source-store proof that privacy-elided cognition was withdrawn."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DurableCognitionWithdrawal:
    """A durable source read proves no execution remains owed for this event.

    Missing recovery data remains retryable. Only the authoritative source
    hook may return this explicit terminal no-op, and the dispatcher must
    persist it against the exact managed lease before publishing a receipt.
    """

    reason: str

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Durable cognition withdrawal requires a concrete reason")
