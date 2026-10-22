"""Explicit host configuration for the native durable push service."""

from collections.abc import Mapping
from dataclasses import dataclass, fields

from kestrel_sovereign.storage.db.polling import DatabasePollRetryPolicy


@dataclass(frozen=True)
class NativePushPolicy:
    """One source for operational defaults, validated before host publication."""

    retention_days: int = 30
    lease_seconds: int = 60
    timeout_seconds: int = 30
    poll_seconds: int = 1
    renewal_seconds: int = 20
    retry_seconds: int = 5
    retry_max_seconds: int = 300
    delivery_retry_attempts: int = 8
    batch_size: int = 50
    database_retry_attempts: int = DatabasePollRetryPolicy().max_attempts

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Native push {field.name} must be a positive integer")
        if self.renewal_seconds >= self.lease_seconds:
            raise ValueError(
                "Native push renewal_seconds must be shorter than lease_seconds"
            )
        if self.retry_max_seconds < self.retry_seconds:
            raise ValueError("Native push retry_max_seconds must be at least retry_seconds")

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]):
        """Hosted native recipients enable push unless explicitly disabled."""
        enabled = environment.get("KESTREL_A2A_PUSH_ENABLED", "true").lower()
        if enabled not in {"true", "false"}:
            raise ValueError("KESTREL_A2A_PUSH_ENABLED must be true or false")
        if enabled == "false":
            return None
        overrides = {}
        for field in fields(cls):
            name = "KESTREL_A2A_PUSH_" + field.name.upper()
            if name in environment:
                try:
                    overrides[field.name] = int(environment[name])
                except ValueError as error:
                    raise ValueError(f"{name} must be a positive integer") from error
        return cls(**overrides)
