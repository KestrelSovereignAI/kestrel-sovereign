"""A backup target's storage quota, as the target last read it (#3894).

The host's Lighthouse account filled its 500 GiB plan (538.4 GB used against a
536.9 GB limit) and every snapshot upload then failed as a bare ``ReadError``
for days, with nothing naming the quota. A target whose account has a quota
reads the account's usage before it uploads, prunes under retention first
when the upload would cross ``threshold`` of the limit, names the quota when
an upload fails for lack of room, and keeps what it read as
``SyncTarget.storage_quota``. The sync service persists that reading and the
health check reports a target near or over its quota.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

#: The fraction of a target's limit at which it counts as near quota: the
#: upload precheck prunes first and the health check warns.
DEFAULT_QUOTA_THRESHOLD = 0.9


def format_bytes(value: int) -> str:
    """``value`` in decimal units (the GB a storage plan is sold in)."""
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(amount) < 1000:
            return f"{int(amount)} B" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1000
    return f"{amount:.1f} TB"


def validate_quota_threshold(threshold: Any) -> float:
    """``threshold`` as a fraction of a limit in (0, 1]; anything else raises
    ``ValueError``."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError(
            f"backup quota threshold must be a number, got {threshold!r}"
        )
    value = float(threshold)
    if not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError(
            "backup quota threshold must be a fraction of the limit in (0, 1], "
            f"got {threshold!r}"
        )
    return value


@dataclass(frozen=True)
class StorageQuota:
    """An account's usage against its limit, read at ``observed_at``."""

    used_bytes: int
    limit_bytes: int
    observed_at: datetime
    threshold: float = DEFAULT_QUOTA_THRESHOLD

    def __post_init__(self) -> None:
        for name in ("used_bytes", "limit_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.used_bytes < 0:
            raise ValueError(f"used_bytes must not be negative, got {self.used_bytes}")
        if self.limit_bytes <= 0:
            raise ValueError(f"limit_bytes must be positive, got {self.limit_bytes}")
        object.__setattr__(self, "threshold", validate_quota_threshold(self.threshold))
        observed_at = self.observed_at
        if not isinstance(observed_at, datetime):
            raise ValueError(f"observed_at must be a datetime, got {observed_at!r}")
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "observed_at", observed_at.astimezone(timezone.utc))

    @property
    def used_fraction(self) -> float:
        return self.used_bytes / self.limit_bytes

    @property
    def over_quota(self) -> bool:
        """The account has no room left."""
        return self.used_bytes >= self.limit_bytes

    @property
    def near_quota(self) -> bool:
        """The account has reached ``threshold`` of its limit."""
        return self.used_bytes >= self.threshold * self.limit_bytes

    def would_cross_threshold(self, upload_bytes: int) -> bool:
        """Uploading ``upload_bytes`` would take the account past ``threshold``."""
        return self.used_bytes + upload_bytes > self.threshold * self.limit_bytes

    def fits(self, upload_bytes: int) -> bool:
        """``upload_bytes`` more stays within the limit."""
        return self.used_bytes + upload_bytes <= self.limit_bytes

    def describe(self) -> str:
        return (
            f"{format_bytes(self.used_bytes)} used of "
            f"{format_bytes(self.limit_bytes)} limit ({self.used_fraction:.1%})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "used_bytes": self.used_bytes,
            "limit_bytes": self.limit_bytes,
            "observed_at": self.observed_at.isoformat(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["StorageQuota"]:
        """The quota ``to_dict`` wrote, or ``None`` for anything else."""
        if not isinstance(data, Mapping):
            return None
        try:
            return cls(
                used_bytes=data["used_bytes"],
                limit_bytes=data["limit_bytes"],
                observed_at=datetime.fromisoformat(str(data["observed_at"])),
                threshold=data.get("threshold", DEFAULT_QUOTA_THRESHOLD),
            )
        except (KeyError, TypeError, ValueError):
            return None


class StorageQuotaExceeded(Exception):
    """An upload failed and the account has no room for it.

    The message names the account's usage against its limit; the transport
    error the upload raised is chained as ``__cause__`` and named in the
    message too.
    """

    def __init__(
        self,
        quota: StorageQuota,
        upload_bytes: int,
        *,
        account: str,
        used_label: str,
        limit_label: str,
        cause: BaseException,
    ):
        self.quota = quota
        self.upload_bytes = upload_bytes
        state = "over" if quota.over_quota else "near"
        super().__init__(
            f"{account} account {state} quota: {used_label} "
            f"{format_bytes(quota.used_bytes)} "
            f"against {limit_label} {format_bytes(quota.limit_bytes)} "
            f"({quota.used_fraction:.1%}); a {format_bytes(upload_bytes)} upload "
            f"does not fit (upload failed: {type(cause).__name__}: {cause})"
        )


def load_quota_threshold() -> float:
    """``[backup.quota] threshold`` from the host config, else the default.

    A value that is not a fraction in (0, 1] raises ``ValueError``.
    """
    try:
        from kestrel_sovereign.config import load_section

        backup = load_section("backup") or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load backup quota config: %s", e)
        backup = {}
    quota = backup.get("quota") if isinstance(backup, Mapping) else None
    if not isinstance(quota, Mapping) or "threshold" not in quota:
        return DEFAULT_QUOTA_THRESHOLD
    return validate_quota_threshold(quota["threshold"])
