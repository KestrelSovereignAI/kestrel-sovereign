"""Backup retention policy shared by sync targets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


class DataClass(str, Enum):
    WORKING_MEMORY = "working_memory"
    IDENTITY = "identity"


IDENTITY_TOKENS = (
    "did",
    "constitution",
    "constitutional",
    "root-key",
    "root_key",
    "bootstrap",
    "personality",
    "migration",
    "manifest",
    "sovereignty",
    "export",
)

WORKING_MEMORY_TOKENS = (
    "kestrel_prime.db",
    "latest.db",
    "latest_snapshot",
    ".db",
    ".wal",
    "-wal",
)


@dataclass(frozen=True)
class RetentionClassPolicy:
    keep_all_days: int
    weekly_until_months: int | None = None
    weekly_forever: bool = False
    monthly_forever: bool = False


@dataclass(frozen=True)
class RetentionDecision:
    item: "RetentionItem"
    keep: bool
    reason: str


@dataclass(frozen=True)
class RetentionItem:
    key: str
    timestamp: datetime
    data_class: DataClass
    name: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        ts = self.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
            object.__setattr__(self, "timestamp", ts)


DEFAULT_RETENTION_POLICY = {
    DataClass.WORKING_MEMORY: RetentionClassPolicy(
        keep_all_days=14,
        weekly_forever=True,
    ),
    DataClass.IDENTITY: RetentionClassPolicy(
        keep_all_days=30,
        weekly_until_months=12,
        monthly_forever=True,
    ),
}


class RetentionPolicy:
    def __init__(
        self,
        policies: Mapping[DataClass, RetentionClassPolicy] | None = None,
    ):
        self.policies = dict(policies or DEFAULT_RETENTION_POLICY)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RetentionPolicy":
        config = config or {}
        backup = config.get("backup", config)
        retention = (
            backup.get("retention", backup) if isinstance(backup, Mapping) else {}
        )
        policies: dict[DataClass, RetentionClassPolicy] = {}
        for data_class, default in DEFAULT_RETENTION_POLICY.items():
            override = retention.get(data_class.value, {})
            if not isinstance(override, Mapping):
                override = {}
            policies[data_class] = RetentionClassPolicy(
                keep_all_days=int(override.get("keep_all_days", default.keep_all_days)),
                weekly_until_months=_optional_int(
                    override.get("weekly_until_months", default.weekly_until_months)
                ),
                weekly_forever=bool(
                    override.get("weekly_forever", default.weekly_forever)
                ),
                monthly_forever=bool(
                    override.get("monthly_forever", default.monthly_forever)
                ),
            )
        return cls(policies)

    def decide(
        self,
        items: Iterable[RetentionItem],
        *,
        now: datetime | None = None,
    ) -> list[RetentionDecision]:
        now = _normalize_ts(now or datetime.now(timezone.utc))
        items_by_class: dict[DataClass, list[RetentionItem]] = {}
        for item in items:
            items_by_class.setdefault(item.data_class, []).append(item)

        decisions: list[RetentionDecision] = []
        for data_class, class_items in items_by_class.items():
            newest = max(class_items, key=lambda item: item.timestamp, default=None)
            policy = self.policies.get(data_class, DEFAULT_RETENTION_POLICY[data_class])
            weekly_keep: dict[tuple[int, int], RetentionItem] = {}
            monthly_keep: dict[tuple[int, int], RetentionItem] = {}

            initial: dict[str, RetentionDecision] = {}
            for item in sorted(class_items, key=lambda i: i.timestamp, reverse=True):
                age_days = (now - _normalize_ts(item.timestamp)).total_seconds() / 86400
                if item is newest:
                    initial[item.key] = RetentionDecision(item, True, "newest")
                    if age_days >= policy.keep_all_days:
                        _reserve_periodic_bucket(
                            item,
                            policy,
                            now,
                            weekly_keep,
                            monthly_keep,
                        )
                    continue

                if age_days < policy.keep_all_days:
                    initial[item.key] = RetentionDecision(item, True, "keep_all_window")
                    continue

                if _uses_weekly(policy, item.timestamp, now):
                    bucket = item.timestamp.isocalendar()[:2]
                    current = weekly_keep.get(bucket)
                    if current is None or item.timestamp > current.timestamp:
                        weekly_keep[bucket] = item
                    initial[item.key] = RetentionDecision(item, False, "weekly_thinned")
                    continue

                if policy.monthly_forever:
                    bucket = (item.timestamp.year, item.timestamp.month)
                    current = monthly_keep.get(bucket)
                    if current is None or item.timestamp > current.timestamp:
                        monthly_keep[bucket] = item
                    initial[item.key] = RetentionDecision(item, False, "monthly_thinned")
                    continue

                initial[item.key] = RetentionDecision(item, False, "expired")

            for item in weekly_keep.values():
                initial[item.key] = RetentionDecision(item, True, "weekly")
            for item in monthly_keep.values():
                initial[item.key] = RetentionDecision(item, True, "monthly")
            decisions.extend(initial.values())

        return sorted(decisions, key=lambda d: d.item.timestamp, reverse=True)

    def deletions(
        self,
        items: Iterable[RetentionItem],
        *,
        now: datetime | None = None,
    ) -> list[RetentionItem]:
        return [
            decision.item
            for decision in self.decide(items, now=now)
            if not decision.keep
        ]


def load_retention_policy() -> RetentionPolicy:
    try:
        from kestrel_sovereign.config import load_section

        backup = load_section("backup") or {}
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to load backup retention config: %s", e)
        backup = {}
    return RetentionPolicy.from_config({"backup": backup})


def classify(item: Any) -> DataClass:
    if isinstance(item, Mapping):
        role = str(item.get("role") or item.get("data_class") or "").lower()
        if role in {DataClass.IDENTITY.value, "constitutional"}:
            return DataClass.IDENTITY
        if role in {DataClass.WORKING_MEMORY.value, "working", "snapshot", "wal"}:
            return DataClass.WORKING_MEMORY
        parts = [
            item.get("name"),
            item.get("filename"),
            item.get("fileName"),
            item.get("key"),
            item.get("blob_name"),
            item.get("tag"),
            item.get("cid"),
        ]
        text = " ".join(str(part) for part in parts if part)
    else:
        text = str(item)

    lowered = Path(text).name.lower()
    full = text.lower()
    if any(token in lowered or token in full for token in IDENTITY_TOKENS):
        return DataClass.IDENTITY
    if any(token in lowered or token in full for token in WORKING_MEMORY_TOKENS):
        return DataClass.WORKING_MEMORY
    return DataClass.WORKING_MEMORY


def parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _normalize_ts(value)
    if not value:
        return None
    text = str(value).strip()
    try:
        return _normalize_ts(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass
    match = re.search(r"(20\d{6})[_-](\d{6})", text)
    if match:
        try:
            return datetime.strptime(
                "".join(match.groups()), "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _uses_weekly(
    policy: RetentionClassPolicy,
    timestamp: datetime,
    now: datetime,
) -> bool:
    if policy.weekly_forever:
        return True
    if policy.weekly_until_months is None:
        return False
    return _months_between(_normalize_ts(timestamp), now) < policy.weekly_until_months


def _reserve_periodic_bucket(
    item: RetentionItem,
    policy: RetentionClassPolicy,
    now: datetime,
    weekly_keep: dict[tuple[int, int], RetentionItem],
    monthly_keep: dict[tuple[int, int], RetentionItem],
) -> None:
    if _uses_weekly(policy, item.timestamp, now):
        weekly_keep[item.timestamp.isocalendar()[:2]] = item
    elif policy.monthly_forever:
        monthly_keep[(item.timestamp.year, item.timestamp.month)] = item


def _months_between(older: datetime, newer: datetime) -> int:
    months = (newer.year - older.year) * 12 + (newer.month - older.month)
    if newer.day < older.day:
        months -= 1
    return max(months, 0)


def _normalize_ts(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
