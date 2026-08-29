"""Stable vocabulary for cooperative Stop requests."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class StopScope(str, Enum):
    """The addressable work boundary affected by a cooperative Stop.

    Process termination is intentionally absent. It is a runtime lifecycle
    operation, not a cooperative cancellation scope.
    """

    HOST = "host"
    AGENT = "agent"
    TURN = "turn"
    TOOL_CALL = "tool_call"


class StopDisposition(str, Enum):
    """One resolved target's truthful terminal Stop result."""

    STOPPED = "stopped"
    ALREADY_COMPLETE = "already_complete"
    REFUSED = "refused"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class StopRequest:
    """A typed request to cooperatively stop work, never a process."""

    scope: StopScope
    actor_id: str
    target: str | None = None
    target_agent_id: str | None = None
    reason: str | None = None
    cascade: bool = True
    correlation_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, StopScope):
            raise TypeError("scope must be a StopScope")
        if not isinstance(self.actor_id, str) or not self.actor_id.strip():
            raise ValueError("actor_id must be a concrete identity")
        if self.scope is StopScope.HOST:
            if self.target is not None:
                raise ValueError("host Stop cannot carry a target")
        elif not isinstance(self.target, str) or not self.target.strip():
            raise ValueError(f"{self.scope.value} Stop requires a target")
        if self.scope in {StopScope.TURN, StopScope.TOOL_CALL} and (
            not isinstance(self.target_agent_id, str)
            or not self.target_agent_id.strip()
        ):
            raise ValueError(
                f"{self.scope.value} Stop requires the owning agent identity"
            )
        if self.scope in {StopScope.HOST, StopScope.AGENT} and (
            self.target_agent_id is not None
        ):
            raise ValueError(
                f"{self.scope.value} Stop cannot carry a separate owning agent"
            )
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("reason must be a non-empty string when supplied")
        if not isinstance(self.cascade, bool):
            raise TypeError("cascade must be boolean")
        if (
            not isinstance(self.correlation_id, str)
            or not self.correlation_id.strip()
        ):
            raise ValueError("correlation_id must be a concrete string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "actor_id": self.actor_id,
            "target": self.target,
            "target_agent_id": self.target_agent_id,
            "reason": self.reason,
            "cascade": self.cascade,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StopRequest":
        return cls(
            scope=StopScope(value["scope"]),
            actor_id=value["actor_id"],
            target=value.get("target"),
            target_agent_id=value.get("target_agent_id"),
            reason=value.get("reason"),
            cascade=value.get("cascade", True),
            correlation_id=value["correlation_id"],
        )


@dataclass(frozen=True, slots=True)
class StopOutcome:
    """One per-target Stop result; fleet fan-out is a tuple of these."""

    scope: StopScope
    requested_target: str | None
    resolved_target: str
    agent_id: str
    disposition: StopDisposition
    correlation_id: str
    detail: str | None = None
    receipt_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, StopScope):
            raise TypeError("scope must be a StopScope")
        if self.scope is StopScope.HOST:
            if self.requested_target is not None:
                raise ValueError("host Stop outcome cannot carry a requested target")
        elif self.requested_target is None:
            raise ValueError(
                f"{self.scope.value} Stop outcome requires a requested target"
            )
        if self.requested_target is not None and (
            not isinstance(self.requested_target, str)
            or not self.requested_target.strip()
        ):
            raise ValueError(
                "requested_target must be a non-empty string when supplied"
            )
        for field_name, value in (
            ("resolved_target", self.resolved_target),
            ("agent_id", self.agent_id),
            ("correlation_id", self.correlation_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a concrete string")
        if not isinstance(self.disposition, StopDisposition):
            raise TypeError("disposition must be a StopDisposition")
        for field_name, value in (
            ("detail", self.detail),
            ("receipt_id", self.receipt_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} must be a non-empty string when supplied"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "requested_target": self.requested_target,
            "resolved_target": self.resolved_target,
            "agent_id": self.agent_id,
            "disposition": self.disposition.value,
            "correlation_id": self.correlation_id,
            "detail": self.detail,
            "receipt_id": self.receipt_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StopOutcome":
        return cls(
            scope=StopScope(value["scope"]),
            requested_target=value.get("requested_target"),
            resolved_target=value["resolved_target"],
            agent_id=value["agent_id"],
            disposition=StopDisposition(value["disposition"]),
            correlation_id=value["correlation_id"],
            detail=value.get("detail"),
            receipt_id=value.get("receipt_id"),
        )
