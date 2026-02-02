"""
Kestrel Hooks - Core Types (Claude Code Aligned).

This module defines the hook event types and data structures aligned with
Claude Code's hooks pattern for stdin/stdout JSON communication.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class HookEvent(Enum):
    """
    Hook events aligned with Claude Code pattern.

    These events map to specific points in the agent lifecycle:
    - SESSION_START: When agent initializes
    - PRE_TOOL_USE: Before a tool executes (can block/modify)
    - POST_TOOL_USE: After a tool completes
    - PRE_SUBAGENT_CALL: Before a feature subagent is invoked
    - POST_SUBAGENT_CALL: After subagent returns
    - USER_PROMPT_SUBMIT: When user sends a message
    - STOP: When agent finishes responding
    """
    SESSION_START = "SessionStart"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_SUBAGENT_CALL = "PreSubagentCall"
    POST_SUBAGENT_CALL = "PostSubagentCall"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"


class PermissionDecision(Enum):
    """
    Permission decisions for PreToolUse hooks.

    - ALLOW: Auto-approve the tool execution
    - DENY: Auto-reject, tool won't execute
    - ASK: Queue for user approval
    """
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class HookInput:
    """
    Input passed to hooks (aligned with Claude Code stdin JSON).

    This dataclass contains all context a hook needs to make decisions.
    Different fields are populated based on the hook event type.
    """
    session_id: str
    hook_event_name: str
    cwd: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # For PreToolUse/PostToolUse
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    feature_name: Optional[str] = None

    # For PostToolUse
    tool_response: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[int] = None

    # For UserPromptSubmit
    user_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "hook_event_name": self.hook_event_name,
            "cwd": self.cwd,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "feature_name": self.feature_name,
            "tool_response": self.tool_response,
            "execution_time_ms": self.execution_time_ms,
            "user_message": self.user_message,
        }


@dataclass
class HookOutput:
    """
    Output from hooks (aligned with Claude Code stdout JSON).

    This dataclass contains the hook's decision and any modifications
    to the execution flow.
    """
    continue_execution: bool = True
    stop_reason: Optional[str] = None
    system_message: Optional[str] = None  # Warning shown to user

    # For PreToolUse - permission decision
    permission_decision: Optional[PermissionDecision] = None
    permission_reason: Optional[str] = None
    updated_input: Optional[Dict[str, Any]] = None  # Modified tool args

    # For ASK - queue info
    approval_id: Optional[str] = None

    @classmethod
    def allow(cls, reason: str = None) -> "HookOutput":
        """Create an ALLOW output - execution continues."""
        return cls(
            continue_execution=True,
            permission_decision=PermissionDecision.ALLOW,
            permission_reason=reason
        )

    @classmethod
    def deny(cls, reason: str) -> "HookOutput":
        """Create a DENY output - execution blocked."""
        return cls(
            continue_execution=False,
            permission_decision=PermissionDecision.DENY,
            permission_reason=reason,
            stop_reason=reason
        )

    @classmethod
    def ask(cls, approval_id: str, reason: str = None) -> "HookOutput":
        """Create an ASK output - queued for user approval."""
        return cls(
            continue_execution=False,  # Pauses until approved
            permission_decision=PermissionDecision.ASK,
            permission_reason=reason,
            approval_id=approval_id
        )

    @classmethod
    def modify(cls, updated_input: Dict[str, Any], reason: str = None) -> "HookOutput":
        """Create a MODIFY output - execution continues with modified args."""
        return cls(
            continue_execution=True,
            permission_decision=PermissionDecision.ALLOW,
            permission_reason=reason,
            updated_input=updated_input
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "continue_execution": self.continue_execution,
            "stop_reason": self.stop_reason,
            "system_message": self.system_message,
            "permission_decision": self.permission_decision.value if self.permission_decision else None,
            "permission_reason": self.permission_reason,
            "updated_input": self.updated_input,
            "approval_id": self.approval_id,
        }


class Hook(ABC):
    """
    Base class for all hooks.

    Hooks are registered with a HooksManager and executed in priority order
    when their associated events fire. Hooks can:
    - Allow execution (continue)
    - Deny execution (block)
    - Ask for user approval (queue)
    - Modify tool arguments

    Example:
        class MySecurityHook(Hook):
            def __init__(self):
                super().__init__(
                    name="security_guard",
                    events=[HookEvent.PRE_TOOL_USE],
                    priority=10  # Run early
                )

            async def execute(self, input: HookInput) -> HookOutput:
                if input.tool_name in self.dangerous_tools:
                    return HookOutput.deny("Tool blocked by security policy")
                return HookOutput.allow()
    """

    def __init__(
        self,
        name: str,
        events: List[HookEvent],
        matcher: Optional[str] = None,  # Regex for tool names
        priority: int = 100,
        timeout: float = 5.0,
    ):
        """
        Initialize a hook.

        Args:
            name: Unique identifier for this hook
            events: List of HookEvent types this hook handles
            matcher: Optional regex pattern to match tool names
            priority: Execution priority (lower = earlier, default 100)
            timeout: Maximum execution time in seconds
        """
        self.name = name
        self.events = events
        self.matcher = matcher
        self.priority = priority
        self.timeout = timeout
        self.enabled = True
        self._compiled_matcher: Optional[re.Pattern] = None

        if matcher:
            try:
                self._compiled_matcher = re.compile(matcher)
            except re.error as e:
                raise ValueError(f"Invalid matcher regex '{matcher}': {e}")

    def matches(self, tool_name: str) -> bool:
        """
        Check if hook matches tool name (regex support).

        Args:
            tool_name: Name of the tool being executed

        Returns:
            True if this hook should process the tool, False otherwise
        """
        if not self.matcher:
            return True
        if not self._compiled_matcher:
            return True
        return bool(self._compiled_matcher.match(tool_name))

    @abstractmethod
    async def execute(self, input: HookInput) -> HookOutput:
        """
        Execute the hook.

        Args:
            input: HookInput containing context for the hook

        Returns:
            HookOutput with the hook's decision
        """
        pass

    def __repr__(self) -> str:
        events_str = ", ".join(e.value for e in self.events)
        return f"Hook({self.name}, events=[{events_str}], priority={self.priority})"
