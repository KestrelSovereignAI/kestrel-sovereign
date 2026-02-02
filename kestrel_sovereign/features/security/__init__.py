"""
Kestrel Security Feature - Permission management and approval queue.

This feature provides:
- Hierarchical permission management for tools and features
- Queue-based approval system for interactive policy building
- Security hooks that intercept tool execution

Usage:
    from kestrel_sovereign.features.security import SecurityFeature

    # Feature is auto-discovered and registered by KestrelAgent
    # Access via agent.features['SecurityFeature']
"""

from kestrel_sovereign.features.security.feature import SecurityFeature
from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
    ToolPermission,
    FeaturePermissions,
)
from kestrel_sovereign.features.security.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
)
from kestrel_sovereign.features.security.hooks import SecurityHook

__all__ = [
    "SecurityFeature",
    "PermissionLevel",
    "PermissionStore",
    "ToolPermission",
    "FeaturePermissions",
    "ApprovalQueue",
    "ApprovalRequest",
    "ApprovalStatus",
    "SecurityHook",
]
