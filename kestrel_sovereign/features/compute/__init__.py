"""
Kestrel Compute Feature - Execute scripts with constitutional security controls.

This feature enables the Kestrel Agent to execute bash scripts and Python code
with a "write-sign-review-execute" pattern that grants computational power
while maintaining sovereignty and security.

Key Innovation: The agent cannot execute code directly. It can only:
1. Write scripts to a staging area
2. Sign them with the agent's cryptographic identity
3. Submit them for security review
4. The SecurityAgent reviews and the user approves
5. Only then does execution happen via uv or Docker sandbox

This is the computational equivalent of "separation of powers" - the agent
that writes the code cannot unilaterally execute it.

Usage:
    from kestrel_sovereign.features.compute import ComputeFeature

    # Feature is auto-discovered and registered by KestrelAgent
    # Access via agent.features['ComputeFeature']
    
    # Or manually:
    feature = ComputeFeature(agent)
    feature.initialize()

CLI Commands:
    !compute-write <name> <language>  Write a new script
    !compute-list                      List all scripts
    !compute-show <id>                 Show script details
    !compute-run <id>                  Submit for execution
    !compute-history                   Show execution history
    !compute-trash                     List files in trash
    !compute-restore <path>            Restore from trash
    !compute-empty-trash               Empty old trash items
    !compute-caps                      Show compute capabilities
"""

from .feature import ComputeFeature
from .models import (
    ComputePolicy,
    ComputeScript,
    DenialResponse,
    ExecutionRecord,
    ScriptState,
    SecurityFinding,
    SuggestedFix,
    calculate_risk_score,
)
from .script_store import ScriptStore
from .script_signer import ScriptSigner
from .script_analyzer import ScriptAnalyzer, AnalysisResult, analyze_script
from .destructive_policy import DestructiveOperationPolicy, rewrite_script_for_safety
from .trash_manager import TrashManager, TrashItem, get_trash_manager
from .security_hook import ComputeSecurityHook, ComputeDebugHook
from .executors import (
    BaseExecutor,
    ExecutionError,
    ExecutionTimeoutError,
    UvExecutor,
    DockerExecutor,
    LocalExecutor,
)

__all__ = [
    # Main feature
    "ComputeFeature",
    
    # Models
    "ComputePolicy",
    "ComputeScript",
    "DenialResponse",
    "ExecutionRecord",
    "ScriptState",
    "SecurityFinding",
    "SuggestedFix",
    "calculate_risk_score",
    
    # Storage
    "ScriptStore",
    "ScriptSigner",
    
    # Security analysis
    "ScriptAnalyzer",
    "AnalysisResult",
    "analyze_script",
    
    # Destructive operations
    "DestructiveOperationPolicy",
    "rewrite_script_for_safety",
    
    # Trash management
    "TrashManager",
    "TrashItem",
    "get_trash_manager",
    
    # Hooks
    "ComputeSecurityHook",
    "ComputeDebugHook",
    
    # Executors
    "BaseExecutor",
    "ExecutionError",
    "ExecutionTimeoutError",
    "UvExecutor",
    "DockerExecutor",
    "LocalExecutor",
]
