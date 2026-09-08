"""Execute scripts with constitutional security controls.

This feature uses a ``write-sign-review-execute`` pattern that grants the
agent computational power while preserving separation of powers.  The agent
cannot execute code directly.  It can only:

1. Write scripts to a staging area.
2. Sign them with the agent's cryptographic identity.
3. Submit them for security review.
4. Wait for the security agent and user approval.
5. Execute the approved script through uv or a Docker sandbox.

The package preserves its historical re-export API, but resolves each export
on first access.  Python executes this initializer before importing *any*
compute submodule, so keeping it dependency-neutral lets pure models and
presenters load without stores, security hooks, or execution backends.

``__all__`` and :func:`dir` always enumerate the complete public API.  Before
first access, unresolved lazy values are intentionally absent from
``vars(module)`` and ``inspect.getmembers_static(module)``; ordinary imports,
attribute access, and dynamic ``inspect.getmembers`` materialize them.

Typical usage::

    from kestrel_sovereign.features.compute import ComputeFeature

    # The feature is normally auto-discovered and registered by KestrelAgent.
    feature = ComputeFeature(agent)
    await feature.initialize()

CLI commands::

    !compute-write <name> <language>  Write a new script
    !compute-list                     List all scripts
    !compute-show <id>                Show script details
    !compute-run <id>                 Submit for execution
    !compute-history                  Show execution history
    !compute-trash                    List files in trash
    !compute-restore <path>           Restore from trash
    !compute-empty-trash              Empty old trash items
    !compute-caps                     Show compute capabilities
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Redundant aliases tell type checkers (and Ruff) that these are deliberate
    # public re-exports while keeping every import off the runtime path.
    from .destructive_policy import (
        DestructiveOperationPolicy as DestructiveOperationPolicy,
        rewrite_script_for_safety as rewrite_script_for_safety,
    )
    from .executors import (
        BaseExecutor as BaseExecutor,
        CommandExecutionUnsupported as CommandExecutionUnsupported,
        DockerExecutor as DockerExecutor,
        ExecutionError as ExecutionError,
        ExecutionTimeoutError as ExecutionTimeoutError,
        LocalExecutor as LocalExecutor,
        UvExecutor as UvExecutor,
    )
    from .feature import ComputeFeature as ComputeFeature
    from .models import (
        ComputeCommand as ComputeCommand,
        ComputePolicy as ComputePolicy,
        ComputeScript as ComputeScript,
        DenialResponse as DenialResponse,
        ExecutionRecord as ExecutionRecord,
        ScriptState as ScriptState,
        SecurityFinding as SecurityFinding,
        SuggestedFix as SuggestedFix,
        calculate_risk_score as calculate_risk_score,
    )
    from .script_analyzer import (
        AnalysisResult as AnalysisResult,
        ScriptAnalyzer as ScriptAnalyzer,
        analyze_script as analyze_script,
    )
    from .script_signer import ScriptSigner as ScriptSigner
    from .script_store import ScriptStore as ScriptStore
    from .security_hook import (
        ComputeDebugHook as ComputeDebugHook,
        ComputeSecurityHook as ComputeSecurityHook,
    )
    from .trash_manager import (
        TrashItem as TrashItem,
        TrashManager as TrashManager,
        get_trash_manager as get_trash_manager,
    )


_EXPORTS_BY_MODULE = {
    ".feature": ("ComputeFeature",),
    ".models": (
        "ComputeCommand",
        "ComputePolicy",
        "ComputeScript",
        "DenialResponse",
        "ExecutionRecord",
        "ScriptState",
        "SecurityFinding",
        "SuggestedFix",
        "calculate_risk_score",
    ),
    ".script_store": ("ScriptStore",),
    ".script_signer": ("ScriptSigner",),
    ".script_analyzer": ("ScriptAnalyzer", "AnalysisResult", "analyze_script"),
    ".destructive_policy": (
        "DestructiveOperationPolicy",
        "rewrite_script_for_safety",
    ),
    ".trash_manager": ("TrashManager", "TrashItem", "get_trash_manager"),
    ".security_hook": ("ComputeSecurityHook", "ComputeDebugHook"),
    ".executors": (
        "BaseExecutor",
        "ExecutionError",
        "ExecutionTimeoutError",
        "CommandExecutionUnsupported",
        "UvExecutor",
        "DockerExecutor",
        "LocalExecutor",
    ),
}
_EXPORT_MODULES = {
    name: module_name
    for module_name, names in _EXPORTS_BY_MODULE.items()
    for name in names
}

__all__ = [
    "ComputeFeature",
    "ComputeCommand",
    "ComputePolicy",
    "ComputeScript",
    "DenialResponse",
    "ExecutionRecord",
    "ScriptState",
    "SecurityFinding",
    "SuggestedFix",
    "calculate_risk_score",
    "ScriptStore",
    "ScriptSigner",
    "ScriptAnalyzer",
    "AnalysisResult",
    "analyze_script",
    "DestructiveOperationPolicy",
    "rewrite_script_for_safety",
    "TrashManager",
    "TrashItem",
    "get_trash_manager",
    "ComputeSecurityHook",
    "ComputeDebugHook",
    "BaseExecutor",
    "ExecutionError",
    "ExecutionTimeoutError",
    "CommandExecutionUnsupported",
    "UvExecutor",
    "DockerExecutor",
    "LocalExecutor",
]

# ``importlib.reload`` re-executes this initializer in the existing module
# namespace.  Drop exports cached by a preceding execution so reload observes
# the current defining submodules, matching the historical eager-import API.
for _cached_export in __all__:
    globals().pop(_cached_export, None)
del _cached_export


def __getattr__(name: str) -> Any:
    """Resolve and cache a public compatibility export on first access."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include unresolved compatibility exports in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
