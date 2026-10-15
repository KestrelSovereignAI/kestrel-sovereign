"""
Kestrel Compute Feature - Data Models.

This module defines the core data structures for script management
and execution tracking.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

#: How long a script or command may run when its caller names no timeout.
#: ``run_script``'s ``timeout`` parameter defaults to it, and that default is
#: part of the tool's published schema, so no policy setting moves it.
DEFAULT_TIMEOUT_SECONDS = 300

#: The longest timeout ``run_script`` accepts unless the policy says otherwise.
DEFAULT_MAX_TIMEOUT_SECONDS = 3600

#: How many days trash is kept before ``empty_trash`` may delete it.
DEFAULT_TRASH_RETENTION_DAYS = 30

#: Where deletion may be real rather than a move to trash. Each value is a
#: parent directory plus a generated-directory name prefix, not an arbitrary
#: string prefix: ``DestructiveOperationPolicy`` resolves both sides and
#: compares path components. A policy may narrow this set, never widen it.
DEFAULT_DELETABLE_PREFIXES: Tuple[str, ...] = (
    "/tmp/kestrel_compute_",
    "/tmp/kestrel_scratch_",
    "/private/tmp/kestrel_compute_",
    "/private/tmp/kestrel_scratch_",
)

#: ``ScriptAnalyzer`` finding categories that mean the script reaches the
#: network: sockets, HTTP clients and netcat (``network``), and fetches with
#: curl or wget (``download``).
NETWORK_FINDING_CATEGORIES = frozenset({"network", "download"})

AUTO_APPROVE_RISK_ENV = "KESTREL_COMPUTE_AUTO_APPROVE_RISK"
MAX_TIMEOUT_ENV = "KESTREL_COMPUTE_MAX_TIMEOUT"
ALLOW_DOCKER_ENV = "KESTREL_COMPUTE_ALLOW_DOCKER"
ALLOW_LOCAL_COMPUTE_ENV = "KESTREL_ALLOW_LOCAL_COMPUTE"
#: Deprecated: read only to warn that it does nothing.
DEFAULT_TIMEOUT_ENV = "KESTREL_COMPUTE_DEFAULT_TIMEOUT"


def local_compute_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the host opted in to the unisolated local executor.

    Only the exact value ``true`` opts in. The policy and ``LocalExecutor``
    both ask here, so they cannot disagree about a spelling such as ``TRUE``.
    """
    env = os.environ if environ is None else environ
    return env.get(ALLOW_LOCAL_COMPUTE_ENV) == "true"


class ScriptState(Enum):
    """
    States in the script lifecycle.
    
    DRAFT → SIGNED → PENDING_REVIEW → APPROVED/REJECTED → QUEUED → RUNNING → COMPLETED/FAILED
    """
    DRAFT = "draft"              # Script written by agent, not yet signed
    SIGNED = "signed"            # Signed with agent DID, ready for review
    PENDING_REVIEW = "pending_review"  # In SecurityAgent review queue
    APPROVED = "approved"        # Passed review, awaiting user confirmation
    REJECTED = "rejected"        # Failed security review
    QUEUED = "queued"            # User approved, waiting for executor
    RUNNING = "running"          # Currently executing in sandbox
    COMPLETED = "completed"      # Execution successful
    FAILED = "failed"            # Execution failed or timed out


@dataclass
class SecurityFinding:
    """A security concern found during script review."""
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str  # "shell_escape", "file_access", "rce", "data_exfil", etc.
    description: str
    pattern_matched: str
    recommendation: str
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "pattern_matched": self.pattern_matched,
            "recommendation": self.recommendation,
            "line_number": self.line_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityFinding":
        """Create from dictionary."""
        return cls(
            severity=data["severity"],
            category=data["category"],
            description=data["description"],
            pattern_matched=data["pattern_matched"],
            recommendation=data["recommendation"],
            line_number=data.get("line_number"),
        )


@dataclass
class SuggestedFix:
    """A suggested modification to make a script acceptable."""
    type: Literal["remove_pattern", "rewrite_pattern", "split_script", "require_flag"]
    description: str
    original: Optional[str] = None
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "description": self.description,
            "original": self.original,
            "replacement": self.replacement,
        }


@dataclass
class DenialResponse:
    """Structured response when script execution is denied."""
    decision: Literal["deny", "auto_deny"]
    reason: str
    findings: List[SecurityFinding]
    suggested_fixes: List[SuggestedFix]
    alternative_approaches: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision": self.decision,
            "reason": self.reason,
            "findings": [f.to_dict() for f in self.findings],
            "suggested_fixes": [f.to_dict() for f in self.suggested_fixes],
            "alternative_approaches": self.alternative_approaches,
        }


@dataclass
class ComputeScript:
    """A script created by the agent for execution."""
    id: str  # UUID
    name: str  # Human-readable name
    language: Literal["bash", "python"]
    content: str  # The actual script code
    purpose: str  # Why the agent created this
    
    # State management
    state: ScriptState = ScriptState.DRAFT
    
    # Signing
    signature: Optional[str] = None  # secp256k1 ECDSA signature of content hash
    signed_by: Optional[str] = None  # Agent DID that signed
    signed_at: Optional[datetime] = None
    
    # Security review
    security_findings: List[SecurityFinding] = field(default_factory=list)
    risk_score: int = 0  # 0-100 from security analysis
    review_notes: Optional[str] = None
    
    # Execution configuration
    execution_id: Optional[str] = None  # Links to ExecutionRecord
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    environment: Dict[str, str] = field(default_factory=dict)  # Env vars for execution
    requirements: List[str] = field(default_factory=list)  # Python packages needed
    
    # Metadata
    parent_task_id: Optional[str] = None  # A2A task that triggered this
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "content": self.content,
            "purpose": self.purpose,
            "state": self.state.value,
            "signature": self.signature,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at.isoformat() if self.signed_at else None,
            "security_findings": [f.to_dict() for f in self.security_findings],
            "risk_score": self.risk_score,
            "review_notes": self.review_notes,
            "execution_id": self.execution_id,
            "timeout_seconds": self.timeout_seconds,
            "environment": self.environment,
            "requirements": self.requirements,
            "parent_task_id": self.parent_task_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputeScript":
        """Create from dictionary."""
        findings = [SecurityFinding.from_dict(f) for f in data.get("security_findings", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            language=data["language"],
            content=data["content"],
            purpose=data["purpose"],
            state=ScriptState(data.get("state", "draft")),
            signature=data.get("signature"),
            signed_by=data.get("signed_by"),
            signed_at=datetime.fromisoformat(data["signed_at"]) if data.get("signed_at") else None,
            security_findings=findings,
            risk_score=data.get("risk_score", 0),
            review_notes=data.get("review_notes"),
            execution_id=data.get("execution_id"),
            timeout_seconds=data.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
            environment=data.get("environment", {}),
            requirements=data.get("requirements", []),
            parent_task_id=data.get("parent_task_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass(frozen=True)
class ComputeCommand:
    """An argv vector to execute directly — not a script.

    A script is *text handed to an interpreter*; a command is an
    *argument vector handed to exec*. The difference is load-bearing,
    not stylistic. In a script the first word is read by the shell's
    grammar, so ``eval``, ``FOO=x`` and ``trap`` each dispatch a program
    of their own choosing; in an argv vector, element zero is the
    program and every other element is an argument to it. #3187 is what
    happens when a caller is promised the second and given the first.

    Deliberately a separate type rather than a mode flag on
    :class:`ComputeScript`: a command has no ``language`` because
    nothing interprets it, and no ``content`` because there is no text,
    while a script has no argv. One type carrying both would let a
    caller set the fields of one execution primitive and get the other.
    """

    id: str  # UUID
    name: str  # Human-readable name
    argv: Tuple[str, ...]  # argv[0] is the program; the rest its arguments
    purpose: str  # Why the agent created this

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    environment: Dict[str, str] = field(default_factory=dict)  # Env vars

    parent_task_id: Optional[str] = None  # A2A task that triggered this
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        # An empty vector is not "run nothing": ``docker run IMAGE`` with
        # no command runs the image's own default CMD, which for the
        # execution images is a shell. Refuse it at construction, where
        # the caller is, rather than discovering it as a container that
        # started something nobody asked for.
        #
        # Validating once is only worth anything if the value cannot
        # change afterwards, which is why this type is frozen and argv
        # is copied into a tuple. A list would stay the caller's: the
        # caller clears it, or assigns a new one, and the check that
        # already passed is describing a vector that no longer exists
        # (codex round 1 P2). The environment is left a plain dict
        # because it has no equivalent cliff — a changed value is a
        # different value, while an emptied argv is a different program.
        if not self.argv:
            raise ValueError("ComputeCommand requires a non-empty argv")
        for index, element in enumerate(self.argv):
            if not isinstance(element, str):
                raise TypeError(
                    f"ComputeCommand argv[{index}] must be str, "
                    f"got {type(element).__name__}"
                )
        if not self.argv[0]:
            raise ValueError("ComputeCommand argv[0] must name a program")
        object.__setattr__(self, "argv", tuple(self.argv))


@dataclass
class ExecutionRecord:
    """Record of one execution — of a :class:`ComputeScript` or a
    :class:`ComputeCommand`."""
    id: str
    script_id: str  # id of the executed script OR command
    
    # Execution details
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    # Whether the executor's output cap discarded anything, per stream.
    # Previously the only signal was a marker appended to ``stdout``/
    # ``stderr`` themselves, which a consumer had to parse back out of
    # caller-controlled text — so a command whose own output happened to end
    # that way was read as truncated. A boolean cannot be spoofed by what the
    # command prints. Two of them, because the streams are capped
    # independently and reporting a clipped stdout as a clipped stderr is a
    # different claim from the true one.
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def output_truncated(self) -> bool:
        """Whether either stream was clipped."""
        return self.stdout_truncated or self.stderr_truncated
    
    # Sandbox info
    executor: Literal["uv", "docker", "local"] = "uv"
    container_id: Optional[str] = None  # If Docker
    resource_usage: Dict[str, Any] = field(default_factory=dict)  # CPU, memory, etc.
    
    # v1.1: Dry-run support
    dry_run: bool = False  # True if simulated execution
    
    # Working directory
    workdir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "script_id": self.script_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            # Round-tripped: a clipped record reconstructed from storage that
            # reported itself whole would defeat the point of moving the fact
            # off the text in the first place.
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "executor": self.executor,
            "container_id": self.container_id,
            "resource_usage": self.resource_usage,
            "dry_run": self.dry_run,
            "workdir": self.workdir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            script_id=data["script_id"],
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            exit_code=data.get("exit_code"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            stdout_truncated=bool(data.get("stdout_truncated", False)),
            stderr_truncated=bool(data.get("stderr_truncated", False)),
            executor=data.get("executor", "uv"),
            container_id=data.get("container_id"),
            resource_usage=data.get("resource_usage", {}),
            dry_run=data.get("dry_run", False),
            workdir=data.get("workdir"),
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def succeeded(self) -> bool:
        """Check if execution succeeded (exit code 0)."""
        return self.exit_code == 0


# A deprecated ``ComputePolicy`` argument nobody passed. Distinct from every
# value, so passing even the value the runtime uses is still reported.
_UNSET: Any = object()


@dataclass(frozen=True, init=False)
class ComputePolicy:
    """The compute security policy, and the only place its settings live.

    ``ComputeFeature`` builds one instance (:meth:`from_env`) and hands that
    same instance to the trash manager and to every executor; the status
    tools report it. It is frozen, so what status reports is what the
    consumers were given. Each setting has one enforcing consumer:

    * ``auto_approve_below_risk`` and ``require_approval_for_network`` --
      :meth:`requires_approval`, which ``run_script`` asks before it runs a
      script. Whether to ask is decided here; asking, and recording the
      answer, belong to the security feature's approval queue.
    * ``max_timeout_seconds`` -- ``run_script`` refuses a longer timeout, and
      the Docker executor leaves an unowned staging directory alone for at
      least this long.
    * ``trash_retention_days`` -- ``TrashManager.empty`` refuses to delete
      trash younger than this.
    * ``allow_docker`` and ``allow_local`` -- whether the feature builds that
      executor at all.
    * ``deletable_prefixes`` -- the executor-owned temp roots where deletion
      is real rather than a move to trash. It may only narrow
      :data:`DEFAULT_DELETABLE_PREFIXES`; anything else is refused here.

    ``default_timeout_seconds`` and ``trash_enabled`` are deprecated and do
    nothing. ``run_script``'s timeout default is fixed by its published tool
    schema, and trash cannot be turned off: without it every ``rm`` would
    delete for real. Passing either logs a warning; reading either returns
    the value the runtime actually uses.
    """

    auto_approve_below_risk: int
    require_approval_for_network: bool
    max_timeout_seconds: int
    trash_retention_days: int
    allow_docker: bool
    allow_local: bool
    deletable_prefixes: Tuple[str, ...]

    def __init__(
        self,
        auto_approve_below_risk: int = 0,  # 0 = never auto-approve
        require_approval_for_network: bool = True,
        max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS,
        default_timeout_seconds: Any = _UNSET,
        trash_enabled: Any = _UNSET,
        trash_retention_days: int = DEFAULT_TRASH_RETENTION_DAYS,
        allow_docker: bool = True,
        allow_local: bool = False,  # Only for trusted dev environments
        deletable_prefixes: Optional[Iterable[str]] = None,
    ) -> None:
        if default_timeout_seconds is not _UNSET:
            logger.warning(
                "ComputePolicy.default_timeout_seconds is deprecated and has no "
                "effect (got %r): run_script's timeout defaults to %ss, fixed by "
                "its tool schema.",
                default_timeout_seconds,
                DEFAULT_TIMEOUT_SECONDS,
            )
        if trash_enabled is not _UNSET:
            logger.warning(
                "ComputePolicy.trash_enabled is deprecated and has no effect "
                "(got %r): compute deletion always moves files to trash.",
                trash_enabled,
            )
        if max_timeout_seconds < 1:
            raise ValueError(
                f"max_timeout_seconds must be at least 1, got {max_timeout_seconds}"
            )
        if trash_retention_days < 0:
            raise ValueError(
                f"trash_retention_days must be at least 0, got {trash_retention_days}"
            )
        if deletable_prefixes is None:
            prefixes = DEFAULT_DELETABLE_PREFIXES
        elif isinstance(deletable_prefixes, str):
            raise TypeError(
                "deletable_prefixes must be a collection of prefixes, not a str"
            )
        else:
            prefixes = tuple(deletable_prefixes)
        widened = [
            prefix for prefix in prefixes if prefix not in DEFAULT_DELETABLE_PREFIXES
        ]
        if widened:
            raise ValueError(
                "deletable_prefixes may only narrow the built-in set "
                f"{list(DEFAULT_DELETABLE_PREFIXES)}; refusing {widened}"
            )

        object.__setattr__(self, "auto_approve_below_risk", auto_approve_below_risk)
        object.__setattr__(
            self, "require_approval_for_network", require_approval_for_network
        )
        object.__setattr__(self, "max_timeout_seconds", max_timeout_seconds)
        object.__setattr__(self, "trash_retention_days", trash_retention_days)
        object.__setattr__(self, "allow_docker", allow_docker)
        object.__setattr__(self, "allow_local", allow_local)
        object.__setattr__(self, "deletable_prefixes", prefixes)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ComputePolicy":
        """Build the policy from the ``KESTREL_COMPUTE_*`` environment."""
        env = os.environ if environ is None else environ
        if DEFAULT_TIMEOUT_ENV in env:
            logger.warning(
                "%s is deprecated and has no effect (set to %r): run_script's "
                "timeout defaults to %ss, fixed by its tool schema. Remove it.",
                DEFAULT_TIMEOUT_ENV,
                env[DEFAULT_TIMEOUT_ENV],
                DEFAULT_TIMEOUT_SECONDS,
            )
        return cls(
            auto_approve_below_risk=int(env.get(AUTO_APPROVE_RISK_ENV, "0")),
            max_timeout_seconds=int(
                env.get(MAX_TIMEOUT_ENV, str(DEFAULT_MAX_TIMEOUT_SECONDS))
            ),
            allow_docker=env.get(ALLOW_DOCKER_ENV, "true").lower() == "true",
            allow_local=local_compute_enabled(env),
        )

    @property
    def default_timeout_seconds(self) -> int:
        """Deprecated: the default ``run_script`` timeout, which is fixed."""
        return DEFAULT_TIMEOUT_SECONDS

    @property
    def trash_enabled(self) -> bool:
        """Deprecated: compute deletion always goes to trash."""
        return True

    def requires_approval(
        self,
        risk_score: int,
        findings: Iterable[SecurityFinding],
    ) -> bool:
        """Whether ``run_script`` must ask before running this script.

        A risk score at or above ``auto_approve_below_risk`` always asks. So
        does a script whose analysis found it reaching the network, whatever
        its score, while ``require_approval_for_network`` is set.
        """
        if risk_score >= self.auto_approve_below_risk:
            return True
        return self.require_approval_for_network and any(
            finding.category in NETWORK_FINDING_CATEGORIES for finding in findings
        )

    def to_dict(self) -> Dict[str, Any]:
        """The values the runtime uses, for JSON serialization."""
        return {
            "auto_approve_below_risk": self.auto_approve_below_risk,
            "require_approval_for_network": self.require_approval_for_network,
            "max_timeout_seconds": self.max_timeout_seconds,
            "default_timeout_seconds": self.default_timeout_seconds,
            "trash_enabled": self.trash_enabled,
            "trash_retention_days": self.trash_retention_days,
            "allow_docker": self.allow_docker,
            "allow_local": self.allow_local,
            "deletable_prefixes": list(self.deletable_prefixes),
        }


def calculate_risk_score(findings: List[SecurityFinding]) -> int:
    """
    Calculate overall risk score (0-100) from security findings.
    
    Args:
        findings: List of security findings from analysis
        
    Returns:
        Risk score from 0 (safe) to 100 (dangerous)
    """
    weights = {
        "critical": 50,
        "high": 25,
        "medium": 10,
        "low": 5,
        "info": 1,
    }
    score = sum(weights.get(f.severity, 0) for f in findings)
    return min(100, score)
