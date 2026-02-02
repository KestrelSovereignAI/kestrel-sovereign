"""
Kestrel Compute Feature - Data Models.

This module defines the core data structures for script management
and execution tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


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
    timeout_seconds: int = 300  # 5 minute default
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
            timeout_seconds=data.get("timeout_seconds", 300),
            environment=data.get("environment", {}),
            requirements=data.get("requirements", []),
            parent_task_id=data.get("parent_task_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class ExecutionRecord:
    """Record of a script execution."""
    id: str
    script_id: str
    
    # Execution details
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    
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


@dataclass
class ComputePolicy:
    """Configuration for compute security policy."""
    # Risk thresholds
    auto_approve_below_risk: int = 0  # 0 = never auto-approve
    require_approval_for_network: bool = True
    
    # Execution limits
    max_timeout_seconds: int = 3600  # 1 hour max
    default_timeout_seconds: int = 300  # 5 minutes
    
    # Trash settings
    trash_enabled: bool = True
    trash_retention_days: int = 30
    
    # Allowed executors
    allow_docker: bool = True
    allow_local: bool = False  # Only for trusted dev environments
    
    # Temp directories where true deletion is allowed
    deletable_prefixes: List[str] = field(default_factory=lambda: [
        "/tmp/kestrel_compute_",
        "/tmp/kestrel_scratch_",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "auto_approve_below_risk": self.auto_approve_below_risk,
            "require_approval_for_network": self.require_approval_for_network,
            "max_timeout_seconds": self.max_timeout_seconds,
            "default_timeout_seconds": self.default_timeout_seconds,
            "trash_enabled": self.trash_enabled,
            "trash_retention_days": self.trash_retention_days,
            "allow_docker": self.allow_docker,
            "allow_local": self.allow_local,
            "deletable_prefixes": self.deletable_prefixes,
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
