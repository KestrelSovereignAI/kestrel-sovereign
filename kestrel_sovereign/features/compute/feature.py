"""
Kestrel Compute Feature - Main Feature Class.

Execute bash and Python scripts with constitutional security controls.
Implements the "write-sign-review-execute" pattern for safe code execution.
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from kestrel_sdk.hooks.base import Hook
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult
from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.kestrel_config.constants import (
    APPROVAL_TIMEOUT_DEFAULT,
    SUBPROCESS_TIMEOUT_SHORT,
)

from .models import (
    ComputePolicy,
    ComputeScript,
    ExecutionRecord,
    ScriptState,
    calculate_risk_score,
)
from .script_store import ScriptStore
from .script_signer import ScriptSigner
from .script_analyzer import ScriptAnalyzer
from .destructive_policy import DestructiveOperationPolicy
from .trash_manager import TrashManager, get_trash_manager
from .executors import BaseExecutor, UvExecutor, DockerExecutor, LocalExecutor
from .security_hook import ComputeSecurityHook, ComputeDebugHook

logger = logging.getLogger(__name__)


class ComputeFeature(Feature):
    """
    Compute Feature - Execute bash and Python scripts with security controls.
    
    The agent writes scripts, signs them, and submits for review.
    Scripts are executed only after security review and user approval.
    All `rm` commands are rewritten to move files to trash.
    
    Key Innovation: The agent cannot execute code directly. It can only:
    1. Write scripts to a staging area
    2. Sign them with the agent's cryptographic identity
    3. Submit them for security review
    4. The SecurityAgent reviews and the user approves
    5. Only then does execution happen via uv or Docker sandbox
    
    This is the computational equivalent of "separation of powers" - 
    the agent that writes the code cannot unilaterally execute it.
    
    CLI Commands:
    - !compute-write <name> <language>: Write a new script
    - !compute-list: List all scripts
    - !compute-show <id>: Show script content and status
    - !compute-run <id>: Submit script for execution
    - !compute-cancel <id>: Cancel pending execution
    - !compute-history: Show execution history
    - !compute-trash: List files in trash
    - !compute-restore <path>: Restore file from trash
    - !compute-empty-trash: Permanently delete old trash
    - !compute-caps: Query available capabilities
    """
    
    def __init__(self, agent):
        super().__init__(agent)
        self.script_store: Optional[ScriptStore] = None
        self.signer: Optional[ScriptSigner] = None
        self.analyzer: Optional[ScriptAnalyzer] = None
        self.policy: Optional[ComputePolicy] = None
        self.executors: Dict[str, Optional[BaseExecutor]] = {}
        self.trash_manager: Optional[TrashManager] = None
        self._destructive_policy: Optional[DestructiveOperationPolicy] = None
    
    @property
    def tool_description(self) -> str:
        return (
            "Execute bash and Python scripts with security controls. "
            "Write scripts, sign them with agent identity, submit for review, "
            "and execute in sandboxed environments (uv or Docker). "
            "All destructive operations (rm) are safely rewritten to use trash."
        )
    
    async def initialize(self):
        """Initialize compute feature components."""
        # Get database path from agent
        # For PostgreSQL mode, storage_path is None - features that need local SQLite
        # should use a fallback path in the agent_data directory
        db_path = getattr(self.agent, "storage_path", None)
        if not db_path:
            # Fallback to agent_data directory for compute scripts
            agent_data_dir = os.environ.get("KESTREL_DB_PATH", "./agent_data")
            os.makedirs(agent_data_dir, exist_ok=True)
            db_path = os.path.join(agent_data_dir, "compute_scripts.db")

        # Get agent DID for signing
        agent_did = getattr(self.agent, "did", None)

        # Initialize components
        self.script_store = ScriptStore(db_path)
        self.signer = ScriptSigner(agent_did, db_path)
        self.analyzer = ScriptAnalyzer()
        self.trash_manager = get_trash_manager()
        self._destructive_policy = DestructiveOperationPolicy()

        # Ensure trash directory exists
        self.trash_manager.ensure_trash_dir()

        # Load policy from environment
        self.policy = ComputePolicy(
            auto_approve_below_risk=int(os.environ.get("KESTREL_COMPUTE_AUTO_APPROVE_RISK", "0")),
            max_timeout_seconds=int(os.environ.get("KESTREL_COMPUTE_MAX_TIMEOUT", "3600")),
            default_timeout_seconds=int(os.environ.get("KESTREL_COMPUTE_DEFAULT_TIMEOUT", "300")),
            allow_docker=os.environ.get("KESTREL_COMPUTE_ALLOW_DOCKER", "true").lower() == "true",
            allow_local=os.environ.get("KESTREL_ALLOW_LOCAL_COMPUTE", "false").lower() == "true",
        )

        # Initialize executors
        self.executors = {
            "uv": UvExecutor(),
            "docker": DockerExecutor() if self._docker_available() else None,
        }

        if self.policy.allow_local:
            self.executors["local"] = LocalExecutor()

        # Track initialization state - async init will complete this
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Complete async initialization directly
        await self._async_initialize()

        logger.info("ComputeFeature initialized")
    
    def get_hooks(self) -> List[Hook]:
        """Return compute security hooks for auto-registration."""
        hooks: List[Hook] = []
        if hasattr(self, 'script_store') and hasattr(self, 'signer'):
            hooks.append(ComputeSecurityHook(
                script_store=self.script_store,
                signer=self.signer,
                priority=5,
            ))
            if os.environ.get("KESTREL_COMPUTE_DEBUG") == "true":
                hooks.append(ComputeDebugHook())
        return hooks

    async def _async_initialize(self):
        """Async initialization tasks."""
        async with self._init_lock:
            if self._initialized:
                return

            # Initialize script store
            await self.script_store.initialize()

            # NOTE: Hook registration is now handled automatically by the
            # Feature lifecycle via get_hooks(). No manual registration needed.

            self._initialized = True
            logger.info("ComputeFeature async initialization complete")
    
    async def _ensure_initialized(self):
        """Ensure async initialization is complete before operations."""
        if not self._initialized:
            await self._async_initialize()
    
    def _docker_available(self) -> bool:
        """Check if Docker is available."""
        import shutil
        if not shutil.which("docker"):
            return False
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SHORT,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False
    
    async def shutdown(self):
        """Clean up resources."""
        # Clean up executors
        for executor in self.executors.values():
            if executor:
                await executor.cleanup()
    
    # =========================================================================
    # Script Management Tools
    # =========================================================================
    
    @tool(
        name="write_script",
        description="Write a new script for later execution. The script is NOT executed immediately - it will be signed, reviewed, and requires user approval.",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-write",
    )
    async def write_script(
        self,
        name: str,
        language: str,
        content: str,
        purpose: str,
        requirements: str = "",
    ) -> ToolResult:
        """
        Write a new script to the staging area.

        The script is NOT executed immediately. It will be:
        1. Stored in the script store
        2. Signed with the agent's DID
        3. Queued for security review
        4. Presented to user for approval

        Args:
            name: Human-readable name for the script
            language: "bash" or "python"
            content: The script content
            purpose: Why this script is needed
            requirements: Python packages needed (comma-separated, python only)
        """
        await self._ensure_initialized()

        if language not in ("bash", "python"):
            return ToolResult.failed(
                f"Error: Unsupported language '{language}'. Use 'bash' or 'python'.",
                data={"name": name, "language": language},
            )

        reqs = []
        if requirements and language == "python":
            reqs = [r.strip() for r in requirements.split(",") if r.strip()]

        script = ComputeScript(
            id=str(uuid4()),
            name=name,
            language=language,
            content=content,
            purpose=purpose,
            state=ScriptState.DRAFT,
            requirements=reqs,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        await self.script_store.save(script)

        # Sign with agent DID. Sign-or-fail: if keys are unavailable the
        # script stays in DRAFT and is reported back to the caller.
        from .script_signer import ScriptSigningKeysUnavailable
        try:
            await self.signer.sign_and_update(script)
        except ScriptSigningKeysUnavailable as e:
            logger.error(f"Cannot sign script {script.id[:8]}…: {e}")
            msg = (
                f"❌ Script '{name}' saved as DRAFT but could not be signed.\n"
                f"   ID: {script.id[:8]}\n"
                f"   Reason: agent secp256k1 signing keys are not available. "
                f"Cannot produce an unforgeable signature; refusing to mark "
                f"as SIGNED."
            )
            return ToolResult.failed(
                msg,
                data={
                    "name": name,
                    "script_id": script.id,
                    "state": ScriptState.DRAFT.value,
                    "signed": False,
                },
            )
        script.state = ScriptState.SIGNED
        await self.script_store.update(script)

        analysis = self.analyzer.analyze(script)

        response = (
            f"✅ Script '{name}' created (ID: {script.id[:8]})\n"
            f"   Language: {language}\n"
            f"   Status: SIGNED (awaiting security review)\n"
        )

        if analysis.has_critical:
            response += "   ⚠️  CRITICAL issues found - will be blocked\n"
        elif analysis.has_rewritable:
            response += "   ℹ️  Destructive operations will be safely rewritten\n"

        if analysis.findings:
            response += (
                f"   Security findings: {len(analysis.findings)} "
                f"(risk: {analysis.risk_score}/100)\n"
            )

        if reqs:
            response += f"   Requirements: {', '.join(reqs)}\n"

        response += (
            f"\n   Use `!compute-run {script.id[:8]}` to submit for execution."
        )

        data = {
            "name": name,
            "script_id": script.id,
            "language": language,
            "state": ScriptState.SIGNED.value,
            "signed": True,
            "risk_score": analysis.risk_score,
            "findings_count": len(analysis.findings),
            "has_critical": analysis.has_critical,
            "has_rewritable": analysis.has_rewritable,
            "requirements": reqs,
        }

        # Honesty: a script with CRITICAL findings will be blocked at
        # run_script time. The write succeeded but the practical effect
        # is "this can never run as-written". PARTIAL forces the agent
        # to speak that the script will be rejected on execution
        # rather than just narrate "created successfully".
        if analysis.has_critical:
            return ToolResult.partial(
                confirmation=response,
                error=(
                    f"script {script.id[:8]} contains CRITICAL findings; "
                    "run_script will auto-reject it. The write/sign step "
                    "succeeded but the script is not executable as written."
                ),
                data=data,
            )

        return ToolResult.ok(confirmation=response, data=data)
    
    @tool(
        name="run_script",
        description="Submit a script for execution (requires security review and user approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-run",
    )
    async def run_script(
        self,
        script_id: str,
        executor: str = "uv",
        timeout: int = 300,
    ) -> ToolResult:
        """
        Submit a script for execution.
        
        This triggers the security review and approval flow:
        1. Security hook analyzes the script
        2. If risky patterns found, auto-reject or require approval
        3. User is notified via approval queue
        4. On approval, script executes in sandbox
        
        Args:
            script_id: ID of the script to run (full or prefix)
            executor: Execution environment ("uv", "docker", or "local")
            timeout: Maximum execution time in seconds
            
        Returns:
            Execution result or status message
        """
        # Ensure async initialization is complete
        await self._ensure_initialized()
        
        # Find script
        script = await self.script_store.find_by_id_prefix(script_id)
        if not script:
            return ToolResult.failed(
                f"Error: Script not found with ID starting with '{script_id}'",
                data={"script_id": script_id},
            )

        # Defense-in-depth: re-verify the signature here, independent of the
        # security-hook chain. Wave 0B (#914) — script.state alone is not
        # sufficient; a host that bypasses or misregisters the hook chain
        # would otherwise trust a forgeable legacy 'hmac:' tag.
        #
        # Gate on state, not on a truthy signature (#925). A manually
        # corrupted DB row with state=SIGNED and signature=None/"" would
        # otherwise skip verify entirely and fall through to the state
        # check that accepts SIGNED. ``signer.verify()`` already rejects
        # null/empty signatures internally; we just have to actually call it.
        if self.signer is not None and script.state in (
            ScriptState.SIGNED, ScriptState.APPROVED,
        ):
            is_valid = await self.signer.verify(script)
            if not is_valid:
                script.state = ScriptState.REJECTED
                script.review_notes = (
                    "Invalid signature on execution attempt — "
                    "possible tampering, legacy 'hmac:' tag, or missing signature."
                )
                await self.script_store.update(script)
                return ToolResult.failed(
                    f"Error: Script '{script.name}' has an invalid signature. "
                    f"Re-create or re-sign with current ECDSA keys before retrying.",
                    data={"script_id": script.id, "state": script.state.value},
                )

        # Check state
        if script.state == ScriptState.REJECTED:
            return ToolResult.failed(
                f"Error: Script '{script.name}' was rejected.\n"
                f"Reason: {script.review_notes}\n"
                f"Please create a new script without the blocked patterns.",
                data={"script_id": script.id, "state": script.state.value},
            )

        if script.state not in (ScriptState.SIGNED, ScriptState.APPROVED, ScriptState.PENDING_REVIEW):
            return ToolResult.failed(
                f"Error: Script is in state '{script.state.value}', cannot execute",
                data={"script_id": script.id, "state": script.state.value},
            )

        # Check executor availability
        if executor not in self.executors or self.executors[executor] is None:
            available = [k for k, v in self.executors.items() if v is not None]
            return ToolResult.failed(
                f"Error: Executor '{executor}' not available. Available: {available}",
                data={"executor": executor, "available": available},
            )

        # Check language support
        exec_obj = self.executors[executor]
        if not exec_obj.supports_language(script.language):
            return ToolResult.failed(
                f"Error: Executor '{executor}' does not support {script.language}",
                data={"executor": executor, "language": script.language},
            )

        # Validate timeout
        if timeout > self.policy.max_timeout_seconds:
            return ToolResult.failed(
                f"Error: Timeout {timeout}s exceeds maximum {self.policy.max_timeout_seconds}s",
                data={"timeout": timeout, "max": self.policy.max_timeout_seconds},
            )
        
        script.timeout_seconds = timeout
        
        # If script hasn't been analyzed yet, analyze it
        if not script.security_findings:
            result = self.analyzer.analyze(script)
            script.security_findings = result.findings
            script.risk_score = result.risk_score
            
            if result.has_critical:
                script.state = ScriptState.REJECTED
                critical = next(f for f in result.findings if f.severity == "critical")
                script.review_notes = f"Auto-rejected: {critical.description}"
                await self.script_store.update(script)
                return ToolResult.failed(
                    f"⛔ Script blocked: {critical.description}\n"
                    f"This pattern is not allowed for security reasons.",
                    data={
                        "script_id": script.id,
                        "state": ScriptState.REJECTED.value,
                        "auto_rejected": True,
                        "critical_finding": critical.description,
                    },
                )
        
        # Demo servers (KESTREL_DEMO_SERVER=1) skip the approval queue entirely:
        # Playwright can't click through it, and the demo agent runs in an
        # isolated DB so the broader-grants are scoped correctly. Without this
        # the modal still pops here even though SecurityFeature defaults to
        # ALLOW for demo servers — ComputeFeature reaches into the queue
        # directly rather than going through SecurityHook.get_permission().
        _is_demo_server = os.environ.get("KESTREL_DEMO_SERVER", "").lower() in (
            "1", "true", "yes",
        )
        # Request user approval unless auto-approved by policy
        if not _is_demo_server and script.risk_score >= self.policy.auto_approve_below_risk:
            # Need user approval - check if SecurityFeature has an approval queue
            security_feature = getattr(self.agent, 'features', {}).get('SecurityFeature')
            if security_feature and hasattr(security_feature, 'approval_queue') and security_feature.approval_queue:
                script.state = ScriptState.PENDING_REVIEW
                await self.script_store.update(script)
                
                # Request approval via the queue
                approved, scope = await security_feature.approval_queue.request_approval(
                    feature_name="ComputeFeature",
                    tool_name="run_script",
                    tool_args={
                        "script_id": script.id,
                        "script_name": script.name,
                        "language": script.language,
                        "risk_score": script.risk_score,
                        "findings_count": len(script.security_findings),
                        "executor": executor,
                        "purpose": script.purpose,
                    },
                    timeout=APPROVAL_TIMEOUT_DEFAULT,  # 5 minutes
                )
                
                if not approved:
                    script.state = ScriptState.REJECTED
                    script.review_notes = f"User denied execution (scope: {scope})"
                    await self.script_store.update(script)
                    return ToolResult.failed(
                        f"❌ Script execution denied by user ({scope})",
                        data={
                            "script_id": script.id,
                            "state": ScriptState.REJECTED.value,
                            "denied_by_user": True,
                            "scope": scope,
                        },
                    )
                
                script.state = ScriptState.APPROVED
                script.review_notes = f"User approved with scope: {scope}"
                await self.script_store.update(script)
        else:
            # Auto-approved due to low risk
            script.state = ScriptState.APPROVED
            script.review_notes = f"Auto-approved (risk {script.risk_score} < threshold {self.policy.auto_approve_below_risk})"
            await self.script_store.update(script)
        
        script.state = ScriptState.QUEUED
        await self.script_store.update(script)
        
        # Execute
        try:
            record = await self._execute_script(script, executor)
            
            # Update script state
            script.state = ScriptState.COMPLETED if record.succeeded else ScriptState.FAILED
            script.execution_id = record.id
            await self.script_store.update(script)
            
            # Save execution record
            await self.script_store.save_execution(record)
            
            # Format response
            status = "✅" if record.succeeded else "❌"
            response = (
                f"{status} Script '{script.name}' {script.state.value}\n"
                f"   Exit code: {record.exit_code}\n"
                f"   Duration: {record.duration_seconds:.2f}s\n"
                f"   Executor: {executor}\n"
            )
            
            if record.stdout:
                response += f"\n📤 Output:\n{record.stdout[:2000]}"
                if len(record.stdout) > 2000:
                    response += "\n... [truncated]"

            if record.stderr:
                response += f"\n📥 Stderr:\n{record.stderr[:1000]}"
                if len(record.stderr) > 1000:
                    response += "\n... [truncated]"

            data = {
                "script_id": script.id,
                "state": script.state.value,
                "exit_code": record.exit_code,
                "duration_seconds": record.duration_seconds,
                "executor": executor,
                "succeeded": record.succeeded,
                "stdout_truncated": (record.stdout[:2000] if record.stdout else ""),
                "stderr_truncated": (record.stderr[:1000] if record.stderr else ""),
            }

            # Honesty: a non-zero exit is a real failure of the
            # script, not the orchestration. The script ran but the
            # command failed — agent must speak that the script's
            # work didn't succeed rather than narrate "executed" off
            # an exit_code != 0.
            if not record.succeeded:
                return ToolResult.partial(
                    confirmation=response,
                    error=(
                        f"script {script.id[:8]} executed but exited with "
                        f"code {record.exit_code} (executor={executor}); "
                        "the work it was supposed to do did not succeed."
                    ),
                    data=data,
                )
            return ToolResult.ok(confirmation=response, data=data)

        except (OSError, ValueError, TypeError) as e:
            script.state = ScriptState.FAILED
            await self.script_store.update(script)
            return ToolResult.failed(
                f"❌ Execution failed: {str(e)}",
                data={"script_id": script.id, "state": ScriptState.FAILED.value},
            )
        except Exception as e:
            script.state = ScriptState.FAILED
            await self.script_store.update(script)
            logger.error(f"Unexpected execution failure: {e}", exc_info=True)
            return ToolResult.failed(
                f"❌ Execution failed: {str(e)}",
                data={"script_id": script.id, "state": ScriptState.FAILED.value},
            )
    
    async def _execute_script(
        self,
        script: ComputeScript,
        executor_name: str,
    ) -> ExecutionRecord:
        """Execute a script with the specified executor."""
        executor = self.executors[executor_name]
        
        script.state = ScriptState.RUNNING
        await self.script_store.update(script)
        
        record = await executor.execute(script)
        return record
    
    @tool(
        name="list_scripts",
        description="List all scripts or filter by state",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-list",
    )
    async def list_scripts(
        self,
        state: str = "",
        limit: int = 20,
    ) -> ToolResult:
        """
        List scripts in the store.

        Args:
            state: Filter by state (draft, signed, pending_review, approved, rejected, running, completed, failed)
            limit: Maximum number of results
        """
        if state:
            try:
                script_state = ScriptState(state)
                scripts = await self.script_store.list_by_state(script_state, limit)
            except ValueError:
                return ToolResult.failed(
                    f"Error: Invalid state '{state}'. Valid states: {[s.value for s in ScriptState]}",
                    data={"state": state},
                )
        else:
            scripts = await self.script_store.list_recent(limit)

        if not scripts:
            return ToolResult.ok(
                confirmation="No scripts found.",
                data={"scripts": [], "count": 0},
            )

        lines = ["📜 Scripts:\n"]
        for s in scripts:
            status_icon = {
                ScriptState.DRAFT: "📝",
                ScriptState.SIGNED: "✍️",
                ScriptState.PENDING_REVIEW: "⏳",
                ScriptState.APPROVED: "✅",
                ScriptState.REJECTED: "⛔",
                ScriptState.QUEUED: "📋",
                ScriptState.RUNNING: "⚡",
                ScriptState.COMPLETED: "✅",
                ScriptState.FAILED: "❌",
            }.get(s.state, "❓")

            lines.append(
                f"  {status_icon} {s.id[:8]} | {s.name[:20]:<20} | {s.language:<6} | "
                f"{s.state.value:<14} | risk:{s.risk_score:>3}"
            )

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(scripts),
                "scripts": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "language": s.language,
                        "state": s.state.value,
                        "risk_score": s.risk_score,
                    }
                    for s in scripts
                ],
            },
        )
    
    @tool(
        name="show_script",
        description="Show detailed information about a script",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-show",
    )
    async def show_script(self, script_id: str) -> ToolResult:
        """
        Show script details including content and security analysis.

        Args:
            script_id: Script ID (full or prefix)
        """
        script = await self.script_store.find_by_id_prefix(script_id)
        if not script:
            return ToolResult.failed(
                f"Error: Script not found with ID starting with '{script_id}'",
                data={"script_id": script_id},
            )
        
        lines = [
            f"📜 Script: {script.name}",
            f"   ID: {script.id}",
            f"   Language: {script.language}",
            f"   State: {script.state.value}",
            f"   Purpose: {script.purpose}",
            f"   Risk Score: {script.risk_score}/100",
            f"   Created: {script.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if script.signed_by:
            lines.append(f"   Signed by: {script.signed_by[:32]}...")
        
        if script.requirements:
            lines.append(f"   Requirements: {', '.join(script.requirements)}")
        
        if script.review_notes:
            lines.append(f"   Review Notes: {script.review_notes}")
        
        # Security findings
        if script.security_findings:
            lines.append("\n🔒 Security Findings:")
            for f in script.security_findings[:5]:  # Limit to 5
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(f.severity, "❓")
                lines.append(f"   {icon} [{f.severity.upper()}] {f.description}")
                if f.line_number:
                    lines.append(f"      Line {f.line_number}: {f.pattern_matched[:50]}")
        
        # Content preview
        lines.append("\n📝 Content:")
        content_lines = script.content.split('\n')
        for i, line in enumerate(content_lines[:20], 1):
            lines.append(f"   {i:3}| {line}")
        if len(content_lines) > 20:
            lines.append(f"   ... ({len(content_lines) - 20} more lines)")

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "script_id": script.id,
                "name": script.name,
                "language": script.language,
                "state": script.state.value,
                "risk_score": script.risk_score,
                "purpose": script.purpose,
                "requirements": script.requirements,
                "review_notes": script.review_notes,
                "findings_count": len(script.security_findings or []),
            },
        )
    
    @tool(
        name="execution_history",
        description="Show script execution history",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-history",
    )
    async def execution_history(
        self,
        script_id: str = "",
        limit: int = 10,
    ) -> ToolResult:
        """
        Show execution history.

        Args:
            script_id: Optional script ID to filter by
            limit: Maximum number of results
        """
        if script_id:
            script = await self.script_store.find_by_id_prefix(script_id)
            if not script:
                return ToolResult.failed(
                    f"Error: Script not found with ID starting with '{script_id}'",
                    data={"script_id": script_id},
                )
            executions = await self.script_store.get_executions_for_script(script.id, limit)
        else:
            executions = await self.script_store.list_recent_executions(limit)

        if not executions:
            return ToolResult.ok(
                confirmation="No executions found.",
                data={"executions": [], "count": 0},
            )

        lines = ["📊 Execution History:\n"]
        for e in executions:
            status = "✅" if e.succeeded else "❌"
            duration = f"{e.duration_seconds:.2f}s" if e.duration_seconds else "N/A"
            lines.append(
                f"  {status} {e.id[:8]} | script:{e.script_id[:8]} | "
                f"exit:{e.exit_code} | {duration} | {e.executor}"
            )

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(executions),
                "executions": [
                    {
                        "id": e.id,
                        "script_id": e.script_id,
                        "exit_code": e.exit_code,
                        "duration_seconds": e.duration_seconds,
                        "executor": e.executor,
                        "succeeded": e.succeeded,
                    }
                    for e in executions
                ],
            },
        )
    
    # =========================================================================
    # Trash Management Tools
    # =========================================================================
    
    @tool(
        name="list_trash",
        description="List files in the trash folder",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-trash",
    )
    async def list_trash(self, days: int = 7) -> ToolResult:
        """
        List recent trash items.

        Args:
            days: Show items from last N days (default: 7)
        """
        items = self.trash_manager.list_items(days=days, limit=50)

        if not items:
            return ToolResult.ok(
                confirmation=f"🗑️ Trash is empty (last {days} days)",
                data={"items": [], "count": 0, "days": days},
            )

        stats = self.trash_manager.get_stats()

        lines = [
            f"🗑️ Trash ({stats['item_count']} items, {self.trash_manager.format_size(stats['total_size_bytes'])})\n"
        ]

        for item in items[:20]:
            icon = "📁" if item.is_dir else "📄"
            age = (datetime.now() - item.deleted_at).days
            size = self.trash_manager.format_size(item.size_bytes)
            lines.append(
                f"  {icon} {item.name[:30]:<30} | {size:>10} | {age}d ago"
            )
            lines.append(f"      Path: {item.path}")

        if len(items) > 20:
            lines.append(f"\n  ... and {len(items) - 20} more items")

        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "count": len(items),
                "days": days,
                "stats": stats,
            },
        )
    
    @tool(
        name="restore_from_trash",
        description="Restore a file from trash to a destination",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-restore",
    )
    async def restore_from_trash(
        self,
        trash_path: str,
        destination: str = "",
    ) -> ToolResult:
        """
        Restore a file from trash.

        Args:
            trash_path: Path to the item in trash
            destination: Where to restore (default: current directory)
        """
        try:
            restored_path = self.trash_manager.restore(
                Path(trash_path),
                destination or None,
            )
        except FileNotFoundError:
            return ToolResult.failed(
                f"Error: Trash item not found: {trash_path}",
                data={"trash_path": trash_path},
            )
        except FileExistsError as e:
            return ToolResult.failed(
                f"Error: {e}",
                data={"trash_path": trash_path, "destination": destination},
            )
        except (PermissionError, OSError) as e:
            return ToolResult.failed(
                f"Error restoring file: {e}",
                data={"trash_path": trash_path},
            )
        except Exception as e:
            logger.error(f"Unexpected error restoring file: {e}", exc_info=True)
            return ToolResult.failed(
                f"Error restoring file: {e}",
                data={"trash_path": trash_path},
            )

        return ToolResult.ok(
            confirmation=f"✅ Restored to: {restored_path}",
            data={"trash_path": trash_path, "restored_path": str(restored_path)},
        )
    
    @tool(
        name="empty_trash",
        description="Permanently delete old trash items (requires approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-empty-trash",
    )
    async def empty_trash(
        self,
        older_than_days: int = 30,
        dry_run: bool = True,
    ) -> ToolResult:
        """
        Permanently delete old trash items.

        This is the ONLY way to truly delete files, and it:
        1. Only deletes items older than specified days
        2. Requires explicit user approval
        3. Logs all deletions to audit trail

        Args:
            older_than_days: Delete items older than this (default: 30)
            dry_run: If True, only count what would be deleted
        """
        if dry_run:
            count = self.trash_manager.empty(older_than_days=older_than_days, dry_run=True)
            body = (
                f"🗑️ Dry run: Would delete {count} items older than {older_than_days} days.\n"
                f"   Run with dry_run=false to actually delete."
            )
            # Honesty: dry-run mode (the default) didn't actually
            # delete anything. PARTIAL forces the agent to speak the
            # caveat — narrating "deleted N items" off a dry run
            # would be a lie. Same pattern as model.cleanup_models
            # (PR #1098) and strategic_memory.backlog_hygiene (#1104).
            return ToolResult.partial(
                confirmation=body,
                error=(
                    f"dry_run=True: {count} items were counted but NOT "
                    "deleted. Re-run with dry_run=False to apply."
                ),
                data={
                    "dry_run": True,
                    "count": count,
                    "older_than_days": older_than_days,
                    "applied": False,
                },
            )

        count = self.trash_manager.empty(older_than_days=older_than_days, dry_run=False)
        return ToolResult.ok(
            confirmation=f"🗑️ Permanently deleted {count} items older than {older_than_days} days.",
            data={
                "dry_run": False,
                "count": count,
                "older_than_days": older_than_days,
                "applied": True,
            },
        )
    
    # =========================================================================
    # Introspection Tools
    # =========================================================================
    
    @tool(
        name="get_compute_capabilities",
        description="Query what compute capabilities are available",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-caps",
    )
    async def get_compute_capabilities(self) -> ToolResult:
        """
        Returns current compute environment so agent can adapt behavior.
        """
        executor_status = {}
        for name, executor in self.executors.items():
            if executor is None:
                executor_status[name] = False
            else:
                executor_status[name] = executor.is_available

        data = {
            "version": "1.0",
            "executors": executor_status,
            "policy": self.policy.to_dict() if self.policy else {},
            "trash": {
                "enabled": self.policy.trash_enabled if self.policy else True,
                "location": str(self.trash_manager.trash_dir) if self.trash_manager else None,
            },
            "supported_languages": ["bash", "python"],
        }
        available = [name for name, ok in executor_status.items() if ok]
        return ToolResult.ok(
            confirmation=(
                f"compute capabilities (v{data['version']}); "
                f"available executors: {', '.join(available) if available else '(none)'}"
            ),
            data=data,
        )
    
    @tool(
        name="get_compute_policy",
        description="Query the current security policy for compute",
        category=ToolCategory.SYSTEM,
    )
    async def get_compute_policy(self) -> ToolResult:
        """
        Returns security policy so agent can explain constraints to user.
        """
        if not self.policy:
            return ToolResult.failed("Policy not initialized")

        data = {
            "policy": self.policy.to_dict(),
            "explanation": {
                "auto_approve_below_risk": (
                    "Scripts with risk score below this are auto-approved. "
                    f"Current: {self.policy.auto_approve_below_risk} (0 = always ask)"
                ),
                "max_timeout": f"Maximum script execution time: {self.policy.max_timeout_seconds}s",
                "trash": (
                    "All rm/delete operations are rewritten to move to trash. "
                    f"Retention: {self.policy.trash_retention_days} days"
                ),
                "executors": {
                    "uv": "Python scripts via isolated uv environments",
                    "docker": f"Sandboxed containers ({'enabled' if self.policy.allow_docker else 'disabled'})",
                    "local": f"Direct execution ({'ENABLED - dangerous!' if self.policy.allow_local else 'disabled'})",
                },
            },
        }
        return ToolResult.ok(
            confirmation=(
                f"compute policy: auto_approve_below_risk="
                f"{self.policy.auto_approve_below_risk}, "
                f"max_timeout={self.policy.max_timeout_seconds}s"
            ),
            data=data,
        )
