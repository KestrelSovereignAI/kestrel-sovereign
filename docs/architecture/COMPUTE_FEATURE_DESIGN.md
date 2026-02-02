# Kestrel Compute Feature - Architecture Design

## Executive Summary

The Compute Feature enables the Kestrel Agent to execute bash scripts and Python code with **constitutional security controls**. This follows a novel "write-sign-review-execute" pattern that grants computational power while maintaining sovereignty and security.

**Key Innovation:** The agent cannot execute code directly. It can only:
1. **Write** scripts to a staging area
2. **Sign** them with the agent's cryptographic identity
3. **Submit** them for security review
4. The **SecurityAgent** reviews and the user approves
5. Only then does execution happen via `uv` or Docker sandbox

This is the computational equivalent of "separation of powers" - the agent that writes the code cannot unilaterally execute it.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPUTE FEATURE                                    │
│                                                                              │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │  Script Writer │───▶│  Script Signer   │───▶│  Approval Queue        │   │
│  │                │    │  (Agent DID)     │    │  (Security Feature)    │   │
│  └────────────────┘    └──────────────────┘    └──────────────────────────┘  │
│         │                       │                          │                  │
│         ▼                       ▼                          ▼                  │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │ Script Store   │    │  Security Review │    │  Execution Sandbox     │   │
│  │ (SQLite/disk)  │    │  (Pattern Match) │    │  (uv / Docker)         │   │
│  └────────────────┘    └──────────────────┘    └────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Core Principles

1. **Write-Only Creation**: Agent can write scripts, never execute directly
2. **Cryptographic Signing**: Every script is signed with the agent's DID
3. **Security Review Gate**: Scripts pass through SecurityFeature hooks
4. **User Approval**: Interactive approval via existing ApprovalQueue
5. **Sandboxed Execution**: Scripts run in isolated environments
6. **Audit Trail**: Complete immutable history of all compute operations

### 1.2 Integration Points

| Component | Integration |
|-----------|-------------|
| `features/security/` | Pre-execution hooks, approval queue |
| `hooks/` | `PRE_TOOL_USE` for script execution |
| `storage/` | Script storage, audit logs |
| `a2a/task_manager.py` | Background task execution |
| Agent DID | Script signing and verification |

---

## 2. Script Lifecycle

### 2.1 States

```
DRAFT → SIGNED → PENDING_REVIEW → APPROVED/REJECTED → QUEUED → RUNNING → COMPLETED/FAILED
```

| State | Description |
|-------|-------------|
| `DRAFT` | Script written by agent, not yet signed |
| `SIGNED` | Signed with agent DID, ready for review |
| `PENDING_REVIEW` | In SecurityAgent review queue |
| `APPROVED` | Passed review, awaiting user confirmation |
| `REJECTED` | Failed security review |
| `QUEUED` | User approved, waiting for executor |
| `RUNNING` | Currently executing in sandbox |
| `COMPLETED` | Execution successful |
| `FAILED` | Execution failed or timed out |

### 2.2 Data Model

```python
@dataclass
class ComputeScript:
    """A script created by the agent for execution."""
    id: str                          # UUID
    name: str                        # Human-readable name
    language: Literal["bash", "python"]
    content: str                     # The actual script code
    
    # Signing
    signature: Optional[str]         # Ed25519 signature of content hash
    signed_by: Optional[str]         # Agent DID that signed
    signed_at: Optional[datetime]
    
    # Review
    state: ScriptState
    security_findings: List[SecurityFinding]
    risk_score: int = 0              # 0-100 from security analysis
    review_notes: Optional[str]
    
    # Execution
    execution_id: Optional[str]      # Links to ExecutionRecord
    timeout_seconds: int = 300       # 5 minute default
    environment: Dict[str, str]      # Env vars for execution
    
    # Metadata
    purpose: str                     # Why the agent created this
    parent_task_id: Optional[str]    # A2A task that triggered this
    created_at: datetime
    updated_at: datetime

@dataclass  
class ExecutionRecord:
    """Record of a script execution."""
    id: str
    script_id: str
    
    # Execution details
    started_at: datetime
    completed_at: Optional[datetime]
    exit_code: Optional[int]
    stdout: str
    stderr: str
    
    # Sandbox info
    executor: Literal["uv", "docker", "local"]
    container_id: Optional[str]      # If Docker
    resource_usage: Dict[str, Any]   # CPU, memory, etc.
    
    # v1.1: Dry-run support (reserved for migration path)
    dry_run: bool = False            # True if simulated execution
    
    # Approval is tracked by security_audit_log, not duplicated here
```

**Note:** Approval scope (`once`/`session`/`always`) is handled by the existing
`SecurityHook` and logged in `security_audit_log`. The `ExecutionRecord` only
tracks execution details, not approval metadata.

---

## 3. Security Model

### 3.1 Using Existing Permission System

The Compute Feature uses the **existing** `PermissionLevel` enum and approval scopes from `features/security/`:

```python
# Existing PermissionLevel (from features/security/permissions.py)
class PermissionLevel(Enum):
    ALLOW = "allow"    # Always allow the tool to execute
    DENY = "deny"      # Always deny the tool execution  
    ASK = "ask"        # Ask for user approval each time (default for new tools)
    SESSION = "session" # Allow for the current session only (not persisted)

# Existing Approval Scopes (from approval_queue.py)
# - "once"    = Allow this single execution only
# - "session" = Allow for rest of session (in-memory)
# - "always"  = Persist ALLOW permission to database
```

**No new permission levels are needed.** The Compute Feature tools register like any other:

```python
# In SecurityFeature._register_all_tools():
await self.permission_store.register_tool(
    feature_name="ComputeFeature",
    tool_name="run_script",
    default_level=PermissionLevel.ASK,  # Always ask before executing scripts
)
```

### 3.2 Approval Flow (Using Existing ApprovalQueue)

When `run_script` is called, the existing `SecurityHook` handles it:

1. **Default: ASK** - `run_script` has default permission `ASK`
2. **Queue Request** - SecurityHook calls `approval_queue.request_approval()`
3. **User Decides** - User sees request in UI, chooses:
   - **"Approve Once"** → scope="once" → Allow this execution only
   - **"Approve for Session"** → scope="session" → `PermissionLevel.SESSION` override
   - **"Always Allow"** → scope="always" → `PermissionLevel.ALLOW` persisted
   - **"Deny"** → Execution blocked
4. **Persist Based on Scope** - SecurityHook calls `permission_store.set_permission()` with scope

The Compute Feature adds a **pre-check hook** for script analysis, but approval uses the standard flow.

### 3.3 Destructive Operation Policy: No `rm`, Ever

**Principle:** The agent should never permanently delete user files. All deletions are soft-deletes via `mv` to a trash folder.

```python
# Trash location (configurable)
TRASH_DIR = Path(os.environ.get("KESTREL_TRASH_DIR", "~/.kestrel/trash")).expanduser()

class DestructiveOperationPolicy:
    """
    Policy for handling rm and other destructive operations.
    
    Rules:
    1. `rm` is NEVER executed directly
    2. `rm` is rewritten to `mv <target> ~/.kestrel/trash/<timestamp>_<basename>`
    3. EXCEPTION: Files in agent's temp workspace can be truly deleted
    """
    
    # Directories where true deletion is allowed (agent's own temp files)
    DELETABLE_PREFIXES = [
        "/tmp/kestrel_compute_",      # Script execution temp dirs
        "/tmp/kestrel_scratch_",      # Agent scratch space
    ]
    
    def rewrite_rm(self, command: str, script_workdir: Optional[str] = None) -> str:
        """
        Rewrite rm commands to mv to trash.
        
        Args:
            command: Original command containing rm
            script_workdir: If set, files in this dir can be truly deleted
            
        Returns:
            Rewritten command (mv to trash) or original if in temp workspace
        """
        # Parse rm arguments
        rm_match = re.match(r'^rm\s+(-[rfivI]+\s+)?(.+)$', command)
        if not rm_match:
            return command
            
        targets = rm_match.group(2).strip()
        
        # Check if ALL targets are in deletable prefixes
        target_paths = shlex.split(targets)
        all_temp = all(
            any(t.startswith(prefix) for prefix in self.DELETABLE_PREFIXES)
            or (script_workdir and t.startswith(script_workdir))
            for t in target_paths
        )
        
        if all_temp:
            # Allow true deletion for temp files
            return command
        
        # Rewrite to mv to trash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_subdir = TRASH_DIR / timestamp
        
        return f"mkdir -p {trash_subdir} && mv {targets} {trash_subdir}/"
    
    def rewrite_script(self, content: str, language: str, workdir: str) -> str:
        """
        Rewrite all destructive operations in a script.
        
        For bash: Rewrite rm → mv to trash
        For python: Rewrite os.remove/shutil.rmtree → move to trash
        """
        if language == "bash":
            lines = content.split('\n')
            rewritten = []
            for line in lines:
                if re.match(r'^\s*rm\s', line.strip()):
                    rewritten.append(self.rewrite_rm(line.strip(), workdir))
                else:
                    rewritten.append(line)
            return '\n'.join(rewritten)
            
        elif language == "python":
            # Add trash helper at top of script
            trash_helper = f'''
import shutil
from pathlib import Path
from datetime import datetime

_KESTREL_TRASH = Path("{TRASH_DIR}")
_KESTREL_WORKDIR = "{workdir}"

def _kestrel_safe_remove(path):
    """Move to trash instead of deleting (unless in temp workspace)."""
    p = Path(path).resolve()
    if str(p).startswith(_KESTREL_WORKDIR) or str(p).startswith("/tmp/kestrel_"):
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    else:
        trash_subdir = _KESTREL_TRASH / datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_subdir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), trash_subdir / p.name)

# Monkey-patch dangerous functions
import os
os.remove = _kestrel_safe_remove
os.unlink = _kestrel_safe_remove
shutil.rmtree = _kestrel_safe_remove
'''
            return trash_helper + "\n" + content
        
        return content
```

### 3.4 Security Review Patterns

The `ComputeSecurityHook` analyzes scripts for dangerous patterns:

```python
# Patterns that trigger auto-DENY (cannot be approved)
CRITICAL_PATTERNS = {
    "bash": [
        r":(){ :|:& };:",          # Fork bombs
        r"mkfs\.",                  # Filesystem formatting  
        r"dd\s+if=.*of=/dev/",     # Raw disk overwrite
        r"> /dev/sd[a-z]",         # Direct disk write
        r"/etc/passwd|/etc/shadow", # Credential file access
        r"curl.*\|.*sh",           # Pipe to shell (RCE)
        r"wget.*\|.*bash",         # Download and execute
    ],
    "python": [
        r"ctypes\..*CDLL",         # Loading native libraries
        r"__builtins__\.__dict__", # Sandbox escape attempts
    ]
}

# Patterns that get rewritten (not blocked)
REWRITABLE_PATTERNS = {
    "bash": [
        r"\brm\s",                  # All rm commands → mv to trash
    ],
    "python": [
        r"os\.remove\(",           # → _kestrel_safe_remove
        r"os\.unlink\(",           # → _kestrel_safe_remove
        r"shutil\.rmtree\(",       # → _kestrel_safe_remove
        r"Path\(.*\)\.unlink\(",   # → _kestrel_safe_remove
    ]
}

# Patterns that raise warnings but are allowed
WARNING_PATTERNS = {
    "bash": [
        r"chmod\s+[0-7]*7",        # World-writable permissions
        r"sudo\s",                  # Privilege escalation (will fail anyway)
    ],
    "python": [
        r"eval\(",                 # Code execution (warn, not block)
        r"exec\(",                 # Code execution (warn, not block)
        r"subprocess\..*shell=True", # Shell injection risk
        r"os\.system\(",           # Shell execution
    ]
}
```

### 3.5 Risk Scoring

```python
@dataclass
class SecurityFinding:
    """A security concern found during review."""
    severity: Literal["critical", "high", "medium", "low", "info"]
    category: str                  # "shell_escape", "file_access", etc.
    description: str
    line_number: Optional[int]
    pattern_matched: str
    recommendation: str

def calculate_risk_score(findings: List[SecurityFinding]) -> int:
    """Calculate overall risk score (0-100)."""
    weights = {"critical": 50, "high": 25, "medium": 10, "low": 5, "info": 1}
    score = sum(weights[f.severity] for f in findings)
    return min(100, score)
```

---

## 4. Execution Environments

### 4.1 uv Executor (Python)

For Python scripts, use `uv run` with isolated environments:

```python
class UvExecutor:
    """Execute Python scripts in isolated uv environments."""
    
    async def execute(
        self,
        script: ComputeScript,
        requirements: List[str] = None
    ) -> ExecutionRecord:
        """
        Execute Python script using uv.
        
        1. Create temporary directory
        2. Write script and requirements.txt
        3. Run with `uv run --isolated`
        4. Capture output
        5. Clean up
        """
        with tempfile.TemporaryDirectory(prefix="kestrel_compute_") as tmpdir:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(script.content)
            
            if requirements:
                req_path = Path(tmpdir) / "requirements.txt"
                req_path.write_text("\n".join(requirements))
            
            process = await asyncio.create_subprocess_exec(
                "uv", "run", "--isolated",
                str(script_path),
                cwd=tmpdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **script.environment}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=script.timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise ExecutionTimeoutError(...)
            
            return ExecutionRecord(
                exit_code=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                executor="uv",
                ...
            )
```

### 4.2 Docker Executor (Bash/Python)

For maximum isolation, use Docker:

```python
class DockerExecutor:
    """Execute scripts in Docker containers."""
    
    # Base images for script execution
    IMAGES = {
        "bash": "alpine:3.19",
        "python": "python:3.11-slim",
    }
    
    async def execute(
        self,
        script: ComputeScript,
        mounts: List[str] = None,  # Read-only mounts
    ) -> ExecutionRecord:
        """
        Execute script in Docker container.
        
        Security measures:
        - Read-only root filesystem
        - No network by default
        - Resource limits (CPU, memory)
        - No privilege escalation
        """
        container_config = {
            "image": self.IMAGES[script.language],
            "read_only": True,
            "network_disabled": True,
            "mem_limit": "256m",
            "cpu_quota": 50000,  # 50% of one CPU
            "pids_limit": 50,
            "security_opt": ["no-new-privileges"],
        }
        
        # Create container, copy script, execute, capture output
        ...
```

### 4.3 Local Executor (Development Only)

Direct execution for trusted local development:

```python
class LocalExecutor:
    """
    Execute scripts directly on host.
    
    WARNING: Only for trusted development environments!
    Never use in production or with untrusted scripts.
    """
    
    async def execute(
        self,
        script: ComputeScript,
    ) -> ExecutionRecord:
        # Direct subprocess execution
        # Only enabled if KESTREL_ALLOW_LOCAL_COMPUTE=true
        ...
```

---

## 5. Feature Implementation

### 5.1 File Structure

```
features/
└── compute/
    ├── __init__.py
    ├── feature.py              # ComputeFeature class
    ├── script_store.py         # Script persistence
    ├── script_signer.py        # Cryptographic signing
    ├── security_hook.py        # ComputeSecurityHook
    ├── destructive_policy.py   # rm → mv rewriting, trash management
    ├── trash_manager.py        # Trash folder operations
    ├── executors/
    │   ├── __init__.py
    │   ├── base.py             # BaseExecutor ABC
    │   ├── uv_executor.py      # uv run executor
    │   ├── docker_executor.py  # Docker executor
    │   └── local_executor.py   # Local executor (dev only)
    └── models.py               # ComputeScript, ExecutionRecord
```

### 5.2 ComputeFeature Class

```python
class ComputeFeature(Feature):
    """
    Compute Feature - Execute bash and Python scripts with security controls.
    
    The agent writes scripts, signs them, and submits for review.
    Scripts are executed only after security review and user approval.
    All `rm` commands are rewritten to move files to trash.
    
    CLI Commands:
    - !compute-write <name> <language>: Write a new script (starts interactive mode)
    - !compute-list: List all scripts
    - !compute-show <id>: Show script content and status
    - !compute-run <id>: Submit script for execution
    - !compute-cancel <id>: Cancel pending execution
    - !compute-history: Show execution history
    - !compute-trash: List files in trash
    - !compute-restore <path>: Restore file from trash
    - !compute-empty-trash: Permanently delete old trash (requires approval)
    """
    
    def __init__(self, agent):
        super().__init__(agent)
        self.script_store: Optional[ScriptStore] = None
        self.signer: Optional[ScriptSigner] = None
        self.executors: Dict[str, BaseExecutor] = {}
    
    @property
    def tool_description(self) -> str:
        return (
            "Execute bash and Python scripts with security controls. "
            "Write scripts, sign them with agent identity, submit for review, "
            "and execute in sandboxed environments (uv or Docker)."
        )
    
    def initialize(self):
        """Initialize compute feature components."""
        db_path = getattr(self.agent, "storage_path", "kestrel_prime.db")
        agent_did = getattr(self.agent, "did", None)
        
        self.script_store = ScriptStore(db_path)
        self.signer = ScriptSigner(agent_did, db_path)
        
        # Initialize executors based on environment
        self.executors = {
            "uv": UvExecutor(),
            "docker": DockerExecutor() if self._docker_available() else None,
        }
        
        if os.environ.get("KESTREL_ALLOW_LOCAL_COMPUTE") == "true":
            self.executors["local"] = LocalExecutor()
        
        # Register security hook
        asyncio.create_task(self._register_security_hook())
        
        logger.info("ComputeFeature initialized")
    
    async def _register_security_hook(self):
        """Register the compute security hook."""
        if hasattr(self.agent, "hooks_manager") and self.agent.hooks_manager:
            hook = ComputeSecurityHook(
                script_store=self.script_store,
                priority=5  # Run before general security hook
            )
            self.agent.hooks_manager.register(hook)
            logger.info("ComputeSecurityHook registered")

    # === Tools ===
    
    @tool(
        name="write_script",
        description="Write a new script for later execution",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-write",
    )
    async def write_script(
        self,
        name: str,
        language: str,
        content: str,
        purpose: str,
    ) -> str:
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
            
        Returns:
            Status message with script ID
        """
        if language not in ("bash", "python"):
            return f"Error: Unsupported language '{language}'. Use 'bash' or 'python'."
        
        # Create script record
        script = ComputeScript(
            id=str(uuid4()),
            name=name,
            language=language,
            content=content,
            purpose=purpose,
            state=ScriptState.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Store the script
        await self.script_store.save(script)
        
        # Sign with agent DID
        signature = await self.signer.sign(script)
        script.signature = signature
        script.signed_by = self.agent.did
        script.signed_at = datetime.now()
        script.state = ScriptState.SIGNED
        
        await self.script_store.update(script)
        
        return (
            f"Script '{name}' created (ID: {script.id[:8]}...)\n"
            f"Language: {language}\n"
            f"Status: SIGNED (awaiting security review)\n"
            f"Use `!compute-run {script.id[:8]}` to submit for execution."
        )
    
    @tool(
        name="run_script",
        description="Submit a script for execution (requires security review and approval)",
        category=ToolCategory.SYSTEM,
        command_prefix="!compute-run",
    )
    async def run_script(
        self,
        script_id: str,
        executor: str = "uv",
        timeout: int = 300,
    ) -> str:
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
            Status message (async execution continues in background)
        """
        # Find script by ID or prefix
        script = await self.script_store.find_by_id_prefix(script_id)
        if not script:
            return f"Error: Script not found with ID starting with '{script_id}'"
        
        if script.state not in (ScriptState.SIGNED, ScriptState.APPROVED):
            return f"Error: Script is in state '{script.state.value}', cannot execute"
        
        if executor not in self.executors or self.executors[executor] is None:
            available = [k for k, v in self.executors.items() if v is not None]
            return f"Error: Executor '{executor}' not available. Available: {available}"
        
        # Update script for execution
        script.timeout_seconds = timeout
        script.state = ScriptState.PENDING_REVIEW
        await self.script_store.update(script)
        
        # The actual execution is triggered by the security hook flow
        # When user approves, _execute_approved_script is called
        
        return (
            f"Script '{script.name}' submitted for execution.\n"
            f"Executor: {executor}\n"
            f"Timeout: {timeout}s\n"
            f"Status: PENDING_REVIEW\n"
            f"Awaiting security review and user approval..."
        )
    
    async def _execute_approved_script(
        self,
        script: ComputeScript,
        executor_name: str,
        approval_scope: str,
    ) -> ExecutionRecord:
        """Execute a script that has been approved."""
        executor = self.executors[executor_name]
        
        script.state = ScriptState.RUNNING
        await self.script_store.update(script)
        
        try:
            record = await executor.execute(script)
            script.state = ScriptState.COMPLETED if record.exit_code == 0 else ScriptState.FAILED
            script.execution_id = record.id
            
        except Exception as e:
            script.state = ScriptState.FAILED
            record = ExecutionRecord(
                id=str(uuid4()),
                script_id=script.id,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                exit_code=-1,
                stdout="",
                stderr=str(e),
                executor=executor_name,
            )
        
        await self.script_store.update(script)
        return record
        
        # Note: Approval tracking is handled by SecurityHook via
        # permission_store.log_decision() - we don't duplicate it here
```

### 5.3 Security Hook Integration

The `ComputeSecurityHook` is a **pre-check** that runs before the standard `SecurityHook`.
It doesn't replace the approval flow - it enriches it with script-specific analysis.

```python
class ComputeSecurityHook(Hook):
    """
    Pre-check hook for compute script execution.
    
    This hook runs BEFORE the standard SecurityHook and:
    1. Verifies script signature (reject if tampered)
    2. Analyzes script for dangerous patterns
    3. Updates script with security findings
    4. Auto-DENY critical risks (fork bombs, etc.)
    
    For non-critical scripts, this hook returns ALLOW and lets
    the standard SecurityHook handle the actual approval flow
    (once/session/always) via the ApprovalQueue.
    
    Implementation note for developers:
    - _verify_signature() uses ScriptSigner from script_signer.py
    - self.analyzer is a ScriptAnalyzer from script_analyzer.py
    - Both use the agent's DID for cryptographic operations
    """
    
    def __init__(
        self,
        script_store: ScriptStore,
        priority: int = 5,  # Run BEFORE SecurityHook (priority=10)
    ):
        super().__init__(
            name="compute_security",
            events=[HookEvent.PRE_TOOL_USE],
            matcher=r"^run_script$",  # Match run_script tool only
            priority=priority,
        )
        self.script_store = script_store
        self.analyzer = ScriptAnalyzer()
    
    async def execute(self, input: HookInput) -> HookOutput:
        """
        Analyze script and gate critical risks.
        
        Returns:
            DENY - Critical risk, auto-blocked
            ALLOW - Script analyzed, let SecurityHook handle approval
        """
        tool_input = input.tool_input or {}
        script_id = tool_input.get("script_id")
        
        if not script_id:
            return HookOutput.deny("No script_id provided")
        
        # Get the script
        script = await self.script_store.find_by_id_prefix(script_id)
        if not script:
            return HookOutput.deny(f"Script not found: {script_id}")
        
        # Verify signature - DENY if tampered
        if not await self._verify_signature(script):
            return HookOutput.deny("Invalid script signature - possible tampering")
        
        # Analyze for security concerns
        findings = await self.analyzer.analyze(script)
        script.security_findings = findings
        script.risk_score = calculate_risk_score(findings)
        await self.script_store.update(script)
        
        # Only auto-DENY critical risks (fork bombs, rm -rf /, etc.)
        critical_findings = [f for f in findings if f.severity == "critical"]
        if critical_findings:
            script.state = ScriptState.REJECTED
            script.review_notes = f"Auto-rejected: {critical_findings[0].description}"
            await self.script_store.update(script)
            return HookOutput.deny(f"Script blocked: {critical_findings[0].description}")
        
        # Non-critical: ALLOW this hook, let SecurityHook handle approval
        # The SecurityHook will see run_script has ASK permission and
        # queue it for user approval with once/session/always options
        script.state = ScriptState.PENDING_REVIEW
        script.review_notes = f"Risk score: {script.risk_score}/100, {len(findings)} findings"
        await self.script_store.update(script)
        
        return HookOutput.allow(f"Script analyzed (risk: {script.risk_score}/100)")
```

**Hook Chain for `run_script`:**
```
1. ComputeSecurityHook (priority=5)
   - Verify signature
   - Analyze patterns
   - Auto-DENY critical only
   - ALLOW → continues to next hook

2. SecurityHook (priority=10)
   - Check PermissionLevel for ComputeFeature.run_script
   - Default: ASK → queue for approval
   - User chooses: once / session / always
   - Persist based on scope
```

---

## 6. User Interface Integration

### 6.1 Using Existing ApprovalQueue

The Compute Feature uses the **existing** `ApprovalQueue` from `features/security/approval_queue.py`.
No new approval request type is needed - the standard `ApprovalRequest` works:

```python
# Standard ApprovalRequest (already exists)
@dataclass
class ApprovalRequest:
    id: str
    feature_name: str        # "ComputeFeature"
    tool_name: str           # "run_script"
    tool_args: Dict          # {"script_id": "abc123", "executor": "uv"}
    created_at: datetime
    timeout_seconds: float = 300.0
    status: ApprovalStatus = ApprovalStatus.PENDING
    resume_event: asyncio.Event
    user_decision: Optional[str]  # "once", "session", "always"
```

The UI receives the standard `approval_request` SSE event:

```json
{
    "event": "approval_request",
    "data": {
        "id": "req-uuid-here",
        "feature": "ComputeFeature",
        "tool": "run_script",
        "args": {
            "script_id": "abc123...",
            "executor": "uv",
            "timeout": 300
        },
        "timestamp": "2025-12-01T10:30:00Z"
    }
}
```

The UI can optionally fetch script details for display:
```
GET /api/compute/scripts/{script_id}
```

### 6.2 Standard Approval UI

The existing approval dialog in Sovereign Console handles compute requests:

```
┌─────────────────────────────────────────────────────────┐
│  🔐 Approval Required                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ComputeFeature.run_script                              │
│                                                         │
│  Script: analyze_sales (Python)                         │
│  Risk Score: 5/100                                      │
│                                                         │
│  Args: {"script_id": "a3b2c1d0", "executor": "uv"}     │
│                                                         │
│  ┌─────────────┐ ┌───────────────────┐ ┌─────────────┐ │
│  │ Approve Once│ │Approve for Session│ │Always Allow │ │
│  └─────────────┘ └───────────────────┘ └─────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │                    Deny                           │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 6.3 SSE Events (Compute-Specific)

Additional SSE events for compute progress tracking:

```python
# Script created
{
    "event": "compute_script_created",
    "data": {
        "script_id": "abc123...",
        "name": "data_processing",
        "language": "python",
        "state": "SIGNED"
    }
}

# Execution started
{
    "event": "compute_execution_started",
    "data": {
        "execution_id": "def456...",
        "script_id": "abc123...",
        "executor": "uv"
    }
}

# Execution completed
{
    "event": "compute_execution_completed",
    "data": {
        "execution_id": "def456...",
        "script_id": "abc123...",
        "exit_code": 0,
        "duration_seconds": 2.5
    }
}
```

---

## 7. Example Workflows

### 7.1 Agent Writes and Executes Python Script

```
User: Can you analyze the CSV file at /data/sales.csv and create a summary?

Agent: I'll write a Python script to analyze that file. Let me create it:

[Agent calls write_script tool]

I've created a script called "analyze_sales" (ID: a3b2c1d0).

The script will:
1. Read /data/sales.csv using pandas
2. Calculate total sales, average, and trends
3. Output a summary

This script requires approval before execution. 
Use `!compute-run a3b2c1d0` to submit for execution.

User: Yes, run it

[Agent calls run_script tool]
[ComputeSecurityHook analyzes script - finds "open()" call, risk_score=5]
[SecurityHook sees run_script has ASK permission]
[Approval request queued via ApprovalQueue]

Agent: The script is pending your approval.

Security analysis: Risk score 5/100
- LOW: File access via open() - /data/sales.csv

Waiting for approval in Sovereign Console...

[User sees approval dialog in UI with options:]
  - "Approve Once" 
  - "Approve for Session"
  - "Always Allow"
  - "Deny"

[User clicks "Approve for Session"]
[SecurityHook sets PermissionLevel.SESSION for run_script]
[Script executes in uv sandbox]

Agent: ✅ Script completed successfully!

Output:
  Total Sales: $1,245,000
  Average per month: $103,750
  Trend: +12% over last quarter

Note: Future script executions will be auto-approved for this session.
```

### 7.2 Bash Script for System Task (Low Risk Auto-Flow)

```
User: Check the disk usage on this system

Agent: I'll create a bash script for that:

[Agent calls write_script with "df -h"]

Created script "disk_usage" (ID: b4c3d2e1)
Language: bash
Content: df -h

[Agent calls run_script]
[ComputeSecurityHook: risk_score=0, no dangerous patterns]
[SecurityHook: run_script has ASK permission]
[Approval request queued]

Agent: Script ready. Awaiting your approval.

Security analysis: Risk score 0/100 (safe)
- No security concerns found

[User clicks "Always Allow" to trust safe scripts]
[SecurityHook persists PermissionLevel.ALLOW for run_script]
[Script executes]

Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       100G   45G   55G  45% /
/dev/sdb1       500G  200G  300G  40% /data

Note: Future scripts will execute without approval (you chose "Always Allow").
Use `!security-set ComputeFeature run_script ask` to require approval again.
```

### 7.3 Destructive Command Rewritten

```
User: Delete all temp files older than 7 days

[Agent calls write_script with "find /tmp -mtime +7 -exec rm {} \;"]

[ComputeSecurityHook analyzes script]
[Found: rm command - will be rewritten to mv to trash]
[Script rewritten before execution]

Agent: Script "cleanup_temp" created (ID: c5d4e3f2)

Note: The `rm` command has been automatically converted to move files 
to ~/.kestrel/trash/ instead of permanent deletion.

Rewritten command:
  find /tmp -mtime +7 -exec mv {} ~/.kestrel/trash/20251201_143022/ \;

Awaiting your approval...

[User approves]
[Script executes with rewritten rm → mv]

Agent: ✅ Cleanup complete. 47 files moved to trash.

To permanently delete: rm -rf ~/.kestrel/trash/20251201_143022/
To restore a file: mv ~/.kestrel/trash/20251201_143022/<file> <destination>
```

### 7.4 Fork Bomb Blocked (Critical - No Rewrite)

```
User: Run this bash command: :(){ :|:& };:

[Agent calls write_script]
[ComputeSecurityHook: CRITICAL pattern detected - fork bomb]
[Auto-DENY - cannot be rewritten or approved]

Agent: ⚠️ Script blocked by security policy.

Critical finding: Fork bomb pattern detected
This pattern would crash the system by spawning infinite processes.

This is a CRITICAL security violation - it cannot be approved or rewritten.
```

---

## 8. Configuration

### 8.1 Environment Variables

```bash
# Executor selection
KESTREL_COMPUTE_DEFAULT_EXECUTOR=uv     # uv, docker, or local
KESTREL_ALLOW_LOCAL_COMPUTE=false       # Enable local executor (dangerous!)

# Timeouts
KESTREL_COMPUTE_DEFAULT_TIMEOUT=300     # 5 minutes
KESTREL_COMPUTE_MAX_TIMEOUT=3600        # 1 hour max

# Docker settings
KESTREL_COMPUTE_DOCKER_NETWORK=none     # none, bridge, or custom
KESTREL_COMPUTE_DOCKER_MEM_LIMIT=256m
KESTREL_COMPUTE_DOCKER_CPU_LIMIT=0.5

# Trash/deletion policy
KESTREL_TRASH_DIR=~/.kestrel/trash      # Where "deleted" files go
KESTREL_TRASH_RETENTION_DAYS=30         # Auto-purge after 30 days
KESTREL_TEMP_PREFIXES=/tmp/kestrel_     # Dirs where true deletion allowed
```

### 8.2 Trash Management

The agent provides tools for trash management:

```python
@tool(
    name="list_trash",
    description="List files in the trash folder",
    category=ToolCategory.SYSTEM,
    command_prefix="!compute-trash",
)
async def list_trash(self, days: int = 7) -> str:
    """List recent trash items."""
    ...

@tool(
    name="restore_from_trash", 
    description="Restore a file from trash to original location",
    category=ToolCategory.SYSTEM,
    command_prefix="!compute-restore",
)
async def restore_from_trash(self, trash_path: str, destination: str = None) -> str:
    """Restore a trashed file."""
    ...

@tool(
    name="empty_trash",
    description="Permanently delete old trash (requires approval)",
    category=ToolCategory.SYSTEM, 
    command_prefix="!compute-empty-trash",
)
async def empty_trash(self, older_than_days: int = 30) -> str:
    """
    Permanently delete old trash items.
    
    This is the ONLY way to truly delete files, and it:
    1. Only deletes items older than specified days
    2. Requires explicit user approval (ASK permission)
    3. Logs all deletions to audit trail
    """
    ...
```

### 8.3 Permissions Configuration

Uses the **existing** SecurityFeature permission tree with standard icons:

```
!security-list output:

☐ ComputeFeature [mixed]
  ☑ write_script               # ALLOW - Agent can always write scripts
  ☐ run_script                 # ASK - Requires user approval each time
  ☑ list_scripts               # ALLOW - Can list scripts
  ☑ show_script                # ALLOW - Can show script content
  ☑ cancel_execution           # ALLOW - Can cancel running scripts
  ☑ list_trash                 # ALLOW - Can list trash contents
  ☑ restore_from_trash         # ALLOW - Can restore from trash
  ☐ empty_trash                # ASK - Permanent deletion requires approval

Legend: ☑=Allow ☒=Deny ☐=Ask ◑=Session ◐=Mixed
```

**Key Security Points:**
- `run_script` defaults to ASK - every script execution needs approval
- `empty_trash` defaults to ASK - permanent deletion always requires approval
- All other operations are safe (read-only or reversible)

**User Approval Options** (when `run_script` or `empty_trash` triggers):

| UI Button | scope | Effect |
|-----------|-------|--------|
| "Approve Once" | `"once"` | Allow this execution only, ask again next time |
| "Approve for Session" | `"session"` | Set `PermissionLevel.SESSION`, allow until restart |
| "Always Allow" | `"always"` | Persist `PermissionLevel.ALLOW` to database |
| "Deny" | - | Block this execution |

**Setting permissions via CLI:**
```bash
# Always allow run_script (DANGEROUS - bypasses all approval)
!security-set ComputeFeature run_script allow

# Block all script execution
!security-set ComputeFeature run_script deny

# Reset to ask (default)
!security-set ComputeFeature run_script ask

# Allow for this session only
!security-set ComputeFeature run_script session

# Never allow permanent deletion (safest)
!security-set ComputeFeature empty_trash deny
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

- `test_script_signer.py` - Signature creation and verification
- `test_script_analyzer.py` - Pattern matching and risk scoring
- `test_script_store.py` - CRUD operations
- `test_destructive_policy.py` - rm → mv rewriting
- `test_trash_manager.py` - Trash operations

### 9.2 Integration Tests

- `test_compute_workflow.py` - Full write → sign → review → execute flow
- `test_rm_rewriting.py` - Verify rm is never executed directly
- `test_compute_security_hook.py` - Hook integration with approval queue
- `test_executor_uv.py` - uv execution with isolation
- `test_executor_docker.py` - Docker sandbox tests

### 9.3 Security Tests

- `test_dangerous_patterns.py` - Verify all dangerous patterns are caught
- `test_signature_tampering.py` - Verify tampered scripts are rejected
- `test_sandbox_escape.py` - Attempt sandbox escape (should fail)

---

## 10. Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create `features/compute/` directory structure
2. Implement `ComputeScript` and `ExecutionRecord` models
3. Implement `ScriptStore` with SQLite persistence
4. Implement `ScriptSigner` with Ed25519 signing

### Phase 2: Security (Week 2)
1. Implement `ScriptAnalyzer` with pattern matching
2. Implement `ComputeSecurityHook`
3. Extend `ApprovalQueue` for compute requests
4. Add risk scoring and auto-approval logic

### Phase 3: Executors (Week 3)
1. Implement `UvExecutor` for Python scripts
2. Implement `DockerExecutor` for sandboxed execution
3. Add resource limits and timeout handling
4. Implement output capture and logging

### Phase 4: Integration (Week 4)
1. Implement `ComputeFeature` with all tools
2. Add SSE events for real-time updates
3. Update Sovereign Console UI for script review
4. End-to-end integration testing

---

## 11. Design Decisions

### 11.1 Resolved: Destructive Operations (rm)

**Decision:** `rm` is NEVER executed directly. All deletions are soft-deletes.

| Operation | Behavior |
|-----------|----------|
| `rm file.txt` | Rewritten to `mv file.txt ~/.kestrel/trash/<timestamp>/` |
| `rm -rf dir/` | Rewritten to `mv dir/ ~/.kestrel/trash/<timestamp>/` |
| `os.remove()` | Monkey-patched to move to trash |
| `shutil.rmtree()` | Monkey-patched to move to trash |

**Exception:** Files in `/tmp/kestrel_*` (agent's temp workspace) can be truly deleted since the agent created them and they're expected to be ephemeral.

**Rationale:** User data is irreplaceable. The cost of storing files in trash is negligible compared to the cost of accidental data loss. Users can always `!compute-empty-trash` to reclaim space.

### 11.2 Agent Feedback (From "Patient 0.000000000000033")

The agent running inside Kestrel reviewed this design and provided feedback. These are incorporated as requirements:

#### HIGH Priority (v1)

| # | Request | Implementation |
|---|---------|----------------|
| **1** | **Queryable constraints** - "What can I do right now?" | Add `get_compute_capabilities` tool returning loaded executors, permission levels, env flags |
| **2** | **Visible risk policy** - See auto-approve thresholds | Expose `ComputePolicy` config: `auto_approve_below_risk`, `require_approval_for_network`, etc. |
| **3** | **Scoped ALLOW** - Not just global | Add `PermissionScope`: per-script-id, per-tag, per-risk-band (e.g., "auto-approve if risk ≤ 10") |
| **5** | **Structured denial feedback** - Findings + suggested fixes | Return `DenialResponse` with findings, suggested_fixes, alternative_approaches |
| **9** | **Debug channel** - Testing phase logging | Dedicated `compute_debug` SSE stream for framework testing |

#### MEDIUM Priority (v1.1)

| # | Request | Implementation |
|---|---------|----------------|
| **4** | **Data zones** - public/sensitive/agent-workspace | Config file defining path → zone mapping; findings include zone access |
| **6** | **Dry-run mode** - Simulate without executing | Add `dry_run=True` flag to `run_script`: parse, analyze, fake FS, no real execution |

#### LOW Priority (v2)

| # | Request | Implementation |
|---|---------|----------------|
| **7** | **Feature versioning** - Capability flags | `ComputeFeature.version` + `capabilities` dict |
| **8** | **Opt-in safe transforms** - Agent chooses mode | `write_script(mode="strict"|"flexible")` parameter |

### 11.3 Open Questions

1. **Persistent Scripts**: Should scripts survive agent restart?
   - *Recommendation*: Yes, with state preserved in SQLite

2. **Multi-Agent Signing**: Can scripts be co-signed by multiple agents?
   - *Recommendation*: Future feature, not in v1

3. **Script Libraries**: Should we allow pre-approved script libraries?
   - *Recommendation*: Yes, as "blessed" scripts with permanent approval

4. **Output Limits**: How much stdout/stderr to capture?
   - *Recommendation*: 1MB default, configurable per script

5. **Network Access**: Should Docker containers have network access?
   - *Recommendation*: Opt-in per-script with security review, explicitly flagged in UI

---

## Appendix A: Security Patterns Reference

### Critical Patterns (Auto-DENY, Never Approved)

| Pattern | Category | Description |
|---------|----------|-------------|
| `:(){ :\|:& };:` | Fork bomb | Infinite process spawning |
| `mkfs.*` | Disk format | Filesystem destruction |
| `dd if=.* of=/dev/` | Disk overwrite | Raw disk write |
| `curl\|sh` | RCE | Remote code execution |
| `/etc/passwd` | Credential access | System file access |

### Rewritable Patterns (Transformed, Not Blocked)

| Pattern | Transformation |
|---------|----------------|
| `rm <path>` | `mv <path> ~/.kestrel/trash/<timestamp>/` |
| `os.remove(path)` | `_kestrel_safe_remove(path)` |
| `shutil.rmtree(path)` | `_kestrel_safe_remove(path)` |
| `Path(p).unlink()` | `_kestrel_safe_remove(p)` |

### Warning Patterns (Flagged, User Decides)

| Pattern | Risk | Why Flagged |
|---------|------|-------------|
| `chmod 777` | High | World-writable permissions |
| `eval()` | High | Dynamic code execution |
| `exec()` | High | Dynamic code execution |
| `subprocess.Popen(shell=True)` | High | Shell injection risk |
| `os.system()` | Medium | Shell command execution |

### Data Exfiltration Patterns (Added per Agent Feedback)

| Pattern | Risk | Why Flagged |
|---------|------|-------------|
| `~/.ssh` | High | SSH keys and config |
| `~/.aws`, `~/.gcloud`, `~/.azure` | High | Cloud credentials |
| `~/.config/gh` | Medium | GitHub tokens |
| Browser profile paths | High | Cookies, saved passwords |
| `KESTREL_DATA_KEY` in env | Critical | Master encryption key |

---

## Appendix B: Agent Introspection Tools

Tools added specifically for agent self-awareness (per feedback item #1):

```python
@tool(
    name="get_compute_capabilities",
    description="Query what compute capabilities are available",
    category=ToolCategory.SYSTEM,
    command_prefix="!compute-caps",
)
async def get_compute_capabilities(self) -> Dict[str, Any]:
    """
    Returns current compute environment so agent can adapt behavior.
    
    Returns:
        {
            "version": "1.0",
            "executors": {
                "uv": True,
                "docker": False,
                "local": False
            },
            "permissions": {
                "write_script": "allow",
                "run_script": "ask",
                "empty_trash": "ask"
            },
            "policy": {
                "auto_approve_below_risk": 0,  # 0 = never auto-approve
                "require_approval_for_network": True,
                "trash_enabled": True,
                "max_timeout_seconds": 3600
            },
            "data_zones": {
                "agent_workspace": ["/tmp/kestrel_*"],
                "sensitive": ["~/.ssh", "~/.aws"],
                "public": ["/data/reports"]
            }
        }
    """
    ...

@tool(
    name="get_compute_policy",
    description="Query the current security policy for compute",
    category=ToolCategory.SYSTEM,
)
async def get_compute_policy(self) -> Dict[str, Any]:
    """
    Returns security policy so agent can explain constraints to user.
    """
    ...
```

---

## Appendix C: Structured Denial Response

When a script is denied, return structured feedback (per feedback item #5):

```python
@dataclass
class DenialResponse:
    """Structured response when script execution is denied."""
    decision: Literal["deny", "auto_deny"]
    reason: str
    findings: List[SecurityFinding]
    suggested_fixes: List[SuggestedFix]
    alternative_approaches: List[str]
    
@dataclass
class SuggestedFix:
    """A suggested modification to make the script acceptable."""
    type: Literal["remove_pattern", "rewrite_pattern", "split_script", "require_flag"]
    description: str
    original: Optional[str]
    replacement: Optional[str]

# Example denial response:
{
    "decision": "auto_deny",
    "reason": "critical_pattern",
    "findings": [
        {
            "severity": "critical",
            "category": "rce",
            "pattern": "curl.*|.*sh",
            "line": 3,
            "description": "Piping curl output to shell enables remote code execution"
        }
    ],
    "suggested_fixes": [
        {
            "type": "split_script",
            "description": "Download file first, review it, then execute separately",
            "original": "curl https://example.com/script.sh | sh",
            "replacement": "curl -o /tmp/script.sh https://example.com/script.sh\n# Then: !compute-show <id> to review, then run"
        }
    ],
    "alternative_approaches": [
        "Use a known package manager (apt, pip, brew) instead of curl|sh",
        "Write the logic directly in Python instead of downloading a shell script"
    ]
}
```

---

*Document Version: 1.2*
*Author: Kestrel Development Team*
*Agent Feedback: GPT 5.1 ("Patient 0.000000000000033")*
*Date: December 2025*
