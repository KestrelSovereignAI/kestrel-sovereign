"""
Code Edit Feature - Self-modification capabilities for Kestrel agents.

This feature enables agents to modify their own source code with proper
constitutional controls and approval workflows.
"""
import asyncio
import functools
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from kestrel_sovereign.features.base import Feature, tool
from kestrel_sovereign.tools.base import ToolCategory

logger = logging.getLogger(__name__)

# Find binaries
GIT_PATH = shutil.which("git") or "/usr/bin/git"
PYTHON_PATH = shutil.which("python3") or shutil.which("python") or "/usr/bin/python3"

# Default to the kestrel-sovereign project root
DEFAULT_CODE_ROOT = os.environ.get(
    "KESTREL_CODE_ROOT",
    str(Path(__file__).parent.parent)  # Up to repo root when co-located
)

# Timeout constants (in seconds)
CODE_REVIEW_TIMEOUT = 300  # 5 minutes for user approval of code changes
TEST_SUITE_TIMEOUT = 300   # 5 minutes for running test suite
LINT_TIMEOUT = 60          # 1 minute for linting operations
GIT_OPERATION_TIMEOUT = 30  # 30 seconds for git commands (diff, commit, rollback)
GIT_QUICK_TIMEOUT = 10     # 10 seconds for quick git commands (rev-parse)


async def _run_subprocess(*args, **kwargs) -> subprocess.CompletedProcess:
    """Run subprocess.run off the event loop via asyncio.to_thread."""
    return await asyncio.to_thread(functools.partial(subprocess.run, *args, **kwargs))


class CodeEditFeature(Feature):
    """Feature for self-modification of source code.

    Enables the agent to:
    - Read its own source files
    - Edit source files (with approval)
    - Commit changes to git
    - Signal for server restart

    All destructive operations require user approval via SecurityFeature.
    """

    tool_name = "code_edit"
    tool_description = "Read and modify the agent's own source code with approval"

    def __init__(self, agent=None, code_root: str = None):
        """Initialize the code edit feature.

        Args:
            agent: The parent agent instance
            code_root: Root directory of the codebase (default: auto-detect)
        """
        super().__init__(agent)
        self.code_root = Path(code_root or DEFAULT_CODE_ROOT).resolve()
        self._pending_restart = False

    async def initialize(self):
        """Initialize the feature."""
        logger.info(f"CodeEditFeature initialized with root: {self.code_root}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to code root, with security checks."""
        # Handle absolute paths by making them relative
        if path.startswith("/"):
            path = path.lstrip("/")

        resolved = (self.code_root / path).resolve()

        # Security: ensure path is within code root
        if not str(resolved).startswith(str(self.code_root)):
            raise ValueError(f"Path escapes code root: {path}")

        return resolved

    def _get_security_feature(self):
        """Get the security feature for approval requests."""
        if hasattr(self.agent, 'get_feature'):
            return self.agent.get_feature("security")
        elif hasattr(self.agent, 'features'):
            return self.agent.features.get("security")
        return None

    async def _request_approval(self, action: str, details: Dict[str, Any]) -> bool:
        """Request approval for a code modification.

        Args:
            action: The action being requested (e.g., "code_edit")
            details: Details about the modification

        Returns:
            True if approved, False otherwise
        """
        security = self._get_security_feature()

        if not security or not hasattr(security, 'approval_queue'):
            logger.warning("SecurityFeature not available, cannot proceed with code edit")
            return False

        try:
            approved, scope = await security.approval_queue.request_approval(
                feature_name="code_edit",
                tool_name=action,
                tool_args=details,
                timeout=CODE_REVIEW_TIMEOUT,
            )
            return approved
        except (TimeoutError, asyncio.TimeoutError) as e:
            logger.error(f"Approval request timed out: {e}")
            return False
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Approval request failed due to invalid arguments: {e}")
            return False
        except Exception as e:
            logger.error(f"Approval request failed: {e}", exc_info=True)
            return False

    # ============== Read Operations (No Approval Required) ==============

    @tool(
        name="code_read",
        description="Read a source file from the agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!code-read"
    )
    async def code_read(
        self,
        path: str,
        start_line: int = None,
        end_line: int = None,
    ) -> Dict[str, Any]:
        """Read a source file.

        Args:
            path: Path to file relative to code root
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)

        Returns:
            Dict with file content and metadata
        """
        try:
            resolved = self._resolve_path(path)

            if not resolved.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if not resolved.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            content = resolved.read_text()
            lines = content.split('\n')

            # Handle line range
            if start_line or end_line:
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                lines = lines[start_idx:end_idx]
                content = '\n'.join(lines)

            return {
                "success": True,
                "path": str(resolved.relative_to(self.code_root)),
                "content": content,
                "total_lines": len(resolved.read_text().split('\n')),
                "shown_lines": len(lines),
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Error reading file: {e}")
            return {"success": False, "error": str(e)}
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Error decoding file content: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error reading file: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_search",
        description="Search for text in the agent's codebase.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!code-search"
    )
    async def code_search(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*.py",
    ) -> Dict[str, Any]:
        """Search for text in source files.

        Args:
            pattern: Text pattern to search for
            path: Directory to search in (relative to code root)
            file_pattern: Glob pattern for files to search

        Returns:
            Dict with matching files and lines
        """
        try:
            resolved = self._resolve_path(path)

            if not resolved.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            matches = []
            for file_path in resolved.rglob(file_pattern):
                if not file_path.is_file():
                    continue
                try:
                    content = file_path.read_text()
                    for i, line in enumerate(content.split('\n'), 1):
                        if pattern in line:
                            matches.append({
                                "file": str(file_path.relative_to(self.code_root)),
                                "line": i,
                                "content": line.strip()[:200],
                            })
                except (UnicodeDecodeError, PermissionError, OSError):
                    continue  # Skip files that can't be read

            return {
                "success": True,
                "pattern": pattern,
                "matches": matches[:50],  # Limit results
                "total_matches": len(matches),
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Error searching: {e}")
            return {"success": False, "error": str(e)}
        except ValueError as e:
            logger.error(f"Invalid search parameters: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error searching: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ============== Write Operations (Require Approval) ==============

    @tool(
        name="code_edit",
        description="Edit a source file by replacing exact text. Requires approval.",
        category=ToolCategory.SYSTEM,
        command_prefix="!code-edit"
    )
    async def code_edit(
        self,
        path: str,
        old_text: str,
        new_text: str,
        description: str = None,
    ) -> Dict[str, Any]:
        """Edit a source file by replacing exact text.

        This operation requires user approval. The old_text must match
        exactly (including whitespace) and appear exactly once in the file.

        Args:
            path: Path to file relative to code root
            old_text: Exact text to find and replace
            new_text: New text to replace with
            description: Optional description of the change

        Returns:
            Dict with success status and details
        """
        try:
            resolved = self._resolve_path(path)

            if not resolved.exists():
                return {"success": False, "error": f"File not found: {path}"}

            content = resolved.read_text()

            # Verify old_text exists exactly once
            count = content.count(old_text)
            if count == 0:
                return {
                    "success": False,
                    "error": "Text to replace not found in file",
                    "hint": "Ensure old_text matches exactly, including whitespace"
                }
            if count > 1:
                return {
                    "success": False,
                    "error": f"Text appears {count} times, must be unique",
                    "hint": "Add more context to make the match unique"
                }

            # Request approval
            approved = await self._request_approval("code_edit", {
                "path": path,
                "old_text": old_text[:500] + ("..." if len(old_text) > 500 else ""),
                "new_text": new_text[:500] + ("..." if len(new_text) > 500 else ""),
                "description": description or "Code modification",
            })

            if not approved:
                return {
                    "success": False,
                    "error": "Edit not approved",
                    "requires_approval": True,
                }

            # Apply the edit
            new_content = content.replace(old_text, new_text, 1)
            resolved.write_text(new_content)

            logger.info(f"Applied code edit to {path}: {description or 'no description'}")

            return {
                "success": True,
                "path": str(resolved.relative_to(self.code_root)),
                "description": description,
                "chars_removed": len(old_text),
                "chars_added": len(new_text),
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Error editing file: {e}")
            return {"success": False, "error": str(e)}
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Error processing file content: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error editing file: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_diff",
        description="Show uncommitted git changes in the codebase.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!code-diff"
    )
    async def code_diff(self, path: str = ".") -> Dict[str, Any]:
        """Show uncommitted changes.

        Args:
            path: Path to check (relative to code root, default: all)

        Returns:
            Dict with diff output
        """
        try:
            resolved = self._resolve_path(path)

            result = await _run_subprocess(
                [GIT_PATH, "diff", str(resolved)],
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=GIT_OPERATION_TIMEOUT,
            )

            if result.returncode != 0:
                return {"success": False, "error": result.stderr}

            return {
                "success": True,
                "diff": result.stdout or "(no changes)",
                "has_changes": bool(result.stdout.strip()),
            }
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git diff timed out: {e}")
            return {"success": False, "error": "Git diff operation timed out"}
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Error getting diff: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error getting diff: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_commit",
        description="Commit staged changes to git. Requires approval.",
        category=ToolCategory.SYSTEM,
        command_prefix="!code-commit"
    )
    async def code_commit(
        self,
        message: str,
        files: str = ".",
    ) -> Dict[str, Any]:
        """Commit changes to git.

        Args:
            message: Commit message
            files: Files to commit (default: all changes)

        Returns:
            Dict with commit details
        """
        try:
            # Request approval
            approved = await self._request_approval("code_commit", {
                "message": message,
                "files": files,
            })

            if not approved:
                return {
                    "success": False,
                    "error": "Commit not approved",
                    "requires_approval": True,
                }

            resolved = self._resolve_path(files)

            # Stage files
            await _run_subprocess(
                [GIT_PATH, "add", str(resolved)],
                cwd=self.code_root,
                check=True,
                timeout=GIT_OPERATION_TIMEOUT,
            )

            # Commit
            result = await _run_subprocess(
                [GIT_PATH, "commit", "-m", message],
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=GIT_OPERATION_TIMEOUT,
            )

            if result.returncode != 0:
                if "nothing to commit" in result.stdout:
                    return {"success": True, "message": "Nothing to commit"}
                return {"success": False, "error": result.stderr}

            # Get commit hash
            hash_result = await _run_subprocess(
                [GIT_PATH, "rev-parse", "HEAD"],
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=GIT_QUICK_TIMEOUT,
            )

            commit_hash = hash_result.stdout.strip()[:8]

            logger.info(f"Committed changes: {commit_hash} - {message}")

            return {
                "success": True,
                "commit": commit_hash,
                "message": message,
            }
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": str(e)}
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git commit timed out: {e}")
            return {"success": False, "error": "Git commit operation timed out"}
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Error committing: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error committing: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_restart",
        description="Signal that the server should restart to apply code changes.",
        category=ToolCategory.SYSTEM,
        command_prefix="!code-restart"
    )
    async def code_restart(self, reason: str = None) -> Dict[str, Any]:
        """Signal server restart.

        This sets a flag that can be checked by external process managers.
        The actual restart is handled by the deployment infrastructure.

        Args:
            reason: Optional reason for restart

        Returns:
            Dict with restart status
        """
        try:
            # Request approval
            approved = await self._request_approval("code_restart", {
                "reason": reason or "Apply code changes",
            })

            if not approved:
                return {
                    "success": False,
                    "error": "Restart not approved",
                    "requires_approval": True,
                }

            self._pending_restart = True

            # Write restart signal file
            restart_file = self.code_root / ".restart_requested"
            restart_file.write_text(reason or "Code changes applied")

            logger.info(f"Restart signaled: {reason}")

            return {
                "success": True,
                "message": "Restart signaled. Server will restart when possible.",
                "reason": reason,
            }
        except (PermissionError, OSError) as e:
            logger.error(f"Error signaling restart: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error signaling restart: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @property
    def pending_restart(self) -> bool:
        """Check if a restart has been requested."""
        return self._pending_restart

    # ============== Testing & Validation ==============

    @tool(
        name="code_test",
        description="Run pytest tests on the codebase. Requires approval for full test suite.",
        category=ToolCategory.SYSTEM,
        command_prefix="!code-test"
    )
    async def code_test(
        self,
        path: str = None,
        verbose: bool = False,
        fail_fast: bool = True,
    ) -> Dict[str, Any]:
        """Run pytest tests.

        Args:
            path: Specific test file or directory (default: all tests)
            verbose: Show verbose output
            fail_fast: Stop on first failure

        Returns:
            Dict with test results
        """
        try:
            # Build pytest command
            cmd = [PYTHON_PATH, "-m", "pytest"]

            if path:
                resolved = self._resolve_path(path)
                cmd.append(str(resolved))
            else:
                # Running full test suite requires approval
                approved = await self._request_approval("code_test", {
                    "scope": "full test suite",
                    "reason": "Running all tests",
                })
                if not approved:
                    return {
                        "success": False,
                        "error": "Full test suite requires approval",
                        "requires_approval": True,
                    }

            if verbose:
                cmd.append("-v")
            if fail_fast:
                cmd.append("-x")

            # Add timeout and capture
            cmd.extend(["--tb=short", "--no-header", "-q"])

            result = await _run_subprocess(
                cmd,
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=TEST_SUITE_TIMEOUT,
                env={**os.environ, "PYTHONPATH": str(self.code_root)},
            )

            passed = result.returncode == 0

            return {
                "success": True,
                "passed": passed,
                "return_code": result.returncode,
                "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "errors": result.stderr[-1000:] if result.stderr else None,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test timeout (5 minutes)"}
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Error running tests: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error running tests: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_lint",
        description="Run linters (ruff) on source files.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!code-lint"
    )
    async def code_lint(self, path: str = ".") -> Dict[str, Any]:
        """Run ruff linter on source files.

        Args:
            path: Path to lint (default: all)

        Returns:
            Dict with linting results
        """
        try:
            resolved = self._resolve_path(path)

            result = await _run_subprocess(
                [PYTHON_PATH, "-m", "ruff", "check", str(resolved), "--output-format=text"],
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=LINT_TIMEOUT,
                env={**os.environ, "PYTHONPATH": str(self.code_root)},
            )

            has_issues = result.returncode != 0

            return {
                "success": True,
                "has_issues": has_issues,
                "output": result.stdout[-2000:] if result.stdout else "(no issues)",
                "issue_count": result.stdout.count("\n") if result.stdout else 0,
            }
        except subprocess.TimeoutExpired as e:
            logger.error(f"Linting timed out: {e}")
            return {"success": False, "error": "Linting operation timed out"}
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Error linting: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error linting: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_logs",
        description="View recent application logs.",
        category=ToolCategory.DATA_ACCESS,
        command_prefix="!code-logs"
    )
    async def code_logs(
        self,
        lines: int = 50,
        errors_only: bool = False,
        log_file: str = None,
    ) -> Dict[str, Any]:
        """View recent logs.

        Args:
            lines: Number of lines to show
            errors_only: Only show ERROR level logs
            log_file: Specific log file (default: auto-detect)

        Returns:
            Dict with log content
        """
        try:
            # Try common log locations
            log_paths = [
                log_file,
                "/tmp/kestrel-claw.log",
                self.code_root / "logs" / "kestrel.log",
                self.code_root / "kestrel.log",
            ]

            log_content = None
            used_path = None

            for lp in log_paths:
                if lp is None:
                    continue
                lp = Path(lp)
                if lp.exists():
                    log_content = lp.read_text()
                    used_path = str(lp)
                    break

            if log_content is None:
                return {"success": False, "error": "No log file found"}

            # Get last N lines
            log_lines = log_content.split('\n')

            if errors_only:
                log_lines = [l for l in log_lines if 'ERROR' in l or 'CRITICAL' in l]

            recent = log_lines[-lines:]

            return {
                "success": True,
                "log_file": used_path,
                "lines": len(recent),
                "content": '\n'.join(recent),
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Error reading logs: {e}")
            return {"success": False, "error": str(e)}
        except (UnicodeDecodeError, ValueError) as e:
            logger.error(f"Error processing log content: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error reading logs: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @tool(
        name="code_rollback",
        description="Rollback to a previous commit. Requires approval.",
        category=ToolCategory.SYSTEM,
        command_prefix="!code-rollback"
    )
    async def code_rollback(
        self,
        commit: str = "HEAD~1",
        hard: bool = False,
    ) -> Dict[str, Any]:
        """Rollback to a previous commit.

        Args:
            commit: Commit to rollback to (default: previous commit)
            hard: Use --hard reset (discards all changes)

        Returns:
            Dict with rollback status
        """
        try:
            # Always require approval for rollback
            approved = await self._request_approval("code_rollback", {
                "commit": commit,
                "hard": hard,
                "warning": "This will modify git history",
            })

            if not approved:
                return {
                    "success": False,
                    "error": "Rollback not approved",
                    "requires_approval": True,
                }

            cmd = [GIT_PATH, "reset"]
            if hard:
                cmd.append("--hard")
            cmd.append(commit)

            result = await _run_subprocess(
                cmd,
                cwd=self.code_root,
                capture_output=True,
                text=True,
                timeout=GIT_OPERATION_TIMEOUT,
            )

            if result.returncode != 0:
                return {"success": False, "error": result.stderr}

            logger.info(f"Rolled back to {commit}")

            return {
                "success": True,
                "message": f"Rolled back to {commit}",
                "output": result.stdout,
            }
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git rollback timed out: {e}")
            return {"success": False, "error": "Git rollback operation timed out"}
        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.error(f"Error rolling back: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error rolling back: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
