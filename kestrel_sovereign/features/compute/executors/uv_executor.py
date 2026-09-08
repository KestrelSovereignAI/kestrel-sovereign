"""
Kestrel Compute Feature - UV Executor.

Execute Python scripts in project-free environments using `uv run`.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from packaging.requirements import InvalidRequirement, Requirement

from kestrel_sovereign.kestrel_config.constants import SUBPROCESS_TIMEOUT_SHORT

from ..destructive_policy import DestructiveOperationPolicy
from ..models import ComputeScript, ExecutionRecord
from .base import (
    _SAFE_ENV_VARS,
    BaseExecutor,
    ExecutionEnvironmentError,
    ExecutionError,
    ExecutionTimeoutError,
    _ExecutionContext,
    _ExecutionResult,
)

logger = logging.getLogger(__name__)

# These variables are consumed by an operating-system dynamic loader before
# the sandbox executable reaches ``main``. Applying a caller value to bwrap or
# sandbox-exec would therefore run caller-controlled code outside the custody
# boundary. They are not forwarded at all; the UV executor's security contract
# takes precedence over scripts that deliberately depend on loader injection.
_DYNAMIC_LOADER_ENV_PREFIXES = ("LD_", "DYLD_")
_DYNAMIC_LOADER_ENV_NAMES = frozenset({"LIBPATH", "SHLIB_PATH"})


def _is_dynamic_loader_environment_variable(name: str) -> bool:
    normalized = name.upper()
    return normalized in _DYNAMIC_LOADER_ENV_NAMES or normalized.startswith(
        _DYNAMIC_LOADER_ENV_PREFIXES
    )


def _validated_requirement(
    requirement: str,
    workspace: Path | None = None,
) -> str:
    """Accept a registry requirement or an executor-owned wheel.

    Resolution deliberately runs before the network namespace is detached, so
    its inputs may not name ambient host files, arbitrary URLs, VCS checkouts,
    or source trees.  Registry packages are constrained to wheels by uv's
    ``--no-build`` flag; an explicit wheel is accepted only when a trusted
    caller has already placed the regular file in this run's private workspace.
    """

    if not isinstance(requirement, str) or not requirement.strip():
        raise ExecutionEnvironmentError("UV requirements must be non-empty strings")
    candidate = requirement.strip()
    try:
        parsed = Requirement(candidate)
    except InvalidRequirement:
        wheel = Path(candidate)
        if workspace is None and wheel.suffix == ".whl":
            # The lifecycle has not allocated the workspace yet. The strict
            # identity/containment proof is repeated once it has.
            return candidate
        try:
            lexical_wheel = wheel.expanduser().absolute()
            resolved_wheel = lexical_wheel.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ExecutionEnvironmentError(
                f"UV requirement is not a valid registry package: {candidate}"
            ) from exc
        if (
            lexical_wheel.is_symlink()
            or not resolved_wheel.is_file()
            or resolved_wheel.stat().st_nlink != 1
            or resolved_wheel.suffix != ".whl"
            or not resolved_wheel.is_relative_to(workspace)
        ):
            raise ExecutionEnvironmentError(
                "UV local requirements must be regular .whl files already inside "
                "the executor-owned workspace"
            )
        return str(resolved_wheel)
    if parsed.url is not None:
        raise ExecutionEnvironmentError(
            "UV direct URL, VCS, and file requirements are not permitted; use a "
            "registry package or an executor-owned wheel"
        )
    return candidate


class UvExecutor(BaseExecutor):
    """
    Execute Python scripts using a project-free ephemeral uv environment.

    In uv 0.9, ``uv run --isolated`` alone still discovers and installs the
    current project. This executor combines ``--isolated``, ``--no-project``,
    and an explicit base interpreter path. The first forces a fresh execution
    environment, the second prevents project/workspace discovery, and the
    third anchors interpreter selection outside Kestrel's runtime as
    defense-in-depth against uv resolver changes.
    
    This executor provides:
    - Project-free ephemeral environment per execution
    - Automatic wheel-only dependency installation
    - Safe deletion rewriting
    - Resource limits via OS controls
    
    Example:
        executor = UvExecutor()
        if executor.is_available:
            record = await executor.execute(script)
    """
    
    def __init__(
        self,
        uv_path: Optional[str] = None,
        max_output_bytes: int = 1024 * 1024,  # 1MB default
        current_agent_data_path: Optional[str | Path] = None,
    ):
        """
        Initialize the UV executor.
        
        Args:
            uv_path: Path to uv binary (default: auto-detect)
            max_output_bytes: Maximum stdout/stderr size to capture
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._uv_path = uv_path
        self._cached_uv_path: Optional[str] = None
        self._policy = DestructiveOperationPolicy(
            current_agent_data_path=current_agent_data_path
        )
    
    @property
    def name(self) -> str:
        return "uv"
    
    @property
    def is_available(self) -> bool:
        """Check uv, an isolated interpreter, and a usable OS sandbox."""
        try:
            path = self._get_uv_path()
            if path is None:
                return False
            base_python_path = self._get_base_python_path()
            sandbox_prefix = self._get_filesystem_sandbox_prefix(
                uv_path=path,
                base_python_path=base_python_path,
            )
            return self._filesystem_sandbox_is_operational(
                sandbox_prefix,
                base_python_path,
            )
        except (
            ExecutionEnvironmentError,
            FileNotFoundError,
            OSError,
            PermissionError,
            subprocess.SubprocessError,
        ):
            return False
    
    def _get_uv_path(self) -> Optional[str]:
        """Find the concrete uv executable rather than a host-only alias.

        The Linux sandbox starts from an empty root.  Shell-visible uv paths
        are commonly symlinks into a package-manager store (Homebrew,
        Linuxbrew, Nix), so preserving the alias would leave the invoked path
        absent even though its resolved target is mounted read-only.
        """
        if self._cached_uv_path:
            return self._cached_uv_path

        # Try common locations
        candidates = [
            self._uv_path,
            shutil.which("uv"),
            os.path.expanduser("~/.cargo/bin/uv"),
            "/usr/local/bin/uv",
            "/opt/homebrew/bin/uv",
        ]

        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved = Path(candidate).resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved.is_file() and os.access(resolved, os.X_OK):
                self._cached_uv_path = str(resolved)
                return self._cached_uv_path

        return None

    @staticmethod
    def _get_base_python_path() -> str:
        """Resolve the executable outside Kestrel's virtual environment.

        ``--isolated --no-project`` provides the currently verified fresh,
        project-free behavior. The concrete executable additionally makes the
        interpreter choice independent of the working directory and fails
        closed if future uv resolution behavior would otherwise select
        Kestrel's runtime. Running Kestrel outside a virtual environment cannot
        provide that independent trust anchor, so it is rejected.
        """
        if sys.prefix == sys.base_prefix:
            raise ExecutionEnvironmentError(
                "UvExecutor requires Kestrel to run inside a Python venv or "
                "virtualenv so compute scripts cannot inherit Kestrel's "
                "site-packages; a Conda environment alone is not sufficient"
            )

        base_prefix = Path(sys.base_prefix)
        candidates: list[Path] = []
        base_executable = getattr(sys, "_base_executable", None)
        if base_executable:
            candidates.append(Path(base_executable))

        if os.name == "nt":
            candidates.extend(
                [
                    base_prefix / "python.exe",
                    base_prefix / "Scripts" / "python.exe",
                ]
            )
        else:
            candidates.extend(
                [
                    base_prefix
                    / "bin"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}",
                    base_prefix / "bin" / "python3",
                    base_prefix / "bin" / "python",
                ]
            )

        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())

        raise ExecutionEnvironmentError(
            "UvExecutor could not resolve an executable base Python interpreter"
        )

    def _get_filesystem_sandbox_prefix(
        self,
        writable_workspace: Optional[str] = None,
        *,
        uv_path: Optional[str] = None,
        base_python_path: Optional[str] = None,
    ) -> list[str]:
        """Build a fail-closed OS boundary around host Hold custody.

        Python-level ``open`` patches cannot mediate C extensions such as
        ``sqlite3`` or a dependency that issues raw syscalls. The UV executor
        therefore runs only when the platform can omit host-control state and
        every ambient credential/service path from the child's mount namespace.
        Linux constructs a minimal filesystem from trusted interpreter/runtime
        roots and the freshly allocated executor workspace. It also creates new
        network and IPC namespaces, so filesystem sockets and remote Hold
        services are not alternate mutation channels.
        """

        protected = self._policy.host_control_data_path.resolve(strict=False)
        if not protected.is_dir():
            raise ExecutionEnvironmentError(
                "UvExecutor requires an existing host Hold custody directory "
                f"before compute can run: {protected}"
            )

        if sys.platform == "darwin":
            # Seatbelt path filters cannot make an inode read-only. A
            # pre-existing hard-link alias outside ``protected`` remains
            # writable even when file-link is denied and every write below the
            # canonical custody path is denied. There is no race-free way for
            # this executor to enumerate every same-filesystem alias, so native
            # macOS UV execution cannot uphold the Hold custody contract.
            raise ExecutionEnvironmentError(
                "UvExecutor is unavailable on macOS because sandbox-exec cannot "
                "protect host Hold custody from pre-existing hard-link aliases; "
                "use the Docker executor"
            )

        if sys.platform.startswith("linux"):
            bubblewrap = shutil.which("bwrap")
            if not bubblewrap:
                raise ExecutionEnvironmentError(
                    "UvExecutor requires bubblewrap on Linux to protect host "
                    "Hold custody; install bwrap or use the Docker executor"
                )
            workspace: Optional[Path] = None
            if writable_workspace is not None:
                workspace = Path(writable_workspace).resolve(strict=True)
                if not workspace.is_dir():
                    raise ExecutionEnvironmentError(
                        "UvExecutor writable workspace must be an existing directory"
                    )
                if self._policy.touches_host_hold_custody(workspace):
                    raise ExecutionEnvironmentError(
                        "UvExecutor writable workspace overlaps host Hold custody"
                    )

            selected_uv = uv_path or self._get_uv_path()
            if not selected_uv:
                raise ExecutionEnvironmentError(
                    "UvExecutor cannot construct a sandbox without the uv binary"
                )
            resolved_uv = Path(selected_uv).resolve(strict=False)
            resolved_python = Path(
                base_python_path or self._get_base_python_path()
            ).resolve(strict=False)

            # Explicitly discard capabilities even when Kestrel itself runs as
            # UID 0 in a service container. Start with an empty root rather
            # than importing ``/``: a read-only host bind still exposes .env
            # files and permits connect(2) through Unix service sockets.
            prefix = [
                bubblewrap,
                "--die-with-parent",
                "--new-session",
                "--unshare-all",
                "--cap-drop",
                "ALL",
                "--tmpfs",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
            ]

            created_directories: set[Path] = set()

            def ensure_parent_directories(destination: Path) -> None:
                for parent in reversed(destination.parents):
                    if parent == Path("/") or parent in created_directories:
                        continue
                    prefix.extend(["--dir", str(parent)])
                    created_directories.add(parent)

            runtime_roots: list[Path] = []

            def add_runtime_root(candidate: Path) -> None:
                candidate = candidate.resolve(strict=False)
                if any(
                    candidate == existing or candidate.is_relative_to(existing)
                    for existing in runtime_roots
                ):
                    return
                runtime_roots[:] = [
                    existing
                    for existing in runtime_roots
                    if not existing.is_relative_to(candidate)
                ]
                runtime_roots.append(candidate)

            # The base interpreter prefix is explicitly outside Kestrel's venv.
            # Standard ELF runtime roots are system-owned and contain the
            # loader/shared libraries needed by that interpreter and uv. No
            # project, home, run, or var directory is imported.
            add_runtime_root(Path(sys.base_prefix))
            for candidate in (Path("/lib"), Path("/lib64"), Path("/usr/lib")):
                if candidate.exists():
                    add_runtime_root(candidate)
            add_runtime_root(resolved_uv)
            add_runtime_root(resolved_python)
            for source in sorted(runtime_roots, key=lambda item: len(item.parts)):
                ensure_parent_directories(source)
                prefix.extend(["--ro-bind", str(source), str(source)])

            # Dynamic executables may need these public loader/time files, but
            # importing /etc wholesale would reintroduce ambient credentials.
            for public_runtime_file in (
                Path("/etc/ld.so.cache"),
                Path("/etc/localtime"),
            ):
                if public_runtime_file.is_file():
                    ensure_parent_directories(public_runtime_file)
                    prefix.extend(
                        [
                            "--ro-bind",
                            str(public_runtime_file),
                            str(public_runtime_file),
                        ]
                    )
            if workspace is not None:
                ensure_parent_directories(workspace)
                prefix.extend(
                    [
                        "--bind",
                        str(workspace),
                        str(workspace),
                    ]
                )
            prefix.append("--")
            return prefix

        raise ExecutionEnvironmentError(
            "UvExecutor has no verified host Hold filesystem sandbox on this "
            "platform; use the Docker executor"
        )

    @staticmethod
    def _filesystem_sandbox_is_operational(
        prefix: list[str],
        base_python_path: str,
    ) -> bool:
        """Prove this process may enter the sandbox before advertising UV.

        The sandbox executable can exist while a parent macOS sandbox forbids
        nested profiles or a Linux host disables unprivileged user namespaces.
        A no-op isolated interpreter launch exercises sandbox creation without
        running caller code or writing to the filesystem.
        """

        result = subprocess.run(
            [*prefix, base_python_path, "-I", "-S", "-c", "pass"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=SUBPROCESS_TIMEOUT_SHORT,
            check=False,
        )
        return result.returncode == 0
    
    def supports_language(self, language: str) -> bool:
        """UV executor only supports Python."""
        return language == "python"
    
    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """
        Execute a Python script using uv run.
        
        Creates a temporary directory, writes the script (with safe deletion
        wrapper), resolves declared registry wheels into its private cache, and
        runs offline with uv inside the OS sandbox.
        
        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory for execution
            
        Returns:
            ExecutionRecord with execution results
        """
        if script.language != "python":
            raise ExecutionError(f"UvExecutor only supports Python, got {script.language}")
        if working_dir is not None:
            raise ExecutionEnvironmentError(
                "UvExecutor does not expose host working directories inside its "
                "minimal Hold-safe namespace; place required inputs in the script "
                "or use the Docker executor"
            )
        for requirement in script.requirements:
            _validated_requirement(requirement)
        
        uv_path = self._get_uv_path()
        if not uv_path:
            raise ExecutionError("uv binary not found")
        base_python_path = self._get_base_python_path()

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            return await self._execute_script(
                script,
                context,
                uv_path,
                base_python_path,
            )

        return await self._execute_with_lifecycle(
            script,
            temp_dir_prefix="kestrel_compute_",
            runner=run,
        )

    async def _execute_script(
        self,
        script: ComputeScript,
        context: _ExecutionContext,
        uv_path: str,
        base_python_path: str,
    ) -> _ExecutionResult:
        # Only the executor-owned temp dir authorizes direct deletion; the
        # Python runtime resolves relative paths against the child's cwd.
        safe_content = self._policy.rewrite_script(
            script.content,
            "python",
            context.workdir,
        )

        script_path = Path(context.workdir) / "script.py"
        script_path.write_text(safe_content)

        workspace = Path(context.workdir).resolve(strict=True)
        requirements = tuple(
            _validated_requirement(requirement, workspace)
            for requirement in script.requirements
        )

        # Only pass safe host variables; never leak host credentials to scripts.
        env = {key: value for key, value in os.environ.items() if key in _SAFE_ENV_VARS}
        # Apply script-supplied overrides, then enforce Python isolation below.
        env.update(script.environment)
        # The minimal namespace does not import host HOME or TMPDIR. Pin every
        # runtime-owned writable/cache path into this execution's sole bind;
        # caller overrides cannot name an ambient host location.
        env["HOME"] = context.workdir
        env["TMPDIR"] = str(Path(context.workdir) / "tmp")
        env["UV_CACHE_DIR"] = str(Path(context.workdir) / ".uv-cache")
        env["UV_NO_CONFIG"] = "1"
        env["UV_NO_BUILD"] = "1"
        env["UV_PYTHON_DOWNLOADS"] = "never"
        env["UV_OFFLINE"] = "1"
        Path(env["TMPDIR"]).mkdir(mode=0o700)
        # PYTHONPATH bypasses uv's interpreter/environment boundary entirely.
        env.pop("PYTHONPATH", None)
        # The subprocess environment is installed on the sandbox process
        # itself. A dynamic loader acts on these keys before bwrap/Seatbelt can
        # establish the read-only Hold mount, so strip every platform spelling
        # rather than trying to sanitize individual values.
        for key in tuple(env):
            if _is_dynamic_loader_environment_variable(key):
                del env[key]

        uv_base_cmd = [
            uv_path,
            "run",
            "--isolated",
            "--no-project",
            "--no-config",
            "--no-build",
            "--no-python-downloads",
            "--python",
            base_python_path,
        ]
        for requirement in requirements:
            uv_base_cmd.extend(["--with", requirement])

        deadline = asyncio.get_running_loop().time() + script.timeout_seconds
        if requirements:
            # uv is trusted infrastructure, but dependency packages are not.
            # Resolve wheel-only registry inputs into the private cache while
            # running only an isolated, site-disabled interpreter no-op.  The
            # caller's environment is deliberately absent here, and the later
            # script process receives neither host networking nor host files.
            resolver_env = {
                key: value
                for key, value in os.environ.items()
                if key in _SAFE_ENV_VARS
            }
            resolver_env.update(
                {
                    "HOME": context.workdir,
                    "TMPDIR": env["TMPDIR"],
                    "UV_CACHE_DIR": env["UV_CACHE_DIR"],
                    "UV_NO_CONFIG": "1",
                    "UV_NO_BUILD": "1",
                    "UV_PYTHON_DOWNLOADS": "never",
                }
            )
            resolver_env.pop("PYTHONPATH", None)
            resolver_env.pop("VIRTUAL_ENV", None)
            resolver_env.pop("UV_OFFLINE", None)
            resolver_cmd = [
                *uv_base_cmd,
                "--",
                base_python_path,
                "-I",
                "-S",
                "-c",
                "pass",
            ]
            resolver = await asyncio.create_subprocess_exec(
                *resolver_cmd,
                cwd=context.workdir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=resolver_env,
                start_new_session=os.name == "posix",
            )
            try:
                resolver_stdout, resolver_stderr = await self._capture_process_output(
                    resolver,
                    timeout_seconds=max(
                        0.001,
                        deadline - asyncio.get_running_loop().time(),
                    ),
                    terminate=lambda: self._kill_process_group(resolver),
                )
            except TimeoutError:
                raise ExecutionTimeoutError(
                    script.id, script.timeout_seconds
                ) from None
            if resolver.returncode != 0:
                return _ExecutionResult(
                    exit_code=resolver.returncode,
                    stdout=resolver_stdout,
                    stderr=resolver_stderr,
                )

        uv_cmd = [*uv_base_cmd, "--offline"]
        uv_cmd.append(str(script_path))
        cmd = [
            *self._get_filesystem_sandbox_prefix(
                context.workdir,
                uv_path=uv_path,
                base_python_path=base_python_path,
            ),
            *uv_cmd,
        ]

        logger.info("Executing script %s... with uv", script.id[:8])
        logger.debug("Command: %s", " ".join(cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=context.workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=os.name == "posix",
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=max(
                    0.001,
                    deadline - asyncio.get_running_loop().time(),
                ),
                terminate=lambda: self._kill_process_group(process),
            )
        except TimeoutError:
            raise ExecutionTimeoutError(script.id, script.timeout_seconds) from None

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
