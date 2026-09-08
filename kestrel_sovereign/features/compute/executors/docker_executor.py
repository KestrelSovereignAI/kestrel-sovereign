"""
Kestrel Compute Feature - Docker Executor.

Execute scripts in isolated Docker containers for maximum security.
"""

import asyncio
import json
import logging
import os
import shlex
import shutil
import stat
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from kestrel_sovereign.kestrel_config.constants import SUBPROCESS_TIMEOUT_SHORT

from ..destructive_policy import DestructiveOperationPolicy
from ..models import ComputeCommand, ComputeScript, ExecutionRecord
from ..trash_manager import _rename_noreplace
from .base import (
    BaseExecutor,
    ExecutionEnvironmentError,
    ExecutionError,
    ExecutionTimeoutError,
    _ExecutionContext,
    _ExecutionResult,
)

logger = logging.getLogger(__name__)


# Default images for script execution
DEFAULT_IMAGES = {
    "bash": "alpine:3.19",
    "python": "python:3.11-slim",
}

# Image for argv execution. Keyed by nothing, because an argv vector has
# no language: element zero names a program, and the image is simply
# where that program has to exist. Named separately from
# ``DEFAULT_IMAGES["bash"]`` even though it is the same image today —
# the script entry means "the image whose shell runs bash scripts", and
# reusing it here would re-attach a shell to a path that has none.
DEFAULT_COMMAND_IMAGE = "alpine:3.19"
_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS = 1.0

_CONTAINER_TRASH_DIR = "/kestrel-trash"


class DockerExecutor(BaseExecutor):
    """
    Execute scripts in Docker containers for maximum isolation.

    Security measures:
    - Read-only root filesystem
    - No network by default
    - Resource limits (CPU, memory)
    - No privilege escalation
    - PID namespace isolation

    Example:
        executor = DockerExecutor()
        if executor.is_available:
            record = await executor.execute(script, network=False)
    """

    def __init__(
        self,
        docker_path: Optional[str] = None,
        images: Optional[Dict[str, str]] = None,
        default_memory_limit: str = "256m",
        default_cpu_quota: int = 50000,  # 50% of one CPU
        default_pids_limit: int = 50,
        max_output_bytes: int = 1024 * 1024,  # 1MB
        current_agent_data_path: Optional[str | Path] = None,
        # Appended rather than grouped with `images`, which is where it
        # belongs by meaning: `DockerExecutor` is a package-level export
        # that already accepted these positions, and inserting a
        # parameter mid-list silently remaps every caller that passed
        # one positionally.
        command_image: Optional[str] = None,
        legacy_staging_age_seconds: Optional[int] = None,
    ):
        """
        Initialize the Docker executor.

        Args:
            docker_path: Path to docker binary (default: auto-detect)
            images: Docker images by language (default: alpine for bash, python:3.11-slim for python)
            command_image: Docker image for argv execution (default: alpine)
            default_memory_limit: Memory limit for containers
            default_cpu_quota: CPU quota (microseconds per 100ms)
            default_pids_limit: Maximum number of processes
            max_output_bytes: Maximum stdout/stderr size
        """
        super().__init__(max_output_bytes=max_output_bytes)
        self._docker_path = docker_path
        self._cached_docker_path: Optional[str] = None
        self._images = images or DEFAULT_IMAGES
        self._command_image = command_image or DEFAULT_COMMAND_IMAGE
        self._memory_limit = default_memory_limit
        self._cpu_quota = default_cpu_quota
        self._pids_limit = default_pids_limit
        self._policy = DestructiveOperationPolicy(
            current_agent_data_path=current_agent_data_path
        )
        # A record-less staging directory (made by code that wrote no record)
        # is swept once older than the longest a script may run under the
        # policy this executor serves; the feature passes its configured
        # maximum, the default is the policy's shipped default.
        self._legacy_staging_age_seconds = (
            int(legacy_staging_age_seconds or self.LEGACY_STAGING_AGE_SECONDS)
            + self.LEGACY_STAGING_GRACE_SECONDS
        )

    @property
    def name(self) -> str:
        return "docker"

    @property
    def is_available(self) -> bool:
        """Check if Docker is installed and the daemon is running."""
        try:
            docker_path = self._get_docker_path()
            if not docker_path:
                return False

            # Check if daemon is running (synchronous check)
            result = subprocess.run(
                [docker_path, "info"],
                capture_output=True,
                timeout=SUBPROCESS_TIMEOUT_SHORT,
            )
            return result.returncode == 0

        except subprocess.TimeoutExpired:
            return False
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def _get_docker_path(self) -> Optional[str]:
        """Find the docker binary path."""
        if self._cached_docker_path:
            return self._cached_docker_path

        if self._docker_path and shutil.which(self._docker_path):
            self._cached_docker_path = self._docker_path
            return self._docker_path

        docker_path = shutil.which("docker")
        if docker_path:
            self._cached_docker_path = docker_path
            return docker_path

        return None

    async def execute(
        self,
        script: ComputeScript,
        working_dir: Optional[str] = None,
        network: bool = False,
        mounts: Optional[List[Dict[str, str]]] = None,
    ) -> ExecutionRecord:
        """
        Execute a script in a Docker container.

        Args:
            script: The ComputeScript to execute
            working_dir: Optional working directory (mounted read-only)
            network: Whether to allow network access (default: False)
            mounts: Additional mounts [{"src": "/host/path", "dst": "/container/path", "ro": True}]

        Returns:
            ExecutionRecord with execution results
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            raise ExecutionEnvironmentError("Docker not found")

        image = self._images.get(script.language)
        if not image:
            raise ExecutionError(
                f"No Docker image configured for language: {script.language}"
            )

        self._validate_additional_mounts(mounts)

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            container_name = self._container_name(context.execution_id)
            return await self._execute_script(
                script,
                working_dir,
                context,
                docker_path=docker_path,
                image=image,
                container_name=container_name,
                network=network,
                mounts=mounts,
            )

        async def cleanup(context: _ExecutionContext) -> None:
            await self._remove_container(
                docker_path,
                self._container_name(context.execution_id),
            )

        return await self._execute_with_lifecycle(
            script,
            temp_dir_prefix="kestrel_compute_docker_",
            runner=run,
            cleanup=cleanup,
        )

    def _validate_additional_mounts(
        self,
        mounts: Optional[List[Dict[str, str]]],
    ) -> None:
        """Keep caller-selected host paths read-only inside compute containers.

        Path containment cannot prove that a writable source is independent of
        Hold custody: a file anywhere below it may be a hard-link alias of a
        protected inode, and another same-user process can add such an alias
        after a recursive preflight scan.  The executor-owned per-run trash
        staging directory is the sole writable host bind and is assembled
        internally, so arbitrary additional mounts have no safe writable mode.
        """

        for mount in mounts or []:
            src = mount.get("src")
            dst = mount.get("dst")
            if not src or not dst:
                continue
            if dst == _CONTAINER_TRASH_DIR or dst.startswith(
                f"{_CONTAINER_TRASH_DIR}/"
            ):
                raise ExecutionError(
                    f"Mount destination is reserved: {_CONTAINER_TRASH_DIR}"
                )
            if not mount.get("ro", True):
                raise ExecutionEnvironmentError(
                    "Refusing writable Docker mount because an arbitrary host "
                    "path cannot prove separation from host Hold custody; "
                    f"additional mounts must be read-only: {src}"
                )

    async def _execute_script(
        self,
        script: ComputeScript,
        working_dir: Optional[str],
        context: _ExecutionContext,
        *,
        docker_path: str,
        image: str,
        container_name: str,
        network: bool,
        mounts: Optional[List[Dict[str, str]]],
    ) -> _ExecutionResult:
        host_trash_dir = self._policy.trash_dir.expanduser().resolve(strict=False)
        host_trash_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Mount a PER-EXECUTION staging directory, never the shared trash
        # root: a read/write bind of the root would let any container script
        # read or corrupt entries trashed by previous runs and other agents.
        # Staged entries are promoted into the real trash root host-side
        # after the container exits (same filesystem, atomic renames).
        await self._promote_stale_staging_dirs(host_trash_dir, docker_path)
        staging_dir = host_trash_dir / f".staging-{uuid.uuid4().hex[:12]}"
        # Guarded from the moment anything exists: the owner record, then
        # the directory, then every exit below (a policy rewrite that
        # refuses the script, the script write, a docker binary that is not
        # there when the process is spawned, a timeout) promotes and removes
        # both. Before this guard began at the process wait, each failed
        # launch left an empty hidden directory in the trash root that no
        # listing could see: 589 of them on one host (#3117).
        try:
            # The record lives BESIDE the directory, outside the bind mount,
            # so the container cannot read or alter its own; it is written
            # first, so a directory with no record is one this code never
            # owned (see ``_promote_stale_staging_dirs``).
            self._write_staging_owner(staging_dir, container_name)
            staging_dir.mkdir(mode=0o700)
            return await self._run_staged_script(
                script,
                working_dir,
                context,
                docker_path=docker_path,
                image=image,
                container_name=container_name,
                network=network,
                mounts=mounts,
                staging_dir=staging_dir,
            )
        finally:
            self._promote_staged_trash(staging_dir, host_trash_dir)

    async def _run_staged_script(
        self,
        script: ComputeScript,
        working_dir: Optional[str],
        context: _ExecutionContext,
        *,
        docker_path: str,
        image: str,
        container_name: str,
        network: bool,
        mounts: Optional[List[Dict[str, str]]],
        staging_dir: Path,
    ) -> _ExecutionResult:
        """Rewrite, stage and run the script against an existing staging dir."""
        # Container mounts (/scripts, /workspace) are read-only, so no
        # workdir is authorized for direct deletion; every delete moves to
        # the trash bind mount.  The container cwd only resolves relative
        # operands for policy checks — which is why it is bound once and
        # passed to both the rewriter and the container: if the two ever
        # disagreed, the rewriter would vet a different path than the one
        # the script actually names.
        container_cwd = "/workspace" if working_dir else "/scripts"
        safe_content = self._policy.rewrite_script(
            script.content,
            script.language,
            None,
            runtime_trash_dir=_CONTAINER_TRASH_DIR,
            script_cwd=container_cwd,
        )

        if script.language == "python":
            interpreter, script_name = "python", "script.py"
        else:
            interpreter, script_name = "sh", "script.sh"
        script_argument = f"/scripts/{script_name}"
        script_path = Path(context.workdir) / script_name
        script_path.write_text(safe_content)
        script_path.chmod(0o755)

        cmd, log_safe_cmd = self._container_invocation(
            docker_path=docker_path,
            container_name=container_name,
            image=image,
            working_dir=working_dir,
            container_cwd=container_cwd,
            network=network,
            mounts=mounts,
            environment=script.environment,
            binds=[
                f"{context.workdir}:/scripts:ro",
                # Safe deletions must survive the container.  The rewriter
                # uses the container path while this dedicated bind mount
                # anchors it to the host's configured Kestrel trash
                # directory.
                f"{staging_dir}:{_CONTAINER_TRASH_DIR}:rw",
            ],
            program=interpreter,
        )
        cmd.append(script_argument)
        log_safe_cmd.append(script_argument)

        logger.info("Executing script %s... in Docker container", script.id[:8])
        logger.debug("Container command: %s", " ".join(log_safe_cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=script.timeout_seconds,
                terminate=lambda: self._kill_container(
                    docker_path,
                    container_name,
                ),
            )
        except TimeoutError:
            # The caller's guard still promotes staged entries: deletions
            # performed before the interruption already happened, and their
            # trash entries must stay restorable from the real trash root.
            raise ExecutionTimeoutError(script.id, script.timeout_seconds) from None

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            container_id=container_name,
        )

    def _container_invocation(
        self,
        *,
        docker_path: str,
        container_name: str,
        image: str,
        working_dir: Optional[str],
        container_cwd: str,
        network: bool,
        mounts: Optional[List[Dict[str, str]]],
        environment: Dict[str, str],
        binds: List[str],
        program: str,
    ) -> tuple[List[str], List[str]]:
        """Build ``docker run`` up to and including the image.

        Both execution modes build their container here so the vetted
        isolation flags (``--read-only``, ``--network=none``,
        ``--security-opt=no-new-privileges``, memory and pid limits)
        cannot drift apart between them.

        Returns ``(cmd, log_safe_cmd)``. The second is identical except
        that environment values are redacted, so a debug log of the
        container command cannot leak a secret the caller passed in.

        ``program`` is what will run, and it is pinned with
        ``--entrypoint`` rather than left to position. Words after the
        image are not the process argv: Docker appends them to whatever
        ``ENTRYPOINT`` the image declares, so an image with one runs its
        own program with the caller's first word demoted to an argument.
        Measured on an image built with
        ``ENTRYPOINT ["/bin/echo", "ENTRYPOINT-RAN"]``: a vector of
        ``["printf", "HACKED"]`` printed ``ENTRYPOINT-RAN printf
        HACKED`` — ``echo`` ran while the policy had vetted ``printf``.
        That is #3187 again, one layer down, and it is why the program
        is named to Docker instead of positioned after the image.

        ``--entrypoint`` also clears the image's default ``CMD``
        (measured, not read: ``docker run --entrypoint printf alpine``
        runs ``printf`` with no arguments, where the same run without
        the override starts the image's shell). So a caller appending
        nothing gets its own program with no arguments, never the
        image's idea of what to do.

        It is a required parameter for the same reason: the two modes
        share this builder, and a mode that forgot to pin its program
        would silently inherit the image's.
        """
        cmd = [
            docker_path,
            "run",
            "--name",
            container_name,
            "--rm",
            "--read-only",
            f"--memory={self._memory_limit}",
            f"--cpu-quota={self._cpu_quota}",
            f"--pids-limit={self._pids_limit}",
            "--security-opt=no-new-privileges",
        ]
        if not network:
            cmd.append("--network=none")

        for bind in binds:
            cmd.extend(["-v", bind])

        if working_dir:
            cmd.extend(["-v", f"{working_dir}:/workspace:ro"])
        cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        for mount in mounts or []:
            src = mount.get("src")
            dst = mount.get("dst")
            read_only = mount.get("ro", True)
            if src and dst:
                ro_flag = ":ro" if read_only else ""
                cmd.extend(["-v", f"{src}:{dst}{ro_flag}"])

        cmd.extend(["-w", container_cwd])
        log_safe_cmd = list(cmd)

        # The program is the caller's text on the command path. Its name
        # is worth logging — it is what the policy vetted — but not
        # raw: a newline in it would forge whole log lines. ``repr``
        # escapes them, and ``shlex.join`` alone would not (it quotes a
        # newline, it does not encode it).
        cmd.extend(["--entrypoint", program])
        log_safe_cmd.extend(["--entrypoint", repr(program)])
        for key, value in environment.items():
            cmd.extend(["-e", f"{key}={value}"])
            log_safe_cmd.extend(["-e", f"{key}=<redacted>"])

        cmd.append(image)
        log_safe_cmd.append(image)
        return cmd, log_safe_cmd

    async def execute_command(
        self,
        command: ComputeCommand,
        working_dir: Optional[str] = None,
    ) -> ExecutionRecord:
        """Execute an argv vector in a container. No script, no shell.

        ``command.argv[0]`` is the program and every later element is an
        argument to it — pinned with ``--entrypoint`` rather than left
        to position, for the reason :meth:`_container_invocation`
        records. That is the whole difference from :meth:`execute`: a
        script's first word is read by a shell's grammar first, which is
        how a vetted ``eval`` ran an unvetted ``printf`` (#3187).

        No trash mount is created. The rewriter that redirects deletions
        into it only rewrites script text, and there is no script text
        here — mounting a writable host directory that nothing can be
        rewritten to use would be a hole with no purpose. Every other
        mount is read-only, so the container has nothing of the host's
        to delete.

        The signature is the base contract exactly: no network, no
        extra mounts. :meth:`execute` takes both because
        ``ComputeFeature`` passes them for a reviewed, signed script;
        nothing asks it of a one-shot vector, and an unused parameter is
        an untested way to widen a container.

        Args:
            command: The :class:`ComputeCommand` to execute
            working_dir: Optional working directory (mounted read-only)

        Returns:
            ExecutionRecord with execution results
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            raise ExecutionEnvironmentError("Docker not found")

        async def run(context: _ExecutionContext) -> _ExecutionResult:
            container_name = self._container_name(context.execution_id)
            return await self._execute_argv(
                command,
                working_dir,
                docker_path=docker_path,
                container_name=container_name,
            )

        async def cleanup(context: _ExecutionContext) -> None:
            await self._remove_container(
                docker_path,
                self._container_name(context.execution_id),
            )

        return await self._execute_with_lifecycle(
            command,
            temp_dir_prefix="kestrel_compute_docker_command_",
            runner=run,
            cleanup=cleanup,
        )

    async def _execute_argv(
        self,
        command: ComputeCommand,
        working_dir: Optional[str],
        *,
        docker_path: str,
        container_name: str,
    ) -> _ExecutionResult:
        cmd, log_safe_cmd = self._container_invocation(
            docker_path=docker_path,
            container_name=container_name,
            image=self._command_image,
            working_dir=working_dir,
            # With no host directory to mount there is no meaningful
            # workspace; the image's read-only root is a defined place
            # to stand rather than an inherited one.
            container_cwd="/workspace" if working_dir else "/",
            network=False,
            mounts=None,
            environment=command.environment,
            binds=[],
            program=command.argv[0],
        )
        cmd.extend(command.argv[1:])
        # Count, not contents. Arguments are where a caller's secrets
        # live — a bearer token, a `--password` — and the environment
        # values on this same line are redacted for exactly that
        # reason. The script path never logged them either: they were
        # inside a file, and the line ended at `sh /scripts/script.sh`.
        # Restoring them here would have been a new sink for secrets,
        # and a way to forge log lines with an embedded newline.
        log_safe_cmd.append(f"<{len(command.argv) - 1} argument(s) not logged>")

        logger.info("Executing command %s... in Docker container", command.id[:8])
        # Quoted, unlike the script path's log line: here the reader is
        # looking at a vector whose word boundaries are the point, and a
        # space-joined rendering of ["printf", "a b"] reads as three
        # arguments. `shlex.join` makes the log line reproduce the run.
        logger.debug("Container command: %s", shlex.join(log_safe_cmd))
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await self._capture_process_output(
                process,
                timeout_seconds=command.timeout_seconds,
                terminate=lambda: self._kill_container(
                    docker_path,
                    container_name,
                ),
            )
        except TimeoutError:
            raise ExecutionTimeoutError(
                command.id, command.timeout_seconds
            ) from None

        return _ExecutionResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            container_id=container_name,
        )

    #: Suffix of the owner record written beside a staging directory
    #: (``.staging-<hex>.owner``), outside the container's bind mount.
    STAGING_OWNER_SUFFIX = ".owner"

    #: A record-bearing directory older than this is swept even though its
    #: recorded pid answers ``kill -0``: after a reboot or a pid wraparound
    #: the pid belongs to another process (one this user cannot signal
    #: counts as alive, which widens it), and no script runs for a week. The
    #: record's ``started`` stamp is the age; the directory's mtime moves
    #: with every staged entry and is not.
    OWNER_MAX_AGE_SECONDS = 7 * 24 * 60 * 60

    #: Added to the legacy floor: a script may run for the whole maximum, and
    #: its promotion happens only after the timeout fires, the container is
    #: killed and its output drained, so the floor must exceed the maximum.
    LEGACY_STAGING_GRACE_SECONDS = 15 * 60

    #: The most wall-clock one sweep may spend asking Docker about dead
    #: owners' containers. Each ``docker inspect`` is bounded, but the sweep
    #: runs at the head of every script and a wedged daemon answers nothing;
    #: directories not reached this run are left for the next.
    SWEEP_INSPECT_BUDGET_SECONDS = 20.0

    #: Default for ``legacy_staging_age_seconds``: the compute policy's
    #: shipped maximum script timeout. A staging directory with NO owner
    #: record was made by code that wrote none (the leak this ticket closes,
    #: or a run of that older code still in flight across an upgrade) and is
    #: swept once older than this; a record-bearing directory is judged by
    #: its owner, never by age.
    LEGACY_STAGING_AGE_SECONDS = 60 * 60

    @classmethod
    def _write_staging_owner(cls, staging_dir: Path, container_name: str) -> None:
        """Record which process and container own ``staging_dir``."""
        record = staging_dir.with_name(staging_dir.name + cls.STAGING_OWNER_SUFFIX)
        record.write_text(json.dumps({
            "pid": os.getpid(),
            "container": container_name,
            "started": time.time(),
        }))
        record.chmod(0o600)

    @staticmethod
    def _read_owner_record(record: Path) -> Optional[Dict[str, object]]:
        """The owner record's fields, or ``None`` when it is not one of ours.

        A record is a small JSON object with a bounded positive ``pid``;
        anything else (prose, a forged value that overflows a C int, a
        symlink) is no record at all.
        """
        try:
            if record.is_symlink() or not record.is_file():
                return None
            data = json.loads(record.read_text(encoding="utf-8", errors="replace")[:4096])
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        pid = data.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or not 0 < pid < 2**31:
            return None
        container = data.get("container")
        started = data.get("started")
        return {
            "pid": pid,
            "container": container if isinstance(container, str) and container else None,
            "started": float(started) if isinstance(started, (int, float)) and not isinstance(started, bool) else None,
        }

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """Whether ``pid`` names a running process; one this user may not
        signal is still a live process."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @classmethod
    async def _container_exists(cls, docker_path: Optional[str], container_name: str) -> bool:
        """Whether Docker still knows ``container_name``.

        ``docker run --rm`` is a foreground client: a killed agent leaves its
        container running with the staging bind for as long as the script
        runs, and Docker removes the container when it exits. So "inspect
        succeeds" means the run is still live; without a docker binary
        nothing can be running.
        """
        if not docker_path:
            return False
        try:
            code = await cls._run_control_command(docker_path, "inspect", container_name)
        except Exception:  # noqa: BLE001 - a wedged daemon must not fail the sweep
            logger.debug("docker inspect %s failed", container_name, exc_info=True)
            return True  # unknown: leave the directory alone
        if code is None:
            # The client timed out or could not be spawned: inconclusive, and
            # an inconclusive answer must never reap a bind a container may
            # still be writing to.
            return True
        return code == 0

    async def _promote_stale_staging_dirs(
        self, host_trash_dir: Path, docker_path: Optional[str]
    ) -> None:
        """Promote and remove staging directories no execution still owns.

        A failed launch used to leave its ``.staging-*`` directory behind
        (#3117), and the trash listing hides dot-directories by design, so
        nothing in band ever showed the accumulation. Each script run sweeps
        the root before staging its own. "Still owned" is a fact about a
        process and its container, not an age: a directory whose record
        names a running process is left alone however old it is; one whose
        process is gone is left alone while Docker still knows its container
        (the container outlives a killed client), and otherwise promoted
        (its entries, if any, into the real trash root) and removed at once,
        record included. A directory with no record is legacy and is swept
        once older than the executor's legacy floor. A record whose
        directory is gone is an orphan and is removed the same way. Nothing
        here follows a symlink or raises: the staging bind is the one
        writable mount a container gets, and the sweep runs at the head of
        every script execution, so a planted file must never turn every
        later run into a failure.
        """
        try:
            await self._sweep_staging_dirs(host_trash_dir, docker_path)
        except Exception:  # noqa: BLE001 - the sweep is best-effort by contract
            logger.warning(
                "Sweep of stale staging directories under %s failed; continuing",
                host_trash_dir,
                exc_info=True,
            )

    async def _sweep_staging_dirs(self, host_trash_dir: Path, docker_path: Optional[str]) -> None:
        try:
            candidates = list(host_trash_dir.iterdir())
        except OSError:
            return
        legacy_cutoff = time.time() - self._legacy_staging_age_seconds
        suffix = self.STAGING_OWNER_SUFFIX
        inspect_deadline = time.monotonic() + self.SWEEP_INSPECT_BUDGET_SECONDS
        for candidate in candidates:
            name = candidate.name
            if not name.startswith(".staging-"):
                continue
            try:
                st = candidate.lstat()
            except OSError:
                continue
            if name.endswith(suffix):
                # An orphan record: its directory is gone (a failed mkdir, or
                # a promotion that did not get to remove it). Reap it once
                # its owner is, or it is unreadable and past the floor.
                directory = candidate.with_name(name[: -len(suffix)])
                if directory.exists() or directory.is_symlink():
                    continue
                owner = self._read_owner_record(candidate)
                if (
                    owner is not None
                    and not self._owner_record_expired(owner)
                    and self._pid_is_alive(owner["pid"])
                ):
                    continue
                if owner is None and st.st_mtime > legacy_cutoff:
                    continue
                self._remove_staging_owner(directory)
                continue
            if not stat.S_ISDIR(st.st_mode):
                # A symlink or file wearing a staging name is not a staging
                # directory; it is never entered, promoted or followed.
                continue
            owner = self._read_owner_record(candidate.with_name(name + suffix))
            if owner is not None:
                expired = self._owner_record_expired(owner)
                if not expired and self._pid_is_alive(owner["pid"]):
                    continue
                if owner["container"]:
                    # A pid that is dead, or too old to be trusted, settles
                    # nothing while Docker still knows the container.
                    if time.monotonic() > inspect_deadline:
                        logger.debug(
                            "Sweep inspect budget spent; %s left for the next run", candidate
                        )
                        continue
                    remaining = inspect_deadline - time.monotonic()
                    try:
                        exists = await asyncio.wait_for(
                            self._container_exists(docker_path, owner["container"]),
                            timeout=max(0.5, remaining),
                        )
                    except TimeoutError:
                        exists = True  # inconclusive: leave the directory alone
                    if exists:
                        continue
                if expired:
                    logger.warning(
                        "Staging directory %s is older than any script may run and "
                        "Docker no longer knows its container; its recorded pid %s "
                        "is treated as reused and the directory is reaped.",
                        candidate, owner["pid"],
                    )
            elif st.st_mtime > legacy_cutoff:
                continue
            self._promote_staged_trash(candidate, host_trash_dir)

    def _owner_record_expired(self, owner: Dict[str, object]) -> bool:
        """Whether the record is older than any script may run, so its pid
        is no longer evidence of a live owner (a reused pid answers
        ``kill -0`` too). The bound follows the configured maximum when that
        is longer than a week."""
        started = owner.get("started")
        if not isinstance(started, float):
            return False
        bound = max(self.OWNER_MAX_AGE_SECONDS, 2 * self._legacy_staging_age_seconds)
        return time.time() - started > bound

    #: How many collision suffixes a move tries before giving up.
    MOVE_SUFFIX_LIMIT = 1000

    @classmethod
    def _move_noreplace(cls, entry: Path, into: Path, label: str) -> Path:
        """Move ``entry`` into ``into`` under ``label`` (a suffix is added on
        collision) without ever replacing a concurrently created target.

        Two promoters (two sweeps, or a sweep and a live promotion) may settle
        on the same destination before either renames; a plain rename would
        make the second replace the first and destroy a restorable entry.
        The rename is anchored on directory descriptors and refused when the
        target exists (``_rename_noreplace``), so a collision is retried with
        the next suffix rather than clobbered.
        """
        source_fd = os.open(entry.parent, os.O_RDONLY)
        try:
            dest_fd = os.open(into, os.O_RDONLY)
            try:
                expected = os.stat(entry.name, dir_fd=source_fd, follow_symlinks=False)
                for suffix in range(cls.MOVE_SUFFIX_LIMIT):
                    name = label if suffix == 0 else f"{label}.{suffix}"
                    try:
                        _rename_noreplace(
                            entry.name, name,
                            source_dir_fd=source_fd, destination_dir_fd=dest_fd,
                            expected_source_stat=expected,
                        )
                    except FileExistsError:
                        continue
                    return into / name
                raise OSError(f"no free name for {entry.name} under {into}")
            finally:
                os.close(dest_fd)
        finally:
            os.close(source_fd)

    #: Where container-made hidden entries go. Not a ``.staging-`` name, so
    #: the sweep never treats it or its contents as staging directories or
    #: owner records; hidden, so the trash listing never shows it.
    QUARANTINE_DIR_NAME = ".quarantine"

    @classmethod
    def _quarantine(cls, entry: Path, staging_dir: Path, host_trash_dir: Path) -> Path:
        """Move ``entry`` into the quarantine; the name records which staging
        directory it came from (the directory itself when moved whole)."""
        quarantine = host_trash_dir / cls.QUARANTINE_DIR_NAME
        quarantine.mkdir(mode=0o700, exist_ok=True)
        label = staging_dir.name if entry == staging_dir else f"{staging_dir.name}-{entry.name}"
        destination = cls._move_noreplace(entry, quarantine, label)
        try:
            held = sum(1 for _ in quarantine.iterdir())
        except OSError:
            held = -1
        logger.warning(
            "Quarantine %s now holds %s entries; container-made entries the host "
            "user cannot delete need an operator's attention.",
            quarantine, held,
        )
        return destination

    @classmethod
    def _remove_staging_owner(cls, staging_dir: Path) -> None:
        record = staging_dir.with_name(staging_dir.name + cls.STAGING_OWNER_SUFFIX)
        try:
            record.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("Could not remove staging owner record %s", record, exc_info=True)

    @classmethod
    def _promote_staged_trash(cls, staging_dir: Path, host_trash_dir: Path) -> None:
        """Move per-execution staged trash entries into the real trash root.

        Renames each staged entry (same filesystem, atomic) with a collision
        suffix, then removes the staging directory and its owner record.
        Best-effort: a promotion failure must not mask the execution result,
        but it is logged loudly because it strands restorable trash entries
        in a hidden directory. Two exceptions are quiet: a staging directory
        that is not a real directory (a symlink planted by the container is
        skipped, never followed, and reported), and entries that vanished
        because another process promoted the same stale directory first
        (the entries are then already in the root).
        """
        try:
            try:
                st = staging_dir.lstat()
            except FileNotFoundError:
                cls._remove_staging_owner(staging_dir)
                return
            if not stat.S_ISDIR(st.st_mode):
                logger.warning(
                    "Refusing to promote %s: not a directory (a symlink or file "
                    "wearing a staging name); left in place.",
                    staging_dir,
                )
                return
            try:
                entries = list(staging_dir.iterdir())
            except FileNotFoundError:
                # Another sweep promoted and removed this directory between
                # our lstat and our listing; its entries are in the root.
                logger.debug("Staging directory %s already promoted by another process", staging_dir)
                cls._remove_staging_owner(staging_dir)
                return
            for entry in entries:
                if entry.is_symlink() or entry.name.startswith("."):
                    # Never promote a link or a hidden entry. A link's target
                    # is whatever the container chose; a hidden name is one
                    # no rewriter-made trash entry ever has, and in the root
                    # it would pass for a staging directory or an owner
                    # record the sweep trusts (review of #3117). A link is
                    # unlinked; anything else is moved aside into the
                    # quarantine rather than deleted, because a container
                    # can make a directory the host user cannot remove
                    # (root-owned, or mode 0500) and a directory that will
                    # not empty would keep this one, and its record, in the
                    # root forever. A rename needs only this directory.
                    if entry.is_symlink():
                        logger.warning(
                            "Refusing to promote symlink %s staged by a container; removed.",
                            entry,
                        )
                        entry.unlink()
                        continue
                    try:
                        quarantined = cls._quarantine(entry, staging_dir, host_trash_dir)
                    except OSError:
                        # Moving a directory rewrites its own '..' entry, so
                        # one the host user cannot write (mode 0500, or
                        # root-owned) will not move; the whole staging
                        # directory, which is ours, is moved aside below.
                        logger.warning(
                            "Refusing to promote hidden entry %s staged by a container; "
                            "it cannot be moved on its own.",
                            entry,
                        )
                        continue
                    logger.warning(
                        "Refusing to promote hidden entry %s staged by a container; "
                        "moved to %s.",
                        entry, quarantined,
                    )
                    continue
                try:
                    cls._move_noreplace(entry, host_trash_dir, entry.name)
                except FileNotFoundError:
                    # Another sweep promoted this stale directory first.
                    logger.debug(
                        "Staged entry %s already promoted by another process", entry
                    )
                except OSError as move_error:
                    # An entry the host user cannot move (a root-owned
                    # directory from a root container: moving it rewrites
                    # its own '..'). Keep promoting the rest; the ENOTEMPTY
                    # fallback below moves this directory aside whole.
                    logger.warning(
                        "Staged entry %s could not be promoted (%s: %s); the "
                        "staging directory will be moved aside whole.",
                        entry, type(move_error).__name__, move_error,
                    )
            try:
                staging_dir.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                # Something a container left could not be moved on its own.
                # The staging directory itself is ours: move the whole of it
                # aside so neither it nor its record stays in the root.
                aside = cls._quarantine(staging_dir, staging_dir, host_trash_dir)
                logger.warning(
                    "Staging directory %s could not be emptied of container-made "
                    "entries; moved whole to %s.",
                    staging_dir, aside,
                )
            cls._remove_staging_owner(staging_dir)
        except OSError:
            logger.warning(
                "Failed to promote staged trash entries from %s into %s; "
                "trashed files remain there and are NOT visible to trash "
                "restore/listing until moved manually.",
                staging_dir,
                host_trash_dir,
                exc_info=True,
            )

    @staticmethod
    def _container_name(execution_id: str) -> str:
        return f"kestrel_compute_{execution_id[:8]}"

    @staticmethod
    async def _kill_container(docker_path: str, container_name: str) -> None:
        returncode = await DockerExecutor._run_control_command(
            docker_path,
            "kill",
            container_name,
        )
        if returncode not in (None, 0):
            logger.debug(
                "Docker kill returned exit code %s for container %s",
                returncode,
                container_name,
            )

    @staticmethod
    async def _remove_container(docker_path: str, container_name: str) -> None:
        returncode = await DockerExecutor._run_control_command(
            docker_path,
            "rm",
            "-f",
            container_name,
        )
        if returncode not in (None, 0):
            logger.debug(
                "Docker rm returned exit code %s for container %s",
                returncode,
                container_name,
            )

    @staticmethod
    async def _run_control_command(
        docker_path: str,
        action: str,
        *arguments: str,
    ) -> Optional[int]:
        """Run one bounded Docker lifecycle command.

        The Docker CLI is a child process too: a wedged daemon must not leave
        its client waiting forever, and cancellation must reap that client
        before propagating.
        """
        process: Optional[asyncio.subprocess.Process] = None
        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    docker_path,
                    action,
                    *arguments,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                ),
                timeout=SUBPROCESS_TIMEOUT_SHORT,
            )
            try:
                return await asyncio.wait_for(
                    process.wait(),
                    timeout=SUBPROCESS_TIMEOUT_SHORT,
                )
            except TimeoutError:
                await DockerExecutor._stop_control_process(process)
                logger.warning(
                    "Timed out running Docker %s %s",
                    action,
                    " ".join(arguments),
                )
                return None
        except asyncio.CancelledError:
            if process is not None:
                stop_task = asyncio.create_task(
                    DockerExecutor._stop_control_process(process)
                )
                _, cancellation = await BaseExecutor._await_owned_task(stop_task)
                if cancellation is not None:
                    raise cancellation
            raise
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
        ) as error:
            logger.debug(
                "Failed to run Docker %s %s: %s",
                action,
                " ".join(arguments),
                error,
            )
            return None

    @staticmethod
    async def _stop_control_process(process: asyncio.subprocess.Process) -> None:
        try:
            process.kill()
        except (ProcessLookupError, OSError) as error:
            logger.debug("Failed to kill Docker control process: %s", error)

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=_DOCKER_CONTROL_REAP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning("Docker control process did not exit after kill")

    async def pull_image(self, language: str) -> bool:
        """
        Pull the Docker image for a language.

        Args:
            language: "bash" or "python"

        Returns:
            True if pull succeeded, False otherwise
        """
        docker_path = self._get_docker_path()
        if not docker_path:
            return False

        image = self._images.get(language)
        if not image:
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                docker_path,
                "pull",
                image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
            return process.returncode == 0
        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
            asyncio.CancelledError,
        ) as e:
            logger.error(f"Failed to pull image {image}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up any orphaned containers."""
        docker_path = self._get_docker_path()
        if not docker_path:
            return

        try:
            # List containers with kestrel_compute prefix
            process = await asyncio.create_subprocess_exec(
                docker_path,
                "ps",
                "-a",
                "-q",
                "--filter",
                "name=kestrel_compute_",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()

            container_ids = stdout.decode().strip().split("\n")
            container_ids = [c for c in container_ids if c]

            if container_ids:
                rm_process = await asyncio.create_subprocess_exec(
                    docker_path,
                    "rm",
                    "-f",
                    *container_ids,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await rm_process.wait()
                logger.info(f"Cleaned up {len(container_ids)} orphaned containers")

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            OSError,
            asyncio.CancelledError,
        ) as e:
            logger.warning(f"Container cleanup failed: {e}")
        except Exception as e:
            logger.warning(f"Container cleanup failed: {e}", exc_info=True)
