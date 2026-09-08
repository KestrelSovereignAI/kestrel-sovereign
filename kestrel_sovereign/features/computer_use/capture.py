"""Durable artifacts for shell runs whose output must outlive the result.

The governed shell surface has no shell (#3129/#3130): ``shlex`` tokenizes
the command and the argv vector is executed directly, so ``> review.txt`` is
a literal argument, not a redirect, and ``policy.py`` refuses the command
rather than run one the caller did not write. That is the right call for the
grammar and the wrong outcome for a long-running review, because the only
way back was the tool result — capped at 1 MiB, with the clip announced in a
field nobody read (#3243).

An 80-minute adversarial review clipped mid-argument still ends in a
paragraph that reads like a verdict. That is the same failure class as a
review tool exiting 0 without reviewing: **a thing shaped like an answer,
produced by a process that did not finish answering.**

So the runtime performs the redirect the caller cannot express. Three files
per run, under a runtime-owned directory:

    <capture_dir>/<run_id>.stdout
    <capture_dir>/<run_id>.stderr
    <capture_dir>/<run_id>.json      the manifest

The paths are chosen here, never by the agent. A ``capture_to`` parameter
would have been a general write primitive reachable through the shell gate
instead of the filesystem-write gate — a way to write any path by naming it
as somewhere to put output. Allocating the path removes that question
rather than answering it.

The manifest is the half that makes a verdict re-checkable rather than
merely re-readable. It records what ran, where, how it ended, and — when the
directory is a git worktree — the ``HEAD`` before and after. A review's
verdict is about a specific tree, and during the 2026-08-31 run the head
moved three times in eight hours. ``head_moved`` is how a later reader can
tell that a verdict was about a tree that no longer exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# How much of a captured stream is echoed back inline. This is a *preview* of
# a complete artifact, which is categorically different from a truncated
# result: nothing was lost, so it must never set ``truncated_stdout``.
PREVIEW_CHARS = 4000

# Reading HEAD is the runtime describing its own work, not the agent running
# a command, so the argv is fixed and the window is short. A repository that
# does not answer in this long simply has no SHA recorded.
_GIT_HEAD_TIMEOUT = 5


@dataclass(frozen=True)
class CaptureBundle:
    """The three paths one captured run owns."""

    run_id: str
    stdout_path: Path
    stderr_path: Path
    manifest_path: Path


def allocate(capture_dir: Path | str, *, run_id: Optional[str] = None) -> CaptureBundle:
    """Reserve the paths for one run. Creates the directory, not the files."""
    rid = run_id or uuid.uuid4().hex
    base = Path(capture_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return CaptureBundle(
        run_id=rid,
        stdout_path=base / f"{rid}.stdout",
        stderr_path=base / f"{rid}.stderr",
        manifest_path=base / f"{rid}.json",
    )


async def git_head(cwd: Optional[Path]) -> Optional[str]:
    """Resolve ``cwd``'s git HEAD, or ``None`` when there isn't one.

    Every failure — not a repository, git absent, timeout, non-zero exit —
    returns ``None``. A manifest that omits the SHA says "unknown", which a
    reader can act on; a manifest carrying a *wrong* SHA would be worse than
    one carrying none, so nothing is guessed here.
    """
    if cwd is None:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(cwd),
            "rev-parse",
            "HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (OSError, FileNotFoundError):
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_HEAD_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None
    if proc.returncode != 0:
        return None
    sha = out.decode("utf-8", errors="replace").strip()
    return sha or None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def build_manifest(
    *,
    bundle: CaptureBundle,
    argv: list[str],
    cwd: Optional[Path],
    backend: str,
    started_at: datetime,
    finished_at: datetime,
    duration_ms: int,
    returncode: int,
    timed_out: bool,
    truncated_stdout: bool,
    truncated_stderr: bool,
    head_before: Optional[str],
    head_after: Optional[str],
) -> dict[str, Any]:
    """Assemble the manifest body.

    ``complete`` is the single field a gate should read. It is the
    conjunction of everything that would make this artifact less than the
    whole run, so a caller cannot satisfy the gate by checking the one
    condition they remembered.
    """
    complete = not (timed_out or truncated_stdout or truncated_stderr)
    return {
        "run_id": bundle.run_id,
        "argv": list(argv),
        "cwd": str(cwd) if cwd else None,
        "backend": backend,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": duration_ms,
        "returncode": returncode,
        "timed_out": timed_out,
        "truncated_stdout": truncated_stdout,
        "truncated_stderr": truncated_stderr,
        "complete": complete,
        "stdout_path": str(bundle.stdout_path),
        "stderr_path": str(bundle.stderr_path),
        "stdout_bytes": _file_size(bundle.stdout_path),
        "stderr_bytes": _file_size(bundle.stderr_path),
        "git": {
            "head_before": head_before,
            "head_after": head_after,
            # Only a claim when both ends are known. Two unknowns are not
            # evidence that nothing moved.
            "head_moved": (
                None
                if head_before is None or head_after is None
                else head_before != head_after
            ),
        },
    }


async def write_manifest(bundle: CaptureBundle, body: dict[str, Any]) -> None:
    """Write the manifest with ``fsync``, matching the audit log's durability."""

    def _write() -> None:
        line = json.dumps(body, indent=2, sort_keys=True) + "\n"
        fd = os.open(bundle.manifest_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    await asyncio.to_thread(_write)


def preview(path: Path, *, max_chars: int = PREVIEW_CHARS) -> str:
    """Echo a bounded window of a captured file.

    Head and tail, not head alone: a review states its verdict at the end,
    and a head-only window is the exact shape that made a clipped review
    look like a finished one. The elision is labelled with the byte count so
    the window is never mistaken for the file.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return f"[capture unreadable: {exc}]"
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    elided = len(text) - (half * 2)
    return f"{text[:half]}\n... [{elided} chars elided; full output in {path}] ...\n{text[-half:]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
