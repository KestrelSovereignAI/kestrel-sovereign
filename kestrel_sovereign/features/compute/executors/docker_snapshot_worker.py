"""Killable subprocess boundary for Docker working-directory snapshots.

Filesystem calls against FUSE or network mounts can remain blocked inside the
kernel after an asyncio deadline.  Keeping that work in this one-purpose child
prevents a stalled snapshot from occupying the host event loop's shared thread
executor or delaying interpreter shutdown.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .base import ExecutionEnvironmentError
from .docker_executor import DockerExecutor


def _emit_result(path: Path, kind: str, message: str = "") -> None:
    """Publish one small result inside the parent's private execution root."""

    payload = (json.dumps({"kind": kind, "message": message}) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as result:
        result.write(payload)


def main(argv: list[str] | None = None) -> int:
    """Run one descriptor-anchored snapshot request supplied by its parent."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 6:
        return 2

    (
        source,
        destination,
        raw_max_bytes,
        raw_max_entries,
        raw_deadline,
        raw_result_path,
    ) = arguments
    result_path = Path(raw_result_path)
    try:
        max_bytes = int(raw_max_bytes)
        max_entries = int(raw_max_entries)
        deadline = float(raw_deadline)
    except ValueError:
        _emit_result(result_path, "protocol", "invalid Docker snapshot worker limits")
        return 2

    try:
        DockerExecutor._snapshot_working_directory(
            source,
            Path(destination),
            max_bytes=max_bytes,
            max_entries=max_entries,
            deadline=deadline,
        )
    except TimeoutError as exc:
        _emit_result(result_path, "timeout", str(exc))
        return 1
    except ExecutionEnvironmentError as exc:
        _emit_result(result_path, "environment", str(exc))
        return 1
    except Exception as exc:  # noqa: BLE001 - child reports a bounded typed failure
        _emit_result(
            result_path,
            "worker",
            f"Docker working directory snapshot worker failed: {exc}",
        )
        return 1
    _emit_result(result_path, "success")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
