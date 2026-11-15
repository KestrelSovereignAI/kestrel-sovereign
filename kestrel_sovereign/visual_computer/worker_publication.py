"""Exclusive publication of complete records in an owned worker directory."""
from pathlib import Path
import os


def publish_record(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    """Publish complete bytes without replacing any existing directory entry.

    The worker has already established ownership of the parent directory.
    A fixed exclusive stage also refuses an interrupted/duplicate publisher;
    there is no retry, overwrite or recovery of an uncertain initialization.
    Readers see the final name only after the original stream has closed.
    """
    if mode not in (0o600, 0o640):
        raise ValueError("fixed_worker_record_mode_required")
    path = Path(path)
    pending = path.with_suffix(".pending")
    if pending == path:
        raise ValueError("distinct_worker_record_stage_required")
    # Do not clean up another publisher's stage if this exclusive open fails.
    stream = open(pending, "xb", opener=lambda p, flags: os.open(p, flags, 0o600))
    failure = None
    try:
        with stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
        # A same-directory hard link publishes the closed inode atomically
        # and refuses an existing file, directory or (even dangling) symlink.
        os.link(pending, path, follow_symlinks=False)
    except BaseException as error:
        failure = error
    try:
        pending.unlink()
    except BaseException as closing:
        if failure is not None:
            from kestrel_sovereign.execution_custody import execution_terminal_error
            raise execution_terminal_error(failure, closing, retain_originals=True)
        raise
    if failure is not None:
        raise failure
