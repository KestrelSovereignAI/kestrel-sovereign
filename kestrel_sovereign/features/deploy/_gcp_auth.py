"""Shared GCP credential discovery for the deploy package.

`CloudRunProvider` and `kestrel deploy secrets sync` both need to
authenticate to Google APIs the same way: project-local credentials,
inline JSON, ``GOOGLE_APPLICATION_CREDENTIALS``, or ADC fallback —
in that order. Extracted from `CloudRunProvider._setup_auth` so the
secrets-sync path inherits the same chain (codex review on PR #1057).

Public surface:
    setup_gcp_auth() -> Optional[str]
        Sets ``GOOGLE_APPLICATION_CREDENTIALS`` in ``os.environ`` if a
        usable credential source is found. Returns the path of any
        temp file it created (caller may want to register cleanup),
        or None.
"""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _project_local_creds_path() -> Path:
    """Return the project-local service account file path.

    Resolves to ``<project_root>/credentials/kestrel-agent-admin.json``
    where ``<project_root>`` is the kestrel-sovereign repo root. The
    repo layout is ``kestrel_sovereign/features/deploy/_gcp_auth.py``
    so four ``parent`` hops land on the project root.
    """
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent.parent
    return project_root / "credentials" / "kestrel-agent-admin.json"


def setup_gcp_auth() -> Optional[str]:
    """Discover GCP credentials and prime ``GOOGLE_APPLICATION_CREDENTIALS``.

    Priority chain:

    1. ``GOOGLE_APPLICATION_CREDENTIALS`` already set + file exists →
       leave it alone.
    2. ``GCP_SERVICE_ACCOUNT_KEY`` (inline JSON) → write to a temp file
       and point ``GOOGLE_APPLICATION_CREDENTIALS`` at it. The temp
       file is registered with ``atexit`` for process-exit cleanup.
    3. ``<project_root>/credentials/kestrel-agent-admin.json`` →
       point ``GOOGLE_APPLICATION_CREDENTIALS`` at it.
    4. Application Default Credentials (no env mutation; warn).

    Returns:
        Path of the temp file written from ``GCP_SERVICE_ACCOUNT_KEY``
        (so callers can clean it up early via :func:`os.unlink`), or
        ``None`` if no temp file was created.
    """
    creds_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_file and os.path.exists(creds_file):
        logger.info(
            f"Using credentials from GOOGLE_APPLICATION_CREDENTIALS: {creds_file}"
        )
        return None

    key_json = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
    if key_json:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(key_json)
            temp_path = f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        logger.info("Using service account key from GCP_SERVICE_ACCOUNT_KEY")
        atexit.register(_cleanup_temp_creds, temp_path)
        return temp_path

    project_creds = _project_local_creds_path()
    if project_creds.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_creds)
        logger.info(f"Using project service account: {project_creds}")
        return None

    logger.warning(
        "No explicit credentials found. Using Application Default Credentials. "
        "Set GOOGLE_APPLICATION_CREDENTIALS or place "
        "credentials/kestrel-agent-admin.json"
    )
    return None


def _cleanup_temp_creds(path: str) -> None:
    """Remove a temp credential file created from
    ``GCP_SERVICE_ACCOUNT_KEY``. Safe to call multiple times."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
            logger.debug(f"Cleaned up temp credentials: {path}")
        except OSError:
            pass
