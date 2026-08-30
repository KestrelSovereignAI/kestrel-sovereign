"""Secret-free environment construction for agent-invoked subprocesses."""

from __future__ import annotations

import os
from collections.abc import Mapping


SAFE_SUBPROCESS_ENV_VARS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TERM",
        "TZ",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    }
)


def sanitized_subprocess_env(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return only non-secret process context from ``source`` or ``os.environ``."""

    values = os.environ if source is None else source
    return {
        key: value
        for key, value in values.items()
        if key in SAFE_SUBPROCESS_ENV_VARS
    }
