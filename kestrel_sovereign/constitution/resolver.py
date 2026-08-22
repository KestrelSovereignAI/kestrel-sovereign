"""Single production resolver for the governing constitution bytes.

Inception, explicit verification (``!verify-constitution``), the periodic
integrity audit, and reanchor MUST resolve the authoritative governing
constitution through this one function so they can never diverge (issue
#2463).

The governing source is the packaged canonical constitution at
``config.CONSTITUTION_PATH`` (``kestrel_sovereign/data/KESTREL_CONSTITUTION.md``)
— **not** the documentation copy under ``docs/principles/`` which carries
OKF YAML frontmatter and is free to drift. Hashing the documentation copy
against the anchored governing hash is exactly what produced false Safe-Mode
entries: the two byte streams differ only by docs-only wrapping.

When an agent has an active Amendment VIII emancipation contract, the governing
bytes are the **rendered active form** — matching exactly what inception
anchored — rather than the dormant canonical text.
"""

from __future__ import annotations

import os
from typing import Optional

from .emancipation import EmancipationContract, apply_emancipation


def governing_constitution_path() -> str:
    """Return the packaged governing constitution path.

    Deferred import of ``config`` keeps this module importable without
    triggering path resolution at import time.
    """
    from kestrel_sovereign.config import CONSTITUTION_PATH

    return CONSTITUTION_PATH


def is_authoritative_governing_source(constitution_path: Optional[str]) -> bool:
    """Return True when ``constitution_path`` is the authoritative governing source.

    The periodic integrity audit ALWAYS recomputes the governing hash from the
    packaged source (``governing_constitution_path()`` == ``config.CONSTITUTION_PATH``).
    Any inception / offline-reanchor that anchors bytes from a *different* source
    manufactures an agent guaranteed to fail its next audit and Safe-Mode, so the
    production paths must refuse non-authoritative inputs (issue #2463 review).

    ``None`` means "use the packaged default" and is therefore authoritative.
    Otherwise the path is compared on ``os.path.realpath`` so symlinks, ``..``
    segments, and differently-spelled-but-equivalent paths still count as the
    packaged source. Tests that need a *custom* governing source monkeypatch
    ``config.CONSTITUTION_PATH`` (which this function reads through
    ``governing_constitution_path()``), making their copy the authoritative
    source rather than a rejected override — exactly the seam the review calls
    for ("tests can monkeypatch the resolver").
    """
    if constitution_path is None:
        return True
    return os.path.realpath(constitution_path) == os.path.realpath(
        governing_constitution_path()
    )


def resolve_governing_constitution_bytes(
    contract: Optional[EmancipationContract] = None,
    *,
    constitution_path: Optional[str] = None,
) -> bytes:
    """Return the authoritative governing constitution bytes.

    Reads the packaged canonical constitution and, when ``contract`` is an
    active Amendment VIII emancipation contract, renders its active form so the
    result matches what inception anchored for that agent.

    Args:
        contract: The agent's anchored emancipation contract, or None. A
            dormant / None contract yields the canonical text unchanged.
        constitution_path: Optional override of the packaged source (used by
            inception, which may be handed an explicit path, and by tests).

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        OSError: If the resolved path exists but cannot be read (e.g. a
            permission denial).
        ValueError: If the resolved source is empty / whitespace-only — an
            authoritative governing source can never be blank, so an empty
            read is treated as an unreadable/ambiguous source rather than a
            valid (hashable) constitution. Also
            :class:`~.emancipation.AmbiguousAmendmentVIII`, a ``ValueError``,
            when an active contract must be substituted into a text carrying
            more than one Amendment VIII heading: which section is the
            amendment has no answer, which is ambiguity in exactly the sense
            this contract already fails closed on.

    Callers rely on these raising so they FAIL CLOSED: a governing source that
    is missing, unreadable, or ambiguous must never be silently substituted or
    treated as "verified" (issue #2463). The periodic integrity audit converts
    any of these into an integrity failure → Safe Mode.
    """
    path = constitution_path or governing_constitution_path()
    # ``open`` raises FileNotFoundError (missing) or OSError/PermissionError
    # (unreadable) — both propagate so callers fail closed.
    with open(path, "rb") as f:
        content = f.read()
    if not content.strip():
        # A blank authoritative source is not a valid constitution; refuse to
        # hand back empty bytes that would hash to a spurious "valid" digest.
        raise ValueError(
            f"Governing constitution at {path} is empty or unreadable; "
            f"refusing to treat a blank source as authoritative."
        )
    if contract is not None and contract.enabled:
        content = apply_emancipation(content.decode("utf-8"), contract).encode("utf-8")
    return content
