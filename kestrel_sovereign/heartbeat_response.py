"""Canonical classification of heartbeat response bodies."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


HEARTBEAT_OK_PATTERN = re.compile(
    r"(?<![\w])(?:<b>)?(?:\*\*)?HEARTBEAT_OK(?:\*\*)?(?:</b>)?[!.]*(?![\w])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class HeartbeatResponseClassification:
    """Semantic result of interpreting one heartbeat response body."""

    is_all_clear: bool
    alert_text: str | None = None


def classify_heartbeat_response(result_body: Any) -> HeartbeatResponseClassification:
    """Classify a heartbeat response without applying presentation limits.

    Empty responses and a standalone ``HEARTBEAT_OK`` sentinel are routine
    all-clears. Residue containing only whitespace or ASCII punctuation is
    treated as a presentation wrapper and remains all-clear. Any substantive
    content remaining beside the sentinel is an alert, as is any non-empty
    response without the sentinel.
    """
    if result_body is None:
        return HeartbeatResponseClassification(is_all_clear=True)

    text = result_body if isinstance(result_body, str) else str(result_body)
    if not text.strip():
        return HeartbeatResponseClassification(is_all_clear=True)

    if not HEARTBEAT_OK_PATTERN.search(text):
        return HeartbeatResponseClassification(
            is_all_clear=False,
            alert_text=text,
        )

    remainder = HEARTBEAT_OK_PATTERN.sub("", text).strip()
    if remainder and not all(
        character.isspace() or character in string.punctuation
        for character in remainder
    ):
        return HeartbeatResponseClassification(
            is_all_clear=False,
            alert_text=remainder,
        )

    return HeartbeatResponseClassification(is_all_clear=True)
