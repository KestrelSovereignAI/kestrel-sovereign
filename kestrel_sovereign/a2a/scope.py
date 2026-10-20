"""Server-selected A2A recipient scope, independent of transport and storage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

_TENANT_SEGMENT = re.compile(r"[A-Za-z0-9._~-]*\Z")


@dataclass(frozen=True)
class NativeRecipientScope:
    """A host-selected recipient and its advertised tenant.

    Construct this from the host registration. A peer's tenant parameter must
    match this scope; it cannot select a different recipient or datastore.
    """

    recipient_id: str
    tenant: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.recipient_id, str) or not self.recipient_id.strip():
            raise ValueError("Native A2A recipient_id must be a concrete identity")
        if (
            not isinstance(self.tenant, str)
            or not _TENANT_SEGMENT.fullmatch(self.tenant)
            or self.tenant in {".", ".."}
        ):
            raise ValueError("Native A2A tenant must be a URL-safe opaque segment")

    def session_id_for_context(self, context_id: str) -> str:
        """Project opaque peer contexts into a separate host-chat namespace."""
        if not isinstance(context_id, str) or not context_id:
            raise ValueError("A native task projection requires a context ID")
        identity = json.dumps(
            [self.recipient_id, self.tenant, context_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return "a2a-context-" + hashlib.sha256(identity).hexdigest()
