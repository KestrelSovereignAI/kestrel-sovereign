"""Agent-scoped Lighthouse records of sovereignty exports (#2874).

A sovereignty export published to Lighthouse is listed in the account under
an upload name and tag that name exactly one agent:

    kestrel_sovereignty__<agent_id>__<YYYYMMDD_HHMMSS>.bin
    tag: kestrel-sovereignty-<agent_id>

These records are the index the Lighthouse health check, the sync target's
retention prune and the backlog cleanup tool read to attribute an export. They
carry an identifier and a time, never content: the payload itself is the
export's sealed envelope.

The family is deliberately distinct from ``kestrel_state__`` snapshots and
``kestrel_manifest__`` manifests. Cold-start restore resolves its database from
those two, and an export is an encrypted archive, not a SQLite file.

Attribution is by exact match, never by prefix: the agent id in the name is
percent-encoded (DIDs are unchanged by it), the timestamp anchors the end of
the name, and a name that does not decode to one canonical agent id, or whose
tag names another agent, belongs to no agent. An upload that belongs to no
agent is never pruned.

An agent's exports are thinned by the agent itself, through the Lighthouse
provider that published them, on the account its PayerPolicy resolves
(``sovereignty_export_deletions``). Nothing else deletes one: the snapshot
target's account need not be the export's, and the account-wide cleanup tools
keep every upload that is, or may be, a sovereignty export
(``is_sovereignty_export_candidate``), attributed or not.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Collection, Iterable, Mapping, Optional
from urllib.parse import quote, unquote

from kestrel_sovereign.storage.sync.retention import (
    DataClass,
    RetentionItem,
    RetentionPolicy,
    parse_timestamp,
)

SOVEREIGNTY_EXPORT_NAME_PREFIX = "kestrel_sovereignty"
SOVEREIGNTY_EXPORT_TAG_PREFIX = "kestrel-sovereignty"
SOVEREIGNTY_EXPORT_SUFFIX = ".bin"
# The name every IPFS-tier upload carried before #2874, sovereignty exports
# included. Such an upload names no agent.
LEGACY_UPLOAD_NAME = "content.bin"

_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_NAME = re.compile(
    rf"^{SOVEREIGNTY_EXPORT_NAME_PREFIX}__(?P<agent>.+)__"
    rf"(?P<timestamp>\d{{8}}_\d{{6}}){re.escape(SOVEREIGNTY_EXPORT_SUFFIX)}$"
)


def _encoded_agent_id(agent_id: str) -> str:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("a sovereignty export's agent id must be a non-empty string")
    return quote(agent_id, safe=":")


def sovereignty_export_upload_name(agent_id: str, uploaded_at: datetime) -> str:
    """The Lighthouse upload name of ``agent_id``'s export made at ``uploaded_at``."""
    timestamp = uploaded_at.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)
    return (
        f"{SOVEREIGNTY_EXPORT_NAME_PREFIX}__{_encoded_agent_id(agent_id)}__"
        f"{timestamp}{SOVEREIGNTY_EXPORT_SUFFIX}"
    )


def sovereignty_export_tag(agent_id: str) -> str:
    """The Lighthouse tag of ``agent_id``'s sovereignty exports."""
    _encoded_agent_id(agent_id)
    return f"{SOVEREIGNTY_EXPORT_TAG_PREFIX}-{agent_id}"


def _upload_name(upload: Mapping[str, Any]) -> str:
    return str(upload.get("fileName") or upload.get("Name") or upload.get("name") or "")


def sovereignty_export_owner(upload: Mapping[str, Any]) -> Optional[str]:
    """The one agent a Lighthouse upload record is a sovereignty export of.

    ``None`` when the record is not one, or cannot be attributed to exactly
    one agent: its name is not in the family, does not decode to a canonical
    agent id, or its tag names another agent.
    """
    match = _NAME.match(_upload_name(upload))
    if match is None:
        return None
    encoded = match.group("agent")
    agent_id = unquote(encoded)
    if not agent_id or _encoded_agent_id(agent_id) != encoded:
        return None
    tag = upload.get("tag")
    if tag and tag != sovereignty_export_tag(agent_id):
        return None
    return agent_id


def sovereignty_export_uploads(
    uploads: Iterable[Mapping[str, Any]], agent_id: str
) -> list[Mapping[str, Any]]:
    """The upload records that are ``agent_id``'s sovereignty exports."""
    return [upload for upload in uploads if sovereignty_export_owner(upload) == agent_id]


def is_sovereignty_export_candidate(upload: Mapping[str, Any]) -> bool:
    """Whether an upload record is, or may be, a sovereignty export.

    True for any name in the export family or any tag in the export tag
    family, whether or not it attributes to one agent, and for the
    agent-less ``content.bin`` name exports carried before #2874. The
    account-wide cleanup tools keep every candidate: a backup they cannot
    attribute is never theirs to delete.
    """
    name = _upload_name(upload)
    tag = str(upload.get("tag") or "")
    return (
        name.startswith(SOVEREIGNTY_EXPORT_NAME_PREFIX)
        or name == LEGACY_UPLOAD_NAME
        or tag == SOVEREIGNTY_EXPORT_TAG_PREFIX
        or tag.startswith(f"{SOVEREIGNTY_EXPORT_TAG_PREFIX}-")
    )


def _upload_cid(upload: Mapping[str, Any]) -> Optional[str]:
    for key in ("cid", "Hash", "hash", "fileHash", "CID"):
        value = upload.get(key)
        if value:
            return str(value)
    return None


def sovereignty_export_deletions(
    uploads: Iterable[Mapping[str, Any]],
    agent_id: str,
    policy: RetentionPolicy,
    *,
    keep_cids: Collection[str] = (),
    now: Optional[datetime] = None,
) -> list[Mapping[str, Any]]:
    """The upload records of ``agent_id``'s exports ``policy`` would delete.

    Only records that attribute to ``agent_id`` are considered, and they are
    thinned among themselves in the identity class, so no other kind of
    upload takes an export's slot. A record with no CID or no time is kept.
    A CID in ``keep_cids`` takes part in the decision, so a just-published
    export is the newest, but is never returned.
    """
    items: list[RetentionItem] = []
    for upload in sovereignty_export_uploads(uploads, agent_id):
        cid = _upload_cid(upload)
        timestamp = parse_timestamp(upload.get("createdAt")) or parse_timestamp(
            _upload_name(upload)
        )
        if cid is None or timestamp is None:
            continue
        items.append(
            RetentionItem(
                key=cid,
                name=_upload_name(upload),
                timestamp=timestamp,
                data_class=DataClass.IDENTITY,
                metadata=upload,
            )
        )
    return [
        item.metadata
        for item in policy.deletions(items, now=now)
        if item.key not in keep_cids
    ]
