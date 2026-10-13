"""Agent-scoped Lighthouse records of sovereignty exports (#2874).

A sovereignty export published to Lighthouse is listed in the account under
an upload name and tag that name exactly one agent:

    kestrel_sovereignty__<agent_id>__<YYYYMMDD_HHMMSS>__<nonce>.bin
    tag: kestrel-sovereignty-<agent_id>

These records are the index the Lighthouse health check, the sync target's
retention prune and the backlog cleanup tool read to attribute an export. They
carry an identifier and a time, never content: the payload itself is the
export's sealed envelope.

The nonce is random and makes each publication's name its own. Lighthouse
answers an upload with its CID alone, not the id of the record it made, so
the name is how the publisher finds that record among others of the same CID
and second (#3602). An export published before #3602 carries no nonce and is
attributed the same way.

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
import secrets
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional
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
# Hex digits of the nonce that makes each publication's name unique (#3602).
_NONCE_LENGTH = 16
# The nonce has no underscore, so it is never read as part of the agent id or
# the timestamp, and a name without one (before #3602) still parses.
_NAME = re.compile(
    rf"^{SOVEREIGNTY_EXPORT_NAME_PREFIX}__(?P<agent>.+)__"
    rf"(?P<timestamp>\d{{8}}_\d{{6}})"
    rf"(?:__(?P<nonce>[0-9a-f]{{{_NONCE_LENGTH}}}))?"
    rf"{re.escape(SOVEREIGNTY_EXPORT_SUFFIX)}$"
)


def _encoded_agent_id(agent_id: str) -> str:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("a sovereignty export's agent id must be a non-empty string")
    return quote(agent_id, safe=":")


def sovereignty_export_upload_name(agent_id: str, uploaded_at: datetime) -> str:
    """A new Lighthouse upload name for ``agent_id``'s export made at ``uploaded_at``.

    Each call returns a name no other publication has: its random nonce sets
    it apart from every other export, even of the same agent, content and
    second (#3602).
    """
    timestamp = uploaded_at.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)
    nonce = secrets.token_hex(_NONCE_LENGTH // 2)
    return (
        f"{SOVEREIGNTY_EXPORT_NAME_PREFIX}__{_encoded_agent_id(agent_id)}__"
        f"{timestamp}__{nonce}{SOVEREIGNTY_EXPORT_SUFFIX}"
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


def _upload_id(upload: Mapping[str, Any]) -> Optional[str]:
    """The upload record's own id: what Lighthouse deletes it by."""
    value = upload.get("id")
    return str(value) if value else None


class AmbiguousPublishedExportError(ValueError):
    """The account does not list the export just published as one record.

    Its upload name is unique to that publication, so either several records
    carry the name or the one that does holds another CID. The publisher
    cannot tell which record it made, so it thins nothing: any record it
    deleted could be the export it just published (#3602).
    """


def _published_record_id(
    exports: Iterable[Mapping[str, Any]], cid: str, name: str
) -> Optional[str]:
    """The id of the record of the export just published as ``name``.

    ``None`` while the account does not list it yet (or lists it with no
    id): a record that is not listed, or has no id, is never deleted.
    """
    named: dict[str, Mapping[str, Any]] = {}
    without_id = 0
    for upload in exports:
        if _upload_name(upload) != name:
            continue
        record_id = _upload_id(upload)
        if record_id is None:
            without_id += 1
        else:
            named.setdefault(record_id, upload)
    if len(named) + without_id > 1:
        raise AmbiguousPublishedExportError(
            f"{len(named) + without_id} upload records are named {name}, the "
            "export just published; none is thinned while it cannot be told "
            "which one is"
        )
    if not named:
        return None
    [(record_id, upload)] = named.items()
    if _upload_cid(upload) != cid:
        raise AmbiguousPublishedExportError(
            f"the upload record named {name}, the export just published, holds "
            f"{_upload_cid(upload)!r}, not its CID {cid!r}; none is thinned"
        )
    return record_id


def sovereignty_export_deletions(
    uploads: Iterable[Mapping[str, Any]],
    agent_id: str,
    policy: RetentionPolicy,
    *,
    published_cid: Optional[str] = None,
    published_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> list[Mapping[str, Any]]:
    """The upload records of ``agent_id``'s exports ``policy`` would delete.

    Only records that attribute to ``agent_id`` are considered, and they are
    thinned among themselves in the identity class, so no other kind of
    upload takes an export's slot. Each record is decided on by its own id,
    never its CID: one CID can be in several records, as when an identical
    export is uploaded again, and each of them holds the account's quota
    (#3602). A record with no id, no CID or no time is kept, and a record
    listed twice is decided on once.

    ``published_cid`` and ``published_name`` are the CID and upload name of
    the export just published. The name is unique to that publication
    (``sovereignty_export_upload_name``), so the one record carrying it is
    the export: it takes part in the decision, where it is the newest unless
    another record is dated later, but is never returned. Every other record,
    of the same CID or not, is decided on like any other. When the account
    lists more than one record under that name, or the record under it holds
    another CID, ``AmbiguousPublishedExportError`` is raised and nothing is
    returned for deletion.
    """
    if (published_cid is None) != (published_name is None):
        raise ValueError(
            "a just-published export is named by its CID and its upload name together"
        )
    exports = sovereignty_export_uploads(uploads, agent_id)
    published_id = (
        _published_record_id(exports, published_cid, published_name)
        if published_name is not None
        else None
    )
    items: list[RetentionItem] = []
    seen: set[str] = set()
    for upload in exports:
        record_id = _upload_id(upload)
        cid = _upload_cid(upload)
        name = _upload_name(upload)
        timestamp = parse_timestamp(upload.get("createdAt")) or parse_timestamp(name)
        if record_id is None or cid is None or timestamp is None or record_id in seen:
            continue
        seen.add(record_id)
        items.append(
            RetentionItem(
                key=record_id,
                name=name,
                timestamp=timestamp,
                data_class=DataClass.IDENTITY,
                metadata=upload,
            )
        )
    # The record just published goes first. ``RetentionPolicy.decide`` keeps
    # the first of equally new records as the newest, so a record of the same
    # time is thinned in its place rather than kept beside it.
    items.sort(key=lambda item: item.key != published_id)
    return [
        item.metadata
        for item in policy.deletions(items, now=now)
        if item.key != published_id
    ]
