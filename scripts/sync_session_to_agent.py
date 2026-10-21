#!/usr/bin/env python3
"""
Sync a Claude Code session transcript into a kestrel agent's memory.

Extracts the user's and the assistant's text turns from a .jsonl session
file, deduplicates against what's already in the agent's database,
and inserts new messages as [SESSION SYNC] tagged entries. A transcript
that yields turns from only one side is refused before anything is written.

Usage:
    python scripts/sync_session_to_agent.py <session_jsonl> <agent_db_path>

    # Sync a session into Meridian:
    python scripts/sync_session_to_agent.py \
        ~/.claude/projects/-Volumes-data2-projects-kestrel-sovereign/<session-id>.jsonl \
        agent_data/meridian/kestrel_prime.db
"""

import json
import hashlib
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from kestrel_sovereign.storage.conversation_created_at import canonical_created_at

# The entry types that are conversation turns. Each one is also the role the
# turn is stored under. Every other entry type (attachment, system,
# api-request-blob, ...) is harness bookkeeping. api-request-blob entries also
# carry a ``message.role``, but they repeat turns already recorded, so the role
# is read from the entry's ``type`` and never from ``message.role``.
TURN_TYPES = ("user", "assistant")

# Most "user" entries were not typed by a person: task notifications, peer
# messages, injected context, compaction summaries, interrupt markers, slash
# command echoes. Claude Code marks the typed ones with ``origin.kind ==
# "human"``, so a user turn is admitted on that mark alone, never by ruling the
# others out one by one.
HUMAN_ORIGIN = "human"

# The model Claude Code records on assistant entries it wrote itself (API error
# notices and the like) rather than receiving from a model.
HARNESS_MODEL = "<synthetic>"


class OneSidedTranscriptError(ValueError):
    """The transcript yielded turns from only one side of the conversation."""


def _text_of(content) -> str:
    """The text of a message: a bare string, or the concatenation of its text blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _is_turn(entry: dict, message: dict) -> bool:
    """Whether a "user" entry was typed by a person, or an "assistant" entry came from the model."""
    if entry.get("type") == "user":
        origin = entry.get("origin")
        return isinstance(origin, dict) and origin.get("kind") == HUMAN_ORIGIN
    return message.get("model") != HARNESS_MODEL


def extract_messages(jsonl_path: str) -> list[dict]:
    """Extract the user's and the assistant's text turns from a Claude Code JSONL.

    Tool calls and tool results carry no text block, so they drop out here.
    """
    messages = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = entry.get("type")
            message = entry.get("message")
            if role not in TURN_TYPES or not isinstance(message, dict) or not _is_turn(entry, message):
                continue
            content = _text_of(message.get("content")).strip()
            if content:
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": entry.get("timestamp", ""),
                })

    return messages


def require_both_sides(messages: list[dict]) -> None:
    """Refuse a transcript with no turns from one of its two sides.

    A conversation always has both, so a missing side means the extractor
    misread the transcript (or the transcript predates the ``origin`` mark).
    Importing anyway produced the Apr 2026 birth-session import: every one of
    the user's turns silently missing, reported as success.
    """
    counts = Counter(m["role"] for m in messages)
    missing = [role for role in TURN_TYPES if not counts[role]]
    if missing:
        raise OneSidedTranscriptError(
            f"transcript yielded no {' or '.join(missing)} turns "
            f"(user={counts['user']}, assistant={counts['assistant']}); "
            "refusing to import a one-sided conversation"
        )


def content_hash(text: str) -> str:
    """Short hash for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sync_to_agent(messages: list[dict], db_path: str, agent_id: str = None) -> dict:
    """Insert new messages into agent's conversation_history, deduplicating."""
    require_both_sides(messages)
    db = sqlite3.connect(db_path)

    # Get agent_id if not provided
    if not agent_id:
        row = db.execute(
            "SELECT node_id FROM graph_nodes WHERE node_type = 'agent' LIMIT 1"
        ).fetchone()
        if not row:
            print("ERROR: No agent found in database")
            sys.exit(1)
        agent_id = row[0]

    # Get existing session sync hashes to deduplicate
    existing = set()
    rows = db.execute(
        "SELECT metadata FROM conversation_history WHERE metadata LIKE '%session_sync%'"
    ).fetchall()
    for row in rows:
        if row[0]:
            try:
                meta = json.loads(row[0])
                h = meta.get("content_hash")
                if h:
                    existing.add(h)
            except json.JSONDecodeError:
                pass

    inserted = 0
    skipped = 0

    for msg in messages:
        h = content_hash(msg["content"])
        if h in existing:
            skipped += 1
            continue

        # Tag as session sync with hash for dedup
        metadata = json.dumps({
            "source": "session_sync",
            "content_hash": h,
            "original_timestamp": msg["timestamp"],
            "synced_at": datetime.now(timezone.utc).isoformat(),
        })

        # Prefix content so the agent knows this is inherited memory
        prefix = "[SESSION SYNC — birth session transcript]\n\n"
        tagged_content = prefix + msg["content"]

        # Use original timestamp if available, otherwise now. Spelled by the
        # module the column's CHECK is computed from (#3009) rather than by
        # hand here: the hand-rolled version only normalized values containing
        # a "T", so a space-separated value carrying an offset was written
        # verbatim — which the CHECK now refuses — and it TRUNCATED the offset
        # rather than applying it, dating "03:04:05+01:00" an hour late.
        ts = canonical_created_at(msg["timestamp"]) or canonical_created_at(
            datetime.now(timezone.utc)
        )

        # No ``session_id`` (#2958). The column is derived from
        # ``metadata.session_id`` and the metadata built above never carries
        # one, so the correct value here is the column's NULL default —
        # stamping it would be a call that can only ever return None. If this
        # script ever starts filing synced turns under a session, that changes:
        # write ``column_session_id(metadata)`` rather than the id directly.
        db.execute(
            "INSERT INTO conversation_history (agent_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (agent_id, msg["role"], tagged_content, metadata, ts),
        )
        existing.add(h)
        inserted += 1

    db.commit()

    total = db.execute("SELECT COUNT(*) FROM conversation_history").fetchone()[0]
    db.close()

    return {"inserted": inserted, "skipped": skipped, "total": total}


def main():
    if len(sys.argv) < 3:
        print("Usage: python sync_session_to_agent.py <session.jsonl> <agent_db_path>")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    db_path = sys.argv[2]

    if not Path(jsonl_path).exists():
        print(f"Session file not found: {jsonl_path}")
        sys.exit(1)
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Extracting messages from {jsonl_path}...")
    messages = extract_messages(jsonl_path)
    counts = Counter(m["role"] for m in messages)
    print(f"Found {counts['user']} user and {counts['assistant']} assistant messages")

    print(f"Syncing to {db_path}...")
    try:
        result = sync_to_agent(messages, db_path)
    except OneSidedTranscriptError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Inserted: {result['inserted']}, Skipped (dupes): {result['skipped']}, Total in DB: {result['total']}")


if __name__ == "__main__":
    main()
