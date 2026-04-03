#!/usr/bin/env python3
"""
Sync a Claude Code session transcript into a kestrel agent's memory.

Extracts human/assistant messages from a .jsonl session file,
deduplicates against what's already in the agent's database,
and inserts new messages as [SESSION SYNC] tagged entries.

Usage:
    python scripts/sync_session_to_agent.py <session_jsonl> <agent_db_path>

    # Sync this session into Meridian:
    python scripts/sync_session_to_agent.py \
        ~/.claude/projects/-Volumes-data2-projects-kestrel-sovereign/e837ea2c-71aa-4e3e-be7c-2ab570b8623d.jsonl \
        agent_data/meridian/kestrel_prime.db
"""

import json
import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def extract_messages(jsonl_path: str) -> list[dict]:
    """Extract human/assistant messages from Claude Code JSONL."""
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

            msg_type = entry.get("type", "")

            # Human messages
            if msg_type == "human":
                content = ""
                for block in entry.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
                    elif isinstance(block, str):
                        content += block
                if content.strip():
                    messages.append({
                        "role": "user",
                        "content": content.strip(),
                        "timestamp": entry.get("timestamp", ""),
                    })

            # Assistant messages
            elif msg_type == "assistant":
                content = ""
                for block in entry.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
                if content.strip():
                    messages.append({
                        "role": "assistant",
                        "content": content.strip(),
                        "timestamp": entry.get("timestamp", ""),
                    })

    return messages


def content_hash(text: str) -> str:
    """Short hash for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sync_to_agent(messages: list[dict], db_path: str, agent_id: str = None) -> dict:
    """Insert new messages into agent's conversation_history, deduplicating."""
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

        # Use original timestamp if available, otherwise now
        ts = msg["timestamp"] if msg["timestamp"] else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # Normalize timestamp format
        ts = ts.replace("T", " ").split("+")[0].split(".")[0] if "T" in ts else ts

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
    print(f"Found {len(messages)} human/assistant messages")

    print(f"Syncing to {db_path}...")
    result = sync_to_agent(messages, db_path)
    print(f"Inserted: {result['inserted']}, Skipped (dupes): {result['skipped']}, Total in DB: {result['total']}")


if __name__ == "__main__":
    main()
