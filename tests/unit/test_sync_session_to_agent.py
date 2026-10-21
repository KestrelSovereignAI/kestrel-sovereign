"""scripts/sync_session_to_agent.py imports both sides of a Claude Code session.

Its first version matched user turns on ``type == "human"``. Claude Code
writes ``"user"``, so the 2026-04-21 import of Meridian's birth session stored
every assistant turn and none of the user's, and it reported success. These
tests feed it the entry shapes Claude Code actually writes.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
spec = importlib.util.spec_from_file_location("sync_session_to_agent", SCRIPTS / "sync_session_to_agent.py")
assert spec and spec.loader
sync_session_to_agent = importlib.util.module_from_spec(spec)
sys.modules["sync_session_to_agent"] = sync_session_to_agent
spec.loader.exec_module(sync_session_to_agent)


def _entry(entry_type: str, content, ts: str, **flags) -> dict:
    role = "user" if entry_type == "api-request-blob" else entry_type
    return {"type": entry_type, "timestamp": ts, "message": {"role": role, "content": content}, **flags}


# One session in the shapes found in real transcripts.
CLAUDE_CODE_SESSION = [
    _entry("user", "What should kestrel-claws be?", "2026-03-29T15:16:00.000Z"),
    _entry("assistant", [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "Let me map the fleet."}],
           "2026-03-29T15:16:30.000Z"),
    _entry("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}], "2026-03-29T15:16:31.000Z"),
    _entry("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}], "2026-03-29T15:16:32.000Z"),
    _entry("user", "<system-reminder>injected</system-reminder>", "2026-03-29T15:16:33.000Z", isMeta=True),
    _entry("user", "Summary of the earlier conversation", "2026-03-29T15:16:34.000Z",
           isCompactSummary=True, isVisibleInTranscriptOnly=True),
    _entry("api-request-blob", "What should kestrel-claws be?", "2026-03-29T15:16:35.000Z"),
    {"type": "attachment", "timestamp": "2026-03-29T15:16:36.000Z"},
    _entry("user", [{"type": "image", "source": {}}, {"type": "text", "text": "Name yourself."}],
           "2026-03-29T15:17:00.000Z"),
    _entry("assistant", [{"type": "text", "text": "Meridian."}], "2026-03-29T15:17:30.000Z"),
]


def _write_jsonl(path: Path, entries: list[dict]) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return path


def _agent_db(path: Path) -> Path:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)")
    db.execute("INSERT INTO graph_nodes VALUES ('did:example:meridian', 'agent')")
    db.execute(
        "CREATE TABLE conversation_history (id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT, "
        "role TEXT, content TEXT, metadata TEXT, created_at TIMESTAMP)"
    )
    db.commit()
    db.close()
    return path


def test_extracts_the_users_turns_as_well_as_the_assistants(tmp_path):
    transcript = _write_jsonl(tmp_path / "session.jsonl", CLAUDE_CODE_SESSION)

    messages = sync_session_to_agent.extract_messages(str(transcript))

    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "What should kestrel-claws be?"),
        ("assistant", "Let me map the fleet."),
        ("user", "Name yourself."),
        ("assistant", "Meridian."),
    ]


def test_sync_stores_both_sides(tmp_path):
    transcript = _write_jsonl(tmp_path / "session.jsonl", CLAUDE_CODE_SESSION)
    db_path = _agent_db(tmp_path / "kestrel_prime.db")

    result = sync_session_to_agent.sync_to_agent(
        sync_session_to_agent.extract_messages(str(transcript)), str(db_path)
    )

    assert result["inserted"] == 4
    db = sqlite3.connect(db_path)
    roles = [r for (r,) in db.execute("SELECT role FROM conversation_history ORDER BY id")]
    db.close()
    assert roles == ["user", "assistant", "user", "assistant"]


@pytest.mark.parametrize("kept", ["assistant", "user"])
def test_one_sided_transcript_is_refused_before_anything_is_written(tmp_path, kept):
    one_side = [e for e in CLAUDE_CODE_SESSION if e["type"] == kept]
    transcript = _write_jsonl(tmp_path / "session.jsonl", one_side)
    db_path = _agent_db(tmp_path / "kestrel_prime.db")
    messages = sync_session_to_agent.extract_messages(str(transcript))
    assert messages, "the fixture must still yield turns from the side it kept"

    with pytest.raises(sync_session_to_agent.OneSidedTranscriptError):
        sync_session_to_agent.sync_to_agent(messages, str(db_path))

    db = sqlite3.connect(db_path)
    assert db.execute("SELECT COUNT(*) FROM conversation_history").fetchone()[0] == 0
    db.close()
