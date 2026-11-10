"""The DDL each native A2A schema component creates, pinned against its revision.

``TaskManager.initialize`` (and the push outbox) skip a component's store
initializers whenever the database already records that component's revision.
A table, column, index or trigger added to those initializers without a new
revision therefore reaches only databases born on the new code. #3995 did
exactly that, and no existing agent database received the columns (#4002).

:func:`capture` initializes each component alone on fresh SQLite and records
what it created: every table's columns and DDL, and every index and trigger.
``tests/unit/test_a2a_native_core_schema_upgrade.py`` compares that with the
committed ``tests/fixtures/native_store_schema.json`` and refuses a change in
a component whose revision did not move. PostgreSQL-only DDL (functions and
the PostgreSQL trigger variants) is outside this snapshot.

Regenerate the snapshot after a deliberate revision change with::

    python -m tests.shared.native_store_schema > tests/fixtures/native_store_schema.json
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "native_store_schema.json"

_COMMENT = re.compile(r"--[^\n]*")
_SPACE = re.compile(r"\s+")


def _normalized(sql: str) -> str:
    return _SPACE.sub(" ", _COMMENT.sub("", sql)).strip()


def _objects(db_path: Path) -> dict[tuple[str, str], dict]:
    """Every schema object in ``db_path`` except SQLite's own and FTS internals."""
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        shadows = {
            name
            for name, kind in connection.execute(
                "SELECT name, type FROM pragma_table_list WHERE schema = 'main'"
            )
            if kind == "shadow"
        }
        objects = {}
        for kind, name, table, sql in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        ):
            if name in shadows:
                continue
            entry = {"sql": _normalized(sql)}
            if kind == "table":
                entry["columns"] = [
                    [column, declared, bool(notnull), default, primary]
                    for _, column, declared, notnull, default, primary in connection.execute(
                        f'PRAGMA table_info("{name}")'
                    )
                ]
            else:
                entry["table"] = table
            objects[(kind, name)] = entry
        return objects
    finally:
        connection.close()


_GROUPS = {"table": "tables", "index": "indexes", "trigger": "triggers", "view": "views"}


def _grouped(objects: dict[tuple[str, str], dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for (kind, name), entry in sorted(objects.items()):
        grouped.setdefault(_GROUPS[kind], {})[name] = entry
    return grouped


async def _initialized(db_path: Path, **components) -> None:
    from kestrel_sovereign.a2a.scope import NativeRecipientScope
    from kestrel_sovereign.a2a.task_manager import TaskManager
    from kestrel_sovereign.storage.db.sqlite import SQLiteBackend

    backend = SQLiteBackend(str(db_path))
    await backend.connect()
    manager = TaskManager.from_backend(
        backend,
        scope=NativeRecipientScope("did:schema-snapshot"),
        owns_backend=False,
        include_memory=components.get("memory", False),
        include_feedback=components.get("feedback", False),
    )
    try:
        await manager.initialize()
        if components.get("push", False):
            from kestrel_sovereign.a2a.push_outbox import NativePushOutbox
            from kestrel_sovereign.a2a.stores.unified.push_config_store import (
                PushConfigStore,
            )
            from kestrel_sovereign.signals.durable import DurableSignalStore

            signals = DurableSignalStore(backend)
            await signals.initialize()
            if components.get("push") == "baseline":
                return
            await NativePushOutbox(
                PushConfigStore(manager.task_store),
                signals,
                retention_days=30,
                lease_seconds=60,
            ).initialize()
    finally:
        await manager.close()
        await backend.close()


async def capture(directory: Path) -> dict:
    """What each component creates, with its current revision.

    The push stores need ``KESTREL_DATA_KEY``, because their initializer
    refuses to run without the key that encrypts callback credentials.
    """
    from kestrel_sovereign.a2a.native_schema import NativeStoreSchema

    async def objects(name: str, **components) -> dict[tuple[str, str], dict]:
        path = directory / f"{name}.db"
        await _initialized(path, **components)
        return _objects(path)

    core = await objects("core")
    created = {
        NativeStoreSchema.CORE: core,
        NativeStoreSchema.MEMORY: await objects("memory", memory=True),
        NativeStoreSchema.FEEDBACK: await objects("feedback", feedback=True),
        NativeStoreSchema.PUSH: await objects("push", push=True),
    }
    baselines = {
        NativeStoreSchema.CORE: {},
        NativeStoreSchema.MEMORY: core,
        NativeStoreSchema.FEEDBACK: core,
        NativeStoreSchema.PUSH: await objects("push-baseline", push="baseline"),
    }
    snapshot = {}
    for component in NativeStoreSchema:
        name, revision = component.value
        own = {
            key: entry
            for key, entry in created[component].items()
            if baselines[component].get(key) != entry
        }
        snapshot[name] = {"revision": revision, **_grouped(own)}
    return snapshot


def load_snapshot() -> dict:
    """The committed snapshot."""
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def render(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def main() -> int:
    import asyncio
    import base64
    import os
    import sys
    import tempfile

    os.environ.setdefault(
        "KESTREL_DATA_KEY", base64.urlsafe_b64encode(os.urandom(32)).decode()
    )
    with tempfile.TemporaryDirectory(prefix="native-store-schema-") as directory:
        sys.stdout.write(render(asyncio.run(capture(Path(directory)))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
