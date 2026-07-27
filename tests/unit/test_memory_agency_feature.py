"""
Tests for the MemoryAgencyFeature.

Verifies:
- Pinning a memory sets decay_protected in metadata
- Releasing a memory clears the pin
- Listing pinned memories returns only active pins
- Pin stats return correct ratios
- Double-pinning is idempotent
- Pinning a nonexistent message returns an error
"""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class FakeDB:
    """In-memory fake database for testing the memory agency feature."""

    def __init__(self):
        self.messages = {}  # id -> (id, content, metadata_json)
        self.pins = {}      # id -> dict with pin fields
        self._next_id = 1
        self.create_table_calls = 0

    def add_message(self, content, metadata=None, agent_id="test-agent", deleted_at=None):
        """Add a fake message and return its ID."""
        msg_id = self._next_id
        self._next_id += 1
        meta_json = json.dumps(metadata or {})
        self.messages[msg_id] = {
            "id": msg_id,
            "content": content,
            "metadata": meta_json,
            "agent_id": agent_id,
            "deleted_at": deleted_at,
        }
        return msg_id

    def trash_message(self, msg_id, when="2026-07-02T00:00:00+00:00"):
        """Soft-delete a message (move it to Trash)."""
        self.messages[msg_id]["deleted_at"] = when

    async def execute(self, sql, params=()):
        """Handle CREATE TABLE, UPDATE, and INSERT statements."""
        sql_lower = sql.strip().lower()

        if sql_lower.startswith("create table"):
            self.create_table_calls += 1
            return 0

        if sql_lower.startswith("update conversation_history"):
            if "json_set" in sql_lower or "jsonb_set" in sql_lower:
                # Atomic single-flag set of $.decay_protected (F217): merge the
                # flag into the EXISTING metadata, preserving other keys — the
                # whole point of the fix (no read-modify-write clobber).
                flag_val, msg_id, *rest = params
                agent_id = rest[0] if rest else None
                msg = self.messages.get(msg_id)
                if msg and (agent_id is None or msg["agent_id"] == agent_id):
                    raw = msg.get("metadata")
                    meta = json.loads(raw) if raw else {}
                    if isinstance(flag_val, str):
                        flag_bool = flag_val.lower() == "true"
                    else:
                        flag_bool = bool(flag_val)
                    meta["decay_protected"] = flag_bool
                    msg["metadata"] = json.dumps(meta)
                return 1
            # Legacy whole-JSON overwrite (kept for any remaining callers).
            meta_json, msg_id, *rest = params
            agent_id = rest[0] if rest else None
            msg = self.messages.get(msg_id)
            if msg and (agent_id is None or msg["agent_id"] == agent_id):
                msg["metadata"] = meta_json
            return 1

        if sql_lower.startswith("update memory_pins"):
            # UPDATE memory_pins SET released_at = ? WHERE id = ? AND agent_id = ?
            if "where id = ?" in sql_lower:
                released_at, pin_id = params[0], params[1]
                agent_id = params[2] if len(params) > 2 else None
                pin = self.pins.get(pin_id)
                if pin and (agent_id is None or pin["agent_id"] == agent_id):
                    pin["released_at"] = released_at
                return 1
            # Bulk unpin-all: WHERE agent_id = ? AND released_at IS NULL (no message_id)
            if "message_id" not in sql_lower:
                released_at = params[0]
                agent_id = params[1] if len(params) > 1 else None
                for pin in self.pins.values():
                    if pin["released_at"] is None and (
                        agent_id is None or pin["agent_id"] == agent_id
                    ):
                        pin["released_at"] = released_at
                return 1
            # UPDATE ... WHERE message_id = ? AND agent_id = ? AND released_at IS NULL
            released_at, message_id = params[0], params[1]
            agent_id = params[2] if len(params) > 2 else None
            for pin in self.pins.values():
                if (
                    pin["message_id"] == message_id
                    and pin["released_at"] is None
                    and (agent_id is None or pin["agent_id"] == agent_id)
                ):
                    pin["released_at"] = released_at
            return 1

        if "insert into memory_pins" in sql_lower:
            pin_id, message_id, agent_id, reason, pinned_at = params
            self.pins[pin_id] = {
                "id": pin_id,
                "message_id": message_id,
                "agent_id": agent_id,
                "pin_reason": reason,
                "pinned_at": pinned_at,
                "released_at": None,
            }
            return 1

        return 0

    async def execute_commit(self, sql, params=()):
        """Handle committed writes used by sovereign override cleanup."""
        sql_lower = sql.strip().lower()

        if sql_lower.startswith("delete from memory_pins"):
            if "message_id in" in sql_lower:
                agent_id, *message_ids = list(params)
                before = len(self.pins)
                self.pins = {
                    pin_id: pin
                    for pin_id, pin in self.pins.items()
                    if not (
                        pin["agent_id"] == agent_id
                        and pin["message_id"] in message_ids
                    )
                }
                return before - len(self.pins)

            agent_id = params[0]
            before = len(self.pins)
            self.pins = {
                pin_id: pin
                for pin_id, pin in self.pins.items()
                if not (
                    pin["agent_id"] == agent_id
                    and pin["released_at"] is None
                )
            }
            return before - len(self.pins)

        if sql_lower.startswith("update conversation_history"):
            agent_id = params[0]
            for msg in self.messages.values():
                if msg["agent_id"] != agent_id:
                    continue
                metadata = json.loads(msg["metadata"])
                if metadata.get("decay_protected"):
                    metadata["decay_protected"] = False
                    msg["metadata"] = json.dumps(metadata)
            return 1

        return await self.execute(sql, params)

    async def fetchone(self, sql, params=()):
        """Handle SELECT queries returning a single row."""
        sql_lower = sql.strip().lower()

        if "from conversation_history" in sql_lower and "where id = ?" in sql_lower:
            msg_id = params[0]
            msg = self.messages.get(msg_id)
            if not msg:
                return None
            # Honor the trash predicate: a soft-deleted row is invisible to
            # reads that filter `deleted_at IS NULL`.
            if "deleted_at is null" in sql_lower and msg.get("deleted_at") is not None:
                return None
            # Honor agent scoping when present.
            if "agent_id = ?" in sql_lower:
                agent_id = params[1]
                if msg["agent_id"] != agent_id:
                    return None
            # Return columns based on SELECT clause
            if "content, metadata" in sql_lower and "id," in sql_lower:
                return (msg["id"], msg["content"], msg["metadata"])
            if sql_lower.startswith("select metadata"):
                return (msg["metadata"],)
            if "metadata" in sql_lower:
                return (msg["id"], msg["metadata"])
            return (msg["id"], msg["content"], msg["metadata"])

        if "from memory_pins" in sql_lower and "released_at is null" in sql_lower:
            message_id = params[0]
            agent_id = params[1] if len(params) > 1 else None
            for pin in self.pins.values():
                if (
                    pin["message_id"] == message_id
                    and pin["released_at"] is None
                    and (agent_id is None or pin["agent_id"] == agent_id)
                ):
                    return (pin["id"],)
            return None

        return None

    async def fetchall(self, sql, params=()):
        """Handle SELECT queries returning multiple rows."""
        sql_lower = sql.strip().lower()

        if "from memory_pins" in sql_lower and "join conversation_history" in sql_lower:
            agent_id = params[0] if params else None
            results = []
            for pin in self.pins.values():
                if pin["released_at"] is not None:
                    continue
                msg = self.messages.get(pin["message_id"])
                # JOIN excludes trashed rows (ch.deleted_at IS NULL).
                if "deleted_at is null" in sql_lower and msg and msg.get("deleted_at") is not None:
                    continue
                # JOIN scopes to ch.agent_id = ?.
                if msg and (agent_id is None or msg["agent_id"] == agent_id):
                    results.append((
                        pin["id"],
                        pin["message_id"],
                        pin["pin_reason"],
                        pin["pinned_at"],
                        msg["content"],
                    ))
            return results

        # SELECT message_id FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        if (
            "select message_id from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            agent_id = params[0] if params else None
            return [
                (p["message_id"],)
                for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]

        # SELECT id, message_id FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        # ORDER BY pinned_at ASC LIMIT ?
        if (
            "select id, message_id from memory_pins" in sql_lower
            and "order by pinned_at asc" in sql_lower
        ):
            agent_id = params[0] if params else None
            limit = params[1] if len(params) > 1 else None
            active = sorted(
                [
                    p for p in self.pins.values()
                    if p["released_at"] is None
                    and (agent_id is None or p["agent_id"] == agent_id)
                ],
                key=lambda p: p["pinned_at"],
            )
            if limit is not None:
                active = active[:limit]
            return [(p["id"], p["message_id"]) for p in active]

        # SELECT pinned_at FROM memory_pins WHERE agent_id = ? AND released_at IS NULL
        if (
            "select pinned_at from memory_pins" in sql_lower
            and "released_at is null" in sql_lower
        ):
            agent_id = params[0] if params else None
            return [
                (p["pinned_at"],)
                for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]

        return []

    async def fetchval(self, sql, params=()):
        """Handle SELECT COUNT(*) and aggregate queries."""
        sql_lower = sql.strip().lower()

        if "count(*)" in sql_lower and "conversation_history" in sql_lower:
            agent_id = params[0] if params else None
            filter_trash = "deleted_at is null" in sql_lower
            return sum(
                1 for m in self.messages.values()
                if (agent_id is None or m["agent_id"] == agent_id)
                and not (filter_trash and m.get("deleted_at") is not None)
            )

        if "count(*)" in sql_lower and "memory_pins" in sql_lower:
            agent_id = params[0] if params else None
            scoped = [
                p for p in self.pins.values()
                if agent_id is None or p["agent_id"] == agent_id
            ]
            if "released_at is null" in sql_lower:
                return sum(1 for p in scoped if p["released_at"] is None)
            if "released_at is not null" in sql_lower:
                return sum(1 for p in scoped if p["released_at"] is not None)
            return len(scoped)

        if "min(pinned_at)" in sql_lower:
            agent_id = params[0] if params else None
            active = [
                p["pinned_at"] for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]
            return min(active) if active else None

        if "max(pinned_at)" in sql_lower:
            agent_id = params[0] if params else None
            active = [
                p["pinned_at"] for p in self.pins.values()
                if p["released_at"] is None
                and (agent_id is None or p["agent_id"] == agent_id)
            ]
            return max(active) if active else None

        return 0


class FakeGraphStore:
    """In-memory fake graph store for testing KG writes."""

    def __init__(self):
        self.nodes = {}   # node_id -> GraphNode
        self.edges = []   # list of (source_id, target_id, label)

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def add_edge(self, source_id, target_id, label, properties=None):
        self.edges.append((source_id, target_id, label))


class FakeCanonicalFactStorage:
    """Narrow governed-storage double for MemoryAgencyFeature receipts."""

    def __init__(self, tenant_id="did:example:memory-agency"):
        from kestrel_sovereign.knowledge import Visibility
        from kestrel_sovereign.storage.semantic_binding import SemanticAssertionBinding

        self.binding = SemanticAssertionBinding(
            tenant_id=tenant_id,
            owning_agent_id=tenant_id,
            privacy_classification="normal",
            release_policy_reference="policy:privacy:normal-v1",
            visibility=Visibility.PRIVATE,
        )
        self.current = []
        self.sources = {}
        self.operations = {}
        self.supersession_operations = {}
        self.restoration_operations = {}
        self.delete_operations = {}
        self.forget_noop_operations = set()
        self.revisions = {}
        self.assertion_currents = {}
        self.put_calls = []
        self.supersede_calls = []
        self.append_calls = []
        self.restore_calls = []
        self.delete_calls = []
        from kestrel_sovereign.privacy import PrivacyMode
        from kestrel_sovereign.storage.privacy_wrapper import (
            PrivacyEnforcingStorage,
        )

        self._governed = PrivacyEnforcingStorage(self, PrivacyMode.NORMAL)

    def semantic_assertion_binding(self):
        return self.binding

    async def save_explicit_fact(self, **kwargs):
        return await self._governed.save_explicit_fact(**kwargs)

    async def forget_explicit_fact(self, **kwargs):
        return await self._governed.forget_explicit_fact(**kwargs)

    async def _replay_governed_assertion_operation(self, operation_id, *_replay_binding):
        if operation_id in self.operations:
            return SimpleNamespace(
                operation="put",
                report=self._report(),
                assertion=self.operations[operation_id],
                predecessor=None,
            )
        if operation_id in self.supersession_operations:
            predecessor, replacement = self.supersession_operations[operation_id]
            return SimpleNamespace(
                operation="supersede",
                report=self._report(),
                assertion=replacement,
                predecessor=predecessor,
            )
        if operation_id in self.restoration_operations:
            predecessor, replacement = self.restoration_operations[
                operation_id
            ]
            return SimpleNamespace(
                operation="restore",
                report=self._report(),
                assertion=replacement,
                predecessor=predecessor,
            )
        return None

    async def _terminalize_legacy_erased_explicit_fact_operation(
        self,
        operation_id,
        binding,
    ):
        return None

    async def _replay_explicit_fact_forget_operation(
        self,
        operation_id,
        *_selector,
    ):
        deletion = self.delete_operations.get(operation_id)
        if deletion is not None:
            return SimpleNamespace(
                deleted=True,
                deletion=deletion,
                idempotent=True,
            )
        if operation_id in self.forget_noop_operations:
            return SimpleNamespace(
                deleted=False,
                deletion=None,
                idempotent=True,
            )
        return None

    async def _record_explicit_fact_forget_noop(
        self,
        operation_id,
        subject,
        predicate,
    ):
        from kestrel_sovereign.storage.async_assertion_store import (
            AssertionConflictError,
        )

        if operation_id in self.forget_noop_operations:
            return SimpleNamespace(
                deleted=False,
                deletion=None,
                idempotent=True,
            )
        if any(
            assertion.subject == subject and assertion.predicate == predicate
            for assertion in self.current
        ):
            raise AssertionConflictError(
                "explicit fact selector gained a current assertion before no-op commit"
            )
        self.forget_noop_operations.add(operation_id)
        return SimpleNamespace(
            deleted=False,
            deletion=None,
            idempotent=False,
        )

    async def query_assertions(self, query):
        return [
            assertion for assertion in self.current
            if assertion.subject == query.subject and assertion.predicate == query.predicate
        ]

    async def list_assertion_sources(self, assertion_id):
        return list(self.sources.get(assertion_id, ()))

    @staticmethod
    def _report():
        return SimpleNamespace(
            state=SimpleNamespace(value="conforms"),
            action=SimpleNamespace(value="accept"),
            report_id="validation-report",
        )

    async def put_assertion(self, assertion, *, source_occurrences, operation_id):
        self.put_calls.append((assertion, source_occurrences, operation_id))
        if operation_id in self.operations:
            return SimpleNamespace(
                accepted=True,
                assertion=self.operations[operation_id],
                report=self._report(),
                idempotent=True,
            )
        self.operations[operation_id] = assertion
        self.current = [assertion]
        self.assertion_currents[assertion.assertion_id] = assertion
        self.sources[assertion.assertion_id] = list(source_occurrences)
        self.revisions[assertion.revision_id] = assertion
        return SimpleNamespace(
            accepted=True,
            assertion=assertion,
            report=self._report(),
            idempotent=False,
        )

    async def put_validated_assertion(
        self,
        assertion,
        *,
        source_occurrences,
        operation_id,
    ):
        return await self.put_assertion(
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def supersede_assertion(self, revision_id, assertion, *, source_occurrences, operation_id):
        self.supersede_calls.append((revision_id, assertion, source_occurrences, operation_id))
        if operation_id in self.supersession_operations:
            predecessor, replacement = self.supersession_operations[operation_id]
            return SimpleNamespace(
                accepted=True,
                predecessor=predecessor,
                replacement=replacement,
                report=self._report(),
                idempotent=True,
            )
        predecessor = self.current[0]
        predecessor_state = replace(
            predecessor,
            revision_id=f"superseded:{predecessor.revision_id}",
            status=type(predecessor.status).SUPERSEDED,
            supersedes_revision_id=predecessor.revision_id,
        )
        replacement = replace(
            assertion,
            supersedes_revision_id=predecessor_state.revision_id,
        )
        self.current = [replacement]
        self.assertion_currents[predecessor.assertion_id] = predecessor_state
        self.assertion_currents[replacement.assertion_id] = replacement
        self.sources.setdefault(assertion.assertion_id, []).extend(
            source_occurrences
        )
        self.revisions[predecessor_state.revision_id] = predecessor_state
        self.revisions[replacement.revision_id] = replacement
        self.supersession_operations[operation_id] = (
            predecessor_state,
            replacement,
        )
        return SimpleNamespace(
            accepted=True,
            predecessor=predecessor_state,
            replacement=replacement,
            report=self._report(),
            idempotent=False,
        )

    async def supersede_validated_assertion(
        self,
        revision_id,
        assertion,
        *,
        source_occurrences,
        operation_id,
    ):
        return await self.supersede_assertion(
            revision_id,
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def append_assertion_source(
        self,
        revision_id,
        assertion,
        *,
        source_occurrences,
        operation_id,
    ):
        self.append_calls.append(
            (revision_id, assertion, source_occurrences, operation_id)
        )
        return await self.supersede_assertion(
            revision_id,
            assertion,
            source_occurrences=source_occurrences,
            operation_id=operation_id,
        )

    async def _restore_explicit_fact_assertion(
        self,
        revision_id,
        assertion,
        *,
        source_occurrences,
        operation_id,
    ):
        self.restore_calls.append(
            (revision_id, assertion, source_occurrences, operation_id)
        )
        if operation_id in self.restoration_operations:
            predecessor, replacement = self.restoration_operations[
                operation_id
            ]
            return SimpleNamespace(
                accepted=True,
                predecessor=predecessor,
                replacement=replacement,
                report=self._report(),
                idempotent=True,
            )
        predecessor = next(
            item
            for item in self.assertion_currents.values()
            if item.revision_id == revision_id
        )
        replacement = replace(
            assertion,
            supersedes_revision_id=predecessor.revision_id,
        )
        self.current = [replacement]
        self.assertion_currents[replacement.assertion_id] = replacement
        self.sources.setdefault(assertion.assertion_id, []).extend(
            source_occurrences
        )
        self.revisions[replacement.revision_id] = replacement
        self.restoration_operations[operation_id] = (
            predecessor,
            replacement,
        )
        return SimpleNamespace(
            accepted=True,
            predecessor=predecessor,
            replacement=replacement,
            report=self._report(),
            idempotent=False,
        )

    async def get_assertion_revision(self, revision_id):
        return self.revisions.get(revision_id)

    async def get_assertion(self, assertion_id, *, include_inactive=False):
        assertion = self.assertion_currents.get(assertion_id)
        if assertion is None:
            return None
        if include_inactive or assertion.status.value == "active":
            return assertion
        return None

    async def _delete_explicit_fact_assertion(
        self,
        assertion_id,
        revision_id,
        *,
        operation_id,
        explicit_fact_selector=None,
    ):
        self.delete_calls.append((assertion_id, revision_id, operation_id))
        if operation_id in self.delete_operations:
            return self.delete_operations[operation_id]
        target = self.current.pop()
        deleted = replace(
            target,
            revision_id=f"deleted:{target.revision_id}",
            status=type(target.status).DELETED,
            supersedes_revision_id=None,
        )
        self.assertion_currents[deleted.assertion_id] = deleted
        self.revisions[deleted.revision_id] = deleted
        result = SimpleNamespace(deleted=deleted, idempotent=False)
        self.delete_operations[operation_id] = result
        return result


class PrivacyBlockedCanonicalFactStorage(FakeCanonicalFactStorage):
    """Models the wrapper's fail-closed anonymous semantic policy."""

    async def save_explicit_fact(self, **kwargs):
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyViolationError

        raise PrivacyViolationError("canonical assertion operation blocked by privacy policy")

    async def forget_explicit_fact(self, **kwargs):
        from kestrel_sovereign.storage.privacy_wrapper import PrivacyViolationError

        raise PrivacyViolationError("canonical assertion operation blocked by privacy policy")


class ErasedCanonicalFactStorage(FakeCanonicalFactStorage):
    """Models a blinded terminal replay after physical assertion erasure."""

    async def _replay_governed_assertion_operation(self, operation_id, *_replay_binding):
        return SimpleNamespace(
            operation="put",
            generation=7,
            terminal_erased=True,
        )


class ValidationRejectedCanonicalFactStorage(FakeCanonicalFactStorage):
    async def put_assertion(self, assertion, *, source_occurrences, operation_id):
        self.put_calls.append((assertion, source_occurrences, operation_id))
        report = SimpleNamespace(
            state=SimpleNamespace(value="nonconformant"),
            action=SimpleNamespace(value="reject"),
            report_id="rejected-report",
        )
        return SimpleNamespace(accepted=False, report=report, idempotent=False)


class ValidationUnavailableCanonicalFactStorage(FakeCanonicalFactStorage):
    async def put_assertion(self, assertion, *, source_occurrences, operation_id):
        from kestrel_sovereign.knowledge.shacl_validation import ShaclCapabilityUnavailable

        raise ShaclCapabilityUnavailable("the pinned SHACL profile is unavailable")


def _make_feature(fake_db, agent_id="test-agent", graph_store=None, semantic_storage=None):
    """Create a MemoryAgencyFeature with a mocked agent and fake database."""
    from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature, PIN_QUOTA_DEFAULT

    storage = semantic_storage or MagicMock()
    storage.db = fake_db
    storage.agent_id = agent_id
    storage.graph = graph_store

    agent = MagicMock()
    agent.storage = storage

    feature = MemoryAgencyFeature(agent)
    feature.storage = storage
    feature._db = fake_db
    feature.agent_id = agent_id
    feature.pin_quota = PIN_QUOTA_DEFAULT
    return feature


class PrivacyWrappedStorage:
    def __init__(self, raw_storage):
        self._storage = raw_storage

    @property
    def db(self):
        raise AssertionError("deprecated wrapper db property was touched")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_uses_raw_storage_without_touching_wrapper_db():
    """Initialization must not trip PrivacyEnforcingStorage.db."""
    from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature

    db = FakeDB()
    raw_storage = MagicMock(db=db)
    agent = MagicMock()
    agent.did = "test-agent"
    agent.storage = PrivacyWrappedStorage(raw_storage)
    agent._raw_storage = raw_storage

    feature = MemoryAgencyFeature(agent)

    await feature.initialize()

    assert feature._db is db


@pytest.mark.asyncio
async def test_memory_pin_refuses_privacy_hidden_mode_without_db_access():
    """Runtime privacy changes must block persistent pin writes."""
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.privacy import PrivacyConfig

    db = FakeDB()
    msg_id = db.add_message("Do not pin while hidden")
    feature = _make_feature(db)
    feature.agent.privacy_config = PrivacyConfig(storage="none", llm_location="local")

    result = await feature.memory_pin(message_id=msg_id, reason="private")

    assert result.status is ToolResultStatus.ERROR
    assert "privacy mode" in result.error
    assert db.pins == {}
    assert json.loads(db.messages[msg_id]["metadata"]) == {}


@pytest.mark.asyncio
async def test_sovereign_override_still_clears_pins_in_privacy_hidden_mode():
    """Privacy mode must not let pins resist sovereign cleanup."""
    from kestrel_sovereign.privacy import PrivacyConfig

    db = FakeDB()
    msg_id = db.add_message("Pinned before privacy switch")
    feature = _make_feature(db)

    pin_result = await feature.memory_pin(message_id=msg_id, reason="cleanup")
    assert pin_result.data["pinned"] is True
    assert db.pins

    feature.agent.privacy_config = PrivacyConfig(storage="none", llm_location="local")

    removed = await feature.sovereign_override_pins(
        "test-agent",
        message_ids=[msg_id],
        reason="privacy_mode_change",
    )

    assert removed == 1
    assert db.pins == {}
    assert json.loads(db.messages[msg_id]["metadata"])["decay_protected"] is False


@pytest.mark.asyncio
async def test_sovereign_override_ensures_pin_table_after_privacy_first_init():
    """Cleanup must not assume memory_pins was created before hidden mode."""
    from kestrel_sovereign.features.memory_agency.feature import MemoryAgencyFeature
    from kestrel_sovereign.privacy import PrivacyConfig

    db = FakeDB()
    raw_storage = MagicMock(db=db)
    agent = MagicMock()
    agent.did = "test-agent"
    agent.storage = PrivacyWrappedStorage(raw_storage)
    agent._raw_storage = raw_storage
    agent.privacy_config = PrivacyConfig(storage="none", llm_location="local")
    feature = MemoryAgencyFeature(agent)

    await feature.initialize()
    assert feature._db is None
    assert db.create_table_calls == 0

    removed = await feature.sovereign_override_pins(
        "test-agent",
        reason="privacy_first_cleanup",
    )

    assert removed == 0
    assert db.create_table_calls == 1


@pytest.mark.asyncio
async def test_pin_memory_sets_decay_protected():
    """Pinning a message should set decay_protected=True in its metadata."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    msg_id = db.add_message("Remember this important event", {"importance": 0.8})
    feature = _make_feature(db)

    result = await feature.memory_pin(message_id=msg_id, reason="milestone")

    # 1 pin / 1 message = 100% ratio → PARTIAL with over-pinning caveat
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["pinned"] is True
    assert result.data["message_id"] == msg_id
    assert "Remember this" in result.data["preview"]

    # Verify metadata was updated AND the pre-existing flag was preserved
    # (F217 — atomic json_set, not a whole-JSON overwrite).
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True
    assert stored_meta["importance"] == 0.8


@pytest.mark.asyncio
async def test_pin_does_not_clobber_concurrent_metadata_writer():
    """F217: setting decay_protected must not drop a flag another writer added
    concurrently. With the atomic single-flag json_set, a flag written to the
    row after the pin's read still survives the pin's write."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    msg_id = db.add_message("Concurrent note", {"importance": 0.5})
    feature = _make_feature(db)

    # Simulate a concurrent writer (e.g. the consolidator) landing a DIFFERENT
    # flag on the row before the pin's write commits.
    original_set = feature._set_decay_protected

    async def _racing_set(dbarg, message_id, agent_id, value):
        meta = json.loads(db.messages[message_id]["metadata"])
        meta["access_count"] = 7  # a concurrent, unrelated metadata write
        db.messages[message_id]["metadata"] = json.dumps(meta)
        return await original_set(dbarg, message_id, agent_id, value)

    feature._set_decay_protected = _racing_set

    result = await feature.memory_pin(message_id=msg_id, reason="milestone")
    assert result.status in (ToolResultStatus.OK, ToolResultStatus.PARTIAL)

    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True
    assert stored_meta["access_count"] == 7  # concurrent write NOT clobbered
    assert stored_meta["importance"] == 0.5


@pytest.mark.asyncio
async def test_release_memory_clears_pin():
    """Releasing a pinned message should clear decay_protected and set released_at."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    msg_id = db.add_message("Temporary note", {"importance": 0.5})
    feature = _make_feature(db)

    # Pin first
    await feature.memory_pin(message_id=msg_id, reason="temp")
    # Release
    result = await feature.memory_release(message_id=msg_id)

    assert result.status is ToolResultStatus.OK
    assert result.data["released"] is True
    assert result.data["message_id"] == msg_id

    # Metadata should have decay_protected=False
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is False

    # Pin record should have released_at set
    active_pins = [p for p in db.pins.values() if p["released_at"] is None]
    assert len(active_pins) == 0


@pytest.mark.asyncio
async def test_list_pinned_returns_active_pins():
    """memory_pinned should return only non-released pins."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    msg1 = db.add_message("First memory")
    msg2 = db.add_message("Second memory")
    msg3 = db.add_message("Third memory")
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg1, reason="reason A")
    await feature.memory_pin(message_id=msg2, reason="reason B")
    await feature.memory_pin(message_id=msg3, reason="reason C")

    # Release the second one
    await feature.memory_release(message_id=msg2)

    result = await feature.memory_pinned()

    assert result.status is ToolResultStatus.OK
    assert result.data["count"] == 2
    pinned_ids = {p["message_id"] for p in result.data["pins"]}
    assert msg1 in pinned_ids
    assert msg3 in pinned_ids
    assert msg2 not in pinned_ids


@pytest.mark.asyncio
async def test_pin_stats_returns_ratios():
    """memory_pin_stats should return correct counts and ratios."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    # Add 10 messages
    ids = [db.add_message(f"Message {i}") for i in range(10)]
    feature = _make_feature(db)

    # Pin 3 messages
    await feature.memory_pin(message_id=ids[0])
    await feature.memory_pin(message_id=ids[1])
    await feature.memory_pin(message_id=ids[2])

    # Release 1
    await feature.memory_release(message_id=ids[1])

    result = await feature.memory_pin_stats()

    # 2/10 = 20% ratio, below threshold → OK
    assert result.status is ToolResultStatus.OK
    assert result.data["total_messages"] == 10
    assert result.data["pinned"] == 2       # 3 pinned - 1 released = 2 active
    assert result.data["released"] == 1
    assert result.data["pin_ratio"] == 0.2  # 2 / 10


@pytest.mark.asyncio
async def test_double_pin_is_idempotent():
    """Pinning the same message twice should not create a duplicate pin record."""
    db = FakeDB()
    msg_id = db.add_message("Pin me twice")
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg_id, reason="first pin")
    await feature.memory_pin(message_id=msg_id, reason="second pin")

    # Should still have only one active pin
    active_pins = [p for p in db.pins.values() if p["released_at"] is None and p["message_id"] == msg_id]
    assert len(active_pins) == 1

    # Metadata should still be protected
    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True


@pytest.mark.asyncio
async def test_pin_nonexistent_message_returns_error():
    """Pinning a message that does not exist should return an error."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    feature = _make_feature(db)

    result = await feature.memory_pin(message_id=99999)

    assert result.status is ToolResultStatus.ERROR
    assert "99999" in result.error


@pytest.mark.asyncio
async def test_release_nonexistent_message_returns_error():
    """Releasing a message that does not exist should return an error."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    feature = _make_feature(db)

    result = await feature.memory_release(message_id=99999)

    assert result.status is ToolResultStatus.ERROR
    assert "99999" in result.error


@pytest.mark.asyncio
async def test_pinned_memory_boost_in_retriever():
    """Verify that _calculate_score boosts importance for decay_protected memories."""
    from kestrel_sovereign.storage.memory_retriever import MemoryRetriever

    store = AsyncMock()
    retriever = MemoryRetriever(conversation_store=store)

    # Score with decay_protected = True and low base importance
    score_pinned = retriever._calculate_score(
        content="test content",
        query="test",
        metadata={"importance": 0.3, "decay_protected": True},
        emotional_context=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        expanded_concepts=[],
    )

    # Score with decay_protected = False and same low base importance
    score_unpinned = retriever._calculate_score(
        content="test content",
        query="test",
        metadata={"importance": 0.3, "decay_protected": False},
        emotional_context=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        expanded_concepts=[],
    )

    # Pinned should score higher due to importance boost
    assert score_pinned > score_unpinned


@pytest.mark.asyncio
async def test_pin_preserves_existing_metadata():
    """Pinning should preserve existing metadata fields while adding decay_protected."""
    db = FakeDB()
    original_meta = {
        "importance": 0.8,
        "emotional_valence": 0.6,
        "emotional_categories": ["joy"],
        "custom_field": "preserved",
    }
    msg_id = db.add_message("Important joyful memory", original_meta)
    feature = _make_feature(db)

    await feature.memory_pin(message_id=msg_id, reason="important")

    stored_meta = json.loads(db.messages[msg_id]["metadata"])
    assert stored_meta["decay_protected"] is True
    assert stored_meta["importance"] == 0.8
    assert stored_meta["emotional_valence"] == 0.6
    assert stored_meta["emotional_categories"] == ["joy"]
    assert stored_meta["custom_field"] == "preserved"


# --------------------------------------------------------------------------
# save_fact canonical assertion tests
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_fact_creates_canonical_receipt_without_graph_write():
    """The explicit tool has one mutation route: the governed assertion facade."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    graph = FakeGraphStore()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, graph_store=graph, semantic_storage=canonical)

    result = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )

    assert result.status is ToolResultStatus.OK
    assert result.data["saved"] is True
    assert result.data["subject"] == "user"
    assert result.data["predicate"] == "preferred_deploy_region"
    assert result.data["value"] == "us-central1"
    assert result.data["assertion_id"].startswith("urn:kestrel:assertion:sha256:")
    assert result.data["revision_id"]
    assert result.data["validation_disposition"] == "conforms:accept"
    assert result.data["provenance_reference"].startswith("source:memory-agency-save-fact-v1:")
    assert result.data["provenance_digest"].startswith("sha256:")

    assertion, sources, operation_id = canonical.put_calls[0]
    assert assertion.tenant_id == canonical.binding.tenant_id
    assert assertion.owning_agent_id == canonical.binding.owning_agent_id
    assert assertion.subject.value.endswith(":principal:user")
    assert assertion.predicate.value == "https://kestrel.ai/vocab/preferredDeployRegion"
    assert assertion.object.datatype_iri == "http://www.w3.org/2001/XMLSchema#string"
    assert assertion.ontology_version.namespace == "https://kestrel.ai/vocab/"
    assert assertion.ontology_version.version == "1.0.0"
    assert assertion.ontology_version.content_digest == "db708b6790e5212bcbfd5040a1d7883da1161b05e73c809ee8d924c31b2a8044"
    assert sources[0].locator == f"tool:memory_agency.save_fact#{operation_id}"
    assert sources[0].actor == canonical.binding.owning_agent_id
    assert "us-central1" not in sources[0].locator
    assert "us-central1" not in sources[0].content_digest
    assert graph.nodes == {}
    assert graph.edges == []


@pytest.mark.asyncio
async def test_save_fact_erased_replay_exposes_only_terminal_disposition():
    """A stale retry cannot recover erased content or semantic identifiers."""
    from kestrel_sdk.tools.result import ToolResultStatus

    feature = _make_feature(
        FakeDB(),
        semantic_storage=ErasedCanonicalFactStorage(),
    )
    result = await feature.save_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value="secret-region",
    )

    assert result.status is ToolResultStatus.ERROR
    assert result.data == {
        "saved": False,
        "assertion_id": None,
        "revision_id": None,
        "validation_disposition": "erased:terminal",
        "validation_report_id": None,
        "provenance_reference": None,
        "provenance_digest": None,
        "operation_id": result.data["operation_id"],
        "idempotent": True,
        "superseded_assertion_id": None,
    }
    assert "secret-region" not in repr(result.data)
    assert "preferred_deploy_region" not in repr(result.data)


@pytest.mark.asyncio
async def test_save_fact_retries_idempotently_and_supersedes_changed_value():
    """Retries retain provenance; changed values use the canonical lifecycle."""
    from kestrel_sovereign.agent.invocation import invocation_scope
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)

    with invocation_scope("request-1"):
        first = await feature.save_fact(
            subject="user", predicate="preferred_deploy_region", value="us-central1"
        )
        replay = await feature.save_fact(
            subject="user", predicate="preferred_deploy_region", value="us-central1"
        )
    with invocation_scope("request-2"):
        replacement = await feature.save_fact(
            subject="user", predicate="preferred_deploy_region", value="europe-west4"
        )

    assert first.status is ToolResultStatus.OK
    assert replay.status is ToolResultStatus.OK
    assert replay.data["idempotent"] is True
    assert replay.data["assertion_id"] == first.data["assertion_id"]
    assert replacement.status is ToolResultStatus.OK
    assert replacement.data["superseded_assertion_id"] == first.data["assertion_id"]
    assert replacement.data["assertion_id"] != first.data["assertion_id"]
    assert len(canonical.supersede_calls) == 1
    assert canonical.current[0].object.value == "europe-west4"


@pytest.mark.asyncio
async def test_save_fact_is_idempotent_after_a_percent_encoded_header_echo():
    """An invoke response header can be retried verbatim without new evidence.

    The ID has both a literal percent and text that resembles a percent escape;
    decode-once is required to preserve both as opaque operation material.
    """
    from kestrel_sovereign.agent.invocation import (
        invocation_id_from_request_header,
        invocation_id_response_header,
        invocation_scope,
    )

    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(FakeDB(), semantic_storage=canonical)
    request_id = "teach ☃ / 100% %E2%98%83?copy=1#retry"
    header_echo = invocation_id_response_header(request_id)

    assert invocation_id_from_request_header(header_echo) == request_id
    # A second application would mutate the semantic retry key; pin that the
    # wire boundary applies exactly one decode and no more.
    assert invocation_id_from_request_header("%2525E2%2525") == "%25E2%25"

    with invocation_scope(invocation_id_from_request_header(header_echo)):
        first = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )
    with invocation_scope(invocation_id_from_request_header(header_echo)):
        replay = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )

    assert replay.data["idempotent"] is True
    assert replay.data["assertion_id"] == first.data["assertion_id"]
    assert len(canonical.sources[first.data["assertion_id"]]) == 1


@pytest.mark.asyncio
async def test_save_fact_distinct_same_value_invocation_appends_governed_provenance():
    """Only an exact retry is idempotent; another request retains its evidence."""
    from kestrel_sovereign.agent.invocation import invocation_scope

    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(FakeDB(), semantic_storage=canonical)

    with invocation_scope("teach-a"):
        first = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )
    with invocation_scope("teach-b"):
        second = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )
        replay = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )

    assert first.data["assertion_id"] == second.data["assertion_id"]
    assert second.data["idempotent"] is False
    assert second.data["provenance_reference"] is not None
    assert len(canonical.append_calls) == 1
    assert len(canonical.sources[first.data["assertion_id"]]) == 2
    assert replay.data["idempotent"] is True
    assert replay.data["provenance_reference"] == second.data["provenance_reference"]
    assert len(canonical.sources[first.data["assertion_id"]]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [
        "retracted",
        "quarantined",
        "superseded",
    ],
)
async def test_save_fact_does_not_restore_non_deleted_terminal_shells(
    terminal_status,
):
    """Ordinary teaching cannot undo deliberate non-delete lifecycle state."""
    from kestrel_sovereign.agent.invocation import invocation_scope
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        FactLifecycleError,
    )
    from kestrel_sovereign.knowledge import AssertionStatus, EpistemicState

    canonical = FakeCanonicalFactStorage()
    with invocation_scope("terminal-shell-first"):
        first = await canonical._governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
        )
    active = canonical.current.pop()
    status = AssertionStatus(terminal_status)
    shell = replace(
        active,
        revision_id=f"{terminal_status}:{active.revision_id}",
        status=status,
        epistemic_state=(
            EpistemicState.RETRACTED
            if status is AssertionStatus.RETRACTED
            else active.epistemic_state
        ),
        supersedes_revision_id=(
            active.revision_id
            if status is AssertionStatus.SUPERSEDED
            else None
        ),
    )
    canonical.assertion_currents[first.assertion_id] = shell
    canonical.revisions[shell.revision_id] = shell

    with invocation_scope(f"terminal-shell-{terminal_status}-retry"):
        with pytest.raises(
            FactLifecycleError,
            match="cannot revive",
        ):
            await canonical._governed.save_explicit_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value="us-central1",
                confidence=0.9,
            )

    assert canonical.restore_calls == []
    assert canonical.current == []


@pytest.mark.asyncio
async def test_save_fact_logs_neither_taught_content_nor_canonical_identifiers():
    """Operator logs retain only fixed outcome metadata for explicit facts."""
    taught_value = "PRIVATE_FACT_VALUE_DO_NOT_LOG"
    feature = _make_feature(FakeDB(), semantic_storage=FakeCanonicalFactStorage())

    with patch("kestrel_sovereign.features.memory_agency.feature.logger.info") as info:
        result = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value=taught_value,
        )

    assert result.data["assertion_id"] is not None
    log_text = " ".join(
        str(part)
        for call in info.call_args_list
        for part in (*call.args, *call.kwargs.values())
    )
    assert taught_value not in log_text
    assert result.data["assertion_id"] not in log_text


@pytest.mark.asyncio
async def test_save_fact_uses_task_local_invocation_not_agent_global_request_id():
    """Concurrent turns retain their own canonical operation provenance."""
    from kestrel_sovereign.agent.invocation import invocation_scope
    from kestrel_sovereign.features.memory_agency.semantic_facts import (
        _operation_material,
    )

    async def teach(invocation_id, value):
        canonical = FakeCanonicalFactStorage()
        feature = _make_feature(FakeDB(), semantic_storage=canonical)
        # This is deliberately wrong for both turns. Provenance must never use
        # the shared lifecycle fallback.
        feature.agent._current_request_id = "wrong-shared-request-id"
        with invocation_scope(invocation_id):
            await asyncio.sleep(0)
            result = await feature.save_fact(
                subject="user",
                predicate="preferred_deploy_region",
                value=value,
            )
        return result, _operation_material(
            action="save",
            subject="user",
            predicate="preferred_deploy_region",
            value=value,
            confidence_requested=1.0,
            invocation_id=invocation_id,
        )[0]

    first, second = await asyncio.gather(
        teach("turn-a", "us-central1"),
        teach("turn-b", "europe-west4"),
    )

    assert first[0].data["operation_id"] == first[1]
    assert second[0].data["operation_id"] == second[1]
    assert first[0].data["operation_id"] != second[0].data["operation_id"]


@pytest.mark.asyncio
async def test_save_fact_uses_authenticated_task_local_provenance_not_agent_owner():
    """The tool cannot choose actor/source fields; the trusted turn can."""
    from kestrel_sovereign.agent.invocation import (
        invocation_scope,
        request_provenance,
    )

    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(FakeDB(), semantic_storage=canonical)
    provenance = request_provenance(
        actor="owner@example.test",
        source_kind="http_request",
        source_locator="POST:/v1/chat/completions",
        received_at="2026-07-26T12:00:00+00:00",
    )

    with invocation_scope("openai-retry-2765", provenance=provenance):
        result = await feature.save_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
        )

    assert result.data["saved"] is True
    _, sources, _ = canonical.put_calls[0]
    source = sources[0]
    assert source.actor == "owner@example.test"
    assert source.source_kind == "http_request"
    assert source.locator.startswith(
        "POST:/v1/chat/completions#tool:memory_agency.save_fact#"
    )
    assert source.received_at.value == "2026-07-26T12:00:00Z"
    assert source.actor != canonical.binding.owning_agent_id


def test_nested_invocation_keeps_trusted_request_provenance_for_command_delegation():
    """Streaming command delegation must not clear the outer HTTP context."""
    from kestrel_sovereign.agent.invocation import (
        current_invocation_provenance,
        invocation_scope,
        request_provenance,
    )

    provenance = request_provenance(
        actor="owner@example.test",
        source_kind="http_request",
        source_locator="POST:/api/agent/stream",
        received_at="2026-07-26T12:00:00Z",
    )
    with invocation_scope("outer-retry-2765", provenance=provenance):
        with invocation_scope("nested-command"):
            assert current_invocation_provenance() is provenance


@pytest.mark.asyncio
async def test_save_fact_without_a_turn_generates_fresh_direct_invocation_ids():
    """Non-HTTP producers no longer share the permanent ``direct`` identity."""
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(FakeDB(), semantic_storage=canonical)

    first = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )
    second = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="europe-west4"
    )
    third = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )

    assert first.data["saved"] is True
    assert second.data["saved"] is True
    assert third.data["saved"] is True
    assert first.data["operation_id"] != second.data["operation_id"]
    assert second.data["operation_id"] != third.data["operation_id"]
    assert len(canonical.supersede_calls) == 2


@pytest.mark.asyncio
async def test_save_fact_clamps_confidence():
    """Out-of-range confidence is clamped and surfaced as PARTIAL."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)

    result = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="x", confidence=2.5
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["confidence"] == 1.0
    assert result.data["confidence_requested"] == 2.5
    assert result.data["confidence_clamped"] is True
    assert "clamped" in result.error
    assert "assertion_id=" in result.confirmation
    assert "provenance_reference=" in result.confirmation

    result = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="y", confidence=-0.5
    )
    assert result.status is ToolResultStatus.PARTIAL
    assert result.data["confidence"] == 0.0
    assert result.data["confidence_clamped"] is True


@pytest.mark.asyncio
async def test_save_fact_rejects_unsupported_or_ambiguous_legacy_terms():
    """The adapter never turns free-form local strings into new ontology terms."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    feature = _make_feature(db, semantic_storage=FakeCanonicalFactStorage())

    result = await feature.save_fact(
        subject="us\u00e9r", predicate="preferred_deploy_region", value="Berlin"
    )
    assert result.status is ToolResultStatus.ERROR
    assert "unsupported subject" in result.error

    result = await feature.save_fact(
        subject="urn:kestrel:agent:forged-tenant:principal:user",
        predicate="preferred_deploy_region",
        value="Berlin",
    )
    assert result.status is ToolResultStatus.ERROR
    assert "unsupported subject" in result.error

    result = await feature.save_fact(
        subject="user", predicate="https://example.test/adversarial", value="Berlin"
    )
    assert result.status is ToolResultStatus.ERROR
    assert "unsupported predicate" in result.error


@pytest.mark.asyncio
async def test_save_fact_keeps_unicode_as_a_typed_literal_not_an_iri():
    """Unicode values are explicit literal data, never inferred semantic terms."""
    from kestrel_sdk.tools.result import ToolResultStatus

    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)

    result = await feature.save_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value="東京-中央",
    )

    assert result.status is ToolResultStatus.OK
    assertion = canonical.put_calls[0][0]
    assert assertion.object.lexical_form == "東京-中央"
    assert assertion.object.datatype_iri == "http://www.w3.org/2001/XMLSchema#string"


@pytest.mark.asyncio
async def test_save_fact_surfaces_validation_rejection_and_unavailability_honestly():
    from kestrel_sdk.tools.result import ToolResultStatus

    db = FakeDB()
    rejected = _make_feature(
        db,
        semantic_storage=ValidationRejectedCanonicalFactStorage(),
    )
    result = await rejected.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )
    assert result.status is ToolResultStatus.ERROR
    assert result.data["assertion_id"] is None
    assert result.data["validation_disposition"] == "nonconformant:reject"

    unavailable = _make_feature(
        db,
        semantic_storage=ValidationUnavailableCanonicalFactStorage(),
    )
    result = await unavailable.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )
    assert result.status is ToolResultStatus.ERROR
    assert "SHACL profile is unavailable" in result.error


@pytest.mark.asyncio
async def test_forget_fact_uses_canonical_delete_for_the_current_adapter_fact():
    from kestrel_sdk.tools.result import ToolResultStatus

    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)
    await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )

    result = await feature.forget_fact("user", "preferred_deploy_region")

    assert result.status is ToolResultStatus.OK
    assert result.data["deleted"] is True
    assert len(canonical.delete_calls) == 1
    assert canonical.current == []


@pytest.mark.asyncio
async def test_forget_fact_refuses_a_current_foreign_lifecycle_target():
    """Matching local terms never let this adapter delete another producer's fact."""
    from kestrel_sdk.tools.result import ToolResultStatus

    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)
    await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="us-central1"
    )
    canonical.current = [
        replace(canonical.current[0], confidence_method="other-producer-v1")
    ]

    result = await feature.forget_fact("user", "preferred_deploy_region")

    assert result.status is ToolResultStatus.ERROR
    assert "outside save_fact" in result.error
    assert canonical.delete_calls == []


async def _finish_fact_after_refused_privacy_transition(
    canonical,
    task,
    entered,
    release,
):
    """Prove the fact lease wins before allowing a restrictive transition."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import (
        PrivacyViolationError,
    )

    await asyncio.wait_for(entered.wait(), timeout=1)
    wrapper = canonical._governed
    with pytest.raises(PrivacyViolationError, match="in flight"):
        wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.NORMAL

    release.set()
    result = await asyncio.wait_for(task, timeout=1)

    # Cancellation, errors, and ordinary returns all release the same lease.
    # This successful retry proves the completed path did not leak it.
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.EPHEMERAL
    return result


@pytest.mark.asyncio
async def test_privacy_transition_is_linearized_with_save_fact_replay():
    """A replay cannot return durable content after a successful mode flip."""
    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    first = await wrapper.save_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value="us-central1",
        confidence=0.9,
        invocation_id="privacy-lease-save-replay",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original = canonical._replay_governed_assertion_operation

    async def blocked_replay(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    canonical._replay_governed_assertion_operation = blocked_replay
    task = asyncio.create_task(
        wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="privacy-lease-save-replay",
        )
    )

    replay = await _finish_fact_after_refused_privacy_transition(
        canonical,
        task,
        entered,
        release,
    )
    assert replay.idempotent is True
    assert replay.assertion_id == first.assertion_id


@pytest.mark.asyncio
async def test_privacy_transition_is_linearized_with_legacy_terminalization():
    """The legacy-erasure check cannot race a later durable fact commit."""
    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    entered = asyncio.Event()
    release = asyncio.Event()
    original = (
        canonical._terminalize_legacy_erased_explicit_fact_operation
    )

    async def blocked_terminalization(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    canonical._terminalize_legacy_erased_explicit_fact_operation = (
        blocked_terminalization
    )
    task = asyncio.create_task(
        wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="privacy-lease-legacy-terminalization",
        )
    )

    result = await _finish_fact_after_refused_privacy_transition(
        canonical,
        task,
        entered,
        release,
    )
    assert result.saved is True
    assert len(canonical.put_calls) == 1


@pytest.mark.asyncio
async def test_privacy_transition_is_linearized_with_forget_fact_replay():
    """A delete replay cannot disclose identifiers after a successful flip."""
    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    await wrapper.save_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value="us-central1",
        confidence=0.9,
        invocation_id="privacy-lease-forget-source",
    )
    first = await wrapper.forget_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        invocation_id="privacy-lease-forget-replay",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original = canonical._replay_explicit_fact_forget_operation

    async def blocked_replay(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    canonical._replay_explicit_fact_forget_operation = blocked_replay
    task = asyncio.create_task(
        wrapper.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="privacy-lease-forget-replay",
        )
    )

    replay = await _finish_fact_after_refused_privacy_transition(
        canonical,
        task,
        entered,
        release,
    )
    assert replay.deleted is True
    assert replay.idempotent is True
    assert replay.assertion_id == first.assertion_id


@pytest.mark.asyncio
async def test_privacy_transition_is_linearized_with_absent_forget_noop():
    """An absent-target receipt is not persisted after a restrictive flip."""
    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    entered = asyncio.Event()
    release = asyncio.Event()
    original = canonical._record_explicit_fact_forget_noop

    async def blocked_noop(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    canonical._record_explicit_fact_forget_noop = blocked_noop
    task = asyncio.create_task(
        wrapper.forget_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            invocation_id="privacy-lease-absent-forget",
        )
    )

    result = await _finish_fact_after_refused_privacy_transition(
        canonical,
        task,
        entered,
        release,
    )
    assert result.deleted is False
    assert result.idempotent is False
    assert len(canonical.forget_noop_operations) == 1


@pytest.mark.asyncio
async def test_privacy_transition_is_linearized_with_fact_restoration():
    """Restoring a deleted fact remains within one privacy-mode epoch."""
    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    await wrapper.save_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        value="us-central1",
        confidence=0.9,
        invocation_id="privacy-lease-restore-source",
    )
    await wrapper.forget_explicit_fact(
        subject="user",
        predicate="preferred_deploy_region",
        invocation_id="privacy-lease-restore-delete",
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original = canonical._restore_explicit_fact_assertion

    async def blocked_restore(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    canonical._restore_explicit_fact_assertion = blocked_restore
    task = asyncio.create_task(
        wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="privacy-lease-restore-new-teaching",
        )
    )

    result = await _finish_fact_after_refused_privacy_transition(
        canonical,
        task,
        entered,
        release,
    )
    assert result.saved is True
    assert result.idempotent is False
    assert len(canonical.restore_calls) == 1


@pytest.mark.asyncio
async def test_cancelled_fact_operation_releases_privacy_transition_lease():
    """Cancellation cannot permanently block later privacy transitions."""
    from kestrel_sovereign.privacy import PrivacyMode

    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    entered = asyncio.Event()
    never_release = asyncio.Event()

    async def blocked_replay(*_args, **_kwargs):
        entered.set()
        await never_release.wait()

    canonical._replay_governed_assertion_operation = blocked_replay
    task = asyncio.create_task(
        wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="us-central1",
            confidence=0.9,
            invocation_id="privacy-lease-cancelled-save",
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)
    assert wrapper.privacy_mode is PrivacyMode.EPHEMERAL


def test_nested_fact_leases_release_exactly_once_before_transition():
    """Nested same-task leases cannot leak or release each other early."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import (
        PrivacyViolationError,
    )

    wrapper = FakeCanonicalFactStorage()._governed
    wrapper._acquire_explicit_fact_lease()
    wrapper._acquire_explicit_fact_lease()
    with pytest.raises(PrivacyViolationError, match="in flight"):
        wrapper.set_privacy_mode(PrivacyMode.ISOLATED)

    wrapper._release_explicit_fact_lease()
    with pytest.raises(PrivacyViolationError, match="in flight"):
        wrapper.set_privacy_mode(PrivacyMode.ISOLATED)

    wrapper._release_explicit_fact_lease()
    wrapper.set_privacy_mode(PrivacyMode.ISOLATED)
    assert wrapper.privacy_mode is PrivacyMode.ISOLATED


@pytest.mark.asyncio
async def test_restrictive_transition_wins_before_fact_lease_acquisition():
    """If the mode flip linearizes first, fact code never reaches raw storage."""
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import (
        PrivacyViolationError,
    )

    canonical = FakeCanonicalFactStorage()
    wrapper = canonical._governed
    wrapper.set_privacy_mode(PrivacyMode.EPHEMERAL)

    with pytest.raises(PrivacyViolationError, match="durable"):
        await wrapper.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="do-not-persist",
            confidence=0.9,
            invocation_id="privacy-transition-won-first",
        )

    assert canonical.put_calls == []
    assert canonical.operations == {}


# --------------------------------------------------------------------------
# Privacy-gating regression tests (F212 re-pin trash, F213 save_fact gating)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repin_after_trash_is_refused():
    """A soft-deleted (Trash) message cannot be re-pinned (F212)."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    msg_id = db.add_message("Secret to be erased")
    feature = _make_feature(db)

    # Pin, then release, then trash the message (as the wrapper would on delete).
    await feature.memory_pin(message_id=msg_id, reason="temp")
    await feature.memory_release(message_id=msg_id)
    db.trash_message(msg_id)

    pins_before = dict(db.pins)

    result = await feature.memory_pin(message_id=msg_id, reason="resurrect")

    assert result.status is ToolResultStatus.ERROR
    assert "not found" in result.error.lower()
    # No new pin record created for the trashed message.
    active = [
        p for p in db.pins.values()
        if p["message_id"] == msg_id and p["released_at"] is None
    ]
    assert active == []
    # The released pin record from before is untouched (no resurrection).
    assert db.pins == pins_before


@pytest.mark.asyncio
async def test_memory_pinned_excludes_trashed_pin():
    """memory_pinned must never surface a trashed row's content (F212)."""
    from kestrel_sdk.tools.result import ToolResultStatus
    db = FakeDB()
    live = db.add_message("Live memory")
    doomed = db.add_message("Content that gets trashed")
    feature = _make_feature(db)

    await feature.memory_pin(message_id=live, reason="keep")
    await feature.memory_pin(message_id=doomed, reason="keep")

    # Message is soft-deleted after being pinned.
    db.trash_message(doomed)

    result = await feature.memory_pinned()

    assert result.status is ToolResultStatus.OK
    surfaced = {p["message_id"] for p in result.data["pins"]}
    assert live in surfaced
    assert doomed not in surfaced


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_mode", ["none", "temp", "deidentified"])
async def test_save_fact_blocked_in_volatile_privacy_modes(storage_mode):
    """EPHEMERAL, ISOLATED, and DEIDENTIFIED never reach semantic storage."""
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.privacy import PrivacyConfig

    db = FakeDB()
    canonical = FakeCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)
    feature.agent.privacy_config = PrivacyConfig(storage=storage_mode, llm_location="local")

    result = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="do-not-persist"
    )

    assert result.status is ToolResultStatus.ERROR
    assert "privacy mode" in result.error
    assert canonical.put_calls == []


@pytest.mark.asyncio
async def test_save_fact_anonymous_mode_fails_closed_without_a_redacted_assertion_pipeline():
    """String redaction is not a semantics-preserving canonical transformation."""
    from kestrel_sdk.tools.result import ToolResultStatus
    from kestrel_sovereign.privacy import PrivacyConfig

    db = FakeDB()
    canonical = PrivacyBlockedCanonicalFactStorage()
    feature = _make_feature(db, semantic_storage=canonical)
    feature.agent.privacy_config = PrivacyConfig(
        storage="pii_redacted", llm_location="local"
    )

    result = await feature.save_fact(
        subject="user", predicate="preferred_deploy_region", value="jane@example.com"
    )

    assert result.status is ToolResultStatus.ERROR
    assert "privacy" in result.error
    assert canonical.put_calls == []
