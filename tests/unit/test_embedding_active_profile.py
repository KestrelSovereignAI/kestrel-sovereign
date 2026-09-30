"""``kestrel embeddings reindex`` targets the profile the agent searches (#3420).

On the fleet, ``ollama:local`` is a member of the shared embedding space
``qwen3-embedding-8b@768``, whose parity probe passed and is recorded in
``embedding_profiles``. The running agent rehydrates that at boot and stamps
and searches profile ``7685e6844167``. The reindex CLI never rehydrated it, so
it resolved the Ollama adapter's own space id ``qwen3-embedding:8b@768``
(profile ``bea8bae1ec47``), counted every row as stale, and rewrote all of
them onto it. Vector search then found nothing.

Covered here:

- the CLI re-applies the verified shared spaces the database records, so it
  resolves the agent's profile (the root cause);
- the agent records the profile it resolves, and ``reindex`` refuses any other
  target, in a dry-run and with ``--yes``, without writing anything; onto the
  recorded profile it re-embeds only the stale rows;
- with no record, ``reindex`` refuses while stored vectors sit on another
  profile.
"""

from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from kestrel_sovereign import cli_embeddings
from kestrel_sovereign.agent.model_preference import ModelPreferenceMixin
from kestrel_sovereign.llm.embedding_service import derive_embedding_profile
from kestrel_sovereign.llm.embedding_space import EmbeddingSpacePin
from kestrel_sovereign.storage.active_embedding_profile import (
    ACTIVE_EMBEDDING_PROFILE_KEY,
    ActiveEmbeddingProfileError,
    load_active_embedding_profiles,
    record_active_embedding_profile,
)
from kestrel_sovereign.storage.async_database import AsyncDatabase
from kestrel_sovereign.storage.sqla.embedding_profile import (
    _clear_profile_upsert_cache_for_tests,
    record_space_parity,
)
from tests.utils.process_local_llm_service import process_local_service

AGENT = "did:test:emma"
PEER = "did:test:claw"
DIM = 4

# The two profiles of the incident, as the fleet derived them.
SHARED_SPACE_PROFILE = "7685e6844167"
ROUTE_SCOPED_PROFILE = "bea8bae1ec47"


class _Service:
    """An embedding service that describes itself and records what it embeds."""

    def __init__(self, model: str, *, provider: str = "ollama") -> None:
        self.profile = derive_embedding_profile(
            provider=provider, model=model, dim=DIM
        )
        self.embedding_dim = DIM
        self.embedded: list[str] = []

    def describe(self):
        return self.profile

    def current_profile_id(self) -> str:
        return self.profile.profile_id

    async def aembed_batch(self, texts):
        self.embedded.extend(texts)
        return [[float(len(t) % 5) + i for i in range(DIM)] for t in texts]


def _vector(seed: float) -> bytes:
    return struct.pack(f"<{DIM}f", *[seed + i for i in range(DIM)])


@pytest.fixture
async def db(tmp_path):
    _clear_profile_upsert_cache_for_tests()
    database = await AsyncDatabase.sqlite(str(tmp_path / "kestrel_prime.db"))
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture(autouse=True)
def _column_dim(monkeypatch):
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: DIM)


async def _add_message(db, content, profile_id, vector=None, *, agent=AGENT):
    await db.execute_commit(
        "INSERT INTO conversation_history "
        "(agent_id, role, content, embedding_vec, embedding_profile_id) "
        "VALUES (?, 'user', ?, ?, ?)",
        (agent, content, vector, profile_id),
    )


async def _seed_corpus(db, profile_id):
    """Rows the agent can find on *profile_id*, plus one never embedded."""
    await _add_message(db, "found one", profile_id, _vector(1.0))
    await _add_message(db, "found two", profile_id, _vector(2.0))
    await _add_message(db, "never embedded", None)
    await db.execute_commit(
        "INSERT INTO saved_items (id, agent_id, item_type, name, summary, "
        "content, embedding_vec, embedding_profile_id) "
        "VALUES ('s1', ?, 'note', 'n', 'saved summary', 'c', ?, ?)",
        (AGENT, _vector(3.0), profile_id),
    )
    await db.execute_commit(
        "INSERT INTO document_chunks (file_hash, content, embedding_vec, "
        "embedding_profile_id) VALUES ('doc', 'chunk text', ?, ?)",
        (_vector(4.0), profile_id),
    )


async def _snapshot(db):
    return {
        "conversation_history": await db.fetchall(
            "SELECT id, content, embedding_vec, embedding_profile_id "
            "FROM conversation_history ORDER BY id"
        ),
        "saved_items": await db.fetchall(
            "SELECT id, embedding_vec, embedding_profile_id "
            "FROM saved_items ORDER BY id"
        ),
        "document_chunks": await db.fetchall(
            "SELECT chunk_id, embedding_vec, embedding_profile_id "
            "FROM document_chunks ORDER BY chunk_id"
        ),
        "embedding_profiles": await db.fetchall(
            "SELECT id FROM embedding_profiles ORDER BY id"
        ),
    }


async def _reindex(db, service, *, apply, agent_id=None, llm_service=None):
    injected = {}
    if service is not None:
        injected = {
            "embedding_service": service,
            "target_profile_id": service.current_profile_id(),
            "target_dim": DIM,
        }
    return await cli_embeddings._reindex(
        db,
        table=None,
        agent_id=agent_id,
        batch=10,
        rate_limit=0.0,
        dry_run=not apply,
        apply=apply,
        llm_service=llm_service,
        **injected,
    )


# ------------------------------------------------ the recorded active profile


async def test_a_recorded_profile_round_trips(db):
    service = _Service("qwen3")

    assert await record_active_embedding_profile(db, AGENT, service) == (
        service.current_profile_id()
    )

    lookup = await load_active_embedding_profiles(db, AGENT)
    (record,) = lookup.records
    assert lookup.profile_id == service.current_profile_id()
    assert not lookup.ambiguous
    assert (record.agent_id, record.provider, record.model, record.dim) == (
        AGENT, "ollama", "qwen3", DIM,
    )
    assert record.recorded_at


@pytest.mark.parametrize("service", [None, SimpleNamespace(describe=lambda: None)])
async def test_nothing_is_recorded_without_a_profile(db, service):
    await record_active_embedding_profile(db, AGENT, _Service("previous"))

    assert await record_active_embedding_profile(db, AGENT, service) is None

    # The last profile the agent's vectors were written in stays recorded.
    lookup = await load_active_embedding_profiles(db, AGENT)
    assert lookup.profile_id == _Service("previous").current_profile_id()


async def test_agents_recording_different_profiles_are_ambiguous(db):
    emma, claw = _Service("qwen3"), _Service("nomic")
    await record_active_embedding_profile(db, AGENT, emma)
    await record_active_embedding_profile(db, PEER, claw)

    everyone = await load_active_embedding_profiles(db)
    assert everyone.ambiguous
    assert everyone.profile_id is None
    assert (await load_active_embedding_profiles(db, AGENT)).profile_id == (
        emma.current_profile_id()
    )


@pytest.mark.parametrize("value", ["not json", json.dumps({"model": "m"})])
async def test_a_malformed_record_raises(db, value):
    await db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        (AGENT, ACTIVE_EMBEDDING_PROFILE_KEY, value),
    )

    with pytest.raises(ActiveEmbeddingProfileError, match=AGENT):
        await load_active_embedding_profiles(db)


# ------------------------------------------------------- the reindex guard


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "yes"])
async def test_reindex_refuses_a_target_other_than_the_recorded_profile(
    db, capsys, apply
):
    active, resolved = _Service("shared-space"), _Service("route-scoped")
    await _seed_corpus(db, active.current_profile_id())
    await record_active_embedding_profile(db, AGENT, active)
    before = await _snapshot(db)

    rc = await _reindex(db, resolved, apply=apply)

    assert rc == 2
    out, err = capsys.readouterr()
    assert f"target_profile: {resolved.current_profile_id()}" in out
    assert f"active_profile: {active.current_profile_id()}" in out
    assert (
        f"is not the profile the agent records as active, "
        f"{active.current_profile_id()}"
    ) in err
    assert resolved.embedded == []
    assert await _snapshot(db) == before


async def test_reindex_onto_the_recorded_profile_rewrites_only_stale_rows(
    db, capsys
):
    active, previous = _Service("shared-space"), _Service("previous-model")
    await _seed_corpus(db, active.current_profile_id())
    await _add_message(db, "old model", previous.current_profile_id(), _vector(9.0))
    await record_active_embedding_profile(db, AGENT, active)
    before = await _snapshot(db)

    rc = await _reindex(db, active, apply=True)

    assert rc == 0
    assert f"active_profile: {active.current_profile_id()}" in capsys.readouterr().out
    assert sorted(active.embedded) == ["never embedded", "old model"]
    after = await _snapshot(db)
    target = active.current_profile_id()
    for table in ("conversation_history", "saved_items", "document_chunks"):
        assert {row[-1] for row in after[table]} == {target}
    # Rows already on the profile keep the exact vector they had.
    kept = [row for row in before["conversation_history"] if row[-1] == target]
    assert kept and all(row in after["conversation_history"] for row in kept)
    assert after["saved_items"] == before["saved_items"]
    assert after["document_chunks"] == before["document_chunks"]


async def test_without_a_record_reindex_refuses_while_vectors_are_elsewhere(
    db, capsys
):
    on_disk, resolved = _Service("shared-space"), _Service("route-scoped")
    await _seed_corpus(db, on_disk.current_profile_id())
    before = await _snapshot(db)

    rc = await _reindex(db, resolved, apply=True)

    assert rc == 2
    out, err = capsys.readouterr()
    assert "active_profile: (none recorded" in out
    assert "no active embedding profile is recorded" in err
    assert f"4 stored vector(s) are on other profile(s): {on_disk.current_profile_id()} (4)" in err
    assert "Start the agent once" in err
    assert resolved.embedded == []
    assert await _snapshot(db) == before


async def test_without_a_record_reindex_embeds_rows_that_have_no_vector(db, capsys):
    # The #3415 constitution chunks: nothing embedded, nothing to strand.
    service = _Service("qwen3")
    await _add_message(db, "never embedded", None)
    await db.execute_commit(
        "INSERT INTO document_chunks (file_hash, content) VALUES ('c', 'chunk')"
    )

    rc = await _reindex(db, service, apply=True)

    assert rc == 0
    assert sorted(service.embedded) == ["chunk", "never embedded"]


async def test_reindex_refuses_when_agents_record_different_profiles(db, capsys):
    emma, claw = _Service("qwen3"), _Service("nomic")
    await _add_message(db, "emma row", None)
    await _add_message(db, "claw row", claw.current_profile_id(), _vector(1.0), agent=PEER)
    await record_active_embedding_profile(db, AGENT, emma)
    await record_active_embedding_profile(db, PEER, claw)
    before = await _snapshot(db)

    assert await _reindex(db, emma, apply=True) == 2
    assert "record different active embedding profiles" in capsys.readouterr().err
    assert await _snapshot(db) == before

    # Scoped to one agent, that agent's record decides and its rows alone move.
    assert await _reindex(db, emma, apply=True, agent_id=AGENT) == 0
    assert emma.embedded == ["emma row"]
    rows = dict(
        (row[1], row[3]) for row in (await _snapshot(db))["conversation_history"]
    )
    assert rows == {
        "emma row": emma.current_profile_id(),
        "claw row": claw.current_profile_id(),
    }


async def test_reindex_refuses_when_the_record_cannot_be_read(db, capsys):
    service = _Service("qwen3")
    await _add_message(db, "never embedded", None)
    await db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        (AGENT, ACTIVE_EMBEDDING_PROFILE_KEY, "not json"),
    )

    assert await _reindex(db, service, apply=True) == 2
    assert "could not read the active embedding profile" in capsys.readouterr().err
    assert service.embedded == []


# --------------------------------------- the root cause: shared-space rehydration


class _OllamaShapedAdapter:
    """Declares its space from the served model slug, as ``OllamaAdapter`` does."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embedding_space_id(self):
        return "qwen3-embedding:8b@768"

    async def aembed_batch(self, client, texts, model=None, **kwargs):
        self.embedded.extend(texts)
        return [[float(len(t) % 5) + i for i in range(768)] for t in texts]


FLEET_PIN = EmbeddingSpacePin(
    name="qwen3",
    model="qwen3-embedding-8b",
    dim=768,
    members=("ollama:local", "openrouter:api"),
)


def _fleet_service(adapter):
    """The fleet's static config: a shared-space member that embeds."""
    route = {
        "name": "ollama:local",
        "vendor": "ollama",
        "route": "local",
        "adapter": adapter,
        "client": object(),
        "model": "auto",
        "is_local": True,
        "is_cloud": False,
        "capabilities": {
            "supports_embeddings": True,
            "embedding_model": "qwen3-embedding:8b",
            "embedding_dim": 768,
        },
    }
    return process_local_service([route], embedding_space_pins=[FLEET_PIN])


async def _persist_fleet_state(db, *, parity_verified):
    await db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        (AGENT, "embedding_route", json.dumps("ollama:local")),
    )
    if parity_verified:
        # What the agent's verify endpoint persisted when the probe passed.
        await record_space_parity(
            db,
            space_id=FLEET_PIN.space_id,
            model=FLEET_PIN.model,
            dim=FLEET_PIN.dim,
            normalized=FLEET_PIN.normalized,
            parity_cosine=0.995,
        )


@pytest.mark.parametrize(
    ("parity_verified", "expected"),
    [(True, SHARED_SPACE_PROFILE), (False, ROUTE_SCOPED_PROFILE)],
    ids=["verified-space", "unverified-space"],
)
async def test_the_cli_resolves_the_shared_space_the_database_verified(
    db, parity_verified, expected
):
    service = _fleet_service(_OllamaShapedAdapter())
    await _persist_fleet_state(db, parity_verified=parity_verified)

    assert await cli_embeddings._apply_persisted_embedding_config(
        service, db, AGENT
    ) is None
    err, _, target = cli_embeddings._resolve_target(service)

    assert err is None
    assert target == expected


class _DiscoveringAdapter:
    """Discovers two models; catalog order lists the corpus's model second."""

    async def list_embedding_models(self, client):
        from kestrel_sovereign.llm.embedding_discovery import EmbeddingModelInfo

        return [
            EmbeddingModelInfo(id="nomic-embed-text", provider="ollama", native_dim=DIM),
            EmbeddingModelInfo(id="qwen3-embedding:8b", provider="ollama", native_dim=DIM),
        ]


async def test_the_cli_resolves_an_unpinned_route_to_the_corpus_model(db):
    # The agent registers the corpus's dominant profile before it reconciles
    # discovery (#2366), so an unpinned route keeps the space the corpus is in.
    corpus = derive_embedding_profile(
        provider="ollama", model="qwen3-embedding:8b", dim=DIM
    )
    await db.execute_commit(
        "INSERT INTO embedding_profiles (id, provider, model, dim, space_id, "
        "normalized) VALUES (?, 'ollama', 'qwen3-embedding:8b', ?, ?, 0)",
        (corpus.profile_id, DIM, corpus.space_id),
    )
    await _add_message(db, "old memory", corpus.profile_id, _vector(1.0))
    await db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        (AGENT, "embedding_route", json.dumps("ollama:local")),
    )
    service = process_local_service([{
        "name": "ollama:local",
        "vendor": "ollama",
        "route": "local",
        "adapter": _DiscoveringAdapter(),
        "client": object(),
        "model": "auto",
        "is_local": True,
        "is_cloud": False,
        "capabilities": {},
    }])
    service._embedding_discovery_cache = None  # discover through the adapter

    assert await cli_embeddings._apply_persisted_embedding_config(
        service, db, AGENT
    ) is None
    err, _, target = cli_embeddings._resolve_target(service)

    assert err is None
    assert target == corpus.profile_id


async def test_the_fleet_reindex_finds_the_corpus_already_on_its_profile(
    db, monkeypatch, capsys
):
    """The incident, end to end, with the real resolution path."""
    monkeypatch.setattr(cli_embeddings, "_resolve_column_dim", lambda: 768)
    adapter = _OllamaShapedAdapter()
    monkeypatch.setattr(
        "kestrel_sovereign.llm.service.LLMService",
        lambda *args, **kwargs: _fleet_service(adapter),
    )
    await _persist_fleet_state(db, parity_verified=True)
    await db.execute_commit(
        "INSERT INTO agent_metadata (agent_id, key, value) VALUES (?, ?, ?)",
        (AGENT, ACTIVE_EMBEDDING_PROFILE_KEY, json.dumps({
            "profile_id": SHARED_SPACE_PROFILE,
            "provider": f"shared:{FLEET_PIN.space_id}",
            "model": FLEET_PIN.model,
            "dim": 768,
        })),
    )
    await _add_message(
        db, "found", SHARED_SPACE_PROFILE, struct.pack("<768f", *[0.5] * 768)
    )
    before = await _snapshot(db)

    rc = await _reindex(db, None, apply=True, agent_id=AGENT)

    assert rc == 0
    out = capsys.readouterr().out
    assert f"target_profile: {SHARED_SPACE_PROFILE}" in out
    assert f"active_profile: {SHARED_SPACE_PROFILE}" in out
    assert "Nothing to do" in out
    assert adapter.embedded == []
    assert await _snapshot(db) == before


# ---------------------------------------------------- the agent's own record


class _Agent(ModelPreferenceMixin):
    """The state ``record_active_embedding_profile`` reads from an agent."""

    def __init__(self, db, service, *, allows_cloud=True) -> None:
        self._raw_storage = SimpleNamespace(db=db)
        self.agent_id = AGENT
        self.llm_service = SimpleNamespace(get_embedding_service=lambda: service)
        self.privacy_agent = SimpleNamespace(
            privacy_config=SimpleNamespace(allows_cloud_llm=lambda: allows_cloud)
        )


async def test_the_agent_records_the_profile_it_resolves(db):
    service = _Service("qwen3")

    await _Agent(db, service).record_active_embedding_profile()

    lookup = await load_active_embedding_profiles(db, AGENT)
    assert lookup.profile_id == service.current_profile_id()


async def test_a_local_only_privacy_mode_records_nothing(db):
    # Its forced route writes no durable rows and does not describe the corpus.
    await _Agent(db, _Service("local-only"), allows_cloud=False).record_active_embedding_profile()

    assert (await load_active_embedding_profiles(db)).records == ()


@pytest.mark.parametrize(
    ("callback", "args"),
    [
        ("_persist_embedding_route", ("ollama:local",)),
        ("_persist_route_embedding_models", ({"ollama:local": {"model": "m"}},)),
        ("_persist_model_preference", ("m", "ollama", "local")),
    ],
)
async def test_a_persisted_embedding_change_records_the_new_profile(
    db, callback, args
):
    before, after = _Service("before"), _Service("after")
    await record_active_embedding_profile(db, AGENT, before)

    await getattr(_Agent(db, after), callback)(*args)

    lookup = await load_active_embedding_profiles(db, AGENT)
    assert lookup.profile_id == after.current_profile_id()


async def test_a_parity_verification_records_the_profile_it_now_resolves(db):
    # A pin that passes moves a member route onto the shared space's profile.
    from kestrel_sovereign.endpoints.models import verify_embedding_space

    shared = _Service("shared-space")
    await record_active_embedding_profile(db, AGENT, _Service("route-scoped"))

    class _ParityLLM:
        def get_embedding_service(self):
            return shared

        async def verify_embedding_space_parity(self, pin_name, *, record_to=None):
            return {}

    agent = _Agent(db, shared)
    agent.llm_service = _ParityLLM()
    agent.storage = SimpleNamespace(db=db, agent_id=AGENT)

    class _Request:
        state = SimpleNamespace(agent=agent)

        async def json(self):
            return {}

    assert (await verify_embedding_space(_Request()))["success"] is True
    lookup = await load_active_embedding_profiles(db, AGENT)
    assert lookup.profile_id == shared.current_profile_id()
