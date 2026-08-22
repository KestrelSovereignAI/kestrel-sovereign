"""Which database names the agent at server startup (#2472, #2894).

Identity is born in the anchor — ``agent_data/<Name>/kestrel_prime.db`` — on
every backend, and governance lives in whatever database the runtime resolves.
#2472 pointed this resolver at the durable database on PostgreSQL, for a
container whose disk carries no identity. That inverted the two on every
*other* PostgreSQL host: ``kestrel create`` writes the birth record to the
anchor, the runtime database has no tables at all until first boot, and the
replication that fills it (#2871) runs inside ``KestrelAgent.initialize()`` —
downstream of this gate. So the gate refused, the boot that would have repaired
the gap never happened, and ``kestrel start`` reported 503 for its whole
window. Reproduced on ``171355ea`` against a real pgvector container.

Both readers still exist; they answer different questions, and when they
disagree about who this agent is that is a custody failure, not a tie to break.
"""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_sovereign import main as main_module


ANCHORED_DID = "did:web:agents.example.com:kestrel"
NEIGHBOUR_DID = "did:web:agents.example.com:someone-else"
DSN = "postgresql://durable.example/kestrel"


def _write_anchor(agent_dir: Path, *dids: str) -> Path:
    """Write the birth record the way inception leaves it.

    Stock ``sqlite3`` and only the columns the reader touches — the reader
    itself deliberately avoids the AsyncStorage stack so that a lookup cannot
    create, migrate, or otherwise write the directory it is inspecting.
    """
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "kestrel_prime.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, node_type TEXT)"
        )
        connection.executemany(
            "INSERT INTO graph_nodes (node_id, node_type) VALUES (?, 'agent')",
            [(did,) for did in dids],
        )
        connection.commit()
    finally:
        connection.close()
    return db_path


class _FakeStorage:
    """The runtime PostgreSQL, holding whatever ``nodes`` is set to."""

    instances = []
    nodes = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.initialized = False
        self.closed = False
        self.__class__.instances.append(self)

    async def initialize(self):
        self.initialized = True

    async def get_nodes_by_type(self, node_type):
        assert node_type == "agent"
        return self.__class__.nodes

    async def close(self):
        self.closed = True


@pytest.fixture
def runtime_db(monkeypatch):
    def _seed(*dids):
        _FakeStorage.instances.clear()
        _FakeStorage.nodes = [SimpleNamespace(node_id=did) for did in dids]
        return _FakeStorage

    monkeypatch.setattr(main_module, "AsyncStorage", _FakeStorage)
    return _seed


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_freshly_incepted_agent_boots_before_its_record_reaches_postgres(
    tmp_path, runtime_db
):
    """#2894. This is the state ``kestrel create`` leaves on a PostgreSQL host:
    a complete birth record in the anchor and an empty runtime database, whose
    schema does not exist until the agent boots. Refusing here refuses the only
    thing that can close the gap."""
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)
    runtime_db()  # no rows — nothing has ever booted against this database

    did = await main_module.get_agent_did_async(
        str(agent_dir), db_backend="postgres", database_url=DSN
    )

    assert did == ANCHORED_DID


@pytest.mark.asyncio
async def test_the_anchor_and_the_runtime_database_agreeing_is_the_steady_state(
    tmp_path, runtime_db
):
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)
    runtime_db(ANCHORED_DID)

    did = await main_module.get_agent_did_async(
        str(agent_dir), db_backend="postgres", database_url=DSN
    )

    assert did == ANCHORED_DID


@pytest.mark.asyncio
async def test_an_anchor_naming_a_different_agent_than_the_database_refuses(
    tmp_path, runtime_db
):
    """Booting this directory's identity against another agent's governance is
    the whole "wrong database" class. Neither side is authoritative over the
    other, so name both and stop."""
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)
    runtime_db(NEIGHBOUR_DID)

    with pytest.raises(ValueError) as excinfo:
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=DSN
        )

    message = str(excinfo.value)
    assert ANCHORED_DID in message
    assert NEIGHBOUR_DID in message


# ---------------------------------------------------------------------------
# The case #2472 added the PostgreSQL branch for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_container_with_no_anchor_still_reads_the_durable_database(
    tmp_path, runtime_db
):
    """An ephemeral disk carries no identity; the durable database is the only
    place it can be."""
    runtime_db(ANCHORED_DID)

    did = await main_module.get_agent_did_async(
        str(tmp_path / "disposable"), db_backend="postgres", database_url=DSN
    )

    assert did == ANCHORED_DID
    storage = _FakeStorage.instances[-1]
    assert storage.args == ()
    assert storage.kwargs == {"backend": "postgres", "dsn": DSN}
    assert storage.initialized and storage.closed


@pytest.mark.asyncio
async def test_no_anchor_and_an_empty_database_is_an_unincepted_host(
    tmp_path, runtime_db
):
    runtime_db()

    with pytest.raises(ValueError, match="No agent found in the database"):
        await main_module.get_agent_did_async(
            str(tmp_path / "disposable"), db_backend="postgres", database_url=DSN
        )


# ---------------------------------------------------------------------------
# Custody
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_agents_in_one_database_refuse_with_no_anchor(
    tmp_path, runtime_db
):
    runtime_db(ANCHORED_DID, NEIGHBOUR_DID)

    with pytest.raises(ValueError, match="exactly one agent node"):
        await main_module.get_agent_did_async(
            str(tmp_path / "disposable"), db_backend="postgres", database_url=DSN
        )

    assert _FakeStorage.instances[-1].closed


@pytest.mark.asyncio
async def test_two_agents_in_one_database_refuse_even_when_the_anchor_names_one(
    tmp_path, runtime_db
):
    """A resolvable identity is not a dedicated database. The custody rule is
    about the database's tenancy, and knowing which of the two this directory
    is does not make the other one belong there."""
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)
    runtime_db(ANCHORED_DID, NEIGHBOUR_DID)

    with pytest.raises(ValueError, match="exactly one agent node"):
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=DSN
        )


@pytest.mark.asyncio
async def test_postgres_identity_discovery_requires_dsn(monkeypatch, tmp_path):
    monkeypatch.delenv("KESTREL_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="KESTREL_DATABASE_URL is required"):
        await main_module.get_agent_did_async(
            str(tmp_path / "disposable"), db_backend="postgres"
        )


# ---------------------------------------------------------------------------
# SQLite: the anchor and the runtime database are the same file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_reads_the_anchor(tmp_path):
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)

    did = await main_module.get_agent_did_async(
        str(agent_dir), db_backend="sqlite"
    )

    assert did == ANCHORED_DID


@pytest.mark.asyncio
async def test_sqlite_refuses_a_missing_anchor_instead_of_creating_one(tmp_path):
    """``AsyncStorage.initialize()`` is write-capable: on a missing path it
    creates the database, WAL, audit tables and schema. Startup asking "who is
    this agent" must not be able to answer by *making* an empty one, which then
    blocks the real inception."""
    agent_dir = tmp_path / "agent_data" / "NeverIncepted"
    agent_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="No agent found"):
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="sqlite"
        )

    assert not (agent_dir / "kestrel_prime.db").exists()


@pytest.mark.asyncio
async def test_sqlite_refuses_a_directory_holding_two_agent_roots(tmp_path):
    """Picking one by incidental row order would authorize the wrong tenant."""
    agent_dir = tmp_path / "agent_data" / "Ambiguous"
    _write_anchor(agent_dir, ANCHORED_DID, NEIGHBOUR_DID)

    with pytest.raises(ValueError, match="invalid agent root set"):
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="sqlite"
        )


# ---------------------------------------------------------------------------
# An anchor that is present but unreadable is not "no anchor"
#
# Only absence lets a caller answer from somewhere else. A corrupt file, a
# permission denial, or two agent roots all mean an anchor IS there and this
# process could not be told what it says — and falling through to the runtime
# database there boots this directory as whichever agent that database holds.
# On SQLite every one of these states is refused loudly; the PostgreSQL path
# must not be weaker for exactly the failures the reader exists to catch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_corrupt_anchor_refuses_rather_than_adopting_a_neighbour(
    tmp_path, runtime_db
):
    """`kestrel start Nellie` where the anchor is truncated by a killed
    inception, or root-owned from a `sudo kestrel create`. Nellie's process
    must not boot as Emma."""
    agent_dir = tmp_path / "agent_data" / "Nellie"
    agent_dir.mkdir(parents=True)
    (agent_dir / "kestrel_prime.db").write_bytes(b"this is not a database")
    runtime_db(NEIGHBOUR_DID)

    with pytest.raises(ValueError, match="Could not read local agent identity"):
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=DSN
        )


@pytest.mark.asyncio
async def test_an_ambiguous_anchor_refuses_rather_than_adopting_a_neighbour(
    tmp_path, runtime_db
):
    """Two agent roots is an integrity failure, not an absence. Answering from
    the runtime database here would pick a third party entirely."""
    agent_dir = tmp_path / "agent_data" / "Ambiguous"
    _write_anchor(agent_dir, ANCHORED_DID, "did:web:agents.example.com:second")
    runtime_db(NEIGHBOUR_DID)

    with pytest.raises(ValueError, match="invalid agent root set"):
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=DSN
        )


@pytest.mark.asyncio
async def test_only_a_genuinely_absent_anchor_defers_to_the_database(
    tmp_path, runtime_db
):
    """The control: the ephemeral-container case still works, so the tests
    above are pinning the distinction rather than a blanket refusal."""
    runtime_db(NEIGHBOUR_DID)

    did = await main_module.get_agent_did_async(
        str(tmp_path / "never-incepted"), db_backend="postgres", database_url=DSN
    )

    assert did == NEIGHBOUR_DID


@pytest.mark.asyncio
async def test_the_conflict_message_names_a_remedy_that_exists(
    tmp_path, runtime_db
):
    """The standalone launcher takes one host-wide KESTREL_DATABASE_URL — there
    is no per-agent DSN to point anywhere. Telling an operator to point it at
    this agent's own database is an instruction they cannot follow; the
    in-process host is the one that runs a fleet against one PostgreSQL."""
    agent_dir = tmp_path / "agent_data" / "Kite"
    _write_anchor(agent_dir, ANCHORED_DID)
    runtime_db(NEIGHBOUR_DID)

    with pytest.raises(ValueError) as excinfo:
        await main_module.get_agent_did_async(
            str(agent_dir), db_backend="postgres", database_url=DSN
        )

    message = str(excinfo.value)
    assert "kestrel start" in message
    assert "#2843" in message
