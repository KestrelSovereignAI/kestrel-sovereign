"""Doctor closes its actual native SQLite readers on success and failure."""
from contextlib import closing
import sqlite3

import pytest

from kestrel_sovereign import doctor


@pytest.mark.parametrize("reader", ["discovery", "ownership", "rows"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_doctor_closes_original_native_sqlite_reader(tmp_path, monkeypatch, reader, corrupt):
    path = tmp_path / "anchor.db"
    did = "did:test:doctor-native-close"
    if corrupt:
        path.write_bytes(b"not a SQLite database")
    else:
        with closing(sqlite3.connect(path)) as seed, seed:
            seed.execute("CREATE TABLE graph_nodes (node_id TEXT, node_type TEXT)")
            seed.execute("INSERT INTO graph_nodes VALUES (?, 'agent')", (did,))
    original_bytes = path.read_bytes()
    source = doctor._GovernanceSource(anchor_path=path, agent_did=did)
    connect, opened = sqlite3.connect, []

    def native_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        assert type(connection) is sqlite3.Connection
        opened.append(connection)
        return connection

    def read():
        if reader == "discovery":
            return doctor._discover_agent_did(path)
        if reader == "ownership":
            return doctor._has_ownership_ledger(source)
        return doctor._fetch_rows(source, "SELECT node_id FROM graph_nodes", "SELECT 1")

    try:
        with monkeypatch.context() as fault:
            fault.setattr(doctor.sqlite3, "connect", native_connect)
            if corrupt and reader == "rows":
                with pytest.raises(sqlite3.DatabaseError):
                    read()
            else:
                result = read()
                if corrupt:
                    assert isinstance(result, doctor._UnreadableDB)
                elif reader == "discovery":
                    assert result == did
                elif reader == "ownership":
                    assert result == doctor._LedgerState(present=False, settled=False)
                else:
                    assert result == [(did,)]
        assert len(opened) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            opened[0].execute("SELECT 1")
        assert path.read_bytes() == original_bytes
    finally:
        for connection in opened:
            connection.close()
