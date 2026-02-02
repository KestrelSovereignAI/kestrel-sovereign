import os
from pathlib import Path

import pytest

from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.retirement_service import retire_test_agent


@pytest.mark.asyncio
async def test_retirement_archives_under_agent_data_dir(tmp_path, monkeypatch):
    """Default retirement archive should be writable in Docker (/data) and local runs.

    Concretely: if no archive_dir is provided, we archive alongside the DB,
    not under the code directory.
    """
    monkeypatch.delenv("KESTREL_DB_PATH", raising=False)

    await create_kestrel_identity_async(
        output_dir=str(tmp_path),
        is_test_instance=True,
        agent_name="Test-Retirement",
    )

    db_path = tmp_path / "kestrel_prime.db"
    assert db_path.exists()

    record = await retire_test_agent(str(db_path), reason="testing_complete")

    archive_path = Path(record.archive_path)
    assert archive_path.exists()
    assert archive_path.parent.name == "retired_agents"
    assert archive_path.parent.parent.name == "archive"

    # DB moved into archive folder
    assert not db_path.exists()
    assert (archive_path / "kestrel_prime.db").exists()

    # Retirement record written
    assert (archive_path / "RETIREMENT_RECORD.json").exists()


@pytest.mark.asyncio
async def test_retirement_prefers_kestrel_db_path_env(tmp_path, monkeypatch):
    """If KESTREL_DB_PATH is set, archives should go under it by default."""
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path))

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        is_test_instance=True,
        agent_name="Test-Retirement-Env",
    )

    db_path = agent_dir / "kestrel_prime.db"
    assert db_path.exists()

    record = await retire_test_agent(str(db_path), reason="testing_complete")

    archive_path = Path(record.archive_path)
    assert str(archive_path).startswith(str(tmp_path))
    assert archive_path.parent == tmp_path / "archive" / "retired_agents"
