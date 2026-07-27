"""Regression tests for #2769 — LLM usage rows must follow the OWNING agent.

Filed after a live incident on the prod host (2026-07-27). ``.env`` set
``KESTREL_DB_PATH=./agent_data/claw`` and all four agents ran in ONE uvicorn
process (in-process multi-agent). Because ``_init_usage_tracking`` read only the
process-global env var, every agent's ``model_usage`` rows were written into that
one agent's ``llm_usage.db``: a Meridian turn on ``anthropic:plan/claude-opus-5``
landed ``use_count=2`` in Claw's database while Claw was pinned to
``openai:plan/gpt-5.6-sol``.

``KESTREL_DB_PATH`` must keep working for single-agent and container deployments
(7 Dockerfiles, the entrypoints, docker-compose and the documented launch
contract all set it) — so the fix is precedence, mirroring #2604's identity-export
resolution, not removal.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kestrel_sovereign.llm.usage_tracking import UsageTrackingMixin


class _Tracker(UsageTrackingMixin):
    """Bare mixin host — ``_init_usage_tracking`` only assigns attributes, so
    this avoids the heavy ``LLMService.__init__`` provider/discovery paths."""


@pytest.fixture(autouse=True)
def _clear_usage_env(monkeypatch):
    """Usage-DB selection reads several env vars; pin them off by default so a
    developer's real environment cannot decide these assertions."""
    for var in ("KESTREL_DB_PATH", "KESTREL_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_agent_data_dir_outranks_process_env(tmp_path, monkeypatch):
    """The exact prod bug: a per-agent binding must beat the host env var."""
    shared_env_dir = tmp_path / "agent_data" / "claw"
    own_dir = tmp_path / "agent_data" / "meridian"
    monkeypatch.setenv("KESTREL_DB_PATH", str(shared_env_dir))

    tracker = _Tracker()
    tracker._init_usage_tracking(agent_data_dir=own_dir)

    assert tracker._usage_db_path == os.path.join(str(own_dir), "llm_usage.db")
    assert "claw" not in tracker._usage_db_path


def test_two_agents_in_one_process_do_not_share_a_usage_db(tmp_path, monkeypatch):
    """In-process multi-agent is the whole point: one env var, several agents.

    Before the fix these two resolved to the SAME file and their usage rows were
    silently merged.
    """
    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent_data" / "claw"))

    first, second = _Tracker(), _Tracker()
    first._init_usage_tracking(agent_data_dir=tmp_path / "agent_data" / "meridian")
    second._init_usage_tracking(agent_data_dir=tmp_path / "agent_data" / "nellie")

    assert first._usage_db_path != second._usage_db_path


def test_env_var_still_authoritative_without_an_agent_binding(tmp_path, monkeypatch):
    """Container/single-agent contract: ``KESTREL_DB_PATH`` still decides when
    the caller holds no agent object. Do not regress Docker/Cloud Run."""
    env_dir = tmp_path / "data"
    monkeypatch.setenv("KESTREL_DB_PATH", str(env_dir))

    tracker = _Tracker()
    tracker._init_usage_tracking()

    assert tracker._usage_db_path == os.path.join(str(env_dir), "llm_usage.db")


def test_falls_back_to_historical_default_when_nothing_is_set():
    tracker = _Tracker()
    tracker._init_usage_tracking()

    assert tracker._usage_db_path == os.path.join("./agent_data", "llm_usage.db")
    assert tracker._db_backend == "sqlite"


def test_accepts_a_string_agent_data_dir(tmp_path):
    """Callers pass either ``Path`` (agent_manager) or ``str``."""
    tracker = _Tracker()
    tracker._init_usage_tracking(agent_data_dir=str(tmp_path / "agent_data" / "emma"))

    assert tracker._usage_db_path.endswith(os.path.join("emma", "llm_usage.db"))


def test_postgres_url_wins_over_any_agent_dir(tmp_path):
    """A Postgres deployment shares one usage table by design; the per-agent
    directory must not silently downgrade it to SQLite."""
    tracker = _Tracker()
    tracker._init_usage_tracking(
        "postgresql://user@host/db", agent_data_dir=tmp_path / "agent_data" / "emma"
    )

    assert tracker._db_backend == "postgres"
    assert not hasattr(tracker, "_usage_db_path")


def test_init_does_not_touch_disk(tmp_path):
    """#2362 invariant: recording the path must not create directories. The
    lazy mkdir lives in ``_ensure_db_initialized``."""
    target = tmp_path / "agent_data" / "meridian"

    tracker = _Tracker()
    tracker._init_usage_tracking(agent_data_dir=target)

    assert not target.exists()


def test_llm_service_threads_agent_data_dir_through(tmp_path, monkeypatch):
    """End-to-end through the real constructor signature, so the wiring between
    ``LLMService`` and the mixin cannot silently drift apart."""
    from unittest.mock import patch

    from kestrel_sovereign.llm import service as svc_mod

    monkeypatch.setenv("KESTREL_DB_PATH", str(tmp_path / "agent_data" / "claw"))
    own_dir = tmp_path / "agent_data" / "meridian"

    with patch.object(svc_mod, "ProviderRegistry") as mock_registry, \
         patch.object(svc_mod.LLMService, "_load_from_disk_cache"), \
         patch.object(svc_mod.LLMService, "_init_constitutional_profiles"):
        mock_registry.return_value.initialize_providers.return_value = []
        service = svc_mod.LLMService(agent_data_dir=own_dir)

    assert service._usage_db_path == os.path.join(str(own_dir), "llm_usage.db")


def test_agent_manager_binds_each_agent_to_its_own_data_dir():
    """The multi-agent construction site is the one that MUST pass the binding.

    Guards the wiring itself: dropping the argument here reintroduces the exact
    prod bug while every unit test above still passes, because the mixin would
    silently fall back to the process environment.
    """
    import inspect
    import re

    from kestrel_sovereign.multi_agent import agent_manager

    source = inspect.getsource(agent_manager)
    constructions = re.findall(r"LLMService\(([^)]*)\)", source)

    assert constructions, "expected agent_manager to construct an LLMService"
    for args in constructions:
        assert "agent_data_dir=" in args, (
            "AgentManager must bind each agent's LLMService to that agent's own "
            f"data dir; found bare construction: LLMService({args})"
        )
