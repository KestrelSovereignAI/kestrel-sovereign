"""
Regression tests for #2135 (F099): bootstrap_config DB persistence must be
wired end-to-end through the ContextBuilder.

Before this fix, ``BootstrapLoader.load_db_config()`` had zero callers and the
loader inside ``ContextBuilder`` was never constructed with a db handle, so
DB-backed bootstrap config (``bootstrap_add`` / ``bootstrap_remove``
persistence) was dead: nothing was persisted or reloaded.

These tests exercise the real wiring:
- ``ContextBuilder`` forwards its ``db`` / ``agent_id`` into the
  ``BootstrapLoader``.
- ``ContextBuilder.load_bootstrap_db_config()`` reads persisted entries back
  and merges them into the file order before the first prompt assembly.
- A ``save_db_entry`` written by one ContextBuilder's loader is reloaded by a
  freshly-constructed ContextBuilder against the same real DB (round-trip).
"""

from types import SimpleNamespace

import pytest

from kestrel_sovereign.agent.context_builder import ContextBuilder
from kestrel_sovereign.features.contribution_runtime import (
    ContextClauseRegistry,
    FeatureContributionRuntimeError,
    ResolvedContextClause,
)


@pytest.fixture
async def real_db(tmp_path, sqlite_database_factory):
    """A real SQLite AsyncDatabase with the bootstrap_config table created."""
    return await sqlite_database_factory(tmp_path / "bootstrap-wired.db")


@pytest.fixture
def agent_dir(tmp_path):
    """Agent data directory seeded with a couple of bootstrap files."""
    d = tmp_path / "agent_data" / "wired_agent"
    d.mkdir(parents=True)
    (d / "AGENTS.md").write_text("Operator policy content.")
    (d / "SOUL.md").write_text("I am the agent.")
    (d / "NOTES.md").write_text("Custom notes content.")
    return d


def _make_builder(db, agent_dir, agent_id="did:test:wired"):
    """Construct a ContextBuilder the way kestrel_agent init does."""
    return ContextBuilder(
        storage=None,
        db=db,
        agent_id=agent_id,
        agent_data_path=str(agent_dir),
    )


class TestContextBuilderWiring:
    def test_loader_receives_db_and_agent_id(self, real_db, agent_dir):
        """ContextBuilder forwards db + agent_id into its BootstrapLoader."""
        cb = _make_builder(real_db, agent_dir)
        loader = cb._bootstrap_loader
        assert loader._db is real_db
        assert loader._agent_id == "did:test:wired"

    def test_db_attempted_gate_lights_up(self, real_db, agent_dir):
        """The db_attempted gate used by bootstrap_add now activates."""
        cb = _make_builder(real_db, agent_dir)
        loader = cb._bootstrap_loader
        db_attempted = bool(
            getattr(loader, "_db", None) and getattr(loader, "_agent_id", None)
        )
        assert db_attempted is True


class TestPersistenceRoundTrip:
    @pytest.mark.asyncio
    async def test_add_entry_persists_and_reloads_on_next_construction(
        self, real_db, agent_dir
    ):
        """A saved bootstrap entry is reloaded by a fresh ContextBuilder.

        Mirrors the acceptance criterion: after agent init with a real DB, a
        ``bootstrap_add`` entry persists and is reloaded by ``load_db_config()``
        on the next construction.
        """
        # First agent lifetime: persist a custom bootstrap file (what
        # ``bootstrap_add`` does under the hood via loader.save_db_entry).
        cb1 = _make_builder(real_db, agent_dir)
        await cb1._bootstrap_loader.save_db_entry(
            file_name="NOTES.md",
            file_path=str(agent_dir / "NOTES.md"),
            enabled=True,
            priority=150,
        )

        # Second agent lifetime: a brand-new ContextBuilder over the same DB.
        # NOTES.md is not in DEFAULT_BOOTSTRAP_FILES, so it only appears if the
        # DB row was actually read back.
        cb2 = _make_builder(real_db, agent_dir)
        assert "NOTES.md" not in cb2._bootstrap_loader.file_order

        await cb2.load_bootstrap_db_config()

        assert "NOTES.md" in cb2._bootstrap_loader.file_order
        assert "NOTES.md" in cb2._bootstrap_loader.get_bootstrap_content()

    @pytest.mark.asyncio
    async def test_disabled_entry_removes_file(self, real_db, agent_dir):
        """A persisted enabled=0 row drops the file on reload."""
        cb1 = _make_builder(real_db, agent_dir)
        await cb1._bootstrap_loader.save_db_entry(
            file_name="AGENTS.md", enabled=False, priority=10
        )

        cb2 = _make_builder(real_db, agent_dir)
        assert "AGENTS.md" in cb2._bootstrap_loader.file_order
        await cb2.load_bootstrap_db_config()
        assert "AGENTS.md" not in cb2._bootstrap_loader.file_order

    @pytest.mark.asyncio
    async def test_persisted_bootstrap_name_cannot_collide_with_active_context(
        self, real_db, agent_dir
    ):
        """DB-restored custom names are checked before first prompt assembly."""

        policy_path = agent_dir / "POLICY.yaml"
        policy_path.write_text("Custom operator policy.")
        seed = _make_builder(real_db, agent_dir)
        await seed._bootstrap_loader.save_db_entry(
            file_name="POLICY.yaml",
            file_path=str(policy_path),
            enabled=True,
            priority=150,
        )
        owner = "tests:persisted-bootstrap-collision"
        registry = ContextClauseRegistry()
        registry.register_batch(
            (
                ResolvedContextClause(
                    owner=owner,
                    name="POLICY.yaml",
                    priority=10,
                    body="feature policy",
                    registration=SimpleNamespace(
                        identity=(owner, "POLICY.yaml")
                    ),
                ),
            )
        )
        builder = ContextBuilder(
            storage=None,
            db=real_db,
            agent_id="did:test:wired",
            agent_data_path=str(agent_dir),
            context_clause_registry=registry,
        )

        with pytest.raises(
            FeatureContributionRuntimeError,
            match="already registered",
        ):
            await builder.load_bootstrap_db_config()


class TestFirstPromptOrdering:
    @pytest.mark.asyncio
    async def test_load_happens_before_first_assembly(self, real_db, agent_dir):
        """load_bootstrap_db_config merges entries into the first prompt.

        The DB-persisted NOTES.md must be visible in the very first
        build_system_prompt after load_bootstrap_db_config — i.e. loading
        happens before assembly (no first-prompt ordering regression).
        """
        cb_seed = _make_builder(real_db, agent_dir)
        await cb_seed._bootstrap_loader.save_db_entry(
            file_name="NOTES.md",
            file_path=str(agent_dir / "NOTES.md"),
            enabled=True,
            priority=150,
        )

        cb = _make_builder(real_db, agent_dir)
        await cb.load_bootstrap_db_config()

        prompt = cb.build_system_prompt(constitution="CONSTITUTION")
        assert "Custom notes content." in prompt
        # Baseline bootstrap files still present in the same assembly.
        assert "Operator policy content." in prompt

    @pytest.mark.asyncio
    async def test_no_db_is_a_noop(self, agent_dir):
        """Without a db handle, load_bootstrap_db_config no-ops cleanly."""
        cb = ContextBuilder(storage=None, agent_data_path=str(agent_dir))
        # Should not raise even though no db/agent_id were provided.
        await cb.load_bootstrap_db_config()
        prompt = cb.build_system_prompt(constitution="CONSTITUTION")
        assert "Operator policy content." in prompt
