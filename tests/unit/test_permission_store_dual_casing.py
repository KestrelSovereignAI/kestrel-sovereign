"""``PermissionStore.get_permission`` resolves rows under both PascalCase and
snake_case forms, picking the more permissive of any duplicates.

The persisted DB has accumulated a mix of casings for the same logical tool
over time: PascalCase (``TaskFeature.respond_to_a2a_task``, written from the
subagent path + the operator's ``!security-set`` calls) and snake_case
(``task_feature.…``, written from the older direct-tool dispatch path
before #1427 normalized it). Honor whichever the operator actually granted.
"""

from __future__ import annotations

import pytest

from kestrel_sovereign.features.security.permissions import (
    PermissionLevel,
    PermissionStore,
    _most_permissive,
    _snake_case,
)


def test_snake_case_matches_feature_tool_name():
    """Must use the exact same regex Feature.tool_name uses
    (kestrel_sovereign/features/base.py) — see test below for the canonical
    examples. Sequential uppercase produces ``m_c_p_agent`` (not
    ``mcp_agent``); ChannelFeature → channel_feature."""
    assert _snake_case("TaskFeature") == "task_feature"
    assert _snake_case("PeersFeature") == "peers_feature"
    assert _snake_case("MCPAgent") == "m_c_p_agent"
    assert _snake_case("ChannelFeature") == "channel_feature"
    assert _snake_case("AlreadySnake_thing") == "already_snake_thing"


def test_most_permissive_picks_allow_over_ask():
    assert _most_permissive([PermissionLevel.ASK, PermissionLevel.ALLOW]) is PermissionLevel.ALLOW
    assert _most_permissive([PermissionLevel.DENY, PermissionLevel.ALLOW]) is PermissionLevel.ALLOW
    assert _most_permissive([PermissionLevel.AUTO, PermissionLevel.ASK]) is PermissionLevel.AUTO
    assert _most_permissive([PermissionLevel.ASK]) is PermissionLevel.ASK


@pytest.mark.asyncio
async def test_get_permission_finds_pascalcase_when_queried_pascal(tmp_path):
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    await store.set_permission(
        "TaskFeature", "respond_to_a2a_task", PermissionLevel.ALLOW,
    )
    assert (
        await store.get_permission("TaskFeature", "respond_to_a2a_task")
        == PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_get_permission_finds_pascalcase_when_queried_snake(tmp_path):
    """Legacy snake-cased callers reading a PascalCase row still get ALLOW.

    Defensive — every NEW direct-tool dispatch already normalizes to
    PascalCase (#1427), but other code paths and tests may still pass the
    snake form. Treat both as the same logical row."""
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    await store.set_permission(
        "TaskFeature", "respond_to_a2a_task", PermissionLevel.ALLOW,
    )
    assert (
        await store.get_permission("task_feature", "respond_to_a2a_task")
        == PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_get_permission_finds_snakecase_when_queried_pascal(tmp_path):
    """Operators who previously approved a tool under the snake form (via the
    old direct-tool path) still see ALLOW after the lookup normalizes to
    PascalCase. Without this, the #1427 fix would have silently revoked
    grants that the operator already made."""
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    await store.set_permission(
        "task_feature", "respond_to_a2a_task", PermissionLevel.ALLOW,
    )
    assert (
        await store.get_permission("TaskFeature", "respond_to_a2a_task")
        == PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_get_permission_picks_more_permissive_of_two_rows(tmp_path):
    """Both forms present, one ALLOW one ASK → the operator's ALLOW wins.

    Uses ``TaskFeature``/``task_feature`` (clean snake↔Pascal mapping) so the
    variant generator finds both rows. Features with non-class-name aliases
    (``computer_use`` for ``ComputerUseFeature``, ``github`` for
    ``GitHubFeature``) require a separate startup migration — out of scope
    here; tracked as the snake-row re-approval follow-up."""
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    await store.set_permission(
        "TaskFeature", "respond_to_a2a_task", PermissionLevel.ASK,
    )
    await store.set_permission(
        "task_feature", "respond_to_a2a_task", PermissionLevel.ALLOW,
    )
    assert (
        await store.get_permission("TaskFeature", "respond_to_a2a_task")
        == PermissionLevel.ALLOW
    )


@pytest.mark.asyncio
async def test_migrate_legacy_feature_aliases_consolidates_alias_rows(tmp_path):
    """Aliased features (``ComputerUseFeature.tool_name = "computer_use"``)
    have rows under both the class name AND the short alias. After
    migration, the canonical (class name) row holds the more permissive
    grant from either side."""
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()

    # Operator granted ALLOW under the short alias only; class-name row is ASK
    await store.set_permission("ComputerUseFeature", "fs_read", PermissionLevel.ASK)
    await store.set_permission("computer_use", "fs_read", PermissionLevel.ALLOW)
    # Tool only present under alias, no class row
    await store.set_permission("computer_use", "fs_list", PermissionLevel.ALLOW)
    # Unrelated row not affected
    await store.set_permission("MemoryFeature", "search_memory", PermissionLevel.ASK)

    aliases = {"computer_use": "ComputerUseFeature"}
    n = await store.migrate_legacy_feature_aliases(aliases)
    assert n == 2, f"expected 2 upserts (fs_read upgrade + fs_list insert), got {n}"

    # The class-name row is now ALLOW (alias was more permissive)
    assert await store.get_permission("ComputerUseFeature", "fs_read") == PermissionLevel.ALLOW
    # The class-name row for fs_list now exists at the granted level
    assert await store.get_permission("ComputerUseFeature", "fs_list") == PermissionLevel.ALLOW
    # Unrelated row untouched
    assert await store.get_permission("MemoryFeature", "search_memory") == PermissionLevel.ASK


@pytest.mark.asyncio
async def test_migrate_legacy_feature_aliases_idempotent(tmp_path):
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    await store.set_permission("computer_use", "fs_read", PermissionLevel.ALLOW)
    aliases = {"computer_use": "ComputerUseFeature"}
    n1 = await store.migrate_legacy_feature_aliases(aliases)
    n2 = await store.migrate_legacy_feature_aliases(aliases)
    assert n1 == 1, f"first pass should upsert 1 row, got {n1}"
    assert n2 == 0, f"second pass should be a no-op, got {n2}"


@pytest.mark.asyncio
async def test_get_permission_returns_ask_default_when_neither_row_present(tmp_path):
    db_path = str(tmp_path / "perms.db")
    store = PermissionStore(db_path)
    await store.initialize()
    assert (
        await store.get_permission("TaskFeature", "respond_to_a2a_task")
        == PermissionLevel.ASK
    )
