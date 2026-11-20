"""Exercise the public Constitution tool and ComputerUse's constitutional gate."""

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kestrel_sovereign import config
from kestrel_sovereign.constitution.hierarchy import DANGEROUS_CAPABILITIES
from kestrel_sovereign.features.computer_use.feature import ComputerUseFeature
from kestrel_sovereign.features.constitution import ConstitutionFeature
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.storage import AsyncStorage, GraphNode

_BASE_IX = (
    "### Amendment IX: Capability Boundaries\n"
    "#### Granted Capabilities\n"
    "- [ ] filesystem_write\n"
    "- [ ] shell_execution_host\n\n"
)
_BASE = f"# Governing text\n\n## Book II: Sovereign Amendments\n\n{_BASE_IX}"
_OVERLAY = "### Amendment IX\n- [x] filesystem_write\n- [x] shell_execution_host\n"


def _snapshot(path):
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as reader:
        return tuple(reader.iterdump())


def _constitution_tool(agent):
    return next(tool for tool in ConstitutionFeature(agent).get_tools() if tool.name == "constitution")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overlay, anchor, expected_status, expected_grants",
    [
        pytest.param(
            _OVERLAY, "matching", "hash verified", {"filesystem_write", "shell_execution_host"},
            id="verified-overlay-grants",
        ),
        ("", "matching", "hash verified", set()),
        (_OVERLAY, None, "unanchored", {"filesystem_read"}),
        (_OVERLAY, "other", "hash mismatch", {"filesystem_read"}),
        (None, "other", "missing", {"filesystem_read"}),
        (None, None, "absent", {"filesystem_read"}),
        (_OVERLAY, "unavailable", "anchor unavailable", {"filesystem_read"}),
        pytest.param(
            _OVERLAY, "unreadable", "source unavailable", {"filesystem_read"},
            id="unreadable-overlay",
        ),
        pytest.param(
            b"\xff", "matching", "unreadable text", {"filesystem_read"},
            id="invalid-utf8-overlay",
        ),
    ],
)
async def test_anchored_tool_view_matches_gate_and_preserves_original(
    tmp_path, monkeypatch, overlay, anchor, expected_status, expected_grants
):
    # The package has a grant, while the independently anchored base leaves
    # its checkboxes unchecked. Those inputs have different roles.
    package = tmp_path / "package.md"
    package.write_text("### Amendment IX\n- [x] filesystem_read\n")
    monkeypatch.setattr(config, "CONSTITUTION_PATH", package)
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_write")
    overlay_path = tmp_path / "CONSTITUTION.md"
    overlay_bytes = overlay.encode("utf-8") if isinstance(overlay, str) else overlay
    if isinstance(overlay, str):
        overlay_path.write_bytes(overlay_bytes)

    path = tmp_path / "agent.db"
    agent = KestrelAgent(did="did:test:effective-grants", storage_path=str(path))
    agent.extension = None
    storage = AsyncStorage(str(path), agent_id=agent.agent_id)
    await storage.initialize()
    agent.storage = agent._raw_storage = storage
    try:
        # Invalid bytes written after startup model a live replacement.
        if isinstance(overlay, bytes):
            overlay_path.write_bytes(overlay)
        digest = await storage.store_file(_BASE.encode("utf-8"), "governing.md")
        overlay_hash = hashlib.sha256(overlay_bytes).hexdigest() if overlay_bytes is not None else None
        properties = {"constitution_hash": digest}
        if anchor in ("matching", "other"):
            properties["constitution_overlay_hash"] = overlay_hash if anchor == "matching" else "0" * 64
        await storage.add_node(
            GraphNode(node_id=agent.agent_id, node_type="agent", label="lookup", properties=properties)
        )
        if anchor == "unavailable":
            with monkeypatch.context() as error_scope:
                error_scope.setattr(storage, "get_node", AsyncMock(side_effect=OSError("read unavailable")))
                await agent.verify_constitution_overlay()
        elif anchor == "unreadable":
            read_bytes = Path.read_bytes

            def read_with_unavailable_overlay(path):
                if path == overlay_path:
                    raise OSError("overlay file unavailable")
                return read_bytes(path)

            with monkeypatch.context() as error_scope:
                error_scope.setattr(Path, "read_bytes", read_with_unavailable_overlay)
                await agent.verify_constitution_overlay()
        else:
            await agent.verify_constitution_overlay()

        before = _snapshot(path)
        tool = _constitution_tool(agent)
        gate = ComputerUseFeature(agent)
        view = await tool.execute(article="amendment", search="IX")
        assert view["status"] == "ok", view
        data = view["data"]
        assert set(data["effective_grants"]) == expected_grants
        assert set(data["effective_grants"]) == {
            capability for capability in DANGEROUS_CAPABILITIES if gate._constitution_allows(capability)
        }
        assert data["overlay_status"] == expected_status
        assert data["signer"] is None
        assert "Signer: unavailable" in view["confirmation"]
        assert "Privacy, runtime configuration, and approval checks still apply." in view["confirmation"]
        if expected_status == "hash verified":
            assert data["source"] == "per-agent overlay"
            assert data["verification"] == "hash verified"
            assert data["sha256"] == overlay_hash
            assert f"SHA-256: {overlay_hash}" in view["confirmation"]
        else:
            assert data["source"] == "packaged Constitution"
            assert data["verification"].startswith("unavailable")
            assert data["sha256"] == hashlib.sha256(package.read_bytes()).hexdigest()

        assert data["base_sha256"] == digest
        assert f"Full anchored text SHA-256: {digest}" in view["confirmation"]
        assert data["base_text"] == _BASE_IX
        assert view["confirmation"].endswith(_BASE_IX)
        assert (await tool.execute())["confirmation"].encode("utf-8") == _BASE.encode("utf-8")
        # Runtime contributions remain visible in the standard governing view.
        # The explicit original path and the IX excerpt use only anchored bytes.
        agent.extension = SimpleNamespace(get_constitution_amendments=lambda: "Additional runtime terms.")
        governing = await tool.execute()
        assert governing["confirmation"].endswith("Additional runtime terms.")
        original = await tool.execute(article="original")
        assert original["confirmation"].encode("utf-8") == _BASE.encode("utf-8")
        with_contributions = await tool.execute(article="amendment", search="9")
        assert with_contributions["data"] == data
        if overlay == _OVERLAY and anchor == "matching":
            # Runtime may refresh cached text before the asynchronous anchor
            # read completes. The existing gate still uses its acceptance flag.
            monkeypatch.setattr(agent, "constitution_text", _OVERLAY.replace("filesystem_write", "filesystem_read"))
            stale = (await tool.execute(article="amendment", search="IX"))["data"]
            assert set(stale["effective_grants"]) == {
                capability for capability in DANGEROUS_CAPABILITIES if gate._constitution_allows(capability)
            } == {"filesystem_read", "shell_execution_host"}
            assert stale["verification"] == "unavailable"
            assert stale["overlay_status"] == "verification unavailable"
            assert stale["sha256"] is None

            # The anchored bytes become unavailable between the two native
            # reads. Do not label the surviving grant data a current result.
            with monkeypatch.context() as error_scope:
                retrieval = AsyncMock(
                    side_effect=[_BASE.encode("utf-8"), OSError("governing bytes unavailable")]
                )
                error_scope.setattr(storage, "retrieve_file", retrieval)
                failed_view = await tool.execute(article="amendment", search="IX")
                assert retrieval.await_count == 2
            assert failed_view["status"] == "error", failed_view
            assert "governing bytes unavailable" in failed_view["error"]
            assert "effective_grants" not in failed_view["data"]

        assert _snapshot(path) == before
        assert (overlay_path.read_bytes() if overlay_path.exists() else None) == overlay_bytes
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attributes, package, env, expected_source, expected_grants",
    [
        (
            {"granted_capabilities": {"filesystem_read", 42}, "capability_grants": {"filesystem_write"}},
            None, "", "agent.granted_capabilities", {"filesystem_read"},
        ),
        (
            {"granted_capabilities": set(), "capability_grants": {"filesystem_write"},
             "amendment_ix": {"shell_execution_sandboxed"}},
            None, "", "agent.capability_grants", {"filesystem_write"},
        ),
        (
            {"amendment_ix": {"shell_execution_sandboxed"}},
            _OVERLAY, "", "agent.amendment_ix", {"shell_execution_sandboxed"},
        ),
        (
            {}, "### Amendment IX\n- [ ] filesystem_read\n",
            " ,filesystem_read,  shell_execution_sandboxed,unknown_token, ",
            "KESTREL_CAPABILITY_GRANTS environment variable", {"filesystem_read", "shell_execution_sandboxed"},
        ),
        ({}, None, "", "default deny (packaged Constitution: unavailable)", set()),
        (
            {"constitution_overlay_verified": True, "constitution_text": _OVERLAY},
            None, "", "per-agent overlay", {"filesystem_write", "shell_execution_host"},
        ),
    ],
)
async def test_legacy_grant_sources_keep_precedence_without_claiming_verification(
    tmp_path, monkeypatch, attributes, package, env, expected_source, expected_grants
):
    path = tmp_path / "package.md"
    if package is not None:
        path.write_text(package)
    monkeypatch.setattr(config, "CONSTITUTION_PATH", path)
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", env)
    agent = SimpleNamespace(_get_governing_constitution=AsyncMock(return_value=_BASE), **attributes)
    view = await _constitution_tool(agent).execute(article="Amendment IX")
    assert view["status"] == "ok", view
    data = view["data"]
    assert data["source"] == expected_source
    assert set(data["effective_grants"]) == expected_grants
    gate = ComputerUseFeature(agent)
    assert set(data["effective_grants"]) == {
        capability for capability in DANGEROUS_CAPABILITIES if gate._constitution_allows(capability)
    }
    assert data["verification"] == "unavailable"
    assert data["signer"] is None
    assert data["sha256"] is None
    if attributes.get("constitution_overlay_verified"):
        assert data["overlay_status"] == "verification unavailable"
    agent._get_governing_constitution.assert_awaited_once_with(allow_lazy_anchor=False)


@pytest.mark.asyncio
async def test_search_lists_amendment_ix_once():
    agent = SimpleNamespace(_get_governing_constitution=AsyncMock(return_value=_BASE))
    view = await _constitution_tool(agent).execute(article="search", search="shell_execution_host")
    assert view["status"] == "ok", view
    assert view["confirmation"].count("### Amendment IX: Capability Boundaries") == 1


@pytest.mark.asyncio
async def test_failed_governing_reads_do_not_show_current_grants():
    error = "Error: Governing constitution changed after this turn's audited admission."
    agent = SimpleNamespace(
        _get_governing_constitution=AsyncMock(return_value=error),
        _get_original_governing_constitution=AsyncMock(return_value=error),
        granted_capabilities={"filesystem_write"},
    )
    tool = _constitution_tool(agent)
    for args in ({"article": "amendment", "search": "IX"}, {"article": "original"}):
        view = await tool.execute(**args)
        assert view["status"] == "error", view
        assert view["error"] == error
        assert "effective_grants" not in view["data"]
    agent._get_original_governing_constitution.assert_awaited_once()
