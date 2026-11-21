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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identifier, heading",
    [(8, "Amendment VIII: Historical Terms"), (9, "Effective ComputerUse constitutional grants")],
)
async def test_public_tool_accepts_integer_amendment_identifier(identifier, heading):
    text = _BASE.replace(_BASE_IX, f"### Amendment VIII: Historical Terms\n\n{_BASE_IX}")
    agent = SimpleNamespace(_get_governing_constitution=AsyncMock(return_value=text))
    view = await _constitution_tool(agent).execute(article="amendment", search=identifier)
    assert view["status"] == "ok", view
    assert heading in view["confirmation"]


@pytest.mark.parametrize("overlay_present", [False, True], ids=["empty-package-falls-through", "verified-empty-overlay"])
def test_edition_2_freezes_legacy_precedence_and_retains_evidence(
    tmp_path, monkeypatch, overlay_present,
):
    from kestrel_sovereign.constitution.grants import (
        build_grant_baseline,
        preview_offline_legacy_grants,
        resolve_amendment_ix_grants,
        validate_grant_cutover,
    )

    package = tmp_path / "package.md"
    package.write_text(_BASE)
    monkeypatch.setattr(config, "CONSTITUTION_PATH", package)
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_read,filesystem_write")
    agent = SimpleNamespace()
    target = "did:test:grant-baseline"
    v2 = "---\nconstitution_version: 2\n---\n"
    overlay_hash = None
    if overlay_present:
        raw = b""
        (tmp_path / "CONSTITUTION.md").write_bytes(raw)
        overlay_hash = hashlib.sha256(raw).hexdigest()

    original = preview_offline_legacy_grants(tmp_path, overlay_hash=overlay_hash)
    record = build_grant_baseline(target, original)
    assert record["legacy_evidence"]["signer"] is None
    assert record["legacy_evidence"]["sha256"] == overlay_hash
    assert record["grants"] == ([] if overlay_present else ["filesystem_read", "filesystem_write"])
    agent._constitution_host_grant_baseline = validate_grant_cutover(
        target_did=target, artifact={"grant_baseline": record}, prior_text=_BASE,
        new_text=v2, legacy_resolution=lambda: original,
    )
    assert resolve_amendment_ix_grants(agent).grants == set(record["grants"])

    agent.granted_capabilities = {"shell_execution_host"}
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_read,filesystem_outside_workspace")
    package.write_text("### Amendment IX\n- [x] shell_execution_host\n")
    effective = resolve_amendment_ix_grants(agent)
    assert effective.grants == (set() if overlay_present else {"filesystem_read"})
    assert effective.environment_narrowed is not overlay_present
    assert effective.content_hash == overlay_hash
    monkeypatch.delenv("KESTREL_CAPABILITY_GRANTS")
    assert resolve_amendment_ix_grants(agent).grants == set()


def test_edition_2_cutover_cannot_expand_remove_or_reattribute_legacy_grants(monkeypatch):
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
    from kestrel_sovereign.constitution.grants import (
        build_grant_baseline,
        resolve_legacy_grants,
        validate_grant_cutover,
    )

    agent = SimpleNamespace(granted_capabilities={"filesystem_read"})
    target = "did:test:grant-baseline"
    v2 = "---\nconstitution_version: 2\n---\n"
    for grants in ([], ["filesystem_read", "filesystem_write"]):
        record = build_grant_baseline(target, resolve_legacy_grants(agent))
        record["grants"] = grants
        with pytest.raises(AmendmentArtifactError, match="differs from the existing decision"):
            validate_grant_cutover(
                target_did=target, artifact={"grant_baseline": record},
                prior_text=_BASE, new_text=v2,
                legacy_resolution=lambda: resolve_legacy_grants(agent),
            )
    with pytest.raises(AmendmentArtifactError, match="Cannot restore legacy"):
        validate_grant_cutover(
            target_did=target, artifact={}, prior_text=v2, new_text=_BASE,
            legacy_resolution=lambda: resolve_legacy_grants(agent),
        )


@pytest.mark.asyncio
async def test_edition_2_runtime_refuses_altered_baseline_and_view_agrees(
    tmp_path, monkeypatch,
):
    import json

    from cryptography.hazmat.primitives.asymmetric import ec

    from kestrel_sovereign.constitution.amendment_artifact import (
        build_legacy_signed_reanchor_artifact,
        did_document_from_legacy_public_key,
    )
    from kestrel_sovereign.constitution.grants import build_grant_baseline, resolve_legacy_grants
    from kestrel_sovereign.constitution.reanchor_receipt import CONSTITUTION_REANCHOR_KEY

    v2_path = (
        Path(__file__).resolve().parents[2] / "kestrel_sovereign" / "data"
        / "constitutions" / "v2" / "KESTREL_CONSTITUTION.md"
    )
    v2_bytes = v2_path.read_bytes()
    monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_read")
    agent = KestrelAgent(did="did:test:baseline", storage_path=str(tmp_path / "baseline.db"))
    agent.extension = None
    sovereign_key = ec.generate_private_key(ec.SECP256K1())
    sovereign = did_document_from_legacy_public_key("did:test:external-root", sovereign_key.public_key())
    monkeypatch.setattr(agent, "_trusted_sovereign_did_document", lambda *_: sovereign)

    async with AsyncStorage(agent.storage_path, agent_id=agent.agent_id) as storage:
        agent.storage = agent._raw_storage = storage
        digest = await storage.store_file(v2_bytes, "governing.md")
        record = build_grant_baseline(agent.agent_id, resolve_legacy_grants(agent))
        assert record["grants"] == ["filesystem_read"]
        artifact = build_legacy_signed_reanchor_artifact(
            signer_did=sovereign["id"], constitution_sha256=digest, private_key=sovereign_key,
            grant_baseline=record,
        )

        async def store_artifact(artifact):
            artifact_hash = await storage.store_file(json.dumps(artifact).encode(), "signed-artifact.json")
            node = GraphNode(
                node_id=agent.agent_id,
                node_type="agent",
                label="Baseline",
                properties={
                    "constitution_hash": digest,
                    CONSTITUTION_REANCHOR_KEY: {"new_hash": digest, "signed_artifact_hash": artifact_hash},
                },
            )
            await storage.add_node(node)
            return node, artifact_hash

        tool = _constitution_tool(agent)
        gate = ComputerUseFeature(agent)
        node, artifact_hash = await store_artifact(artifact)
        ok, message = await agent._refresh_constitution_host_grants(node)
        assert not ok and "differs from the authoritative source" in message
        assert not gate._constitution_allows("filesystem_read")
        monkeypatch.setattr(config, "CONSTITUTION_PATH", v2_path)
        ok, message = await agent._refresh_constitution_host_grants(node)
        assert ok, message
        monkeypatch.setenv("KESTREL_CAPABILITY_GRANTS", "filesystem_read,filesystem_write")
        assert gate._constitution_allows("filesystem_read")
        assert not gate._constitution_allows("filesystem_write")
        baseline_view = (await tool.execute(article="amendment", search="IX"))["data"]
        assert baseline_view["effective_grants"] == ["filesystem_read"]
        assert baseline_view["signer"] is None
        assert baseline_view["baseline_signer"] == sovereign["id"]
        assert baseline_view["baseline_sha256"] == artifact_hash

        from copy import deepcopy

        conflicting_record = deepcopy(record)
        conflicting_record["grants"].append("filesystem_write")
        conflicting_artifact = build_legacy_signed_reanchor_artifact(
            signer_did=sovereign["id"], constitution_sha256=digest, private_key=sovereign_key,
            grant_baseline=conflicting_record,
        )
        conflict_node, conflict_hash = await store_artifact(conflicting_artifact)
        conflict_node.properties["constitution_reanchor_history"] = [{
            "receipt": node.properties[CONSTITUTION_REANCHOR_KEY],
            "superseded_by_constitution_hash": digest,
            "superseded_by_artifact_hash": conflict_hash,
        }]
        await storage.add_node(conflict_node)
        ok, message = await agent._refresh_constitution_host_grants(conflict_node)
        assert ok, message
        assert gate._constitution_allows("filesystem_read")
        assert not gate._constitution_allows("filesystem_write")
        conflict_view = (await tool.execute(article="amendment", search="IX"))["data"]
        assert conflict_view["effective_grants"] == ["filesystem_read"]
        assert conflict_view["conflicts"] == ["filesystem_write"]

        record["grants"].append("filesystem_write")
        tampered_node, _ = await store_artifact(artifact)
        ok, message = await agent._refresh_constitution_host_grants(tampered_node)
        assert not ok and "signature" in message
        assert not gate._constitution_allows("filesystem_read")
        assert not gate._constitution_allows("filesystem_write")
        denied_view = (await tool.execute(article="amendment", search="IX"))["data"]
        assert denied_view["effective_grants"] == []
        assert denied_view["baseline_signer"] is None


def test_grant_repair_classifies_only_exact_replacement_bytes():
    from kestrel_sovereign.constitution.grants import prior_text_for_grant_repair

    historical_hash = hashlib.sha256(_BASE.encode()).hexdigest()
    assert prior_text_for_grant_repair(
        anchored_text=None, historical_hash=historical_hash,
        new_text=_BASE, new_hash=historical_hash,
    ) == _BASE
    assert prior_text_for_grant_repair(
        anchored_text=None, historical_hash="0" * 64,
        new_text=_BASE, new_hash=historical_hash,
    ) is None


def test_same_hash_edition_2_repair_still_requires_prior_baseline():
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
    from kestrel_sovereign.constitution.grants import (
        build_grant_baseline,
        prior_text_for_grant_repair,
        resolve_legacy_grants,
        validate_grant_cutover,
    )

    edition_2 = "---\nconstitution_version: 2\n---\n"
    digest = hashlib.sha256(edition_2.encode()).hexdigest()
    prior_text = prior_text_for_grant_repair(
        anchored_text=None, historical_hash=digest, new_text=edition_2, new_hash=digest,
    )
    target = "did:test:edition-2-repair"
    legacy = resolve_legacy_grants(SimpleNamespace(granted_capabilities={"filesystem_read"}))
    artifact = {"grant_baseline": build_grant_baseline(target, legacy)}
    with pytest.raises(AmendmentArtifactError, match="differs from the existing decision"):
        validate_grant_cutover(
            target_did=target,
            artifact=artifact,
            prior_text=prior_text,
            new_text=edition_2,
            legacy_resolution=lambda: pytest.fail("edition 2 must not fall back to legacy grants"),
        )


@pytest.mark.asyncio
async def test_only_independently_pinned_admitted_grant_artifacts_survive_root_rotation(
    tmp_path, monkeypatch,
):
    import json

    from cryptography.hazmat.primitives.asymmetric import ec

    from kestrel_sovereign.constitution.amendment_artifact import (
        AmendmentArtifactError, build_legacy_signed_reanchor_artifact,
        did_document_from_legacy_public_key, verify_reanchor_artifact,
    )
    from kestrel_sovereign.constitution.grants import (
        HISTORY_PIN_ENV, build_grant_baseline, load_anchored_grant_baseline,
        resolve_legacy_grants, sovereign_document_fingerprint,
    )

    old_key, new_key, unrelated_key = (ec.generate_private_key(ec.SECP256K1()) for _ in range(3))
    old = did_document_from_legacy_public_key("did:test:old-root", old_key.public_key())
    new = did_document_from_legacy_public_key("did:test:new-root", new_key.public_key())
    unrelated = did_document_from_legacy_public_key("did:test:new-root", unrelated_key.public_key())
    target = "did:test:host-grant-history"
    digest = "1" * 64
    baseline = build_grant_baseline(target, resolve_legacy_grants(SimpleNamespace(
        granted_capabilities={"filesystem_read"},
    )))

    def artifact(signer, key, *, reason="preserve decision"):
        return build_legacy_signed_reanchor_artifact(
            signer_did=signer["id"], constitution_sha256=digest, private_key=key,
            grant_baseline=baseline, reason=reason,
        )

    historic = artifact(old, old_key)
    latest = artifact(new, new_key)
    assert not verify_reanchor_artifact(
        historic, trusted_did_document=new, expected_constitution_sha256=digest,
    ).ok
    pin_path = tmp_path / "archived-grant-credentials.json"
    env = {HISTORY_PIN_ENV: str(pin_path)}

    async with AsyncStorage(str(tmp_path / "history.db"), agent_id=target) as storage:
        async def receipt(value):
            artifact_hash = await storage.store_file(json.dumps(value).encode(), "artifact.json")
            return {"new_hash": digest, "signed_artifact_hash": artifact_hash}

        prior = await receipt(historic)
        current = await receipt(latest)
        properties = {
            "constitution_hash": digest,
            "constitution_reanchor": current,
            "constitution_reanchor_history": [{
                "receipt": prior, "superseded_by_constitution_hash": digest,
                "superseded_by_artifact_hash": current["signed_artifact_hash"],
            }],
        }

        async def load(root, properties=properties, environ=env):
            return await load_anchored_grant_baseline(
                storage.db, target_did=target, properties=properties,
                trusted_did_document=root, environ=environ,
            )

        with pytest.raises(AmendmentArtifactError, match="signature verification"):
            await load(new, environ={})
        pin_path.write_text(json.dumps({
            "active_document_sha256": sovereign_document_fingerprint(new),
            "admitted_artifacts": {prior["signed_artifact_hash"]: old},
        }))
        loaded = await load(new)
        assert loaded.grants == frozenset({"filesystem_read"})
        assert loaded.baseline_signer == new["id"]

        # Rotation preserves the original artifact even before a new repair writes one.
        old_current = {"constitution_hash": digest, "constitution_reanchor": prior}
        assert (await load(new, old_current)).baseline_signer == old["id"]
        with pytest.raises(AmendmentArtifactError, match="do not bind"):
            await load(unrelated, old_current)

        # An archived public key alone cannot authorize a different artifact.
        unadmitted = await receipt(artifact(old, old_key, reason="another signed artifact"))
        with pytest.raises(AmendmentArtifactError, match="signature verification"):
            await load(new, {"constitution_hash": digest, "constitution_reanchor": unadmitted})


@pytest.mark.parametrize(
    "observed_at, created_at, valid",
    [
        ("2000-01-01T00:00:00+00:00", "2000-01-02T00:00:00+00:00", True),
        ("2000-01-02T00:00:00+00:00", "2000-01-01T00:00:00+00:00", False),
        ("2000-01-01T00:00:00+00:00", "9999-01-02T00:00:00+00:00", False),
        ("9999-01-01T00:00:00+00:00", "9999-01-02T00:00:00+00:00", False),
    ],
)
def test_signed_offline_grant_observation_must_be_chronological(observed_at, created_at, valid):
    from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
    from kestrel_sovereign.constitution.grants import (
        GrantResolution, build_offline_grant_observation, require_offline_grant_observation,
    )

    prior = "1" * 64
    resolution = GrantResolution(frozenset(), "default deny", "unavailable", "absent")
    observation = build_offline_grant_observation(
        target_did="did:test:agent", prior_sha256=prior, observed_at=observed_at, resolution=resolution,
    )
    artifact = {
        "created_at": created_at, "grant_baseline": observation["baseline"],
        "offline_grant_observation": observation,
    }
    if valid:
        require_offline_grant_observation(artifact, prior_sha256=prior)
    else:
        with pytest.raises(AmendmentArtifactError, match="Invalid timestamp"):
            require_offline_grant_observation(artifact, prior_sha256=prior)
