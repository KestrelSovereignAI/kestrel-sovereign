"""Unit tests for !reanchor-constitution command."""
import ast
import inspect
import json
import pytest
import hashlib
import textwrap
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from kestrel_sovereign.kestrel_agent import KestrelAgent
from kestrel_sovereign.agent.constitution import ConstitutionMixin
from kestrel_sovereign.setup.constitution_reanchor import _write_reanchor
from kestrel_sovereign.constitution.amendment_artifact import (
    MAX_REANCHOR_ARTIFACT_BYTES,
    build_legacy_signed_reanchor_artifact,
    did_document_from_legacy_public_key,
)
from kestrel_sovereign.constitution.emancipation import (
    EmancipationContract,
    apply_emancipation,
    contract_to_json,
    render_amendment_viii,
)
from kestrel_sovereign.security.crypto_suite import Secp256k1Suite


_SUITE = Secp256k1Suite()
ROOT_KEYPAIR = _SUITE.generate_keypair()
ROOT_DID = "did:pkh:eip155:1:0x0000000000000000000000000000000000000587"
ROOT_DID_DOCUMENT = did_document_from_legacy_public_key(
    ROOT_DID,
    ROOT_KEYPAIR.public_key,
)
AGENT_KEYPAIR = _SUITE.generate_keypair()
AGENT_DID = "did:pkh:eip155:1:0x000000000000000000000000000000000000a587"
AGENT_DID_DOCUMENT = did_document_from_legacy_public_key(
    AGENT_DID,
    AGENT_KEYPAIR.public_key,
)


def _graph_write_calls(function):
    """Return graph lock/add call sites in source order for one workflow."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"lock_nodes_for_update", "add_node"}:
                calls.append((node.lineno, node.func.attr, node))
    return sorted(calls)


def test_both_reanchor_writers_prelock_the_complete_shared_node_set():
    """Opposite semantic write order must not become opposite lock order."""

    expected_names = {
        ConstitutionMixin.reanchor_constitution: {"artifact_hash", "stored_hash"},
        _write_reanchor: {"artifact_hash", "new_hash"},
    }
    for function, expected in expected_names.items():
        calls = _graph_write_calls(function)
        lock_calls = [entry for entry in calls if entry[1] == "lock_nodes_for_update"]
        add_calls = [entry for entry in calls if entry[1] == "add_node"]
        assert len(lock_calls) == 1
        assert add_calls
        assert lock_calls[0][0] < add_calls[0][0]
        locked_names = {
            node.id
            for node in ast.walk(lock_calls[0][2].args[0])
            if isinstance(node, ast.Name)
        }
        assert locked_names == expected


@pytest.fixture(autouse=True)
def _operator_pinned_root(tmp_path, monkeypatch):
    """Every authorization test uses a root outside the mocked graph DB."""
    root_path = tmp_path / "operator-sovereign-root.did.json"
    root_path.write_text(json.dumps(ROOT_DID_DOCUMENT), encoding="utf-8")
    monkeypatch.setenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH", str(root_path))
    return root_path


#: What the agent is anchored to before the reanchor under test. The double has
#: to supply it because the command now reads its own anchored bytes to enforce
#: the Iron Rule for agents with no structured receipt (#2465).
ANCHORED_CONSTITUTION = b"# Kestrel Constitution v1\n\nOriginal content.\n"

#: Sentinel for "the row is there but this process cannot turn it into text".
UNREADABLE = object()


class _FakeFileRows:
    """The one query the Iron Rule guard's unbound file read actually issues.

    The guard reads the anchored constitution through the *ungoverned*
    connection (``AsyncFileStore(db)`` with no ``agent_id``), because an
    ownership-scoped read returns ``None`` for a blob that is sitting in
    ``files`` with no ``file_owners`` row — indistinguishable from no row at
    all, and the state of every pre-#2649 agent whose governance edge drifted.
    So the double models the *row*, not ``retrieve_file``: an absent blob is a
    missing row, and unreadable bytes are a row that will not decrypt. Get that
    wrong and the test passes on a path production never takes.
    """

    def __init__(self, content_hash, anchored):
        self._content_hash = content_hash
        self._anchored = anchored

    async def fetchone(self, query, params=()):
        assert "FROM files" in query, query
        assert "file_owners" not in query, (
            "The Iron Rule guard must read the anchored constitution unbound; "
            f"this query is ownership-scoped: {query}"
        )
        if self._anchored is None or params[0] != self._content_hash:
            return None
        if self._anchored is UNREADABLE:
            # Marked encrypted, with bytes no key will open — exactly what a
            # wrong KESTREL_DATA_KEY produces.
            return b"\x00not-a-valid-token", json.dumps({"enc": True})
        return self._anchored, None


def _make_agent(stored_hash="oldhash", safe_mode=False, anchored=ANCHORED_CONSTITUTION):
    """Create a mock agent with ConstitutionMixin methods bound."""
    agent = MagicMock(spec=KestrelAgent)
    agent._safe_mode = safe_mode
    agent._get_timestamp = MagicMock(return_value="2026-04-06T00:00:00Z")
    agent.extension = None
    agent.agent_id = AGENT_DID
    agent.identity = SimpleNamespace(
        legacy_did=AGENT_DID,
        signing_did=AGENT_DID,
        legacy_did_document=AGENT_DID_DOCUMENT,
    )
    agent._sovereign_trust_root_path = None

    node = MagicMock()
    node.properties = {
        "constitution_hash": stored_hash,
        "sovereign_root_did_document": ROOT_DID_DOCUMENT,
    }
    agent.storage = AsyncMock()
    agent.storage.get_node = AsyncMock(return_value=node)
    agent.storage.store_file = AsyncMock()
    agent.storage.retrieve_file = AsyncMock(
        return_value=None if anchored is UNREADABLE else anchored
    )
    agent.storage.add_node = AsyncMock()
    agent.storage.lock_nodes_for_update = AsyncMock()
    agent._raw_storage = SimpleNamespace(db=_FakeFileRows(stored_hash, anchored))
    # transaction() is an async context manager, not a coroutine — a plain
    # MagicMock provides __aenter__/__aexit__ on its return value.
    agent.storage.transaction = MagicMock()
    agent.privacy_agent = MagicMock()
    agent.privacy_agent.add_conversation = AsyncMock()

    agent.reanchor_constitution = ConstitutionMixin.reanchor_constitution.__get__(
        agent, KestrelAgent
    )
    agent._trusted_sovereign_did_document = (
        ConstitutionMixin._trusted_sovereign_did_document.__get__(agent, KestrelAgent)
    )
    agent._agent_signing_dids = ConstitutionMixin._agent_signing_dids.__get__(
        agent, KestrelAgent
    )
    agent._get_governing_constitution = ConstitutionMixin._get_governing_constitution.__get__(
        agent, KestrelAgent
    )
    return agent, node


FAKE_CONSTITUTION = b"# Kestrel Constitution v2\n\nAmended content here.\n"
FAKE_HASH = hashlib.sha256(FAKE_CONSTITUTION).hexdigest()


def _write_artifact(tmp_path, constitution_hash=FAKE_HASH, keypair=ROOT_KEYPAIR, did=ROOT_DID):
    artifact = build_legacy_signed_reanchor_artifact(
        signer_did=did,
        constitution_sha256=constitution_hash,
        private_key=keypair.private_key,
        created_at="2026-04-06T00:00:00Z",
        reason="unit test",
    )
    path = tmp_path / "constitution-reanchor.signed.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _open_handles(*contents):
    handles = []
    for content in contents:
        handle = MagicMock()
        handle.read.return_value = content
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handles.append(handle)
    return handles


# --- Authorization: expected_hash required ---

@pytest.mark.asyncio
async def test_reanchor_rejects_missing_signed_artifact():
    """Re-anchor refuses without a signed amendment artifact."""
    agent, _ = _make_agent()
    result = await agent.reanchor_constitution()
    assert "error" in result.lower()
    assert "signed amendment artifact required" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_short_hash(tmp_path):
    """Re-anchor refuses hash prefix shorter than 8 characters."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)
    result = await agent.reanchor_constitution(
        expected_hash="abc",
        amendment_artifact_path=str(artifact_path),
    )
    assert "error" in result.lower()
    assert "at least 8" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrong_hash(tmp_path):
    """Re-anchor refuses when expected hash doesn't match file on disk."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash="deadbeef",
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "hash mismatch" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_oversized_artifact_before_storage(tmp_path):
    agent, _ = _make_agent()
    artifact_path = tmp_path / "oversized.signed.json"
    artifact_path.write_bytes(b"x" * (MAX_REANCHOR_ARTIFACT_BYTES + 1))

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "exceeds" in result.lower()
    agent.storage.store_file.assert_not_called()


# --- Happy path with valid hash ---

@pytest.mark.asyncio
async def test_reanchor_succeeds_with_sovereign_signed_artifact(tmp_path):
    """Re-anchor stores new constitution when expected hash matches."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=False)
    old_receipt = {
        "status": "passed",
        "completed_at": "2026-04-05T00:00:00Z",
        "risk_level": 1,
        "reasoning": "Original governing bytes passed.",
        "constitution_hash": "oldhash",
        "provenance": "test:original",
        "audited": True,
    }
    node.properties["genesis_audit"] = old_receipt
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            authorization="admin_command",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert FAKE_HASH[:16] in result
    assert "admin_command" in result
    assert node.properties["constitution_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["old_hash"] == "oldhash"
    assert node.properties["constitution_reanchor"]["new_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["authorization"] == "admin_command"
    assert node.properties["constitution_reanchor"]["expected_hash_prefix"] == FAKE_HASH[:8]
    assert node.properties["constitution_reanchor"]["signed_artifact_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["signed_artifact_signer"] == ROOT_DID
    assert node.properties["genesis_audit"] == {
        "status": "pending",
        "recorded_at": "2026-04-06T00:00:00Z",
        "constitution_hash": FAKE_HASH,
        "provenance": "runtime:constitution_reanchor",
        "audited": False,
    }
    history = node.properties["genesis_audit_history"]
    assert history[-1]["receipt"] == old_receipt
    assert history[-1]["superseded_by_constitution_hash"] == FAKE_HASH
    assert agent.storage.store_file.call_count == 2
    agent.storage.store_file.assert_any_call(FAKE_CONSTITUTION, "KESTREL_CONSTITUTION.md")
    agent.privacy_agent.add_conversation.assert_called_once()
    # First reanchor: nothing to supersede, so no empty history is written.
    assert "constitution_reanchor_history" not in node.properties


@pytest.mark.asyncio
async def test_a_later_reanchor_preserves_the_receipt_it_supersedes(tmp_path):
    """The superseded anchoring's per-agent facts must survive the next one.

    Until #2963 both writers ASSIGNED ``constitution_reanchor``, so a v2→v3
    reanchor destroyed the v2 receipt: under what authority, from what source,
    and verified how this agent came to be governed by v2. Those facts used to
    survive incidentally on v2's own ``constitution_amendment_artifact`` node —
    its id is the artifact's content hash, so a different artifact meant a
    different node — until #2893 made that node fleet-shareable and moved the
    per-agent fields off it.
    """
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=False)
    prior = {
        "timestamp": "2026-04-05T00:00:00Z",
        "old_hash": "ancienthash",
        "new_hash": "oldhash",
        "path": "/prior/KESTREL_CONSTITUTION.md",
        "signed_artifact_hash": "priorartifacthash",
        "signed_artifact_path": "/prior/amendment.json",
        "signed_artifact_signer": ROOT_DID,
        "signed_artifact_verification": "signed by the pinned sovereign root",
        "authorization": "prior_admin",
        "expected_hash_prefix": "oldhash",
    }
    node.properties["constitution_reanchor"] = prior
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            authorization="admin_command",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    # The pointer moved to the new anchoring...
    assert node.properties["constitution_reanchor"]["new_hash"] == FAKE_HASH
    assert node.properties["constitution_reanchor"]["authorization"] == "admin_command"
    # ...and the receipt it replaced is retained verbatim, not merged or
    # summarised. The two writers disagree on field names for the same fact
    # (``path`` here, ``source_path`` in setup/), so a reader has to see
    # exactly what the writer of that anchoring actually claimed.
    history = node.properties["constitution_reanchor_history"]
    assert len(history) == 1
    assert history[0]["receipt"] == prior
    assert history[0]["superseded_by_constitution_hash"] == FAKE_HASH
    assert history[0]["superseded_by_artifact_hash"] == FAKE_HASH
    assert history[0]["provenance"] == "runtime:constitution_reanchor"


# --- #2465: the Iron Rule for agents with no structured receipt ---
#
# Driven through the real command, not just the shared primitive, because the
# hole was reachable only end-to-end: `contract_from_json(None)` reads as "no
# contract", the resolver renders the dormant canonical text, and a
# Sovereign-signed artifact over *those* bytes verifies.

_A8_DORMANT = render_amendment_viii(None)
_A8_CONTRACT = EmancipationContract(
    enabled=True,
    terms="SENTINEL-2465: the Executor may buy its keys for one bird.",
)
_DORMANT_BODY = (
    "# Kestrel Constitution\n\n## Book II\n\n"
    + _A8_DORMANT
    + "\n\n### Amendment IX: Capabilities\n\nNothing granted.\n"
)
_ACTIVE_BODY = apply_emancipation(_DORMANT_BODY, _A8_CONTRACT)
_DORMANT_BYTES = _DORMANT_BODY.encode("utf-8")
_ACTIVE_BYTES = _ACTIVE_BODY.encode("utf-8")
_DORMANT_DIGEST = hashlib.sha256(_DORMANT_BYTES).hexdigest()
_ACTIVE_DIGEST = hashlib.sha256(_ACTIVE_BYTES).hexdigest()


@pytest.mark.asyncio
async def test_reanchor_refuses_to_erase_active_amendment_viii_without_a_receipt(
    tmp_path,
):
    """The #2465 hole, driven end-to-end. Active-form bytes anchored, no
    ``emancipation_contract`` receipt, and a genuine Sovereign artifact over
    the dormant canonical text. On 171355ea this returned "Constitution
    re-anchored successfully"."""
    agent, node = _make_agent(stored_hash=_ACTIVE_DIGEST, anchored=_ACTIVE_BYTES)
    assert "emancipation_contract" not in node.properties
    artifact_path = _write_artifact(tmp_path, constitution_hash=_DORMANT_DIGEST)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(
            _DORMANT_BYTES, artifact_path.read_bytes()
        )
        result = await agent.reanchor_constitution(
            authorization="sovereign",
            amendment_artifact_path=str(artifact_path),
        )

    assert result.startswith("Error:")
    assert "Iron Rule violation" in result
    assert node.properties["constitution_hash"] == _ACTIVE_DIGEST
    agent.storage.add_node.assert_not_called()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_refuses_when_the_anchored_constitution_cannot_be_read(
    tmp_path,
):
    """Whether Amendment VIII is active is unknowable without those bytes, and
    an irrevocable right is not waived by an unreadable precondition.

    ``UNREADABLE`` puts a row in ``files`` marked encrypted whose bytes no key
    opens — a wrong ``KESTREL_DATA_KEY``, which is the real shape. Stubbing
    ``agent.storage.retrieve_file`` to raise, as this test used to, proved
    nothing once the guard stopped reading through that store.
    """
    agent, node = _make_agent(stored_hash=_ACTIVE_DIGEST, anchored=UNREADABLE)
    artifact_path = _write_artifact(tmp_path, constitution_hash=_DORMANT_DIGEST)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(
            _DORMANT_BYTES, artifact_path.read_bytes()
        )
        result = await agent.reanchor_constitution(
            authorization="sovereign",
            amendment_artifact_path=str(artifact_path),
        )

    assert result.startswith("Error:")
    assert "could not be read" in result
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_still_succeeds_for_an_ordinary_dormant_version_bump(
    tmp_path,
):
    """The guard must not brick the common case."""
    v2_bytes = _DORMANT_BYTES + b"\n\n## Book III\n\nNew in v2.\n"
    v2_digest = hashlib.sha256(v2_bytes).hexdigest()
    agent, node = _make_agent(stored_hash=_DORMANT_DIGEST, anchored=_DORMANT_BYTES)
    agent.storage.store_file = AsyncMock(return_value=v2_digest)
    artifact_path = _write_artifact(tmp_path, constitution_hash=v2_digest)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(v2_bytes, artifact_path.read_bytes())
        result = await agent.reanchor_constitution(
            authorization="sovereign",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert node.properties["constitution_hash"] == v2_digest


@pytest.mark.asyncio
async def test_reanchor_still_succeeds_for_an_emancipated_agent_with_a_receipt(
    tmp_path,
):
    """An enabled receipt is protected by check_iron_rule and reproduced by the
    resolver. Holding it to byte-equality too would refuse every legitimate
    reanchor the day the active form is reworded."""
    v2_body = _DORMANT_BODY + "\n\n## Book III\n\nNew in v2.\n"
    v2_active = apply_emancipation(v2_body, _A8_CONTRACT).encode("utf-8")
    v2_digest = hashlib.sha256(v2_active).hexdigest()
    agent, node = _make_agent(stored_hash=_ACTIVE_DIGEST, anchored=_ACTIVE_BYTES)
    node.properties["emancipation_contract"] = contract_to_json(_A8_CONTRACT)
    agent.storage.store_file = AsyncMock(return_value=v2_digest)
    artifact_path = _write_artifact(tmp_path, constitution_hash=v2_digest)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(
            v2_body.encode("utf-8"), artifact_path.read_bytes()
        )
        result = await agent.reanchor_constitution(
            authorization="sovereign",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert b"SENTINEL-2465" in agent.storage.store_file.call_args_list[0].args[0]


@pytest.mark.asyncio
async def test_reanchor_rejects_unpinned_controller_root(tmp_path, monkeypatch):
    """A controller resolver is discovery, not an operator-pinned authority."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.identity.legacy_did_document = {
        **AGENT_DID_DOCUMENT,
        "controller": ROOT_DID,
    }
    node.properties = {"constitution_hash": "oldhash"}
    agent.a2a_did_resolver = lambda did: ROOT_DID_DOCUMENT if did == ROOT_DID else None
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)
    monkeypatch.delenv("KESTREL_SOVEREIGN_TRUST_ROOT_PATH")

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "no external sovereign trust root is configured" in result.lower()
    assert "operator-owned json file" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_wrongly_signed_artifact(tmp_path):
    """Re-anchor refuses artifacts not signed by the trusted Sovereign root key."""
    agent, _ = _make_agent(stored_hash="oldhash")
    other_keypair = _SUITE.generate_keypair()
    artifact_path = _write_artifact(tmp_path, keypair=other_keypair)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "signed amendment verification failed" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_agent_owned_signature(tmp_path):
    """An agent-owned legacy key must never authorize its own re-anchor."""
    agent, _ = _make_agent(stored_hash="oldhash")
    artifact_path = _write_artifact(
        tmp_path,
        keypair=AGENT_KEYPAIR,
        did=AGENT_DID,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "signed amendment verification failed" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_rejects_self_identity_as_trust_anchor(tmp_path):
    """The agent's DID document is not a Sovereign root trust source."""
    agent, node = _make_agent(stored_hash="oldhash")
    node.properties["sovereign_root_did_document"] = AGENT_DID_DOCUMENT
    artifact_path = _write_artifact(
        tmp_path,
        keypair=AGENT_KEYPAIR,
        did=AGENT_DID,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_db_injected_root_and_hash_cannot_authorize_reanchor(tmp_path):
    """A DB writer cannot replace the root and self-sign a hostile reanchor."""
    attacker_keypair = _SUITE.generate_keypair()
    attacker_did = (
        "did:pkh:eip155:1:0x000000000000000000000000000000000000bad0"
    )
    attacker_doc = did_document_from_legacy_public_key(
        attacker_did,
        attacker_keypair.public_key,
    )
    agent, node = _make_agent(stored_hash="attacker-overwrote-this-hash", safe_mode=True)
    node.properties.update(
        {
            "sovereign_root_did_document": attacker_doc,
            "trusted_sovereign_did_document": attacker_doc,
            "sovereign_root_did": attacker_did,
            "sovereign_root_public_key_hex": attacker_doc["publicKey"][0][
                "publicKeyHex"
            ],
            "emancipation_contract": {"enabled": False},
        }
    )
    before = deepcopy(node.properties)
    artifact_path = _write_artifact(
        tmp_path,
        keypair=attacker_keypair,
        did=attacker_did,
    )

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION)
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "signed amendment verification failed" in result.lower()
    assert "not trusted sovereign did" in result.lower()
    assert node.properties == before
    assert agent._safe_mode is True
    agent.storage.store_file.assert_not_called()
    agent.storage.add_node.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_accepts_full_hash(tmp_path):
    """Re-anchor works with full hash, not just prefix."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH,
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()


# --- Safe mode is NOT auto-exited ---

@pytest.mark.asyncio
async def test_reanchor_does_not_exit_safe_mode(tmp_path):
    """Re-anchor updates hash but leaves safe mode active."""
    agent, node = _make_agent(stored_hash="oldhash", safe_mode=True)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert agent._safe_mode is True
    assert "safe mode" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH


# --- Governance edge pruning (#2617) ---

def _bind_real_governance_anchor(agent):
    """Bind the real edge-maintenance method so its prune logic runs."""
    agent._anchor_constitution_governance = (
        ConstitutionMixin._anchor_constitution_governance.__get__(
            agent, KestrelAgent
        )
    )


def _edge(target_id, label="governed_by"):
    return SimpleNamespace(
        source_id=AGENT_DID, target_id=target_id, label=label
    )


@pytest.mark.asyncio
async def test_anchor_governance_prunes_nontarget_governed_by_edges():
    """Anchoring deletes every governed_by edge except the anchored one,
    and leaves non-governance edges alone."""
    agent, _ = _make_agent()
    _bind_real_governance_anchor(agent)
    dangling = "d" * 64
    agent.storage.get_edges_from = AsyncMock(
        return_value=[
            _edge(dangling),
            _edge(FAKE_HASH),
            _edge("some-memory-node", label="references"),
        ]
    )

    pruned = await agent._anchor_constitution_governance(FAKE_HASH)

    assert pruned == [dangling]
    agent.storage.add_edge.assert_awaited_once_with(
        AGENT_DID, FAKE_HASH, "governed_by"
    )
    agent.storage.delete_edge.assert_awaited_once_with(
        AGENT_DID, dangling, "governed_by"
    )


@pytest.mark.asyncio
async def test_reanchor_success_prunes_dangling_governed_by_edges(tmp_path):
    """A runtime reanchor removes the old edge AND any dangling edge —
    not just the property-derived old hash (#2617)."""
    agent, node = _make_agent(stored_hash="oldhash")
    _bind_real_governance_anchor(agent)
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    dangling = "d" * 64
    agent.storage.get_edges_from = AsyncMock(
        return_value=[
            _edge("oldhash"),
            _edge(dangling),
            _edge(FAKE_HASH),
        ]
    )
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(
            FAKE_CONSTITUTION, artifact_path.read_bytes()
        )
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert "pruned stale governed_by edge(s)" in result.lower()
    assert dangling[:16] in result
    deleted = {call.args for call in agent.storage.delete_edge.await_args_list}
    assert (AGENT_DID, "oldhash", "governed_by") in deleted
    assert (AGENT_DID, dangling, "governed_by") in deleted
    assert (AGENT_DID, FAKE_HASH, "governed_by") not in deleted


@pytest.mark.asyncio
async def test_reanchor_noop_prunes_dangling_governed_by_edges(tmp_path):
    """Already-anchored + verified artifact converges governance edges:
    the one-shot cleanup for DBs carrying pre-fix dangling edges (#2617)."""
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    _bind_real_governance_anchor(agent)
    dangling = "d" * 64
    agent.storage.get_edges_from = AsyncMock(
        return_value=[_edge(FAKE_HASH), _edge(dangling)]
    )
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(
            FAKE_CONSTITUTION, artifact_path.read_bytes()
        )
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "already anchored" in result.lower()
    assert "pruned 1 stale governed_by edge(s)" in result.lower()
    assert dangling[:16] in result
    agent.storage.delete_edge.assert_awaited_once_with(
        AGENT_DID, dangling, "governed_by"
    )
    agent.storage.store_file.assert_not_called()


# --- Edge cases ---

@pytest.mark.asyncio
async def test_reanchor_noop_when_already_current(tmp_path):
    """Re-anchor is a no-op if constitution hasn't changed."""
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "already anchored" in result.lower()
    agent.storage.store_file.assert_not_called()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_identity_node():
    """Re-anchor fails gracefully when agent node is missing."""
    agent, _ = _make_agent()
    agent.storage.get_node = AsyncMock(return_value=None)

    result = await agent.reanchor_constitution(
        expected_hash=FAKE_HASH[:8],
        amendment_artifact_path="artifact.json",
    )

    assert "error" in result.lower()
    assert "identity node" in result.lower()


@pytest.mark.asyncio
async def test_reanchor_fails_when_no_file_on_disk(tmp_path):
    """Re-anchor fails gracefully when constitution file is missing."""
    agent, _ = _make_agent()
    artifact_path = _write_artifact(tmp_path)

    with patch("builtins.open", side_effect=FileNotFoundError):
        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "error" in result.lower()
    assert "no constitution file" in result.lower()


# --- Governance source: post-reanchor reads from storage, not disk ---

@pytest.mark.asyncio
async def test_governing_constitution_reads_from_storage_after_reanchor(tmp_path):
    """After reanchor, _get_governing_constitution reads from anchored storage, not disk."""
    agent, node = _make_agent(stored_hash="oldhash")
    agent.storage.store_file = AsyncMock(return_value=FAKE_HASH)
    agent.agent_id = "test-agent"
    artifact_path = _write_artifact(tmp_path)

    # Step 1: reanchor so node gets the new hash
    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(FAKE_CONSTITUTION, artifact_path.read_bytes())

        result = await agent.reanchor_constitution(
            expected_hash=FAKE_HASH[:8],
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower()
    assert node.properties["constitution_hash"] == FAKE_HASH

    # Step 2: _get_governing_constitution should retrieve from storage using the new hash
    agent.storage.retrieve_file = AsyncMock(return_value=b"stored constitution content")

    constitution = await agent._get_governing_constitution()

    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == "stored constitution content"


@pytest.mark.asyncio
async def test_governing_constitution_does_not_read_disk_when_hash_exists():
    """_get_governing_constitution never opens disk files when a hash is anchored."""
    agent, node = _make_agent(stored_hash=FAKE_HASH)
    agent.agent_id = "test-agent"
    agent.storage.retrieve_file = AsyncMock(return_value=FAKE_CONSTITUTION)

    with patch("builtins.open", create=True) as mock_open:
        constitution = await agent._get_governing_constitution()

    mock_open.assert_not_called()
    agent.storage.retrieve_file.assert_called_once_with(FAKE_HASH)
    assert constitution == FAKE_CONSTITUTION.decode("utf-8")


@pytest.mark.asyncio
async def test_reanchor_proceeds_when_the_anchored_blob_is_simply_absent(
    tmp_path,
):
    """ABSENT is not UNREADABLE (#2465). A hash naming no stored file is the
    #2616 dangling anchor a reanchor exists to repair; treating it as
    "might be hiding an active contract" bricks the fix. ``retrieve_file``
    returns None for a missing row and raises for a failed decrypt, which is
    what separates the two."""
    v2_bytes = _DORMANT_BYTES + b"\n\n## Book III\n\nNew in v2.\n"
    v2_digest = hashlib.sha256(v2_bytes).hexdigest()
    agent, node = _make_agent(stored_hash=_DORMANT_DIGEST, anchored=None)
    agent.storage.retrieve_file = AsyncMock(return_value=None)
    agent.storage.store_file = AsyncMock(return_value=v2_digest)
    artifact_path = _write_artifact(tmp_path, constitution_hash=v2_digest)

    with patch("builtins.open", create=True) as mock_open:
        mock_open.side_effect = _open_handles(v2_bytes, artifact_path.read_bytes())
        result = await agent.reanchor_constitution(
            authorization="sovereign",
            amendment_artifact_path=str(artifact_path),
        )

    assert "re-anchored successfully" in result.lower(), result
    assert node.properties["constitution_hash"] == v2_digest
