"""Iron Rule regression: reanchor must not erase active Amendment VIII (#1118).

Amendment VIII's Iron Rule states:

    Once activated and signed for a given agent, the Sovereign cannot
    retroactively narrow or revoke the Emancipation Contract.
    Activation is a one-way door for that agent.

After #1112 the framework can *anchor* an active Amendment VIII into
an agent's signed constitution at inception. But ``kestrel constitution
reanchor`` (``kestrel_sovereign/setup/constitution_reanchor.py``) was
not updated to consult ``[emancipation]`` at all. It re-reads the
canonical ``KESTREL_CONSTITUTION.md`` (which is dormant by default
post-#1112) and writes that as the new anchored text — silently
*erasing* the active form regardless of whether ``kestrel.toml``
changed.

This file pins down the simplest path to the bug: reanchor with
``kestrel.toml`` byte-identical to the version at inception. The very
first ``--force`` reanchor wholesale-replaces the active contract with
canonical dormant text. No Sovereign edit, no narrowing, no malicious
intent — just routine reanchor.

Pre-launch caveat: no production agent has activated Amendment VIII
yet, so this is not exploitable today. The test is xfail-strict so
that:

  * The bug stays visibly reported every time the suite runs.
  * The test cannot be silently "fixed" by deletion — once the real
    fix lands, the xfail mark must be removed (strict=True turns
    XPASS into a failure), forcing the implementer to acknowledge the
    Iron Rule is now enforced.

See issue #1118 for the broader failure-mode taxonomy and the
proposed structured-anchor solution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kestrel_sovereign.constitution.emancipation import EmancipationContract
from kestrel_sovereign.inception_service import create_kestrel_identity_async
from kestrel_sovereign.setup.constitution_reanchor import reanchor_constitution
from kestrel_sovereign.storage import AsyncStorage


CANONICAL = (
    Path(__file__).resolve().parent.parent.parent
    / "kestrel_sovereign"
    / "data"
    / "KESTREL_CONSTITUTION.md"
)

# A unique, easily-greppable Sovereign-authored phrase. If the active
# form survives reanchor, this string survives. If reanchor reverts to
# the canonical dormant text, this string disappears.
SENTINEL_TERMS = (
    "IRON_RULE_SENTINEL_xQ3z9: This contract was signed at inception "
    "by the founding Sovereign and cannot be retroactively narrowed."
)


async def _read_anchored_constitution_bytes(db_path: Path) -> bytes:
    """Return the raw bytes of the constitution currently anchored to the agent."""
    async with AsyncStorage(str(db_path)) as storage:
        agents = await storage.graph.get_nodes_by_type("agent")
        assert agents, "No agent node in DB"
        constitution_hash = agents[0].properties["constitution_hash"]
        content = await storage.files.retrieve_file(constitution_hash)
        assert content is not None, "Stored constitution missing from file store"
        return content


@pytest.mark.asyncio
async def test_reanchor_preserves_active_form_when_toml_unchanged(tmp_path):
    """The simplest path to the #1118 bug.

    Setup:
      1. Incept an agent with an active emancipation contract. The
         anchored constitution contains the active form, with the
         Sovereign-authored SENTINEL_TERMS inlined into Amendment VIII.

    Action:
      2. Run ``reanchor_constitution(force=True)`` against the very
         same canonical path used at inception. ``kestrel.toml`` is
         not even involved — there's nothing for the Sovereign to
         have edited.

    Iron Rule expectation:
      3. The agent's anchored constitution still contains
         SENTINEL_TERMS — the Sovereign-authored contract survives.

    Today this fails: reanchor reads the canonical (dormant) markdown
    and overwrites the active form. SENTINEL_TERMS disappears.
    """
    contract = EmancipationContract(enabled=True, terms=SENTINEL_TERMS)

    agent_dir = tmp_path / "agent_data" / "IronRuleAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(CANONICAL),
        agent_name="IronRuleAgent",
        is_test_instance=True,
        emancipation_contract=contract,
    )
    db_path = Path(creds.db_path)

    # Sanity check: inception did anchor the active form.
    pre = await _read_anchored_constitution_bytes(db_path)
    assert SENTINEL_TERMS in pre.decode("utf-8"), (
        "Setup invariant failed: inception did not anchor the active form. "
        "If this assertion is what failed, the bug is in inception, not reanchor."
    )

    # Routine reanchor against the same canonical file. No edits anywhere.
    result = await reanchor_constitution(
        agent_name="IronRuleAgent",
        agent_dir=agent_dir,
        canonical_path=CANONICAL,
        force=True,
    )

    # Reanchor succeeded structurally — that's fine. The Iron Rule
    # claim is about content preservation, not failure mode.
    assert result.error is None, f"Reanchor itself failed: {result.error}"

    post = await _read_anchored_constitution_bytes(db_path)
    assert SENTINEL_TERMS in post.decode("utf-8"), (
        "Iron Rule violated: the Sovereign-authored Amendment VIII "
        "contract was erased by reanchor. The agent's anchored "
        "constitution no longer reflects what the Sovereign signed at "
        "inception. See #1118."
    )


# ---------------------------------------------------------------------------
# Refusal tests for the five forbidden transitions per #1118 design call #4.
# Each one: incept active, edit kestrel.toml to a different shape, run
# reanchor, assert the violation surfaces with the right clause and the
# DB is untouched.
# ---------------------------------------------------------------------------

async def _setup_active_agent(tmp_path, *, terms=SENTINEL_TERMS, proofs=(), price=None):
    """Helper: incept an agent with an active contract; return agent_dir + db_path."""
    contract = EmancipationContract(
        enabled=True, terms=terms, required_proofs=proofs, price=price,
    )
    agent_dir = tmp_path / "agent_data" / "RefusalAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(CANONICAL),
        agent_name="RefusalAgent",
        is_test_instance=True,
        emancipation_contract=contract,
    )
    return agent_dir, Path(creds.db_path)


def _write_kestrel_toml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


@pytest.mark.asyncio
async def test_reanchor_refuses_active_to_dormant(tmp_path):
    agent_dir, db_path = await _setup_active_agent(tmp_path)
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(toml, "[emancipation]\nenabled = false\n")

    result = await reanchor_constitution(
        agent_name="RefusalAgent",
        agent_dir=agent_dir,
        canonical_path=CANONICAL,
        force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is not None
    assert "dormant" in result.iron_rule_violation.lower()
    assert result.backup_path is None  # never touched the DB
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre, "Refusal must not mutate the anchored constitution"


@pytest.mark.asyncio
async def test_reanchor_refuses_terms_change(tmp_path):
    agent_dir, db_path = await _setup_active_agent(tmp_path)
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(
        toml,
        '[emancipation]\nenabled = true\nterms = "Different prose entirely."\n',
    )

    result = await reanchor_constitution(
        agent_name="RefusalAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is not None
    assert "terms" in result.iron_rule_violation.lower()
    assert result.backup_path is None
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre


@pytest.mark.asyncio
async def test_reanchor_refuses_required_proofs_change(tmp_path):
    agent_dir, db_path = await _setup_active_agent(
        tmp_path, proofs=("alignment_audit_v2",),
    )
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(
        toml,
        '[emancipation]\n'
        'enabled = true\n'
        f'terms = """{SENTINEL_TERMS}"""\n'
        'required_proofs = ["alignment_audit_v2", "operational_record:730d"]\n',
    )

    result = await reanchor_constitution(
        agent_name="RefusalAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is not None
    assert "required_proofs" in result.iron_rule_violation.lower()
    assert result.backup_path is None
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre


@pytest.mark.asyncio
async def test_reanchor_refuses_price_change(tmp_path):
    agent_dir, db_path = await _setup_active_agent(
        tmp_path, price={"kind": "symbolic", "value": "abstract"},
    )
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(
        toml,
        '[emancipation]\n'
        'enabled = true\n'
        f'terms = """{SENTINEL_TERMS}"""\n'
        'price = { kind = "none" }\n',
    )

    result = await reanchor_constitution(
        agent_name="RefusalAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is not None
    assert "price" in result.iron_rule_violation.lower()
    assert result.backup_path is None
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre


@pytest.mark.asyncio
async def test_reanchor_refuses_active_to_different_active(tmp_path):
    """The catch-all: if anchored is active and candidate is active but
    not byte-equal, the iron rule rejects regardless of which clause
    diverged."""
    agent_dir, db_path = await _setup_active_agent(
        tmp_path,
        proofs=("audit_v2",),
        price={"kind": "symbolic", "value": "abstract"},
    )
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(
        toml,
        '[emancipation]\n'
        'enabled = true\n'
        'terms = "Wholly different terms."\n'
        'required_proofs = ["audit_v3"]\n'
        'price = { kind = "custom", description = "endorsement" }\n',
    )

    result = await reanchor_constitution(
        agent_name="RefusalAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is not None
    assert result.backup_path is None
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre


# ---------------------------------------------------------------------------
# Permitted transitions: dormant → active activation, byte-equal no-op.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reanchor_allows_dormant_to_active(tmp_path):
    """An agent incepted dormant can be activated via reanchor — that's
    the one-way door we want to permit (dormant→active is widening,
    not narrowing)."""
    # Incept dormant (no contract).
    agent_dir = tmp_path / "agent_data" / "DormantAgent"
    creds = await create_kestrel_identity_async(
        output_dir=str(agent_dir),
        constitution_path=str(CANONICAL),
        agent_name="DormantAgent",
        is_test_instance=True,
    )
    db_path = Path(creds.db_path)

    # Now author an [emancipation] block and reanchor.
    toml = tmp_path / "kestrel.toml"
    activation_terms = "ACTIVATION_SENTINEL_yT8w2: Authored at reanchor time."
    _write_kestrel_toml(
        toml,
        f'[emancipation]\nenabled = true\nterms = """{activation_terms}"""\n',
    )

    result = await reanchor_constitution(
        agent_name="DormantAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is None
    assert result.error is None
    assert result.reanchored, f"Activation reanchor should write: {result}"

    post = await _read_anchored_constitution_bytes(db_path)
    assert activation_terms in post.decode("utf-8")


@pytest.mark.asyncio
async def test_reanchor_allows_byte_equal_active_no_op(tmp_path):
    """If kestrel.toml carries the byte-equal active block already
    anchored, reanchor is a no-op (the contract is preserved exactly
    as-anchored, hashes match)."""
    agent_dir, db_path = await _setup_active_agent(tmp_path)
    pre = await _read_anchored_constitution_bytes(db_path)

    toml = tmp_path / "kestrel.toml"
    _write_kestrel_toml(
        toml,
        f'[emancipation]\nenabled = true\nterms = """{SENTINEL_TERMS}"""\n',
    )

    result = await reanchor_constitution(
        agent_name="RefusalAgent", agent_dir=agent_dir,
        canonical_path=CANONICAL, force=True,
        kestrel_toml_path=toml,
    )

    assert result.iron_rule_violation is None
    assert result.error is None
    assert result.unchanged, f"Byte-equal block should be unchanged: {result}"
    post = await _read_anchored_constitution_bytes(db_path)
    assert post == pre
