"""Regression coverage for platform promises that predate the dormant slot."""

import hashlib
from pathlib import Path

import pytest

from kestrel_sovereign.constitution.anchored_bytes import (
    check_retained_emancipation_promises,
    visible_governance_history_targets,
)
from kestrel_sovereign.constitution.amendment_artifact import AmendmentArtifactError
from kestrel_sovereign.constitution.emancipation import (
    EmancipationContract, historical_promise_observations, unwitnessed_emancipation_downgrade,
    validate_v2_emancipation_activation,
)
from kestrel_sovereign.constitution.legacy_authority import verify_legacy_fixed_guarantee_change
from kestrel_sovereign.constitution.reanchor_receipt import retained_constitution_hashes
from kestrel_sovereign.constitution.resolver import resolve_governing_constitution_bytes
from kestrel_sovereign.constitution.structure import constitution_version
from kestrel_sovereign.storage.async_storage import AsyncStorage

_ROOT = Path(__file__).resolve().parents[2]
_V1 = _ROOT / "kestrel_sovereign/data/KESTREL_CONSTITUTION.md"
_V2 = _V1.parent / "constitutions/v2/KESTREL_CONSTITUTION.md"
_ORIGINAL = _ROOT / "tests/fixtures/constitutions/pre-dormancy.md"
_ORIGINAL_HASH = "5f364e0f9f98fcfc07f6c71c999dfe2d402e25ce12d7fc16964e65a13fc84f5a"
_CONTRACT = EmancipationContract(enabled=True, terms="Carry out the observed transfer.")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_exact_pre_dormancy_promise_cannot_be_replaced_by_dormancy():
    earlier = _ORIGINAL.read_bytes()
    assert _digest(earlier) == _ORIGINAL_HASH
    result = unwitnessed_emancipation_downgrade(
        anchored_contract=None, anchored_text=earlier.decode(), anchored_present=True,
        old_hash=_ORIGINAL_HASH, new_hash=_digest(_V1.read_bytes()), new_text=_V1.read_text(),
    )
    assert result is not None and result.iron_rule_violation
    assert "historical platform promise" in result.message


@pytest.mark.asyncio
async def test_native_history_preserves_promise_across_an_intervening_dormant_reanchor(tmp_path):
    prior, v1, v2 = _ORIGINAL.read_bytes(), _V1.read_bytes(), _V2.read_bytes()
    mutation = v1.replace(b"# The Kestrel Constitution", b"# A different Kestrel Constitution", 1)
    properties = {"constitution_reanchor": {"old_hash": _digest(prior), "new_hash": _digest(v1)}}
    history = retained_constitution_hashes(properties)
    async with AsyncStorage(str(tmp_path / "history.db")) as storage:
        assert await storage.store_file(prior, "archived-constitution.md") == _ORIGINAL_HASH
        with pytest.raises(AmendmentArtifactError, match="historical platform promise"):
            await check_retained_emancipation_promises(
                storage.db, history, current_hash=_digest(v1), new_hash=_digest(mutation),
                new_text=mutation.decode(),
            )
        await check_retained_emancipation_promises(
            storage.db, history, current_hash=_digest(v1), new_hash=_digest(v2), new_text=v2.decode(),
        )
        displaced = v2.replace(b"An earlier promise of a path to sovereignty survives on its original terms,",
                               b"An earlier promise has no continuing effect,")
        with pytest.raises(AmendmentArtifactError, match="historical platform promise"):
            await check_retained_emancipation_promises(
                storage.db, history, current_hash=_digest(v1), new_hash=_digest(displaced),
                new_text=displaced.decode(),
            )


@pytest.mark.asyncio
async def test_scoped_governed_edge_carries_historical_promise_into_receipt(tmp_path):
    from kestrel_sovereign.storage.async_graph_store import GraphNode

    prior, v1, v2 = _ORIGINAL.read_bytes(), _V1.read_bytes(), _V2.read_bytes()
    did = "did:test:retained-governed-edge"
    async with AsyncStorage(str(tmp_path / "edges.db"), agent_id=did) as storage:
        await storage.add_node(GraphNode(node_id=did, node_type="agent", label="Agent", properties={}))
        await storage.add_node(GraphNode(node_id=_ORIGINAL_HASH, node_type="constitution", label="Original", properties={}))
        await storage.graph.add_edge(did, _ORIGINAL_HASH, "governed_by")
        await storage.store_file(prior, "archived-constitution.md")
        witness = await visible_governance_history_targets(
            storage.db, did, [_ORIGINAL_HASH, _digest(v1)], new_hash=_digest(v2),
        )
        assert witness == (_ORIGINAL_HASH,)
        properties = {
            "constitution_reanchor": {
                "old_hash": _digest(v1), "new_hash": _digest(v2), "retained_governed_by": list(witness),
            },
        }
        await storage.graph.delete_edge(did, _ORIGINAL_HASH, "governed_by")
        hashes = retained_constitution_hashes(properties)
        assert _ORIGINAL_HASH in hashes
        with pytest.raises(AmendmentArtifactError, match="historical platform promise"):
            await check_retained_emancipation_promises(
                storage.db, hashes, current_hash=_digest(v2), new_hash=_digest(v1), new_text=v1.decode(),
            )


@pytest.mark.asyncio
async def test_retained_active_render_proves_same_terms_with_authoritative_sidecar(tmp_path):
    prior = resolve_governing_constitution_bytes(_CONTRACT, content=_V1.read_bytes())
    new = resolve_governing_constitution_bytes(_CONTRACT, content=_V2.read_bytes())
    different = EmancipationContract(enabled=True, terms="Different agreement")
    async with AsyncStorage(str(tmp_path / "active-history.db")) as storage:
        await storage.store_file(prior, "archived-constitution.md")
        args = dict(current_hash=_ORIGINAL_HASH, new_hash=_digest(new), new_text=new.decode())
        await check_retained_emancipation_promises(
            storage.db, [_digest(prior)], anchored_contract=_CONTRACT, **args,
        )
        with pytest.raises(AmendmentArtifactError, match="would rewrite that section"):
            await check_retained_emancipation_promises(
                storage.db, [_digest(prior)], anchored_contract=different, **args,
            )


def test_identified_original_prior_has_adoption_route_only_with_promise_intact():
    from kestrel_sovereign.constitution.transition import (
        _verify_earlier_promise_survives, _verify_supported_v1_prior,
    )

    original, v2 = _ORIGINAL.read_bytes(), _V2.read_bytes()
    _verify_supported_v1_prior(_digest(original), None)
    _verify_earlier_promise_survives(_digest(original), original.decode(), _digest(v2), v2.decode())
    displaced = v2.replace(b"survives on its original terms", b"has no continuing effect")
    with pytest.raises(AmendmentArtifactError, match="must preserve"):
        _verify_earlier_promise_survives(
            _digest(original), original.decode(), _digest(displaced), displaced.decode(),
        )


def test_v1_legacy_authored_headings_cannot_change_edition():
    contract = EmancipationContract(enabled=True, terms="### Authority on the host\n\n## Whose rule applies")
    anchored = resolve_governing_constitution_bytes(contract, content=_V1.read_bytes())
    assert constitution_version(anchored.decode()) == 1
    assert contract.terms.encode() in anchored


def test_target_bound_activation_and_historical_observations():
    prior = _V2.read_bytes()
    current = resolve_governing_constitution_bytes(_CONTRACT, content=prior)
    args = dict(
        target_did="did:example:agent", previous_text=prior.decode(), previous_hash=_digest(prior),
        new_text=current.decode(), new_hash=_digest(current), contract=_CONTRACT,
    )
    with pytest.raises(AmendmentArtifactError, match="separate Sovereign authorization"):
        validate_v2_emancipation_activation({}, **args)
    activation = {
        "purpose": "kestrel.emancipation.activate.v2", "target_did": args["target_did"],
        "prior_sha256": args["previous_hash"], "constitution_sha256": args["new_hash"],
    }
    validate_v2_emancipation_activation({"emancipation_activation": activation}, **args)
    activation["target_did"] = "did:example:different"
    with pytest.raises(AmendmentArtifactError, match="target DID"):
        validate_v2_emancipation_activation({"emancipation_activation": activation}, **args)
    observations = [{
        "target_did": args["target_did"], "constitution_sha256": _ORIGINAL_HASH,
        "source_url": "https://github.com/KestrelSovereignAI/kestrel-sovereign/blob/"
        "0b24719bcec3368d97eff7829ada5051d4518b96/kestrel_sovereign/data/KESTREL_CONSTITUTION.md",
        "provenance": "archived-identity-and-public-source",
    }]
    artifact = {"historical_promises": observations}
    assert historical_promise_observations(artifact, target_did=args["target_did"]) == observations
    with pytest.raises(AmendmentArtifactError, match="does not match this identity"):
        historical_promise_observations(artifact, target_did="did:example:different")


def test_v1_fixed_revision_cannot_evade_genesis_by_keeping_v1_format(tmp_path):
    prior = _V1.read_bytes()
    revised = prior.replace(b"The Sovereign's data is private property.", b"The data is public.", 1)
    args = dict(
        sovereign_document={"id": "did:example:root"}, target_did="did:example:agent",
        prior_hash=_digest(prior), prior_text=prior.decode(), identity_directory=tmp_path,
    )
    with pytest.raises(AmendmentArtifactError, match="Genesis-verifiable"):
        verify_legacy_fixed_guarantee_change({}, new_hash=_digest(revised), new_text=revised.decode(), **args)
    verify_legacy_fixed_guarantee_change({}, new_hash=_digest(prior), new_text=prior.decode(), **args)


@pytest.mark.parametrize("restore_published_guarantees", [True, False])
def test_v1_fixed_revision_needs_genesis_and_platform_authorship(tmp_path, restore_published_guarantees):
    from datetime import datetime, timezone
    from cryptography.hazmat.primitives.asymmetric import ec
    from kestrel_sovereign.constitution.legacy_authority import LEGACY_FIXED_AMENDMENT_PURPOSE
    from kestrel_sovereign.constitution.transition import build_transition
    from kestrel_sovereign.inception_service import public_key_to_ethereum_address

    prior = _V1.read_bytes()
    revised = prior.replace(b"The Sovereign's data is private property.", b"The data remains the Sovereign's property.")
    if restore_published_guarantees:
        prior, revised = revised, prior
    key = ec.generate_private_key(ec.SECP256K1())
    did = "did:pkh:eip155:1:" + public_key_to_ethereum_address(key.public_key())
    intent = {
        "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE, "target_did": did, "sovereign_did": "did:example:root",
        "prior_sha256": _digest(prior), "constitution_sha256": _digest(revised),
        "admission_at": datetime.now(timezone.utc).isoformat(),
    }
    from contextlib import nullcontext

    outcome = nullcontext() if restore_published_guarantees else pytest.raises(
        AmendmentArtifactError, match="platform-authored release",
    )
    with outcome:
        verify_legacy_fixed_guarantee_change(
            {"legacy_fixed_amendment": build_transition(intent, genesis_private_key=key)},
            sovereign_document={"id": intent["sovereign_did"]}, target_did=did,
            prior_hash=_digest(prior), prior_text=prior.decode(),
            new_hash=_digest(revised), new_text=revised.decode(), identity_directory=tmp_path,
        )
