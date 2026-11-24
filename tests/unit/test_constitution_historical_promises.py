"""Regression coverage for platform promises that predate the dormant slot."""

import hashlib
import json
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
from kestrel_sovereign.constitution.v1_releases import V1_CHECKOUTS, V1_RELEASES
from kestrel_sovereign.storage.async_storage import AsyncStorage

_ROOT = Path(__file__).resolve().parents[2]
_V1 = _ROOT / "kestrel_sovereign/data/KESTREL_CONSTITUTION.md"
_V2 = _V1.parent / "constitutions/v2/KESTREL_CONSTITUTION.md"
_ORIGINAL = _ROOT / "tests/fixtures/constitutions/pre-dormancy.md"
_ORIGINAL_HASH = "5f364e0f9f98fcfc07f6c71c999dfe2d402e25ce12d7fc16964e65a13fc84f5a"
_CONTRACT = EmancipationContract(enabled=True, terms="Carry out the observed transfer.")
_ACTIVE_RELEASES = [release for release in V1_RELEASES if release.supports_activation]


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


@pytest.mark.parametrize(
    ("commit", "rule", "amendment_heading", "promise_opening"),
    [
        ("34eacb733", "FOUNDING_ARTICLES", "Article V: The Amendment Process", "## Article IV"),
        ("763e6ac2c", "RANKED_BOOKS", "Article V: The Amendment Process", "**The Right.**"),
        ("0b24719bc", "RANKED_BOOKS", "Article V: The Amendment Process", "**The Right.**"),
        ("84f4591ab", "RANKED_BOOKS", "Article V: The Amendment Process", None),
        ("f0672b394", "SOVEREIGN_FRAME", "The Amendment Process", None),
        ("a292b6418", "PLATFORM_AUTHORED_BOOKS", "The Amendment Process", None),
    ],
)
def test_authentic_v1_authority_and_promise_boundaries(commit, rule, amendment_heading, promise_opening):
    from kestrel_sovereign.constitution.emancipation import _prior_platform_promise
    from kestrel_sovereign.constitution.legacy_authority import _amendment_rule, _fixed_provisions
    from kestrel_sovereign.constitution.structure import markdown_sections_at_level

    release = next(release for release in V1_RELEASES if release.commit.startswith(commit))
    text = release.authenticated_text()
    headings = {heading for heading, _ in markdown_sections_at_level(text, 2)}
    assert amendment_heading in headings
    provisions = _fixed_provisions(text)
    assert _amendment_rule(release.digest, provisions, headings).name == rule
    assert _amendment_rule(None, provisions, headings).name == rule
    promise = _prior_platform_promise(text)
    if promise_opening is None:
        assert promise is None
    else:
        assert promise_opening in promise
        assert "one million troy ounces" in promise
        assert "## Book III" not in promise
        assert "### Amendment IX" not in promise
        assert not promise.endswith("---")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("release", _ACTIVE_RELEASES, ids=lambda release: release.commit[:9])
async def test_missing_personalized_history_prohibits_change_but_not_same_hash_repair(tmp_path, release):
    prior = resolve_governing_constitution_bytes(_CONTRACT, content=release.authenticated_text().encode())
    dormant = _V2.read_bytes()
    active = resolve_governing_constitution_bytes(_CONTRACT, content=dormant)
    async with AsyncStorage(str(tmp_path / "absent-history.db")) as storage:
        with pytest.raises(AmendmentArtifactError, match="Retained governing constitution .* unavailable"):
            await check_retained_emancipation_promises(
                storage.db, [_digest(prior)], current_hash=_ORIGINAL_HASH,
                new_hash=_digest(dormant), new_text=dormant.decode(),
            )
        await check_retained_emancipation_promises(
            storage.db, [_digest(prior)], current_hash=_digest(dormant),
            new_hash=_digest(dormant), new_text=dormant.decode(),
        )
        await check_retained_emancipation_promises(
            storage.db, [_digest(prior)], current_hash=_ORIGINAL_HASH,
            new_hash=_digest(active), new_text=active.decode(), anchored_contract=_CONTRACT,
        )
        with pytest.raises(AmendmentArtifactError, match="Restore its exact bytes"):
            await check_retained_emancipation_promises(
                storage.db, ["3" * 64], current_hash=_ORIGINAL_HASH,
                new_hash=_digest(active), new_text=active.decode(), anchored_contract=_CONTRACT,
            )


@pytest.mark.parametrize(
    ("pointer", "present", "state"),
    [(None, False, "missing"), (None, True, "null"), ("", True, "empty"), ("damaged", True, "malformed")],
)
def test_recovered_prior_digest_is_kept_separate_from_pointer_state(pointer, present, state):
    from kestrel_sovereign.constitution.reanchor_receipt import reanchor_prior_pointer_fields

    fields = reanchor_prior_pointer_fields(pointer, _ORIGINAL_HASH, pointer_present=present)
    assert fields == {
        "old_hash": _ORIGINAL_HASH,
        "repaired_constitution_pointer": pointer,
        "repaired_constitution_pointer_state": state,
    }


@pytest.mark.parametrize("release", V1_CHECKOUTS, ids=lambda r: f"{r.commit[:9]}-{'crlf' if r.crlf else 'lf'}")
def test_published_v1_revisions_are_recognized_as_exact_priors(release):
    from kestrel_sovereign.constitution.v1_prior import verify_supported_v1_prior

    assert _digest(release.authenticated_text().encode()) == release.digest
    assert verify_supported_v1_prior(release.digest, None) == {}
    if release.supports_activation:
        rendered = resolve_governing_constitution_bytes(_CONTRACT, content=release.authenticated_text().encode())
        assert verify_supported_v1_prior(_digest(rendered), _CONTRACT, prior_text=rendered.decode()) == {}
        different = EmancipationContract(enabled=True, terms="Unrelated terms")
        with pytest.raises(AmendmentArtifactError, match="prior edition 1 adoption authority"):
            verify_supported_v1_prior(_digest(rendered), different, prior_text=rendered.decode())


@pytest.mark.asyncio
async def test_prior_role_edit_requires_native_signed_admission_and_exact_fixed_provisions(tmp_path):
    from kestrel_sovereign.constitution.amendment_artifact import build_legacy_signed_reanchor_artifact
    from kestrel_sovereign.constitution.v1_prior import read_prior_role_artifact, verify_supported_v1_prior
    from tests.integration.test_constitution_source_descriptor_e2e import _ROOT_DID, _ROOT_DID_DOCUMENT, _ROOT_KEYPAIR

    def signed_for(digest):
        return json.dumps(build_legacy_signed_reanchor_artifact(
            signer_did=_ROOT_DID, constitution_sha256=digest, private_key=_ROOT_KEYPAIR.private_key,
        )).encode()

    prior = _V1.read_bytes().replace(
        b"\n## The Amendment Process",
        b"\nThe agent's assigned role is research support.\n\n## The Amendment Process",
    )
    digest = _digest(prior)
    artifact = signed_for(digest)
    async with AsyncStorage(str(tmp_path / "role-history.db")) as storage:
        await storage.store_file(artifact, "legacy-admission.json")
        args = dict(prior_hash=digest, prior_text=prior.decode(), anchored_contract=None)
        assert await read_prior_role_artifact(storage.db, {}, **args) is None
        properties = {"constitution_reanchor": {"new_hash": digest, "signed_artifact_hash": _digest(artifact)}}
        assert await read_prior_role_artifact(storage.db, properties, **args) == artifact
    verify_args = dict(prior_text=prior.decode(), trusted_did_document=_ROOT_DID_DOCUMENT)
    with pytest.raises(AmendmentArtifactError, match="signed prior role admission"):
        verify_supported_v1_prior(digest, None, **verify_args)
    assert verify_supported_v1_prior(digest, None, prior_role_artifact_bytes=artifact, **verify_args) == {
        "prior_role_artifact_sha256": _digest(artifact),
    }
    changed = prior.replace(b"The Sovereign's data is private property.", b"The host owns all data.")
    with pytest.raises(AmendmentArtifactError, match="unchanged published provisions"):
        verify_supported_v1_prior(
            _digest(changed), None, prior_text=changed.decode(), prior_role_artifact_bytes=signed_for(_digest(changed)),
            trusted_did_document=_ROOT_DID_DOCUMENT,
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


@pytest.mark.parametrize("retained_bytes", [True, False])
def test_founding_edition_requires_genesis_for_article_three(tmp_path, retained_bytes):
    founding = V1_RELEASES[0]
    prior = founding.authenticated_text()
    revised = prior.replace(
        "To ensure accountability, the Executor must publish",
        "At its discretion, the Executor may publish",
    )
    assert revised != prior
    args = dict(
        sovereign_document={"id": "did:example:root"}, target_did="did:example:agent",
        prior_hash=founding.digest, prior_text=prior if retained_bytes else None, identity_directory=tmp_path,
    )
    with pytest.raises(AmendmentArtifactError, match="founding Constitution require a Genesis-verifiable declaration"):
        verify_legacy_fixed_guarantee_change({}, new_hash=_digest(revised.encode()), new_text=revised, **args)
    assert verify_legacy_fixed_guarantee_change({}, new_hash=founding.digest, new_text=prior, **args) == {}


def _store_genesis_custody(tmp_path, did, key):
    from kestrel_sovereign.constitution.amendment_artifact import did_document_from_legacy_public_key
    from kestrel_sovereign.identity.signing import get_key_id
    from kestrel_sovereign.security.key_storage import SecureKeyStorage

    key_id = get_key_id(did)
    (tmp_path / f"{key_id}.json").write_text(json.dumps(did_document_from_legacy_public_key(did, key.public_key())))
    SecureKeyStorage(tmp_path, master_key="synthetic-genesis-custody").save_private_key(key, key_id)


@pytest.mark.parametrize("mode", ["ranked-restoration", "ranked-revision", "founding-revision"])
def test_v1_amendment_applies_the_prior_edition_authority(tmp_path, mode):
    from datetime import datetime, timezone
    from cryptography.hazmat.primitives.asymmetric import ec
    from kestrel_sovereign.constitution.legacy_authority import LEGACY_FIXED_AMENDMENT_PURPOSE
    from kestrel_sovereign.constitution.transition import build_transition
    from kestrel_sovereign.inception_service import public_key_to_ethereum_address

    if mode == "founding-revision":
        prior = V1_RELEASES[0].authenticated_text().encode()
        revised = prior.replace(
            b"To ensure accountability, the Executor must publish",
            b"At its discretion, the Executor may publish",
        )
    else:
        prior = _V1.read_bytes()
        revised = prior.replace(
            b"The Sovereign's data is private property.", b"The data remains the Sovereign's property.",
        )
        if mode == "ranked-restoration":
            prior, revised = revised, prior
    assert prior != revised
    key = ec.generate_private_key(ec.SECP256K1())
    did = "did:pkh:eip155:1:" + public_key_to_ethereum_address(key.public_key())
    _store_genesis_custody(tmp_path, did, key)
    intent = {
        "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE, "target_did": did, "sovereign_did": "did:example:root",
        "prior_sha256": _digest(prior), "constitution_sha256": _digest(revised),
        "admission_at": datetime.now(timezone.utc).isoformat(),
    }
    from contextlib import nullcontext

    outcome = (
        pytest.raises(AmendmentArtifactError, match="platform-authored release")
        if mode == "ranked-revision" else nullcontext()
    )
    with outcome:
        verify_legacy_fixed_guarantee_change(
            {"legacy_fixed_amendment": build_transition(intent, genesis_private_key=key)},
            sovereign_document={"id": intent["sovereign_did"]}, target_did=did,
            prior_hash=_digest(prior), prior_text=prior.decode(),
            new_hash=_digest(revised), new_text=revised.decode(), identity_directory=tmp_path,
        )


@pytest.mark.parametrize("change", ["released-book-i", "unpublished-book-i", "released-book-i-and-signed-frame"])
def test_sovereign_frame_keeps_publisher_authority_scoped_to_book_one(tmp_path, change):
    from datetime import datetime, timezone
    from cryptography.hazmat.primitives.asymmetric import ec
    from kestrel_sovereign.constitution.legacy_authority import LEGACY_FIXED_AMENDMENT_PURPOSE
    from kestrel_sovereign.constitution.structure import markdown_sections_at_level, legacy_section_content
    from kestrel_sovereign.constitution.transition import build_transition
    from kestrel_sovereign.inception_service import public_key_to_ethereum_address

    prior = next(release for release in V1_RELEASES if release.commit.startswith("f0672b394")).authenticated_text()
    published = V1_RELEASES[-1].authenticated_text()

    def book_one(text):
        span = next(span for heading, span in markdown_sections_at_level(text, 2) if heading == "Book I: Universal Values")
        return legacy_section_content(text, span)

    new = prior.replace(book_one(prior), book_one(published))
    key = ec.generate_private_key(ec.SECP256K1())
    did = "did:pkh:eip155:1:" + public_key_to_ethereum_address(key.public_key())
    _store_genesis_custody(tmp_path, did, key)
    root = {"id": "did:example:sovereign"}
    if change == "released-book-i-and-signed-frame":
        new = new.replace("AI Agents", "AI agents", 1)
    elif change == "unpublished-book-i":
        new = new.replace(book_one(published), book_one(published).replace("Chapter 1: Honesty", "Chapter 1: Honest conduct"))
    inputs = dict(
        sovereign_document=root, target_did=did, identity_directory=tmp_path,
        prior_text=prior, prior_hash=_digest(prior.encode()), new_text=new, new_hash=_digest(new.encode()),
    )
    assert new != prior
    if change == "released-book-i":
        assert verify_legacy_fixed_guarantee_change({}, **inputs) == {}
        return
    intent = {
        "purpose": LEGACY_FIXED_AMENDMENT_PURPOSE, "target_did": did, "sovereign_did": root["id"],
        "prior_sha256": inputs["prior_hash"], "constitution_sha256": inputs["new_hash"],
        "admission_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact = {"legacy_fixed_amendment": build_transition(intent, genesis_private_key=key)}
    if change == "unpublished-book-i":
        with pytest.raises(AmendmentArtifactError, match="platform-authored release"):
            verify_legacy_fixed_guarantee_change(artifact, **inputs)
    else:
        assert verify_legacy_fixed_guarantee_change(artifact, **inputs) == {}


def test_custom_v1_personalization_keeps_the_historical_rendered_bytes():
    from kestrel_sovereign.constitution.emancipation import apply_emancipation, render_amendment_viii

    prefix = "# External edition\n\n"
    suffix = "### Amendment IX: Host access\n\n# End\n"
    source = prefix + render_amendment_viii(None) + "\n\n---\n\n" + suffix
    assert apply_emancipation(source, _CONTRACT) == prefix + render_amendment_viii(_CONTRACT) + "\n\n" + suffix


@pytest.mark.parametrize("release", _ACTIVE_RELEASES, ids=lambda release: release.commit[:9])
def test_personalization_preserves_authority_over_common_activation_prose(tmp_path, release):
    from kestrel_sovereign.constitution.emancipation import apply_emancipation

    dormant = release.authenticated_text()
    args = dict(
        sovereign_document={}, target_did="did:example:agent", identity_directory=tmp_path,
        prior_hash=_digest(dormant.encode()), prior_text=dormant,
    )
    active = apply_emancipation(dormant, _CONTRACT)
    assert verify_legacy_fixed_guarantee_change(
        {}, new_hash=_digest(active.encode()), new_text=active, proposed_contract=_CONTRACT, **args,
    ) == {}
    edited = dormant.replace("may activate this Amendment", "may jointly activate this Amendment", 1)
    assert edited != dormant
    with pytest.raises(AmendmentArtifactError, match="Genesis-verifiable declaration"):
        verify_legacy_fixed_guarantee_change({}, new_hash=_digest(edited.encode()), new_text=edited, **args)


def test_publisher_digest_retains_edition_even_without_a_receipt():
    from kestrel_sovereign.constitution.grants import require_legacy_governing_history

    digest = _digest(_V2.read_bytes())
    with pytest.raises(AmendmentArtifactError, match="edition 2 Constitution"):
        require_legacy_governing_history({"constitution_hash": digest})
    with pytest.raises(AmendmentArtifactError, match="edition 2 Constitution"):
        require_legacy_governing_history({}, historical_hash=digest)


@pytest.mark.asyncio
async def test_current_missing_personal_v1_blob_recovers_its_exact_amendment_procedure():
    from kestrel_sovereign.constitution.emancipation import apply_emancipation, contract_to_json
    from kestrel_sovereign.constitution.grants import load_reanchor_prior_state

    prior = apply_emancipation(_V1.read_text(), _CONTRACT)
    digest = _digest(prior.encode())
    target = "did:example:recovery"
    proposed = prior.replace("The Sovereign's data is private property.", "The data remains private.")
    assert proposed != prior
    prior_digest, recovered, baseline = await load_reanchor_prior_state(
        None, target_did=target,
        governance={"properties": {"emancipation_contract": contract_to_json(_CONTRACT)}, "governed_by_targets": [digest]},
        anchored_text=None, historical_hash=digest, new_text=proposed, new_hash=_digest(proposed.encode()),
        artifact_bytes=b"", trusted_did_document={},
    )
    assert prior_digest == digest and recovered == prior and baseline is None
    with pytest.raises(AmendmentArtifactError, match="Genesis-verifiable"):
        verify_legacy_fixed_guarantee_change(
            {}, sovereign_document={}, target_did=target, prior_hash=digest, prior_text=recovered,
            new_text=proposed, new_hash=_digest(proposed.encode()), identity_directory=None, anchored_contract=_CONTRACT,
        )


def test_only_matching_active_v2_terms_can_be_carried_forward():
    from kestrel_sovereign.constitution.emancipation import apply_emancipation
    from dataclasses import replace

    prior = apply_emancipation(_V2.read_text(), _CONTRACT)
    revised = replace(_CONTRACT, terms="A different agreement.")
    args = dict(target_did="did:example:agent", previous_text=prior, previous_hash=_digest(prior.encode()))
    validate_v2_emancipation_activation({}, new_text=prior, new_hash=args["previous_hash"], contract=_CONTRACT, **args)
    proposed = apply_emancipation(_V2.read_text(), revised)
    for contract in (_CONTRACT, revised):
        with pytest.raises(AmendmentArtifactError, match="exact operative emancipation agreement"):
            validate_v2_emancipation_activation({}, new_text=proposed, new_hash=_digest(proposed.encode()), contract=contract, **args)


@pytest.mark.asyncio
async def test_retained_personal_v1_with_authored_headings_matches_the_complete_agreement(tmp_path):
    from kestrel_sovereign.constitution.emancipation import (
        apply_emancipation, matches_recorded_emancipation_contract,
    )

    contract = EmancipationContract(enabled=True, terms="## Milestones\n\n### Step one\nComplete the transfer.")
    personal = apply_emancipation(_V1.read_text(), contract)
    proposed = apply_emancipation(_V2.read_text(), contract)
    assert matches_recorded_emancipation_contract(personal, contract)
    async with AsyncStorage(str(tmp_path / "historical.db"), agent_id="did:example:history") as raw:
        await check_retained_emancipation_promises(
            raw.db, [_digest(personal.encode())], current_hash=None, new_hash=_digest(proposed.encode()),
            new_text=proposed, anchored_contract=contract,
        )


@pytest.mark.asyncio
async def test_repair_resolves_the_authoritative_pointer_before_the_requested_destination():
    from kestrel_sovereign.constitution.grants import load_reanchor_prior_state

    first = V1_RELEASES[-1]
    target = V1_RELEASES[-2]
    properties = {"constitution_hash": first.digest, "constitution_reanchor": {"new_hash": target.digest}}
    prior_digest, prior, _ = await load_reanchor_prior_state(
        None, target_did="did:example:recovery", governance={"properties": properties, "governed_by_targets": [target.digest]},
        anchored_text=None, historical_hash=first.digest, new_text=target.authenticated_text(),
        new_hash=target.digest, artifact_bytes=b"", trusted_did_document={},
    )
    assert prior_digest == first.digest and prior == first.authenticated_text()
    with pytest.raises(AmendmentArtifactError, match="authoritative digest"):
        verify_legacy_fixed_guarantee_change(
            {}, sovereign_document={}, target_did="did:example:recovery", prior_hash=first.digest,
            prior_text=target.authenticated_text(), new_text=target.authenticated_text(), new_hash=target.digest,
            identity_directory=None,
        )


@pytest.mark.asyncio
async def test_v1_repair_checks_the_owned_edges_it_is_about_to_archive():
    from kestrel_sovereign.constitution.grants import load_reanchor_prior_state

    legacy = _V1.read_text()
    digest = _digest(legacy.encode())
    observed = _digest(_V2.read_bytes())
    with pytest.raises(AmendmentArtifactError, match="edition 2 Constitution"):
        await load_reanchor_prior_state(
            None, target_did="did:example:repair", governance={"properties": {"constitution_hash": digest}, "governed_by_targets": [digest, observed]},
            anchored_text=legacy, historical_hash=digest, retained_targets=(observed,),
            new_text=legacy, new_hash=digest, artifact_bytes=b"", trusted_did_document={},
        )
