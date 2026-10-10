"""Pure admission preserves malformed evidence before any in-place change."""

from copy import deepcopy

import pytest

from kestrel_sovereign.constitution.reanchor_receipt import supersede_constitution_reanchor


@pytest.mark.parametrize("old_hash", [None, "none", "b" * 64])
def test_legacy_and_first_anchor_evidence_is_preserved_without_inventing_signature(old_hash):
    prior = {"old_hash": old_hash, "new_hash": "a" * 64, "legacy_prose": "retained verbatim"}
    properties = {"constitution_reanchor": prior}
    new = {"old_hash": "a" * 64, "new_hash": "c" * 64, "signed_artifact_hash": "d" * 64}
    supersede_constitution_reanchor(properties, receipt=new, provenance="native-test")
    assert properties["constitution_reanchor_history"][0]["receipt"] == prior
    assert "signed_artifact_hash" not in properties["constitution_reanchor_history"][0]["receipt"]


@pytest.mark.parametrize("properties", [
    {"constitution_reanchor": None},
    {"constitution_reanchor": {"new_hash": "bad"}},
    {"constitution_reanchor_history": None},
    {"constitution_reanchor_history": [{}]},
    {"constitution_reanchor_history": [{"receipt": {"new_hash": "a" * 64}, "superseded_by_constitution_hash": "bad"}]},
])
def test_reanchor_receipt_admission_never_mutates_unreadable_history(properties):
    before = deepcopy(properties)
    with pytest.raises(ValueError, match="existing evidence is preserved"):
        supersede_constitution_reanchor(properties, receipt={"new_hash": "a" * 64}, provenance="native-test")
    assert properties == before


def test_new_receipt_validation_precedes_history_publication():
    properties = {"constitution_reanchor": {"new_hash": "a" * 64}}
    before = deepcopy(properties)
    with pytest.raises(ValueError, match="new_hash"):
        supersede_constitution_reanchor(properties, receipt={"new_hash": None}, provenance="native-test")
    assert properties == before
