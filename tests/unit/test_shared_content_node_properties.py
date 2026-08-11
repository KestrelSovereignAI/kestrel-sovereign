"""The key-set guard on properties a PostgreSQL fleet co-owns (#2893).

A content-addressed node — the governing constitution document, or a
Sovereign-signed reanchor artifact — can carry more than one owner, because
every tenant that possesses the bytes computes the same node id and the same
properties. That is only true while the properties stay *content-derived*, so
``add_node`` admits co-ownership only when both the stored row and the incoming
node pass a shareability predicate.

These tests pin the predicates' key-set guard specifically. It was originally
written as ``set(properties) > ALLOWED`` — a *proper superset* test, which asks
a weaker question than "does this carry an unlisted key". Because ``created_at``
is optional, swapping it for an unlisted key produces a key set that is not a
superset at all, so the guard never fired and an operator filesystem path could
ride onto a row another tenant co-owns.
"""
import pytest

from kestrel_sovereign.constitution.amendment_artifact import ARTIFACT_TYPE
from kestrel_sovereign.storage.async_graph_store import (
    _is_shareable_amendment_artifact_properties,
    _is_shareable_constitution_properties,
)

NODE_ID = "a" * 64
CONSTITUTION_HASH = "b" * 64
SIGNER = "did:web:example.com:kestrel-sovereign"


def _artifact_properties(**overrides):
    properties = {
        "hash": NODE_ID,
        "type": "SignedConstitutionAmendment",
        "artifact_type": ARTIFACT_TYPE,
        "constitution_hash": CONSTITUTION_HASH,
        "signer": SIGNER,
        "created_at": "2026-08-11T00:00:00+00:00",
    }
    properties.update(overrides)
    return properties


class TestConstitutionAnchor:
    def test_the_canonical_shape_is_shareable(self):
        assert _is_shareable_constitution_properties(
            {"hash": NODE_ID, "type": "Constitution"}, NODE_ID
        )
        assert _is_shareable_constitution_properties(
            {
                "hash": NODE_ID,
                "type": "Constitution",
                "created_at": "2026-08-11T00:00:00+00:00",
            },
            NODE_ID,
        )

    def test_an_extra_key_is_refused(self):
        assert not _is_shareable_constitution_properties(
            {
                "hash": NODE_ID,
                "type": "Constitution",
                "created_at": "2026-08-11T00:00:00+00:00",
                "source_path": "/home/operator/secret/KESTREL_CONSTITUTION.md",
            },
            NODE_ID,
        )

    def test_an_unlisted_key_swapped_for_an_optional_one_is_refused(self):
        """The case a proper-superset guard misses.

        ``{hash, type, source_path}`` is not a superset of
        ``{hash, type, created_at}`` — it is missing ``created_at`` — so
        ``set(properties) > ALLOWED`` is ``False`` and the guard falls through.
        Every remaining check then passes, and the operator's path is admitted
        onto a shared row.
        """
        assert not _is_shareable_constitution_properties(
            {
                "hash": NODE_ID,
                "type": "Constitution",
                "source_path": "/home/operator/secret/KESTREL_CONSTITUTION.md",
            },
            NODE_ID,
        )


class TestAmendmentArtifact:
    def test_the_canonical_shape_is_shareable(self):
        assert _is_shareable_amendment_artifact_properties(
            _artifact_properties(), NODE_ID
        )

    def test_a_null_created_at_is_still_shareable(self):
        """The writer passes ``amendment_artifact.get("created_at")`` straight
        through, so an artifact without one stores an explicit ``None``. The
        key is present; the value is absent."""
        assert _is_shareable_amendment_artifact_properties(
            _artifact_properties(created_at=None), NODE_ID
        )

    def test_an_extra_key_is_refused(self):
        assert not _is_shareable_amendment_artifact_properties(
            _artifact_properties(source_path="/home/operator/artifact.json"),
            NODE_ID,
        )

    def test_an_unlisted_key_swapped_for_an_optional_one_is_refused(self):
        """The proper-superset hole, on the artifact predicate."""
        properties = _artifact_properties()
        del properties["created_at"]
        properties["source_path"] = "/home/operator/artifact.json"
        assert not _is_shareable_amendment_artifact_properties(
            properties, NODE_ID
        )

    @pytest.mark.parametrize(
        "key, value",
        [
            ("source_path", "/home/operator/artifact.json"),
            ("anchored_at", "2026-08-11T00:00:00+00:00"),
            ("verification", "signature verified against did:web:…"),
        ],
    )
    def test_a_per_agent_field_is_refused(self, key, value):
        """Parametrized rather than looped, because each is a distinct
        disclosure — a filesystem path, this agent's anchoring time, and the
        result of verifying against *this* agent's trust root — and a loop
        stops reporting at the first one that regresses.

        Each replaces the optional ``created_at``, which is the shape that
        slips past a proper-superset guard.
        """
        properties = _artifact_properties()
        del properties["created_at"]
        properties[key] = value
        assert not _is_shareable_amendment_artifact_properties(
            properties, NODE_ID
        ), f"{key} was admitted onto a fleet-shared row"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"hash": "c" * 64},
            {"type": "Constitution"},
            {"artifact_type": "constitution_reanchor"},
            {"constitution_hash": "not-a-digest"},
            {"constitution_hash": "B" * 64},
            {"signer": "https://example.com/key"},
            {"signer": "did:" + "x" * 300},
            {"created_at": "2026-08-11T00:00:00"},
        ],
    )
    def test_a_field_outside_its_declared_shape_is_refused(self, overrides):
        assert not _is_shareable_amendment_artifact_properties(
            _artifact_properties(**overrides), NODE_ID
        )
