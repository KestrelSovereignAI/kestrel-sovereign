from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_conversation_store,
    resolve_feature_database,
    resolve_workflow_owner_did,
)
from kestrel_sovereign.privacy import PrivacyConfig
from tests.utils.agent_identities import (
    born_hybrid_identity,
    legacy_identity,
    rotated_identity,
)

LEGACY_DID = "did:pkh:eip155:1:0x00000000000000000000000000000000000000a1"
SUCCESSOR_DID = "did:web:agents.example.test:rotated"
BORN_HYBRID_DID = "did:web:agents.example.test:born-hybrid"


class PrivacyWrappedStorage:
    def __init__(self, raw_storage):
        self._storage = raw_storage

    @property
    def db(self):
        raise AssertionError("deprecated wrapper db property was touched")

    @property
    def conversation(self):
        raise AssertionError("deprecated wrapper conversation property was touched")


class PropertyBackedPrivacyAgent:
    def __init__(self, config):
        self._privacy_config = config

    @property
    def privacy_config(self):
        return self._privacy_config


def test_resolve_feature_database_prefers_raw_storage():
    raw_db = object()
    wrapped_db = object()
    agent = SimpleNamespace(
        _raw_storage=SimpleNamespace(db=raw_db),
        storage=PrivacyWrappedStorage(SimpleNamespace(db=wrapped_db)),
    )

    assert resolve_feature_database(agent) is raw_db


def test_resolve_feature_database_unwraps_privacy_storage_without_touching_property():
    db = object()
    agent = SimpleNamespace(
        _raw_storage=None,
        storage=PrivacyWrappedStorage(SimpleNamespace(db=db)),
    )

    assert resolve_feature_database(agent) is db


def test_resolve_feature_database_supports_legacy_unwrapped_storage_names():
    db = object()
    agent = SimpleNamespace(
        _raw_storage=None,
        storage=SimpleNamespace(database=db),
    )

    assert resolve_feature_database(agent) is db


def test_resolve_feature_database_ignores_magicmock_fabricated_attributes():
    agent = MagicMock()

    assert resolve_feature_database(agent) is None


def test_resolve_feature_database_supports_explicit_magicmock_db():
    db = object()
    agent = MagicMock()
    storage = MagicMock()
    storage.db = db
    agent.storage = storage
    agent._raw_storage = None

    assert resolve_feature_database(agent) is db


def test_resolve_feature_conversation_store_unwraps_without_touching_property():
    conversation = object()
    agent = SimpleNamespace(
        _raw_storage=None,
        storage=PrivacyWrappedStorage(SimpleNamespace(conversation=conversation)),
    )

    assert resolve_feature_conversation_store(agent) is conversation


def test_hides_persisted_user_content_reads_real_privacy_property():
    agent = PropertyBackedPrivacyAgent(
        PrivacyConfig(storage="none", llm_location="local")
    )

    assert hides_persisted_user_content(agent) is True


def test_hides_persisted_user_content_ignores_fabricated_magicmock_attrs():
    assert hides_persisted_user_content(MagicMock()) is False


# -- resolve_workflow_owner_did: who owns an agent's Workflows runs (#3533) --


def test_workflow_owner_of_a_legacy_identity_is_its_did():
    assert resolve_workflow_owner_did(legacy_identity(LEGACY_DID)) == LEGACY_DID


def test_workflow_owner_of_a_rotated_identity_is_its_legacy_did():
    identity = rotated_identity(LEGACY_DID, SUCCESSOR_DID)

    assert identity.signing_did == SUCCESSOR_DID
    assert resolve_workflow_owner_did(identity) == LEGACY_DID


def test_workflow_owner_of_a_born_hybrid_identity_is_its_signing_did():
    identity = born_hybrid_identity(BORN_HYBRID_DID)

    assert identity.legacy_did is None
    assert resolve_workflow_owner_did(identity) == BORN_HYBRID_DID


_NO_OWNER = [
    pytest.param(None, id="no-identity"),
    pytest.param(MagicMock(), id="fabricated-attributes"),
    pytest.param(SimpleNamespace(legacy_did="", signing_did=""), id="empty"),
    pytest.param(SimpleNamespace(legacy_did=None, signing_did=7), id="non-string-signing"),
    # A malformed legacy DID is refused, not passed over for the signing
    # DID: the Workflows runner refuses it too, so it stamped no run under
    # the signing DID either.
    pytest.param(
        SimpleNamespace(legacy_did=b"did:pkh", signing_did=SUCCESSOR_DID),
        id="non-string-legacy",
    ),
]


@pytest.mark.parametrize("identity", _NO_OWNER)
def test_workflow_owner_refuses_an_identity_without_a_did(identity):
    assert resolve_workflow_owner_did(identity) is None


@pytest.mark.parametrize(
    "identity",
    [
        pytest.param(legacy_identity(LEGACY_DID), id="legacy"),
        pytest.param(rotated_identity(LEGACY_DID, SUCCESSOR_DID), id="rotated"),
        pytest.param(born_hybrid_identity(BORN_HYBRID_DID), id="born-hybrid"),
        *_NO_OWNER,
    ],
)
def test_workflow_owner_is_the_owner_the_workflows_runner_stamps(identity):
    runner = pytest.importorskip("kestrel_feature_workflows.runner")

    assert resolve_workflow_owner_did(identity) == runner.resolve_owner_did(identity)
