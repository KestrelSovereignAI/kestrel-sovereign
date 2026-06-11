from types import SimpleNamespace
from unittest.mock import MagicMock

from kestrel_sovereign.features.storage_access import (
    hides_persisted_user_content,
    resolve_feature_conversation_store,
    resolve_feature_database,
)
from kestrel_sovereign.privacy import PrivacyConfig


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
