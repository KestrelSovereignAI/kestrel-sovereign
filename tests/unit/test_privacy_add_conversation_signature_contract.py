"""Contract test: PrivacyAgent.add_conversation must accept every keyword that
the storage-layer add_conversation accepts.

Regression guard for #1920: #1906 added ``model``/``provider`` to the storage
``add_conversation`` and threaded them through callers as ``**kwargs``, but the
agent-level ``PrivacyAgent.add_conversation`` wrapper kept the old signature, so
every assistant turn after a model identity was resolved raised
``TypeError: ... unexpected keyword argument 'model'`` and 500'd the chat-turn
persistence path. This asserts the wrapper signature stays a superset of the
storage signature so that class of drift fails fast in CI, not in production.
"""

import inspect

from kestrel_sovereign.features.privacy.feature import PrivacyAgent
from kestrel_sovereign.storage.async_storage import AsyncStorage
from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage


def _kwargs(func) -> set[str]:
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name != "self"
    }


def test_privacy_agent_add_conversation_is_superset_of_storage():
    agent_kwargs = _kwargs(PrivacyAgent.add_conversation)
    for storage_cls in (AsyncStorage, PrivacyEnforcingStorage):
        storage_kwargs = _kwargs(storage_cls.add_conversation)
        missing = storage_kwargs - agent_kwargs
        assert not missing, (
            "PrivacyAgent.add_conversation is missing kwargs accepted by "
            f"{storage_cls.__name__}.add_conversation: {sorted(missing)}. "
            "Callers forward these as **kwargs; drift here 500s assistant turns "
            "(#1920)."
        )


def test_model_and_provider_are_accepted():
    # The specific #1920 regression: these must be present.
    agent_kwargs = _kwargs(PrivacyAgent.add_conversation)
    assert {"model", "provider"} <= agent_kwargs
