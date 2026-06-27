"""v5 capability negotiation (#1983).

Covers the registration-time capability validator
(:func:`_warn_unimplemented_capabilities`) and the typed streaming routing
gates that replaced the dead ``provider_name in [...]`` literals. The gates must
stay False for any adapter that hasn't opted in via ``contract_features()`` so
no in-tree adapter changes behavior.
"""
import logging

from kestrel_sdk.llm import ProviderCapabilities, ProviderInfo, StructuredOutputMode

from kestrel_sovereign.llm import streaming as streaming_mod
from kestrel_sovereign.llm.adapter import LLMAdapter
from kestrel_sovereign.llm.provider_registry import _warn_unimplemented_capabilities

_REGISTRY_LOGGER = "kestrel_sovereign.llm.provider_registry"


class _MinimalAdapter(LLMAdapter):
    """Adapter implementing only the abstract surface (inherits v5 defaults)."""

    async def get_response(self, client, model, messages, **kwargs):  # pragma: no cover
        return None


def _info(adapter: LLMAdapter, caps: ProviderCapabilities) -> ProviderInfo:
    return ProviderInfo(
        name="x:api",
        vendor="x",
        route="api",
        client=object(),
        adapter=adapter,
        model="m",
        capabilities=caps,
    )


# --------------------------------------------------------------------------- #
# Registration-time validator
# --------------------------------------------------------------------------- #


def test_validator_warns_when_flag_on_but_method_unimplemented(caplog):
    info = _info(_MinimalAdapter(), ProviderCapabilities(supports_batch=True))
    with caplog.at_level(logging.WARNING, logger=_REGISTRY_LOGGER):
        _warn_unimplemented_capabilities(info)
    messages = [r.getMessage() for r in caplog.records]
    assert any("supports_batch" in m and "x:api" in m for m in messages)


def test_validator_silent_when_no_flags_advertised(caplog):
    info = _info(_MinimalAdapter(), ProviderCapabilities())
    with caplog.at_level(logging.WARNING, logger=_REGISTRY_LOGGER):
        _warn_unimplemented_capabilities(info)
    assert caplog.records == []


def test_validator_silent_when_feature_declared(caplog):
    class _DeclaresBatch(_MinimalAdapter):
        def contract_features(self):
            return frozenset({"batch"})

    info = _info(_DeclaresBatch(), ProviderCapabilities(supports_batch=True))
    with caplog.at_level(logging.WARNING, logger=_REGISTRY_LOGGER):
        _warn_unimplemented_capabilities(info)
    assert caplog.records == []


def test_validator_silent_when_method_overridden(caplog):
    class _ImplementsBatch(_MinimalAdapter):
        async def batch_submit(self, client, requests, **kwargs):  # pragma: no cover
            return None

    info = _info(_ImplementsBatch(), ProviderCapabilities(supports_batch=True))
    with caplog.at_level(logging.WARNING, logger=_REGISTRY_LOGGER):
        _warn_unimplemented_capabilities(info)
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# Typed streaming routing gates (no behavior change until opt-in)
# --------------------------------------------------------------------------- #


def test_streaming_structured_gate_default_false():
    assert streaming_mod._route_supports_streaming_structured(_MinimalAdapter()) is False


def test_streaming_structured_gate_true_on_optin():
    class _OptIn(_MinimalAdapter):
        def provider_capabilities(self):
            return ProviderCapabilities(
                supports_streaming=True,
                structured_output_mode=StructuredOutputMode.JSON_SCHEMA,
            )

        def contract_features(self):
            return frozenset({streaming_mod._FEATURE_STREAMING_STRUCTURED_OUTPUT})

    assert streaming_mod._route_supports_streaming_structured(_OptIn()) is True


def test_streaming_structured_gate_false_when_mode_not_streamable():
    class _ToolForced(_MinimalAdapter):
        def provider_capabilities(self):
            return ProviderCapabilities(
                supports_streaming=True,
                structured_output_mode=StructuredOutputMode.TOOL_FORCED,
            )

        def contract_features(self):
            return frozenset({streaming_mod._FEATURE_STREAMING_STRUCTURED_OUTPUT})

    assert streaming_mod._route_supports_streaming_structured(_ToolForced()) is False


def test_tool_stream_system_prompt_gate_default_false():
    assert (
        streaming_mod._route_wants_tool_stream_system_prompt(_MinimalAdapter()) is False
    )


def test_tool_stream_system_prompt_gate_true_on_optin():
    class _OptIn(_MinimalAdapter):
        def provider_capabilities(self):
            return ProviderCapabilities(supports_inline_system=True)

        def contract_features(self):
            return frozenset({streaming_mod._FEATURE_TOOL_STREAM_SYSTEM_PROMPT})

    assert streaming_mod._route_wants_tool_stream_system_prompt(_OptIn()) is True
