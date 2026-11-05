"""Paid provider tests in ``tests/llm`` skip unless explicitly opted in (#3941).

A provider key alone must never authorize a paid request. Each live test is
called here with every network-capable constructor it uses replaced by one
that fails, so a missing gate fails this module instead of reaching a provider.
"""

import anthropic
import openai
import pytest

import kestrel_sovereign.llm.service as llm_service_module
from tests.llm import test_streaming_tool_detection as streaming
from tests.llm import test_token_tracking as token_tracking
from tests.llm._live import LIVE_TESTS_ENV, PROVIDER_CREDENTIAL_ENV_VARS
from tests.shared.environment_guard import unset_environment

LIVE_PROVIDER_TESTS = (
    (streaming.TestOpenAIStreamingToolDetectionIntegration, "test_real_openai_text_streaming", "openai"),
    (streaming.TestOpenAIStreamingToolDetectionIntegration, "test_real_openai_tool_detection", "openai"),
    (streaming.TestAnthropicStreamingToolDetectionIntegration, "test_real_anthropic_text_streaming", "anthropic"),
    (streaming.TestAnthropicStreamingToolDetectionIntegration, "test_real_anthropic_tool_detection", "anthropic"),
    (streaming.TestLLMServiceStreamWithToolDetection, "test_service_streaming_text_only", "openai"),
    (streaming.TestLLMServiceStreamWithToolDetection, "test_service_streaming_with_tools", "openai"),
    (token_tracking.TestRealOpenAITokenTracking, "test_real_openai_returns_tokens", "openai"),
    (token_tracking.TestRealAnthropicTokenTracking, "test_real_anthropic_returns_tokens", "anthropic"),
    (
        token_tracking.TestLLMServiceTokenTracking,
        "test_service_passes_tokens_through_with_structured_output",
        "openai",
    ),
    (token_tracking.TestLLMServiceTokenTracking, "test_service_passes_tokens_through_with_tools", "openai"),
)


def _refuse_construction(*args, **kwargs):
    raise AssertionError("network-capable client constructed without crossing the live gate")


@pytest.fixture
def refuse_provider_clients(monkeypatch):
    """Make every client the live tests construct fail before it could connect."""
    monkeypatch.setattr(openai, "AsyncOpenAI", _refuse_construction)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _refuse_construction)
    monkeypatch.setattr(llm_service_module, "LLMService", _refuse_construction)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_class,method_name,provider",
    LIVE_PROVIDER_TESTS,
    ids=[method_name for _, method_name, _ in LIVE_PROVIDER_TESTS],
)
async def test_credentials_without_opt_in_skip_before_any_client(
    monkeypatch, refuse_provider_clients, test_class, method_name, provider
):
    unset_environment(monkeypatch, LIVE_TESTS_ENV)
    for variable in PROVIDER_CREDENTIAL_ENV_VARS[provider]:
        monkeypatch.setenv(variable, "test-credential-must-never-reach-network")

    with pytest.raises(pytest.skip.Exception, match=f"set {LIVE_TESTS_ENV}=1"):
        await getattr(test_class(), method_name)()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_class,method_name,provider",
    LIVE_PROVIDER_TESTS,
    ids=[method_name for _, method_name, _ in LIVE_PROVIDER_TESTS],
)
async def test_opt_in_without_credentials_skips_for_missing_credentials(
    monkeypatch, refuse_provider_clients, test_class, method_name, provider
):
    monkeypatch.setenv(LIVE_TESTS_ENV, "1")
    unset_environment(monkeypatch, *PROVIDER_CREDENTIAL_ENV_VARS[provider])

    with pytest.raises(pytest.skip.Exception, match=f"{provider} requires"):
        await getattr(test_class(), method_name)()
