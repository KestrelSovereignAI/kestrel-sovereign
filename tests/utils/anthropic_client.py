"""Stand-ins for the two Anthropic SDK client surfaces a non-streaming
``AnthropicAdapter`` request touches (#3300).

* ``client.models.retrieve`` — the adapter asks the Models API for a model's
  output ceiling (its ``max_tokens``) the first time it sends a request for
  that model without a caller-chosen ``max_tokens``. The record returned here
  is the SDK's own ``anthropic.types.ModelInfo``, so the adapter reads the
  real field names.
* ``client.messages.stream(...)`` + ``get_final_message()`` — how
  ``get_response`` now carries a complete response, because the SDK refuses
  a non-streaming ``messages.create`` at a full-ceiling ``max_tokens``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from anthropic.types import ModelInfo as AnthropicModelInfo

#: The output ceiling the fake Models API reports unless a test says otherwise.
DEFAULT_TEST_OUTPUT_CEILING = 128_000


def anthropic_model_record(model_id: str, *, max_tokens: Any) -> AnthropicModelInfo:
    """A Models API record for ``model_id`` reporting ``max_tokens``."""
    return AnthropicModelInfo(
        id=model_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        display_name=model_id,
        max_input_tokens=1_000_000,
        max_tokens=max_tokens,
        type="model",
    )


def install_models_api(
    client: Any, *, max_tokens: Any = DEFAULT_TEST_OUTPUT_CEILING,
) -> AsyncMock:
    """Give ``client`` a ``models.retrieve`` reporting ``max_tokens``."""
    client.models.retrieve = AsyncMock(
        side_effect=lambda model_id, **_: anthropic_model_record(
            model_id, max_tokens=max_tokens,
        )
    )
    return client.models.retrieve


def models_api(*, max_tokens: Any = DEFAULT_TEST_OUTPUT_CEILING) -> SimpleNamespace:
    """A stand-alone ``client.models`` for hand-built (``SimpleNamespace``)
    clients."""
    namespace = SimpleNamespace()
    install_models_api(SimpleNamespace(models=namespace), max_tokens=max_tokens)
    return namespace


class FinalMessageStream:
    """What ``client.messages.stream(...)`` returns, reduced to the part
    ``get_response`` uses: an async context whose ``get_final_message()``
    is the complete response."""

    def __init__(self, message: Any) -> None:
        self._message = message

    async def __aenter__(self) -> "FinalMessageStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def get_final_message(self) -> Any:
        return self._message


def install_final_message(client: Any, message: Any) -> MagicMock:
    """Make every ``client.messages.stream(...)`` resolve to ``message``."""
    client.messages.stream = MagicMock(
        side_effect=lambda **_: FinalMessageStream(message)
    )
    return client.messages.stream


def anthropic_client(message: Any, **models_api: Any) -> MagicMock:
    """A fake Anthropic client whose non-streaming responses are ``message``."""
    client = MagicMock()
    install_models_api(client, **models_api)
    install_final_message(client, message)
    return client


#: The 400 Anthropic returns when a model refuses a forced tool choice
#: (verbatim from claude-opus-5-5 / claude-sonnet-5-5, #3492).
FORCED_TOOL_CHOICE_REFUSAL = (
    'tool_choice: type "tool" and "any" are not supported for this model.'
)


def anthropic_bad_request(message: str) -> Any:
    """The SDK's own ``BadRequestError`` for a 400 ``invalid_request_error``."""
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return anthropic.BadRequestError(
        f"Error code: 400 - {body}",
        response=httpx.Response(400, request=request, json=body),
        body=body,
    )


class ForcedToolRefusingMessages:
    """``client.messages`` for a model that, like claude-opus-5-5, refuses a
    forced ``tool_choice`` and answers native structured output.

    A request carrying ``tool_choice`` type "tool" or "any" raises the 400 the
    live API returns. A request with ``output_config.format`` is answered with
    a thinking block followed by ``payload`` as JSON text, the shape the live
    model returns with adaptive thinking on. Every request is recorded.
    """

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def stream(self, **params: Any) -> FinalMessageStream:
        import json

        self.requests.append(params)
        choice = params.get("tool_choice") or {}
        if choice.get("type") in ("tool", "any"):
            raise anthropic_bad_request(FORCED_TOOL_CHOICE_REFUSAL)
        fmt = (params.get("output_config") or {}).get("format") or {}
        if fmt.get("type") != "json_schema":
            raise AssertionError("expected a native structured-output request")
        return FinalMessageStream(SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=json.dumps(self.payload)),
            ],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=5, output_tokens=10),
        ))


def forced_tool_refusing_client(payload: dict) -> SimpleNamespace:
    """A fake client behaving like claude-opus-5-5 for structured output."""
    return SimpleNamespace(
        messages=ForcedToolRefusingMessages(payload), models=models_api(),
    )
