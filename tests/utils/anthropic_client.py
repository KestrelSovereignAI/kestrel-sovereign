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
