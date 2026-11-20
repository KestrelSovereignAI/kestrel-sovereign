"""What a provider's own response says served a request (#4198).

Kestrel records the route and model it *asked for*. That is not evidence of
what answered: a provider can serve a dated snapshot of a requested alias, and
a turn that asked for one model may have been answered by another. A reviewer
checking that an exact one-turn override reached the model it named needs the
provider's own word, per call.

An adapter copies that word off the provider's response and attaches it to the
``LLMResponse`` it returns, or to the one that ends its stream: the provider
API it called, the model the response names, and the provider's response
id. The service records it on the call's durable ``llm_calls`` row and on its
``llm.usage`` line, under :data:`PROVIDER_SERVED_KEY`. Nothing reads it to
route; it is evidence only. A failed attempt keeps it too: an adapter puts it
on the usage evidence its error carries
(:func:`~kestrel_sovereign.llm.output_ceiling.reported_usage`), and the
usage-only copy a failed attempt is recorded from keeps it
(:func:`carry_provider_served`).

Only short identifiers are kept. A value that is not a non-empty string of at
most :data:`MAX_IDENTIFIER_CHARS` characters is not recorded, so a provider
that returns something else in these fields cannot write content into a log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

#: The key the evidence travels under, on the ``llm.usage`` line and in the
#: durable ``llm_calls`` row's metadata. Absent when the adapter attached none.
PROVIDER_SERVED_KEY = "provider_served"

#: The attribute a response carries its evidence on, from adapter to service.
_RESPONSE_ATTR = "_kestrel_provider_served"

#: A model name or response id longer than this is not recorded.
MAX_IDENTIFIER_CHARS = 256


def _identifier(value: Any) -> Optional[str]:
    if isinstance(value, str) and value and len(value) <= MAX_IDENTIFIER_CHARS:
        return value
    return None


@dataclass(frozen=True)
class ProviderServed:
    """The provider's own account of one response.

    ``api`` names the provider API the adapter called, as the provider's SDK
    names the operation (for example ``responses`` or ``chat.completions``),
    which holds whatever base URL a route uses. ``model`` is the model the
    response names, and ``response_id`` the provider's id for it; either is
    ``None`` when the provider did not report one Kestrel keeps.
    """

    api: str
    model: Optional[str] = None
    response_id: Optional[str] = None

    def record(self) -> Dict[str, Optional[str]]:
        """The JSON shape recorded."""
        return {
            "api": self.api,
            "model": self.model,
            "response_id": self.response_id,
        }


def provider_served(
    api: str, *, model: Any = None, response_id: Any = None,
) -> Optional[ProviderServed]:
    """The evidence for one response, or ``None`` when the provider reported
    neither a model nor a response id Kestrel keeps."""
    kept_model = _identifier(model)
    kept_id = _identifier(response_id)
    if kept_model is None and kept_id is None:
        return None
    return ProviderServed(api=api, model=kept_model, response_id=kept_id)


def attach_provider_served(response: Any, served: Optional[ProviderServed]) -> Any:
    """Carry ``served`` on ``response`` to the service, and return it."""
    if served is not None:
        setattr(response, _RESPONSE_ATTR, served)
    return response


def carry_provider_served(source: Any, target: Any) -> Any:
    """Carry the evidence ``source`` holds on ``target`` too, and return
    ``target``: a usage-only copy of a response keeps what served it."""
    served = getattr(source, _RESPONSE_ATTR, None)
    return attach_provider_served(
        target, served if isinstance(served, ProviderServed) else None,
    )


def response_provider_served(response: Any) -> Optional[Dict[str, Optional[str]]]:
    """The record of the evidence an adapter attached to ``response``, if any."""
    served = getattr(response, _RESPONSE_ATTR, None)
    return served.record() if isinstance(served, ProviderServed) else None


__all__ = [
    "MAX_IDENTIFIER_CHARS",
    "PROVIDER_SERVED_KEY",
    "ProviderServed",
    "attach_provider_served",
    "carry_provider_served",
    "provider_served",
    "response_provider_served",
]
