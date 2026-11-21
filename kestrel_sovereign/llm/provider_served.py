"""What a provider reported serving one call, carried to its ``llm_calls`` row.

The service records the route and model it *asked for*. The provider's reply
says what actually answered: the model it served and the id it gave the
response (#4198). An adapter that observes them attaches a
:class:`ProviderServed` to the response it returns, or to the error it raises
when the attempt fails after the provider had already identified the
response, and the service copies it into the attempt's metadata under
:data:`PROVIDER_SERVED_KEY`.

The evidence is content-free: an API name, a model id, a response id, the
continuation boundary the request crossed, and the ids of any replayed
reasoning items the provider reported removing. Never text, tool arguments
or encrypted reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

#: The ``llm_calls.metadata`` key the evidence is recorded under.
PROVIDER_SERVED_KEY = "provider_served"

_ATTR = "_kestrel_provider_served"


@dataclass(frozen=True)
class ProviderServed:
    """The provider's own account of the call it answered.

    ``continuation`` names the request's continuation boundary (see
    :mod:`kestrel_sovereign.llm.responses_continuation`). ``prefix_mismatch``
    is ``None`` when the provider reported nothing about removed input, and
    otherwise the ids of the reasoning items it removed for a changed prefix
    (empty when it removed none).
    """

    api: str
    model: Optional[str] = None
    response_id: Optional[str] = None
    continuation: Mapping[str, Any] = field(default_factory=dict)
    prefix_mismatch: Optional[Tuple[str, ...]] = None

    def with_response(
        self, *, model: Optional[str], response_id: Optional[str],
    ) -> "ProviderServed":
        """This evidence, updated with an id or model the provider just named.

        A value already observed is kept when the provider names none.
        """
        return replace(
            self,
            model=model if isinstance(model, str) and model else self.model,
            response_id=(
                response_id
                if isinstance(response_id, str) and response_id
                else self.response_id
            ),
        )

    def as_metadata(self) -> Dict[str, Any]:
        """The JSON-ready form stored on the attempt's row."""
        record: Dict[str, Any] = {"api": self.api}
        if self.model is not None:
            record["model"] = self.model
        if self.response_id is not None:
            record["response_id"] = self.response_id
        if self.continuation:
            record["continuation"] = dict(self.continuation)
        if self.prefix_mismatch is not None:
            record["prefix_mismatch"] = list(self.prefix_mismatch)
        return record


def attach_provider_served(target: Any, served: Optional[ProviderServed]) -> Any:
    """Attach ``served`` to a response or an exception; return ``target``."""
    if served is not None and target is not None:
        setattr(target, _ATTR, served)
    return target


def provider_served(source: Any) -> Optional[ProviderServed]:
    """The evidence attached to ``source`` or to an error it wraps, if any.

    An error is followed through the same explicit links
    :func:`~kestrel_sovereign.llm.output_ceiling.incomplete_generation`
    follows: ``__cause__``, ``original_error`` and ``underlying``, never the
    implicit ``__context__``.
    """
    seen: set[int] = set()
    pending = [source]
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        served = getattr(current, _ATTR, None)
        if isinstance(served, ProviderServed):
            return served
        if isinstance(current, BaseException):
            pending.extend(
                link for link in (
                    current.__cause__,
                    getattr(current, "original_error", None),
                    getattr(current, "underlying", None),
                )
                if isinstance(link, BaseException)
            )
    return None


__all__ = [
    "PROVIDER_SERVED_KEY",
    "ProviderServed",
    "attach_provider_served",
    "provider_served",
]
