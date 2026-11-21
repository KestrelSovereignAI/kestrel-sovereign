"""What a provider reported serving one call, carried to its ``llm_calls`` row.

The service records the route and model it *asked for*. The provider's reply
says what actually answered: the model it served and the id it gave the
response (#4198). An adapter that observes them attaches a
:class:`ProviderServed` to the response it returns, or to the error it raises
when the attempt fails after the provider had already identified the
response, and the service copies it into the attempt's metadata under
:data:`PROVIDER_SERVED_KEY`. While an attempt runs, what its provider has
reported so far is kept on one :class:`ProviderAttemptReceipt` (#4207).

The evidence is content-free: an API name, a model id, a response id, how the
request continued its chain or which boundary it crossed, whether the reply's
continuation snapshot was kept, and the ids of any replayed items the
provider reported removing. Never text, tool arguments or encrypted
reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from kestrel_sdk.llm import LLMResponse

from .output_ceiling import attach_reported_usage

#: The ``llm_calls.metadata`` key the evidence is recorded under.
PROVIDER_SERVED_KEY = "provider_served"

_ATTR = "_kestrel_provider_served"


@dataclass(frozen=True)
class ProviderServed:
    """The provider's own account of the call it answered.

    ``continuation`` describes the request as it was sent: a fresh chain, a
    continuation, or the boundary it crossed (see
    :mod:`kestrel_sovereign.llm.responses_continuation`). ``snapshot`` says
    what was stored for the reply: ``{"stored": True}``, or ``stored`` false
    with the reason it was not kept; empty when the reply was never settled
    (a failed attempt). ``prefix_mismatch`` is ``None`` when the provider
    reported no prefix mismatch, and otherwise the ids of the replayed items
    it removed for one (empty when it named none).
    """

    api: str
    model: Optional[str] = None
    response_id: Optional[str] = None
    continuation: Mapping[str, Any] = field(default_factory=dict)
    prefix_mismatch: Optional[Tuple[str, ...]] = None
    snapshot: Mapping[str, Any] = field(default_factory=dict)

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
        if self.snapshot:
            record["snapshot"] = dict(self.snapshot)
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


class ProviderAttemptReceipt:
    """What one provider attempt has reported, recorded as it arrives (#4207).

    An adapter opens one receipt for each provider attempt and keeps it for
    the attempt's whole lifetime: provider execution, stream iteration and
    the physical close of its stream or response. The receipt records the
    provider's identification of its reply (:attr:`served`) the moment the
    provider gives it, and the usage the provider reported (:attr:`usage`)
    the moment that arrives, never only once the attempt ends. An attempt
    that then fails, while closing its stream included, has lost nothing the
    provider had reported.

    :attr:`served` is ``None`` while the provider has identified nothing,
    and :attr:`usage` while it has reported no usage: unknown, never zero.
    A receipt belongs to the one attempt that opened it, so concurrent or
    successive attempts never share one.

    :meth:`retain` records the receipt on the error that ends the attempt,
    whatever it is: an ordinary failure, a cancellation, a closed stream, an
    execution-control error or exception group. It sets attributes on that
    same object and returns it; it never wraps, replaces or swallows the
    error, and leaves a group's members, its cause and its native custody as
    they are. A caller that catches the error reads the receipt from it with
    :func:`provider_served` and
    :func:`~kestrel_sovereign.llm.output_ceiling.reported_usage`. Recording
    evidence decides nothing about whether the attempt is recorded or
    billed: the service decides that from the error's own classification.
    """

    __slots__ = ("served", "usage")

    def __init__(self) -> None:
        self.served: Optional[ProviderServed] = None
        self.usage: Optional[LLMResponse] = None

    def identify(self, served: ProviderServed) -> None:
        """Record the provider's identification of its reply, as it is now."""
        self.served = served

    def report_usage(self, usage: Optional[LLMResponse]) -> None:
        """Record the usage the provider reported; ``None`` records nothing,
        so a report never seen stays unknown."""
        if usage is not None:
            self.usage = usage

    def reported(self) -> Optional[LLMResponse]:
        """The reported usage with the provider's identification attached,
        or ``None`` when none was reported."""
        return attach_provider_served(self.usage, self.served)

    def retain(self, error: BaseException) -> BaseException:
        """Record this receipt on ``error`` and return the same ``error``.

        Only what this attempt observed is recorded, on ``error`` itself, so
        a reader finds it before anything ``error`` wraps: the caller that
        catches ``error`` reads the attempt it just ran. A field the provider
        never reported is left as ``error`` has it. Synchronous, so a caller
        that re-raises ``error``, cancellation included, does so with no
        await in between.
        """
        attach_provider_served(error, self.served)
        attach_reported_usage(error, self.reported())
        return error


__all__ = [
    "PROVIDER_SERVED_KEY",
    "ProviderAttemptReceipt",
    "ProviderServed",
    "attach_provider_served",
    "provider_served",
]
