"""Re-exports of model metadata types from the SDK.

The canonical definitions of :class:`ModelInfo` and
:class:`ModelCategory` live in :mod:`kestrel_sdk.llm.model_info`
(promoted to the SDK in 0.5.0 alongside :class:`LLMAdapter`) so
third-party LLM provider plugins can depend only on
``kestrel-sovereign-sdk`` without pulling in the framework.

This module preserves the historical import path
``kestrel_sovereign.llm.model_metadata`` for in-tree callers and
external consumers that imported from here pre-0.5.0. New code
should prefer ``from kestrel_sdk.llm import ModelInfo, ModelCategory``.

Behavioral note for callers constructing :class:`ModelInfo`
directly: the SDK's ``ModelInfo`` defaults ``supports_streaming``
to ``False`` (conservative, matching ``supports_vision`` and
``supports_tools``). Adapters that produce ``ModelInfo`` for
streaming-capable models must set ``supports_streaming=True``
explicitly. ``ModelInfo.from_dict`` preserves the legacy
"streaming when key absent" assumption for serialized catalogs
written before the field existed.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from kestrel_sdk.llm import ModelCategory
from kestrel_sdk.llm import ModelInfo as _SDKModelInfo


@dataclass
class ModelInfo(_SDKModelInfo):
    """Framework ModelInfo — the SDK dataclass plus ``underlying_provider``.

    Meta-provider catalogs (OpenRouter, and any future aggregator) route to
    many upstream vendors, so a single ``provider="openrouter"`` bucket hides
    the real substrate. OpenRouter's ``/models`` ids already encode it as a
    prefix (``anthropic/claude-3-opus`` → ``anthropic``); this field carries
    that prefix so UI faceting doesn't have to re-parse the id (#2262).

    The canonical shape still lives in the SDK; this subclass only ADDS the
    optional field (never reshapes existing ones) so external adapters that
    construct the SDK ``ModelInfo`` remain forward-compatible — they simply
    leave ``underlying_provider`` at ``None``. All in-tree code imports
    ``ModelInfo`` from this module, so it uniformly gets the extended type.
    """

    underlying_provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["underlying_provider"] = self.underlying_provider
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        # Reuse the SDK round-trip for the shared fields, then layer the
        # framework-only field on top. ``super().from_dict`` returns a
        # ``cls`` instance (classmethod), so the base fields land correctly
        # and only ``underlying_provider`` needs to be applied here.
        model = super().from_dict(data)
        model.underlying_provider = data.get("underlying_provider")
        return model


__all__ = ["ModelCategory", "ModelInfo"]
