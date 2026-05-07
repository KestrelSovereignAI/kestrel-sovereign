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

from kestrel_sdk.llm import ModelCategory, ModelInfo

__all__ = ["ModelCategory", "ModelInfo"]
