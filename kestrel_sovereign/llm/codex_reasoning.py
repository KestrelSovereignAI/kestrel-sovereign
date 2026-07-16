"""Model-aware reasoning-effort policy for the Codex app-server route.

Codex owns the account-specific model catalog. Kestrel owns one compatibility
rule: a configured effort above the selected model's advertised ceiling is
clamped to that ceiling before either app-server request starts. Keep that
policy isolated from provider-neutral configuration and the standard OpenAI
API adapter.

``max`` is intentionally *not* a global alias. It remains a real effort for
models that advertise it, and aliases to the highest recognized advertised
level (currently ``xhigh`` for GPT-5.5) only when the selected model proves a
lower ceiling. The rank table can be retired in Kestrel 1.0 once the Codex
protocol advertises stable semantic ranks rather than only ordered picker
values; capability negotiation itself remains the source of truth.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .codex_app_server import CodexAppServerError

logger = logging.getLogger(__name__)

# Ordered compatibility vocabulary. The top levels are model-dependent:
# current GPT-5.6 catalogs advertise ``max``/``ultra`` while GPT-5.5 tops out
# at ``xhigh``. Directly advertised future values pass through even before this
# compatibility order learns about them.
_CODEX_REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)
_CODEX_REASONING_EFFORT_RANK = {
    effort: rank for rank, effort in enumerate(_CODEX_REASONING_EFFORTS)
}
_MODEL_LIST_PAGE_SIZE = 100
_MODEL_LIST_MAX_PAGES = 100
_MODEL_METADATA_MAX_EFFORTS = 100
_DISK_CACHE_MAX_MODELS = 1_000


class CodexReasoningCapabilityState(str, Enum):
    """What the selected model's effort metadata actually proves."""

    KNOWN = "known"
    NO_EFFORT = "no_effort"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CodexReasoningCapability:
    """Tri-state evidence for one selected model's reasoning vocabulary."""

    state: CodexReasoningCapabilityState
    efforts: tuple[str, ...] = ()
    source: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, CodexReasoningCapabilityState):
            raise ValueError("invalid Codex reasoning capability state")
        if self.state is CodexReasoningCapabilityState.KNOWN and not self.efforts:
            raise ValueError("known Codex reasoning capability requires efforts")
        if self.state is not CodexReasoningCapabilityState.KNOWN and self.efforts:
            raise ValueError(
                "non-known Codex reasoning capability cannot carry efforts"
            )


class AppServerRequester(Protocol):
    """Minimal app-server RPC boundary needed for catalog negotiation."""

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 120,
    ) -> Any: ...


def _known(efforts: tuple[str, ...], source: str) -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.KNOWN,
        efforts=efforts,
        source=source,
    )


def _no_effort(source: str) -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.NO_EFFORT,
        source=source,
    )


def _unknown(source: str, detail: str) -> CodexReasoningCapability:
    return CodexReasoningCapability(
        CodexReasoningCapabilityState.UNKNOWN,
        source=source,
        detail=detail,
    )


def normalize_codex_reasoning_effort(
    configured: Any,
    capability: CodexReasoningCapability,
    *,
    model: str | None,
) -> str | None:
    """Validate Codex effort and clamp it to the selected model's ceiling.

    A configured effort is never forwarded without capability evidence. A
    successful live lookup that explicitly advertises no efforts is distinct
    from missing or unavailable metadata, but both fail closed when an effort
    was configured. An absent setting needs no capability evidence and remains
    absent.
    """
    if configured is None:
        return None

    advertised_future_value = (
        isinstance(configured, str)
        and capability.state is CodexReasoningCapabilityState.KNOWN
        and configured in capability.efforts
    )
    if not isinstance(configured, str) or (
        configured not in _CODEX_REASONING_EFFORT_RANK and not advertised_future_value
    ):
        values = ", ".join(_CODEX_REASONING_EFFORTS)
        raise CodexAppServerError(
            f"Invalid Codex model_reasoning_effort {configured!r} for model "
            f"{model or 'auto'!r}. Supported compatibility values: {values}; "
            "a future value is valid only when model/list advertises it for "
            "the selected model. Update the effective Codex config before "
            "starting a turn."
        )

    if capability.state is CodexReasoningCapabilityState.UNKNOWN:
        detail = capability.detail or "selected-model metadata was unavailable"
        raise CodexAppServerError(
            f"Cannot apply configured Codex model_reasoning_effort "
            f"{configured!r} for model {model or 'auto'!r}: its reasoning "
            f"ceiling could not be proved from {capability.source or 'the model catalog'} "
            f"({detail}). Refresh the Codex model catalog or remove the "
            "configured effort before starting a turn."
        )
    if capability.state is CodexReasoningCapabilityState.NO_EFFORT:
        raise CodexAppServerError(
            f"Cannot apply configured Codex model_reasoning_effort "
            f"{configured!r} for model {model or 'auto'!r}: "
            f"{capability.source or 'the model catalog'} explicitly advertises "
            "no reasoning-effort support. Remove the configured effort before "
            "starting a turn."
        )

    supported = capability.efforts
    if configured in supported:
        # The protocol permits newly advertised non-empty values. Accept the
        # authoritative vocabulary first so a new level does not require a
        # framework release merely to pass through unchanged.
        return configured

    # The catalog advertises selectable UI levels and may omit valid low-end
    # protocol values (for example ``none``/``minimal``). Use recognized
    # entries to establish the upper ceiling; do not reject a lower known value
    # merely because the picker omitted it.
    recognized = tuple(
        effort for effort in supported if effort in _CODEX_REASONING_EFFORT_RANK
    )
    if not recognized:
        raise CodexAppServerError(
            f"Cannot apply configured Codex model_reasoning_effort "
            f"{configured!r} for model {model or 'auto'!r}: "
            f"{capability.source or 'the model catalog'} advertises "
            f"{', '.join(supported)}, but none establishes a compatible "
            "ceiling. Update the Codex app/CLI or choose an advertised effort."
        )
    ceiling = max(recognized, key=_CODEX_REASONING_EFFORT_RANK.__getitem__)
    if (
        _CODEX_REASONING_EFFORT_RANK[configured]
        <= _CODEX_REASONING_EFFORT_RANK[ceiling]
    ):
        return configured
    return ceiling


def _parse_efforts(
    options: Any,
    *,
    value_key: str,
    source: str,
) -> CodexReasoningCapability:
    if not isinstance(options, list):
        return _unknown(source, f"missing or non-list {value_key} metadata")
    if len(options) > _MODEL_METADATA_MAX_EFFORTS:
        return _unknown(
            source,
            f"effort metadata exceeded {_MODEL_METADATA_MAX_EFFORTS} entries",
        )
    if not options:
        return _no_effort(source)

    efforts: list[str] = []
    for option in options:
        if not isinstance(option, dict):
            return _unknown(source, "effort metadata contained a non-object entry")
        value = option.get(value_key)
        if not isinstance(value, str) or not value:
            return _unknown(source, f"effort metadata omitted non-empty {value_key}")
        if value not in efforts:
            efforts.append(value)
    return _known(tuple(efforts), source)


async def _app_server_reasoning_capability(
    app: AppServerRequester,
    model: str | None,
) -> CodexReasoningCapability:
    """Read authoritative effort evidence from paginated ``model/list``."""
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _ in range(_MODEL_LIST_MAX_PAGES):
        params: dict[str, Any] = {
            "includeHidden": True,
            "limit": _MODEL_LIST_PAGE_SIZE,
        }
        if cursor is not None:
            params["cursor"] = cursor
        response = await app.request("model/list", params, timeout=30)
        if not isinstance(response, dict) or not isinstance(response.get("data"), list):
            return _unknown(
                "live model/list",
                "the app-server returned a malformed catalog",
            )

        for entry in response["data"]:
            if not isinstance(entry, dict):
                continue
            is_selected = entry.get("model") == model or entry.get("id") == model
            if model is None:
                is_selected = entry.get("isDefault") is True
            if not is_selected:
                continue
            return _parse_efforts(
                entry.get("supportedReasoningEfforts"),
                value_key="reasoningEffort",
                source="live model/list selected-model entry",
            )

        next_cursor = response.get("nextCursor")
        if next_cursor is None:
            return _unknown(
                "live model/list",
                f"no metadata entry matched model {model or 'the app-server default'!r}",
            )
        if (
            not isinstance(next_cursor, str)
            or not next_cursor
            or next_cursor in seen_cursors
        ):
            return _unknown(
                "live model/list",
                "the app-server returned an invalid pagination cursor",
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return _unknown(
        "live model/list",
        f"catalog exceeded {_MODEL_LIST_MAX_PAGES} pages",
    )


def _cached_reasoning_capability(
    models: list[dict[str, Any]],
    model: str | None,
) -> CodexReasoningCapability:
    """Bounded compatibility evidence from raw ``models_cache.json`` rows."""
    source = "models_cache.json compatibility metadata"
    if not model:
        return _unknown(source, "no explicit selected model was available")
    if not isinstance(models, list):
        return _unknown(source, "the cache did not contain a model list")
    for entry in models[:_DISK_CACHE_MAX_MODELS]:
        if not isinstance(entry, dict) or entry.get("slug") != model:
            continue
        if "supported_reasoning_levels" not in entry:
            return _unknown(source, "the selected model omitted effort metadata")
        return _parse_efforts(
            entry["supported_reasoning_levels"],
            value_key="effort",
            source=source,
        )
    suffix = (
        f" within the first {_DISK_CACHE_MAX_MODELS} rows"
        if len(models) > _DISK_CACHE_MAX_MODELS
        else ""
    )
    return _unknown(source, f"the selected model was absent{suffix}")


async def resolve_codex_reasoning_capability(
    app: AppServerRequester,
    model: str | None,
    read_cached_models: Callable[[], list[dict[str, Any]]],
) -> CodexReasoningCapability:
    """Resolve tri-state capability evidence, preferring the live catalog.

    A successful ``model/list`` response is authoritative, including a missing
    selected model or explicit empty effort list. Disk metadata is consulted
    only when the RPC itself is unavailable, and only an explicit matching
    model row can establish a ceiling. This prevents stale cache rows from
    overriding a current live catalog or turning missing evidence into a
    permissive pass-through.
    """
    try:
        return await _app_server_reasoning_capability(app, model)
    except CodexAppServerError as exc:
        logger.warning(
            "codex: model/list unavailable while resolving reasoning effort "
            "for %r; consulting bounded models_cache.json compatibility "
            "metadata: %s",
            model or "auto",
            exc,
        )
        return _cached_reasoning_capability(read_cached_models(), model)
