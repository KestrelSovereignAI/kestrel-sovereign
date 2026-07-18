"""Canonical active-substrate resolution for identity portability.

Identity export and the ``assess_substrate`` tool both need the substrate that
the active agent is actually configured to use.  The LLM service already owns
that routing truth: its mandate preference identifies the selected route and
its provider entries carry the adapter metadata.  This module translates that
runtime state into the stable values used by :class:`SubstrateType`.

No process-global LLM service is consulted.  That is important in multi-agent
hosts, where each agent can have a different route and model.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from .identity_package import SubstrateType

logger = logging.getLogger(__name__)


_FAMILY_TO_SUBSTRATE = {
    "claude": SubstrateType.ANTHROPIC_CLAUDE.value,
    SubstrateType.ANTHROPIC_CLAUDE.value: SubstrateType.ANTHROPIC_CLAUDE.value,
    "gpt": SubstrateType.OPENAI_GPT.value,
    SubstrateType.OPENAI_GPT.value: SubstrateType.OPENAI_GPT.value,
    "gemini": SubstrateType.GOOGLE_GEMINI.value,
    SubstrateType.GOOGLE_GEMINI.value: SubstrateType.GOOGLE_GEMINI.value,
    "llama": SubstrateType.META_LLAMA.value,
    SubstrateType.META_LLAMA.value: SubstrateType.META_LLAMA.value,
    "mistral": SubstrateType.MISTRAL.value,
    "mixtral": SubstrateType.MISTRAL.value,
    SubstrateType.MISTRAL.value: SubstrateType.MISTRAL.value,
}

_PROFILED_SUBSTRATES = frozenset({
    SubstrateType.ANTHROPIC_CLAUDE.value,
    SubstrateType.OPENAI_GPT.value,
    SubstrateType.GOOGLE_GEMINI.value,
    SubstrateType.META_LLAMA.value,
    SubstrateType.OLLAMA_LOCAL.value,
    SubstrateType.OPENROUTER.value,
})


@dataclass(frozen=True, slots=True)
class SubstrateResolution:
    """Resolved substrate plus the route/model evidence used to select it."""

    substrate: str
    vendor: Optional[str] = None
    route: Optional[str] = None
    model: Optional[str] = None
    provider_name: Optional[str] = None
    adapter_family: Optional[str] = None
    capability_profile_known: bool = False
    reason: Optional[str] = None

    @property
    def provider_selector(self) -> Optional[str]:
        """Return the canonical ``vendor[:route]`` selector when available."""

        if self.vendor and self.route:
            return f"{self.vendor}:{self.route}"
        return self.vendor or self.provider_name


def _preference(llm_service: Any) -> dict[str, Any]:
    getter = getattr(llm_service, "get_model_preference", None)
    if not callable(getter):
        return {}
    value = getter()
    return dict(value) if isinstance(value, Mapping) else {}


def _providers(llm_service: Any) -> list[Mapping[str, Any]]:
    value = getattr(llm_service, "providers", None)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [provider for provider in value if isinstance(provider, Mapping)]


def _select_provider(
    providers: list[Mapping[str, Any]], preference: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Choose the provider entry named by the mandate, then the route default."""

    vendor = preference.get("vendor")
    route = preference.get("route")
    if vendor and route:
        selector = f"{vendor}:{route}"
        for provider in providers:
            if provider.get("name") == selector or (
                provider.get("vendor") == vendor
                and provider.get("route") == route
            ):
                return provider
    if vendor:
        for provider in providers:
            if provider.get("vendor") == vendor:
                return provider

    # A model-only mandate deliberately carries no vendor.  Prefer an exact
    # model match when it identifies one route; otherwise provider order is the
    # LLM service's canonical default ordering.
    model = preference.get("model")
    if model and model != "auto":
        matches = [provider for provider in providers if provider.get("model") == model]
        if len(matches) == 1:
            return matches[0]
    return providers[0] if providers else None


def _model_family(model: Optional[str]) -> Optional[str]:
    """Infer a heterogeneous adapter's concrete family from its active model."""

    normalized = (model or "").strip().lower()
    if not normalized or normalized == "auto":
        return None
    if "claude" in normalized:
        return "claude"
    if "gemini" in normalized:
        return "gemini"
    if "llama" in normalized:
        return "llama"
    if "mistral" in normalized or "mixtral" in normalized:
        return "mistral"
    if (
        "gpt" in normalized
        or normalized.startswith(("o1", "o3", "o4"))
        or "/o1" in normalized
        or "/o3" in normalized
        or "/o4" in normalized
    ):
        return "gpt"
    return None


def _substrate_value(family: Optional[str]) -> str:
    if not family:
        return SubstrateType.UNKNOWN.value
    normalized = family.strip().lower()
    # Plugin families are intentionally preserved when they do not yet have a
    # first-party SubstrateType.  Returning the adapter's stable family token
    # is more truthful and useful than erasing it to UNKNOWN.
    return _FAMILY_TO_SUBSTRATE.get(normalized, normalized)


def resolve_active_substrate(llm_service: Any) -> SubstrateResolution:
    """Resolve the active agent's model-family substrate from LLM route truth.

    Adapters are the primary source.  Heterogeneous adapters return ``None``;
    for those, the active model identifies a concrete family when possible.
    OpenRouter/Ollama retain their explicit multi/local substrate value when
    the model name itself is not informative.
    """

    if llm_service is None:
        return SubstrateResolution(
            substrate=SubstrateType.UNKNOWN.value,
            reason="llm_service_unavailable",
        )

    preference = _preference(llm_service)
    providers = _providers(llm_service)
    provider = _select_provider(providers, preference)
    if provider is None:
        return SubstrateResolution(
            substrate=SubstrateType.UNKNOWN.value,
            model=preference.get("model"),
            reason="no_configured_provider_routes",
        )

    vendor = provider.get("vendor") or preference.get("vendor")
    route = provider.get("route") or preference.get("route")
    model = preference.get("model")
    if not model or model == "auto":
        model = provider.get("model") or "auto"
    provider_name = provider.get("name")
    adapter = provider.get("adapter")

    family: Optional[str] = None
    substrate_method = getattr(adapter, "substrate_type", None)
    if callable(substrate_method):
        try:
            reported = substrate_method()
        except Exception as exc:  # noqa: BLE001 - third-party adapter boundary
            logger.warning(
                "Adapter substrate metadata failed for %s: %s",
                provider_name or vendor or "unknown route",
                exc,
            )
            return SubstrateResolution(
                substrate=SubstrateType.UNKNOWN.value,
                vendor=vendor,
                route=route,
                model=model,
                provider_name=provider_name,
                reason=f"adapter_metadata_failed:{type(exc).__name__}",
            )
        if isinstance(reported, str) and reported.strip():
            family = reported.strip().lower()

    if family is None:
        family = _model_family(model)

    if family is not None:
        substrate = _substrate_value(family)
        profile_known = substrate in _PROFILED_SUBSTRATES
        return SubstrateResolution(
            substrate=substrate,
            vendor=vendor,
            route=route,
            model=model,
            provider_name=provider_name,
            adapter_family=family,
            capability_profile_known=profile_known,
            reason=None if profile_known else "capability_profile_unavailable",
        )

    vendor_key = str(vendor or "").lower()
    if vendor_key == "openrouter":
        substrate = SubstrateType.OPENROUTER.value
        reason = "heterogeneous_route_model_family_unknown"
    elif vendor_key == "ollama":
        substrate = SubstrateType.OLLAMA_LOCAL.value
        reason = "local_route_model_family_unknown"
    else:
        substrate = SubstrateType.UNKNOWN.value
        reason = "adapter_and_model_family_unknown"
    return SubstrateResolution(
        substrate=substrate,
        vendor=vendor,
        route=route,
        model=model,
        provider_name=provider_name,
        capability_profile_known=substrate in _PROFILED_SUBSTRATES,
        reason=reason,
    )


__all__ = ["SubstrateResolution", "resolve_active_substrate"]
