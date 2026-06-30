"""Derive the UI capability set from an agent's enabled features (#2041).

Frontend capability gating (``API.hasCapability('voice')``) used to read a
*hardcoded* default set: every new feature that wanted a UI gate required a core
edit to ``CAPABILITY_KEYS`` in ``api_client.mjs``. That defeats the
"discovered, not curated" doctrine and blocks out-of-tree features.

This module computes the **feature-backed** slice of the capability set from the
features actually enabled on an agent. The capability key for a feature is its
*registry name* (the discovered source of truth — see
``feature_registry.get_registry``), so a feature gains a working
``hasCapability(<name>)`` with zero frontend edits. A feature being enabled →
its capability is ``True``; disabling it at runtime → ``False``.

The map is injected into ``window.KESTREL_UI_CONFIG.featureCapabilities`` at page
render and is also served from ``GET /api/ui/capabilities`` so the frontend can
re-fetch it when a feature is enabled/disabled at runtime (no page reload).
"""

from __future__ import annotations

import json
from typing import Any, Dict

from kestrel_sovereign.feature_registry import FeatureStatus, get_registry


def _active_class_names(agent) -> set:
    """Feature class names that are loaded AND not runtime-disabled.

    The Feature Store enable/disable endpoints toggle a per-feature ``enabled``
    flag; a feature that has never been disabled has no flag and is treated as
    enabled. A feature absent from ``agent.features`` is not loaded at all.
    """
    features = getattr(agent, "features", {}) or {}
    return {
        name
        for name, feat in features.items()
        if getattr(feat, "enabled", True)
    }


def compute_feature_capabilities(agent) -> Dict[str, bool]:
    """Map each catalogued feature's UI capability key(s) → enabled bool.

    The capability key is what the frontend gates on via ``hasCapability(...)``.
    A feature declares its UI capability key(s) in the registry's
    ``ui_capabilities`` field; when unset it defaults to the registry short
    name. This matters because the short name and the frontend key diverge for
    some panels — ``observability`` backs ``metrics``; ``security`` backs both
    ``audit`` and ``permissions`` — and emitting the bare short name would leave
    the real UI key absent (and therefore default-*true* on the frontend), so a
    disabled feature would never gate its panel off.

    A capability is ``True`` iff at least one of the features mapping to that key
    is loaded and not runtime-disabled. Disabled or uninstalled features are
    reported as ``False`` (rather than omitted) so the frontend can treat them as
    *authoritatively* off: a host override may force a feature capability off, but
    a force-*true* on a disabled feature is ignored — a disabled feature has no UI
    assets to gate into.
    """
    active = _active_class_names(agent)
    registry = get_registry(enabled_class_names=active)
    caps: Dict[str, bool] = {}
    for info in registry.values():
        enabled = info.status == FeatureStatus.ENABLED
        # ``ui_capabilities`` overrides the short name when the two diverge;
        # multiple features may map to the same key (none today, but the OR keeps
        # the key true if *any* backing feature is enabled).
        keys = info.ui_capabilities or [info.name]
        for key in keys:
            caps[key] = caps.get(key, False) or enabled
    return caps


def render_ui_config_script(agent) -> str:
    """Return an inline ``<script>`` that seeds ``window.KESTREL_UI_CONFIG``.

    Injected into the served console HTML *before* the module scripts load so
    the capability set is known before ``app.js`` runs feature registrations and
    ``initNavigation`` prunes panels (the boot-ordering requirement). Existing
    host-supplied config is preserved — only ``featureCapabilities`` is merged
    in. ``<`` is escaped so a feature name can never break out of the script.
    """
    payload = {"featureCapabilities": compute_feature_capabilities(agent)}
    encoded = json.dumps(payload).replace("<", "\\u003c")
    return (
        "<script>window.KESTREL_UI_CONFIG = Object.assign("
        f"{{}}, window.KESTREL_UI_CONFIG || {{}}, {encoded});</script>"
    )
