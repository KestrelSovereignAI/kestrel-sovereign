"""Helpers for council scripts after council moved to an optional package."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace


_PACKAGE = "kestrel_feature_council"
_INSTALL_HINT = (
    "Council scripts require the optional package. "
    "Install it with: pip install kestrel-feature-council"
)


def _load_module(name: str):
    try:
        return import_module(f"{_PACKAGE}.{name}")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == _PACKAGE or missing.startswith(f"{_PACKAGE}."):
            raise RuntimeError(_INSTALL_HINT) from exc
        raise


def load_council_exports() -> SimpleNamespace:
    """Load council package symbols without making core depend on the package."""
    models = _load_module("models")
    deliberation = _load_module("deliberation")
    costing = _load_module("costing")
    storage = _load_module("storage")

    return SimpleNamespace(
        Evidence=models.Evidence,
        CouncilConfig=models.CouncilConfig,
        CouncilMember=models.CouncilMember,
        ConsensusRule=models.ConsensusRule,
        convene_council=deliberation.convene_council,
        print_token_usage_summary=costing.print_token_usage_summary,
        get_storage=storage.get_storage,
    )
