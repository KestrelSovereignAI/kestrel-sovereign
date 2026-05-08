"""
Model preference and solvency mixin for KestrelAgent.

Extracted from kestrel_agent.py — manages model selection, persistence,
and economic operating mode based on wallet solvency.
"""

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional


class ModelPreferenceMixin:
    """Mixin providing model preference and solvency methods for KestrelAgent."""

    MODEL_PREFERENCE_KEY = "model_preference"

    async def list_available_models(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        List all available models from configured LLM providers.

        Args:
            use_cache: If True, use cached models if available (default: True)

        Returns:
            List of model dictionaries with id, provider, name, description
        """
        # ModelAgent.list_models() now returns a ToolResult envelope
        # (#1061 wave 10); the legacy list[dict] payload lives under
        # .data["models"]. Unwrap so existing public callers see the
        # documented shape. On ERROR raise — the legacy method raised
        # on provider/cache failures, and silently returning [] would
        # mask "discovery failed" as "no models available".
        from kestrel_sdk.tools.result import ToolResultStatus

        envelope = await self.model_agent.list_models(use_cache=use_cache)
        if envelope.status is ToolResultStatus.ERROR:
            raise RuntimeError(envelope.error or "list_models failed")
        if envelope.data is None:
            return []
        return envelope.data.get("models", [])

    def set_model(self, model_id: str) -> str:
        """
        Set the LLM model for this agent.

        Args:
            model_id: The model ID to use (e.g., "gpt-5", "claude-sonnet-4-5")

        Returns:
            Confirmation message
        """
        return self.model_agent.set_model_preference(model_id)

    def get_current_model(self) -> str:
        """
        Get the current LLM model being used by this agent.

        Delegates to LLMService.get_active_model_id() as the single
        source of truth, then formats with provider prefix.

        Returns:
            Current model ID (provider/model format)
        """
        from kestrel_sovereign.llm.service import resolve_active_model_selection
        return resolve_active_model_selection(self.llm_service)["model"]

    async def _load_model_preference(self) -> None:
        """Load persisted model preference from agent_metadata table.

        Persistence schema: ``{"vendor": str, "model": str, "route": str|None}``.
        Rows using the legacy ``{"model", "provider"}`` shape are dropped
        silently — the agent starts with no mandate and the user re-selects
        via the UI.
        """
        try:
            result = await self._raw_storage.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.MODEL_PREFERENCE_KEY),
            )
            if not result:
                return
            pref = json.loads(result[0][0])
            # New shape.
            model = pref.get("model")
            vendor = pref.get("vendor")
            route = pref.get("route")
            if vendor is None and "provider" in pref:
                # Legacy row — drop it. User re-selects via UI.
                logging.warning(
                    "Ignoring legacy model_preference row for %s (old shape {model, provider}); "
                    "re-select via the UI to persist in the new {vendor, model, route} shape.",
                    self.agent_id,
                )
                return
            if model and model != "auto":
                self.llm_service.set_model_preference(model, vendor, route)
                if vendor and route:
                    logging.info("Loaded persisted model preference: %s:%s/%s", vendor, route, model)
                elif vendor:
                    logging.info("Loaded persisted model preference: %s/%s", vendor, model)
                else:
                    logging.info("Loaded persisted model preference: %s", model)
        except Exception as e:
            logging.warning(f"Failed to load model preference: {e}")

    async def _persist_model_preference(
        self,
        model: str | None,
        vendor: str | None,
        route: str | None,
    ) -> None:
        """Persist model preference to agent_metadata table."""
        try:
            value = json.dumps({"vendor": vendor, "model": model, "route": route})
            await self._raw_storage.db.execute(
                """INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (self.agent_id, self.MODEL_PREFERENCE_KEY, value, datetime.now(timezone.utc)),
            )
        except Exception as e:
            logging.warning(f"Failed to persist model preference: {e}")

    def _get_local_model_fallback(self) -> str:
        """Get the configured local (ollama) model for economy/solvency fallback."""
        # Check ollama provider in the providers list. Providers are keyed
        # by composite "<vendor>:<route>" (e.g. "ollama:local") in the
        # vendor/route/model architecture — match on the ``vendor`` field
        # to find any ollama route, not the bare ``name``.
        for provider in self.llm_service.providers:
            if provider.get("vendor") == "ollama":
                return provider.get("model", "auto")
        # Fall back to config
        if hasattr(self.llm_service, 'config'):
            return self.llm_service.config.get("ollama", {}).get("model", "auto")
        return "auto"

    async def check_solvency(self) -> Optional[str]:
        """
        Checks the agent's wallet balance and determines the economic operating mode.
        Returns the model preference based on solvency.

        Uses total USD-equivalent value across all currencies so that agents holding
        ETH, MATIC, or other non-FIL assets are correctly classified as solvent.
        FIL balance is checked as a fallback for FIL-only wallets.
        """
        try:
            fil_balance = self.wallet.get_balance()
            usd_balance = self.wallet.get_total_balance_usd()

            # Green Zone: > $5 USD equivalent (or > 10 FIL for FIL-only wallets)
            if usd_balance > Decimal("5.0") or fil_balance > Decimal("10.0"):
                if self._current_model_preference != "NORMAL":
                    logging.info(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"Operating in NORMAL mode."
                    )
                    self._current_model_preference = "NORMAL"
                return None  # No override, use default/mandated models

            # Yellow Zone: > $0.50 USD equivalent (or > 1 FIL)
            elif usd_balance > Decimal("0.50") or fil_balance > Decimal("1.0"):
                if self._current_model_preference != "ECONOMY":
                    logging.warning(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"Switching to ECONOMY mode (Local Models)."
                    )
                    self._current_model_preference = "ECONOMY"
                return self._get_local_model_fallback()

            # Red Zone: Critical (< $0.50 USD and < 1 FIL)
            else:
                if self._current_model_preference != "CRITICAL":
                    logging.error(
                        f"Solvency Check: ${usd_balance:.2f} USD / {fil_balance} FIL. "
                        f"CRITICAL SOLVENCY. Forced to minimal model."
                    )
                    self._current_model_preference = "CRITICAL"
                return self._get_local_model_fallback()

        except Exception as e:
            logging.error(f"Solvency check failed: {e}", exc_info=True)
            return None
