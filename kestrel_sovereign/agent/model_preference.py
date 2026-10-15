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

from kestrel_sovereign.execution_custody import is_execution_control_error
from kestrel_sovereign.llm.provider_policy import (
    OverrideProvenance,
    agent_provenance,
    override_writer,
)


class ModelPreferenceMixin:
    """Mixin providing model preference and solvency methods for KestrelAgent."""

    MODEL_PREFERENCE_KEY = "model_preference"
    # Who wrote the stored preference (#2652), kept beside it rather than in
    # its row, whose shape readers and backups already rely on.
    MODEL_PREFERENCE_PROVENANCE_KEY = "model_preference_provenance"
    EMBEDDING_ROUTE_KEY = "embedding_route"
    EMBEDDING_MODEL_OVERRIDES_KEY = "embedding_model_overrides"

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

        Raises:
            ProviderPolicyRefusal: the override is the agent's own, so it may
                select only a provider in the approved chain (#2652).
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

        See :func:`apply_persisted_model_preference`, which offline tools run
        too so they resolve what this agent resolves.
        """
        await apply_persisted_model_preference(
            self.llm_service,
            getattr(self._raw_storage, "db", None),
            self.agent_id,
        )

    async def _persist_model_preference(
        self,
        model: str | None,
        vendor: str | None,
        route: str | None,
        provenance: Optional[OverrideProvenance] = None,
    ) -> None:
        """Persist model preference to agent_metadata table.

        The ``LLMService`` persistence callback: a failed write is logged, not
        raised. A caller that must know the mandate reached storage uses
        :meth:`_write_model_preference`.
        """
        try:
            await self._write_model_preference(
                model, vendor, route, provenance=provenance
            )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to persist model preference: {e}")
        # With an auto embedding_route, embeddings follow the chat route.
        await self.refresh_active_embedding_profile()

    async def _write_model_preference(
        self,
        model: str | None,
        vendor: str | None,
        route: str | None,
        *,
        provenance: Optional[OverrideProvenance] = None,
    ) -> None:
        """Write the model preference row; a database error propagates.

        ``provenance`` records who wrote the override, so boot replays it under
        the same authority (#2652). It is written in the same statement as the
        preference, with the mandate it describes; a preference written without
        one records no writer, and is replayed as the agent's.
        """
        mandate = {"vendor": vendor, "model": model, "route": route}
        record = {
            "mandate": mandate,
            "provenance": (
                provenance.to_dict() if provenance is not None and model else None
            ),
        }
        now = datetime.now(timezone.utc)
        await self._raw_storage.db.execute(
            """INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
               VALUES (?, ?, ?, ?), (?, ?, ?, ?)""",
            (
                self.agent_id, self.MODEL_PREFERENCE_KEY, json.dumps(mandate), now,
                self.agent_id, self.MODEL_PREFERENCE_PROVENANCE_KEY,
                json.dumps(record), now,
            ),
        )

    async def _load_embedding_route(self) -> None:
        """Load the persisted top-level embedding_route knob (#2263).

        Mirrors ``_load_model_preference``: a persisted runtime value in
        agent_metadata overrides the config default. Absence leaves the
        config-seeded value (or None = auto) in place.
        """
        try:
            result = await self._raw_storage.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.EMBEDDING_ROUTE_KEY),
            )
            if not result:
                return
            route = json.loads(result[0][0])
            # ``route`` may be a string (explicit selector) or None (auto).
            # For an explicit route, fold live embedding discovery into route
            # capabilities BEFORE the sync validator runs (#2338), mirroring the
            # settings endpoint. A route discovered dynamically (e.g. OpenRouter
            # with no TOML ``embedding_model`` pin) advertises embedding support
            # only after reconciliation; without this, ``set_embedding_route`` ->
            # ``_validate_embedding_route`` would reject the persisted route as
            # "does not advertise embedding support", the exception would be
            # swallowed, and the persisted route would silently not apply after
            # restart. Best-effort — a discovery hiccup must not fail boot.
            if route not in (None, "", "auto", "none") and hasattr(
                self.llm_service, "reconcile_embedding_capabilities"
            ):
                try:
                    await self.llm_service.reconcile_embedding_capabilities(
                        use_cache=True
                    )
                except Exception as e:  # pragma: no cover - never block boot
                    if is_execution_control_error(e):
                        raise
                    logging.debug(
                        "embedding capability reconcile skipped at boot: %s", e
                    )
            # Boot applies the persisted value WITHOUT the live upstream probe
            # that explicit sets run (#2326): probing at boot could fail startup
            # on a transient upstream outage. Instead we log loudly so a
            # persisted route that is dead upstream is visible in the boot logs
            # (the write path still degrades gracefully to keyword search).
            self.llm_service.set_embedding_route(route, persist=False)
            if route and route != "none":
                logging.warning(
                    "Loaded persisted embedding_route %r without a live upstream "
                    "probe (#2326): if the route's model is not currently served, "
                    "storage writes will degrade to keyword search. Re-set it via "
                    "the settings API/UI to live-validate.",
                    route,
                )
            elif route == "none":
                logging.info(
                    "Loaded persisted embedding_route: none (embeddings disabled)"
                )
            else:
                logging.info("Loaded persisted embedding_route: auto (follow chat)")
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to load embedding_route: {e}")

    async def _persist_embedding_route(self, route: str | None) -> None:
        """Persist the top-level embedding_route knob to agent_metadata (#2263)."""
        try:
            value = json.dumps(route)
            await self._raw_storage.db.execute(
                """INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (self.agent_id, self.EMBEDDING_ROUTE_KEY, value, datetime.now(timezone.utc)),
            )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to persist embedding_route: {e}")
        await self.refresh_active_embedding_profile()

    async def refresh_active_embedding_profile(self) -> None:
        """Act on the embedding profile this agent now resolves.

        The one path for boot, once the embedding config is loaded, for
        every persisted runtime change to that config, and for every
        resolution that changes a route's embedding model or width after it
        (the LLM service's embedding profile listener): record the profile
        for offline tools (#3420), then compare it with the declared width of
        the conversation vector column (#2913). Re-running the comparison on a
        change keeps ``/health/detailed``'s misconfiguration about the profile
        the agent writes with now, not the one it booted with (#3606).

        With no storage (torn down by a failed boot) there is no column to
        compare and nothing written to flag.
        """
        await self.record_active_embedding_profile()
        conversation = getattr(self._raw_storage, "conversation", None)
        if conversation is None:
            return
        await conversation.check_embedding_column_dimensions()

    async def record_active_embedding_profile(self) -> None:
        """Record the embedding profile this agent resolves (#3420).

        Offline tools compare against it: ``kestrel embeddings reindex``
        refuses to move rows to any other profile, and ``embeddings verify``
        counts the vectors that are off it (see
        :mod:`kestrel_sovereign.storage.active_embedding_profile`). Called
        through :meth:`refresh_active_embedding_profile`, at boot once the
        embedding config is loaded, after every persisted change to it, and
        after a resolution that changes the profile.

        Skipped while a local-only privacy mode forces the resolution: it
        writes no durable rows, and its forced route does not describe the
        corpus. Best-effort, like the persistence it follows.
        """
        try:
            privacy_agent = getattr(self, "privacy_agent", None)
            if privacy_agent is not None and not (
                privacy_agent.privacy_config.allows_cloud_llm()
            ):
                return
            db = getattr(self._raw_storage, "db", None)
            if db is None or not self.llm_service:
                return
            from kestrel_sovereign.llm.embedding_service import (
                get_provider_embedding_service,
            )
            from kestrel_sovereign.storage.active_embedding_profile import (
                record_active_embedding_profile,
            )

            profile_id = await record_active_embedding_profile(
                db,
                self.agent_id,
                get_provider_embedding_service(self.llm_service),
            )
            if profile_id:
                logging.info("Recorded active embedding profile: %s", profile_id)
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to record active embedding profile: {e}")

    async def _dominant_embedding_profile(self) -> Optional[Dict[str, Any]]:
        """Return the DB's dominant existing embedding profile, or ``None`` (#2366).

        Wired into the LLM service as the corpus-profile provider so auto
        embedding-model resolution prefers a model that keeps new memories in
        the same coordinate space the existing corpus already uses. Best-effort:
        an empty corpus, an unreadable table, or a missing DB handle yields
        ``None`` and resolution falls back to hint/catalog order.
        """
        try:
            db = getattr(self._raw_storage, "db", None)
            if db is None:
                return None
            from kestrel_sovereign.storage.embedding_reindex import (
                dominant_embedding_profile,
            )

            return await dominant_embedding_profile(
                db, agent_id=getattr(self, "agent_id", None)
            )
        except Exception as e:  # pragma: no cover - never break resolution
            if is_execution_control_error(e):
                raise
            logging.debug(f"dominant embedding profile lookup failed: {e}")
            return None

    async def _load_route_embedding_models(self) -> None:
        """Load persisted per-route embedding_model overrides (#2337).

        Mirrors ``_load_embedding_route``: a persisted runtime map in
        agent_metadata (``{"<vendor>:<route>": {"model": str, "dim": int|None}}``)
        re-applies the operator's UI-chosen model pins over the config/discovery
        default. Applied WITHOUT the live probe explicit sets run — probing at
        boot could fail startup on a transient upstream outage; the write path
        still degrades gracefully to keyword search if a pinned model is dead.
        """
        try:
            result = await self._raw_storage.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.EMBEDDING_MODEL_OVERRIDES_KEY),
            )
            if not result:
                return
            overrides = json.loads(result[0][0])
            if not isinstance(overrides, dict):
                return
            for route, spec in overrides.items():
                if not isinstance(spec, dict):
                    continue
                model = spec.get("model")
                dim = spec.get("dim")
                if not model:
                    continue
                try:
                    self.llm_service.set_route_embedding_model(
                        route, model, dim, persist=False
                    )
                    logging.info(
                        "Loaded persisted embedding_model pin for %s: %s%s",
                        route,
                        model,
                        f" @ {dim}" if dim is not None else "",
                    )
                except Exception as e:  # pragma: no cover - never block boot
                    if is_execution_control_error(e):
                        raise
                    logging.warning(
                        "Skipping persisted embedding_model pin for %s: %s", route, e
                    )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to load embedding_model overrides: {e}")

    async def _persist_route_embedding_models(self, overrides: dict) -> None:
        """Persist per-route embedding_model overrides to agent_metadata (#2337)."""
        try:
            value = json.dumps(overrides or {})
            await self._raw_storage.db.execute(
                """INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (
                    self.agent_id,
                    self.EMBEDDING_MODEL_OVERRIDES_KEY,
                    value,
                    datetime.now(timezone.utc),
                ),
            )
        except Exception as e:
            if is_execution_control_error(e):
                raise
            logging.warning(f"Failed to persist embedding_model overrides: {e}")
        await self.refresh_active_embedding_profile()

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
        wallet = getattr(self, "wallet", None)
        if wallet is None:
            logging.debug("Solvency check skipped: wallet feature is not available.")
            return None

        try:
            fil_balance = wallet.get_balance()
            usd_balance = wallet.get_total_balance_usd()

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
            if is_execution_control_error(e):
                raise
            logging.error(f"Solvency check failed: {e}", exc_info=True)
            return None


def replay_model_preference(
    llm_service: Any,
    model: str,
    vendor: Optional[str],
    route: Optional[str],
    *,
    provenance: Optional[OverrideProvenance] = None,
) -> None:
    """Re-apply a model mandate this agent already accepted.

    Used for the persisted preference at boot and for the mandate a local-only
    privacy mode displaced, when the agent leaves that mode (#3554). Raises
    ``ValueError`` when a local model-ignoring route no longer serves ``model``.

    Re-applying our OWN decision, so do not re-validate it against the live
    catalog (#3190).

    Validation belongs at the boundary where NEW information enters —
    `set_model_preference` from an operator or a tool, where #1927/#1946 guard
    against a hallucinated triple being written. Replaying a triple this agent
    already accepted is a different boundary, and gating it on a catalog
    fetched seconds earlier means a transient discovery problem silently
    revokes a deliberate choice.

    That is exactly the 2026-08-31 outage: the Anthropic key was disabled,
    `GET /v1/models` 401ed, the collapsed catalog "proved" claude-opus-5
    unservable, the boot replay raised, its caller swallowed it, and all four
    agents booted unpinned onto a 1B local model. The route itself was fine
    throughout — only discovery was broken.

    If the pin really is unservable, `resolve_provider_routing` raises
    `LLMProviderUnavailableError` at use time. Loud at the point of use beats
    silent at boot.

    Local model-ignoring routes keep their validation (#3190 r8 P2).
    `llama_cpp` and `ollama` serve whatever model is currently loaded and
    ignore the requested id, and the streaming paths never call
    `_model_available_for_route`. So a stale id restored without validation is
    not merely a wrong pin — responses from the newly loaded model get reported
    and METERED as the model that is no longer there.

    The bypass exists because a remote vendor's catalog can be transiently
    unfetchable while the route still serves the pinned model perfectly. That
    argument does not hold for a local server: its catalog is what it has
    actually loaded, so a populated local catalog that disagrees is evidence,
    not a discovery artefact.

    The provider policy is not bypassed (#2652): ``provenance`` is who wrote
    the mandate, and a replay is admitted for it as the write was. Without
    one, the mandate counts as the agent's, so an unattributed override
    outside the approved chain is refused here, not reinstated.
    """
    with override_writer(provenance or agent_provenance(via="replay")):
        llm_service.set_model_preference(
            model, vendor, route, validate=_revalidated_on_replay(vendor)
        )


def resolve_replayed_model_preference(
    llm_service: Any,
    model: str,
    vendor: Optional[str],
    route: Optional[str],
    *,
    provenance: Optional[OverrideProvenance] = None,
) -> Dict[str, Optional[str]]:
    """The mandate :func:`replay_model_preference` would install.

    Validated the same way, and raising the same ``ValueError``, without
    installing it: a privacy-mode exit records the mandate it restores before
    it takes effect (#3554).
    """
    with override_writer(provenance or agent_provenance(via="replay")):
        return llm_service.resolve_model_preference(
            model, vendor, route, validate=_revalidated_on_replay(vendor)
        )


async def _stored_provenance(
    db: Any, agent_id: Optional[str], mandate: Dict[str, Optional[str]]
) -> Optional[OverrideProvenance]:
    """Who wrote the stored ``mandate``, when its record still describes it."""
    key = ModelPreferenceMixin.MODEL_PREFERENCE_PROVENANCE_KEY
    if agent_id:
        rows = await db.fetchall(
            "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
            (agent_id, key),
        )
    else:
        rows = await db.fetchall(
            "SELECT value FROM agent_metadata WHERE key = ?", (key,)
        )
    if not rows or len(rows) != 1:
        return None
    try:
        record = json.loads(rows[0][0])
    except (TypeError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("mandate") != mandate:
        return None
    return OverrideProvenance.from_dict(record.get("provenance"))


def _revalidated_on_replay(vendor: Optional[str]) -> bool:
    """Whether a replayed mandate on ``vendor`` is checked against the catalog.

    Only on a local model-ignoring route; see :func:`replay_model_preference`.
    """
    from kestrel_sovereign.llm.service import _MODEL_IGNORING_VENDORS

    return vendor in _MODEL_IGNORING_VENDORS


async def apply_persisted_model_preference(
    llm_service: Any, db: Any, agent_id: Optional[str]
) -> Optional[str]:
    """Re-apply the chat-model preference an agent persisted in agent_metadata.

    The one loader for that preference. The agent runs it at boot, and
    ``kestrel embeddings reindex`` / ``kestrel constitution reanchor`` run it
    before they resolve the agent's embedding profile: with an auto
    ``embedding_route``, embeddings follow the chat route this preference
    selects, so a tool that skipped it resolved another profile than the
    agent (#3420).

    Persistence schema: ``{"vendor": str, "model": str, "route": str|None}``.
    Who wrote it is a separate ``model_preference_provenance`` row,
    ``{"mandate": {...}, "provenance": {...}}``; see :class:`OverrideProvenance`
    (#2652). A preference without a matching record is replayed as the agent's.
    Rows using the legacy ``{"model", "provider"}`` shape are dropped
    silently — the agent starts with no mandate and the user re-selects
    via the UI.

    ``agent_id`` ``None`` reads the single stored preference (a single-agent
    database) and applies nothing when several agents stored one.

    Returns ``None`` when the preference was applied or none is stored, and
    the reason when a stored preference could not be applied. That failure is
    also logged and recorded on the service for the health surface; the
    service is left unpinned, as the agent then runs.
    """
    try:
        if agent_id:
            result = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (agent_id, ModelPreferenceMixin.MODEL_PREFERENCE_KEY),
            )
        else:
            result = await db.fetchall(
                "SELECT value FROM agent_metadata WHERE key = ?",
                (ModelPreferenceMixin.MODEL_PREFERENCE_KEY,),
            )
        if not result or len(result) != 1:
            return None
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
                agent_id,
            )
            return None
        if model and model != "auto":
            replay_model_preference(
                llm_service,
                model,
                vendor,
                route,
                provenance=await _stored_provenance(
                    db, agent_id, {"vendor": vendor, "model": model, "route": route}
                ),
            )
            if vendor and route:
                logging.info("Loaded persisted model preference: %s:%s/%s", vendor, route, model)
            elif vendor:
                logging.info("Loaded persisted model preference: %s/%s", vendor, model)
            else:
                logging.info("Loaded persisted model preference: %s", model)
    except Exception as e:
        if is_execution_control_error(e):
            raise
        logging.warning(f"Failed to load model preference: {e}")
        # Record the drop for the health surface (#3190). A persisted pin
        # that fails to apply is an operator-visible degradation: the agent
        # then runs unpinned and routing falls through to route_priority,
        # which with allow_paid_fallback=false can land on a 1B local model.
        # A WARNING line in a multi-million-line host log is not a surface.
        try:
            llm_service._mandate_load_error = str(e)[:300]
        except Exception:  # pragma: no cover - never fail boot on reporting
            pass
        return str(e)
    return None
