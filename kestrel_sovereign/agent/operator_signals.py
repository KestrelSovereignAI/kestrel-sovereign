"""Operator-signal producer seam for turn-time LLM delivery.

Operator facts are trusted runtime state, not user content. This module
centralizes when those facts become model-visible and whether they travel as
native inline system turns or as an explicit user-visible fallback.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SOURCE_AUTO_MODE = "operator.auto_mode"
SOURCE_TOKEN_BUDGET = "operator.token_budget"
SOURCE_GOVERNANCE_DELTA = "operator.governance_delta"


class OperatorSignalProducer:
    """Build trusted operator facts for the next model request.

    The producer is deliberately stateful and append-only: once it emits a
    notice for a turn, the caller appends that exact text to conversation
    history with ``metadata.sent_form=True`` so replay is byte-stable.
    """

    def __init__(self, agent: Any):
        self._agent = agent
        self._pending_auto_mode: List[bool] = []
        self._last_budget_low_by_session: Dict[str, bool] = {}
        self._last_governance_fingerprint_by_session: Dict[str, str] = {}

    def enqueue_auto_mode(self, enabled: bool) -> None:
        self._pending_auto_mode.append(bool(enabled))

    async def collect_for_turn(
        self,
        *,
        session_id: Optional[str],
        llm_service: Any,
        model_override: Optional[str],
        force_local_only: bool,
        budget_summary: Optional[Dict[str, Any]],
        state_of_mind: Any,
    ) -> "OperatorSignalBatch":
        use_inline, route_label = supports_inline_system_for_next_route(
            llm_service,
            model_override=model_override,
            force_local_only=force_local_only,
        )
        delivery_role = "system" if use_inline else "user"

        events: List[OperatorSignalEvent] = []
        for enabled in self._pending_auto_mode:
            events.append(
                OperatorSignalEvent(
                    source=SOURCE_AUTO_MODE,
                    content=_auto_mode_notice(enabled),
                    payload={"enabled": enabled},
                )
            )
        self._pending_auto_mode.clear()

        budget_event = _budget_event(
            budget_summary,
            session_key=session_id or "__global__",
            low_state=self._last_budget_low_by_session,
        )
        if budget_event is not None:
            events.append(budget_event)

        governance_event = _governance_delta_event(
            state_of_mind,
            session_key=session_id or "__global__",
            last_fingerprints=self._last_governance_fingerprint_by_session,
        )
        if governance_event is not None:
            events.append(governance_event)

        if not events:
            return OperatorSignalBatch.empty()

        if use_inline:
            content = "\n\n".join(event.content for event in events)
        else:
            content = _fallback_user_notice(
                "\n\n".join(event.content for event in events)
            )
            logger.info(
                "Operator signal fallback: route=%s does not support inline "
                "system messages; delivering %d event(s) as user-visible "
                "operator notice.",
                route_label,
                len(events),
            )

        batch = OperatorSignalBatch(
            role=delivery_role,
            content=content,
            keep_trailing_system=use_inline,
            events=events,
            route_label=route_label,
            fallback=not use_inline,
        )
        await self._audit(batch, session_id=session_id)
        return batch

    async def _audit(
        self,
        batch: "OperatorSignalBatch",
        *,
        session_id: Optional[str],
    ) -> None:
        store = getattr(self._agent, "signal_log_store", None)
        if store is None:
            logger.info(
                "Operator signal batch delivered without signal_log store "
                "(role=%s, events=%s)",
                batch.role,
                [event.source for event in batch.events],
            )
            return

        try:
            from kestrel_sdk.signals import (
                RedactionPolicy,
                Signal,
                SignalMode,
                SignalResult,
                SourceRegistration,
                Status,
                Trust,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Operator signal audit unavailable: %s", exc)
            return

        async def _noop(_payload: dict) -> None:
            return None

        for event in batch.events:
            registration = SourceRegistration(
                name=event.source,
                schema=dict,
                default_mode=SignalMode.ACTION,
                allowed_modes=frozenset({SignalMode.ACTION}),
                handler=_noop,
                trust=Trust.TRUSTED,
                log_redaction=RedactionPolicy(
                    summarize=lambda p, source=event.source: (
                        f"{source} delivery={p.get('delivery_role')} "
                        f"fallback={p.get('fallback')}"
                    ),
                    store_raw_trusted=True,
                    redact_caller_identifier=True,
                ),
                retention_days=14,
            )
            payload = {
                **event.payload,
                "delivery_role": batch.role,
                "fallback": batch.fallback,
                "route": batch.route_label,
            }
            signal = Signal(
                source=event.source,
                kind="operator_fact",
                mode=SignalMode.ACTION,
                payload=payload,
                target_agent=getattr(self._agent, "did", "unknown"),
                session_id=session_id,
            )
            result = SignalResult(
                signal_id=signal.id,
                status=Status.OK,
                mode=SignalMode.ACTION,
                duration_ms=0,
                action_result={
                    "delivered": True,
                    "delivery_role": batch.role,
                    "fallback": batch.fallback,
                },
            )
            try:
                await store.append(signal, registration, result)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to write operator signal audit for %s: %s",
                    event.source,
                    exc,
                    exc_info=True,
                )


class OperatorSignalEvent:
    def __init__(self, *, source: str, content: str, payload: Dict[str, Any]):
        self.source = source
        self.content = content
        self.payload = payload


class OperatorSignalBatch:
    def __init__(
        self,
        *,
        role: str,
        content: str,
        keep_trailing_system: bool,
        events: List[OperatorSignalEvent],
        route_label: str,
        fallback: bool,
    ):
        self.role = role
        self.content = content
        self.keep_trailing_system = keep_trailing_system
        self.events = events
        self.route_label = route_label
        self.fallback = fallback

    @classmethod
    def empty(cls) -> "OperatorSignalBatch":
        return cls(
            role="",
            content="",
            keep_trailing_system=False,
            events=[],
            route_label="unknown",
            fallback=False,
        )

    @property
    def has_events(self) -> bool:
        return bool(self.events)


def supports_inline_system_for_next_route(
    llm_service: Any,
    *,
    model_override: Optional[str],
    force_local_only: bool,
) -> Tuple[bool, str]:
    """Centralized capability decision for operator-signal delivery."""
    provider: Optional[Dict[str, Any]] = None
    target_model: Optional[str] = None
    try:
        if hasattr(llm_service, "resolve_provider_routing"):
            providers, target_model = llm_service.resolve_provider_routing(
                model_override=model_override,
                force_local_only=force_local_only,
            )
        else:
            providers = getattr(llm_service, "_available_providers", lambda: [])()
        provider = providers[0] if providers else None
    except Exception as exc:  # noqa: BLE001
        logger.info("Operator signal route capability probe failed: %s", exc)
        provider = None

    if not provider:
        return False, "unknown"

    model = target_model or provider.get("model") or ""
    route_label = provider.get("name") or provider.get("vendor") or "unknown"
    caps = provider.get("capabilities") or {}
    route_supports = bool(caps.get("supports_inline_system"))
    adapter = provider.get("adapter")
    model_supports = True
    checker = getattr(adapter, "_model_supports_inline_system", None)
    if callable(checker):
        try:
            model_supports = bool(checker(model))
        except Exception:  # noqa: BLE001
            model_supports = False
    return route_supports and model_supports, f"{route_label}/{model or 'auto'}"


def state_of_mind_snapshot(state: Any) -> Dict[str, Any]:
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    if is_dataclass(state):
        raw = asdict(state)
    else:
        raw = {
            "provider": getattr(state, "provider", None),
            "model": getattr(state, "model", None),
            "governance_mode": getattr(state, "governance_mode", None),
            "transparency": getattr(state, "transparency", None),
            "delegated_principles": getattr(state, "delegated_principles", None),
            "active_conflicts": getattr(state, "active_conflicts", None),
            "complements": getattr(state, "complements", None),
        }
    raw.pop("prompt_adaptation", None)
    return raw


def _auto_mode_notice(enabled: bool) -> str:
    if enabled:
        return (
            "Operator context: auto-mode is now enabled for this server "
            "session. The operator has given standing consent for "
            "multi-agent workflows and non-denied tool calls to proceed "
            "without additional approval prompts when earlier constitutional, "
            "honesty, and security checks do not flag the action."
        )
    return (
        "Operator context: auto-mode is now disabled for this server session. "
        "Standing consent for unprompted tool use has ended; approval prompts "
        "apply according to the configured permission policy."
    )


def _budget_event(
    budget_summary: Optional[Dict[str, Any]],
    *,
    session_key: str,
    low_state: Dict[str, bool],
) -> Optional[OperatorSignalEvent]:
    if not budget_summary:
        return None
    try:
        total = int(budget_summary.get("total_budget") or 0)
        used = int(budget_summary.get("total_used") or 0)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    remaining = max(0, total - used)
    threshold = _budget_threshold(total)
    is_low = remaining <= threshold
    was_low = low_state.get(session_key, False)
    low_state[session_key] = is_low
    if not is_low or was_low:
        return None
    return OperatorSignalEvent(
        source=SOURCE_TOKEN_BUDGET,
        content=(
            "Operator context: remaining token budget for this turn is now "
            f"{remaining} tokens out of {total}. The low-budget threshold is "
            f"{threshold} tokens."
        ),
        payload={
            "remaining_tokens": remaining,
            "total_budget": total,
            "threshold": threshold,
        },
    )


def _budget_threshold(total_budget: int) -> int:
    return max(2048, int(total_budget * 0.10))


def _governance_delta_event(
    state: Any,
    *,
    session_key: str,
    last_fingerprints: Dict[str, str],
) -> Optional[OperatorSignalEvent]:
    snapshot = state_of_mind_snapshot(state)
    if not snapshot:
        return None
    mutable = {
        "provider": snapshot.get("provider"),
        "model": snapshot.get("model"),
        "governance_mode": snapshot.get("governance_mode"),
        "transparency": snapshot.get("transparency"),
        "delegated_principles": snapshot.get("delegated_principles") or [],
        "active_conflicts": snapshot.get("active_conflicts") or [],
    }
    fingerprint = repr(mutable)
    if last_fingerprints.get(session_key) == fingerprint:
        return None
    last_fingerprints[session_key] = fingerprint
    return OperatorSignalEvent(
        source=SOURCE_GOVERNANCE_DELTA,
        content=_format_governance_notice(mutable),
        payload=mutable,
    )


def _format_governance_notice(state: Dict[str, Any]) -> str:
    lines = [
        "Operator context: constitutional governance state has changed.",
        (
            f"Current model route: {state.get('provider') or 'unknown'} / "
            f"{state.get('model') or 'unknown'}."
        ),
        (
            "Governance mode: "
            f"{str(state.get('governance_mode') or 'unknown').upper()}."
        ),
        f"Transparency: {state.get('transparency') or 'unknown'}.",
    ]
    delegated = state.get("delegated_principles") or []
    if delegated:
        lines.append(
            "Delegated principles: "
            + ", ".join(str(p) for p in delegated)
            + "."
        )
    else:
        lines.append("Delegated principles: none.")
    conflicts = state.get("active_conflicts") or []
    if conflicts:
        rendered = []
        for conflict in conflicts:
            if isinstance(conflict, dict):
                principle = conflict.get("principle", "unknown")
                severity = conflict.get("severity", "unknown")
                description = conflict.get("description", "")
                rendered.append(
                    f"{principle} ({severity})"
                    + (f": {description}" if description else "")
                )
            else:
                rendered.append(str(conflict))
        lines.append("Active conflicts: " + "; ".join(rendered) + ".")
    else:
        lines.append("Active conflicts: none.")
    lines.append(f"Observed at: {datetime.now(timezone.utc).isoformat()}.")
    return "\n".join(lines)


def _fallback_user_notice(content: str) -> str:
    return (
        "<operator_notice>\n"
        "The following trusted operator context is being relayed as a "
        "user-visible notice because the selected route/model does not "
        "support inline mid-conversation system messages.\n\n"
        f"{content}\n"
        "</operator_notice>"
    )
