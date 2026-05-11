"""Signal source registrations for FeatureFeature workflows.

The constitutional-review source deliberately sets
``require_constitution_echo=True`` because FeatureFeature changes the agent's
own executable surface; the workflow must verify that constitutional context
was present before allowing generated feature code to proceed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable

from kestrel_sdk.signals import (
    ActionHandler,
    RateLimit,
    RedactionPolicy,
    SignalMode,
    SourceRegistration,
    Trust,
)

from kestrel_sovereign.features.feature_features.workflows import (
    FEATURE_FEATURES_COMPENSATION_SOURCES,
    FEATURE_FEATURES_REVIEWER_SOURCES,
    FEATURE_FEATURES_STAGE_SOURCES,
)

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def build_feature_feature_registrations(agent: Any = None) -> list[SourceRegistration]:
    """Build SourceRegistrations used by the FeatureFeature workflow specs."""

    return [
        *_stage_registrations(agent),
        *_reviewer_registrations(),
        *[
            _action_registration(source, agent=agent)
            for source in FEATURE_FEATURES_COMPENSATION_SOURCES.values()
        ],
    ]


def _stage_registrations(agent: Any) -> list[SourceRegistration]:
    return [
        _action_registration(
            FEATURE_FEATURES_STAGE_SOURCES["explore"],
            agent=agent,
            fallback=_explore_handler(agent),
        ),
        _cognition_registration(
            FEATURE_FEATURES_STAGE_SOURCES["design_plan"],
            prompt_template=_PROMPT_DIR / "feature_feature_design.md",
            require_constitution_echo=False,
        ),
        _cognition_registration(
            FEATURE_FEATURES_STAGE_SOURCES["constitutional_review"],
            prompt_template=_PROMPT_DIR / "feature_feature_constitutional_review.md",
            require_constitution_echo=True,
        ),
        *[
            _action_registration(source, agent=agent)
            for key, source in FEATURE_FEATURES_STAGE_SOURCES.items()
            if key
            not in {
                "explore",
                "design_plan",
                "constitutional_review",
            }
        ],
    ]


def _reviewer_registrations() -> list[SourceRegistration]:
    return [
        _cognition_registration(
            source,
            prompt_template=_PROMPT_DIR / "feature_feature_red_team_review.md",
            require_constitution_echo=False,
        )
        for source in FEATURE_FEATURES_REVIEWER_SOURCES.values()
    ]


def _action_registration(
    source: str,
    *,
    agent: Any,
    fallback: ActionHandler | None = None,
    required: bool = True,
) -> SourceRegistration:
    return SourceRegistration(
        name=source,
        schema=_dict_payload_schema,
        default_mode=SignalMode.ACTION,
        allowed_modes=frozenset({SignalMode.ACTION}),
        handler=fallback or _agent_handler(source, agent=agent, required=required),
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(per_hour=120, burst=20),
        log_redaction=_REDACTION,
        result_summary=_result_summary,
        retention_days=30,
    )


def _cognition_registration(
    source: str,
    *,
    prompt_template: Path,
    require_constitution_echo: bool,
) -> SourceRegistration:
    return SourceRegistration(
        name=source,
        schema=_dict_payload_schema,
        default_mode=SignalMode.COGNITION,
        allowed_modes=frozenset({SignalMode.COGNITION}),
        prompt_template=prompt_template,
        trust=Trust.TRUSTED,
        rate_limit=RateLimit(per_hour=30, burst=4),
        log_redaction=_REDACTION,
        retention_days=30,
        require_constitution_echo=require_constitution_echo,
        constitution_injection="full" if require_constitution_echo else "none",
    )


def _dict_payload_schema(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("FeatureFeature signal payload must be an object")
    return dict(payload)


def _agent_handler(
    source: str,
    *,
    agent: Any,
    required: bool,
) -> ActionHandler:
    stage = source.split(".")[-1]

    async def handler(payload: dict) -> Any:
        provider = _resolve_provider(agent, stage)
        if provider is None:
            if required:
                raise RuntimeError(
                    f"FeatureFeature source {source!r} has no provider. "
                    f"Configure agent.feature_feature_{stage} before running."
                )
            return {"status": "not_configured", "source": source}
        result = provider(payload)
        if inspect.isawaitable(result):
            result = await result
        return result

    return handler


def _explore_handler(agent: Any) -> ActionHandler:
    from kestrel_sovereign.features.feature_features.feature import (
        _core_feature_inventory,
    )

    async def handler(payload: dict) -> dict[str, Any]:
        provider = _resolve_provider(agent, "explore")
        if provider is not None:
            result = provider(payload)
            if inspect.isawaitable(result):
                result = await result
            return result
        return {
            "status": "ok",
            "features": _core_feature_inventory(),
            "payload": dict(payload),
        }

    return handler


def _resolve_provider(agent: Any, stage: str) -> Callable[[dict], Awaitable[Any]] | None:
    for name in (
        f"feature_feature_{stage}",
        f"feature_feature_handle_{stage}",
    ):
        provider = getattr(agent, name, None)
        if callable(provider):
            return provider
    return None


def _result_summary(result_body: Any) -> str:
    if isinstance(result_body, dict):
        status = result_body.get("status")
        if isinstance(status, str):
            return status[:120]
    return ""


def _redact_payload(payload: dict) -> str:
    keys = ",".join(sorted(str(key) for key in payload))
    return f"keys={keys}" if keys else "<empty>"


_REDACTION = RedactionPolicy(
    summarize=_redact_payload,
    store_raw_trusted=False,
    redact_caller_identifier=True,
)


__all__ = ["build_feature_feature_registrations"]
