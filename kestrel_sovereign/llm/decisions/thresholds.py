"""Per-(caller, model, question) calibration thresholds (§2.5).

``[decisions.thresholds.<caller>]`` declares an ``uncalibrated`` policy, a
``default`` table, and per-model tables keyed ``"<vendor>:<route>/<model>"``.
A model is calibrated for a request only when its table covers every question
id in that request; thresholds from two calibrations are never mixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Collection, Mapping, Optional

from kestrel_sdk.llm.decisions import DecisionRequestInvalid


class UncalibratedPolicy(str, Enum):
    DEFAULT = "default"
    REFUSE = "refuse"


def _thresholds(raw: Any, where: str) -> Mapping[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a table of question id = threshold")
    values: dict[str, float] = {}
    for question_id, value in raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{where}.{question_id} must be a number in [0, 1]")
        values[str(question_id)] = float(value)
    return MappingProxyType(values)


@dataclass(frozen=True)
class CallerThresholds:
    policy: UncalibratedPolicy
    default: Mapping[str, float]
    models: Mapping[str, Mapping[str, float]]

    def calibrated_for(self, model_key: str, question_ids: Collection[str]) -> bool:
        table = self.models.get(model_key)
        return table is not None and all(q in table for q in question_ids)


@dataclass(frozen=True)
class ThresholdResolution:
    thresholds: Mapping[str, float]
    calibrated: Optional[bool]


_EMPTY: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True)
class ThresholdBook:
    """All callers' threshold tables, loaded from ``[decisions.thresholds]``."""

    callers: Mapping[str, CallerThresholds]

    @classmethod
    def from_config(cls, decisions_cfg: Mapping[str, Any]) -> "ThresholdBook":
        raw = decisions_cfg.get("thresholds", {}) if decisions_cfg else {}
        if not isinstance(raw, Mapping):
            raise ValueError("[decisions.thresholds] must be a table of callers")
        callers: dict[str, CallerThresholds] = {}
        for caller, table in raw.items():
            where = f"[decisions.thresholds.{caller}]"
            if not isinstance(table, Mapping):
                raise ValueError(f"{where} must be a table")
            try:
                policy = UncalibratedPolicy(table.get("uncalibrated", "default"))
            except ValueError as error:
                raise ValueError(
                    f'{where}.uncalibrated must be "default" or "refuse"'
                ) from error
            default = _thresholds(table.get("default", {}), f"{where}.default")
            raw_models = table.get("models", {})
            if not isinstance(raw_models, Mapping):
                raise ValueError(f"{where}.models must be a table")
            models = {
                str(key): _thresholds(value, f"{where}.models.{key!r}")
                for key, value in raw_models.items()
            }
            callers[str(caller)] = CallerThresholds(
                policy=policy, default=default, models=MappingProxyType(models)
            )
        return cls(callers=MappingProxyType(callers))

    def caller(self, caller: str) -> Optional[CallerThresholds]:
        return self.callers.get(caller)

    def _policy_and_defaults(
        self, caller: str, caller_defaults: Optional[Mapping[str, float]]
    ) -> tuple[Optional[UncalibratedPolicy], Mapping[str, float]]:
        """The effective policy and default table for one request.

        A caller may ship its own defaults (the threshold it uses before any
        calibration exists); the operator's ``default`` table overrides them
        key by key. ``None`` policy means the caller has neither: thresholds do
        not apply to it.
        """

        table = self.callers.get(caller)
        if table is None and caller_defaults is None:
            return None, _EMPTY
        defaults = dict(caller_defaults or {})
        if table is not None:
            defaults.update(table.default)
        policy = table.policy if table is not None else UncalibratedPolicy.DEFAULT
        return policy, MappingProxyType(defaults)

    def check_request(
        self,
        caller: str,
        keys: Collection[str],
        caller_defaults: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Refuse a request whose default thresholds cannot cover it.

        ``keys`` are the request's threshold keys (question ids, or the shared
        keys a caller maps them to). Under the ``default`` policy every key must
        have a default; a gap is a configuration error raised before routing.
        This is checked even when a calibrated model would answer: which model
        answers depends on routing (privacy mode, a route going down), so a gap
        must fail the same way on every route rather than only on the day an
        uncalibrated route is chosen (spec §2.5).
        """

        policy, defaults = self._policy_and_defaults(caller, caller_defaults)
        if policy is not UncalibratedPolicy.DEFAULT:
            return
        missing = sorted(k for k in set(keys) if k not in defaults)
        if missing:
            raise DecisionRequestInvalid(
                "thresholds",
                f"[decisions.thresholds.{caller}].default has no threshold for "
                f"key(s) {missing!r}; add them, or set "
                'uncalibrated = "refuse" to allow only calibrated models',
            )

    def admits(self, caller: str, model_key: str, keys: Collection[str]) -> bool:
        """Whether resolution may dispatch this request to ``model_key``."""

        table = self.callers.get(caller)
        if table is None or table.policy is not UncalibratedPolicy.REFUSE:
            return True
        return table.calibrated_for(model_key, set(keys))

    def resolve(
        self,
        caller: str,
        model_key: str,
        keys: Collection[str],
        caller_defaults: Optional[Mapping[str, float]] = None,
    ) -> ThresholdResolution:
        """Thresholds by key for the model that answers."""

        policy, defaults = self._policy_and_defaults(caller, caller_defaults)
        if policy is None:
            return ThresholdResolution(thresholds=_EMPTY, calibrated=None)
        unique = set(keys)
        table = self.callers.get(caller)
        if table is not None and table.calibrated_for(model_key, unique):
            source = table.models[model_key]
            calibrated = True
        else:
            # Only reachable under the "default" policy: "refuse" removed the
            # model during resolution, and check_request proved coverage.
            source = defaults
            calibrated = False
        return ThresholdResolution(
            thresholds=MappingProxyType({k: source[k] for k in unique}),
            calibrated=calibrated,
        )


def model_key(route_name: str, model: str) -> str:
    return f"{route_name}/{model}"
