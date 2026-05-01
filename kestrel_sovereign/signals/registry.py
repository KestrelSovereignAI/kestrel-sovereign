"""Source Registry — the v1 boundary.

Sources cannot dispatch without registering. The registry validates every
registration against the constraints in SIGNAL_DISPATCHER.md
(§"The Source Registry"); a partially-specified registration is rejected
at register time, not at dispatch time.

Validators enforced here:
- `default_mode` must be in `allowed_modes`
- ACTION in allowed_modes → `handler` required
- ARTIFACT in allowed_modes → `artifact_handler` required
- COGNITION in allowed_modes → `prompt_template` required
- UNTRUSTED + any non-ACTION mode allowed → `sanitizer` required
- COGNITION sources MUST NOT declare `CONVERSATION` in `resources`
  (the turn lifecycle is the sole owner — see Concern #1)
- `log_redaction` is required (no defaults; this is too important to default)
"""

from __future__ import annotations

from typing import Iterator, Optional

from kestrel_sdk.signals import (
    ResourceLock,
    SignalMode,
    SourceRegistration,
    Trust,
)


class RegistrationError(ValueError):
    """Raised when a `SourceRegistration` violates a v1 invariant.

    Caught only at startup / source-add time. At dispatch time, an unknown
    source name produces `Status.DROPPED_VALIDATION` instead.
    """


class SourceRegistry:
    """In-memory registry of signal sources keyed by `source.name`.

    Single instance per dispatcher. Registration is one-shot: re-registering
    the same name raises (mutating the contract behind a running dispatcher
    is the road back to the accretion mess this fixes).
    """

    def __init__(self) -> None:
        self._sources: dict[str, SourceRegistration] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, registration: SourceRegistration) -> None:
        self._validate(registration)
        if registration.name in self._sources:
            raise RegistrationError(
                f"Source '{registration.name}' is already registered. "
                "Re-registration is not supported; restart the process to change."
            )
        self._sources[registration.name] = registration

    @staticmethod
    def _validate(reg: SourceRegistration) -> None:
        # Identity sanity
        if not reg.name or not reg.name.strip():
            raise RegistrationError("Source registration requires a non-empty name.")

        # Mode coverage
        if not reg.allowed_modes:
            raise RegistrationError(
                f"Source '{reg.name}' must allow at least one mode."
            )
        if reg.default_mode not in reg.allowed_modes:
            raise RegistrationError(
                f"Source '{reg.name}': default_mode {reg.default_mode.value} "
                f"is not in allowed_modes {{{', '.join(m.value for m in reg.allowed_modes)}}}."
            )

        # Mode-specific contracts
        if SignalMode.ACTION in reg.allowed_modes and reg.handler is None:
            raise RegistrationError(
                f"Source '{reg.name}' allows ACTION but provides no handler."
            )
        if SignalMode.ARTIFACT in reg.allowed_modes and reg.artifact_handler is None:
            raise RegistrationError(
                f"Source '{reg.name}' allows ARTIFACT but provides no artifact_handler."
            )
        if SignalMode.COGNITION in reg.allowed_modes and reg.prompt_template is None:
            raise RegistrationError(
                f"Source '{reg.name}' allows COGNITION but provides no prompt_template."
            )

        # Trust + sanitizer
        non_action_modes = reg.allowed_modes - {SignalMode.ACTION}
        if reg.trust == Trust.UNTRUSTED and non_action_modes and reg.sanitizer is None:
            raise RegistrationError(
                f"Source '{reg.name}' is UNTRUSTED with non-ACTION modes "
                f"{{{', '.join(m.value for m in non_action_modes)}}} "
                "but provides no sanitizer. Untrusted payloads must be "
                "sanitized before reaching cognition or artifact handlers."
            )

        # CONVERSATION ownership — see SIGNAL_DISPATCHER.md §Concern 1, 2
        if ResourceLock.CONVERSATION in reg.resources:
            raise RegistrationError(
                f"Source '{reg.name}' declares CONVERSATION in resources. "
                "The turn lifecycle is the sole owner of CONVERSATION; "
                "sources must not pre-acquire it. Remove it from resources."
            )

        # Privacy
        if reg.log_redaction is None:
            raise RegistrationError(
                f"Source '{reg.name}' has no log_redaction policy. "
                "Every source must declare one explicitly — no defaults. "
                "If the source emits no payload data worth redacting, "
                "supply a policy whose summarize() returns an empty string."
            )

        # Retention
        if reg.retention_days < 0:
            raise RegistrationError(
                f"Source '{reg.name}': retention_days must be >= 0, "
                f"got {reg.retention_days}."
            )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[SourceRegistration]:
        return self._sources.get(name)

    def require(self, name: str) -> SourceRegistration:
        """Lookup that raises `RegistrationError` instead of returning None.

        Used inside the dispatcher's validation step where unknown source
        is a `DROPPED_VALIDATION` outcome, not a registration error.
        Callers that want the dropped-validation path should use `get`
        and check for None.
        """
        reg = self._sources.get(name)
        if reg is None:
            raise RegistrationError(f"Unknown source: '{name}'")
        return reg

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def __iter__(self) -> Iterator[SourceRegistration]:
        return iter(self._sources.values())

    def __len__(self) -> int:
        return len(self._sources)
