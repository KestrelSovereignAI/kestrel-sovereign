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

Constitutional-injection invariants (kestrel-sovereign#1137 chunk 1D —
see `docs/architecture/CONSTITUTION_INJECTION.md` §"`SourceRegistration`
additions"):

- `prompt_template_format in {"codex", "local"}` requires
  `require_constitution_echo=True`. Those reviewer formats exist
  precisely to verify; opting out is contradictory.
- `prompt_template_format == "claude_code"` may set
  `require_constitution_echo` to either value; setting it to True
  without documenting the rationale in the source module's docstring
  emits a `UserWarning` (best-effort introspection — registration
  still succeeds).
- `prompt_template_format == "bare"` is caller-responsibility; the
  echo flag is unconstrained.
- `system_prompt_budget_bytes`, when set, must be a positive int.
  `None` falls back to the operator default at injection time.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Iterator, Optional

from kestrel_sdk.signals import (
    ResourceLock,
    SignalMode,
    SourceRegistration,
    Trust,
)

# Allowed values for the SDK-typed Literal fields. The SDK dataclass
# only declares these via `Literal[...]` annotations, which Python does
# NOT enforce at runtime — a typo like `"codxe"` would otherwise slip
# past every format-specific rule below and surface as an opaque
# dispatcher failure later. We allowlist explicitly here so registration
# is the single point that catches the error.
_VALID_PROMPT_TEMPLATE_FORMATS = frozenset(
    {"claude_code", "codex", "local", "bare"}
)
_VALID_CONSTITUTION_INJECTION = frozenset({"full", "none"})

# Reviewer formats that exist solely to verify constitution receipt.
# Echo opt-out is rejected for these.
_ECHO_REQUIRED_FORMATS = frozenset({"codex", "local"})

# Phrases we accept as evidence the source author documented why
# they're opting an in-agent (`claude_code`) source into the phantom-
# tool receipt path. Best-effort — silence the warning by writing
# anything substantive about constitutional echo into the module
# docstring.
_ECHO_RATIONALE_PHRASES = (
    "require_constitution_echo",
    "constitution_echo",
    "phantom tool",
    "constitution receipt",
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

        # Constitutional injection — kestrel-sovereign#1137 chunk 1D.
        # See CONSTITUTION_INJECTION.md §"`SourceRegistration` additions".
        SourceRegistry._validate_constitution_injection(reg)

    @staticmethod
    def _validate_constitution_injection(reg: SourceRegistration) -> None:
        fmt = reg.prompt_template_format

        # Allowlist enum-like fields BEFORE applying format-specific
        # rules. SDK Literal annotations are not runtime-enforced; a
        # typo like `prompt_template_format="codxe"` would otherwise
        # silently bypass the codex/local echo requirement (codex
        # round-1 P2 finding). Same for `constitution_injection`.
        if fmt not in _VALID_PROMPT_TEMPLATE_FORMATS:
            raise RegistrationError(
                f"Source '{reg.name}': prompt_template_format='{fmt}' is "
                "not a recognized format. Allowed: "
                f"{sorted(_VALID_PROMPT_TEMPLATE_FORMATS)}."
            )
        if reg.constitution_injection not in _VALID_CONSTITUTION_INJECTION:
            raise RegistrationError(
                f"Source '{reg.name}': constitution_injection="
                f"'{reg.constitution_injection}' is not a recognized "
                "value. Allowed: "
                f"{sorted(_VALID_CONSTITUTION_INJECTION)}."
            )

        # Hard error — reviewer formats may not opt out of echo.
        if fmt in _ECHO_REQUIRED_FORMATS and not reg.require_constitution_echo:
            raise RegistrationError(
                f"Source '{reg.name}': prompt_template_format='{fmt}' "
                "requires require_constitution_echo=True. The "
                f"'{fmt}' format is a non-in-agent reviewer path "
                "where echo verification is the entire point; "
                "opting out is contradictory."
            )

        # Hard error — reviewer formats with echo MUST also commit to
        # full constitutional injection (codex round-2 P2 finding).
        # Without `constitution_injection="full"` the dispatcher's
        # audit builder short-circuits, no constitution_hash is
        # resolved, no canary is derived, and every dispatch fails
        # with `constitution_not_received` — a configuration
        # mismatch the validator should catch.
        if (
            fmt in _ECHO_REQUIRED_FORMATS
            and reg.constitution_injection != "full"
        ):
            raise RegistrationError(
                f"Source '{reg.name}': prompt_template_format='{fmt}' "
                "with require_constitution_echo=True requires "
                "constitution_injection='full'. The reviewer formats "
                "exist to verify a constitution that was injected; "
                f"got constitution_injection='{reg.constitution_injection}'."
            )

        # Budget sanity — None means use operator default; if set,
        # must be a positive int. Type-check before the comparison
        # because untyped config (TOML / env / JSON) can hand us a
        # string `"8192"` or a float `1.5` that would otherwise either
        # raise an opaque TypeError on `<= 0` (str case) or silently
        # pass with the wrong type (float case). bool is rejected
        # explicitly because `True` is technically an int but
        # obviously a misuse here.
        budget = reg.system_prompt_budget_bytes
        if budget is not None:
            if isinstance(budget, bool) or not isinstance(budget, int):
                raise RegistrationError(
                    f"Source '{reg.name}': system_prompt_budget_bytes "
                    f"must be a positive int (or None), got "
                    f"{type(budget).__name__} {budget!r}."
                )
            if budget <= 0:
                raise RegistrationError(
                    f"Source '{reg.name}': system_prompt_budget_bytes "
                    f"must be > 0 when set, got {budget}."
                )

        # Soft warning — claude_code sources opting INTO echo should
        # document why in the module docstring. The legacy default
        # for in-agent COGNITION is echo=False; True is unusual and
        # auditors looking at this source should be able to read the
        # rationale without grepping commit history.
        if fmt == "claude_code" and reg.require_constitution_echo:
            if not _has_echo_rationale(reg):
                warnings.warn(
                    f"Source '{reg.name}': require_constitution_echo=True "
                    "on claude_code format; document the rationale in "
                    "the source module's docstring (mention "
                    "'require_constitution_echo' or 'phantom tool'). "
                    "Registration proceeds.",
                    UserWarning,
                    stacklevel=4,
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


def _has_echo_rationale(reg: SourceRegistration) -> bool:
    """Best-effort check that an opt-in `require_constitution_echo`
    has a documented rationale.

    Walks the registration's callable members (handler, artifact_handler,
    sanitizer) plus the `prompt_template` Path's containing repo dir to
    find a module-level docstring containing one of `_ECHO_RATIONALE_PHRASES`.
    Returns True if any candidate yields a match, False otherwise.

    Introspection is deliberately forgiving: returns True when no candidate
    callable can be located (e.g. closures, partials, C-extensions) so that
    sources whose source module is not loadable don't get spurious warnings.
    """
    candidates: list[object] = []
    for member in (reg.handler, reg.artifact_handler, reg.sanitizer):
        if member is not None:
            candidates.append(member)

    if not candidates:
        # Nothing to introspect — give the registration the benefit of
        # the doubt rather than firing a warning the author cannot act on.
        return True

    introspectable_found = False
    for candidate in candidates:
        module = inspect.getmodule(candidate)
        if module is None:
            continue
        introspectable_found = True
        doc = (module.__doc__ or "").lower()
        if any(phrase in doc for phrase in _ECHO_RATIONALE_PHRASES):
            return True

    # All candidates were callables we could not place in a module
    # (e.g. lambdas declared in REPL, dynamically generated closures).
    # Same forgiving stance as above.
    return not introspectable_found
