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

import contextlib
import enum
import functools
import inspect
import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from kestrel_sdk.signals import (
    ResourceLock,
    SignalMode,
    SourceRegistration,
    Trust,
)

logger = logging.getLogger(__name__)

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


#: A feature holds sources in two independent roles: ones it registered itself,
#: and ones its declared contributions activated. They tear down by different
#: paths and either can fail alone, so they are separate claims (issue #3053).
CLAIM_IMPERATIVE = "imperative"
CLAIM_CONTRIBUTION = "contribution"

#: Marks the host as a holder of a source (issue #3053). The host outlives every
#: feature, so this claim is never released — only `unregister` clears it.
_HOST_CLAIM = "host"

#: The only two roles a claim can be held in. A third value is a claim nothing
#: releases: `Feature.shutdown()` releases CLAIM_IMPERATIVE and the contribution
#: runtime releases CLAIM_CONTRIBUTION, and neither would recognise it.
_CLAIM_ROLES = (CLAIM_IMPERATIVE, CLAIM_CONTRIBUTION)


def _claim_role_for(owner, role):
    """The role a claim is recorded under — validated, never defaulted.

    A HOST claim (``owner is None``) has no role: it is one sentinel that
    `release` can never drop, so nothing reads a role for it.

    An OWNED claim must name one of :data:`_CLAIM_ROLES`, because the two are
    released by different code paths. A value neither path knows retains the
    source and the owner object forever, and an empty string quietly becoming
    the contribution role is exactly the silent downgrade that requiring a
    stated role exists to stop — so membership is checked, not truthiness.
    """
    if owner is None:
        return CLAIM_CONTRIBUTION  # unused: the host claim is keyed by sentinel
    if role not in _CLAIM_ROLES:
        raise TypeError(
            "an owned registration must state its claim role as one of "
            f"{_CLAIM_ROLES}; got {role!r}"
        )
    return role


class RegistrationPolicy(enum.Enum):
    """Explicit policy for how a source registration resolves a name clash.

    The bare :meth:`SourceRegistry.register` is one-shot and raises on ANY
    duplicate name — safe for the single core-boot path where every name is
    registered exactly once, but too blunt for the feature/boot paths that
    historically hand-rolled precheck-by-name skips, catch-and-continue, or
    partial-set loops (kestrel-sovereign#2522). Those paths silently equated a
    same-name source that in fact carried a *different* trust, mode set,
    redaction, handler, or ownership. This enum makes the intended behavior a
    declared policy instead of an accident of the surrounding try/except:

    * ``MANDATORY`` — a name clash with a non-equivalent contract is a hard
      :class:`RegistrationError`; an equivalent re-registration is a no-op
      success. Boot-critical sources use this. In a batch it is atomic
      (see :meth:`SourceRegistry.register_batch`).
    * ``OPTIONAL`` — never raises. A validation failure, or a clash with a
      non-equivalent contract, is *reported* (a ``MISMATCH`` / ``INVALID``
      outcome, logged loudly) and the existing registration is kept. Feature
      sources whose absence is a degraded-but-tolerable state use this.
    * ``IDEMPOTENT`` — an equivalent re-registration is a no-op success; a
      clash with a *different* contract raises. Same failure contract as
      ``MANDATORY`` but names the intent ("register once; re-runs are only
      valid if identical").
    """

    MANDATORY = "mandatory"
    OPTIONAL = "optional"
    IDEMPOTENT = "idempotent"


class RegistrationState(enum.Enum):
    """Outcome state of a single policy-driven registration."""

    REGISTERED = "registered"           # newly added
    ALREADY_EQUIVALENT = "already_equivalent"  # present with an equivalent contract
    MISMATCH = "mismatch"               # present with a DIFFERENT contract (reported)
    INVALID = "invalid"                 # failed validation (reported, OPTIONAL only)


@dataclass(frozen=True)
class RegistrationOutcome:
    """Structured result of a policy-driven registration (never a silent skip)."""

    name: str
    state: RegistrationState
    detail: str = ""

    @property
    def ok(self) -> bool:
        """True when the source is present with the intended contract."""
        return self.state in (
            RegistrationState.REGISTERED,
            RegistrationState.ALREADY_EQUIVALENT,
        )


class SourceRegistry:
    """In-memory registry of signal sources keyed by `source.name`.

    Single instance per dispatcher. Registration is one-shot: re-registering
    the same name raises (mutating the contract behind a running dispatcher
    is the road back to the accretion mess this fixes).
    """

    def __init__(self) -> None:
        self._sources: dict[str, SourceRegistration] = {}
        # Who holds each source, in registration order — THE ownership record
        # (issue #3053).
        #
        # Ownership used to be kept by the callers instead, in two places that
        # could not see each other: `Feature._owned_signal_source_names` for
        # sources a feature registers itself, and
        # `ActiveFeatureContributions.registered_sources` for ones it declares.
        # Both applied the same rule — only a newly-created source is yours —
        # and neither knew the other existed, so a feature that did both could
        # tear down a source another feature was still dispatching against.
        #
        # It belongs here: "who registered this source" is a fact about the
        # source, and this is what holds the sources. A claim is released by its
        # holder; the source itself goes when the last claim does.
        self._claims: dict[str, list] = {}
        self._claim_owners: dict[int, object] = {}
        #: Active `claims_acquired` scopes, innermost last.
        self._acquisition_logs: list = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, registration: SourceRegistration, *, owner=None) -> None:
        self._validate(registration)
        if registration.name in self._sources:
            raise RegistrationError(
                f"Source '{registration.name}' is already registered. "
                "Re-registration is not supported; restart the process to change."
            )
        self._sources[registration.name] = registration
        self._claim(registration.name, owner)

    # ------------------------------------------------------------------
    # Ownership (issue #3053)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def claims_acquired(self, owner):
        """Record exactly the claims ACQUIRED inside the block.

        Callers with a rollback path kept deriving "what did I just take?" by
        hand, and three separate sites got it wrong three different ways: a host
        claim retained so a feature could never release its source; a claim that
        PREDATED the operation released, deleting a live source; and owners
        compared by equality, so a distinct-but-equal instance's claim went
        untracked.

        Only the registry knows whether a claim was actually added, so it
        reports it. Yields the list of names acquired; unwind with
        :meth:`release_acquired` (issue #3053).
        """
        acquired: list = []
        self._acquisition_logs.append(acquired)
        try:
            yield acquired
        finally:
            self._acquisition_logs.pop()

    def release_acquired(
        self, acquired, owner, role: str = CLAIM_CONTRIBUTION
    ) -> None:
        """Release exactly the claims recorded by :meth:`claims_acquired`."""
        key = _HOST_CLAIM if owner is None else (id(owner), role)
        for name in acquired:
            holders = self._claims.get(name)
            if not holders or key not in holders:
                continue
            holders.remove(key)
            self._forget_owner_if_unreferenced(key)
            if not holders:
                self._claims.pop(name, None)
                self._sources.pop(name, None)

    def _record_acquisition(self, name: str) -> None:
        # EVERY active scope, not just the innermost: `register_batch` opens
        # its own scope inside a caller's, and an acquisition made there
        # happened inside both. Recording only the innermost left the outer
        # rollback believing it had taken nothing.
        for log in self._acquisition_logs:
            log.append(name)

    def _claim(self, name: str, owner, role: str = CLAIM_CONTRIBUTION) -> None:
        """Record *owner* as a holder of *name*. ``None`` means the host.

        The host IS a holder — it just never lets go. Recording nothing for it
        made a feature that claimed an equivalent host source the sole holder,
        so that feature's teardown deleted the host's own source. Explicit
        sentinel: `release` can never drop it, and `unregister` (the boot
        rollback's unconditional removal) clears it outright.
        """
        holders = self._claims.setdefault(name, [])
        if owner is None:
            if _HOST_CLAIM not in holders:
                holders.append(_HOST_CLAIM)
                self._record_acquisition(name)
            return
        key = (id(owner), role)
        if key not in holders:
            holders.append(key)
            self._claim_owners[key] = owner
            self._record_acquisition(name)

    def _forget_owner_if_unreferenced(self, key) -> None:
        """Drop the owner object once no claim list mentions it.

        `_claim_owners` holds STRONG references so `owners_of` can hand back the
        objects; leaving one behind after its last claim went would retain a
        feature instance (and everything it holds) for the registry's lifetime.
        """
        if key is _HOST_CLAIM:
            return
        if not any(key in holders for holders in self._claims.values()):
            self._claim_owners.pop(key, None)

    def owners_of(self, name: str) -> tuple:
        """The objects currently holding *name*, in the order they claimed it."""
        seen: list = []
        for key in self._claims.get(name, ()):
            if key is _HOST_CLAIM or key not in self._claim_owners:
                continue
            owner = self._claim_owners[key]
            if not any(owner is existing for existing in seen):
                seen.append(owner)
        return tuple(seen)

    def release(self, name: str, owner, role: str = CLAIM_CONTRIBUTION) -> bool:
        """Drop *owner*'s claim on *name*; remove the source with the last one.

        Returns True when the source itself was removed. This is what replaces
        both private ledgers: a holder releases what it holds, and a source
        another holder still needs simply does not go away — no transfer, no
        reference count kept somewhere else, no second list to agree with.
        """
        holders = self._claims.get(name)
        key = (id(owner), role)
        if not holders or key not in holders:
            return False
        holders.remove(key)
        self._forget_owner_if_unreferenced(key)
        if holders:
            return False
        self._claims.pop(name, None)
        return self._sources.pop(name, None) is not None

    def release_all(self, owner, role: str = CLAIM_CONTRIBUTION) -> tuple:
        """Release every claim *owner* holds IN THIS ROLE.

        A feature is two independent dependents: what it registered itself
        (``CLAIM_IMPERATIVE``) and what its declared contributions activated
        (``CLAIM_CONTRIBUTION``). They are torn down by different code paths and
        either can fail on its own — `_unregister_feature_runtime` deliberately
        continues to `shutdown()` after a rejected `deactivate()` — so releasing
        both together dropped a still-active contribution's claim and could take
        its source with it (issue #3053).
        """
        removed = []
        key = (id(owner), role)
        for name in [n for n, h in self._claims.items() if key in h]:
            if self.release(name, owner, role):
                removed.append(name)
        return tuple(removed)

    def unregister(self, name: str) -> bool:
        """Remove a source by name. Returns True if one was present.

        Used by the agent boot state machine to roll a partial signal-source
        set back in reverse order when a later boot phase fails
        (kestrel-sovereign#2522). Registration is otherwise one-shot; this is
        the deliberate inverse for the teardown path, not a mutate-behind-a-
        running-dispatcher tool.
        """
        for key in self._claims.pop(name, ()):
            self._forget_owner_if_unreferenced(key)
        return self._sources.pop(name, None) is not None

    def register_with_policy(
        self,
        registration: SourceRegistration,
        policy: RegistrationPolicy = RegistrationPolicy.MANDATORY,
        *,
        owner=None,
        role: Optional[str] = None,
    ) -> RegistrationOutcome:
        """Register a source under an explicit name-clash :class:`RegistrationPolicy`.

        Returns a :class:`RegistrationOutcome`. A same-name source is only
        accepted when it is *contract-equivalent* to the existing one (see
        :meth:`contract_signature`); a differing trust/mode/redaction/handler/
        ownership is a ``MISMATCH`` — reported (and raised for MANDATORY /
        IDEMPOTENT) rather than silently equated.

        ``owner=None`` means the HOST — a permanent holder — and nothing else;
        the policy has no say in who is claiming (#3074). An owner MUST name
        its ``role``, because the two roles are torn down by different code
        paths: a default here would hand an imperative registration a
        contribution claim that ``shutdown()`` never releases, which is the
        silent-downgrade shape a default is always at risk of.
        """
        role = _claim_role_for(owner, role)
        try:
            self._validate(registration)
        except RegistrationError as exc:
            if policy is RegistrationPolicy.OPTIONAL:
                logger.warning(
                    "signal source '%s' failed validation under OPTIONAL "
                    "policy; not registered: %s",
                    registration.name,
                    exc,
                )
                return RegistrationOutcome(
                    registration.name, RegistrationState.INVALID, str(exc)
                )
            raise

        existing = self._sources.get(registration.name)
        if existing is None:
            self._sources[registration.name] = registration
            self._claim(registration.name, owner, role)
            return RegistrationOutcome(registration.name, RegistrationState.REGISTERED)

        if self.contract_equivalent(existing, registration):
            # An equivalent re-registration is a no-op for the SOURCE and a real
            # claim for the CALLER: it now depends on this source and must keep
            # it alive until it lets go. Recording that is what stops the first
            # holder's teardown pulling the source out from under the second.
            #
            # Unconditional, because ``owner`` now says who is claiming and
            # nothing has to be inferred from the policy. It used to: an
            # ownerless call was ambiguous — core saying "I need this" or the
            # imperative feature path, which registered ownerless and claimed a
            # moment later — and the policy split happened to separate them.
            # Three review rounds each found a different edge of that guess
            # (issue #3074). Features state their ownership at registration
            # (:meth:`KestrelFeature._register_signal_sources`), so ownerless
            # means the host and only the host.
            self._claim(registration.name, owner, role)
            return RegistrationOutcome(
                registration.name, RegistrationState.ALREADY_EQUIVALENT
            )

        detail = (
            f"Source '{registration.name}' is already registered with a "
            "DIFFERENT contract (trust/modes/redaction/handler/ownership); "
            "refusing to silently treat them as equivalent."
        )
        if policy is RegistrationPolicy.OPTIONAL:
            logger.error(
                "%s Keeping the existing registration; the new one is dropped "
                "(existing=%s incoming=%s).",
                detail,
                self.contract_signature(existing),
                self.contract_signature(registration),
            )
            return RegistrationOutcome(
                registration.name, RegistrationState.MISMATCH, detail
            )
        raise RegistrationError(detail)

    def register_batch(
        self,
        registrations: Iterable[SourceRegistration],
        policy: RegistrationPolicy = RegistrationPolicy.MANDATORY,
        *,
        owner=None,
        role: Optional[str] = None,
    ) -> list[RegistrationOutcome]:
        """Register several sources under one policy.

        Under ``MANDATORY``/``IDEMPOTENT`` the batch is **atomic**: if any
        registration raises, every source newly added *by this batch* is
        removed before the error propagates, so a mid-sequence failure never
        leaves a partial source set behind (kestrel-sovereign#2522 partial-set
        defect). Under ``OPTIONAL`` each source is independent and reported.
        """
        # Validated HERE as well as per-registration, because the rollback below
        # has to release under the same role the batch claimed under — reading
        # it off a default there left a failed owner attached to every incumbent
        # the batch had ridden.
        role = _claim_role_for(owner, role)
        outcomes: list[RegistrationOutcome] = []
        newly_added: list[str] = []
        # Atomic means the CLAIMS unwind too: exactly the ones this batch took,
        # host claims included, and nothing that predated it.
        with self.claims_acquired(owner) as acquired:
            try:
                for registration in registrations:
                    outcome = self.register_with_policy(
                        registration, policy, owner=owner, role=role
                    )
                    outcomes.append(outcome)
                    if outcome.state is RegistrationState.REGISTERED:
                        newly_added.append(outcome.name)
            except RegistrationError:
                self.release_acquired(acquired, owner, role)
                for name in newly_added:
                    self._sources.pop(name, None)
                    self._claims.pop(name, None)
                raise
        return outcomes

    @staticmethod
    def contract_signature(reg: SourceRegistration) -> tuple:
        """A hashable fingerprint of *every* behavior-affecting field of a source.

        Two registrations are *contract-equivalent* iff their signatures are
        equal. The fingerprint deliberately covers the whole dispatch contract,
        not a subset (kestrel-sovereign#2522 P1): identity + schema, the mode
        set + default, trust, the handler/artifact/sanitizer/result callables,
        throttling (rate limit + coalescing window), attention policy, resource
        ownership + self-loop policy, the redaction policy's *flags and
        summarizer* (not merely its class), retention, the four
        constitutional-injection fields, and the per-signal prompt-override
        opt-in (``allow_prompt_override``). A re-registration that changes any
        of them is therefore caught as a MISMATCH instead of being silently
        accepted as equivalent.

        ``allow_prompt_override`` is validated at registration time (only a
        ``bool`` is accepted) yet governs a real dispatch decision — whether a
        signal's ``prompt_template_override`` is honored — so two otherwise
        identical registrations that differ only in that flag are a genuine
        contract mismatch and must not compare equivalent (#2522 P1).

        Callables are fingerprinted by :func:`_callable_identity`, which folds
        in a bound method's owner and a closure's *captured free variables* by
        object identity. That is what distinguishes two
        ``build_*_registration(coordinator)`` handlers that share a qualname but
        capture *different* coordinators (or two ``fleet_stalled_sweep``
        registrations bound to different discovery callbacks): a genuine
        same-owner re-init compares equivalent (the same captured instance), a
        handler bound to a new owner/dependency is a mismatch — the exact
        false-equivalence the signal-duplication audit found. Immutable scalars
        captured by a closure (e.g. a task name) still compare by value, so a
        rebuilt-but-identical registration is not spuriously flagged.
        """
        rl = reg.rate_limit
        ap = reg.attention_policy
        red = reg.log_redaction
        return (
            reg.name,
            _callable_identity(reg.schema),
            reg.default_mode,
            frozenset(reg.allowed_modes),
            _callable_identity(reg.handler),
            _callable_identity(reg.artifact_handler),
            str(reg.prompt_template) if reg.prompt_template is not None else None,
            reg.trust,
            _callable_identity(reg.sanitizer),
            (rl.per_minute, rl.per_hour, rl.burst) if rl is not None else None,
            reg.coalescing_window,
            (
                (
                    ap.quiet_hours,
                    ap.tz,
                    frozenset(ap.modes_governed),
                    ap.urgency_override,
                )
                if ap is not None
                else None
            ),
            frozenset(reg.resources),
            reg.allow_self_loops,
            (
                (
                    _callable_identity(red.summarize),
                    red.store_raw_trusted,
                    red.redact_caller_identifier,
                )
                if red is not None
                else None
            ),
            reg.retention_days,
            _callable_identity(reg.result_summary),
            reg.require_constitution_echo,
            reg.prompt_template_format,
            reg.constitution_injection,
            reg.system_prompt_budget_bytes,
            getattr(reg, "allow_prompt_override", False),
        )

    @classmethod
    def contract_equivalent(
        cls, a: SourceRegistration, b: SourceRegistration
    ) -> bool:
        """True when ``a`` and ``b`` declare the same source contract."""
        return cls.contract_signature(a) == cls.contract_signature(b)

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
        allow_prompt_override = getattr(reg, "allow_prompt_override", False)
        if not isinstance(allow_prompt_override, bool):
            raise RegistrationError(
                f"Source '{reg.name}': allow_prompt_override must be a bool "
                f"when declared, got {type(allow_prompt_override).__name__}."
            )

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


def _value_identity(value: object) -> object:
    """A hashable, comparison-stable identity for a closure-captured value.

    Used by :func:`_callable_identity` to fingerprint each free variable a
    handler closure captures (kestrel-sovereign#2522 P1):

    * ``None`` and immutable scalars (``str``/``bytes``/``bool``/numbers) and
      enums compare **by value**, so a rebuilt closure that captured an equal
      ``task_name`` string is still equivalent — no spurious mismatch.
    * A captured *callable* recurses through :func:`_callable_identity`, so a
      captured bound method compares by its owner instance (stable across a
      same-instance re-init even though the bound-method wrapper object is
      freshly created on each attribute access).
    * Any other object (a coordinator, a discovery callback object, …) compares
      **by ``id()``** — a distinct owner/dependency instance is correctly a
      different contract.

    ``id()`` is reliable for this comparison because both the existing and the
    incoming registration hold a live reference to their callables (and thus to
    everything those capture) for the duration of the equivalence check, so no
    id can be reused between the two operands.
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    if isinstance(value, enum.Enum):
        return value
    if callable(value):
        return _callable_identity(value)
    return id(value)


def _callable_identity(fn: object) -> Optional[tuple]:
    """A stable, hashable identity for a handler / sanitizer / schema callable.

    Fingerprints the *whole behavioral identity* of the callable so two
    callables that merely share a qualified name are distinguished when they
    would in fact behave differently (kestrel-sovereign#2522 P1). A genuine
    same-definition rebuild yields an equal identity; a callable that differs
    in owner, module, compiled code, default-bound arguments, or captured
    dependencies does not.

    * ``None`` → ``None``.
    * bound method / bound builtin → ``("bound", qualname, id(owner))``.
    * :class:`functools.partial` → recurse into ``func`` plus the bound
      positional/keyword arguments (each via :func:`_value_identity`).
    * plain function OR closure → ``("function", module, qualname, code-id,
      defaults, kwdefaults, closure-cells)``. Reducing this to ``qualname``
      alone (the old behavior) equated two functions built from the same
      factory that bake in *different* default-bound behavior, or two same-name
      functions defined in different modules — the false-equivalence the audit
      found. The compiled ``__code__`` object is shared across every function
      produced from one ``def``, so a genuine rebuild keeps an equal
      ``code-id`` while a different definition (or a different default) does
      not. Defaults / kwdefaults / captured free variables each fold through
      :func:`_value_identity`, so immutable captures still compare by value.
    * bound/builtin method without ``__self__`` → ``("function", module,
      qualname)`` (no Python code/defaults to fold).
    * any other callable object → ``("object", type-qualname, id(obj))`` — a new
      instance is (correctly) a different contract.
    """
    if fn is None:
        return None

    qual = (
        getattr(fn, "__qualname__", None)
        or getattr(fn, "__name__", None)
        or type(fn).__qualname__
    )

    # Bound method (or bound builtin): the owner instance is the
    # behavior-affecting binding, and it survives a same-instance re-init even
    # though ``obj.method`` yields a fresh wrapper object on each access.
    self_obj = getattr(fn, "__self__", None)
    if self_obj is not None:
        return ("bound", qual, id(self_obj))

    if isinstance(fn, functools.partial):
        return (
            "partial",
            _callable_identity(fn.func),
            tuple(_value_identity(a) for a in fn.args),
            tuple(sorted((k, _value_identity(v)) for k, v in fn.keywords.items())),
        )

    # A Python function — closure or not. Fold module + compiled code identity
    # + default-bound arguments + captured free variables, NOT just the
    # qualified name. Two ``build_*(dep)`` handlers sharing a qualname but
    # baking in a different default (or a different captured dependency, or
    # defined in a different module) are then correctly a mismatch, while a
    # same-``def`` rebuild — which reuses the one compiled code object and the
    # same immutable defaults/captures — stays equivalent.
    if inspect.isfunction(fn):
        code = getattr(fn, "__code__", None)
        cells: list = []
        for cell in getattr(fn, "__closure__", None) or ():
            try:
                cells.append(_value_identity(cell.cell_contents))
            except ValueError:
                # Empty cell — a free variable not yet bound; nothing to fold.
                cells.append(None)
        defaults = tuple(
            _value_identity(d) for d in (getattr(fn, "__defaults__", None) or ())
        )
        kwdefaults = tuple(
            sorted(
                (k, _value_identity(v))
                for k, v in (getattr(fn, "__kwdefaults__", None) or {}).items()
            )
        )
        return (
            "function",
            getattr(fn, "__module__", None),
            qual,
            id(code) if code is not None else None,
            defaults,
            kwdefaults,
            tuple(cells),
        )

    # A builtin function (``len``) or a stray method object without
    # ``__self__`` has no Python ``__code__``/defaults to fold — its behavior
    # is fixed by module + name.
    if inspect.ismethod(fn) or inspect.isbuiltin(fn):
        return ("function", getattr(fn, "__module__", None), qual)

    # Some other callable object (an instance with ``__call__``). Its identity
    # IS the instance; a fresh instance is correctly a different contract.
    return ("object", type(fn).__qualname__, id(fn))


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
