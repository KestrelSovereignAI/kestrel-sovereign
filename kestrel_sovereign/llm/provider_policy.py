"""Which providers an agent's LLM work may use, and who may widen that (#2652).

On 2026-07-20 the Talon coordinator's LLM work ran on OpenRouter's Gemini.
Anthropic was down, OpenAI was over quota, and a model override the agent had
written for itself pinned its turns to a provider the Sovereign had never
approved. Nothing refused it, and nothing recorded who had chosen it.

Two rules, both read from the host's ``[llm]`` section:

* ``route_priority`` is the approved chain. A model override the agent writes
  for itself may select only a vendor in that chain; any other is refused
  before it is stored (:class:`ProviderPolicyRefusal`). An override the
  Sovereign or an operator wrote is not narrowed by the chain. With no
  ``route_priority`` declared, the chain is every initialized route, as the
  default route order already is.
* ``denied_providers`` names vendors (``"openrouter"``) or routes
  (``"google:vertex"``) that are never used, whatever their health and
  whoever wrote an override. It matches a route's vendor or its
  ``vendor:route`` name, not a model id: OpenRouter's ``google/...`` models
  are an ``openrouter`` route. Empty by default.

A write path that holds a principal names it with :func:`override_writer`. An
override counts as the Sovereign's or an operator's only when an
endpoint-bound principal is attested for that write: the ``/api/model/set``
caller, or the caller of a ``!model-set`` command typed through
:meth:`TaskManager.execute_command`. Everything else counts as the agent's: a
``set_model`` call the agent's own model made, a scheduled or subagent call, a
command with no caller, a stored override with no recorded writer, and any
write that names no writer at all (the agent's ``set_model()`` method, or a
feature calling the service directly). A path that forgets to name its writer
is therefore held to the chain, not exempted from it.
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .error_handling import LLMError

DENIED_PROVIDERS_KEY = "denied_providers"

# How many override decisions an LLM service keeps for the audit surface.
OVERRIDE_HISTORY_LIMIT = 100


class OverrideSource(str, enum.Enum):
    """Who wrote a model override."""

    SOVEREIGN = "sovereign"
    """An endpoint-bound sovereign-key caller."""

    OPERATOR = "operator"
    """An authenticated human principal of this host that is not the Sovereign."""

    AGENT = "agent"
    """The agent's own cognition, or a write no principal attested or named."""

    SYSTEM = "system"
    """Kestrel reinstating a mandate: one a local-only privacy mode forces or restores."""


class ProviderPolicyConfigError(ValueError):
    """``[llm] denied_providers`` is not a list of non-empty strings."""


class ProviderPolicyRefusal(LLMError, ValueError):
    """A provider was refused by the provider policy.

    A ``ValueError`` as well, so the callers that already turn an unusable
    mandate into a 400 or a kept preference treat a refused one the same way.
    """

    PROVIDER_DENIED = "provider_denied"
    OUTSIDE_APPROVED_CHAIN = "provider_outside_approved_chain"

    def __init__(
        self,
        code: str,
        *,
        vendor: Optional[str],
        route: Optional[str],
        model: Optional[str],
        reason: str,
        source: Optional[OverrideSource] = None,
        approved_vendors: Iterable[str] = (),
    ) -> None:
        self.code = code
        self.vendor = vendor
        self.route = route
        self.model = model
        self.reason = reason
        self.source = source
        self.approved_vendors = sorted(set(approved_vendors))
        super().__init__(f"{code}: {reason}")

    @property
    def provider(self) -> Optional[str]:
        """The refused provider as ``vendor`` or ``vendor:route``."""
        if self.vendor and self.route:
            return f"{self.vendor}:{self.route}"
        return self.vendor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "provider": self.provider,
            "vendor": self.vendor,
            "route": self.route,
            "model": self.model,
            "reason": self.reason,
            "source": self.source.value if self.source else None,
            "approved_vendors": list(self.approved_vendors),
        }


@dataclass(frozen=True)
class OverrideProvenance:
    """Who wrote a model override, and where."""

    source: OverrideSource
    role: str
    """The writer's caller role (``sovereign``, ``authenticated``...), or ``none``."""
    identity: Optional[str] = None
    """The caller's identity (an email, ``api_key``); never a credential."""
    via: Optional[str] = None
    """The write path: ``set_model``, ``/api/model/set``, ``replay``, ``privacy``."""
    task: Optional[str] = None
    """The turn or invocation the override was written in."""
    at: Optional[str] = None
    """When the override was written (ISO 8601, UTC)."""

    @property
    def narrowed_by_chain(self) -> bool:
        """Whether the approved chain bounds this override."""
        return self.source is OverrideSource.AGENT

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "source": self.source.value,
            "role": self.role,
            "identity": self.identity,
            "via": self.via,
            "task": self.task,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["OverrideProvenance"]:
        """The provenance ``data`` records, or None when it records none."""
        if not isinstance(data, Mapping):
            return None
        try:
            source = OverrideSource(data.get("source"))
        except ValueError:
            return None
        role = data.get("role")

        def _text(key: str) -> Optional[str]:
            value = data.get(key)
            return value if isinstance(value, str) and value else None

        return cls(
            source=source,
            role=role if isinstance(role, str) and role else "none",
            identity=_text("identity"),
            via=_text("via"),
            task=_text("task"),
            at=_text("at"),
        )


def agent_provenance(*, via: str, role: str = "none") -> OverrideProvenance:
    """Provenance for a write no principal attested."""
    return OverrideProvenance(
        source=OverrideSource.AGENT,
        role=role,
        via=via,
        task=_current_task(),
        at=utc_now_iso(),
    )


def system_provenance(*, via: str) -> OverrideProvenance:
    """Provenance for a mandate Kestrel reinstates itself."""
    return OverrideProvenance(
        source=OverrideSource.SYSTEM,
        role="system",
        via=via,
        task=_current_task(),
        at=utc_now_iso(),
    )


def provenance_for_caller(caller: Any, *, via: str) -> OverrideProvenance:
    """Provenance for a write an endpoint-bound ``caller`` made directly.

    A sovereign caller is the Sovereign. An authenticated human principal is
    an operator. A transport-only A2A caller, an anonymous caller and no
    caller at all are not principals, so the write is the agent's.
    """
    from kestrel_sovereign.auth import AuthMethod, CallerRole

    if caller is None:
        return agent_provenance(via=via)
    role = getattr(getattr(caller, "role", None), "value", None) or "none"
    identity = getattr(caller, "identity", None)
    if getattr(caller, "is_sovereign", False) is True:
        source = OverrideSource.SOVEREIGN
    elif (
        getattr(caller, "role", None) is CallerRole.AUTHENTICATED
        and getattr(caller, "auth_method", None) is not AuthMethod.A2A_TRANSPORT
    ):
        source = OverrideSource.OPERATOR
    else:
        source = OverrideSource.AGENT
    return OverrideProvenance(
        source=source,
        role=role,
        identity=identity if isinstance(identity, str) else None,
        via=via,
        task=_current_task(),
        at=utc_now_iso(),
    )


# Who is writing a model override, for the duration of one synchronous call to
# ``LLMService.set_model_preference`` or ``resolve_model_preference``. Each
# write path names its writer here rather than threading it through every
# service and test double that shares that interface.
_OVERRIDE_WRITER: ContextVar[Optional[OverrideProvenance]] = ContextVar(
    "kestrel_override_writer", default=None
)


@contextmanager
def override_writer(provenance: OverrideProvenance) -> Iterator[None]:
    """Name ``provenance`` as the writer of overrides made in this scope.

    Hold it around the synchronous preference call only, never across an
    await, so it cannot attach to another write.
    """
    if not isinstance(provenance, OverrideProvenance):
        raise TypeError("override_writer needs an OverrideProvenance")
    token = _OVERRIDE_WRITER.set(provenance)
    try:
        yield
    finally:
        _OVERRIDE_WRITER.reset(token)


def current_override_writer(*, via: str) -> OverrideProvenance:
    """The writer named for this scope, else the agent.

    An unnamed writer is the agent's, so the chain bounds it: only a path that
    attests a principal can write an override the chain does not narrow.
    """
    return _OVERRIDE_WRITER.get() or agent_provenance(via=via)


# The tool a principal invoked by typing its command, for the duration of that
# invocation. ``TaskManager.execute_command`` is the only place core knows a
# tool call was typed rather than composed by the model (#3112), so it sets
# this; a tool reads it to attribute what it writes.
_COMMAND_AUTHORED_SKILL: ContextVar[Optional[str]] = ContextVar(
    "kestrel_command_authored_skill", default=None
)


@contextmanager
def command_authored_scope(skill_id: str) -> Iterator[None]:
    """Mark ``skill_id`` as invoked by a typed command for this scope."""
    token = _COMMAND_AUTHORED_SKILL.set(skill_id)
    try:
        yield
    finally:
        _COMMAND_AUTHORED_SKILL.reset(token)


def tool_write_provenance(skill_id: str) -> OverrideProvenance:
    """Provenance for an override the tool ``skill_id`` is writing.

    The current caller is the writer only when this very tool was invoked by a
    typed command. Otherwise the model, a schedule or a subagent chose the
    call, and the write is the agent's even inside a Sovereign's turn: being
    asked by the Sovereign is not the Sovereign choosing the provider.
    """
    if _COMMAND_AUTHORED_SKILL.get() != skill_id:
        return agent_provenance(via=skill_id)
    from kestrel_sovereign.auth import current_caller_context

    return provenance_for_caller(current_caller_context(), via=f"command:{skill_id}")


def _current_task() -> Optional[str]:
    """The turn, else the invocation, this write happens in."""
    from kestrel_sovereign.agent.invocation import current_invocation_id
    from kestrel_sovereign.telemetry import current_turn_id

    return current_turn_id() or current_invocation_id()


def parse_denied_providers(config: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    """``[llm] denied_providers``, validated.

    A malformed value raises rather than being ignored: a deny-list that
    silently denies nothing is the failure it exists to prevent.
    """
    if not config:
        return ()
    raw = config.get(DENIED_PROVIDERS_KEY)
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ProviderPolicyConfigError(
            f"[llm] {DENIED_PROVIDERS_KEY} must be a list of vendor or "
            f"vendor:route names, not {type(raw).__name__}"
        )
    entries = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise ProviderPolicyConfigError(
                f"[llm] {DENIED_PROVIDERS_KEY} entries must be non-empty "
                f"strings; got {entry!r}"
            )
        entries.append(entry.strip())
    return tuple(entries)


def route_priority_entries(config: Optional[Mapping[str, Any]]) -> List[str]:
    """The declared ``route_priority`` names, in order."""
    if not config:
        return []
    priority = config.get("route_priority") or []
    if not isinstance(priority, (list, tuple)):
        return []
    return [key for key in priority if isinstance(key, str) and key]


def selector_is_denied(
    vendor: Optional[str], route: Optional[str], denied: Sequence[str]
) -> bool:
    """Whether ``vendor``/``route`` is named by ``denied``.

    A vendor entry denies all its routes; a ``vendor:route`` entry denies that
    route. A vendor with no route is denied only by a vendor entry, since it may
    still reach its routes that are not denied.
    """
    if not vendor:
        return False
    for entry in denied:
        if entry == vendor:
            return True
        if route and entry == f"{vendor}:{route}":
            return True
    return False


def provider_is_denied(provider: Mapping[str, Any], denied: Sequence[str]) -> bool:
    """Whether the route ``provider`` is named by ``denied``."""
    if not denied:
        return False
    name = provider.get("name")
    if isinstance(name, str) and name in denied:
        return True
    return selector_is_denied(provider.get("vendor"), provider.get("route"), denied)


def approved_vendors(
    config: Optional[Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
    denied: Sequence[str],
) -> set[str]:
    """Vendors an override the agent writes may select.

    The vendors ``route_priority`` names, or with none declared every
    initialized route's vendor, less any vendor ``denied`` names whole. An
    empty set admits no agent override: denying the whole chain leaves the
    agent nothing to select, not everything.
    """
    declared = route_priority_entries(config)
    if declared:
        vendors = {key.split(":", 1)[0] for key in declared}
    else:
        vendors = {p.get("vendor") for p in providers if p.get("vendor")}
    return {vendor for vendor in vendors if vendor not in denied}


def describe_chain(
    config: Optional[Mapping[str, Any]],
    providers: Sequence[Mapping[str, Any]],
    denied: Sequence[str],
    disabled: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """The effective chain, in order, with each route's state.

    ``route_priority`` names first, then any other initialized route, which
    default routing appends but only follows when no chain is declared.
    """
    declared = route_priority_entries(config)
    by_name = {p.get("name"): p for p in providers if p.get("name")}
    names = list(declared)
    names.extend(name for name in by_name if name not in declared)
    chain = []
    for name in names:
        provider = by_name.get(name)
        vendor, _, route = name.partition(":")
        if provider is not None:
            vendor = provider.get("vendor") or vendor
            route = provider.get("route") or route
        if provider_is_denied({"name": name, "vendor": vendor, "route": route}, denied):
            state = "denied"
        elif provider is None:
            state = "not_initialized"
        elif name in disabled:
            state = "disabled"
        else:
            state = "available"
        chain.append({
            "name": name,
            "vendor": vendor,
            "route": route or None,
            "model": provider.get("model") if provider is not None else None,
            "approved": (name in declared) if declared else True,
            "state": state,
        })
    return chain


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
