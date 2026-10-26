"""
Dynamic tool registry mixin for KestrelAgent.

Extracted from kestrel_agent.py — manages the tool catalog exposed to the
orchestrator LLM, including feature dispatcher tools and directly-promoted
individual tools with LRU eviction.
"""

import hashlib
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from uuid import uuid4

from kestrel_sdk.resources import ResourceConflict, ResourceOutcomeUnknown

from kestrel_sdk.tools import GovernedToolBinding

from kestrel_sovereign.agent.tool_allowlist import (
    KnownToolNames,
    bump_registry_generation,
    names_held_by_live_turn,
    registry_generation,
)
from kestrel_sovereign.tools.result_contract import enforce_tool_result_contract


@dataclass(frozen=True)
class _GovernedToolBinding:
    """An in-process reference to one owner's mounted registration batch.

    The existing owner registry carries the batch identity. No second tool
    registry or serialized bearer credential is involved.
    """

    agent: Any
    name: str
    owner: str
    registration: object
    tool: Any
    security_feature: str

    @property
    def is_current(self) -> bool:
        return (
            self.agent._explored_features.get(self.owner) is self.registration
            and self.agent._direct_tools.get(self.name) is self.tool
            and self.agent._tool_to_feature.get(self.name) == self.owner
        )

    def require_current(self) -> None:
        if not self.is_current:
            raise ResourceConflict("Tool registration expired; reopen the originating view")

    async def invoke(self, arguments: Dict[str, Any], *, session_id: str,
                     source: str = "external") -> Any:
        from kestrel_sovereign.llm.adapter import ToolCall
        from kestrel_sovereign.storage.db.interface import DatabaseError

        self.require_current()
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a dictionary")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("A tool invocation requires a session_id")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("A tool invocation requires a source")
        try:
            return await self.agent._dispatch_tool_call(
                ToolCall(id=str(uuid4()), name=self.name, arguments=deepcopy(arguments)),
                features_by_tool_name={}, known_tools=self.agent._known_tool_names(),
                messages=[], iteration=0, user_message=None, session_id=session_id,
                bound_tool=self, source=source,
            )
        except DatabaseError as exc:
            raise ResourceOutcomeUnknown(
                "Tool audit could not be confirmed; reconcile before retrying"
            ) from exc


@dataclass(frozen=True)
class _RegistryNames:
    """Every name the live registry resolves, split by where it comes from."""

    #: Dispatcher names and ``@tool`` names of enabled, loaded features. These
    #: stay resolvable whether or not they are mounted as direct tools.
    feature_backed: frozenset
    #: Names currently mounted in ``_direct_tools``, including those (MCP
    #: tools, collision-renamed tools) that no feature's ``get_tools()`` carries.
    mounted: frozenset
    #: Features whose ``get_tools()`` raised, so their ``@tool`` names are absent.
    unreadable_features: tuple
    features_loaded: int
    features_enabled: int


def _derive_registry_names(agent: Any) -> _RegistryNames:
    """Derive the registry's names from what is actually loaded (#2929).

    A module function rather than a method so the orchestrator's test doubles,
    which bind a fixed list of real methods onto a ``MagicMock``, reach it
    through the methods they do bind.
    """
    mounted = frozenset(
        name for name in (getattr(agent, "_direct_tools", None) or {})
        if isinstance(name, str)
    )
    backed = {
        name for name in agent._registered_features_by_tool_name()
        if isinstance(name, str)
    }
    unreadable = []
    features = list((getattr(agent, "features", None) or {}).items())
    enabled = 0
    for key, feature in features:
        if not bool(getattr(feature, "enabled", True)):
            continue
        enabled += 1
        get_tools = getattr(feature, "get_tools", None)
        if not callable(get_tools):
            continue
        try:
            for tool in get_tools() or []:
                name = getattr(tool, "name", None)
                if isinstance(name, str) and name:
                    backed.add(name)
        except Exception:  # noqa: BLE001 — one broken feature mustn't
            # shrink the allowlist for every other feature; it is named in the
            # refusal evidence instead, since its own names DO drop out (#2935).
            unreadable.append(str(key))
    return _RegistryNames(
        feature_backed=frozenset(backed),
        mounted=mounted,
        unreadable_features=tuple(sorted(unreadable)),
        features_loaded=len(features),
        features_enabled=enabled,
    )


def _resolvable_tool_names(agent: Any) -> KnownToolNames:
    """Every name the live registry resolves, stamped with its provenance."""
    generation = registry_generation(agent)
    names = _derive_registry_names(agent)
    return KnownToolNames(
        names.feature_backed | names.mounted,
        registry_generation=generation,
        features_loaded=names.features_loaded,
        features_enabled=names.features_enabled,
        unreadable_features=names.unreadable_features,
    )


class ToolRegistryMixin:
    """Mixin providing dynamic tool loading and management for KestrelAgent."""

    # Startup-promoted operational features currently pin 50 tools
    # (Task, Todo, Peers, Save, Spawn, StrategicMemory, Wait). Keep enough
    # headroom for the common model_agent + memory_feature exploration
    # path (31 tools) without immediately evicting the first explored
    # feature. ``test_startup_promotion_stays_under_budget`` derives the
    # promoted set from discovery and fails when this is outgrown.
    MAX_DIRECT_TOOLS = 88

    # OpenAI (and most providers) cap function names at 64 chars matching
    # ``^[a-zA-Z0-9_-]+$``. A single over-length name rejects the entire tool
    # list request, so every dynamically-generated name is bounded here.
    MAX_TOOL_NAME_LEN = 64

    def _bounded_unique_tool_name(self, candidate: str) -> str:
        """Return a provider-safe (<=64 char), registry-unique tool name.

        Provider function names must match ``^[a-zA-Z0-9_-]+$``; MCP tool names
        routinely contain ``.``/``/``/``:`` (e.g. ``server.tool``,
        ``fetch/raw``), so any invalid char is replaced first — otherwise a
        single bad name rejects the whole tool-list request. The schema name is
        purely the LLM-facing handle; the tool's own ``.execute`` knows the real
        downstream name, so renaming here is safe.

        Over-length candidates (common for MCP owners that are package names
        or URLs once prefixed) are then truncated and suffixed with a short
        stable hash of the full candidate. Any residual collision (including
        ones created by sanitisation, e.g. ``fetch.raw`` and ``fetch/raw``) gets
        a numeric suffix. Uniqueness is checked against ``_direct_tools`` so two
        owners can never end up writing the same schema name.
        """
        candidate = re.sub(r"[^a-zA-Z0-9_-]", "_", str(candidate))
        limit = self.MAX_TOOL_NAME_LEN
        if len(candidate) > limit:
            digest = hashlib.sha1(candidate.encode()).hexdigest()[:8]
            candidate = f"{candidate[: limit - 9]}_{digest}"
        if candidate not in self._direct_tools:
            return candidate
        base = candidate[: limit - 4]
        suffix = 2
        name = f"{base}_{suffix}"
        while name in self._direct_tools:
            suffix += 1
            name = f"{base}_{suffix}"
        return name

    def _build_feature_tools(self) -> List[Dict[str, Any]]:
        """
        Build the list of feature tools for the orchestrator LLM.

        Each feature is exposed as a high-level tool that the orchestrator
        can call. The feature then handles the task using its own tools
        and context (A2A pattern).

        Returns:
            List of tools in OpenAI function calling format
        """
        tools = []
        failed_features = []

        for feature in self.features.values():
            try:
                if not self._feature_visible_to_llm(feature):
                    continue
                # Skip subagent dispatcher for pre-explored features
                # (their individual tools are already in the direct tool list)
                if feature.tool_name in self._explored_features:
                    continue
                tool_def = feature.to_orchestrator_tool()
                tools.append(tool_def)
                logging.info(f"[AGENTIC] Added tool: {feature.tool_name}")
            except (AttributeError, TypeError, KeyError, ValueError) as e:
                logging.error(f"[AGENTIC] FAILED to build tool for {feature.name}: {e}")
                failed_features.append({"name": feature.name, "error": str(e)})
            except Exception as e:
                logging.error(f"[AGENTIC] FAILED to build tool for {feature.name}: {e}", exc_info=True)
                failed_features.append({"name": feature.name, "error": str(e)})

        # Log summary
        logging.info(f"[AGENTIC] Built {len(tools)} tools from {len(self.features)} features")
        if failed_features:
            logging.error(f"[AGENTIC] Failed features: {failed_features}")

        return tools

    def _build_all_tools(self) -> list:
        """Build combined tool list: feature dispatchers + explored individual tools."""
        tools = self._build_feature_tools()
        hidden_tools = self._hidden_context_tools()
        hidden_features = self._hidden_context_features()
        tools.extend(
            tool_def
            for tool_def in self._direct_tool_defs
            if not self._direct_tool_hidden_from_context(
                tool_def,
                hidden_tools=hidden_tools,
                hidden_features=hidden_features,
            )
        )
        return tools

    def build_progressive_tool_schemas(
        self,
        *,
        include_direct_tools: bool = True,
        max_direct_tools: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Return the current progressive-disclosure tool view.

        Non-chat transports such as voice realtime use the same registry
        state as the chat orchestrator: feature dispatcher tools are visible
        first, and direct ``@tool`` schemas appear only after the feature has
        been explored through subagent dispatch.  ``max_direct_tools`` caps the
        direct-tool portion of this view for sessions that need a smaller
        transport-level tool budget without mutating the chat registry.
        """
        tools = self._build_feature_tools()
        if not include_direct_tools:
            return tools

        hidden_tools = self._hidden_context_tools()
        hidden_features = self._hidden_context_features()
        direct_tools = [
            tool_def
            for tool_def in self._direct_tool_defs
            if not self._direct_tool_hidden_from_context(
                tool_def,
                hidden_tools=hidden_tools,
                hidden_features=hidden_features,
            )
        ]
        if max_direct_tools is not None and max_direct_tools >= 0:
            direct_tools = direct_tools[-max_direct_tools:]
        tools.extend(direct_tools)
        return tools

    def _visible_features_by_tool_name(self) -> Dict[str, Any]:
        """Return feature dispatch targets currently visible to the LLM."""
        return {
            feature.tool_name: feature
            for feature in self.features.values()
            if self._feature_visible_to_llm(feature)
        }

    def _visible_known_tool_names(self) -> set[str]:
        """Return the tool names currently *advertised* to the LLM.

        This is the schema view — what ``_build_all_tools`` puts in the request
        — and it is deliberately narrower than what the agent can dispatch:
        progressive disclosure hides tools, exploration replaces a feature's
        dispatcher with its individual tools, and LRU eviction sheds direct
        tools. Use :meth:`_known_tool_names` for the guardrail allowlist;
        anything narrower rejects live capability as "unknown" (#2929).
        """
        return {
            tool_def["function"]["name"]
            for tool_def in self._build_all_tools()
            if isinstance(tool_def.get("function", {}).get("name"), str)
        }

    def _registered_features_by_tool_name(self) -> Dict[str, Any]:
        """Return every dispatchable loaded feature keyed by dispatcher name.

        Same enabled/dispatch-capability gates as
        :meth:`_visible_features_by_tool_name`, minus the context-profile
        filter: hiding a feature is a prompt-budget decision, not a capability
        boundary, so a hidden feature is still *known* and still dispatchable
        if the model names it. A soft-disabled feature (``enabled=False``) is
        excluded — that one IS a capability boundary (#2522).
        """
        return {
            feature.tool_name: feature
            for feature in self.features.values()
            if bool(getattr(feature, "enabled", True))
            and self._feature_supports_subagent_dispatch(feature)
        }

    def _registered_tool_names(self) -> KnownToolNames:
        """Return every tool name the live registry can resolve.

        Derived from what is actually loaded, never hand-maintained (#2929):

        * subagent dispatcher names of every enabled, dispatchable feature —
          whether or not that feature has been explored (exploration drops the
          dispatcher from the *schema* view, which is what made a name that
          worked one call ago bounce on the next),
        * every ``@tool`` name those features expose, promoted or not,
        * every direct tool currently mounted, including dynamically-mounted
          ones (MCP servers etc.) that belong to no feature.

        This is the set ``execute_named_tool`` resolves against, so it is what
        that path records as its allowlist (#2935).
        """
        return _resolvable_tool_names(self)

    def _known_tool_names(self) -> KnownToolNames:
        """Return the guardrail allowlist for this turn.

        The union of what the LLM is currently shown
        (:meth:`_visible_known_tool_names`) and what the live registry can
        resolve (:meth:`_registered_tool_names`). Membership is *derived*, so
        it cannot drift from the features that are loaded, and it cannot shrink
        mid-turn as exploration, eviction, or a context profile changes the
        advertised view — both of which produced pre-permission, unaudited
        bounces of legitimately registered tools (#2929).

        The result carries the registry generation and feature counts it was
        derived from, so a refusal can say what produced the set (#2935).
        """
        return _resolvable_tool_names(self).widened(
            self._visible_known_tool_names()
        )

    def register_dynamic_tools(
        self, owner: str, tools, *, pin: bool = False,
        on_registered: Callable[[Any, GovernedToolBinding], None] | None = None,
    ) -> int:
        """Mount runtime tools owned by ``owner`` into the direct-tool registry.

        This is the single primitive for adding first-class, LLM-callable tools
        at runtime. Both feature exploration (``owner`` = ``feature.tool_name``)
        and out-of-band tool sources such as MCP servers (``owner`` =
        ``"mcp:<server>"``) go through here, so every dynamically-mounted tool
        gets the same progressive-disclosure schema slot, LRU eviction, and —
        because execution flows through ``_dispatch_direct_tool`` —
        ``ToolResult`` envelope, ``a2a_tool_dispatches`` row, hook, and
        permission treatment.

        ``tools`` is any iterable of tool handles exposing ``.name`` and
        ``.schema.to_openai_format()`` plus an awaitable ``.execute(**kwargs)``
        (the SDK ``AgentTool`` protocol). Names that collide with an existing
        direct tool are disambiguated as ``<owner>__<tool.name>`` (the owner is
        sanitised to a schema-safe token first). Returns how many of ``tools``
        are still mounted once LRU eviction has run: fewer than offered, and
        possibly none, when the cap is reached and every older owner is pinned
        or held by the live turn, so the mount itself is what gets evicted
        (#2935). That eviction is logged at WARNING.

        ``owner`` is recorded in ``_explored_features`` so eviction and
        :meth:`unregister_dynamic_tools` treat feature- and non-feature owners
        uniformly. ``pin=True`` exempts the owner from LRU eviction.

        ``schema.model_visible=False`` keeps a tool mounted and governed but
        omits it from model schemas. Exposure is not authorization. A synchronous
        ``on_registered(tool, binding)`` callback receives only tools retained
        after eviction, with their actual disambiguated names. Bindings expire
        when this owner is registered again, unmounted, or evicted, even when
        the same tool object is reused. Callbacks must not mutate the registry.

        Permissions: a tool not present in the security permission store
        resolves to ``PermissionLevel.ASK`` (the safe default — never a silent
        ALLOW), so dynamically-mounted tools are always gated. Letting an
        operator *preconfigure* ALLOW/DENY for non-feature owners (the security
        feature only enumerates ``self.agent.features`` today) is handled in
        #1979 PR5; it is a convenience, not a safety gap.
        """
        if on_registered is not None and not callable(on_registered):
            raise TypeError("on_registered must be callable")
        safe_owner = re.sub(r"\W+", "_", str(owner))
        registration = object()
        self._explored_features[owner] = registration
        if pin:
            self._pinned_features.add(owner)
        mounted = []
        for tool in tools:
            name = tool.name
            if name in self._direct_tools:
                name = f"{safe_owner}__{tool.name}"
            # Bound to the provider name cap AND guarantee uniqueness — covers
            # long MCP owner/tool identifiers and the case where two owners
            # sanitise to the same prefix and expose a colliding tool name.
            name = self._bounded_unique_tool_name(name)
            self._direct_tools[name] = tool
            tool_def = tool.schema.to_openai_format()
            tool_def["function"]["name"] = name
            if getattr(tool.schema, "model_visible", True) is not False:
                self._direct_tool_defs.append(tool_def)
            self._tool_to_feature[name] = owner
            mounted.append((name, tool))
        bump_registry_generation(self)
        self._maybe_evict_direct_tools()
        retained = [(name, tool) for name, tool in mounted
                    if self._direct_tools.get(name) is tool]
        registered = len(retained)
        logging.info(
            f"[DYNAMIC-TOOLS] Registered {registered} of {len(mounted)} direct "
            f"tools for '{owner}'. Total: {len(self._direct_tools)}"
        )
        if on_registered is not None:
            for name, tool in retained:
                on_registered(tool, _GovernedToolBinding(
                    self, name, owner, registration, tool,
                    self._security_feature_name_for_tool(name),
                ))
        return registered

    def unregister_dynamic_tools(self, owner: str) -> int:
        """Remove all direct tools owned by ``owner`` (inverse of
        :meth:`register_dynamic_tools`).

        Used when a feature is disabled or an MCP server is removed. Returns the
        number of tools removed.
        """
        to_remove = [
            name for name, tool_owner in self._tool_to_feature.items()
            if tool_owner == owner
        ]
        for name in to_remove:
            self._direct_tools.pop(name, None)
            self._tool_to_feature.pop(name, None)
        if to_remove:
            removed = set(to_remove)
            self._direct_tool_defs = [
                tool_def for tool_def in self._direct_tool_defs
                if tool_def.get("function", {}).get("name") not in removed
            ]
        was_mounted = self._explored_features.pop(owner, None) is not None
        self._pinned_features.discard(owner)
        if to_remove or was_mounted:
            bump_registry_generation(self)
        return len(to_remove)

    def _register_explored_feature_tools(self, feature) -> None:
        """Register a feature's individual tools for direct calling.

        After a successful subagent dispatch, the feature's @tool methods
        become available for the orchestrator to call directly without
        a subagent LLM hop. Thin wrapper over :meth:`register_dynamic_tools`
        that adds the feature-specific idempotency guard and ToolResult
        contract enforcement.
        """
        if feature.tool_name in self._explored_features:
            return
        # Enforce the ToolResult return contract for migrated features
        # (#1042 layer 4 / #1061). No-op for non-migrated modules; raises
        # ToolResultContractError for migrated modules whose @tool
        # methods don't annotate ``-> ToolResult``. Catching the
        # registration failure rather than letting it propagate would
        # silently ship an honesty-violating tool, so we let it surface.
        enforce_tool_result_contract(feature)
        self.register_dynamic_tools(feature.tool_name, feature.get_tools())

    def _maybe_evict_direct_tools(self) -> None:
        """Evict least-recently-explored feature's tools if over limit.

        Skips features in ``_pinned_features`` (#1580 / D). The pin
        tier protects startup-promoted operationally-critical features
        (Peers / Tasks / Spawn / Save / StrategicMemory) so a long
        session can't silently drop tools the agent depends on for
        basic orchestration. Logged eviction lists the actual tool
        names (not just a count) so the operator can spot a
        regression in the audit trail.

        Also skips, while the agent's turn is live, an owner holding a tool
        that turn has already been allowed to call that no feature's
        ``get_tools()`` carries (#2935). Evicting a feature's tools leaves them
        resolvable through the feature; evicting an MCP tool or a
        collision-renamed one unmounts the only copy, and exploration runs
        mid-turn, so the next call to a tool that just answered was refused as
        not in the allowlist.

        When every older owner is pinned or held, the owner whose registration
        overflowed the cap is the oldest evictable one, so it goes: an explored
        feature stays reachable through its dispatcher, while an MCP mount is
        refused and ``register_dynamic_tools`` reports none mounted. That
        eviction, and any other that passed over a held owner, is a WARNING
        naming the evicted tools and the held owners.
        When nothing at all is evictable, nothing is unmounted: the count stays
        over the cap, with a WARNING, until the first registration after the
        turn.
        """
        pinned = getattr(self, "_pinned_features", set())
        in_use = names_held_by_live_turn(self)
        feature_backed = None

        def _held_by_live_turn(owner) -> bool:
            nonlocal feature_backed
            at_risk = [
                name for name, tool_owner in self._tool_to_feature.items()
                if tool_owner == owner and name in in_use
            ]
            if not at_risk:
                return False
            if feature_backed is None:
                feature_backed = _derive_registry_names(self).feature_backed
            return any(name not in feature_backed for name in at_risk)

        while len(self._direct_tools) > self.MAX_DIRECT_TOOLS:
            # Find oldest UNPINNED feature the live turn is not holding.
            held = []
            oldest = None
            for owner in self._explored_features:
                if owner in pinned:
                    continue
                if _held_by_live_turn(owner):
                    held.append(owner)
                    continue
                oldest = owner
                break
            if oldest is None:
                # Every remaining feature is pinned or held by the live
                # turn — eviction is impossible. Log and bail so we don't
                # loop forever. A held owner is evictable again by the
                # first registration after its turn ends; a pinned one never.
                logging.warning(
                    "[DYNAMIC-TOOLS] Cannot evict: all %d explored owners "
                    "are pinned or hold a tool this turn used (pinned: %s; "
                    "held by this turn: %s). Direct-tool count %d exceeds "
                    "cap %d.",
                    len(self._explored_features),
                    sorted(pinned),
                    sorted(held),
                    len(self._direct_tools),
                    self.MAX_DIRECT_TOOLS,
                )
                return
            del self._explored_features[oldest]
            to_remove = [k for k, v in self._tool_to_feature.items() if v == oldest]
            for name in to_remove:
                del self._direct_tools[name]
                del self._tool_to_feature[name]
            self._direct_tool_defs = [
                d for d in self._direct_tool_defs
                if d["function"]["name"] not in to_remove
            ]
            bump_registry_generation(self)
            # Log the actual evicted names, not just a count (#1580 /
            # D). Without names, regressions are impossible to spot
            # in retro audit.
            preview = sorted(to_remove)[:10]
            tail = (
                f" (+{len(to_remove) - 10} more)"
                if len(to_remove) > 10 else ""
            )
            if held:
                # LRU would have taken a held owner. What went instead may be
                # the owner being registered right now: say so out loud.
                logging.warning(
                    "[DYNAMIC-TOOLS] Evicted %d tool(s) from %s in place of "
                    "owner(s) holding a tool this turn used (%s): %s%s",
                    len(to_remove), oldest, ", ".join(held),
                    ", ".join(preview), tail,
                )
            else:
                logging.info(
                    "[DYNAMIC-TOOLS] Evicted %d tool(s) from %s: %s%s",
                    len(to_remove), oldest, ", ".join(preview), tail,
                )

    def _build_features_prompt_section(self) -> str:
        """
        Build a dynamic system prompt section describing loaded features.

        This informs the LLM about what features/subagents are available,
        their capabilities, and the commands they provide.

        Returns:
            Formatted string describing loaded features
        """
        if not self.features:
            return ""

        feature_sections = []

        hidden_tools = self._hidden_context_tools()
        for feature in self.features.values():
            try:
                if not self._feature_visible_to_llm(feature):
                    continue
                # Feature name and description
                feature_sections.append(f"\n### {feature.name}")
                feature_sections.append(f"**Capabilities:** {feature.tool_description}")

                # List the feature's tools/commands
                tools = feature.get_tools()
                if tools:
                    feature_sections.append("\n**Available commands:**")
                    for tool in tools:
                        if tool.name in hidden_tools:
                            continue
                        cmd_prefix = tool.schema.command_prefix or ""
                        if cmd_prefix:
                            feature_sections.append(f"- `{cmd_prefix}` - {tool.schema.description}")
                        else:
                            feature_sections.append(f"- {tool.name}: {tool.schema.description}")
            except (AttributeError, TypeError, KeyError) as e:
                logging.warning(f"Failed to build prompt section for feature {feature.name}: {e}")
            except Exception as e:
                logging.warning(f"Failed to build prompt section for feature {feature.name}: {e}", exc_info=True)

        if not feature_sections:
            return ""

        sections = ["\n\n## LOADED FEATURES (Active Subagents)\n"]
        sections.append("These are your ACTIVE subagents. They are loaded and ready to use RIGHT NOW:\n")
        sections.extend(feature_sections)
        sections.append("\n\n**CRITICAL:** When asked about your subagents, capabilities, or available tools, LIST the features above by name. They ARE your active subagents. Never say 'no active subagents' - that is incorrect.")
        return "\n".join(sections)

    def _feature_supports_subagent_dispatch(self, feature: Any) -> bool:
        return (
            getattr(feature, "supports_subagent_dispatch", True) is True
            and callable(getattr(feature, "to_orchestrator_tool", None))
            and callable(getattr(feature, "execute_as_subagent", None))
        )

    def _feature_visible_to_llm(self, feature: Any) -> bool:
        """Whether a loaded feature is currently exposed to the orchestrator LLM.

        Three gates, all required:

        * **enabled** — a feature soft-disabled through the ``/disable`` endpoint
          stays LOADED in ``self.features`` (so ``/enable`` can restore the same
          instance) with ``enabled=False``, but it must NOT appear as a callable
          dispatcher tool, in the features prompt, or in the visible-subagent map
          while disabled (kestrel-sovereign#2522 P1). ``getattr(..., True)``
          keeps features that never set the flag (the normal booted state)
          visible.
        * **not hidden by the active tool-context profile** — progressive
          disclosure can hide a feature for the current turn.
        * **supports subagent dispatch** — the feature has not opted out of
          model-facing execution and exposes both ``to_orchestrator_tool`` and
          ``execute_as_subagent``. HTTP-only services can remain enabled without
          advertising an executable capability they do not implement.

        One source of truth shared by :meth:`_build_feature_tools`,
        :meth:`_visible_features_by_tool_name`, and
        :meth:`_build_features_prompt_section`, so the disabled/hidden gate can
        never drift between the tool list, the subagent map, and the prompt.
        """
        return (
            bool(getattr(feature, "enabled", True))
            and not self._feature_hidden_from_context(feature)
            and self._feature_supports_subagent_dispatch(feature)
        )

    def _hidden_context_features(self) -> set[str]:
        hidden = getattr(self, "_tool_context_hidden_features", set())
        return {str(item) for item in hidden}

    def _hidden_context_tools(self) -> set[str]:
        hidden = getattr(self, "_tool_context_hidden_tools", set())
        return {str(item) for item in hidden}

    def _feature_hidden_from_context(self, feature: Any) -> bool:
        hidden = self._hidden_context_features()
        return feature.tool_name in hidden or feature.name in hidden

    def _direct_tool_hidden_from_context(
        self,
        tool_def: Dict[str, Any],
        *,
        hidden_tools: set[str],
        hidden_features: set[str],
    ) -> bool:
        name = tool_def.get("function", {}).get("name")
        if not isinstance(name, str):
            return False
        if name in hidden_tools:
            return True
        feature_name = self._tool_to_feature.get(name)
        return feature_name in hidden_features
