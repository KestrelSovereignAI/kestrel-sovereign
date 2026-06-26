"""
Dynamic tool registry mixin for KestrelAgent.

Extracted from kestrel_agent.py — manages the tool catalog exposed to the
orchestrator LLM, including feature dispatcher tools and directly-promoted
individual tools with LRU eviction.
"""

import logging
import re
from typing import Any, Dict, List

from kestrel_sovereign.tools.result_contract import enforce_tool_result_contract


class ToolRegistryMixin:
    """Mixin providing dynamic tool loading and management for KestrelAgent."""

    # Startup-promoted operational features currently pin 44 tools
    # (Task, Todo, Peers, Save, Spawn, StrategicMemory). Keep enough
    # headroom for the common model_agent + memory_feature exploration
    # path without immediately evicting the first explored feature.
    MAX_DIRECT_TOOLS = 80

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
                if self._feature_hidden_from_context(feature):
                    continue
                if not self._feature_supports_subagent_dispatch(feature):
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
            if not self._feature_hidden_from_context(feature)
            and self._feature_supports_subagent_dispatch(feature)
        }

    def _visible_known_tool_names(self) -> set[str]:
        """Return tool names the LLM may call under the active context profile."""
        return {
            tool_def["function"]["name"]
            for tool_def in self._build_all_tools()
            if isinstance(tool_def.get("function", {}).get("name"), str)
        }

    def register_dynamic_tools(self, owner: str, tools, *, pin: bool = False) -> int:
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
        sanitised to a schema-safe token first). Returns the number registered.

        ``owner`` is recorded in ``_explored_features`` so eviction and
        :meth:`unregister_dynamic_tools` treat feature- and non-feature owners
        uniformly. ``pin=True`` exempts the owner from LRU eviction.
        """
        safe_owner = re.sub(r"\W+", "_", owner)
        self._explored_features[owner] = True
        if pin:
            self._pinned_features.add(owner)
        registered = 0
        for tool in tools:
            name = tool.name
            if name in self._direct_tools:
                name = f"{safe_owner}__{tool.name}"
            # Guarantee the final name is unique even when the prefixed name
            # also collides (e.g. two owners that sanitise to the same token,
            # 'mcp:foo-bar' and 'mcp:foo_bar', each exposing 'search'). Without
            # this, the second registration would overwrite the first and leave
            # duplicate schema/owner bookkeeping.
            if name in self._direct_tools:
                base, suffix = name, 2
                while name in self._direct_tools:
                    name = f"{base}_{suffix}"
                    suffix += 1
            self._direct_tools[name] = tool
            tool_def = tool.schema.to_openai_format()
            tool_def["function"]["name"] = name
            self._direct_tool_defs.append(tool_def)
            self._tool_to_feature[name] = owner
            registered += 1
        self._maybe_evict_direct_tools()
        logging.info(
            f"[DYNAMIC-TOOLS] Registered {registered} direct tools for "
            f"'{owner}'. Total: {len(self._direct_tools)}"
        )
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
        self._explored_features.pop(owner, None)
        self._pinned_features.discard(owner)
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
        """
        pinned = getattr(self, "_pinned_features", set())
        while len(self._direct_tools) > self.MAX_DIRECT_TOOLS:
            # Find oldest UNPINNED feature.
            oldest = next(
                (k for k in self._explored_features if k not in pinned),
                None,
            )
            if oldest is None:
                # Every remaining feature is pinned — eviction is
                # impossible. Log and bail so we don't loop forever.
                logging.warning(
                    "[DYNAMIC-TOOLS] Cannot evict: all %d explored "
                    "features are pinned (%s). Direct-tool count %d "
                    "exceeds cap %d; raise MAX_DIRECT_TOOLS or unpin.",
                    len(self._explored_features),
                    sorted(pinned), len(self._direct_tools),
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
            # Log the actual evicted names, not just a count (#1580 /
            # D). Without names, regressions are impossible to spot
            # in retro audit.
            preview = sorted(to_remove)[:10]
            tail = (
                f" (+{len(to_remove) - 10} more)"
                if len(to_remove) > 10 else ""
            )
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
                if self._feature_hidden_from_context(feature):
                    continue
                if not self._feature_supports_subagent_dispatch(feature):
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
            callable(getattr(feature, "to_orchestrator_tool", None))
            and callable(getattr(feature, "execute_as_subagent", None))
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
