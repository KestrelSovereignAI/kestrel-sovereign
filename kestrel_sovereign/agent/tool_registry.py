"""
Dynamic tool registry mixin for KestrelAgent.

Extracted from kestrel_agent.py — manages the tool catalog exposed to the
orchestrator LLM, including feature dispatcher tools and directly-promoted
individual tools with LRU eviction.
"""

import logging
from typing import Any, Dict, List

from kestrel_sovereign.tools.result_contract import enforce_tool_result_contract


class ToolRegistryMixin:
    """Mixin providing dynamic tool loading and management for KestrelAgent."""

    MAX_DIRECT_TOOLS = 60

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

    def _register_explored_feature_tools(self, feature) -> None:
        """Register a feature's individual tools for direct calling.

        After a successful subagent dispatch, the feature's @tool methods
        become available for the orchestrator to call directly without
        a subagent LLM hop.
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
        self._explored_features[feature.tool_name] = True
        registered = 0
        for tool in feature.get_tools():
            if tool.name in self._direct_tools:
                name = f"{feature.tool_name}__{tool.name}"
            else:
                name = tool.name
            self._direct_tools[name] = tool
            tool_def = tool.schema.to_openai_format()
            tool_def["function"]["name"] = name
            self._direct_tool_defs.append(tool_def)
            self._tool_to_feature[name] = feature.tool_name
            registered += 1
        self._maybe_evict_direct_tools()
        logging.info(
            f"[DYNAMIC-TOOLS] Explored {feature.tool_name}, "
            f"registered {registered} direct tools. "
            f"Total: {len(self._direct_tools)}"
        )

    def _maybe_evict_direct_tools(self) -> None:
        """Evict least-recently-explored feature's tools if over limit."""
        while len(self._direct_tools) > self.MAX_DIRECT_TOOLS:
            oldest = next(iter(self._explored_features))
            del self._explored_features[oldest]
            to_remove = [k for k, v in self._tool_to_feature.items() if v == oldest]
            for name in to_remove:
                del self._direct_tools[name]
                del self._tool_to_feature[name]
            self._direct_tool_defs = [
                d for d in self._direct_tool_defs
                if d["function"]["name"] not in to_remove
            ]
            logging.info(f"[DYNAMIC-TOOLS] Evicted {len(to_remove)} tools from {oldest}")

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
