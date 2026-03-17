"""
Command Handler for Kestrel Agent.

Extracts command parsing and dispatch logic from the main agent,
making the agent class cleaner and commands easier to test.
"""

import os
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from kestrel_sovereign.privacy import PrivacyMode

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Categories of commands for organization and permissions."""
    SYSTEM = "system"           # Status, help, audit
    PRIVACY = "privacy"         # Privacy mode controls
    SOVEREIGNTY = "sovereignty" # Export/import sovereignty
    MODEL = "model"             # Model management
    BACKUP = "backup"           # Backup and restore
    EXTENSION = "extension"     # App-specific extensions


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    
    @staticmethod
    def ok(message: str, data: Dict[str, Any] = None) -> "CommandResult":
        return CommandResult(success=True, message=message, data=data)
    
    @staticmethod
    def error(message: str) -> "CommandResult":
        return CommandResult(success=False, message=f"❌ {message}")
    
    @staticmethod
    def usage(expected: str) -> "CommandResult":
        return CommandResult(success=False, message=f"Usage: {expected}")


class CommandHandler:
    """
    Handles command parsing and dispatch for the Kestrel agent.

    Commands are prefixed with '!' and can be:
    - Built-in commands (handled directly by this class)
    - Feature commands (delegated via A2A TaskManager)
    """

    def __init__(self, agent, task_manager=None):
        """
        Initialize with reference to the agent and optional TaskManager.

        Args:
            agent: The KestrelAgent instance
            task_manager: Optional TaskManager for A2A routing (if None, uses agent.task_manager)
        """
        self.agent = agent
        self._task_manager = task_manager
        self._command_handlers: Dict[str, Callable] = {}
        self._register_builtin_commands()

    @property
    def task_manager(self):
        """Get the TaskManager, preferring the one passed at init."""
        if self._task_manager:
            return self._task_manager
        return getattr(self.agent, 'task_manager', None)
    
    def _register_builtin_commands(self):
        """Register built-in command handlers.

        Only commands that are NOT handled by features should be here.
        Feature-based commands are automatically discovered via A2A TaskManager.
        """
        # System commands
        self._command_handlers["!status"] = self._cmd_status
        self._command_handlers["!help"] = self._cmd_help
        self._command_handlers["!audit"] = self._cmd_audit
        
        # Constitution/integrity commands
        self._command_handlers["!verify-constitution"] = self._cmd_verify_constitution
        self._command_handlers["!safe-mode"] = self._cmd_safe_mode
        # !constitution handled by ConstitutionFeature via tool registry
        
        # Privacy session management (not in PrivacyAgent feature)
        self._command_handlers["!privacy"] = self._cmd_privacy
        self._command_handlers["!set-privacy-mode"] = self._cmd_privacy
        self._command_handlers["!get-privacy-mode"] = self._cmd_get_privacy_mode
        self._command_handlers["!privacy-status"] = self._cmd_privacy_status
        self._command_handlers["!privacy-save"] = self._cmd_privacy_save
        self._command_handlers["!privacy-discard"] = self._cmd_privacy_discard
        
        # Backup commands (agent-level, not feature-level)
        self._command_handlers["!backup"] = self._cmd_backup
        self._command_handlers["!promote-backup"] = self._cmd_promote_backup

        # Sleep/consolidation commands (agent-level)
        self._command_handlers["!sleep"] = self._cmd_sleep
        self._command_handlers["!consolidate"] = self._cmd_consolidate
        self._command_handlers["!compress"] = self._cmd_compress
        
        # Sovereignty commands handled by SovereigntyFeature via tool registry
        # !export-sovereignty, !import-sovereignty, !sovereignty-status, etc.
        
        # Agent creation (agent-level)
        self._command_handlers["!create-agent"] = self._cmd_create_agent
        self._command_handlers["!anchor"] = self._cmd_anchor

        # Model commands (!model, !model-set, !model-list, !model-pull, !model-info)
        # All handled by ModelAgent feature via tool registry

        # App context
        self._command_handlers["!set-app-context"] = self._cmd_set_app_context
        self._command_handlers["!legacy-echo"] = self._cmd_legacy_echo

        # Task management
        self._command_handlers["!tasks"] = self._cmd_tasks
        
        # Continue from stopped request
        self._command_handlers["!continue"] = self._cmd_continue

        # Context reload
        self._command_handlers["!reload-context"] = self._cmd_reload_context

        # Heartbeat
        self._command_handlers["!heartbeat"] = self._cmd_heartbeat
    
    async def handle(self, user_input: str) -> Optional[str]:
        """
        Handle a command input.

        Args:
            user_input: The raw user input starting with '!'

        Returns:
            Command result string, or None if command not recognized
        """
        parts = user_input.strip().split()
        if not parts:
            return None

        command = parts[0].lower()

        # Check built-in handlers
        if command in self._command_handlers:
            handler = self._command_handlers[command]
            result = handler(user_input)
            # Handle both sync and async handlers
            if hasattr(result, '__await__'):
                return await result
            return result

        # Try A2A TaskManager for feature-based commands
        if self.task_manager:
            task_result = await self.task_manager.execute_command(user_input)
            if task_result:
                if task_result.get("success"):
                    # Prefer 'message' (used by tools like set_model), fallback to 'result', then str(task_result)
                    result = task_result.get("message") or task_result.get("result") or str(task_result)
                    # Ensure result is a string (some tools return lists or dicts)
                    if not isinstance(result, str):
                        result = self._format_result(result)
                    return result
                else:
                    return f"❌ Error: {task_result.get('error', 'Unknown error')}"

        return None  # Command not recognized

    def _get_feature_commands(self) -> Dict[str, str]:
        """
        Dynamically discover commands from registered features.

        Returns:
            Dict mapping command prefix to description
        """
        commands = {}
        if not self.task_manager:
            return commands

        # Get all registered agents and their skills
        for agent_id, (agent_card, handler) in self.task_manager._agents.items():
            for skill in agent_card.skills:
                # Get the tool from the handler to access command_prefix
                if hasattr(handler, 'get_tools'):
                    for tool in handler.get_tools():
                        if tool.schema.command_prefix:
                            commands[tool.schema.command_prefix] = tool.schema.description

        return commands

    def _format_result(self, result) -> str:
        """
        Format a non-string result into a human-readable string.

        Handles:
        - List[ModelInfo] from list_models()
        - Dict results from various tools
        - Other iterables
        """
        import json
        from kestrel_sovereign.llm.model_metadata import ModelInfo

        # Handle list of ModelInfo objects
        if isinstance(result, list) and result and hasattr(result[0], 'provider'):
            # Group models by provider
            by_provider = {}
            for m in result:
                provider = getattr(m, 'provider', None) or "unknown"
                if provider not in by_provider:
                    by_provider[provider] = []
                by_provider[provider].append(m)

            # Format output
            lines = ["**Available Models:**\n"]
            for provider, provider_models in sorted(by_provider.items()):
                lines.append(f"\n**{provider.upper()}**")
                for m in sorted(provider_models, key=lambda x: getattr(x, 'display_name', None) or getattr(x, 'id', '')):
                    star = "★ " if getattr(m, 'is_featured', False) else "  "
                    name = getattr(m, 'display_name', None) or getattr(m, 'id', str(m))
                    lines.append(f"{star}{name}")

            lines.append(f"\n\n_Total: {len(result)} models across {len(by_provider)} providers_")
            return "\n".join(lines)

        # Handle dict
        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)

        # Handle list of other objects
        if isinstance(result, list):
            return "\n".join(str(item) for item in result)

        # Fallback
        return str(result)

    # === System Commands ===
    
    def _cmd_status(self, user_input: str) -> str:
        """Handle !status command."""
        return f"Agent ID: {self.agent.agent_id}\n{self.agent.privacy_agent.get_status()}"
    
    def _cmd_help(self, user_input: str) -> str:
        """Handle !help command - dynamically generates help from registered features."""
        lines = [
            "Kestrel Agent Commands",
            "======================",
            "",
            "System:",
            "  !status              - Show agent status",
            "  !help                - Show this help",
            "  !audit [on|off]      - Toggle or check audit status",
            "  !reload-context      - Hot-reload bootstrap files from disk",
            "  !heartbeat           - Trigger a manual heartbeat check",
            "",
            "Constitution:",
            "  !verify-constitution - Verify constitution integrity",
            "  !safe-mode [exit]    - Check or exit safe mode",
            "",
            "Privacy:",
            "  !privacy [mode]      - Get or set privacy mode",
            "  !privacy-status      - Detailed privacy status",
            "  !privacy-save        - Save isolated session",
            "  !privacy-discard     - Discard isolated session",
            "",
            "Memory:",
            "  !compress [--keep N] - Compress session (summarize older messages, keep N recent)",
            "  !sleep [--tier ...]  - Consolidate memories + export to sovereignty storage",
            "  !consolidate         - Consolidate memories only (create episodes, archive)",
            "",
            "Backup:",
            "  !backup [--tier local|ipfs|filecoin] [--no-encrypt] - Create backup",
            "  !promote-backup [--tier ...] - Save isolated session and backup",
            "",
            "Tasks:",
            "  !tasks               - List recent background tasks",
            "  !tasks all           - List all tasks",
            "",
            "Other:",
            "  !create-agent <name> - Create trusted agent",
            "  !anchor              - Anchor memory state",
            "  !send-mail <addr> <file> - Send physical mail",
        ]

        # Dynamically add feature commands from TaskManager
        if self.task_manager:
            feature_commands = self._get_feature_commands()
            if feature_commands:
                lines.append("")
                lines.append("Feature Commands (via registered agents):")
                for cmd, desc in sorted(feature_commands.items()):
                    lines.append(f"  {cmd:<20} - {desc}")

        return "\n".join(lines)
    
    def _cmd_audit(self, user_input: str) -> str:
        """Handle !audit command."""
        parts = user_input.split()
        if len(parts) > 1:
            opt = parts[1].lower()
            if opt == "off":
                self.agent.audit_enabled = False
                return "Audit disabled."
            if opt == "on":
                self.agent.audit_enabled = True
                return "Audit enabled."
        return f"Audit is {'enabled' if self.agent.audit_enabled else 'disabled'}."
    
    # === Constitution Commands ===
    
    def _cmd_verify_constitution(self, user_input: str) -> str:
        """Handle !verify-constitution command."""
        is_valid, message = self.agent._verify_constitution_integrity()
        self.agent._constitution_verified = is_valid
        if is_valid:
            return f"✅ {message}"
        else:
            self.agent.enter_safe_mode(message)
            return f"🚨 {message}\n\nAgent has entered SAFE MODE. Contact administrator."
    
    def _cmd_safe_mode(self, user_input: str) -> str:
        """Handle !safe-mode command."""
        parts = user_input.split()
        if len(parts) > 1 and parts[1].lower() == "exit":
            return self.agent.exit_safe_mode(authorization="user_command")
        if self.agent._safe_mode:
            return "🚨 SAFE MODE ACTIVE: Agent functionality restricted due to integrity failure."
        return "✅ Normal operation mode. No integrity issues detected."
    
    # !constitution command now handled by ConstitutionFeature via tool registry
    
    # === Privacy Commands ===
    
    def _cmd_privacy(self, user_input: str) -> str:
        """Handle !privacy and !set-privacy-mode commands."""
        parts = user_input.split()
        if len(parts) > 1:
            try:
                mode = PrivacyMode(parts[1].lower())
                return self.agent.privacy_agent.set_mode(mode)
            except ValueError:
                valid_modes = ", ".join([m.value for m in PrivacyMode])
                return f"Invalid privacy mode. Valid modes are: {valid_modes}"
        return self.agent.privacy_agent.get_status()
    
    def _cmd_get_privacy_mode(self, user_input: str) -> str:
        """Handle !get-privacy-mode command."""
        mode = self.agent.privacy_agent.privacy_mode
        mode_info = {
            PrivacyMode.EPHEMERAL: ("🔒", "EPHEMERAL: Nothing stored, local LLM only"),
            PrivacyMode.ISOLATED: ("🔐", "ISOLATED: Temporary session storage, local LLM only"),
            PrivacyMode.ANONYMOUS: ("🎭", "ANONYMOUS: Stored with PII removed, cloud LLM allowed"),
            PrivacyMode.NORMAL: ("📝", "NORMAL: Standard persistent storage"),
            PrivacyMode.PUBLIC: ("🌐", "PUBLIC: Shareable and exportable"),
        }
        icon, description = mode_info.get(mode, ("", f"Current mode: {mode.value}"))
        return f"{icon} {description}"
    
    def _cmd_privacy_status(self, user_input: str) -> str:
        """Handle !privacy-status command."""
        status = self.agent.privacy_agent.get_detailed_status()
        return f"""
Privacy Status Report
=====================
Current Mode: {status['privacy_mode'].upper()}

Storage:
- Messages stored: {status['message_count']}
- Storage location: {status['storage_location']}
- Persistent: {status['persistent_storage']}
- PII filtering: {'Enabled' if status['pii_filtering'] else 'Disabled'}

LLM Providers:
- Local (Ollama): {'Allowed' if status['llm_providers']['local_ollama'] else 'Disabled'}
- Cloud (OpenAI): {'Allowed' if status['llm_providers']['cloud_openai'] else 'Disabled'}
- Cloud (Anthropic): {'Allowed' if status['llm_providers']['cloud_anthropic'] else 'Disabled'}

Backups:
- Status: {status['backup_status']}
- Encryption: {status['backup_encryption']}

Sharing:
- Can share: {status['shareable']}
"""
    
    def _cmd_privacy_save(self, user_input: str) -> str:
        """Handle !privacy-save command."""
        return self.agent.privacy_agent.save_isolated_session()
    
    def _cmd_privacy_discard(self, user_input: str) -> str:
        """Handle !privacy-discard command."""
        return self.agent.privacy_agent.discard_isolated_session()
    
    # === Backup Commands ===
    
    async def _cmd_backup(self, user_input: str) -> str:
        """Handle !backup command."""
        return await self.agent._command_backup(user_input)
    
    async def _cmd_promote_backup(self, user_input: str) -> str:
        """Handle !promote-backup command."""
        return await self.agent._command_promote_backup(user_input)

    # === Sleep/Consolidation Commands ===

    async def _cmd_sleep(self, user_input: str) -> str:
        """
        Handle !sleep command - full memory consolidation + sovereignty export.

        Usage:
            !sleep                    - Full sleep (consolidate + export to IPFS)
            !sleep --tier local       - Export to local only
            !sleep --tier filecoin    - Export to Filecoin for permanent storage
            !sleep --consolidate-only - Only run memory consolidation
            !sleep --export-only      - Only run sovereignty export
        """
        return await self.agent._command_sleep(user_input)

    async def _cmd_consolidate(self, user_input: str) -> str:
        """
        Handle !consolidate command - memory consolidation only.

        This is a shortcut for !sleep --consolidate-only.
        Creates episodes, detects patterns, archives decayed memories.
        """
        return await self.agent._command_sleep(user_input + " --consolidate-only")

    async def _cmd_compress(self, user_input: str) -> str:
        """
        Handle !compress command - in-session context compression.

        Summarizes older messages to free up context window space
        while preserving recent messages verbatim.

        Usage:
            !compress             - Compress with default settings (keep 10 recent)
            !compress --keep 20   - Keep 20 most recent messages
            !compress --force     - Force compression even if not needed
            !compress --check     - Check if compression is recommended
        """
        import re

        parts = user_input.split()

        # Parse --keep N
        preserve_recent = 10  # default
        keep_match = re.search(r'--keep\s+(\d+)', user_input)
        if keep_match:
            preserve_recent = int(keep_match.group(1))

        force = '--force' in user_input
        check_only = '--check' in user_input

        # Check if context_manager exists
        if not hasattr(self.agent, 'context_manager') or not self.agent.context_manager:
            return "❌ Context manager not available"

        # Check-only mode
        if check_only:
            result = await self.agent.context_manager.check_compression_needed()
            status = "✅ Recommended" if result["compression_recommended"] else "ℹ️ Not needed"
            return (
                f"{status}\n"
                f"  Utilization: {result['utilization_percent']:.1f}% (threshold: {result['threshold']}%)\n"
                f"  Messages: {result['message_count']}\n"
                f"  Tokens: {result['total_tokens']:,} / {result['budget_limit']:,}"
            )

        # Check if llm_service exists
        if not hasattr(self.agent, 'llm_service') or not self.agent.llm_service:
            return "❌ LLM service not available for compression"

        # Perform compression
        result = await self.agent.context_manager.compress_session(
            llm_service=self.agent.llm_service,
            preserve_recent=preserve_recent,
            force=force
        )

        if result["success"]:
            return (
                f"✅ Session compressed\n"
                f"  Messages compressed: {result['messages_compressed']}\n"
                f"  Messages preserved: {result['messages_preserved']}\n"
                f"  Tokens saved: {result['tokens_saved']:,} ({result['tokens_before']:,} → {result['tokens_after']:,})\n"
                f"  Summary preview: {result['summary_preview']}"
            )
        else:
            return f"ℹ️ {result.get('reason', 'Compression not performed')}"

    # Sovereignty commands now handled by SovereigntyFeature via tool registry
    # !export-sovereignty, !import-sovereignty, !sovereignty-status, etc.
    
    # === Agent/Memory Commands ===
    
    def _cmd_create_agent(self, user_input: str) -> str:
        """Handle !create-agent command."""
        parts = user_input.split()
        if len(parts) > 1:
            agent_name = parts[1]
            return self.agent.create_trusted_agent(agent_name)
        return "Usage: !create-agent <agent_name>"
    
    def _cmd_anchor(self, user_input: str) -> str:
        """Handle !anchor command."""
        return self.agent.anchor_memory_state()

    # Model commands (!model, !model-set, !model-list, !model-pull, !model-info)
    # All handled by ModelAgent feature via tool registry

    # === App Context Commands ===
    
    def _cmd_set_app_context(self, user_input: str) -> str:
        """Handle !set-app-context command."""
        parts = user_input.split()
        if len(parts) > 1:
            self.agent.app_context = parts[1]
            return f"Application context set to: {self.agent.app_context}"
        return "Usage: !set-app-context <app_name>"
    
    def _cmd_legacy_echo(self, user_input: str) -> str:
        """Handle !legacy-echo command."""
        if not self.agent.extension:
            return "No application extension loaded."
        return self.agent.extension.handle_legacy_echo()

    # === Task Management Commands ===

    async def _cmd_tasks(self, user_input: str) -> str:
        """
        Handle !tasks command - list background tasks.

        Usage:
            !tasks              - List recent tasks (default: 10)
            !tasks all          - List all tasks
            !tasks completed    - List only completed tasks
            !tasks working      - List only in-progress tasks
            !tasks failed       - List only failed tasks
            !tasks 20           - List 20 most recent tasks
        """
        if not self.task_manager:
            return "❌ Task manager not available"

        parts = user_input.split()
        limit = 10
        status_filter = None

        # Parse arguments
        if len(parts) > 1:
            arg = parts[1].lower()
            if arg == "all":
                limit = 1000
            elif arg in ("completed", "working", "failed", "submitted", "canceled"):
                status_filter = arg.upper()
            elif arg.isdigit():
                limit = int(arg)

        try:
            # Get tasks from store
            tasks = await self.task_manager.task_store.list_tasks(limit=limit)

            if not tasks:
                return "📋 No tasks found"

            # Filter by status if specified
            if status_filter:
                from kestrel_sovereign.a2a.types import TaskState
                try:
                    filter_state = TaskState[status_filter]
                    tasks = [t for t in tasks if t.status.state == filter_state]
                except KeyError:
                    pass

            if not tasks:
                return f"📋 No {status_filter.lower()} tasks found"

            # Format output
            lines = ["📋 Background Tasks", "=" * 40]

            # State emoji mapping
            state_emoji = {
                "COMPLETED": "✅",
                "FAILED": "❌",
                "WORKING": "🔄",
                "SUBMITTED": "📥",
                "CANCELED": "⚫",
                "INPUT_REQUIRED": "❓",
            }

            for task in tasks:
                state = task.status.state.value
                emoji = state_emoji.get(state, "❓")
                task_id = task.id[:8]

                # Get description from metadata
                agent_id = task.metadata.get("agent_id", "unknown") if task.metadata else "unknown"
                skill_id = task.metadata.get("skill", "") if task.metadata else ""

                # Build task line
                desc = f"{agent_id}/{skill_id}" if skill_id else agent_id
                lines.append(f"{emoji} [{task_id}] {desc} - {state}")

            lines.append(f"\nShowing {len(tasks)} task(s)")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return f"❌ Error listing tasks: {e}"

    async def _cmd_continue(self, user_input: str) -> str:
        """
        Handle !continue command - continue from where a stopped request left off.

        This sends a continuation prompt to the LLM to resume the previous response.
        """
        # This is a special command - we return None to let the agent handle it
        # by sending a continuation request to the LLM
        return None  # Signal to process_input to continue the conversation

    def _cmd_reload_context(self, user_input: str) -> str:
        """Handle !reload-context — hot-reload bootstrap files from disk."""
        try:
            builder = getattr(self.agent, 'context_builder', None)
            if builder is None:
                cm = getattr(self.agent, 'context_manager', None)
                if cm:
                    builder = getattr(cm, 'context_builder', None)
            if builder and hasattr(builder, 'reload_bootstrap_files'):
                builder.reload_bootstrap_files()
                loaded = list(builder._bootstrap_files.keys())
                if loaded:
                    return f"Bootstrap files reloaded: {', '.join(loaded)}"
                return "No bootstrap files found in agent data directory."
            return "Context builder not available."
        except Exception as e:
            return f"Error reloading context: {e}"

    async def _cmd_heartbeat(self, user_input: str) -> str:
        """Handle !heartbeat — trigger a manual heartbeat check."""
        runner = getattr(self.agent, 'heartbeat_runner', None)
        if not runner:
            return "Heartbeat not configured. Add [heartbeat] section to kestrel.toml."
        try:
            result = await runner.run_once()
            if result.status == "ok":
                return f"Heartbeat OK ({result.duration_ms}ms)"
            elif result.status == "alert":
                return f"Heartbeat ALERT ({result.duration_ms}ms):\n{result.message}"
            elif result.status == "skipped":
                return f"Heartbeat skipped: {result.reason}"
            else:
                return f"Heartbeat error: {result.reason or result.message}"
        except Exception as e:
            return f"Heartbeat failed: {e}"
