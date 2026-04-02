import re
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone

from kestrel_sovereign.privacy import PrivacyMode, PrivacyConfig, PRIVACY_PRESETS, get_privacy_preset
from kestrel_sovereign.kestrel_types.storage_types import StorageProvider
from kestrel_sovereign.ephemeral_session import EphemeralSession
from .pii_detector import get_pii_detector, anonymize_text


class PrivacyAgent:
    """
    A Feature Agent that manages the Kestrel agent's privacy modes and conversation history.
    
    Now uses PrivacyConfig with independent flags for storage, llm_location, and shareable.
    Presets (ephemeral, isolated, anonymous, normal, public) provide named combinations.
    """

    def __init__(self, storage: StorageProvider, initial_mode: Union[PrivacyMode, PrivacyConfig, str] = PrivacyMode.NORMAL):
        self.storage = storage
        
        # Convert to PrivacyConfig internally
        self._privacy_config = self._to_config(initial_mode)
        
        self.isolated_session: List[Dict] = []
        self.ephemeral_session: Optional[EphemeralSession] = None

        # Initialize ephemeral session if starting in ephemeral mode
        if self._privacy_config.is_ephemeral():
            self.ephemeral_session = EphemeralSession()

    def _to_config(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> PrivacyConfig:
        """Convert various mode representations to PrivacyConfig."""
        if isinstance(mode, PrivacyConfig):
            return mode
        elif isinstance(mode, PrivacyMode):
            return mode.to_config()
        elif isinstance(mode, str):
            return get_privacy_preset(mode)
        else:
            raise TypeError(f"Expected PrivacyMode, PrivacyConfig, or str, got {type(mode)}")

    @property
    def privacy_mode(self) -> PrivacyMode:
        """Backward compatibility: return PrivacyMode enum."""
        return PrivacyMode.from_config(self._privacy_config)
    
    @property
    def privacy_config(self) -> PrivacyConfig:
        """Get the current privacy configuration."""
        return self._privacy_config

    def set_mode(self, mode: Union[PrivacyMode, PrivacyConfig, str]) -> str:
        """Sets the privacy mode/config for future conversations."""
        old_config = self._privacy_config
        new_config = self._to_config(mode)

        # Handle transitions that require warnings
        if old_config.shareable and new_config.is_ephemeral():
            return "WARNING: Switching from PUBLIC to EPHEMERAL will prevent storage of future messages. Previous PUBLIC messages remain stored. Use !confirm-privacy-mode ephemeral to confirm."

        # Handle ephemeral session lifecycle
        if new_config.is_ephemeral():
            if self.ephemeral_session is None:
                self.ephemeral_session = EphemeralSession()
            # Clear isolated session if switching from isolated
            if old_config.uses_temp_storage() and self.isolated_session:
                self.isolated_session.clear()
        else:
            # Clear ephemeral session when leaving ephemeral mode
            if self.ephemeral_session is not None:
                self.ephemeral_session.clear()
                self.ephemeral_session = None

        self._privacy_config = new_config
        
        # Get preset name for display
        preset_name = self._get_preset_name(new_config)
        old_preset_name = self._get_preset_name(old_config)

        mode_descriptions = {
            "ephemeral": "EPHEMERAL mode - nothing stored, local LLM only, no memory.",
            "isolated": "ISOLATED mode - temporary session storage, local LLM only.",
            "anonymous": "ANONYMOUS mode - stored with PII removed, cloud LLM allowed.",
            "normal": "NORMAL mode - standard persistence with all features.",
            "public": "PUBLIC mode - can be shared and exported publicly."
        }
        
        description = mode_descriptions.get(preset_name, f"Custom config: {new_config}")

        llm_restrictions = ""
        if not new_config.allows_cloud_llm():
            llm_restrictions = " Note: Only local LLM (Ollama) will be used. Cloud providers disabled."

        return f"Privacy mode changed from {old_preset_name} to {preset_name}. {description}{llm_restrictions}"

    def _get_preset_name(self, config: PrivacyConfig) -> str:
        """Get the preset name for a config, or 'custom' if no match."""
        for name, preset in PRIVACY_PRESETS.items():
            if (config.storage == preset.storage and 
                config.llm_location == preset.llm_location and
                config.shareable == preset.shareable):
                return name
        return "custom"

    def get_status(self) -> str:
        """Gets the current privacy mode and session status."""
        preset_name = self._get_preset_name(self._privacy_config)
        status = f"Current privacy mode: {preset_name}"
        
        if self._privacy_config.uses_temp_storage():
            status += f" (session has {len(self.isolated_session)} messages)"
        elif self._privacy_config.is_ephemeral():
            if self.ephemeral_session:
                stats = self.ephemeral_session.get_stats()
                status += f" (in-memory buffer has {stats['message_count']} messages)"
            else:
                status += " (no active ephemeral session)"
        return status

    def get_detailed_status(self) -> Dict:
        """Gets detailed privacy status and storage information."""
        config = self._privacy_config
        preset_name = self._get_preset_name(config)

        # Get storage statistics
        message_count = 0
        storage_size_mb = 0.0

        if config.is_ephemeral():
            if self.ephemeral_session:
                stats = self.ephemeral_session.get_stats()
                message_count = stats['message_count']
            storage_location = "in-memory only"
            persistent = False
        elif config.uses_temp_storage():
            message_count = len(self.isolated_session)
            storage_location = "temporary session"
            persistent = False
        else:
            # For persistent modes - query actual storage
            try:
                history = self.storage.get_conversation_history(limit=10000)
                message_count = len(history)
            except Exception:
                message_count = 0
            storage_location = "persistent database"
            persistent = True

        # Determine LLM provider restrictions based on config flags
        llm_providers = {
            "local_ollama": True,  # Always allowed - data stays local
            "cloud_openai": config.allows_cloud_llm(),
            "cloud_anthropic": config.allows_cloud_llm()
        }

        # Backup settings
        backup_status = "disabled" if config.is_ephemeral() else "enabled"
        backup_encryption = "required" if config.requires_anonymization() else "optional"

        return {
            "privacy_mode": preset_name,
            "privacy_config": {
                "storage": config.storage,
                "llm_location": config.llm_location,
                "shareable": config.shareable
            },
            "message_count": message_count,
            "storage_location": storage_location,
            "persistent_storage": persistent,
            "storage_size_mb": storage_size_mb,
            "llm_providers": llm_providers,
            "backup_status": backup_status,
            "backup_encryption": backup_encryption,
            "pii_filtering": config.requires_anonymization(),
            "shareable": config.shareable
        }

    async def save_isolated_session(self) -> str:
        """Saves the isolated session to permanent storage."""
        if not self._privacy_config.uses_temp_storage():
            return "Error: Not in isolated mode."

        if not self.isolated_session:
            return "No isolated session to save."

        saved_count = 0
        for entry in self.isolated_session:
            await self.storage.add_conversation(entry['role'], entry['content'], entry.get('metadata'))
            saved_count += 1

        self.isolated_session.clear()
        return f"Saved {saved_count} messages from isolated session to permanent storage."

    def discard_isolated_session(self) -> str:
        """Discards the isolated session without saving."""
        if not self._privacy_config.uses_temp_storage():
            return "Error: Not in isolated mode."

        if not self.isolated_session:
            return "No isolated session to discard."

        discard_count = len(self.isolated_session)
        self.isolated_session.clear()
        return f"Discarded {discard_count} messages from isolated session."

    async def add_conversation(self, role: str, content: str, metadata: Optional[Dict] = None,
                               session_id: Optional[str] = None):
        """
        Adds a conversation entry according to the current privacy config.
        This is the central method for enforcing privacy rules.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata dict
            session_id: If provided, link this message to a specific session.
                       This allows resuming old conversations beyond the 30-min gap.
        """
        config = self._privacy_config

        if config.is_ephemeral():
            # Store in ephemeral session only (in-memory)
            if self.ephemeral_session is None:
                self.ephemeral_session = EphemeralSession()
            self.ephemeral_session.add_message(role, content, metadata)
            return  # Do NOT persist to storage

        if config.uses_temp_storage():
            self.isolated_session.append({"role": role, "content": content, "metadata": metadata})
            return

        final_content = content
        if config.requires_anonymization():
            final_content = self._anonymize_text(content)

        # Add privacy mode to metadata for tracking
        if metadata is None:
            metadata = {}
        metadata["privacy_mode"] = self._get_preset_name(config)
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        await self.storage.add_conversation(role, final_content, metadata, session_id)

    async def get_conversation_history(self, limit: int = 100, session_id: str = None) -> List[Dict]:
        """
        Gets conversation history respecting privacy config.

        Args:
            limit: Maximum number of messages to return
            session_id: Optional session ID to load context from a specific session

        Returns:
            List of conversation messages
        """
        config = self._privacy_config

        if config.is_ephemeral():
            if self.ephemeral_session:
                return self.ephemeral_session.get_history(limit)
            return []

        if config.uses_temp_storage():
            return self.isolated_session[-limit:] if len(self.isolated_session) > limit else self.isolated_session.copy()

        # For persistent modes - get from storage
        return await self.storage.get_conversation_history(limit, session_id=session_id)

    # === Unified Privacy Decision API ===
    # All features should consult these methods instead of making
    # independent privacy decisions based on raw config flags.

    def can_store(self, data_type: str = "conversation") -> bool:
        """
        Central "can I store this?" check for all features.

        Features MUST call this before writing any user data to persistent
        storage. This is the single decision point that replaces ad-hoc
        privacy checks scattered across features.

        Args:
            data_type: Type of data being stored. Supported types:
                - "conversation": Chat messages (default)
                - "file": Files, audio, images
                - "metadata": Non-PII structural metadata (always allowed)
                - "backup": Full backup blobs

        Returns:
            True if persistent storage is allowed for this data type.
        """
        config = self._privacy_config

        # Structural metadata is always allowed (graph nodes, edges, etc.)
        if data_type == "metadata":
            return True

        # Ephemeral: nothing persisted
        if config.is_ephemeral():
            return False

        # Isolated: only temp storage (not persistent)
        if config.uses_temp_storage():
            return False

        # Backups have their own gate
        if data_type == "backup":
            return config.allows_persistent_storage() and not config.is_ephemeral()

        # Normal/anonymous/public: persistent storage allowed
        return config.allows_persistent_storage()

    def can_use_cloud(self) -> bool:
        """
        Check whether cloud services (LLM, TTS, STT) are allowed.

        Features MUST call this before sending data to any cloud provider.
        This is the single decision point for cloud access.

        Returns:
            True if cloud providers are allowed under the current privacy config.
        """
        return self._privacy_config.allows_cloud_llm()

    def get_storage_policy(self) -> str:
        """
        Get the current storage policy string.

        Returns one of: "none", "temp", "scrubbed", "full".
        Features use this to decide HOW to store (not IF — use can_store() for that).
        """
        return self._privacy_config.storage

    def requires_anonymization(self) -> bool:
        """Check if PII scrubbing is required before storage."""
        return self._privacy_config.requires_anonymization()

    def get_mode_name(self) -> str:
        """Get the current privacy preset name (e.g. 'ephemeral', 'normal', 'custom')."""
        return self._get_preset_name(self._privacy_config)

    # === Privacy Command Handlers ===
    # These methods contain the logic for privacy-related commands.
    # CommandHandler delegates to these instead of reimplementing.

    def handle_get_privacy_mode(self) -> str:
        """Format the current privacy mode for display."""
        mode = self.privacy_mode
        mode_info = {
            PrivacyMode.EPHEMERAL: ("\U0001f512", "EPHEMERAL: Nothing stored, local LLM only"),
            PrivacyMode.ISOLATED: ("\U0001f510", "ISOLATED: Temporary session storage, local LLM only"),
            PrivacyMode.ANONYMOUS: ("\U0001f3ad", "ANONYMOUS: Stored with PII removed, cloud LLM allowed"),
            PrivacyMode.NORMAL: ("\U0001f4dd", "NORMAL: Standard persistent storage"),
            PrivacyMode.PUBLIC: ("\U0001f310", "PUBLIC: Shareable and exportable"),
        }
        icon, description = mode_info.get(mode, ("", f"Current mode: {mode.value}"))
        return f"{icon} {description}"

    def handle_privacy_status(self) -> str:
        """Format detailed privacy status for display."""
        status = self.get_detailed_status()
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

    def _anonymize_text(self, text: str) -> str:
        """
        PII redaction function for ANONYMOUS mode.
        Uses NER-based detection via spaCy (if available) with regex fallback.

        Detects:
        - PERSON names via NER (John Smith, Dr. Jane Doe)
        - ORG names via NER (Acme Corp, Bank of America)
        - Locations via NER (New York, California)
        - Email addresses via regex
        - Phone numbers via regex
        - SSN/National IDs via regex
        - Credit card numbers via regex
        - Addresses via regex
        - ZIP codes via regex
        """
        return anonymize_text(text)