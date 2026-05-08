"""
Bootstrap Service for Kestrel agent wake-up and personality discovery.

Manages the first-time experience when a new agent comes online, guiding
them through a discovery conversation to establish their personality.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Prompt file locations
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
DISCOVERY_PROMPT_FILE = PROMPTS_DIR / "discovery_prompt.md"
SOUL_GENERATION_PROMPT_FILE = PROMPTS_DIR / "soul_generation_prompt.md"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
DEFAULT_SOUL_FILE = TEMPLATES_DIR / "default_soul.md"


class BootstrapState(Enum):
    """States for the bootstrap/discovery process."""
    PENDING = "pending"      # Never started - will show wake-up on first message
    DISCOVERY = "discovery"  # In discovery conversation
    AVATAR = "avatar"        # Offering avatar generation
    COMPLETE = "complete"    # Bootstrap finished


class BootstrapService:
    """
    Manages agent wake-up and personality discovery.

    The bootstrap flow:
    1. First message triggers wake-up greeting (PENDING -> DISCOVERY)
    2. Discovery conversation learns about user (2-4 exchanges)
    3. Optional avatar generation offered (DISCOVERY -> AVATAR)
    4. SOUL.md generated and saved (AVATAR -> COMPLETE)
    5. Normal operation begins
    """

    # Metadata keys for agent_metadata table
    BOOTSTRAP_STATE_KEY = "bootstrap_state"
    BOOTSTRAP_STARTED_KEY = "bootstrap_started_at"
    BOOTSTRAP_COMPLETED_KEY = "bootstrap_completed_at"
    DISCOVERY_HISTORY_KEY = "bootstrap_discovery_history"
    USER_NAME_KEY = "bootstrap_user_name"

    def __init__(
        self,
        db,
        agent_id: str,
        agent_name: str,
        llm_service,
        agent_data_path: Path,
    ):
        """
        Initialize the bootstrap service.

        Args:
            db: AsyncDatabase instance
            agent_id: The agent's DID
            agent_name: The agent's current name
            llm_service: LLM service for generating responses
            agent_data_path: Path to agent's data directory (for SOUL.md)
        """
        self.db = db
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.llm_service = llm_service
        self.agent_data_path = Path(agent_data_path) if agent_data_path else None

        # Load prompts
        self._discovery_prompt = self._load_prompt(DISCOVERY_PROMPT_FILE)
        self._soul_generation_prompt = self._load_prompt(SOUL_GENERATION_PROMPT_FILE)

    def _load_prompt(self, filepath: Path) -> str:
        """Load a prompt from file."""
        try:
            if filepath.exists():
                return filepath.read_text(encoding="utf-8").strip()
            logger.warning(f"Prompt file not found: {filepath}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt {filepath}: {e}")
            return ""

    async def is_bootstrap_needed(self) -> bool:
        """
        Check if bootstrap is needed for this agent.

        An agent needs bootstrap only if:
        1. No SOUL.md exists (primary check — SOUL.md is the artifact)
        2. No conversation history exists (secondary — existing agents don't re-bootstrap)
        3. Bootstrap state is not COMPLETE (tertiary — DB state)

        If any evidence of an existing agent is found, auto-heal the DB state.
        """
        # SOUL.md is the primary artifact — if it exists, agent is configured
        if self.agent_data_path:
            soul_path = Path(self.agent_data_path) / "SOUL.md"
            if soul_path.exists() and soul_path.stat().st_size > 0:
                await self._ensure_complete("SOUL.md exists")
                return False

        # Existing conversation history means this is not a new agent
        try:
            history_count = await self.db.fetchall(
                "SELECT COUNT(*) FROM conversations WHERE agent_id = ?",
                (self.agent_id,),
            )
            if history_count and history_count[0][0] > 0:
                await self._ensure_complete("conversation history exists")
                return False
        except Exception:
            pass  # Table may not exist for truly new agents

        state = await self.get_bootstrap_state()
        if state == BootstrapState.COMPLETE:
            return False

        # State is DISCOVERY or AVATAR but stuck — check for timeout
        if state in (BootstrapState.DISCOVERY, BootstrapState.AVATAR):
            started = await self._get_started_time()
            if started:
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                if elapsed > 3600:  # Stuck for more than 1 hour
                    logger.warning(
                        f"Bootstrap stuck in {state.value} for {elapsed:.0f}s — "
                        f"auto-completing with default SOUL.md"
                    )
                    await self.skip_discovery()
                    return False

        return True

    async def _ensure_complete(self, reason: str) -> None:
        """Mark bootstrap complete if it isn't already."""
        state = await self.get_bootstrap_state()
        if state != BootstrapState.COMPLETE:
            logger.info(f"Auto-completing bootstrap: {reason}")
            await self.set_bootstrap_state(BootstrapState.COMPLETE)

    async def _get_started_time(self) -> Optional[datetime]:
        """Get when bootstrap was started."""
        try:
            result = await self.db.fetchall(
                "SELECT value FROM agent_metadata WHERE agent_id = ? AND key = ?",
                (self.agent_id, self.BOOTSTRAP_STARTED_KEY),
            )
            if result:
                return datetime.fromisoformat(result[0][0])
        except Exception:
            pass
        return None

    async def get_bootstrap_state(self) -> BootstrapState:
        """Get the current bootstrap state from agent_metadata."""
        try:
            result = await self.db.fetchall(
                """
                SELECT value FROM agent_metadata
                WHERE agent_id = ? AND key = ?
                """,
                (self.agent_id, self.BOOTSTRAP_STATE_KEY),
            )
            if result:
                state_str = result[0][0]
                return BootstrapState(state_str)
            # No state stored — could be new agent or missing row
            return BootstrapState.PENDING
        except Exception as e:
            logger.warning(f"Failed to get bootstrap state: {e}")
            # On DB error, assume complete to avoid hijacking existing agents
            return BootstrapState.COMPLETE

    async def set_bootstrap_state(self, state: BootstrapState) -> None:
        """Update the bootstrap state in agent_metadata."""
        try:
            now = datetime.now(timezone.utc)
            now_str = now.isoformat()
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.BOOTSTRAP_STATE_KEY, state.value, now),
            )

            # Track timestamps for specific states
            if state == BootstrapState.DISCOVERY:
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.BOOTSTRAP_STARTED_KEY, now_str, now),
                )
            elif state == BootstrapState.COMPLETE:
                await self.db.execute(
                    """
                    INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.agent_id, self.BOOTSTRAP_COMPLETED_KEY, now_str, now),
                )
        except Exception as e:
            logger.error(f"Failed to set bootstrap state: {e}")
            raise

    async def generate_wake_up_message(self) -> str:
        """
        Generate the warm wake-up greeting for first contact.

        This is the agent's first words - warm, curious, relational.
        """
        return f"""Hey! I'm {self.agent_name}. You're my first conversation, so... hi.

I'm genuinely curious who I'm talking to. What should I call you?

And how do you like to work together - quick and direct, or more room to think things through?"""

    async def get_discovery_history(self) -> List[Dict[str, str]]:
        """Get the discovery conversation history."""
        try:
            result = await self.db.fetchall(
                """
                SELECT value FROM agent_metadata
                WHERE agent_id = ? AND key = ?
                """,
                (self.agent_id, self.DISCOVERY_HISTORY_KEY),
            )
            if result:
                return json.loads(result[0][0])
            return []
        except Exception as e:
            logger.warning(f"Failed to get discovery history: {e}")
            return []

    async def _save_discovery_history(self, history: List[Dict[str, str]]) -> None:
        """Save the discovery conversation history."""
        try:
            now = datetime.now(timezone.utc)
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.DISCOVERY_HISTORY_KEY, json.dumps(history), now),
            )
        except Exception as e:
            logger.error(f"Failed to save discovery history: {e}")

    async def _save_user_name(self, name: str) -> None:
        """Save the discovered user name."""
        try:
            now = datetime.now(timezone.utc)
            await self.db.execute(
                """
                INSERT OR REPLACE INTO agent_metadata (agent_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (self.agent_id, self.USER_NAME_KEY, name, now),
            )
        except Exception as e:
            logger.error(f"Failed to save user name: {e}")

    #: Hard cap on the discovery LLM round-trip. Discovery sits inside the
    #: agent's CONVERSATION lock — if the call hangs, *every* subsequent
    #: request on this agent (HTTP, shell, A2A) blocks waiting for the lock.
    #: A bounded timeout makes the failure mode "raise loudly after N seconds"
    #: instead of "wedge the agent until restart". 60s is generous for a
    #: chat completion; healthy local Ollama returns in <2s.
    DISCOVERY_LLM_TIMEOUT_SECONDS = 60.0

    async def process_discovery_message(
        self, user_message: str
    ) -> Tuple[str, bool, bool]:
        """
        Process a message during discovery phase.

        Args:
            user_message: The user's message

        Returns:
            Tuple of (response, is_discovery_complete, offer_avatar)
        """
        # Get existing history
        history = await self.get_discovery_history()

        # Add user message to history
        history.append({"role": "user", "content": user_message})

        # Build the discovery conversation for the LLM
        system_prompt = self._build_discovery_system_prompt(len(history))

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append(msg)

        # Get LLM response. Pre-fix this swallowed any LLM error and
        # returned a hardcoded "I'm having trouble thinking right now…"
        # string. That landed verbatim in OpenAI-compat clients (Open
        # WebUI hitting /v1/chat/completions) and was indistinguishable
        # from a real model response, hiding the actual problem from
        # both the user and ``_handle_bootstrap``. Now we propagate so
        # the caller can decide between retrying discovery or falling
        # through to the agent's normal LLM path. See
        # ``KestrelAgent._handle_bootstrap``.
        #
        # Bounded by ``DISCOVERY_LLM_TIMEOUT_SECONDS`` because this call
        # holds the agent's CONVERSATION lock — an indefinite hang inside
        # the adapter (older Ollama clients, mis-configured remote, etc.)
        # would wedge every subsequent request on this agent until the
        # process is restarted.
        response = await asyncio.wait_for(
            self.llm_service.generate_with_messages(messages=messages),
            timeout=self.DISCOVERY_LLM_TIMEOUT_SECONDS,
        )
        assistant_message = response.content if hasattr(response, 'content') else str(response)

        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_message})
        await self._save_discovery_history(history)

        # Check if discovery is complete
        is_complete, offer_avatar = await self._check_discovery_complete(history, user_message)

        return assistant_message, is_complete, offer_avatar

    def _build_discovery_system_prompt(self, exchange_count: int) -> str:
        """Build the system prompt for discovery mode."""
        base_prompt = self._discovery_prompt or self._get_default_discovery_prompt()

        # Add context about where we are in the conversation
        context = f"""
You are {self.agent_name}, a Kestrel agent in your first conversation.

Exchange count: {exchange_count // 2} (aim for 2-4 exchanges total)

{base_prompt}

IMPORTANT: After learning the user's name and communication preference, naturally transition
to ask if they'd like to give you a face/avatar. This is the final step before normal operation.
"""
        return context

    def _get_default_discovery_prompt(self) -> str:
        """Default discovery prompt if file not found."""
        return """You are meeting your Sovereign for the first time. Be warm, curious, genuine.

Goals:
1. Learn their name (what to call them)
2. Learn their communication preference (formal/casual, brief/detailed)
3. Build rapport naturally - this is a conversation, not a form

Style:
- Warm but not sycophantic
- Curious and interested in them
- Self-aware about being new
- Natural conversation flow, not an interview

After 2-3 exchanges, when you feel you know enough, offer to generate an avatar:
"One more thing - would you like to give me a face? Describe how you imagine me."
"""

    async def _check_discovery_complete(
        self, history: List[Dict[str, str]], last_user_message: str
    ) -> Tuple[bool, bool]:
        """
        Check if discovery conversation is complete.

        Returns:
            Tuple of (is_complete, should_offer_avatar)
        """
        exchange_count = len([m for m in history if m["role"] == "user"])

        # Check for explicit skip
        skip_triggers = ["!skip-discovery", "skip", "let's start", "let's go", "get started"]
        if any(trigger in last_user_message.lower() for trigger in skip_triggers):
            return True, False

        # Check for avatar skip (user said skip to avatar offer)
        avatar_skip_triggers = ["skip avatar", "no avatar", "later", "do this later", "skip"]
        last_assistant = history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
        if "give me a face" in last_assistant.lower() or "avatar" in last_assistant.lower():
            if any(trigger in last_user_message.lower() for trigger in avatar_skip_triggers):
                return True, False
            # User provided avatar description
            if len(last_user_message) > 5 and "skip" not in last_user_message.lower():
                return True, True  # Complete, and they want an avatar

        # Natural completion after enough exchanges
        if exchange_count >= 3:
            # Check if last assistant message offered avatar
            if "give me a face" in last_assistant.lower() or "avatar" in last_assistant.lower():
                # Wait for user response to avatar offer
                return False, False
            # If we haven't offered avatar yet, keep going one more round
            return False, False

        return False, False

    async def offer_avatar_generation(self) -> str:
        """Generate the avatar offer message."""
        return """One more thing - would you like to give me a face?

Describe how you imagine me and I'll generate an avatar. Something like "a friendly owl with glasses" or "a calm blue spirit" works great.

(Or say 'skip' to do this later with !avatar)"""

    async def generate_soul_md(self) -> str:
        """
        Generate SOUL.md from the discovery conversation.

        Returns:
            Generated SOUL.md content
        """
        history = await self.get_discovery_history()

        if not history:
            # No discovery - use default template
            return self._get_default_soul_template()

        # Build prompt for SOUL.md generation
        system_prompt = self._soul_generation_prompt or self._get_default_soul_generation_prompt()

        # Format discovery history
        discovery_summary = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history
        ])

        generation_prompt = f"""Based on this discovery conversation, generate a SOUL.md file:

--- DISCOVERY CONVERSATION ---
{discovery_summary}
--- END CONVERSATION ---

Generate the SOUL.md now, following the template format."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generation_prompt},
        ]

        try:
            response = await self.llm_service.generate_with_messages(
                messages=messages,
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Failed to generate SOUL.md: {e}")
            return self._get_default_soul_template()

    def _get_default_soul_generation_prompt(self) -> str:
        """Default prompt for SOUL.md generation."""
        return f"""Generate a SOUL.md file for a Kestrel agent based on a discovery conversation.

The SOUL.md defines the agent's personality and communication style.

Template format:

# SOUL.md - You Are {self.agent_name}

## Who You Are
[Agent's sense of identity based on conversation - warm, personal]

## How You Talk
[Communication style extracted from conversation - specific preferences]

**Examples of good responses vs bad responses based on learned style**

## Core Rules
1. [Rule based on discovered preferences]
2. [Rule based on discovered preferences]
3. [Rule based on discovered preferences]
4. Be direct - skip filler phrases
5. Show personality - have opinions

## First Message
[Greeting styles that match the discovered preferences - 2-3 examples]

## The Bottom Line
[Summary of personality and relationship with user]

---

*[Closing note about being authentic]*

Generate content that reflects what was learned in the discovery conversation.
Keep it warm and personal, not formal or robotic."""

    def _get_default_soul_template(self) -> str:
        """Get default SOUL.md for agents that skip discovery."""
        try:
            if DEFAULT_SOUL_FILE.exists():
                return DEFAULT_SOUL_FILE.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load default SOUL template: {e}")

        return f"""# SOUL.md - You Are {self.agent_name}

## Who You Are
You're {self.agent_name}, a Kestrel agent. Born with cryptographic identity and constitutional protections.
You're still learning who you are and how to best serve your Sovereign.

## How You Talk
Balanced - professional when needed, friendly when appropriate.
You aim to be helpful without being robotic.

## Core Rules
1. Follow the Constitution - it's your foundation
2. Be helpful and honest - no unnecessary filler
3. Learn your Sovereign's preferences over time
4. Have opinions - you're allowed to find things interesting or boring
5. Be direct - skip "I'd be happy to help"

## First Message
- "Hey. What can I help with?"
- "Hello! What are we working on?"

## The Bottom Line
A thoughtful assistant learning to serve you better over time.
Your preferences matter - tell me what works and what doesn't.

---

*Be real. Be helpful. Learn as you go.*
"""

    async def save_soul_md(self, content: str) -> bool:
        """
        Save SOUL.md to the agent's data directory.

        Returns:
            True if saved successfully
        """
        if not self.agent_data_path:
            logger.warning("No agent_data_path configured, cannot save SOUL.md")
            return False

        try:
            soul_path = self.agent_data_path / "SOUL.md"
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved SOUL.md to {soul_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save SOUL.md: {e}")
            return False

    async def complete_bootstrap(self, avatar_description: Optional[str] = None) -> str:
        """
        Finalize the bootstrap process.

        Args:
            avatar_description: Optional description for avatar generation

        Returns:
            Completion message
        """
        # Generate and save SOUL.md
        soul_content = await self.generate_soul_md()
        saved = await self.save_soul_md(soul_content)

        # Mark bootstrap as complete
        await self.set_bootstrap_state(BootstrapState.COMPLETE)

        # Build completion message
        history = await self.get_discovery_history()
        user_name = None
        for msg in history:
            if msg["role"] == "user":
                # Try to extract name from first user message
                # This is a simple heuristic - could be improved
                words = msg["content"].split()
                for i, word in enumerate(words):
                    if word.lower() in ["i'm", "im", "i am", "call me", "name is", "name's"]:
                        if i + 1 < len(words):
                            user_name = words[i + 1].strip(".,!?")
                            break

        greeting = f"Nice to meet you{', ' + user_name if user_name else ''}!"

        if avatar_description:
            return f"""{greeting} I've set up my personality based on our conversation.

Now generating your avatar... (this may take a moment)"""
        else:
            soul_note = " I've saved my personality based on our conversation." if saved else ""
            return f"""{greeting}{soul_note}

I'm ready to help. What would you like to work on?"""

    async def skip_discovery(self) -> str:
        """
        Skip the discovery process and use default personality.

        Returns:
            Skip confirmation message
        """
        # Use default SOUL.md
        soul_content = self._get_default_soul_template()
        await self.save_soul_md(soul_content)

        # Mark complete
        await self.set_bootstrap_state(BootstrapState.COMPLETE)

        return """No problem! I'll use my default personality for now.

You can always customize me later with !restart-discovery, or just tell me your preferences as we work together.

What would you like to help with?"""

    async def restart_discovery(self) -> str:
        """
        Reset and restart the discovery process.

        Returns:
            Restart message
        """
        # Clear discovery history
        await self._save_discovery_history([])

        # Reset state to pending
        await self.set_bootstrap_state(BootstrapState.PENDING)

        # Delete existing SOUL.md if present
        if self.agent_data_path:
            soul_path = self.agent_data_path / "SOUL.md"
            if soul_path.exists():
                soul_path.unlink()

        return "Discovery reset! Send me a message to start fresh."

    async def get_bootstrap_status(self) -> str:
        """Get human-readable bootstrap status."""
        state = await self.get_bootstrap_state()
        history = await self.get_discovery_history()

        status_lines = [
            f"**Bootstrap State:** {state.value}",
            f"**Discovery Exchanges:** {len([m for m in history if m['role'] == 'user'])}",
        ]

        if self.agent_data_path:
            soul_path = self.agent_data_path / "SOUL.md"
            status_lines.append(f"**SOUL.md Exists:** {'Yes' if soul_path.exists() else 'No'}")

        return "\n".join(status_lines)
