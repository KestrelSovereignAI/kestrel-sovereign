"""
Token counting for context management.

Uses tiktoken for accurate OpenAI model token counts,
with fallback estimation for Ollama and other models.

Context limits are sourced from (in priority order):
1. Discovered limits (from API discovery at runtime)
2. ModelCatalogService (model_catalog.toml overrides)
3. Family pattern defaults (prefix-based)
4. DEFAULT_CONTEXT_LIMIT fallback
"""

import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_catalog_service = None

def _get_catalog_service():
    """Get catalog service lazily to avoid circular imports."""
    global _catalog_service
    if _catalog_service is None:
        try:
            from kestrel_sovereign.llm.model_catalog import get_catalog_service
            _catalog_service = get_catalog_service()
        except ImportError:
            logger.debug("ModelCatalogService not available")
    return _catalog_service

# Try to import tiktoken, but don't fail if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using character-based estimation")


# --- Discovered limits registry (populated by model_discovery at runtime) ---
_discovered_context_limits: Dict[str, int] = {}

def register_discovered_limits(models: list) -> None:
    """Register context limits from discovered models.

    Called by model_discovery after API discovery completes.
    Models with a non-None context_limit populate this registry.
    """
    for model in models:
        ctx = getattr(model, 'context_limit', None)
        if ctx:
            _discovered_context_limits[model.id] = ctx
    if _discovered_context_limits:
        logger.info(f"Registered {len(_discovered_context_limits)} discovered context limits")


# Family-based pattern defaults (more specific prefixes first)
MODEL_FAMILY_DEFAULTS = [
    ("claude-opus-4-5", 1000000),
    ("claude-", 200000),
    ("gpt-5.1", 256000),
    ("gpt-4o", 128000),
    ("gpt-4-turbo", 128000),
    ("gpt-5", 128000),
    ("gpt-4", 8192),
    ("gpt-3.5", 16385),
    ("gemini-3", 2000000),
    ("gemini-2.5", 1000000),
    ("gemini-2", 1000000),
    ("gemini-", 1000000),
    ("llama", 128000),
    ("qwen", 32768),
    ("mistral", 32768),
    ("mixtral", 32768),
    ("deepseek", 128000),
    ("phi", 16384),
    ("grok", 128000),
    ("gpt-oss", 32768),
]

DEFAULT_CONTEXT_LIMIT = 32768

# Characters per token estimate for non-tiktoken models
CHARS_PER_TOKEN_ESTIMATE = 4


class TokenCounter:
    """
    Counts tokens in text using tiktoken when available,
    falling back to character-based estimation.
    """

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter for a specific model.

        Args:
            model: The model name (e.g., "gpt-4", "llama3.2:3b")
        """
        self.model = model
        self.encoder = None
        self._use_tiktoken = False

        if TIKTOKEN_AVAILABLE:
            try:
                # Try to get encoding for the specific model
                self.encoder = tiktoken.encoding_for_model(model)
                self._use_tiktoken = True
                logger.debug(f"Using tiktoken for model: {model}")
            except KeyError:
                # Model not found, try cl100k_base (GPT-4 family)
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                    self._use_tiktoken = True
                    logger.debug(f"Using cl100k_base encoding for unknown model: {model}")
                except Exception:
                    logger.debug(f"No tiktoken encoding available for model: {model}")

        if not self._use_tiktoken:
            logger.debug(f"Using character estimation for model: {model}")

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self._use_tiktoken and self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as e:
                logger.warning(f"tiktoken encoding failed, using estimation: {e}")

        # Fallback: estimate based on characters
        return len(text) // CHARS_PER_TOKEN_ESTIMATE

    def count_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of chat messages.

        Accounts for message structure overhead (role, separators).

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Total token count including overhead
        """
        total = 0

        # Base overhead per message (role name, separators)
        # OpenAI uses ~4 tokens per message for structure
        MESSAGE_OVERHEAD = 4

        for msg in messages:
            content = msg.get("content", "")
            total += self.count(content) + MESSAGE_OVERHEAD

        # Add priming tokens (assistant response start)
        total += 3

        return total

    def get_context_limit(self) -> int:
        """
        Get the context window limit for the current model.

        Sources (in order of priority):
        1. Discovered limits (from API discovery at runtime)
        2. ModelCatalogService (model_catalog.toml overrides)
        3. Family pattern defaults (prefix-based)
        4. DEFAULT_CONTEXT_LIMIT fallback

        Returns:
            Maximum tokens allowed in context
        """
        # 1. Discovered limits (populated by API discovery)
        if self.model in _discovered_context_limits:
            return _discovered_context_limits[self.model]

        # 2. Catalog TOML overrides
        catalog = _get_catalog_service()
        if catalog is not None:
            limit = catalog.get_context_limit(self.model)
            if limit is not None:
                return limit

        # 3. Family pattern defaults (prefix match)
        model_lower = self.model.lower()
        base_model = self.model.split(":")[0].lower()
        for prefix, limit in MODEL_FAMILY_DEFAULTS:
            if model_lower.startswith(prefix) or base_model.startswith(prefix) or prefix in model_lower:
                return limit

        logger.warning(f"Unknown model {self.model}, using default context limit {DEFAULT_CONTEXT_LIMIT}")
        return DEFAULT_CONTEXT_LIMIT

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        if not text:
            return text

        current_tokens = self.count(text)
        if current_tokens <= max_tokens:
            return text

        if self._use_tiktoken and self.encoder:
            # Precise truncation with tiktoken
            try:
                tokens = self.encoder.encode(text)
                truncated_tokens = tokens[:max_tokens]
                return self.encoder.decode(truncated_tokens)
            except Exception:
                pass

        # Fallback: estimate character limit
        char_limit = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        return text[:char_limit]

    def fits_in_context(self, text: str, reserved_tokens: int = 0) -> bool:
        """
        Check if text fits within model's context window.

        Args:
            text: Text to check
            reserved_tokens: Tokens already used or reserved

        Returns:
            True if text fits
        """
        available = self.get_context_limit() - reserved_tokens
        return self.count(text) <= available


def get_token_counter(model: str) -> TokenCounter:
    """
    Factory function to get a token counter for a model.

    Args:
        model: Model name

    Returns:
        TokenCounter instance
    """
    return TokenCounter(model)
