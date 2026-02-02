"""
Token counting for context management.

Uses tiktoken for accurate OpenAI model token counts,
with fallback estimation for Ollama and other models.

Context limits are sourced from:
1. ModelCatalogService (model_catalog.toml) - primary source
2. MODEL_CONTEXT_LIMITS dict - fallback for unconfigured models
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


# Known model context windows (tokens)
MODEL_CONTEXT_LIMITS = {
    # OpenAI models
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-5": 128000,
    "gpt-5-mini": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    # Anthropic models (for reference, though we use OpenAI-style counting)
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-sonnet": 200000,
    # Ollama models (approximate)
    "llama3.2:3b": 8192,
    "llama3.2:1b": 8192,
    "llama3.1:8b": 8192,
    "llama3.1:70b": 8192,
    "mistral:7b": 8192,
    "mixtral:8x7b": 32768,
    "qwen2.5:0.5b": 8192,
    "qwen2.5:3b": 8192,
    "qwen2.5:7b": 8192,
    "phi3:3.8b": 4096,
    "gemma2:9b": 8192,
    "deepseek-coder:6.7b": 8192,
}

# Default context window if model not found
DEFAULT_CONTEXT_LIMIT = 8192

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
        1. ModelCatalogService (model_catalog.toml) - authoritative source
        2. MODEL_CONTEXT_LIMITS dict - fallback for unconfigured models
        3. DEFAULT_CONTEXT_LIMIT - final fallback

        Returns:
            Maximum tokens allowed in context
        """
        # 1. Try catalog service first (authoritative source)
        catalog = _get_catalog_service()
        if catalog is not None:
            limit = catalog.get_context_limit(self.model)
            if limit is not None:
                return limit

        # 2. Fallback to hardcoded limits
        # Try exact match first
        if self.model in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[self.model]

        # Try base model name (before :)
        base_model = self.model.split(":")[0]
        if base_model in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[base_model]

        # Try partial match
        model_lower = self.model.lower()
        for known_model, limit in MODEL_CONTEXT_LIMITS.items():
            if known_model in model_lower or model_lower in known_model:
                return limit

        logger.warning(f"Unknown model {self.model}, using default context limit")
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
