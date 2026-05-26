"""
Token counting for context management.

Uses tiktoken for accurate OpenAI model token counts,
with fallback estimation for Ollama and other models.

Context limits are sourced from (in priority order):
1. Discovered limits (from API discovery at runtime)
2. Cached limits (persisted from previous discovery runs)
3. ModelCatalogService (model_catalog.toml overrides)
4. DEFAULT_CONTEXT_LIMIT fallback
"""

import json
import logging
from pathlib import Path
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

# --- Persistent cache of discovered limits ---
CACHE_FILE = Path.home() / ".kestrel" / "discovered_context_limits.json"
_cached_limits: Optional[Dict[str, int]] = None  # None = not loaded yet


def _load_cached_limits() -> Dict[str, int]:
    """Load cached context limits from disk. Lazy, one-time read."""
    global _cached_limits
    if _cached_limits is not None:
        return _cached_limits
    _cached_limits = {}
    try:
        if CACHE_FILE.exists():
            _cached_limits = json.loads(CACHE_FILE.read_text())
            logger.info(f"Loaded {len(_cached_limits)} cached context limits from {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to load cached context limits: {e}")
    return _cached_limits


def _persist_cache() -> None:
    """Write discovered limits to disk for cold starts."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Merge with existing cache (preserve entries from providers not discovered this run)
        existing = _load_cached_limits()
        merged = {**existing, **_discovered_context_limits}
        CACHE_FILE.write_text(json.dumps(merged, indent=2, sort_keys=True))
        logger.debug(f"Persisted {len(merged)} context limits to {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to persist context limits cache: {e}")


def register_discovered_limits(models: list) -> None:
    """Register context limits from discovered models and persist to cache.

    Called by model_discovery after API discovery completes.
    Models with a non-None context_limit populate the in-memory registry
    and are written to ~/.kestrel/discovered_context_limits.json.
    """
    for model in models:
        ctx = getattr(model, 'context_limit', None)
        if ctx:
            _discovered_context_limits[model.id] = ctx
    if _discovered_context_limits:
        logger.info(f"Registered {len(_discovered_context_limits)} discovered context limits")
        _persist_cache()


DEFAULT_CONTEXT_LIMIT = 32768

# Characters per token estimate for non-tiktoken models
CHARS_PER_TOKEN_ESTIMATE = 4


class TokenCounter:
    """
    Counts tokens in text using tiktoken when available,
    falling back to character-based estimation.
    """

    def __init__(self, model: str = "auto"):
        """
        Initialize token counter for a specific model.

        Args:
            model: The model name for tiktoken encoding lookup
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
        1. Discovered limits (from API discovery this session)
        2. Cached limits (from previous discovery runs on disk)
        3. ModelCatalogService (model_catalog.toml overrides)
        4. DEFAULT_CONTEXT_LIMIT fallback

        Each source is consulted with the full model string first, then
        with the model-name-only portion (post-slash). The LLM service
        routes by ``"vendor:route/model"`` while the cache stores bare
        model IDs (``"claude-opus-4-7"``) — same model reachable via
        multiple routes, so the cache shouldn't carry route prefixes.
        This is normalization, not a workaround: the actual model ID is
        what discovery records and what catalog lookups expect.

        Returns:
            Maximum tokens allowed in context
        """
        # Build candidate keys for discovered/cache/catalog lookup.
        # Route-qualified ``"<vendor>:<route>/<model_name>"`` splits on
        # the FIRST ``/`` so OpenRouter/HF-style model names containing
        # additional slashes (``google/gemini-pro``,
        # ``meta-llama/Llama-3.1-8B-Instruct``) survive intact and the
        # bare-model cache entry is actually found (codex round-7 P2
        # on PR #1396). For vendor-only ``"vendor/model"`` we keep the
        # legacy ``rsplit`` fallback (some cache entries are stored
        # under the bare model name too).
        candidates = [self.model]
        if ":" in self.model and "/" in self.model:
            candidates.append(self.model.split("/", 1)[1])
        elif "/" in self.model:
            candidates.append(self.model.rsplit("/", 1)[1])

        # Route-qualified models: give the catalog FIRST shot at any
        # route-keyed cap (key containing ``:``) before the bare-model
        # discovered/cache fallthrough. Discovery records the model's
        # full context window — it cannot know about route-level
        # per-turn caps (e.g. ChatGPT-Plus's ~20K payload limit on
        # openai:plan), so the manual catalog override is the only
        # authoritative source for those.
        #
        # The discriminator requires BOTH ``:`` AND ``/`` — the kestrel
        # route form is ``"<vendor>:<route>/<model_name>"``. This
        # excludes Ollama tags like ``"gemma2:9b"`` (which have ``:``
        # but no ``/`` — codex round-2 P2 on PR #1396) and vendor-only
        # forms like ``"openai/gpt-5.5"`` (no route to cap). And it
        # consults ONLY route-keyed catalog entries via
        # ``get_route_context_cap`` so bare-model catalog entries
        # cannot bypass discovery on non-capped routes like
        # ``openai:api/gpt-5.5`` (codex round-1 P2 on PR #1396).
        is_route_qualified = ":" in self.model and "/" in self.model
        if is_route_qualified:
            catalog = _get_catalog_service()
            if catalog is not None:
                limit = catalog.get_route_context_cap(self.model)
                if limit is not None:
                    return limit

        # 1. Discovered limits (populated by API discovery this session)
        for key in candidates:
            if key in _discovered_context_limits:
                return _discovered_context_limits[key]

        # 2. Cached limits (from previous discovery runs)
        cached = _load_cached_limits()
        for key in candidates:
            if key in cached:
                return cached[key]

        # 3. Catalog TOML overrides (bare-model fallback path — route
        # cases already handled above).
        catalog = _get_catalog_service()
        if catalog is not None:
            for key in candidates:
                limit = catalog.get_context_limit(key)
                if limit is not None:
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
