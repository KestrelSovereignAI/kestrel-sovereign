"""
Standardized Model Metadata Types

Provides consistent model information across all LLM providers.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ModelCategory(str, Enum):
    """Model category for filtering."""
    CHAT = "chat"
    EMBEDDING = "embedding"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class ModelInfo:
    """
    Standardized model information across all providers.

    Required fields come from API discovery.
    Optional fields are enriched from config or usage tracking.
    """
    # Required (from API)
    id: str                     # API model ID (e.g., "gpt-5-mini", "claude-sonnet-4-5-20250929")
    provider: str               # Provider name (e.g., "openai", "anthropic", "ollama")
    display_name: str           # Human-readable name (from API or config override)
    category: ModelCategory = ModelCategory.CHAT  # chat, embedding, image, audio

    # Presentation flags (set by ModelCatalogService)
    is_featured: bool = False   # In featured list for this provider
    is_hidden: bool = False     # Truly broken/useless models to hide

    # Usage tracking (populated from DB)
    frecency_score: float = 0.0  # MRU with decay
    last_used: Optional[datetime] = None
    use_count: int = 0

    # Optional metadata (when available from API)
    description: Optional[str] = None
    created_at: Optional[str] = None

    # Capabilities (inferred from config or API)
    supports_vision: bool = False
    supports_tools: bool = False
    supports_streaming: bool = True

    # For local models (Ollama)
    size_gb: Optional[float] = None

    # Context window (tokens) - critical for budget allocation
    context_limit: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "provider": self.provider,
            "display_name": self.display_name,
            "category": self.category.value,
            "is_featured": self.is_featured,
            "is_hidden": self.is_hidden,
            "frecency_score": self.frecency_score,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "use_count": self.use_count,
            "description": self.description,
            "created_at": self.created_at,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "size_gb": self.size_gb,
            "context_limit": self.context_limit,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            provider=data["provider"],
            display_name=data.get("display_name", data["id"]),
            category=ModelCategory(data.get("category", "chat")),
            is_featured=data.get("is_featured", False),
            is_hidden=data.get("is_hidden", False),
            frecency_score=data.get("frecency_score", 0.0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            use_count=data.get("use_count", 0),
            description=data.get("description"),
            created_at=data.get("created_at"),
            supports_vision=data.get("supports_vision", False),
            supports_tools=data.get("supports_tools", False),
            supports_streaming=data.get("supports_streaming", True),
            size_gb=data.get("size_gb"),
            context_limit=data.get("context_limit"),
        )
