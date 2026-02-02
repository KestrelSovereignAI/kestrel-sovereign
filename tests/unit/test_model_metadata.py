"""
Unit tests for ModelInfo and ModelCategory in model_metadata.py
"""
import pytest
from datetime import datetime

from kestrel_sovereign.llm.model_metadata import ModelInfo, ModelCategory


class TestModelCategory:
    """Test ModelCategory enum"""

    def test_category_values(self):
        """Test that all expected categories exist"""
        assert ModelCategory.CHAT.value == "chat"
        assert ModelCategory.EMBEDDING.value == "embedding"
        assert ModelCategory.IMAGE.value == "image"
        assert ModelCategory.AUDIO.value == "audio"

    def test_category_from_string(self):
        """Test creating category from string"""
        assert ModelCategory("chat") == ModelCategory.CHAT
        assert ModelCategory("embedding") == ModelCategory.EMBEDDING
        assert ModelCategory("image") == ModelCategory.IMAGE
        assert ModelCategory("audio") == ModelCategory.AUDIO

    def test_invalid_category_raises(self):
        """Test that invalid category raises ValueError"""
        with pytest.raises(ValueError):
            ModelCategory("invalid")


class TestModelInfo:
    """Test ModelInfo dataclass"""

    def test_create_minimal(self):
        """Test creating ModelInfo with minimal fields"""
        model = ModelInfo(
            id="gpt-5-mini",
            provider="openai",
            display_name="GPT-5 Mini"
        )

        assert model.id == "gpt-5-mini"
        assert model.provider == "openai"
        assert model.display_name == "GPT-5 Mini"
        assert model.category == ModelCategory.CHAT  # default
        assert model.is_featured is False
        assert model.is_hidden is False
        assert model.frecency_score == 0.0
        assert model.last_used is None
        assert model.use_count == 0

    def test_create_full(self):
        """Test creating ModelInfo with all fields"""
        now = datetime.utcnow()
        model = ModelInfo(
            id="claude-sonnet-4-5-20250929",
            provider="anthropic",
            display_name="Claude Sonnet 4.5",
            category=ModelCategory.CHAT,
            is_featured=True,
            is_hidden=False,
            frecency_score=10.5,
            last_used=now,
            use_count=25,
            description="Latest Claude model",
            created_at="2025-09-29",
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
            size_gb=None
        )

        assert model.id == "claude-sonnet-4-5-20250929"
        assert model.is_featured is True
        assert model.frecency_score == 10.5
        assert model.use_count == 25
        assert model.supports_vision is True

    def test_to_dict(self):
        """Test serialization to dict"""
        now = datetime(2025, 11, 30, 12, 0, 0)
        model = ModelInfo(
            id="llama3.2:latest",
            provider="ollama",
            display_name="Llama 3.2",
            category=ModelCategory.CHAT,
            is_featured=True,
            last_used=now,
            use_count=5,
            size_gb=4.5
        )

        d = model.to_dict()

        assert d["id"] == "llama3.2:latest"
        assert d["provider"] == "ollama"
        assert d["display_name"] == "Llama 3.2"
        assert d["category"] == "chat"
        assert d["is_featured"] is True
        assert d["last_used"] == now.isoformat()
        assert d["use_count"] == 5
        assert d["size_gb"] == 4.5

    def test_from_dict(self):
        """Test deserialization from dict"""
        data = {
            "id": "gpt-5.1",
            "provider": "openai",
            "display_name": "GPT-5.1",
            "category": "chat",
            "is_featured": True,
            "is_hidden": False,
            "frecency_score": 5.0,
            "last_used": "2025-11-30T12:00:00",
            "use_count": 10,
            "description": "Latest GPT model",
            "supports_vision": True,
            "supports_tools": True,
        }

        model = ModelInfo.from_dict(data)

        assert model.id == "gpt-5.1"
        assert model.provider == "openai"
        assert model.category == ModelCategory.CHAT
        assert model.is_featured is True
        assert model.use_count == 10
        assert model.last_used == datetime(2025, 11, 30, 12, 0, 0)

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data"""
        data = {
            "id": "phi4",
            "provider": "ollama"
        }

        model = ModelInfo.from_dict(data)

        assert model.id == "phi4"
        assert model.provider == "ollama"
        assert model.display_name == "phi4"  # defaults to id
        assert model.category == ModelCategory.CHAT
        assert model.is_featured is False

    def test_embedding_category(self):
        """Test embedding model category"""
        model = ModelInfo(
            id="text-embedding-3-large",
            provider="openai",
            display_name="Text Embedding 3 Large",
            category=ModelCategory.EMBEDDING
        )

        assert model.category == ModelCategory.EMBEDDING
        d = model.to_dict()
        assert d["category"] == "embedding"


class TestModelInfoRoundTrip:
    """Test that ModelInfo can be serialized and deserialized"""

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip"""
        original = ModelInfo(
            id="gemini-3-pro",
            provider="google",
            display_name="Gemini 3 Pro",
            category=ModelCategory.CHAT,
            is_featured=True,
            frecency_score=15.5,
            last_used=datetime(2025, 11, 30, 10, 30, 0),
            use_count=50,
            description="Google's latest model",
            supports_vision=True,
            supports_tools=True,
            supports_streaming=True,
        )

        d = original.to_dict()
        restored = ModelInfo.from_dict(d)

        assert restored.id == original.id
        assert restored.provider == original.provider
        assert restored.display_name == original.display_name
        assert restored.category == original.category
        assert restored.is_featured == original.is_featured
        assert restored.frecency_score == original.frecency_score
        assert restored.use_count == original.use_count
        assert restored.description == original.description
        assert restored.supports_vision == original.supports_vision
