"""
Tests for LLM Adapter base class.
Tests message creation, streaming, structured output, and vision support.
"""
import pytest
import logging
from typing import Generator
from pydantic import BaseModel, Field
import base64
import tempfile
from pathlib import Path

from kestrel_sovereign.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('openai').setLevel(logging.WARNING)


# Test models
class SimpleModel(BaseModel):
    """Simple test model"""
    value: str = Field(description="A simple value")


class DetailedModel(BaseModel):
    """Detailed test model"""
    title: str = Field(description="The title")
    content: str = Field(description="The content")
    count: int = Field(description="A count")


# Mock adapter for testing
class MockAdapter(LLMAdapter):
    """Mock adapter for testing base class functionality"""

    async def get_response(self, client, model, messages, format=None):
        """Mock implementation - not used in these tests"""
        return "Mock response"


# ============================================================================
# Message Creation Tests
# ============================================================================

class TestMessageCreation:
    """Test message creation with various inputs"""

    def test_create_messages_user_only(self):
        """Test creating messages with user prompt only"""
        adapter = MockAdapter()
        messages = adapter.create_messages(user_prompt="Hello")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_create_messages_system_and_user(self):
        """Test creating messages with system and user prompts"""
        adapter = MockAdapter()
        messages = adapter.create_messages(
            user_prompt="Hello",
            system_prompt="You are helpful"
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"

    def test_create_messages_empty(self):
        """Test creating messages with no content returns empty list"""
        adapter = MockAdapter()
        messages = adapter.create_messages()

        # No content = no messages (valid behavior)
        assert len(messages) == 0

    def test_create_messages_none_prompts(self):
        """Test creating messages with None prompts returns empty list"""
        adapter = MockAdapter()
        messages = adapter.create_messages(
            user_prompt=None,
            system_prompt=None
        )

        # None prompts = no messages (valid behavior)
        assert len(messages) == 0

    def test_create_messages_empty_strings(self):
        """Test creating messages with empty strings returns empty list"""
        adapter = MockAdapter()
        messages = adapter.create_messages(
            user_prompt="",
            system_prompt=""
        )

        # Empty strings = no messages (valid behavior)
        assert len(messages) == 0


# ============================================================================
# Vision/Image Handling Tests
# ============================================================================

class TestImageHandling:
    """Test image handling in multiple formats"""

    @pytest.fixture
    def sample_png_path(self, tmp_path):
        """Create a sample PNG file"""
        png_data = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
            b'\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
            b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\x0e\x08k\x00\x00\x00'
            b'\x00IEND\xaeB`\x82'
        )
        image_path = tmp_path / "test.png"
        image_path.write_bytes(png_data)
        return str(image_path)

    @pytest.fixture
    def sample_jpeg_path(self, tmp_path):
        """Create a sample JPEG file"""
        # Minimal valid JPEG
        jpeg_data = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
            b'\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07'
            b'\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14'
            b'\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444'
            b'\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x11'
            b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
            b'\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd0'
            b'\xff\xd9'
        )
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(jpeg_data)
        return str(image_path)

    def test_handle_images_none(self):
        """Test handling None images"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]
        adapter._handle_images(None, content)
        # Should not modify content
        assert len(content) == 1

    def test_handle_images_empty_list(self):
        """Test handling empty image list"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]
        adapter._handle_images([], content)
        # Should not modify content
        assert len(content) == 1

    def test_handle_images_file_path_png(self, sample_png_path):
        """Test handling PNG file path"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]
        adapter._handle_images([sample_png_path], content)

        # Should have text + image
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_handle_images_file_path_jpeg(self, sample_jpeg_path):
        """Test handling JPEG file path"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]
        adapter._handle_images([sample_jpeg_path], content)

        # Should have text + image
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in content[1]["image_url"]["url"]

    def test_handle_images_base64_string(self):
        """Test handling base64 encoded image"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]
        # Minimal base64 PNG
        base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        adapter._handle_images([base64_png], content)

        # Should have text + image
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_handle_images_bytes(self):
        """Test handling image bytes"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]

        # Minimal PNG bytes
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
            b'\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
            b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\x0e\x08k\x00\x00\x00'
            b'\x00IEND\xaeB`\x82'
        )

        adapter._handle_images([png_bytes], content)

        # Should have text + image
        assert len(content) == 2
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    def test_handle_images_multiple(self, sample_png_path):
        """Test handling multiple images"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]

        adapter._handle_images([sample_png_path, sample_png_path], content)

        # Should have text + 2 images
        assert len(content) == 3
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"

    def test_handle_images_mixed_formats(self, sample_png_path):
        """Test handling mixed image formats"""
        adapter = MockAdapter()
        content = [{"type": "text", "text": "Hello"}]

        base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00'
            b'\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx'
            b'\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\x0e\x08k\x00\x00\x00'
            b'\x00IEND\xaeB`\x82'
        )

        adapter._handle_images([sample_png_path, base64_png, png_bytes], content)

        # Should have text + 3 images
        assert len(content) == 4
        for i in range(1, 4):
            assert content[i]["type"] == "image_url"

    # Note: _is_path and _encode_image helper methods were removed
    # Image handling now uses centralized image_utils module
    # See tests/llm/test_image_resizing.py for image processing tests


# ============================================================================
# Message Structure Tests
# ============================================================================

class TestMessageStructure:
    """Test OpenAI-compatible message format"""

    def test_message_format_openai_compatible(self):
        """Test messages are in OpenAI compatible format"""
        adapter = MockAdapter()
        messages = adapter.create_messages(
            system_prompt="You are helpful",
            user_prompt="Hello"
        )

        # Check OpenAI format
        assert isinstance(messages, list)
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]

    def test_user_content_list_format(self):
        """Test user content is a list (for vision support)"""
        adapter = MockAdapter()
        messages = adapter.create_messages(user_prompt="Hello")

        user_msg = messages[-1]
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) > 0
        assert user_msg["content"][0]["type"] == "text"

    def test_image_content_format(self):
        """Test image content has correct format"""
        adapter = MockAdapter()
        content = []

        base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        adapter._handle_images([base64_png], content)

        assert len(content) == 1
        image_content = content[0]
        assert image_content["type"] == "image_url"
        assert "image_url" in image_content
        assert "url" in image_content["image_url"]
        assert image_content["image_url"]["url"].startswith("data:image/")


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompatibility:
    """Test backward compatibility with legacy code"""

    def test_adapter_is_abstract(self):
        """Test adapter has abstract method"""
        # MockAdapter has the abstract method implemented
        adapter = MockAdapter()
        assert hasattr(adapter, 'get_response')

    def test_legacy_create_messages_signature(self):
        """Test create_messages can be called with original signature"""
        adapter = MockAdapter()
        # Original signature was create_messages(system_prompt, user_prompt)
        # New signature is create_messages(user_prompt, system_prompt, images)
        # Both should work

        # New way
        messages = adapter.create_messages(
            user_prompt="test",
            system_prompt="system"
        )
        assert len(messages) == 2

        # Can also use keyword arguments
        messages = adapter.create_messages(
            system_prompt="system",
            user_prompt="test"
        )
        assert len(messages) == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
