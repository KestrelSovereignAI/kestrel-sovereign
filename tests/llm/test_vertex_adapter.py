"""
Tests for Vertex AI Adapter.

REAL INTEGRATION TESTS - require GCP credentials.
All tests use actual Vertex AI API calls.

Required environment variables:
- GOOGLE_APPLICATION_CREDENTIALS or GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT
"""
import pytest
import logging
import os
import base64
from typing import List, Dict, Any

from kestrel_sovereign.llm.vertex_adapter import VertexAIAdapter, create_vertex_adapter, VertexAIConfig
from kestrel_sovereign.llm.adapter import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Skip all tests if no credentials
# ============================================================================

def has_gcp_credentials() -> bool:
    """Check if GCP credentials are available."""
    # Check for credentials file
    creds_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_file and os.path.exists(creds_file):
        return True
    # Check for project ID env vars
    if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID"):
        return True
    return False


pytestmark = pytest.mark.skipif(
    not has_gcp_credentials(),
    reason="Requires GCP credentials (GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT or GCP_PROJECT_ID)"
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def adapter():
    """Create a fresh adapter for each test (no client caching between tests)."""
    a = VertexAIAdapter()
    yield a
    # Clean up the client to avoid state pollution between tests
    a._client = None


@pytest.fixture
def sample_png_path(tmp_path):
    """Create a sample PNG file (10x10 red square - valid for Vertex AI)."""
    # Create a proper 10x10 PNG with PIL if available, otherwise skip vision tests
    try:
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color='red')
        image_path = tmp_path / "test.png"
        img.save(str(image_path), 'PNG')
        return str(image_path)
    except ImportError:
        # Fallback: Create a minimal but valid PNG
        # This is a 10x10 red PNG created with proper structure
        import zlib
        import struct

        def create_png(width, height, rgb):
            def png_chunk(chunk_type, data):
                chunk_len = struct.pack('>I', len(data))
                chunk_crc = struct.pack('>I', zlib.crc32(chunk_type + data) & 0xffffffff)
                return chunk_len + chunk_type + data + chunk_crc

            # IHDR
            ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)

            # IDAT - raw image data (RGB for each pixel, with filter byte per row)
            raw_data = b''
            for _ in range(height):
                raw_data += b'\x00'  # filter byte
                raw_data += bytes(rgb) * width
            compressed = zlib.compress(raw_data)

            # Build PNG
            png = b'\x89PNG\r\n\x1a\n'
            png += png_chunk(b'IHDR', ihdr_data)
            png += png_chunk(b'IDAT', compressed)
            png += png_chunk(b'IEND', b'')
            return png

        png_data = create_png(100, 100, (255, 0, 0))  # 100x100 red
        image_path = tmp_path / "test.png"
        image_path.write_bytes(png_data)
        return str(image_path)


# ============================================================================
# Initialization Tests (Real Credentials)
# ============================================================================

class TestVertexAIAdapterInit:
    """Test adapter initialization with real credentials."""

    def test_init_auto_discovers_project(self):
        """Test initialization auto-discovers project from credentials."""
        adapter = VertexAIAdapter()
        assert adapter.project_id is not None
        logger.info(f"Auto-discovered project: {adapter.project_id}")

    def test_init_default_location(self):
        """Test initialization uses default location."""
        adapter = VertexAIAdapter()
        assert adapter.location == "us-central1"

    def test_init_custom_location(self):
        """Test initialization with custom location."""
        adapter = VertexAIAdapter(location="europe-west1")
        assert adapter.location == "europe-west1"

    def test_factory_function(self):
        """Test create_vertex_adapter factory function."""
        adapter = create_vertex_adapter()
        assert adapter.project_id is not None
        assert adapter.location is not None

    def test_client_creation(self):
        """Test actual client can be created."""
        adapter = VertexAIAdapter()
        client = adapter._get_client()
        assert client is not None
        logger.info(f"Successfully created Vertex AI client for project: {adapter.project_id}")


# ============================================================================
# Message Creation Tests
# ============================================================================

class TestMessageCreation:
    """Test message creation in Vertex AI format."""

    def test_create_messages_user_only(self, adapter):
        """Test creating messages with user prompt only."""
        messages = adapter.create_messages(user_prompt="Hello")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["parts"]) == 1
        assert messages[0]["parts"][0]["text"] == "Hello"

    def test_create_messages_with_system_prompt(self, adapter):
        """Test creating messages with system prompt."""
        messages = adapter.create_messages(
            user_prompt="Hello",
            system_prompt="You are helpful"
        )

        # Should have _system marker and user message
        assert len(messages) == 2
        assert messages[0]["role"] == "_system"
        assert messages[0]["parts"][0]["text"] == "You are helpful"
        assert messages[1]["role"] == "user"

    def test_create_messages_with_images(self, adapter, sample_png_path):
        """Test creating messages with images."""
        messages = adapter.create_messages(
            user_prompt="What's in this image?",
            images=[sample_png_path]
        )

        assert len(messages) == 1
        parts = messages[0]["parts"]
        assert len(parts) == 2  # text + image
        assert parts[0]["text"] == "What's in this image?"
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/png"

    def test_extract_system_prompt(self, adapter):
        """Test system prompt extraction."""
        messages = [
            {"role": "_system", "parts": [{"text": "Be helpful"}]},
            {"role": "user", "parts": [{"text": "Hello"}]}
        ]

        system_prompt, filtered = adapter._extract_system_prompt(messages)

        assert system_prompt == "Be helpful"
        assert len(filtered) == 1
        assert filtered[0]["role"] == "user"


# ============================================================================
# Tool Conversion Tests
# ============================================================================

class TestToolConversion:
    """Test OpenAI to Vertex AI tool format conversion."""

    def test_convert_simple_tool(self, adapter):
        """Test converting a simple tool."""
        openai_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }]

        vertex_tools = adapter._convert_tools_to_vertex_format(openai_tools)

        assert len(vertex_tools) == 1
        assert vertex_tools[0]["name"] == "get_weather"
        assert vertex_tools[0]["description"] == "Get weather for a location"
        assert "location" in vertex_tools[0]["parameters"]["properties"]

    def test_convert_multiple_tools(self, adapter):
        """Test converting multiple tools."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "Tool A",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "description": "Tool B",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        vertex_tools = adapter._convert_tools_to_vertex_format(openai_tools)

        assert len(vertex_tools) == 2
        assert vertex_tools[0]["name"] == "tool_a"
        assert vertex_tools[1]["name"] == "tool_b"

    def test_convert_empty_tools(self, adapter):
        """Test converting empty tools list."""
        vertex_tools = adapter._convert_tools_to_vertex_format([])
        assert len(vertex_tools) == 0


# ============================================================================
# Model Discovery Tests (Real API)
# ============================================================================

class TestModelDiscovery:
    """Test model discovery with real Vertex AI API."""

    @pytest.mark.asyncio
    async def test_list_models_returns_results(self, adapter):
        """Test listing models returns actual results."""
        models = await adapter.list_models()

        assert len(models) > 0
        logger.info(f"Discovered {len(models)} models from Vertex AI")

    @pytest.mark.asyncio
    async def test_list_models_structure(self, adapter):
        """Test model list has correct structure (ModelInfo objects)."""
        from kestrel_sovereign.llm.model_metadata import ModelInfo

        models = await adapter.list_models()

        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.id is not None
            assert model.provider == "vertex_ai"
            assert model.display_name is not None
            assert model.supports_vision is not None
            assert model.supports_tools is not None
            assert model.supports_streaming is not None

    @pytest.mark.asyncio
    async def test_list_models_includes_gemini(self, adapter):
        """Test that model list includes Gemini models."""
        models = await adapter.list_models()
        model_ids = [m.id for m in models]

        # Should have at least some Gemini models
        gemini_models = [m for m in model_ids if "gemini" in m.lower()]
        assert len(gemini_models) > 0
        logger.info(f"Found Gemini models: {gemini_models[:5]}")

    def test_fallback_models(self, adapter):
        """Test fallback model list structure (ModelInfo objects)."""
        from kestrel_sovereign.llm.model_metadata import ModelInfo

        fallback = adapter._get_fallback_models()

        assert len(fallback) >= 3
        for model in fallback:
            assert isinstance(model, ModelInfo)
            assert model.id is not None
            assert model.provider == "vertex_ai"


# ============================================================================
# Generation Tests (Real API)
# ============================================================================

class TestGeneration:
    """Test actual generation with Vertex AI."""

    @pytest.mark.asyncio
    async def test_simple_generation(self, adapter):
        """Test simple text generation."""
        messages = adapter.create_messages(
            user_prompt="Say 'hello' in exactly one word.",
            system_prompt="You are a helpful assistant. Be concise."
        )

        response = await adapter.get_response(
            client=None,  # Use internal client
            model="gemini-2.0-flash-001",  # Use available model
            messages=messages
        )

        assert response.content is not None
        assert len(response.content) > 0
        logger.info(f"Response: {response.content}")

    @pytest.mark.asyncio
    async def test_generation_with_system_prompt(self, adapter):
        """Test generation respects system prompt."""
        messages = adapter.create_messages(
            user_prompt="What is 2 + 2?",
            system_prompt="You are a math teacher. Always explain your reasoning step by step."
        )

        response = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages
        )

        assert response.content is not None
        # Should have some explanation, not just "4"
        assert len(response.content) > 10
        logger.info(f"Math response: {response.content[:200]}...")

    @pytest.mark.asyncio
    async def test_generation_with_temperature(self, adapter):
        """Test generation with temperature parameter."""
        messages = adapter.create_messages(
            user_prompt="Write a haiku about coding."
        )

        response = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages,
            temperature=0.9
        )

        assert response.content is not None
        logger.info(f"Creative response: {response.content}")

    @pytest.mark.asyncio
    async def test_generation_with_max_tokens(self, adapter):
        """Test generation with max_tokens parameter."""
        messages = adapter.create_messages(
            user_prompt="Tell me a very long story about a dragon."
        )

        response = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages,
            max_tokens=50
        )

        assert response.content is not None
        # Response should be truncated
        logger.info(f"Truncated response ({len(response.content)} chars): {response.content}")

    @pytest.mark.asyncio
    async def test_generation_different_models(self, adapter):
        """Test generation works with different models."""
        messages = adapter.create_messages(user_prompt="Say 'test' and nothing else.")

        # Test with flash model (fast)
        response_flash = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages
        )
        assert response_flash.content is not None

        # Test with pro model (more capable)
        response_pro = await adapter.get_response(
            client=None,
            model="gemini-2.5-pro",
            messages=messages
        )
        assert response_pro.content is not None

        logger.info(f"Flash: {response_flash.content}")
        logger.info(f"Pro: {response_pro.content}")


# ============================================================================
# Streaming Tests (Real API)
# ============================================================================

class TestStreaming:
    """Test streaming responses with real Vertex AI."""

    @pytest.mark.asyncio
    async def test_streaming_response(self, adapter):
        """Test streaming text generation."""
        messages = adapter.create_messages(
            user_prompt="Count from 1 to 5, one number per line."
        )

        chunks = []
        async for chunk in adapter.get_streaming_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages
        ):
            chunks.append(chunk)
            logger.info(f"Chunk: {repr(chunk)}")

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert len(full_response) > 0
        logger.info(f"Full streamed response: {full_response}")

    @pytest.mark.asyncio
    async def test_streaming_accumulation(self, adapter):
        """Test that streaming chunks accumulate to complete response."""
        messages = adapter.create_messages(
            user_prompt="What is Python? Answer in one sentence."
        )

        chunks = []
        async for chunk in adapter.get_streaming_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        # Should be a coherent sentence
        assert "Python" in full_response or "python" in full_response.lower()


# ============================================================================
# Vision Tests (Real API)
# ============================================================================

class TestVision:
    """Test vision capabilities with real Vertex AI."""

    @pytest.mark.asyncio
    async def test_image_description(self, adapter, sample_png_path):
        """Test describing an image."""
        messages = adapter.create_messages(
            user_prompt="What do you see in this image? Describe it briefly.",
            images=[sample_png_path]
        )

        response = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",  # Vision-capable model
            messages=messages
        )

        assert response.content is not None
        logger.info(f"Image description: {response.content}")

    @pytest.mark.asyncio
    async def test_base64_image(self, adapter, sample_png_path):
        """Test with base64 encoded image."""
        # Read the sample image and encode as base64
        with open(sample_png_path, 'rb') as f:
            png_data = f.read()
        b64_image = base64.b64encode(png_data).decode('utf-8')

        messages = adapter.create_messages(
            user_prompt="What color is this image?",
            images=[b64_image]
        )

        response = await adapter.get_response(
            client=None,
            model="gemini-2.0-flash-001",
            messages=messages
        )

        assert response.content is not None


# ============================================================================
# Config Tests
# ============================================================================

class TestVertexAIConfig:
    """Test VertexAIConfig dataclass."""

    def test_config_defaults(self):
        """Test config with defaults."""
        config = VertexAIConfig(project_id="test-project")
        assert config.project_id == "test-project"
        assert config.location == "us-central1"
        assert config.credentials_file is None

    def test_config_all_fields(self):
        """Test config with all fields."""
        config = VertexAIConfig(
            project_id="my-project",
            location="europe-west1",
            credentials_file="/path/to/creds.json"
        )
        assert config.project_id == "my-project"
        assert config.location == "europe-west1"
        assert config.credentials_file == "/path/to/creds.json"


# ============================================================================
# Error Handling Tests (Real API)
# ============================================================================

class TestErrorHandling:
    """Test error handling with real API."""

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, adapter):
        """Test error when using invalid model."""
        messages = adapter.create_messages(user_prompt="Hello")

        with pytest.raises(Exception):
            await adapter.get_response(
                client=None,
                model="nonexistent-model-12345",
                messages=messages
            )

    @pytest.mark.asyncio
    async def test_empty_messages_handled(self, adapter):
        """Test handling of empty messages."""
        with pytest.raises(Exception):
            await adapter.get_response(
                client=None,
                model="gemini-2.0-flash-001",
                messages=[]
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
