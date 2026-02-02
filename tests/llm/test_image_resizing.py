"""
Tests for image resizing functionality in llm/image_utils.py
"""
import base64
import io
import pytest
from typing import Tuple

# Try to import PIL, skip tests if not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from kestrel_sovereign.llm.image_utils import (
    resize_image_if_needed,
    process_image,
    process_images,
    get_base64_only,
    get_max_dimensions_for_provider,
    PROVIDER_MAX_DIMENSIONS,
    DEFAULT_MAX_DIMENSIONS,
)


def create_test_image(width: int, height: int, color: Tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Create a test image with the specified dimensions."""
    if not HAS_PIL:
        pytest.skip("PIL not installed")

    img = Image.new('RGB', (width, height), color)
    output = io.BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()


def get_image_size(image_bytes: bytes) -> Tuple[int, int]:
    """Get the dimensions of an image from bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.size


@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestImageResizing:
    """Tests for image resizing functionality."""

    def test_no_resize_when_under_limit(self):
        """Test that images under the limit are not resized."""
        # Create a 100x100 image
        original = create_test_image(100, 100)
        max_dims = (200, 200)

        result = resize_image_if_needed(original, max_dims)

        # Size should be unchanged
        assert get_image_size(result) == (100, 100)

    def test_resize_when_width_exceeds_limit(self):
        """Test resizing when width exceeds limit."""
        # Create a 300x200 image (aspect ratio 1.5)
        original = create_test_image(300, 200)
        max_dims = (150, 150)

        result = resize_image_if_needed(original, max_dims)

        # Width should be limited to 150, height scaled proportionally
        new_size = get_image_size(result)
        assert new_size[0] <= 150
        assert new_size[1] <= 150
        # Aspect ratio should be preserved (approximately)
        original_ratio = 300 / 200
        new_ratio = new_size[0] / new_size[1]
        assert abs(original_ratio - new_ratio) < 0.01

    def test_resize_when_height_exceeds_limit(self):
        """Test resizing when height exceeds limit."""
        # Create a 200x300 image (aspect ratio 0.67)
        original = create_test_image(200, 300)
        max_dims = (150, 150)

        result = resize_image_if_needed(original, max_dims)

        # Height should be limited to 150, width scaled proportionally
        new_size = get_image_size(result)
        assert new_size[0] <= 150
        assert new_size[1] <= 150
        # Aspect ratio should be preserved (approximately)
        original_ratio = 200 / 300
        new_ratio = new_size[0] / new_size[1]
        assert abs(original_ratio - new_ratio) < 0.01

    def test_resize_large_square_image(self):
        """Test resizing a large square image."""
        # Create a 1200x1200 image
        original = create_test_image(1200, 1200)
        max_dims = (1120, 1120)

        result = resize_image_if_needed(original, max_dims)

        new_size = get_image_size(result)
        assert new_size == (1120, 1120)


@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestProcessImageWithResize:
    """Tests for process_image with resizing."""

    def test_process_image_with_max_dimensions(self):
        """Test process_image with explicit max_dimensions."""
        # Create a 300x200 image
        original = create_test_image(300, 200)

        result = process_image(original, max_dimensions=(150, 150))

        assert result is not None
        # Decode and check size
        decoded = base64.b64decode(result.data)
        new_size = get_image_size(decoded)
        assert new_size[0] <= 150
        assert new_size[1] <= 150

    def test_process_image_with_provider(self):
        """Test process_image with provider-specific limits."""
        # Create a large image
        original = create_test_image(2000, 2000)

        result = process_image(original, provider="ollama")

        assert result is not None
        # Decode and check size (Ollama limit is 1120x1120)
        decoded = base64.b64decode(result.data)
        new_size = get_image_size(decoded)
        assert new_size[0] <= 1120
        assert new_size[1] <= 1120

    def test_process_image_without_resize(self):
        """Test process_image without resize (default behavior)."""
        # Create a 100x100 image
        original = create_test_image(100, 100)

        result = process_image(original)

        assert result is not None
        # Should not be resized
        decoded = base64.b64decode(result.data)
        new_size = get_image_size(decoded)
        assert new_size == (100, 100)


@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestProcessImagesWithResize:
    """Tests for process_images with resizing."""

    def test_process_multiple_images_with_provider(self):
        """Test processing multiple images with provider limits."""
        # Create images of different sizes
        images = [
            create_test_image(2000, 1500),  # Needs resize
            create_test_image(500, 500),     # Under limit
            create_test_image(1500, 2000),  # Needs resize
        ]

        results = process_images(images, provider="ollama")

        assert len(results) == 3
        for result in results:
            decoded = base64.b64decode(result.data)
            new_size = get_image_size(decoded)
            assert new_size[0] <= 1120
            assert new_size[1] <= 1120


class TestProviderDimensions:
    """Tests for provider dimension lookups."""

    def test_known_providers(self):
        """Test that known providers have correct limits."""
        assert get_max_dimensions_for_provider("openai") == (2048, 2048)
        assert get_max_dimensions_for_provider("anthropic") == (1568, 1568)
        assert get_max_dimensions_for_provider("vertex_ai") == (3072, 3072)
        assert get_max_dimensions_for_provider("ollama") == (1120, 1120)
        assert get_max_dimensions_for_provider("bedrock") == (1120, 1120)

    def test_unknown_provider_returns_default(self):
        """Test that unknown provider returns default limits."""
        assert get_max_dimensions_for_provider("unknown") == DEFAULT_MAX_DIMENSIONS

    def test_provider_dimensions_dict(self):
        """Test that PROVIDER_MAX_DIMENSIONS is properly populated."""
        assert len(PROVIDER_MAX_DIMENSIONS) >= 4
        for provider, dims in PROVIDER_MAX_DIMENSIONS.items():
            assert isinstance(dims, tuple)
            assert len(dims) == 2
            assert all(isinstance(d, int) for d in dims)


@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestGetBase64OnlyWithResize:
    """Tests for get_base64_only with resizing."""

    def test_get_base64_only_with_provider(self):
        """Test get_base64_only with provider limits."""
        images = [create_test_image(2000, 2000)]

        results = get_base64_only(images, provider="ollama")

        assert len(results) == 1
        # Decode and check size
        decoded = base64.b64decode(results[0])
        new_size = get_image_size(decoded)
        assert new_size[0] <= 1120
        assert new_size[1] <= 1120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
