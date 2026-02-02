"""
Centralized image handling utilities for LLM adapters.

Provides functions for loading, encoding, and detecting MIME types of images
for use across all LLM adapter implementations.

Usage:
    from kestrel_sovereign.llm.image_utils import process_image, process_images

    # Process a single image (file path, base64 string, or bytes)
    img_data, mime_type = process_image(image)

    # Process multiple images
    processed = process_images(images)
    # Returns list of (base64_data, mime_type) tuples

    # Process with auto-resize for provider limits
    processed = process_images(images, max_dimensions=(1024, 1024))
"""
import base64
import io
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Provider-specific image dimension limits
PROVIDER_MAX_DIMENSIONS = {
    "openai": (2048, 2048),      # GPT-4V limit
    "anthropic": (1568, 1568),  # Claude vision limit
    "vertex_ai": (3072, 3072),  # Gemini limit
    "ollama": (1120, 1120),     # LLaVA/Llama vision limit
    "bedrock": (1120, 1120),    # Bedrock Llama limit
}

# Default max dimensions if provider not specified
DEFAULT_MAX_DIMENSIONS = (1120, 1120)


@dataclass
class ProcessedImage:
    """Represents a processed image ready for API submission."""
    data: str  # Base64-encoded image data
    mime_type: str  # MIME type (e.g., "image/png", "image/jpeg")


def detect_mime_type_from_bytes(data: bytes) -> str:
    """
    Detect MIME type from image bytes using magic numbers.

    Args:
        data: Raw image bytes

    Returns:
        MIME type string (defaults to "image/jpeg" if unknown)
    """
    if data.startswith(b'\x89PNG'):
        return "image/png"
    if data.startswith(b'\xff\xd8\xff'):
        return "image/jpeg"
    if data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
        return "image/gif"
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return "image/webp"
    # Default to JPEG
    return "image/jpeg"


def detect_mime_type_from_base64(data: str) -> str:
    """
    Detect MIME type from base64-encoded image data.

    Args:
        data: Base64-encoded image string

    Returns:
        MIME type string (defaults to "image/jpeg" if unknown)
    """
    # PNG base64 starts with iVBORw0KGgo
    if data.startswith('iVBORw0KGgo'):
        return "image/png"
    # JPEG base64 starts with /9j/
    if data.startswith('/9j/'):
        return "image/jpeg"
    # GIF base64 starts with R0lGOD
    if data.startswith('R0lGOD'):
        return "image/gif"
    # WebP base64 starts with UklGR
    if data.startswith('UklGR'):
        return "image/webp"
    # Default to JPEG
    return "image/jpeg"


def detect_mime_type_from_extension(path: str) -> str:
    """
    Detect MIME type from file extension.

    Args:
        path: File path

    Returns:
        MIME type string (defaults to "image/jpeg" if unknown)
    """
    ext = path.split('.')[-1].lower()
    mime_map = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp',
        'svg': 'image/svg+xml',
    }
    return mime_map.get(ext, 'image/jpeg')


def _get_image_bytes(image: Union[str, bytes]) -> Tuple[Optional[bytes], str]:
    """
    Get raw bytes and mime type from an image input.

    Args:
        image: Image as file path, base64 string, or bytes

    Returns:
        Tuple of (raw_bytes, mime_type) on success.
        Returns (None, "") for invalid input - callers must check for None.

    Error Handling:
        - Invalid base64 strings return (None, "")
        - Unknown image types return (None, "")
        - File read errors are propagated as exceptions
    """
    if isinstance(image, str):
        # Heuristic: short string that exists as file = file path
        if len(image) < 1000 and os.path.exists(image):
            with open(image, 'rb') as f:
                raw_data = f.read()
            mime_type = detect_mime_type_from_extension(image)
            return raw_data, mime_type
        else:
            # Already base64 encoded - decode to bytes
            try:
                raw_data = base64.b64decode(image)
                mime_type = detect_mime_type_from_base64(image)
                return raw_data, mime_type
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")
                return None, ""
    elif isinstance(image, bytes):
        mime_type = detect_mime_type_from_bytes(image)
        return image, mime_type
    logger.warning(f"Invalid image type: {type(image).__name__}")
    return None, ""


def resize_image_if_needed(
    image_bytes: bytes,
    max_dimensions: Tuple[int, int],
    output_format: str = "PNG"
) -> bytes:
    """
    Resize an image if it exceeds maximum dimensions while preserving aspect ratio.

    Requires PIL/Pillow to be installed. Falls back to original bytes if PIL unavailable.

    Args:
        image_bytes: Raw image bytes
        max_dimensions: Maximum (width, height) tuple
        output_format: Output image format (PNG, JPEG, WEBP)

    Returns:
        Resized image bytes (or original if no resize needed or PIL unavailable)
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not installed, skipping image resize. Install with: pip install Pillow")
        return image_bytes

    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        original_size = img.size
        max_width, max_height = max_dimensions

        # Check if resize is needed
        if img.size[0] <= max_width and img.size[1] <= max_height:
            return image_bytes

        # Calculate new size preserving aspect ratio
        aspect_ratio = img.size[0] / img.size[1]
        if aspect_ratio > 1:  # Width is larger
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:  # Height is larger or equal
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        # Ensure dimensions don't exceed max
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        if new_height > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

        # Convert to RGB if necessary (for JPEG output)
        if output_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        # Resize using high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        logger.info(f"Resized image from {original_size} to {img.size}")

        # Save to bytes
        output_buffer = io.BytesIO()
        img.save(output_buffer, format=output_format, optimize=True)
        return output_buffer.getvalue()

    except Exception as e:
        logger.warning(f"Failed to resize image: {e}, returning original")
        return image_bytes


def process_image(
    image: Union[str, bytes],
    max_dimensions: Optional[Tuple[int, int]] = None,
    provider: Optional[str] = None
) -> Optional[ProcessedImage]:
    """
    Process a single image into base64-encoded data with MIME type.

    Handles three input types:
    1. File path (str < 1000 chars that exists on disk)
    2. Base64-encoded string (str >= 1000 chars or doesn't exist as file)
    3. Raw bytes

    Args:
        image: Image as file path, base64 string, or bytes
        max_dimensions: Optional (width, height) tuple for auto-resize
        provider: Optional provider name to use provider-specific limits

    Returns:
        ProcessedImage with data and mime_type, or None if invalid
    """
    # Get raw bytes
    raw_data, mime_type = _get_image_bytes(image)
    if raw_data is None:
        return None

    # Determine max dimensions
    if max_dimensions is None and provider is not None:
        max_dimensions = PROVIDER_MAX_DIMENSIONS.get(provider, DEFAULT_MAX_DIMENSIONS)

    # Resize if needed
    if max_dimensions is not None:
        raw_data = resize_image_if_needed(raw_data, max_dimensions)
        # Re-detect mime type after potential resize
        mime_type = detect_mime_type_from_bytes(raw_data)

    # Encode to base64
    img_data = base64.b64encode(raw_data).decode('utf-8')
    return ProcessedImage(data=img_data, mime_type=mime_type)


def process_images(
    images: List[Union[str, bytes]],
    max_dimensions: Optional[Tuple[int, int]] = None,
    provider: Optional[str] = None
) -> List[ProcessedImage]:
    """
    Process a list of images with optional auto-resize.

    Args:
        images: List of images (file paths, base64 strings, or bytes)
        max_dimensions: Optional (width, height) tuple for auto-resize
        provider: Optional provider name to use provider-specific limits

    Returns:
        List of ProcessedImage objects (invalid images are filtered out)
    """
    result = []
    for image in images:
        processed = process_image(image, max_dimensions=max_dimensions, provider=provider)
        if processed is not None:
            result.append(processed)
    return result


def get_base64_only(
    images: List[Union[str, bytes]],
    max_dimensions: Optional[Tuple[int, int]] = None,
    provider: Optional[str] = None
) -> List[str]:
    """
    Process images and return only the base64 data strings.

    Useful for APIs like Ollama that don't need MIME types.

    Args:
        images: List of images
        max_dimensions: Optional (width, height) tuple for auto-resize
        provider: Optional provider name to use provider-specific limits

    Returns:
        List of base64-encoded image strings
    """
    processed = process_images(images, max_dimensions=max_dimensions, provider=provider)
    return [p.data for p in processed]


def get_max_dimensions_for_provider(provider: str) -> Tuple[int, int]:
    """
    Get the maximum image dimensions for a specific provider.

    Args:
        provider: Provider name (openai, anthropic, vertex_ai, ollama, bedrock)

    Returns:
        Tuple of (max_width, max_height)
    """
    return PROVIDER_MAX_DIMENSIONS.get(provider, DEFAULT_MAX_DIMENSIONS)
