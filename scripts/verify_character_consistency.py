#!/usr/bin/env python3
"""
Character Consistency Verification using Claude.

Analyzes generated images to verify the same person appears across all scenes.
Uses Claude's vision capabilities to compare facial features.

Usage:
    cd kestrel-sovereign
    uv run python scripts/verify_character_consistency.py

This script reads the verification manifest created by run_lora_consistency_test.py
and uses Claude to analyze character consistency.
"""

import base64
import json
import os
import sys
from pathlib import Path

from kestrel_sovereign.llm.model_selection import resolve_provider_default

# Directory with generated images
TEST_OUTPUT_DIR = Path.home() / "models" / "local-training" / "consistency-test-output"
MANIFEST_PATH = TEST_OUTPUT_DIR / "verification_manifest.json"


def resolve_verification_model() -> str:
    """Resolve the Anthropic model used for consistency verification."""
    return os.getenv("CHARACTER_VERIFY_MODEL") or resolve_provider_default("anthropic")


def get_anthropic_client():
    """Construct the Anthropic client lazily so the script remains importable."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed")
        print("Run: uv add anthropic")
        sys.exit(1)

    return anthropic.Anthropic()


def load_image_as_base64(path: str) -> str:
    """Load image and convert to base64."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def get_image_media_type(path: str) -> str:
    """Get media type from file extension."""
    ext = Path(path).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")


def verify_character_consistency(manifest: dict) -> dict:
    """
    Use Claude to verify character consistency across generated images.

    Returns a verification report with:
    - Overall consistency score (0-100%)
    - Per-image analysis
    - Confidence level
    - Detailed reasoning
    """
    client = get_anthropic_client()

    # Build message with all images
    content = []

    # Add reference image first
    ref_path = manifest.get("reference_image")
    if ref_path and Path(ref_path).exists():
        content.append({
            "type": "text",
            "text": "REFERENCE IMAGE (used for training):"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": get_image_media_type(ref_path),
                "data": load_image_as_base64(ref_path),
            }
        })

    # Add generated images
    for i, img in enumerate(manifest.get("generated_images", [])):
        img_path = img.get("path")
        if img_path and Path(img_path).exists():
            content.append({
                "type": "text",
                "text": f"\nGENERATED IMAGE {i+1} - Scene: {img.get('scene', 'unknown')}"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": get_image_media_type(img_path),
                    "data": load_image_as_base64(img_path),
                }
            })

    # Add analysis prompt
    content.append({
        "type": "text",
        "text": """
TASK: Analyze these images for character consistency.

The first image is the REFERENCE used to train a LoRA model.
The subsequent images were GENERATED using that LoRA with the trigger word in different scenes.

Please analyze:

1. **Facial Feature Comparison**
   - Compare key facial features: eye shape, eye color, nose shape, mouth shape, face shape
   - Note any distinctive features (moles, freckles, dimples, etc.)
   - Assess hair color and style (may vary by scene but underlying color should match)

2. **Identity Consistency Score**
   Rate from 0-100% how confident you are that ALL generated images show the SAME PERSON.
   - 90-100%: Definitely the same person
   - 70-89%: Very likely the same person with minor variations
   - 50-69%: Possibly the same person but notable differences
   - Below 50%: Likely different people or significant inconsistencies

3. **Per-Image Analysis**
   For each generated image, note:
   - Similarity to reference (High/Medium/Low)
   - Any features that match or differ
   - Scene-appropriate variations (e.g., different lighting, angle)

4. **Overall Assessment**
   - Is the LoRA successfully capturing the person's identity?
   - What features are most consistently preserved?
   - What improvements could be made?

Format your response as JSON:
```json
{
    "consistency_score": 85,
    "confidence": "high",
    "same_person": true,
    "facial_features": {
        "preserved": ["eye color", "face shape", "..."],
        "inconsistent": ["...", "..."]
    },
    "per_image_analysis": [
        {"scene": "portrait", "similarity": "high", "notes": "..."},
        {"scene": "casual", "similarity": "medium", "notes": "..."}
    ],
    "overall_assessment": "...",
    "recommendations": ["...", "..."]
}
```
"""
    })

    # Call Claude
    print("Sending images to Claude for analysis...")
    response = client.messages.create(
        model=resolve_verification_model(),
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": content
        }]
    )

    # Parse response
    response_text = response.content[0].text

    # Try to extract JSON from response
    try:
        # Find JSON block
        if "```json" in response_text:
            json_start = response_text.index("```json") + 7
            json_end = response_text.index("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            json_start = response_text.index("{")
            json_end = response_text.rindex("}") + 1
            json_str = response_text[json_start:json_end]
        else:
            json_str = response_text

        result = json.loads(json_str)
        result["raw_response"] = response_text
        return result

    except (json.JSONDecodeError, ValueError):
        return {
            "consistency_score": None,
            "error": "Could not parse JSON response",
            "raw_response": response_text,
        }


def main():
    """Run character consistency verification."""
    print("=" * 60)
    print("Character Consistency Verification")
    print("=" * 60)

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"\nError: Manifest not found at {MANIFEST_PATH}")
        print("Run run_lora_consistency_test.py first to generate images.")
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"\nLoaded manifest: {MANIFEST_PATH}")
    print(f"Reference image: {manifest.get('reference_image')}")
    print(f"Generated images: {len(manifest.get('generated_images', []))}")

    # Verify images exist
    ref_path = manifest.get("reference_image")
    if ref_path and not Path(ref_path).exists():
        print(f"\nError: Reference image not found: {ref_path}")
        sys.exit(1)

    for img in manifest.get("generated_images", []):
        if not Path(img.get("path", "")).exists():
            print(f"\nWarning: Image not found: {img.get('path')}")

    # Run verification
    print("\n" + "-" * 40)
    result = verify_character_consistency(manifest)

    # Display results
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    if result.get("error"):
        print(f"\nError: {result['error']}")
        print("\nRaw response:")
        print(result.get("raw_response", "No response"))
    else:
        score = result.get("consistency_score")
        print(f"\nConsistency Score: {score}%")
        print(f"Confidence: {result.get('confidence', 'unknown')}")
        print(f"Same Person: {'Yes' if result.get('same_person') else 'No'}")

        features = result.get("facial_features", {})
        if features.get("preserved"):
            print(f"\nPreserved Features: {', '.join(features['preserved'])}")
        if features.get("inconsistent"):
            print(f"Inconsistent Features: {', '.join(features['inconsistent'])}")

        print("\nPer-Image Analysis:")
        for analysis in result.get("per_image_analysis", []):
            print(f"  - {analysis.get('scene')}: {analysis.get('similarity')} similarity")
            if analysis.get("notes"):
                print(f"    {analysis['notes'][:80]}...")

        print(f"\nOverall Assessment:")
        print(f"  {result.get('overall_assessment', 'N/A')}")

        if result.get("recommendations"):
            print("\nRecommendations:")
            for rec in result.get("recommendations", []):
                print(f"  - {rec}")

    # Save result
    result_path = TEST_OUTPUT_DIR / "verification_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nFull result saved to: {result_path}")

    # Return exit code based on score
    score = result.get("consistency_score", 0)
    if score and score >= 70:
        print("\n SUCCESS: Character consistency verified!")
        return 0
    elif score and score >= 50:
        print("\n WARNING: Moderate character consistency")
        return 0
    else:
        print("\n FAIL: Poor character consistency")
        return 1


if __name__ == "__main__":
    sys.exit(main())
