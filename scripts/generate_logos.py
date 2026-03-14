#!/usr/bin/env python3
"""Generate Kestrel Sovereign logo concepts using Nano Banana 2 (Gemini 3.1 Flash Image Preview)."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-3.1-flash-image-preview"

output_dir = Path(__file__).parent.parent / "docs" / "logo-concepts"
output_dir.mkdir(parents=True, exist_ok=True)

prompts = {
    "heraldic": (
        "Design a heraldic coat of arms logo. A medieval rook castle tower (chess piece shape) "
        "stands at the center with three kestrel falcons launching upward from its battlements. "
        "One kestrel is perched, one is spreading its wings, one is in full soaring flight. "
        "Gold and deep royal blue color palette. Shield-shaped background. "
        "Clean vector illustration style, professional logo design, symmetrical composition, "
        "white background. The word 'KESTREL' is NOT included — just the emblem."
    ),
    "minimalist": (
        "Minimalist tech company logo: a stylized rook castle tower silhouette with a single "
        "kestrel falcon taking flight from the top. The bird shape emerges from the tower using "
        "clever negative space. Single color: dark slate blue (#2C3E50). Clean geometric lines, "
        "suitable for app icon and favicon. Flat design, white background, no text, no gradients. "
        "Think Stripe or Linear logo quality."
    ),
    "dramatic": (
        "Dramatic illustration: a weathered stone castle rook tower at golden hour dawn. "
        "A squadron of kestrel falcons launches from the battlements into a warm golden sky. "
        "The tower has subtle glowing circuit-board patterns etched into the stonework, "
        "suggesting technology within ancient architecture. Rich painterly style with bold colors — "
        "deep charcoal stone, warm gold sky, brown and slate kestrels with detailed feathers. "
        "Low angle cinematic perspective looking upward. No text."
    ),
    "woodcut": (
        "Black and white woodcut illustration in the style of a medieval printer's mark or "
        "bookplate. A castle rook tower with kestrel falcons departing from the top in flight. "
        "Traditional cross-hatching technique, high contrast black ink on white. "
        "Circular composition like a wax seal. Detailed feather work on the birds, "
        "weathered stone texture on the tower. No text, just the image."
    ),
    "modern_tech": (
        "Modern tech startup logo: an isometric 3D-style rook castle tower acts as a launchpad. "
        "Geometric kestrel falcon shapes fly upward from the tower in a spiral formation, "
        "each trailing a subtle digital particle stream. Gradient from deep navy (#1a1a2e) to "
        "teal (#16a085). Clean vector lines with slight depth/shadow. Professional SaaS aesthetic "
        "like Vercel or Supabase branding. White background, no text."
    ),
    "playful": (
        "A whimsical illustrated logo: a friendly rook castle tower with a face-like window "
        "arrangement, serving as a birdhouse/rookery. Cute but elegant kestrel falcons perch on "
        "and launch from various levels. Warm earth tones — terracotta, sage green, cream, "
        "with accents of gold. Storybook illustration quality, hand-drawn feel with clean lines. "
        "Could be a children's book cover detail. White background, no text."
    ),
}

print(f"Generating {len(prompts)} logo concepts with Nano Banana 2...")
print(f"Output directory: {output_dir}\n")

for name, prompt in prompts.items():
    print(f"  Generating: {name}...", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        saved = False
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                outpath = output_dir / f"kestrel-{name}.png"
                image.save(str(outpath))
                print(f"saved -> {outpath.name}")
                saved = True
            elif part.text is not None:
                print(f"(model note: {part.text[:80]})")
        if not saved:
            print("no image returned")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone! Check {output_dir}/")
