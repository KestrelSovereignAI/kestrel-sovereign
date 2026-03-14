#!/usr/bin/env python3
"""Iterate on minimalist and heraldic logo concepts — round 2."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-3.1-flash-image-preview"

output_dir = Path(__file__).parent.parent / "docs" / "logo-concepts" / "v2"
output_dir.mkdir(parents=True, exist_ok=True)

prompts = {
    # === MINIMALIST VARIATIONS ===
    "min-v1-upward": (
        "Minimalist single-color logo mark. A rook castle tower (chess piece silhouette with "
        "crenellations) with a kestrel falcon bursting upward from the top, wings swept back "
        "in a steep climb. The bird and tower share a continuous outline — the bird IS the top "
        "of the tower. Dark slate blue (#2C3E50) on white. No text, no gradients. "
        "Extremely clean vector style, suitable for 32px favicon. Professional logo design."
    ),
    "min-v2-negative": (
        "Minimalist logo: a solid dark slate blue (#2C3E50) rook castle tower shape. "
        "A kestrel falcon silhouette is CUT OUT of the tower as negative white space, "
        "wings spread wide, visible through the tower body. The bird is revealed by its absence. "
        "Single color on white background. Ultra-clean geometric vector style, "
        "no gradients, no text. Think Apple or Nike level simplicity."
    ),
    "min-v3-circle": (
        "Minimalist circular logo mark. Inside a perfect circle: a stylized rook castle tower "
        "with a kestrel falcon launching from its battlements. The composition fills the circle "
        "tightly. Single color: dark slate blue (#2C3E50) on white. Balanced negative space. "
        "Clean geometric vector, no text, no gradients. App icon ready. "
        "Think GitHub's octocat level of iconic simplicity."
    ),
    "min-v4-merged": (
        "Minimalist logo: the silhouette of a rook chess piece (castle tower) that also reads "
        "as a kestrel falcon in profile when you look at it differently — an optical illusion "
        "where tower and bird are the same shape. Rubin vase style ambiguity. "
        "Single color dark slate (#2C3E50) on white. Ultra-minimal, geometric, vector style. "
        "No text. The genius is in the dual reading of one simple shape."
    ),
    "min-v5-launching": (
        "Minimalist tech logo: a geometric rook castle tower with three small kestrel falcon "
        "silhouettes launching upward in a diagonal formation from the tower top, like jets "
        "taking off from an aircraft carrier. The birds get smaller as they go higher, "
        "suggesting distance and acceleration. Dark slate blue (#2C3E50) on white. "
        "Clean vector, no gradients, no text. Startup logo quality."
    ),

    # === HERALDIC VARIATIONS ===
    "herald-v1-sharp": (
        "Heraldic coat of arms logo with a modern edge. A pointed shield shape contains a "
        "rook castle tower at center. Three kestrel falcons with sharp, detailed raptor features "
        "(hooked beaks, fierce eyes, pointed wings) launch from the battlements in a fan pattern. "
        "Color palette: deep royal blue (#1B3A6B) field, gold (#D4A843) for the tower and birds. "
        "Clean heraldic illustration, ornate but not cluttered. White background, no text."
    ),
    "herald-v2-modern": (
        "Modern heraldic emblem: a simplified shield shape with a rook castle tower. "
        "Two kestrel falcons flank the tower symmetrically, wings raised in a displayed pose "
        "(like heraldic eagles). Clean flat-color heraldic style — no 3D effects. "
        "Navy blue (#1a2744) and metallic gold (#c9a84c). Sharp geometric edges on the shield. "
        "Professional and authoritative, like a university or government seal but modern. "
        "White background, no banner, no text, no crown."
    ),
    "herald-v3-seal": (
        "Circular heraldic seal design. A rook castle tower at center with kestrel falcons "
        "in flight around it, forming a circular pattern. Reminiscent of national seals or "
        "military unit insignias. Deep navy and gold color scheme. "
        "Outer ring with subtle geometric pattern (not text). Inner scene is the tower and birds. "
        "Professional, authoritative, high quality heraldic illustration. White background."
    ),
    "herald-v4-crest": (
        "A family crest style logo: a rook castle tower crowned with a perched kestrel falcon "
        "at the very top (the crest element). Two more kestrels as supporters on either side, "
        "wings spread. Small shield at the base of the tower. Color: royal blue and antique gold. "
        "Classical heraldic proportions but clean modern rendering. "
        "No text, no motto banner. White background. Elegant and regal."
    ),
    "herald-v5-minimal": (
        "Heraldic logo that bridges traditional and modern: a simple shield outline containing "
        "just a rook tower silhouette with a single kestrel in flight above it. "
        "Minimal detail — the power comes from clean shapes and strong composition. "
        "Two colors only: deep blue (#1B3A6B) and gold (#D4A843). Flat vector style. "
        "No ornamental flourishes. The sophistication is in restraint. White background, no text."
    ),
}

print(f"Generating {len(prompts)} variations with Nano Banana 2...")
print(f"Output directory: {output_dir}\n")

for name, prompt in prompts.items():
    print(f"  {name}...", end=" ", flush=True)
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
                outpath = output_dir / f"{name}.png"
                image.save(str(outpath))
                print(f"-> {outpath.name}")
                saved = True
            elif part.text is not None:
                print(f"(note: {part.text[:60]})", end=" ")
        if not saved:
            print("no image returned")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nDone! {output_dir}/")
