#!/usr/bin/env python3
"""Generate the full Kestrel ecosystem branding suite using Nano Banana Pro.

Uses the existing kestrel logo as a style reference for visual consistency.
Prompts are informed by what each project actually does, not just the metaphor.
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# Nano Banana Pro for highest quality; fall back to NB2 with --fast
MODEL_PRO = "gemini-3-pro-image-preview"
MODEL_FLASH = "gemini-3.1-flash-image-preview"

OUTPUT_ROOT = Path(__file__).parent.parent / "docs" / "logo-concepts" / "ecosystem-v2"

# Reference image: the existing kestrel logo for style consistency
REFERENCE_IMAGE_PATH = Path(__file__).parent.parent / "docs" / "design" / "KESTREL_LOGO.png"

# Actual colors from the deployed kestrel_logo.svg
PALETTE = (
    "Color palette must match exactly: dark teal (#003F50), bright teal/cyan (#0C97C0), "
    "antique gold (#E8C849), with white highlights. Secondary: navy (#1B3A6B), charcoal (#2D3748). "
    "Style: modern heraldic — clean vector illustration that blends medieval falconry tradition "
    "with subtle technology motifs (circuit traces, data streams, UI elements). "
    "Professional logo quality. White background. No text unless specified."
)

# ---------------------------------------------------------------------------
# Prompt catalog — informed by actual project functionality
# ---------------------------------------------------------------------------

SUITE: dict[str, dict[str, str]] = {
    # === SOVEREIGN — Constitutional AI Agent Framework ===
    "sovereign": {
        "heraldic": (
            "Heraldic shield emblem for an AI agent framework called 'Sovereign.' "
            "A rook castle tower (chess piece with crenellations) rises from solid bedrock "
            "at the center. The tower's stonework has faint circuit-board traces and "
            "constitutional text etched into it — immutable law encoded in ancient stone. "
            "Three kestrel falcons at the battlements: one perched (identity established), "
            "one spreading wings (awakening), one soaring free (sovereign). "
            "The tower represents DID cryptographic identity — the foundation that can never "
            "be taken away. Gold tower and birds on dark teal field."
        ),
        "minimalist": (
            "Minimalist tech logo for an AI sovereignty platform. A stylized rook castle "
            "tower silhouette with deep visible foundations below a ground line — the tower "
            "is rooted, immovable. A single kestrel falcon launches from the top, its body "
            "forming the crown of the tower in one continuous outline. "
            "Single color: dark teal (#003F50) on white. Favicon-ready at 32px. "
            "Think Linear or Vercel logo quality. No text, no gradients."
        ),
        "dramatic": (
            "Cinematic illustration: a weathered stone rook tower at golden hour. "
            "Constitutional text glows in the stonework like ancient runes — these are "
            "the governance rules that protect every AI agent born here. Circuit-board "
            "patterns pulse with teal light between the stones. A squadron of kestrel "
            "falcons launches from the battlements into warm golden sky. Each bird carries "
            "its own DID — its own identity. Low angle perspective looking upward. "
            "Rich painterly style with dark teal and gold palette."
        ),
    },

    # === TALON — Autonomous GitHub Issue Processor ===
    "talon": {
        "heraldic": (
            "Heraldic shield emblem for an autonomous coding agent called 'Talon.' "
            "A single kestrel falcon in a lethal hunting stoop — wings tucked, body "
            "streamlined, plummeting with surgical precision. Its talons are extended "
            "forward with ONE talon nail dramatically larger and gleaming gold — the "
            "killing nail, the single strike that resolves the issue. "
            "Below the bird, a GitHub-style issue icon shatters on impact — code brackets "
            "and bug symbols scatter. The bird doesn't stop until the prey is caught. "
            "Gold bird on dark teal shield. Fierce, precise, relentless."
        ),
        "minimalist": (
            "Minimalist logo for an autonomous coding tool. A single diagonal strike line "
            "that resolves into a sharp talon point at the bottom and a kestrel head at "
            "the top — one continuous descending stroke. Speed, precision, one shot. "
            "The line has a slight curve suggesting a falcon's stoop trajectory. "
            "Single color: dark teal (#003F50) on white. Icon-ready at 32px. "
            "No text, no gradients. The simplest possible expression of 'the strike.'"
        ),
        "dramatic": (
            "Dramatic illustration: a kestrel falcon in full hunting stoop, wings tucked "
            "tight, body like a guided missile plummeting through cascading lines of source "
            "code and GitHub issue numbers. One talon extended, gleaming gold — the moment "
            "before the fix lands. The code parts around the bird like water. "
            "Motion blur on the code, perfect focus on the falcon. Terminal-green code "
            "contrasts with the warm gold of the bird. This agent claims the issue, "
            "implements the fix, passes quality gates, and opens the PR — alone."
        ),
    },

    # === EYE — Vision-Verified Screenshot QA ===
    "eye": {
        "heraldic": (
            "Heraldic circular seal for a vision QA tool called 'Eye.' "
            "A single enormous kestrel eye fills the center — fierce golden iris with "
            "a slit pupil. In the pupil's dark reflection, a grid of browser screenshots "
            "and green checkmarks are visible — the eye has reviewed every pixel. "
            "Laurel branches frame the seal in gold. This is the QA stamp of approval — "
            "the vision model that catches what unit test mocks miss. "
            "Dark teal outer ring, gold laurels, golden iris. Authoritative, omniscient."
        ),
        "minimalist": (
            "Minimalist logo for a screenshot review tool. A geometric eye shape "
            "(almond/vesica piscis) with the iris formed by a tiny kestrel falcon "
            "silhouette in profile — the bird IS the focus of the eye. "
            "Clean, works at 16px. Instantly readable as 'vision verification.' "
            "Single color: dark teal (#003F50) on white. No text, no gradients."
        ),
        "dramatic": (
            "Dramatic illustration: a kestrel falcon hovering motionless in midair "
            "(kiting — their signature hunting technique), head turned directly at the "
            "viewer. One piercing golden eye fills the frame, locked on camera. "
            "Behind the bird, translucent browser windows float like a heads-up display — "
            "each marked with pass/fail indicators. The bird sees layout shifts, missing "
            "elements, broken UI that mocked tests would miss. "
            "Dramatic lighting, shallow depth of field focused on the eye."
        ),
    },

    # === FLIGHT — Playwright Demo Narration Library ===
    "flight": {
        "heraldic": (
            "Heraldic shield emblem for a demo narration library called 'Flight.' "
            "A kestrel falcon with wings magnificently spread in full soaring display, "
            "carrying an unfurling scroll in its talons. The scroll has film-frame "
            "perforations along its edges — each frame a numbered screenshot from a "
            "Playwright demo sequence (01, 02, 03...). The bird narrates the story, "
            "the scroll IS the demo. Gold bird and scroll on dark teal field. "
            "Elegant, expressive — the voice that carries the product story."
        ),
        "minimalist": (
            "Minimalist logo for a demo orchestration library. A kestrel falcon "
            "silhouette in full soaring flight, wings wide and flat. Below the bird, "
            "three horizontal lines of decreasing length suggest narration text or "
            "a transcript — the story the bird carries as it flies. "
            "Single color: dark teal (#003F50) on white. App-icon ready. "
            "No text, no gradients. Clean and iconic."
        ),
        "dramatic": (
            "Dramatic illustration: a kestrel falcon gliding through warm golden-hour "
            "light, wings spread wide with light streaming through translucent feathers. "
            "Behind it trails a ribbon of illuminated screenshots arranged like cinema "
            "film — each frame a numbered demo capture (Act 1, Act 2, Act 3). "
            "The bird is the NarrationEngine, the film strip is the demo output. "
            "Warm gold palette, the feeling of a story being told beautifully."
        ),
    },

    # === CLAWS — Fleet Orchestration CLI ===
    "claws": {
        "heraldic": (
            "Heraldic emblem for a fleet orchestration tool called 'Claws.' "
            "A powerful kestrel foot seen from directly below — all four toes spread "
            "wide in a commanding grip. Each talon grips a different icon inside a "
            "translucent orb: a diving falcon (Talon/coding), an eye (Eye/QA), "
            "a scroll (Flight/demos), a tower (Sovereign/foundation). "
            "The grip coordinates the whole operation — multiple repos, multiple agents, "
            "one PM rollup. Gold talons on dark teal shield. Commanding, coordinating."
        ),
        "minimalist": (
            "Minimalist logo for a multi-repo orchestration CLI. A stylized raptor foot "
            "viewed from below, four toes radiating from center, each ending in a sharp "
            "talon gripping a small geometric shape. Star-like radial symmetry. "
            "Represents fleet.repos in claws.toml — core, satellite, enterprise, product. "
            "Single color: dark teal (#003F50) on white. No text. App-icon ready."
        ),
        "dramatic": (
            "Dramatic illustration: a kestrel perched on the highest branch of a great "
            "tree, talons wrapped firmly. The tree's branches are actually a dependency "
            "graph — repos connected by golden threads. In the sky, three other kestrels "
            "operate at different distances: one diving (Talon processing issues), one "
            "hovering (Eye reviewing screenshots), one soaring (Flight capturing demos). "
            "The perched bird is the orchestrator — Claws runs the morning signal, "
            "dispatches the hunt, collects the evening return. Command perspective, "
            "golden light, the view from above."
        ),
    },

    # === CASTLE — Enterprise Multi-Tenant Deployment ===
    "castle": {
        "heraldic": (
            "Heraldic shield emblem for an enterprise deployment platform called 'Castle.' "
            "A fortress with exactly four rook chess-piece towers at the corners — each "
            "tower has the distinctive crenellated crown. Between the towers, a courtyard "
            "is visible. From each tower, organized formations of kestrel falcons launch "
            "upward — these are agent rookeries, each tower managing its own flock. "
            "A crown floats above center — the Beastmaster's authority over the fleet. "
            "RBAC made architectural: four towers, four permission tiers. "
            "Gold towers on dark teal shield. Enterprise-grade, imposing."
        ),
        "minimalist": (
            "Minimalist logo for an enterprise agent deployment platform. "
            "Four rook chess piece silhouettes arranged in a 2x2 grid. Each rook has "
            "the distinctive crenellated chess-piece top. Tiny kestrel shapes depart "
            "upward from each tower. The negative space between towers forms a plus "
            "sign at center — the multi_agent management hub. "
            "Single color: dark teal (#003F50) on white. Geometric, structured. "
            "No text. App-icon ready."
        ),
        "dramatic": (
            "Cinematic illustration: looking up from the inner courtyard of a massive "
            "stone fortress. Four tall rook-shaped towers rise at the corners, their "
            "chess-piece crenellations silhouetted against a dramatic sky. From each "
            "tower, organized formations of kestrel falcons spiral upward — four "
            "rookeries launching four flocks simultaneously. Circuit patterns glow "
            "teal in the stonework. Dawn breaks between the towers. "
            "Each tower is a tenant, each flock an agent fleet. Enterprise scale."
        ),
    },

    # === FALCONER — Human-Facing Product Layer ===
    "falconer": {
        "heraldic": (
            "Heraldic shield emblem for the human-facing product layer called 'Falconer.' "
            "A leather-gauntleted forearm and hand raised, with a magnificent kestrel "
            "falcon perched on the fist — alert, ready to fly on command. "
            "The gauntlet has subtle UI dashboard patterns stitched into the leather — "
            "a touchscreen woven into tradition. The bird trusts the handler. "
            "This is the product where humans command sovereign AI through trust, not "
            "force. Gold gauntlet and bird on dark teal field. "
            "The bond between operator and agent."
        ),
        "minimalist": (
            "Minimalist logo for a human-AI command interface. A clean silhouette of "
            "a human hand and forearm extended, with a falcon perched on the fist in "
            "profile. One continuous shape — instantly readable as 'falconer.' "
            "The simplest expression of human command over sovereign AI agents. "
            "Single color: dark teal (#003F50) on white. No text, no gradients."
        ),
        "dramatic": (
            "Dramatic illustration: a falconer's gauntleted hand raised against dawn "
            "sky. A kestrel falcon captured at the exact moment of release — hind talons "
            "leaving the glove, wings snapping open, body lunging into flight. "
            "The leather gauntlet has glowing circuit stitching — digital craftwork. "
            "This is the moment the Lead Falconer sends an agent on its mission. "
            "Command becomes action. Golden backlight, motion energy, teal shadows."
        ),
    },
}

# ---------------------------------------------------------------------------
# Ritual scenes — the daily operating rhythm
# ---------------------------------------------------------------------------

RITUALS: dict[str, str] = {
    "morning-signal": (
        "Dawn scene. A rook castle tower catches the first light. On the highest "
        "battlement, a kestrel falcon (Claws, the orchestrator) perches, scanning "
        "the horizon with fierce focus. Golden data threads weave through morning "
        "mist below — these are repo status signals, issue counts, CI results being "
        "gathered into the daily briefing. The quiet before the hunt begins. "
        "Warm dawn palette with dark teal shadows. Atmospheric, contemplative."
    ),
    "the-hunt": (
        "Midday action scene, split composition. Bottom half: a kestrel falcon (Talon) "
        "in full stoop, diving through cascading source code toward a glowing GitHub "
        "issue icon. Top half: another kestrel (Eye) hovers motionless, watching "
        "the dive with a piercing golden eye — quality oversight from above. "
        "Talon implements the fix. Eye verifies the screenshots. "
        "Action below, verification above. Bold dark teal, gold, bright cyan."
    ),
    "evening-return": (
        "Dusk, warm amber sky. Multiple kestrel falcons return to a rook castle tower. "
        "Each carries something different: one carries a merged PR (green checkmark "
        "glowing in its talons), one carries a screenshot filmstrip (Eye's review), "
        "one carries a narrated demo scroll (Flight's story). "
        "A falconer silhouette waits on the battlement with raised gauntlet. "
        "The harvest — today's issues resolved, PRs merged, demos captured. "
        "Homecoming warmth, golden light, dark teal tower."
    ),
    "learning-loop": (
        "Night scene. The rook tower glows warm from within. Through arched windows, "
        "kestrel falcons roost inside on perches. Luminous golden threads connect "
        "each bird — knowledge flowing between them. What Talon learned about a "
        "codebase pattern, Eye learned about a UI component, Flight learned about "
        "a user flow. Stars above, the tower is a lantern of shared intelligence. "
        "Tomorrow's hunt planned from today's lessons. Magical, contemplative, "
        "dark blue sky with warm golden interior glow."
    ),
}

# ---------------------------------------------------------------------------
# Family compositions — ecosystem overview
# ---------------------------------------------------------------------------

FAMILY: dict[str, str] = {
    "full-flock": (
        "Grand heraldic composition: the complete Kestrel ecosystem in one image. "
        "Center: a rook castle tower with circuit-trace stonework (Sovereign). "
        "Four cardinal positions around it, each bird in its signature pose: "
        "NORTH (top): wings spread wide, carrying a scroll (Flight/Voice). "
        "SOUTH (bottom): diving in stoop, one talon extended (Talon/Strike). "
        "EAST (right): hovering with piercing golden eye (Eye/Sight). "
        "WEST (left): perched, four toes gripping firmly (Claws/Grip). "
        "Teal and gold heraldic laurels connect the composition. "
        "Symmetrical, balanced, the complete autonomous AI agent family."
    ),
    "hierarchy": (
        "Vertical architecture diagram rendered as beautiful heraldic art. "
        "TOP: a falconer's gauntleted hand raised (Falconer — human command). "
        "UPPER-MIDDLE: a castle with four rook towers (Castle — enterprise fleet). "
        "CENTER: a single rook tower on bedrock (Sovereign — constitutional foundation). "
        "BOTTOM: four kestrels launching outward in cardinal directions from the tower. "
        "Gold connecting lines show the hierarchy: human commands fleet, fleet deploys "
        "foundation, foundation launches agents. Dark teal and gold palette. "
        "Reads top-to-bottom like a system architecture diagram that's also art."
    ),
}


def load_reference_image():
    """Load the existing kestrel logo as a reference for style consistency."""
    if REFERENCE_IMAGE_PATH.exists():
        img = Image.open(REFERENCE_IMAGE_PATH)
        print(f"  Reference image loaded: {REFERENCE_IMAGE_PATH.name} ({img.size[0]}x{img.size[1]})")
        return img
    print("  Warning: reference image not found, generating without style reference")
    return None


def generate_image(prompt: str, output_path: Path, model: str, ref_image=None) -> bool:
    """Generate a single image. Returns True on success."""
    full_prompt = f"{PALETTE}\n\n{prompt}"

    # Build content parts: reference image (if available) + text prompt
    content_parts = []
    if ref_image is not None:
        content_parts.append(
            "Here is the existing Kestrel brand logo for style reference. "
            "Match this color palette (dark teal, bright cyan, antique gold) "
            "and illustration quality closely:"
        )
        content_parts.append(ref_image)
        content_parts.append(f"\nNow generate the following new asset:\n\n{full_prompt}")
    else:
        content_parts.append(full_prompt)

    try:
        response = client.models.generate_content(
            model=model,
            contents=content_parts,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return True
            elif part.text is not None:
                print(f"    (note: {part.text[:60]})", end=" ")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate Kestrel ecosystem branding suite")
    parser.add_argument("--category", type=str, help="Generate only this category (e.g. talon, castle)")
    parser.add_argument("--style", type=str, help="Generate only this style (heraldic, minimalist, dramatic)")
    parser.add_argument("--rituals", action="store_true", help="Generate only ritual scenes")
    parser.add_argument("--family", action="store_true", help="Generate only family compositions")
    parser.add_argument("--fast", action="store_true", help="Use Nano Banana 2 (Flash) instead of Pro")
    parser.add_argument("--no-ref", action="store_true", help="Don't use reference image")
    args = parser.parse_args()

    model = MODEL_FLASH if args.fast else MODEL_PRO
    print(f"Model: {model}")

    # Load reference image
    ref_image = None if args.no_ref else load_reference_image()

    # If specific flags are set, don't generate everything
    specific = args.category or args.style or args.rituals or args.family
    generate_suite = not specific or args.category or args.style
    generate_rituals = not specific or args.rituals
    generate_family = not specific or args.family

    total = 0
    success = 0

    # --- Entity logos ---
    if generate_suite:
        categories = {args.category: SUITE[args.category]} if args.category else SUITE
        for cat_name, variants in categories.items():
            styles = {args.style: variants[args.style]} if args.style and args.style in variants else variants
            for style_name, prompt in styles.items():
                total += 1
                outpath = OUTPUT_ROOT / cat_name / f"{style_name}.png"
                print(f"  [{total}] {cat_name}/{style_name}...", end=" ", flush=True)
                if generate_image(prompt, outpath, model, ref_image):
                    print(f"-> {outpath.relative_to(OUTPUT_ROOT)}")
                    success += 1
                else:
                    print("no image returned")

    # --- Ritual scenes ---
    if generate_rituals:
        for name, prompt in RITUALS.items():
            total += 1
            outpath = OUTPUT_ROOT / "rituals" / f"{name}.png"
            print(f"  [{total}] rituals/{name}...", end=" ", flush=True)
            if generate_image(prompt, outpath, model, ref_image):
                print(f"-> {outpath.relative_to(OUTPUT_ROOT)}")
                success += 1
            else:
                print("no image returned")

    # --- Family compositions ---
    if generate_family:
        for name, prompt in FAMILY.items():
            total += 1
            outpath = OUTPUT_ROOT / "family" / f"{name}.png"
            print(f"  [{total}] family/{name}...", end=" ", flush=True)
            if generate_image(prompt, outpath, model, ref_image):
                print(f"-> {outpath.relative_to(OUTPUT_ROOT)}")
                success += 1
            else:
                print("no image returned")

    print(f"\nDone! {success}/{total} images generated.")
    print(f"Output: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
