# Kestrel Ecosystem Brand Guide

## The Metaphor

Kestrel is a falconry ecosystem. Every name is deliberate — rooted in real falconry anatomy, tradition, and the relationship between human and raptor.

> "To know the true name of a thing is to have power over it."

---

## Color Palette

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| **Primary** | Navy Blue | `#1B3A6B` | Shield backgrounds, text, headers |
| **Secondary** | Antique Gold | `#D4A843` | Birds, towers, accents, highlights |
| **Tertiary** | Deep Teal | `#006D77` | Technology overlays, circuit patterns |
| **Accent** | Electric Blue | `#00A6FB` | Digital elements, data streams |
| **Neutral** | Charcoal | `#2D3748` | Body text, subtle backgrounds |

---

## The Foundation

### Kestrel Sovereign — *The Roost / The Demesne*

The constitutional ground where agents exist. A rook tower rooted in bedrock — ancient architecture with circuit-board runes glowing in the stonework. Law and technology fused at the foundation.

**Visual motif:** Rook chess-piece tower with constitutional text etched in stone. Kestrels launch from its battlements.

---

## The Birds

Each bird is a sub-project with a true name drawn from kestrel anatomy.

### Talon — *The Strike*

The autonomous GitHub issue processor. The single killing nail — precision execution. It doesn't stop until the issue is resolved.

**Visual motif:** A kestrel in full hunting stoop, one talon nail dramatically larger than the rest. Lethal, focused, descending.

### Eye — *The Sight*

Vision-verified E2E screenshot QA. Sees what others miss. The quality gate that catches what mocks hide.

**Visual motif:** A piercing kestrel eye with golden iris. The pupil reflects browser windows and screenshots. Omniscient verification.

### Flight — *The Voice*

Playwright demo narration library. Carries the story. Every demo, every screenshot sequence, every narrative is Flight's domain.

**Visual motif:** A kestrel soaring with wings fully spread, carrying an unfurling scroll with film-frame edges. The narrator.

### Claws — *The Grip*

Fleet orchestration CLI. The whole foot — multiple toes gripping, kneading, processing. Holds the entire operation together across repositories.

**Visual motif:** A kestrel foot from below, four toes radiating outward, each gripping a piece of the ecosystem. Coordination.

---

## The Architecture

### Kestrel Castle — *The Enterprise Fortress*

The enterprise deployment layer. A castle with four rook-shaped towers (the chess piece pun). Each tower is a multi_agent that launches falcon agents. Multi-tenant fleet management, RBAC, governance policies.

**Visual motif:** Four chess-rook towers in square formation, kestrels launching from each. A crown above the center. Enterprise scale.

### Kestrel Falconer — *The Handler*

The human-facing product layer. The human at the top of the flock. Command through trust, not control.

**Visual motif:** A gauntleted hand with a perched kestrel about to launch. Digital stitching in the leather. The moment command becomes action.

---

## The Handlers (Roles)

| Role | Title | Description |
|------|-------|-------------|
| **Beastmaster** | Architect | Built the birds. Speaks to them directly. |
| **Master Falconer** | Trainer | Trains falconers, decides which birds exist. |
| **Lead Falconer** | Operator | Flies all birds daily, runs the operation. |
| **Falconer** | User | Flies specific birds for specific purposes. |

---

## The Rituals

| Ritual | Time | Description |
|--------|------|-------------|
| **Morning Signal** | Dawn | Claws scans repos, generates daily briefing. Data threads in the mist. |
| **The Hunt** | Day | Talon dives, Eye watches from above. Action and oversight. |
| **Evening Return** | Dusk | Birds return to the tower carrying trophies — fixes, verifications, stories. |
| **The Learning Loop** | Night | Tower glows from within. Knowledge flows between roosting birds. Tomorrow's hunt planned. |

---

## Logo Variants & Usage

Each entity has three visual treatments:

| Variant | Use Case | Characteristics |
|---------|----------|-----------------|
| **Heraldic** | README headers, formal brand, presentations | Coat-of-arms/crest, shield, gold on navy |
| **Minimalist** | Favicons, app icons, small format | Single color (#2C3E50), geometric, 32px ready |
| **Dramatic** | Documentation art, splash screens, pitch decks | Cinematic illustration, painterly, atmospheric |

### When to use which

- **GitHub README** → Heraldic (authoritative, eye-catching at scroll speed)
- **Favicon / App icon** → Minimalist (must read at 16-32px)
- **Documentation page** → Dramatic (sets mood, draws readers in)
- **Slide deck** → Dramatic for full-bleed, Heraldic for logo placement
- **Social media avatar** → Heraldic (strong at small circular crop)

---

## File Structure

```
docs/logo-concepts/ecosystem/
    sovereign/      heraldic.png  minimalist.png  dramatic.png
    talon/          heraldic.png  minimalist.png  dramatic.png
    eye/            heraldic.png  minimalist.png  dramatic.png
    flight/         heraldic.png  minimalist.png  dramatic.png
    claws/          heraldic.png  minimalist.png  dramatic.png
    castle/         heraldic.png  minimalist.png  dramatic.png
    falconer/       heraldic.png  minimalist.png  dramatic.png
    rituals/        morning-signal.png  the-hunt.png  evening-return.png  learning-loop.png
    family/         full-flock.png  hierarchy.png
    ecosystem-grid.png
```

---

## Generation

All images generated via Gemini 3.1 Flash with a shared palette preamble for visual coherence.

```bash
# Full suite (27 images)
python scripts/generate_branding_suite.py

# Single category
python scripts/generate_branding_suite.py --category talon

# Single style across all categories
python scripts/generate_branding_suite.py --style heraldic

# Just rituals or family shots
python scripts/generate_branding_suite.py --rituals
python scripts/generate_branding_suite.py --family

# Rebuild comparison grid
python scripts/make_ecosystem_grid.py
```
