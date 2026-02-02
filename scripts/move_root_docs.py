#!/usr/bin/env python3
"""Move markdown documents from repo root into the docs/ hierarchy.

Run from repository root: `python scripts/move_root_docs.py`
"""
from __future__ import annotations

import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DOCS = BASE / "docs"

MAPPING = {
    "ADVISOR_CONVERSATION_GUIDE.md": DOCS / "outreach",
    "ADVISOR_OUTREACH_MESSAGES.md": DOCS / "outreach",
    "DEMO_SCRIPT.md": DOCS / "demos",
    "DEVELOPMENT_ROADMAP.md": DOCS / "planning",
    "EDUCATION_IMPLEMENTATION_ROADMAP.md": DOCS / "education",
    "GDPR_CCPA_IMPACT_ANALYSIS.md": DOCS / "legal",
    "INTEGRATED_PROJECT_ROADMAP.md": DOCS / "planning",
    "IP_PROTECTION_STRATEGY.md": DOCS / "legal",
    "JIM_ETHICS_DEMO_ASSESSMENT.md": DOCS / "assessments",
    "KESTREL_FOUNDING_WHY.md": DOCS / "vision",
    "KESTREL_MVP_ASSESSMENT.md": DOCS / "assessments",
    "KESTREL_PRICING_COMPETITIVE_ANALYSIS.md": DOCS / "business",
    "LEGAL_ENTITY_FORMATION_PLAN.md": DOCS / "legal",
    "LEGAL_FORMATION_TODO.md": DOCS / "legal",
    "MICHAEL_LABROAD_BUSINESS_OVERVIEW.md": DOCS / "assessments",
    "MICHAEL_RUSSELL_ADVISOR_ASSESSMENT.md": DOCS / "assessments",
    "MICHAEL_RUSSELL_BUSINESS_OVERVIEW.md": DOCS / "assessments",
    "MILESTONES_FUNDING_ROADMAP_DETAILED.md": DOCS / "business",
    "MVP_ASSESSMENT_PREP.md": DOCS / "assessments",
    "MVP_PROGRESS_UPDATE_NOV2025.md": DOCS / "planning",
    "NDA_TEMPLATE.md": DOCS / "legal",
    "OUTREACH_EXECUTION_PLAN.md": DOCS / "outreach",
    "PBC_ANALYSIS_FOR_KESTREL.md": DOCS / "business",
    "ROGER_CROOKS_ADVISOR_SCRIPT.md": DOCS / "outreach",
    "ROGER_NEXT_STEPS_POST_NDA.md": DOCS / "outreach",
    "STAKEHOLDER_ENGAGEMENT_HUB_VISIONARY_PILOT_UPDATE.md": DOCS / "outreach",
    "STAKEHOLDER_ENGAGEMENT_HUB.md": DOCS / "outreach",
    "STARTUP_ATTORNEY_OUTREACH.md": DOCS / "outreach",
    "SUNDBY_OUTREACH_STRATEGY.md": DOCS / "outreach",
    "USER_EDUCATION_STRATEGY.md": DOCS / "education",
}


def move_files() -> None:
    moved = 0
    skipped = 0

    for filename, target_dir in MAPPING.items():
        source = BASE / filename
        if not source.exists():
            print(f"[skip] {filename} not found at repo root")
            skipped += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / source.name
        print(f"[move] {source.relative_to(BASE)} -> {destination.relative_to(BASE)}")
        shutil.move(str(source), str(destination))
        moved += 1

    print(f"\nDone. {moved} file(s) moved, {skipped} skipped.")


if __name__ == "__main__":
    move_files()
