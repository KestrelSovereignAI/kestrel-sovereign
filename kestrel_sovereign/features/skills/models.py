"""Skill data model and (de)serialization.

A Skill is a structured capture of tacit procedural knowledge:
the trigger (when it applies), the steps (what to do), the verification
(how to know it worked), and provenance (where it came from).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import re


@dataclass
class Skill:
    """A reusable procedural skill extracted from a work session."""

    id: str
    title: str
    trigger: str           # When this skill applies (one sentence)
    steps: List[str]       # Ordered procedural steps
    verification: str      # How to know the skill worked
    tags: List[str] = field(default_factory=list)
    source_insight_id: Optional[str] = None  # Originating reflection insight
    source_session_id: Optional[str] = None  # Originating session
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "trigger": self.trigger,
            "steps": list(self.steps),
            "verification": self.verification,
            "tags": list(self.tags),
            "source_insight_id": self.source_insight_id,
            "source_session_id": self.source_session_id,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(
            id=data["id"],
            title=data["title"],
            trigger=data["trigger"],
            steps=list(data.get("steps", [])),
            verification=data.get("verification", ""),
            tags=list(data.get("tags", [])),
            source_insight_id=data.get("source_insight_id"),
            source_session_id=data.get("source_session_id"),
            confidence=float(data.get("confidence", 0.5)),
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
        )

    def to_markdown(self) -> str:
        """Serialize to a skill file: YAML-ish frontmatter + markdown body.

        Intentionally hand-rolled rather than depending on PyYAML — skill files
        are small and the escape surface is trivial.
        """
        fm_lines = [
            "---",
            f"id: {self.id}",
            f"title: {_yaml_scalar(self.title)}",
            f"trigger: {_yaml_scalar(self.trigger)}",
            f"confidence: {self.confidence:.2f}",
            f"created_at: {self.created_at}",
        ]
        if self.tags:
            fm_lines.append(f"tags: [{', '.join(_yaml_scalar(t) for t in self.tags)}]")
        if self.source_insight_id:
            fm_lines.append(f"source_insight_id: {self.source_insight_id}")
        if self.source_session_id:
            fm_lines.append(f"source_session_id: {self.source_session_id}")
        fm_lines.append("---")

        body = [f"# {self.title}", "", "## When to apply", self.trigger, "", "## Steps"]
        for i, step in enumerate(self.steps, 1):
            body.append(f"{i}. {step}")
        body.extend(["", "## Verification", self.verification, ""])
        return "\n".join(fm_lines) + "\n\n" + "\n".join(body)

    @classmethod
    def from_markdown(cls, text: str) -> "Skill":
        """Parse a skill markdown file back into a Skill."""
        fm, body = _split_frontmatter(text)
        meta = _parse_simple_yaml(fm)

        steps: List[str] = []
        verification_lines: List[str] = []
        trigger_lines: List[str] = []
        section = None
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].lower().strip()
                continue
            if stripped.startswith("# "):
                continue
            if section == "when to apply" and stripped:
                trigger_lines.append(stripped)
            elif section == "steps":
                m = re.match(r"^\s*\d+\.\s+(.*)$", line)
                if m:
                    steps.append(m.group(1).strip())
            elif section == "verification" and stripped:
                verification_lines.append(stripped)

        # Frontmatter trigger takes precedence; body trigger is a fallback.
        trigger = meta.get("trigger") or " ".join(trigger_lines)
        verification = " ".join(verification_lines)

        tags_raw = meta.get("tags", "")
        if tags_raw.startswith("[") and tags_raw.endswith("]"):
            tags = [t.strip().strip('"').strip("'") for t in tags_raw[1:-1].split(",") if t.strip()]
        else:
            tags = []

        return cls(
            id=meta.get("id", ""),
            title=meta.get("title", ""),
            trigger=trigger,
            steps=steps,
            verification=verification,
            tags=tags,
            source_insight_id=meta.get("source_insight_id") or None,
            source_session_id=meta.get("source_session_id") or None,
            confidence=float(meta.get("confidence", 0.5)),
            created_at=meta.get("created_at") or datetime.now(timezone.utc).isoformat(),
        )


# =============================================================================
# Helpers — deliberately small, no PyYAML dep
# =============================================================================


def _yaml_scalar(value: str) -> str:
    """Quote a scalar only when the YAML subset we accept would misparse it."""
    if value == "":
        return '""'
    if any(c in value for c in (":", "#", "[", "]", "{", "}", ",", "\n", '"', "'")):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a markdown file into (frontmatter, body)."""
    if not text.startswith("---"):
        return "", text
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        return "", text
    fm = parts[0].lstrip("-").strip("\n")
    body = parts[1].lstrip("\n")
    return fm, body


def _parse_simple_yaml(text: str) -> Dict[str, str]:
    """Parse `key: value` lines — the only YAML shape we emit."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip()
        # Unquote simple quoted strings
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        out[key.strip()] = value
    return out


def normalize_title(title: str) -> str:
    """Normalized title for dedup — lowercased, alnum-only."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def skill_id_from_title(title: str) -> str:
    """Stable ID from a title. SHA-1 of the normalized title, first 12 hex."""
    n = normalize_title(title)
    return "skill_" + hashlib.sha1(n.encode("utf-8")).hexdigest()[:12]
