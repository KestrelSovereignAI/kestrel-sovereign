#!/usr/bin/env python3
"""Generate an OKF index of Kestrel demo and visual-review evidence."""

from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMOS_ROOT = PROJECT_ROOT / "demos"
OUTPUT = PROJECT_ROOT / "docs" / "generated" / "DEMO_EVIDENCE.md"


@dataclass(frozen=True)
class DemoEvidence:
    name: str
    demo_script: Path
    eye_config: Path | None
    narration: Path | None
    screenshot_dir: str
    test_cmd: str
    screenshot_count: int

    @property
    def rel_demo_script(self) -> str:
        return "/" + self.demo_script.relative_to(PROJECT_ROOT).as_posix()

    @property
    def rel_eye_config(self) -> str:
        if self.eye_config is None:
            return ""
        return "/" + self.eye_config.relative_to(PROJECT_ROOT).as_posix()

    @property
    def rel_narration(self) -> str:
        if self.narration is None:
            return ""
        return "/" + self.narration.relative_to(PROJECT_ROOT).as_posix()


def read_eye_config(path: Path) -> tuple[str, str, int]:
    if not path.exists():
        return "", "", 0
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    eye = data.get("eye", {})
    screenshots = data.get("eye", {}).get("screenshots")
    if screenshots is None:
        screenshots = data.get("screenshots", [])
    return (
        str(eye.get("screenshot_dir", "")),
        str(eye.get("test_cmd", "")),
        len(screenshots) if isinstance(screenshots, list) else 0,
    )


def discover_demo_evidence(root: Path = DEMOS_ROOT) -> list[DemoEvidence]:
    demos: list[DemoEvidence] = []
    for demo_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if demo_dir.name in {"shared", "TEMPLATE"}:
            continue
        demo_script = demo_dir / "demo.cjs"
        if not demo_script.exists():
            continue

        eye_config = demo_dir / "eye.toml"
        screenshot_dir, test_cmd, screenshot_count = read_eye_config(eye_config)
        narration = demo_dir / "narration.md"
        demos.append(
            DemoEvidence(
                name=demo_dir.name,
                demo_script=demo_script,
                eye_config=eye_config if eye_config.exists() else None,
                narration=narration if narration.exists() else None,
                screenshot_dir=screenshot_dir,
                test_cmd=test_cmd,
                screenshot_count=screenshot_count,
            )
        )
    return demos


def build_frontmatter(generated_at: datetime) -> str:
    metadata = {
        "type": "Generated Reference",
        "title": "Kestrel Demo Evidence Index",
        "description": "Generated inventory of executable demos, kestrel-flight scripts, and kestrel-eye review configs.",
        "resource": "/docs/generated/DEMO_EVIDENCE.md",
        "tags": ["demos", "generated-docs", "kestrel-flight", "kestrel-eye"],
        "timestamp": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "generated",
        "generated": True,
        "canonical": False,
        "source": "/demos/",
        "generator": "scripts/generate_demo_evidence_docs.py",
    }
    dumped = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{dumped}\n---\n\n"


def render(demos: list[DemoEvidence], *, generated_at: datetime) -> str:
    lines = [
        build_frontmatter(generated_at).rstrip(),
        "",
        "# Kestrel Demo Evidence Index",
        "",
        "This generated reference links human demo docs to executable `kestrel-flight` demos and `kestrel-eye` visual review configs.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "uv run python scripts/generate_demo_evidence_docs.py",
        "```",
        "",
        "| Demo | Script | Eye config | Expected screenshots | Test command | Narration |",
        "|---|---|---|---:|---|---|",
    ]
    for demo in demos:
        eye = demo.rel_eye_config or "none"
        narration = demo.rel_narration or "none"
        test_cmd = demo.test_cmd or "not configured"
        lines.append(
            f"| `{demo.name}` | `{demo.rel_demo_script}` | `{eye}` | {demo.screenshot_count} | `{test_cmd}` | `{narration}` |"
        )
    lines.extend(
        [
            "",
            "## Talon Gate",
            "",
            "Use a demo-specific gate when a PR changes a documented UI workflow:",
            "",
            "```talon-verify",
            "kestrel demo run technical",
            "kestrel-eye review --config demos/technical/eye.toml",
            "```",
            "",
            "Keep generated screenshots and video in `demo-output/` or CI artifacts unless a PR intentionally updates stable documentation evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def check_output(expected: str, path: Path = OUTPUT) -> int:
    if not path.exists():
        print(f"ERROR: {path.relative_to(PROJECT_ROOT)} does not exist", file=sys.stderr)
        return 1
    actual = path.read_text(encoding="utf-8")
    if actual == expected:
        print("Demo evidence index is current.")
        return 0
    diff = difflib.unified_diff(
        actual.splitlines(),
        expected.splitlines(),
        fromfile=str(path.relative_to(PROJECT_ROOT)),
        tofile="generated",
        lineterm="",
    )
    print("\n".join(diff), file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the generated index is stale")
    args = parser.parse_args()

    # Fixed timestamp keeps --check deterministic; this file is an inventory
    # snapshot, not proof that demos were executed at generation time.
    content = render(
        discover_demo_evidence(),
        generated_at=datetime(2026, 6, 18, tzinfo=timezone.utc),
    )
    if args.check:
        return check_output(content)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
