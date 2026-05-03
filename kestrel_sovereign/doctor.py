"""``kestrel doctor`` — diagnose readiness without making any changes.

The doctor produces a structured :class:`DoctorReport` that the CLI
formats. It is also reused as the ``verify`` step at the end of
``kestrel setup``.

Checks performed (v1):

  - ``KESTREL_DATA_KEY`` set in ``.env``
  - ``[llm]`` section present with a non-empty ``route_priority``
  - For each cloud route in ``route_priority``, the matching
    ``api_key_env`` is set in ``.env``
  - At least one agent registered in ``rookery.toml``
  - For each registered agent, ``kestrel_prime.db`` exists

This is deliberately minimal. We avoid reaching out to Ollama / OpenAI
in v1 — that's flaky in CI and out of scope for "is the config sane?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kestrel_sovereign.rookery.config import ROOKERY_CONFIG_FILENAME, RookeryConfig
from kestrel_sovereign.setup.env_file import read_env
from kestrel_sovereign.setup.toml_file import read_toml


@dataclass
class DoctorReport:
    """Outcome of a doctor run.

    Each entry in ``ok`` / ``warn`` / ``fail`` is a single line a user
    can act on. ``fail`` makes ``ready`` False; ``warn`` is informational.
    """

    ok: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)
    fail: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.fail


def diagnose(project_dir: Path) -> DoctorReport:
    """Run all v1 readiness checks and return a structured report."""
    report = DoctorReport()
    env_path = project_dir / ".env"
    toml_path = project_dir / "kestrel.toml"
    rookery_path = project_dir / ROOKERY_CONFIG_FILENAME

    env = read_env(env_path)
    config = read_toml(toml_path)

    _check_data_key(env, env_path, report)
    _check_llm(config, env, toml_path, report)
    _check_rookery(rookery_path, project_dir, report)

    return report


def format_report(report: DoctorReport) -> str:
    """Render a report as human-readable text suitable for stdout."""
    lines: list[str] = []
    for msg in report.ok:
        lines.append(f"  ✅ {msg}")
    for msg in report.warn:
        lines.append(f"  ⚠️  {msg}")
    for msg in report.fail:
        lines.append(f"  ❌ {msg}")
    if report.ready:
        lines.append("")
        lines.append("Ready. Start with: kestrel start")
    else:
        lines.append("")
        lines.append("Not ready. Fix the items above, or run: kestrel setup")
    return "\n".join(lines)


def _check_data_key(env: dict, env_path: Path, report: DoctorReport) -> None:
    if env.get("KESTREL_DATA_KEY"):
        report.ok.append("KESTREL_DATA_KEY is set")
    else:
        report.fail.append(f"KESTREL_DATA_KEY missing in {env_path}")


def _check_llm(
    config: dict, env: dict, toml_path: Path, report: DoctorReport
) -> None:
    llm = config.get("llm") or {}
    priority = llm.get("route_priority") or []
    if not priority:
        report.fail.append(
            f"[llm] route_priority is empty in {toml_path} — no provider configured"
        )
        return

    report.ok.append(f"[llm] route_priority: {', '.join(priority)}")
    vendors = llm.get("vendors") or {}
    for route_id in priority:
        vendor_key, _, route_key = route_id.partition(":")
        vendor = vendors.get(vendor_key) or {}
        route = (vendor.get("routes") or {}).get(route_key) or {}
        api_key_env = route.get("api_key_env")
        if api_key_env and not env.get(api_key_env):
            report.fail.append(
                f"{api_key_env} not set in .env (required for {route_id})"
            )
        elif api_key_env:
            report.ok.append(f"{api_key_env} set for {route_id}")


def _check_rookery(
    rookery_path: Path, project_dir: Path, report: DoctorReport
) -> None:
    rookery = RookeryConfig.load(rookery_path, auto_discover_fallback=False)
    agents = rookery.get_local_agents()
    if not agents:
        report.fail.append(
            f"No local agents in {rookery_path} — run `kestrel setup agent`"
        )
        return

    report.ok.append(f"{len(agents)} agent(s) registered: {', '.join(agents.keys())}")
    for name, cfg in agents.items():
        db_path = (project_dir / cfg.data_dir / "kestrel_prime.db").resolve()
        if db_path.exists():
            report.ok.append(f"{name}: kestrel_prime.db present")
        else:
            report.fail.append(
                f"{name}: kestrel_prime.db missing at {db_path} — re-run inception"
            )
