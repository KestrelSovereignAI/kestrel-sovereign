"""Unit tests for kestrel_sovereign.doctor."""

from __future__ import annotations

from pathlib import Path

import toml
from cryptography.fernet import Fernet

from kestrel_sovereign.doctor import diagnose, format_report
from kestrel_sovereign.rookery.config import (
    HostConfig,
    LocalAgentConfig,
    ROOKERY_CONFIG_FILENAME,
    RookeryConfig,
)
from kestrel_sovereign.setup.env_file import write_env
from kestrel_sovereign.setup.toml_file import write_toml


def _seed_ready(tmp_path: Path) -> None:
    """Build a fully-ready project tree."""
    write_env(
        tmp_path / ".env",
        {
            "KESTREL_DATA_KEY": Fernet.generate_key().decode("ascii"),
            "OPENAI_API_KEY": "sk-x",
        },
    )
    write_toml(
        tmp_path / "kestrel.toml",
        {
            "llm": {
                "route_priority": ["openai:api"],
                "vendors": {
                    "openai": {
                        "is_cloud": True,
                        "routes": {
                            "api": {
                                "adapter": "OpenAIAdapter",
                                "api_key_env": "OPENAI_API_KEY",
                            }
                        },
                    }
                },
            }
        },
    )
    rookery = RookeryConfig(
        host=HostConfig(),
        agents={
            "Test": LocalAgentConfig(
                data_dir=Path("agent_data/test"), port=8801, autostart=True
            )
        },
    )
    rookery.save(tmp_path / ROOKERY_CONFIG_FILENAME)
    db_dir = tmp_path / "agent_data" / "test"
    db_dir.mkdir(parents=True)
    (db_dir / "kestrel_prime.db").write_bytes(b"")


def test_doctor_reports_ready_when_everything_set(tmp_path):
    _seed_ready(tmp_path)
    report = diagnose(tmp_path)
    assert report.ready, f"fail={report.fail}"
    assert report.fail == []


def test_doctor_blocks_on_missing_data_key(tmp_path):
    _seed_ready(tmp_path)
    # Wipe .env
    (tmp_path / ".env").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("KESTREL_DATA_KEY" in m for m in report.fail)


def test_doctor_blocks_on_empty_route_priority(tmp_path):
    _seed_ready(tmp_path)
    write_toml(tmp_path / "kestrel.toml", {"llm": {"route_priority": []}}, deep_merge=False)
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("route_priority" in m for m in report.fail)


def test_doctor_blocks_on_missing_api_key_env(tmp_path):
    _seed_ready(tmp_path)
    # Remove OPENAI_API_KEY but keep route
    p = tmp_path / ".env"
    text = p.read_text()
    p.write_text("\n".join(
        line for line in text.splitlines() if not line.startswith("OPENAI_API_KEY=")
    ) + "\n")
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("OPENAI_API_KEY" in m for m in report.fail)


def test_doctor_blocks_when_no_agents(tmp_path):
    _seed_ready(tmp_path)
    (tmp_path / "rookery.toml").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("agent" in m.lower() for m in report.fail)


def test_doctor_blocks_when_agent_db_missing(tmp_path):
    _seed_ready(tmp_path)
    (tmp_path / "agent_data" / "test" / "kestrel_prime.db").unlink()
    report = diagnose(tmp_path)
    assert not report.ready
    assert any("kestrel_prime.db" in m for m in report.fail)


def test_format_report_renders_lines(tmp_path):
    _seed_ready(tmp_path)
    report = diagnose(tmp_path)
    text = format_report(report)
    assert "✅" in text
    assert "Ready" in text


def test_format_report_says_not_ready_when_blocked(tmp_path):
    """Empty project should produce a not-ready message."""
    report = diagnose(tmp_path)
    text = format_report(report)
    assert "Not ready" in text
    assert "❌" in text
