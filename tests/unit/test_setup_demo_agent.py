"""``scripts/setup_demo_agent.py`` options used by ``kestrel demo smoke`` (#2682).

The default invocation (``kestrel demo run``) is unchanged; the smoke adds a
fresh ``--data-dir``, an inception ``--manifest``, and ``--local-llm-only``.
"""

from __future__ import annotations

import json
import tomllib

import pytest

from scripts import setup_demo_agent
from scripts.setup_demo_agent import build_demo_kestrel_toml


def test_local_llm_only_config_has_no_paid_route():
    llm = tomllib.loads(build_demo_kestrel_toml(local_llm_only=True))["llm"]

    assert llm["route_priority"] == ["ollama:local"]
    assert set(llm["vendors"]) == {"ollama"}
    route = llm["vendors"]["ollama"]["routes"]["local"]
    assert route["adapter"] == "OllamaAdapter"
    assert route["host"] == "http://localhost:11434"
    assert "api_key_env" not in route


def test_default_config_is_unchanged_by_the_local_option():
    assert build_demo_kestrel_toml() == build_demo_kestrel_toml(local_llm_only=False)
    assert "AnthropicAdapter" in build_demo_kestrel_toml()


def test_parse_args_defaults_to_the_demo_sandbox():
    args = setup_demo_agent._parse_args([])
    assert args.data_dir == setup_demo_agent.DEMO_DIR
    assert args.manifest is None
    assert args.local_llm_only is False


def test_prepare_data_dir_wipes_only_the_default_sandbox(tmp_path, monkeypatch):
    sandbox = tmp_path / "agent_data" / "demo"
    sandbox.mkdir(parents=True)
    (sandbox / "kestrel_prime.db").write_text("previous demo")
    monkeypatch.setattr(setup_demo_agent, "DEMO_DIR", str(sandbox))

    assert setup_demo_agent.prepare_data_dir(str(sandbox)) == str(sandbox)
    assert list(sandbox.iterdir()) == []


def test_prepare_data_dir_never_deletes_a_supplied_directory(tmp_path):
    populated = tmp_path / "someone-elses-agent"
    populated.mkdir()
    (populated / "kestrel_prime.db").write_text("live")

    with pytest.raises(SystemExit, match="not empty"):
        setup_demo_agent.prepare_data_dir(str(populated))
    assert (populated / "kestrel_prime.db").read_text() == "live"


def test_prepare_data_dir_creates_a_fresh_supplied_directory(tmp_path):
    fresh = tmp_path / "home" / "agent_data" / "console-smoke"
    assert setup_demo_agent.prepare_data_dir(str(fresh)) == str(fresh)
    assert fresh.is_dir()


def test_write_manifest_records_the_inception_facts(tmp_path):
    data_dir = str(tmp_path / "data")
    path = tmp_path / "inception.json"
    setup_demo_agent.write_manifest(
        str(path),
        agent_did="did:web:localhost:kestrel-demo-agent-abc123",
        db_path=str(tmp_path / "data" / "kestrel_prime.db"),
        data_dir=data_dir,
    )
    assert json.loads(path.read_text()) == {
        "agent_name": setup_demo_agent.DEMO_AGENT_NAME,
        "agent_did": "did:web:localhost:kestrel-demo-agent-abc123",
        "db_path": str(tmp_path / "data" / "kestrel_prime.db"),
        "data_dir": data_dir,
    }
