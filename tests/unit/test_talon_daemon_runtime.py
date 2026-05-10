"""Daemon command contracts for backend-aware Talon runtime."""

from argparse import Namespace

from scripts.talon_daemon import DaemonConfig, RepoConfig, build_talon_command, load_config


def test_daemon_builds_codex_command(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    repo = RepoConfig(
        repo="org/repo",
        repo_dir="/tmp/repo",
        max_iterations=2,
        max_turns=40,
    )
    config = DaemonConfig(
        backend="codex",
        model="gpt-5.4-mini",
        auth_lane="oauth",
    )

    cmd = build_talon_command(repo, 42, config)

    assert cmd[:3] == ["uv", "run", "kestrel-talon"]
    assert cmd[cmd.index("--backend") + 1] == "codex"
    assert cmd[cmd.index("--codex-model") + 1] == "gpt-5.4-mini"
    assert "--model" not in cmd
    assert cmd[cmd.index("--max-iterations") + 1] == "2"
    assert cmd[cmd.index("--max-turns") + 1] == "40"


def test_daemon_loads_single_model_field_for_codex(tmp_path):
    path = tmp_path / "talon_daemon.toml"
    path.write_text(
        """
[daemon]
backend = "codex"
model = "gpt-5.4-mini"
auth_lane = "oauth"

[[repos]]
repo = "org/repo"
repo_dir = "/tmp/repo"
""",
        encoding="utf-8",
    )

    config = load_config(
        path,
        Namespace(
            backend=None,
            opencode_model=None,
            model=None,
            auth_lane=None,
            verbose=False,
            poll_interval=None,
        ),
    )

    assert config.backend == "codex"
    assert config.model == "gpt-5.4-mini"
    assert config.auth_lane == "oauth"

