"""``kestrel runpod`` CLI tests — sub-PR 4 of epic #1050 (bash-to-Python
port of ``scripts/runpod/deploy_lora_trainer.sh``).

Covers:
- argparse wiring under the local subparser and the real ``kestrel``
  parser
- Missing ``RUNPOD_API_KEY`` → exit code 1 with a clear hint
- Unknown profile in ``runpod_config.toml`` → exit code 1 listing the
  available profiles
- ``deploy lora-trainer`` invokes :meth:`RunPodManager.start_session`
  with the right kwargs
- ``status`` calls :meth:`get_status` and surfaces both active and
  resumable pods
- ``stop`` calls :meth:`stop_session`
- ``kill`` calls :meth:`get_status` then ``provider.terminate_pod``
- Top-level dispatcher prints usage when no subverb is supplied
"""

from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kestrel_sovereign import cli_runpod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_profile() -> SimpleNamespace:
    return SimpleNamespace(
        gpu_type_id="NVIDIA RTX 3090",
        image_name="gcr.io/test/lora:latest",
        template_id="tmpl-1",
        network_volume_id="vol-1",
        inference_port=8080,
    )


def _make_manager(*, profile_name: str = "training", with_profile: bool = True,
                  start_session_result=None,
                  start_session_exc=None,
                  status_active=True,
                  pod_id="pod-abc",
                  base_url="https://lora.runpod.io",
                  stop_result="stopped",
                  stopped_pod=None) -> MagicMock:
    """Mock RunPodManager that mirrors the real surface the CLI uses."""
    mgr = MagicMock()
    mgr.profiles = ({profile_name: _make_profile()} if with_profile else {})

    async def get_status():
        if status_active:
            return {
                "active": True,
                "pod_id": pod_id,
                "status": "RUNNING",
                "base_url": base_url,
                "expires_at": "2026-12-31T00:00:00Z",
            }
        return {"active": False, "pod_id": pod_id if status_active else None}

    async def start_session(**kw):
        if start_session_exc is not None:
            raise start_session_exc
        return (start_session_result or {
            "pod_id": pod_id,
            "status": "RUNNING",
            "base_url": base_url,
            "ttl_seconds": 3600,
        })

    async def stop_session():
        return stop_result

    async def find_stopped_pod(*a, **kw):
        return stopped_pod

    mgr.get_status = get_status
    mgr.start_session = start_session
    mgr.stop_session = stop_session
    mgr.find_stopped_pod = find_stopped_pod

    # provider.terminate_pod is sync and called via asyncio.to_thread.
    mgr.provider = MagicMock()
    mgr.provider.terminate_pod = MagicMock(return_value=None)
    return mgr


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kestrel")
    sub = p.add_subparsers(dest="command")
    cli_runpod.add_runpod_subcommand(sub)
    return p


def test_argparse_deploy_defaults():
    parser = _build_parser()
    args = parser.parse_args(["runpod", "deploy", "lora-trainer"])
    assert args.command == "runpod"
    assert args.runpod_command == "deploy"
    assert args.target == "lora-trainer"
    assert args.profile == "training"
    assert args.test is False


def test_argparse_deploy_overrides():
    parser = _build_parser()
    args = parser.parse_args([
        "runpod", "deploy", "lora-trainer",
        "--profile", "training-4090", "--test",
    ])
    assert args.profile == "training-4090"
    assert args.test is True


def test_argparse_deploy_unknown_target_rejected():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["runpod", "deploy", "ollama"])


def test_argparse_status_stop_kill():
    parser = _build_parser()
    for verb in ("status", "stop", "kill"):
        args = parser.parse_args(["runpod", verb])
        assert args.runpod_command == verb
        assert args.profile == "training"


def test_kestrel_cli_registers_runpod():
    from kestrel_sovereign.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["runpod", "status"])
    assert args.command == "runpod"
    assert args.runpod_command == "status"


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def test_cmd_runpod_no_subverb_prints_usage(capsys):
    rc = cli_runpod.cmd_runpod(_Args(runpod_command=None))
    assert rc == 1
    err = capsys.readouterr().err
    assert "Usage" in err
    assert "deploy" in err and "status" in err and "stop" in err


# ---------------------------------------------------------------------------
# Missing RUNPOD_API_KEY
# ---------------------------------------------------------------------------

def test_deploy_without_api_key_errors_with_hint(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    args = _Args(
        runpod_command="deploy", target="lora-trainer",
        profile="training", test=False,
    )
    rc = cli_runpod.cmd_runpod(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err


def test_status_without_api_key_errors(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = cli_runpod.cmd_runpod(
        _Args(runpod_command="status", profile="training")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err


def test_stop_without_api_key_errors(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = cli_runpod.cmd_runpod(
        _Args(runpod_command="stop", profile="training")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err


def test_kill_without_api_key_errors(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = cli_runpod.cmd_runpod(
        _Args(runpod_command="kill", profile="training")
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "RUNPOD_API_KEY" in err


# ---------------------------------------------------------------------------
# Unknown profile
# ---------------------------------------------------------------------------

def test_deploy_unknown_profile_errors_with_available_list(
    monkeypatch, capsys,
):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(profile_name="training")
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="deploy", target="lora-trainer",
            profile="bogus-profile", test=False,
        ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "bogus-profile" in err
    assert "training" in err  # listed as available


def test_deploy_no_profiles_at_all_hints_at_config(
    monkeypatch, capsys,
):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(with_profile=False)
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="deploy", target="lora-trainer",
            profile="training", test=False,
        ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "runpod_config.toml" in err


# ---------------------------------------------------------------------------
# deploy lora-trainer
# ---------------------------------------------------------------------------

def test_deploy_unknown_target_errors(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    args = _Args(
        runpod_command="deploy", target="elephants",
        profile="training", test=False,
    )
    rc = cli_runpod.cmd_runpod(args)
    assert rc == 1
    err = capsys.readouterr().err
    assert "elephants" in err


def test_deploy_invokes_start_session(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager()
    captured: dict = {}
    orig = mgr.start_session

    async def spy(**kw):
        captured.update(kw)
        return await orig(**kw)

    mgr.start_session = spy

    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="deploy", target="lora-trainer",
            profile="training", test=False,
        ))
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pod created successfully" in out
    assert "pod-abc" in out
    assert captured.get("task_profile") == "training"
    assert captured.get("model_name") == "FLUX.1-dev"
    assert captured.get("ttl_seconds") == 3600


def test_deploy_failure_surfaces_error_and_returns_1(
    monkeypatch, capsys,
):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(start_session_exc=RuntimeError("boom"))
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="deploy", target="lora-trainer",
            profile="training", test=False,
        ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "boom" in err


def test_deploy_no_gpu_avail_emits_recovery_hints(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(start_session_exc=RuntimeError(
        "There are no longer any instances available with this configuration."
    ))
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="deploy", target="lora-trainer",
            profile="training", test=False,
        ))
    assert rc == 1
    err = capsys.readouterr().err
    assert "GPU Availability" in err
    assert "training-4090" in err  # cross-profile suggestion


# ---------------------------------------------------------------------------
# status / stop / kill
# ---------------------------------------------------------------------------

def test_status_active_pod(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(status_active=True)
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="status", profile="training",
        ))
    assert rc == 0
    out = capsys.readouterr().out
    assert "ACTIVE" in out
    assert "pod-abc" in out


def test_status_no_active_with_resumable(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(
        status_active=False, stopped_pod={"id": "pod-frozen"},
    )
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="status", profile="training",
        ))
    assert rc == 0
    out = capsys.readouterr().out
    assert "No active pod" in out
    assert "pod-frozen" in out


def test_stop_invokes_stop_session(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(stop_result="STOPPED-OK")
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="stop", profile="training",
        ))
    assert rc == 0
    out = capsys.readouterr().out
    assert "STOPPED-OK" in out


def test_kill_terminates_pod_via_provider(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(status_active=True)
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="kill", profile="training",
        ))
    assert rc == 0
    mgr.provider.terminate_pod.assert_called_once_with("pod-abc")
    out = capsys.readouterr().out
    assert "Terminated" in out


def test_kill_no_active_pod_returns_zero(monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    mgr = _make_manager(status_active=False)
    # Simulate "no active pod" state — get_status returns active=False
    # AND no pod_id.
    async def get_status():
        return {"active": False, "pod_id": None}
    mgr.get_status = get_status
    with patch.object(cli_runpod, "_load_manager", return_value=mgr):
        rc = cli_runpod.cmd_runpod(_Args(
            runpod_command="kill", profile="training",
        ))
    assert rc == 0
    mgr.provider.terminate_pod.assert_not_called()
    out = capsys.readouterr().out
    assert "No active pod" in out


# ---------------------------------------------------------------------------
# Lazy import — kestrel-cloud-runpod absent
# ---------------------------------------------------------------------------

def test_load_manager_missing_package_exits_clean(monkeypatch, capsys):
    """When kestrel-cloud-runpod isn't installed, the CLI should exit
    1 with a clear hint, not raise an uncaught ImportError."""
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    # Force the lazy import path to fail.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "kestrel_cloud_runpod.manager":
            raise ImportError("no such module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit) as exc:
        cli_runpod._load_manager()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "kestrel-cloud-runpod" in err
