"""Accepted operator venvs stay immutable after settings/process loss."""

from dataclasses import replace
import hashlib
import os
import shutil
import subprocess

import pytest

from kestrel_sovereign.features import isolated_runtime as native
from tests.unit.test_isolated_feature_runtime import (
    _hosted_postgres_agent, _isolated_runtime, _write_console_script_fixture_wheel,
    _write_real_venv_module,
)


def _inventory(root):
    return {
        str(path.relative_to(root)): (
            ("link", os.readlink(path)) if path.is_symlink()
            else ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        )
        for path in root.rglob("*") if path.is_file() or path.is_symlink()
    }


@pytest.mark.parametrize("restart", [False, True])
def test_historical_operator_default_venv_never_becomes_mutable(tmp_path, monkeypatch, restart):
    runtime = replace(
        _isolated_runtime(), distribution="kestrel-console-migration-fixture",
        service="kestrel-whatsapp-service",
    )
    root = tmp_path / "runtime"
    agent = _hosted_postgres_agent(root, "immutable-original")
    feature = native.ProxyFeature(agent, runtime)
    feature._prepare_runtime_workspace()
    venv = feature._default_venv_path()
    python = _write_real_venv_module(venv, "review68_native", "value = 'offline native fixture'\n")
    wheel = _write_console_script_fixture_wheel(tmp_path)
    subprocess.run(
        [shutil.which("uv"), "pip", "install", "--offline", "--no-deps", "--python", str(python), str(wheel)],
        check=True, capture_output=True, timeout=15,
    )
    setting = "KESTREL_FEATURE_WHATSAPPFEATURE_VENV"
    monkeypatch.setenv(setting, str(venv))
    feature = native.ProxyFeature(agent, runtime)
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature.ensure_venv() is False
    assert not feature._provision_manifest_path().exists()
    before = _inventory(venv)
    monkeypatch.delenv(setting)
    if restart:
        agent._isolated_runtime_cleanup_owners.clear()
        agent = _hosted_postgres_agent(root, "immutable-original")
        feature = native.ProxyFeature(agent, runtime)
        assert not feature._validated_hosted_immutable_venv_custody_paths
    feature._venv_path, feature._bin_path = feature.resolve_runtime_paths()
    assert feature._venv_path == venv
    attempted = []

    def refuse_mutation(command):
        attempted.append(command)
        raise AssertionError("historical operator venv reached actual provisioning boundary")

    monkeypatch.setattr(feature, "_run_provisioning_command", refuse_mutation)
    with pytest.raises(native.IsolatedRuntimePreparationError, match="immutable"):
        feature.ensure_venv()
    assert attempted == []
    assert _inventory(venv) == before
    assert not feature._provision_manifest_path().exists()
