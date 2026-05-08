"""Tests for the shared GCP credential discovery in
``kestrel_sovereign.features.deploy._gcp_auth``.

Sub-PR 1.2 of epic #1050 extracted this logic from
``CloudRunProvider._setup_auth`` so ``kestrel deploy secrets sync`` can
authenticate the same way as ``kestrel deploy <profile>``.
"""

from __future__ import annotations

import os

import pytest

from kestrel_sovereign.features.deploy._gcp_auth import setup_gcp_auth


def test_setup_gcp_auth_clears_stale_credentials_env_var(
    tmp_path, monkeypatch, caplog
):
    """Codex review on PR #1057: if ``GOOGLE_APPLICATION_CREDENTIALS``
    points at a missing file, Google clients still check it first and
    fail before ADC can take effect. ``setup_gcp_auth`` must clear the
    stale env var so the fallback chain can proceed.
    """
    missing_path = tmp_path / "definitely-not-here.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing_path))
    monkeypatch.delenv("GCP_SERVICE_ACCOUNT_KEY", raising=False)

    # No project-local creds either — force the ADC fallback path.
    # The function uses the real project root, but the test runs in a
    # fresh checkout where credentials/kestrel-agent-admin.json doesn't
    # exist. If it ever does, this test would need a guard; for now we
    # simply assert the env-var clearing behavior, which is independent.

    with caplog.at_level("WARNING", logger="kestrel_sovereign.features.deploy._gcp_auth"):
        setup_gcp_auth()

    # Stale env var must be gone so SDK clients can retry via ADC.
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ
    # And we must have logged a warning so operators know what happened.
    assert any(
        "missing file" in rec.message
        for rec in caplog.records
    )


def test_setup_gcp_auth_keeps_valid_credentials_env_var(tmp_path, monkeypatch):
    """If ``GOOGLE_APPLICATION_CREDENTIALS`` points at a real file we
    leave it alone — that's the operator's explicit choice."""
    creds_file = tmp_path / "creds.json"
    creds_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds_file))
    monkeypatch.delenv("GCP_SERVICE_ACCOUNT_KEY", raising=False)

    result = setup_gcp_auth()

    assert result is None  # no temp file written
    assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(creds_file)
