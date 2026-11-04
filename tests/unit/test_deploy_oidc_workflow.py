"""Deployment must authenticate every cloud job through the shared OIDC action."""

from pathlib import Path
import os
import subprocess

import pytest
import yaml


def _documents():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/deploy.yml").read_text())
    action = yaml.safe_load((root / ".github/actions/gcp-deploy-auth/action.yml").read_text())
    return workflow, action


def test_all_cloud_deployment_jobs_use_the_shared_federated_identity():
    workflow, action = _documents()
    jobs = workflow["jobs"]
    for name in ["build", "deploy-dev", "deploy-prod"]:
        job = jobs[name]
        assert job["permissions"]["id-token"] == "write", name
        (authentication,) = [step for step in job["steps"]
                            if step.get("uses") == "./.github/actions/gcp-deploy-auth"]
        assert authentication["with"] == {
            "project_id": "${{ env.PROJECT_ID }}",
            "workload_identity_provider": "${{ vars.GCP_WIF_PROVIDER }}",
            "service_account": "${{ vars.GCP_DEPLOY_SERVICE_ACCOUNT }}",
        }, name
        assert not any(step.get("uses", "").startswith("google-github-actions/auth@")
                       for step in job["steps"]), name
    (authentication,) = [step for step in action["runs"]["steps"]
                        if step.get("uses", "").startswith("google-github-actions/auth@")]
    assert authentication["with"] == {
        "project_id": "${{ inputs.project_id }}",
        "workload_identity_provider": "${{ inputs.workload_identity_provider }}",
        "service_account": "${{ inputs.service_account }}",
    }
    assert "credentials_json" not in str(workflow) + str(action)
    assert "GCP_SA_KEY" not in str(workflow) + str(action)


@pytest.mark.parametrize("missing,value", [(None, None)] + [
    (field, value)
    for field in ("GCP_WIF_PROVIDER", "GCP_DEPLOY_SERVICE_ACCOUNT", "GCP_DEPLOY_PROJECT_ID")
    for value in (None, "")
])
def test_federation_configuration_fails_before_authentication_when_incomplete(missing, value):
    _, action = _documents()
    guard, authentication = action["runs"]["steps"]
    assert authentication["uses"].startswith("google-github-actions/auth@")
    assert guard["shell"] == "bash"
    env = {"PATH": os.environ["PATH"], "GCP_WIF_PROVIDER": "test-provider",
           "GCP_DEPLOY_SERVICE_ACCOUNT": "test-service-account",
           "GCP_DEPLOY_PROJECT_ID": "test-project"}
    if missing:
        if value is None:
            del env[missing]
        else:
            env[missing] = value
    result = subprocess.run(["bash", "--noprofile", "--norc", "-e", "-c", guard["run"]],
                            env=env, capture_output=True, text=True)
    if missing:
        assert result.returncode != 0
        assert missing in result.stderr
    else:
        assert result.returncode == 0, result.stderr
