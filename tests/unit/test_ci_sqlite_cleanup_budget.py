"""Integration runners reserve bounded native SQLite checkpoint retirement."""
import pytest
import yaml

from tests.utils.ci_budget import CI_WORKFLOW


@pytest.mark.parametrize('job', ['integration-tests', 'native-postgres-tests'])
def test_ci_database_tiers_reserve_bounded_checkpoint_retirement(job):
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    key = 'KESTREL_AIOSQLITE_WORKER_SHUTDOWN_TIMEOUT_S'
    budget = float(workflow['jobs'][job].get('env', {}).get(key, 0))
    assert 3 <= budget <= 10, 'shared-runner fsync/worker retirement needs an explicit finite reservation'
    # Do not change production defaults or the unit suite's deadline checks.
    assert key not in workflow.get('env', {})
    assert key not in workflow['jobs']['unit-tests'].get('env', {})
