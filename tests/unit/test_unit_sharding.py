"""Real pytest shards execute the complete collected suite exactly once."""

import json
from pathlib import Path

import pytest
import yaml

from tests.shared.child_pytest import run_child_pytest
from tests.utils.ci_budget import CI_WORKFLOW
from tests.utils.unit_sharding import shard_for_node, verify_manifests


@pytest.fixture
def actual_shard_reports(tmp_path: Path, request):
    reports = tmp_path / 'reports'
    witness = tmp_path / 'executed'
    for index in (1, 2):
        root = tmp_path / f'run-{index}'
        root.mkdir()
        distributed = ['-p', 'xdist.plugin', '-n', '2'] if getattr(request, 'param', False) else []
        result = run_child_pytest(root, conftest='', test_file='test_owned_shards.py',
            source=f'''import pytest
from pathlib import Path

@pytest.mark.parametrize('case', ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])
def test_owned_case(case):
    with Path({str(witness)!r}).open('a') as output:
        output.write(case + '\\n')
''', args=['-q', *distributed, '-p', 'tests.utils.unit_sharding', '--unit-shard', str(index),
                '--unit-shard-count', '2', '--unit-shard-manifests', str(reports)], timeout=15)
        assert result.returncode == 0, result.stdout
    return reports, witness


@pytest.mark.parametrize('actual_shard_reports', [False, True], indirect=True,
    ids=['one-process', 'native-xdist-workers'])
def test_actual_pytest_shards_execute_every_collected_test_once(actual_shard_reports):
    reports, witness = actual_shard_reports
    result = verify_manifests(reports, 2)
    assert result == dict(complete=True, shard_count=2, worker_manifests=len(list(reports.glob('*.json'))),
        collected_tests=8, assigned_tests=8)
    assert sorted(witness.read_text().splitlines()) == sorted(
        ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])


@pytest.mark.parametrize('corrupt', ['missing-shard', 'omitted-test', 'wrong-shard',
    'changed-suite', 'duplicate-node', 'duplicate-worker', 'wrong-count',
    'invalid-schema', 'boolean-version', 'invalid-worker', 'invalid-nodes', 'empty-suite'])
def test_manifest_verification_refuses_incomplete_or_ambiguous_coverage(actual_shard_reports, corrupt):
    reports, _ = actual_shard_reports
    paths = sorted(reports.glob('*.json'))
    first, second = paths
    data = json.loads(first.read_text())
    if corrupt == 'missing-shard':
        second.unlink()
    elif corrupt == 'omitted-test':
        data['selected_nodeids'].pop()
    elif corrupt == 'wrong-shard':
        data['selected_nodeids'] = json.loads(second.read_text())['selected_nodeids']
    elif corrupt == 'changed-suite':
        data['all_nodeids'].pop()
    elif corrupt == 'duplicate-node':
        data['selected_nodeids'].append(data['selected_nodeids'][0])
    elif corrupt == 'duplicate-worker':
        (reports / 'duplicate.json').write_text(first.read_text())
    elif corrupt == 'invalid-schema':
        data['unexpected'] = True
    elif corrupt == 'boolean-version':
        data['format_version'] = True
    elif corrupt == 'invalid-worker':
        data['worker'] = []
    elif corrupt == 'invalid-nodes':
        data['selected_nodeids'] = 'not a collection'
    elif corrupt == 'empty-suite':
        data['all_nodeids'] = data['selected_nodeids'] = []
    else:
        data['shard_count'] = 3
    first.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_manifests(reports, 2)


def test_shard_verification_refuses_absent_reports(tmp_path):
    with pytest.raises(ValueError, match='No unit shard'):
        verify_manifests(tmp_path, 2)


@pytest.mark.parametrize('count', [0, -1, True])
def test_invalid_shard_counts_refuse_before_selection(count):
    with pytest.raises(ValueError):
        shard_for_node('tests/unit/test_owned.py::test_case', count)


def test_pytest_refuses_partial_shard_configuration(tmp_path):
    result = run_child_pytest(tmp_path, conftest='', test_file='test_owned.py',
        source='def test_owned(): pass\n',
        args=['-q', '-p', 'tests.utils.unit_sharding', '--unit-shard', '1'], timeout=15)
    assert result.returncode != 0
    assert 'Supply a valid unit shard' in result.stdout


def test_ci_runs_and_verifies_all_unit_shards_without_widening_deadlines():
    jobs = yaml.safe_load(CI_WORKFLOW.read_text())['jobs']
    unit = jobs['unit-tests']
    assert unit['strategy'] == {'fail-fast': False, 'matrix': {'shard': [1, 2]}}
    run = next(step for step in unit['steps'] if step.get('name') == 'Run unit tests')
    assert '-p tests.utils.unit_sharding' in run['run']
    assert '--unit-shard=${{ matrix.shard }}' in run['run']
    assert '--unit-shard-count=2' in run['run']
    assert '--unit-shard-manifests=unit-shard-manifests' in run['run']
    assert '--session-timeout=2100' in run['run'] and '--timeout=60' in run['run']
    assert run['timeout-minutes'] == 45
    upload = next(step for step in unit['steps'] if step.get('name') == 'Upload unit shard collection manifests')
    assert upload['if'] == 'always()' and upload['with']['if-no-files-found'] == 'error'
    assert upload['with']['name'] == 'unit-shard-${{ matrix.shard }}'
    coverage = jobs['unit-shard-coverage']
    assert coverage['needs'] == 'unit-tests'
    assert 'unit_sharding.py --verify unit-shard-manifests --count 2' in str(coverage['steps'])
    assert 'unit-shard-coverage' in jobs['a2a-go-interop']['needs']
