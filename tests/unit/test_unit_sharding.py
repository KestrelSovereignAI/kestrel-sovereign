"""Real pytest shards execute the complete collected suite exactly once."""

import json
from pathlib import Path

import pytest
import yaml

from tests.shared.child_pytest import run_child_pytest
from tests.utils.ci_budget import CI_WORKFLOW
from tests.utils.unit_sharding import shard_for_node, verify_manifests


def _run_actual_shards(tmp_path: Path, *, selectors=(), distributed=False, reorder=False, lastfailed=False):
    reports = tmp_path / 'reports'
    witness = tmp_path / 'executed'
    results = []
    for index in (1, 2):
        root = tmp_path / f'run-{index}'
        root.mkdir()
        workers = ['-p', 'xdist.plugin', '-n', '2'] if distributed else []
        cached = []
        if lastfailed:
            cache = root / '.pytest_cache' / 'v' / 'cache'
            cache.mkdir(parents=True)
            (cache / 'lastfailed').write_text(json.dumps({
                f'test_owned_shards.py::test_owned_case[{case}]': True for case in ['alpha', 'beta']}))
            cached = ['-p', 'cacheprovider', '--lf', 'test_owned_shards.py']
        conftest = 'def pytest_collection_modifyitems(items):\n    items.reverse()\n' if reorder else ''
        result = run_child_pytest(root, conftest=conftest, test_file='test_owned_shards.py',
            source=f'''import pytest
from pathlib import Path

@pytest.mark.parametrize('case', ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])
def test_owned_case(case):
    with Path({str(witness)!r}).open('a') as output:
        output.write(case + '\\n')
''', args=['-q', *workers, '-p', 'tests.utils.unit_sharding', '--unit-shard', str(index),
                '--unit-shard-count', '2', '--unit-shard-manifests', str(reports), *selectors, *cached], timeout=15)
        results.append(result)
    return reports, witness, results


@pytest.fixture
def actual_shard_reports(tmp_path: Path, request):
    reports, witness, results = _run_actual_shards(tmp_path, distributed=getattr(request, 'param', False))
    for result in results:
        assert result.returncode == 0, result.stdout
    return reports, witness


@pytest.mark.parametrize('distributed', [False, True], ids=['one-process', 'native-xdist-workers'])
def test_actual_pytest_shards_execute_every_collected_test_once(tmp_path, distributed):
    reports, witness, results = _run_actual_shards(tmp_path, distributed=distributed)
    for child in results:
        assert child.returncode == 0, child.stdout
    result = verify_manifests(reports, 2)
    assert result == dict(complete=True, shard_count=2, worker_manifests=len(list(reports.glob('*.json'))),
        collected_tests=8, assigned_tests=8)
    assert sorted(witness.read_text().splitlines()) == sorted(
        ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])


@pytest.mark.parametrize('corrupt', ['missing-shard', 'omitted-test', 'wrong-shard',
    'changed-suite', 'duplicate-node', 'duplicate-worker', 'wrong-count',
    'invalid-schema', 'boolean-version', 'invalid-worker', 'invalid-nodes', 'empty-suite',
    'different-worker-order'])
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
    elif corrupt == 'different-worker-order':
        other = dict(data, worker='another-worker', selected_nodeids=list(reversed(data['selected_nodeids'])))
        (reports / 'other-worker.json').write_text(json.dumps(other))
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


@pytest.mark.parametrize('selectors', [
    ('--deselect=test_owned_shards.py::test_owned_case[alpha]',),
    ('-k', 'not alpha'),
], ids=['deselect', 'keyword'])
def test_later_pytest_selection_cannot_certify_unexecuted_tests(tmp_path, selectors):
    reports, witness, results = _run_actual_shards(tmp_path, selectors=selectors)
    assert results[0].returncode == 4, results[0].stdout
    assert 'Unit shard final collection changed after assignment' in results[0].stdout
    assert results[1].returncode == 0, results[1].stdout
    with pytest.raises(ValueError, match='Missing unit shard'):
        verify_manifests(reports, 2)
    assert sorted(witness.read_text().splitlines()) == sorted(
        ['beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta'])


@pytest.mark.parametrize('distributed', [False, True], ids=['one-process', 'native-xdist-workers'])
def test_later_hooks_may_reorder_the_complete_shard(tmp_path, distributed):
    reports, witness, results = _run_actual_shards(tmp_path, distributed=distributed, reorder=True)
    for child in results:
        assert child.returncode == 0, child.stdout
    assert verify_manifests(reports, 2)['assigned_tests'] == 8
    assert sorted(witness.read_text().splitlines()) == sorted(
        ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'])


def test_late_lastfailed_wrapper_cannot_certify_unexecuted_tests(tmp_path):
    reports, witness, results = _run_actual_shards(tmp_path, lastfailed=True)
    for child in results:
        assert child.returncode == 4, child.stdout
        assert 'Unit shard final collection changed after assignment' in child.stdout
    with pytest.raises(ValueError, match='No unit shard'):
        verify_manifests(reports, 2)
    assert not witness.exists(), 'A filtered shard started execution before final collection validation'
