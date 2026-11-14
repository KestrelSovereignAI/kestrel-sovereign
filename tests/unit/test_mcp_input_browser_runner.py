"""Actual wheel custody failures must stop native browser acceptance."""

import hashlib
import importlib.util
from pathlib import Path
import tomllib
import warnings
import zipfile

import pytest
from packaging.requirements import Requirement
from packaging.version import Version
import yaml


def runner():
    path = Path(__file__).parents[1] / 'native_mcp/run_input_browser.py'
    spec = importlib.util.spec_from_file_location('input_browser_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_input_browser_support_matches_the_public_host_feature_minimum():
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    requirement, = [Requirement(raw) for raw in project['optional-dependencies']['mcp-integration']
                    if Requirement(raw).name == 'kestrel-feature-mcp']
    minimum = max(Version(spec.version) for spec in requirement.specifier if spec.operator == '>=')
    workflow = yaml.safe_load((root / '.github/workflows/ci.yml').read_text())
    support, = [step['with'] for step in workflow['jobs']['mcp-input-browser']['steps']
                if step.get('with', {}).get('repository') == 'KestrelSovereignAI/kestrel-feature-mcp']
    assert support['ref'] == f'v{minimum}', (
        'Native input acceptance must use the same public MCP release as the declared host minimum')


def test_input_browser_runtime_install_resolves_candidate_dependencies():
    python = Path('/original runtime/bin/python')
    wheel = Path('/owned/candidate core.whl')
    command = runner().runtime_install_command(python, '0.17.0', wheel)
    assert command[:3] == ['uv', 'pip', 'install']
    assert '--no-deps' not in command, (
        'Replacing Core without its declared dependencies leaves the base SDK installed')
    assert command[command.index('--python') + 1] == str(python)
    assert command[command.index('--index-url') + 1] == 'https://pypi.org/simple'
    assert command[-2:] == ['kestrel-feature-mcp==0.17.0', str(wheel)]


def test_native_browser_refuses_unknown_suite_before_building():
    with pytest.raises(ValueError, match='Unknown native MCP browser suite'):
        runner().build_input_image(None, None, None, None, None, suite='untrusted-suite')


def test_native_consent_ci_runs_and_retires_the_same_owned_namespace():
    root = Path(__file__).parents[2]
    workflow = yaml.safe_load((root / '.github/workflows/ci.yml').read_text())
    steps = workflow['jobs']['mcp-input-browser']['steps']
    consent = [step for step in steps if '--suite consent' in step.get('run', '')]
    assert len(consent) == 2
    run, cleanup = consent
    assert '--cleanup' not in run['run'] and not run.get('continue-on-error', False)
    assert cleanup.get('if') == 'always()' and '--cleanup' in cleanup['run']
    destination = '--output "${{ runner.temp }}/mcp-consent-browser"'
    assert all(destination in step['run'] for step in consent)
    artifact, = [step['with'] for step in steps if step.get('uses', '').startswith('actions/upload-artifact@')]
    assert '${{ runner.temp }}/mcp-consent-browser/' in artifact['path']


@pytest.mark.parametrize('fault', ['altered', 'missing', 'duplicate', 'empty'])
def test_input_browser_rejects_wheel_without_exact_unique_source_bytes(tmp_path, fault):
    source = tmp_path / 'source'
    source.mkdir()
    package = source / 'kestrel_sovereign'
    package.mkdir()
    (package / '__init__.py').write_bytes(b'original source\n')
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        if fault != 'empty':
            name = 'kestrel_sovereign/missing.py' if fault == 'missing' else 'kestrel_sovereign/__init__.py'
            archive.writestr(name, b'altered bytes\n' if fault == 'altered' else b'original source\n')
            if fault == 'duplicate':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    archive.writestr(name, b'original source\n')
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner().wheel_inventory(wheel, source)


def test_input_browser_binds_the_installed_runtime_to_real_wheel_bytes(tmp_path):
    package = tmp_path / 'kestrel_sovereign'
    package.mkdir()
    body = b'original runtime\n'
    (package / '__init__.py').write_bytes(body)
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('kestrel_sovereign/', b'')
        archive.writestr('kestrel_sovereign/__init__.py', body)
    assert runner().wheel_inventory(wheel, tmp_path) == {
        'kestrel_sovereign/__init__.py': hashlib.sha256(body).hexdigest(),
    }
