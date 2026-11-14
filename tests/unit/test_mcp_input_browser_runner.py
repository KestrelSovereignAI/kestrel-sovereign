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


@pytest.fixture
def source_tree(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'pyproject.toml').write_text('[tool.hatch.build]\npackages = ["kestrel_sovereign"]\n')
    return source


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


@pytest.mark.parametrize('fault', ['altered', 'missing', 'duplicate', 'empty'])
def test_input_browser_rejects_wheel_without_exact_unique_source_bytes(tmp_path, source_tree, fault):
    source = source_tree
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


def test_input_browser_binds_the_installed_runtime_to_real_wheel_bytes(tmp_path, source_tree):
    package = source_tree / 'kestrel_sovereign'
    package.mkdir()
    body = b'original runtime\n'
    (package / '__init__.py').write_bytes(body)
    wheel = tmp_path / 'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('kestrel_sovereign/', b'')
        archive.writestr('kestrel_sovereign/__init__.py', body)
    assert runner().wheel_inventory(wheel, source_tree) == {
        'kestrel_sovereign/__init__.py': hashlib.sha256(body).hexdigest(),
    }


def test_input_browser_binds_actual_declared_force_includes(tmp_path):
    source = Path(__file__).parents[2]
    build = tomllib.loads((source / 'pyproject.toml').read_text())['tool']['hatch']['build']
    included = build['targets']['wheel']['force-include']
    assert included, 'Exercise the actual declared private program mappings'
    wheel = tmp_path / 'mapped-programs.whl'
    expected = {}
    with zipfile.ZipFile(wheel, 'w') as archive:
        for original, target in included.items():
            body = (source / original).read_bytes()
            assert not (source / target).exists(), 'Mapped source and wheel destination must differ'
            archive.writestr(target, body)
            expected[target] = hashlib.sha256(body).hexdigest()
    assert runner().wheel_inventory(wheel, source) == expected


@pytest.mark.parametrize('fault', ['altered-source', 'missing-source', 'altered-wheel'])
def test_input_browser_checks_canonical_mapped_bytes(tmp_path, source_tree, fault):
    original = source_tree / 'tools/canonical/program.py'
    original.parent.mkdir(parents=True)
    original.write_bytes(b'original mapped program\n')
    target = 'kestrel_sovereign/private/installed.py'
    (source_tree / 'pyproject.toml').write_text(
        '[tool.hatch.build]\npackages = ["kestrel_sovereign"]\n'
        '[tool.hatch.build.targets.wheel.force-include]\n'
        '"tools/canonical/program.py" = "kestrel_sovereign/private/installed.py"\n')
    # An unmapped same-name file must never substitute for the declared input.
    decoy = source_tree / target
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(original.read_bytes())
    wheel = tmp_path / 'mapped.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr(target, b'altered mapped program\n' if fault == 'altered-wheel' else original.read_bytes())
    if fault == 'altered-source':
        original.write_bytes(b'altered mapped source\n')
    elif fault == 'missing-source':
        original.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        runner().wheel_inventory(wheel, source_tree)
