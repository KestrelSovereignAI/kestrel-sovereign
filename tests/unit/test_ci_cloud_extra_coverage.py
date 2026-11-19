"""The unit gates must retain the cloud SDK used by deployment regressions."""

import ast
import shlex
from pathlib import Path

import yaml


def test_unit_install_and_pytest_keep_the_same_frozen_cloud_extra():
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / '.github/workflows/ci.yml').read_text())
    steps = workflow['jobs']['unit-tests']['steps']
    for name, prefix in [('Install dependencies', ['uv', 'sync']),
                         ('Validate cloud provider test dependencies', ['uv', 'run']),
                         ('Run unit tests', ['uv', 'run'])]:
        command = shlex.split(next(step['run'] for step in steps if step.get('name') == name))
        assert command[:2] == prefix
        assert '--frozen' in command
        assert '--extra' in command and command[command.index('--extra') + 1] == 'cloud'
        if name == 'Run unit tests':
            assert command.index('--extra') < command.index('pytest')
        if name == 'Validate cloud provider test dependencies':
            assert command.index('--extra') < command.index('python')
            code = ast.parse(command[command.index('-c') + 1])
            imports = {alias.name for node in ast.walk(code) if isinstance(node, ast.Import)
                       for alias in node.names}
            assert {'google.cloud.run_v2', 'azure.mgmt.appcontainers'} <= imports
