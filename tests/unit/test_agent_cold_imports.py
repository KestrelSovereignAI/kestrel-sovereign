"""Agent leaf imports and pure terminal classification must work from cold start."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("operation", [
    "import importlib; importlib.import_module('kestrel_sovereign.tools.effect_failures')",
    "import importlib; importlib.import_module('kestrel_sovereign.tools.effect_journal')",
    "import importlib; importlib.import_module('kestrel_sovereign.tools.effect_invocation')",
    """
from kestrel_sovereign.execution_custody import (
    execution_terminal_error, is_tool_execution_terminal_error,
)
original = ValueError('declared ordinary effect failure')
assert not is_tool_execution_terminal_error(original)
assert execution_terminal_error(original, retain_originals=True,
    isolate_control_originals=True) is original
assert original.__cause__ is None and original.__context__ is None
""",
    """
from kestrel_sovereign.execution_custody import (
    execution_terminal_error, is_execution_control_error,
    is_tool_execution_terminal_error,
)
from kestrel_sovereign.tools.effect_journal import ToolEffectOutcomeUnconfirmed
original = ToolEffectOutcomeUnconfirmed('declared original uncertain delivery')
assert is_tool_execution_terminal_error(original)
assert not is_execution_control_error(original)
assert execution_terminal_error(original, retain_originals=True,
    isolate_control_originals=True) is original
""",
], ids=['effect-failures', 'effect-journal', 'effect-invocation', 'ordinary-terminal',
        'uncertain-tool-terminal'])
def test_cold_effect_entrypoints_do_not_require_agent_startup(tmp_path, operation):
    _run_cold(tmp_path, operation + "\nassert 'kestrel_sovereign.agent.streaming' not in sys.modules\n")


def test_agent_public_exports_remain_canonical_and_discoverable(tmp_path):
    _run_cold(tmp_path, """
import importlib
agent = importlib.import_module('kestrel_sovereign.agent')
assert set(agent.__all__) == {
    'ContextManager', 'ContextResult', 'ContextBuilder', 'ConstitutionMixin',
    'StreamingMixin', 'BackupMixin', 'OrchestratorEngineMixin', 'ContextStats',
    'ToolRegistryMixin', 'ModelPreferenceMixin', 'EventManagerMixin',
    'EventDeliveryReceipt', 'RequestLifecycleMixin', 'TokenCounter',
    'get_token_counter', 'TokenBudget', 'AdaptiveTokenBudget', 'create_budget',
}
assert set(agent.__all__) <= set(dir(agent))
for name in agent.__all__:
    value = getattr(agent, name)
    canonical = getattr(importlib.import_module(value.__module__), value.__name__)
    assert value is canonical and getattr(agent, name) is value, name
star_exports = {}
exec('from kestrel_sovereign.agent import *', star_exports)
assert {name for name in star_exports if not name.startswith('__')} == set(agent.__all__)
for name in agent.__all__:
    assert star_exports[name] is getattr(agent, name), name
try:
    agent.unknown_public_agent_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown agent export was accepted')
""")


@pytest.mark.parametrize('exports', [
    ('ContextBuilder', 'EventManagerMixin'),
    ('ContextManager', 'ContextResult', 'ContextBuilder', 'ConstitutionMixin',
     'StreamingMixin', 'BackupMixin', 'OrchestratorEngineMixin', 'ContextStats',
     'ToolRegistryMixin', 'ModelPreferenceMixin', 'EventManagerMixin',
     'EventDeliveryReceipt', 'RequestLifecycleMixin', 'TokenCounter',
     'get_token_counter', 'TokenBudget', 'AdaptiveTokenBudget', 'create_budget'),
], ids=['shared-dependency-pair', 'all-public-exports'])
def test_cold_public_exports_share_one_initialization(tmp_path, exports):
    _run_cold(tmp_path, f"""
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import kestrel_sovereign.agent as agent
exports = {exports!r}
start = Barrier(len(exports))
def resolve(name):
    start.wait(timeout=10)
    return getattr(agent, name)
with ThreadPoolExecutor(max_workers=len(exports)) as pool:
    resolved = tuple(pool.map(resolve, exports))
for name, value in zip(exports, resolved):
    assert value is getattr(agent, name), name
    assert value is getattr(importlib.import_module(value.__module__), value.__name__)
""")


def test_cold_public_and_named_imports_share_initialization(tmp_path):
    _run_cold(tmp_path, """
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import kestrel_sovereign.agent as agent
start = Barrier(2)
def public():
    start.wait(timeout=10)
    return agent.ContextBuilder
def named():
    start.wait(timeout=10)
    return importlib.import_module('kestrel_sovereign.agent.event_manager').EventManagerMixin
with ThreadPoolExecutor(max_workers=2) as pool:
    public_result = pool.submit(public)
    named_result = pool.submit(named)
    assert public_result.result(timeout=20) is agent.ContextBuilder
    assert named_result.result(timeout=20) is agent.EventManagerMixin
""")


def _run_cold(tmp_path, operation):
    repo = Path(__file__).resolve().parents[2]
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(('KESTREL_', 'PYTHON', 'OPENAI_', 'ANTHROPIC_',
                                'GOOGLE_', 'RUNPOD_', 'OTEL_', 'PHOENIX_'))
    }
    environment.update({
        'KESTREL_HOME': str(tmp_path / 'owned-home'),
        'KESTREL_SKIP_DOTENV': '1',
        'PYTHON_DOTENV_DISABLED': '1',
        'PYTHONNOUSERSITE': '1',
        'KESTREL_TRACING_ENABLED': '0',
        'PHOENIX_ENABLED': 'false',
    })
    source = """
import sys
from pathlib import Path
import kestrel_sovereign
assert Path(kestrel_sovereign.__file__).resolve().is_relative_to(Path.cwd())
""" + operation
    result = subprocess.run([sys.executable, '-B', '-c', source], cwd=repo,
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
