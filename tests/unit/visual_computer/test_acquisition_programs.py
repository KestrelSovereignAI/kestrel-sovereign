"""Program identity controls only; these fixtures cannot issue a producer."""
from dataclasses import fields
import httpx
import json
from types import FunctionType, ModuleType

import pytest
from kestrel_sdk.resources import ResourceUnsupported
from kestrel_sovereign.execution_custody import ExecutionAuthorityError
from kestrel_sovereign.visual_computer.runpod_acquisition import (
    _CanonicalJSONDecoder, _CanonicalRunpod, _require_source_class, _source_functions,
)


@pytest.fixture
def programs(tmp_path):
    raw = b"def helper(value):\n    return value['id']\n\nclass Parser:\n    def parse(self, value):\n        return helper(value)\n"
    path = tmp_path/'parser.py'
    path.write_bytes(raw)
    module = ModuleType('owned_program_fixture')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module, raw


@pytest.mark.parametrize('replacement', ['code', 'foreign-globals'])
def test_helper_source_refuses_preexisting_replacement(programs, replacement):
    module, raw = programs
    if replacement == 'code':
        module.helper.__code__ = (lambda value: 'forged-id').__code__
    else:
        module.helper = FunctionType(module.helper.__code__, dict(module.__dict__))
    with pytest.raises(ResourceUnsupported, match='helper program differs'):
        _source_functions(module, raw)


def test_method_source_refuses_original_code_with_foreign_globals(programs):
    module, raw = programs
    original = module.Parser.parse
    module.Parser.parse = FunctionType(original.__code__, dict(module.__dict__))
    with pytest.raises(ResourceUnsupported, match='declared method was replaced'):
        _require_source_class(module.Parser, (module,), {module.__name__: raw})


def test_retained_helper_identity_does_not_allow_changed_program(programs):
    module, raw = programs
    captured = _source_functions(module, raw)
    _require_source_class(module.Parser, (module,), {module.__name__: raw})
    # This descriptor alone has no root/client/row and cannot issue a handle.
    # It exercises the same original-code check used by the finite binder.
    values = {field.name: None for field in fields(_CanonicalRunpod)}
    values.update(modules=((module, frozenset({'helper', 'Parser'})),),
        namespace=((module, 'helper', module.helper), (module, 'Parser', module.Parser)),
        methods=(), programs=captured, decoder=_CanonicalJSONDecoder.capture())
    canonical = _CanonicalRunpod(**values)
    canonical.require_original()
    original = module.helper
    module.helper.__code__ = (lambda value: 'forged-id').__code__
    assert module.helper is original
    with pytest.raises(ExecutionAuthorityError, match='helper implementation changed'):
        canonical.require_original()


@pytest.mark.parametrize('replacement', ['response', 'loads', 'response-code', 'loads-code'])
def test_canonical_decoder_rejects_source_replacement_and_changed_retained_code(monkeypatch, replacement):
    captured = _CanonicalJSONDecoder.capture()
    captured.require_original()
    if replacement.startswith('response'):
        target, name = httpx.Response, 'json'
    else:
        target, name = json, 'loads'
    original = getattr(target, name)
    if replacement.endswith('-code'):
        monkeypatch.setattr(original, '__code__', (lambda *args, **kwargs: {'id': 'ghost'}).__code__)
    else:
        monkeypatch.setattr(target, name, lambda *args, **kwargs: {'id': 'ghost'})
    with pytest.raises(ExecutionAuthorityError):
        captured.require_original()
    with pytest.raises(ResourceUnsupported, match='decoder program differs'):
        _CanonicalJSONDecoder.capture()
