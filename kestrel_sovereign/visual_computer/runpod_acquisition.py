"""Optional closed HOST composition with the installed canonical Runpod code.

No provider is registered or enabled. Source identities restrict one original
host admission; they never grant access, spending, a retry or an SQL program.
"""
from dataclasses import dataclass
import ast
import hashlib
import hmac
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import types

import httpx
from kestrel_sdk.computer_use import EnvironmentLease
from kestrel_sdk.computer_use.acquisition import EnvironmentAcquisitionPreparation
from kestrel_sdk.resources import ResourceOutcomeUnknown, ResourceUnsupported
from kestrel_sovereign.execution_custody import ExecutionAuthorityError

_CANONICAL = None


def _function(value):
    if type(value) is property:
        return value.fget
    return value.__func__ if type(value) in (classmethod, staticmethod) else value


def _nested_codes(code):
    yield code
    for value in code.co_consts:
        if type(value) is types.CodeType:
            yield from _nested_codes(value)


def _source_functions(module, raw):
    """Capture original module helpers, including the canonical parser program."""
    filename = Path(module.__file__).resolve()
    source = ast.parse(raw)
    names = {value.name for value in source.body
        if type(value) in (ast.FunctionDef, ast.AsyncFunctionDef)}
    compiled = compile(raw, str(filename), 'exec', dont_inherit=True)
    declared = {value.co_name: value for value in compiled.co_consts
        if type(value) is types.CodeType and value.co_name in names}
    programs = []
    for name, expected in declared.items():
        actual = module.__dict__.get(name)
        if (type(actual) is not types.FunctionType
                or actual.__globals__ is not module.__dict__
                or Path(actual.__code__.co_filename).resolve() != filename
                or actual.__code__ != expected.replace(co_filename=actual.__code__.co_filename)):
            raise ResourceUnsupported('canonical Runpod helper program differs')
        programs.append((module, name, actual, actual.__code__))
    return tuple(programs)


def _require_source_class(cls, modules, sources):
    """Reject an ordinary pre-import replacement as well as later changes.

    Compile only for code comparison, never execute a second module/parser.
    Generated dataclass methods are captured by identity after checking that
    every explicitly declared canonical method still has its source program.
    """
    module = next((value for value in modules if value.__name__ == cls.__module__), None)
    if module is None:
        raise ResourceUnsupported('canonical Runpod source class required')
    filename = Path(module.__file__).resolve()
    raw = sources[module.__name__]
    compiled = compile(raw, str(filename), 'exec', dont_inherit=True)
    code = next((value for value in compiled.co_consts if type(value) is types.CodeType and value.co_name == cls.__name__), None)
    if code is None:
        raise ResourceUnsupported('canonical Runpod source class is unavailable')
    source_class = next((value for value in ast.parse(raw).body if type(value) is ast.ClassDef and value.name == cls.__name__), None)
    if source_class is None:
        raise ResourceUnsupported('canonical Runpod source class is unavailable')
    names = {value.name for value in source_class.body if type(value) in (ast.FunctionDef, ast.AsyncFunctionDef)}
    declared = {value.co_name: value for value in code.co_consts if type(value) is types.CodeType and value.co_name in names}
    for name, expected in declared.items():
        actual = _function(cls.__dict__.get(name))
        if (type(actual) is not types.FunctionType or actual.__globals__ is not module.__dict__
                or Path(actual.__code__.co_filename).resolve() != filename):
            raise ResourceUnsupported('canonical Runpod declared method was replaced')
        # Runtime import may retain a symlink spelling of the same source.
        expected = expected.replace(co_filename=actual.__code__.co_filename)
        if actual.__code__ != expected:
            raise ResourceUnsupported('canonical Runpod source program differs')
    generated = {'__init__', '__eq__', '__hash__', '__setattr__', '__delattr__'}
    for name, value in cls.__dict__.items():
        actual = _function(value)
        if type(actual) is types.FunctionType and name not in declared:
            if any(type(candidate) is types.CodeType and actual.__code__ == candidate
                    for candidate in code.co_consts):
                continue  # The interpreter's original annotation helper.
            import dataclasses
            import reprlib
            if name in {'__replace__', '__getstate__', '__setstate__'} and any(
                    actual is getattr(dataclasses, candidate, None)
                    for candidate in ('_replace', '_dataclass_getstate', '_dataclass_setstate')):
                continue
            if name == '__repr__' and any(
                    type(generator) is types.FunctionType and actual.__globals__ is generator.__globals__
                    and any(actual.__code__ is candidate for candidate in _nested_codes(generator.__code__))
                    for generator in (reprlib.recursive_repr, getattr(dataclasses, '_recursive_repr', None))):
                continue
            if name == '__hash__' and actual.__module__.startswith('pydantic.'):
                import pydantic
                if Path(actual.__code__.co_filename).resolve().is_relative_to(Path(pydantic.__file__).resolve().parent):
                    continue
            if name not in generated or actual.__code__.co_filename not in {'<string>', '__create_fn__'}:
                raise ResourceUnsupported('canonical Runpod method was substituted')


def _declared_decoder_program(module, name, *, class_name=None):
    """One installed decoder entry, compared without executing another parser."""
    raw = Path(module.__file__).read_bytes()
    filename = Path(module.__file__).resolve()
    code = compile(raw, str(filename), 'exec', dont_inherit=True)
    if class_name is not None:
        code = next(value for value in code.co_consts
            if type(value) is types.CodeType and value.co_name == class_name)
    expected = next(value for value in code.co_consts
        if type(value) is types.CodeType and value.co_name == name)
    return expected, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, eq=False, repr=False)
class _CanonicalJSONDecoder:
    module: object
    function: object
    code: object
    json_module: object
    loads: object
    loads_code: object
    source_sha256: tuple

    @classmethod
    def capture(cls):
        module = importlib.import_module('httpx._models')
        json_module = importlib.import_module('json')
        function, loads = httpx.Response.__dict__.get('json'), json_module.__dict__.get('loads')
        expected, source = _declared_decoder_program(module, 'json', class_name='Response')
        expected_loads, loads_source = _declared_decoder_program(json_module, 'loads')
        if (type(function) is not types.FunctionType or function.__globals__ is not module.__dict__
                or type(loads) is not types.FunctionType or loads.__globals__ is not json_module.__dict__
                or function.__code__ != expected.replace(co_filename=function.__code__.co_filename)
                or loads.__code__ != expected_loads.replace(co_filename=loads.__code__.co_filename)
                or module.__dict__.get('jsonlib') is not json_module):
            raise ResourceUnsupported('Canonical native JSON decoder program differs')
        return cls(module, function, function.__code__, json_module, loads, loads.__code__,
            (('httpx._models', source), ('json', loads_source)))

    def require_storage_original(self):
        # Fixed Core private-row codecs also consume stdlib loads. A changed
        # storage decoder cannot be borrowed even for an original fact write.
        if (self.json_module.__dict__.get('loads') is not self.loads
                or self.loads.__code__ is not self.loads_code
                or self.loads.__globals__ is not self.json_module.__dict__):
            raise ExecutionAuthorityError('Original private storage decoder changed')

    def require_original(self):
        self.require_storage_original()
        # These are the directly consumed canonical Response.json entry and
        # global loads callable. This is not recursive Python/C attestation.
        if (httpx.Response.__dict__.get('json') is not self.function
                or self.function.__code__ is not self.code
                or self.function.__globals__ is not self.module.__dict__
                or self.module.__dict__.get('jsonlib') is not self.json_module
                or self.json_module.__dict__.get('loads') is not self.loads
                or self.loads.__code__ is not self.loads_code
                or self.loads.__globals__ is not self.json_module.__dict__):
            raise ExecutionAuthorityError('Canonical native JSON decoder changed')


@dataclass(frozen=True, eq=False, repr=False)
class _CanonicalRunpod:
    modules: tuple
    namespace: tuple
    methods: tuple
    programs: tuple
    source_sha256: tuple
    decoder: _CanonicalJSONDecoder
    client_type: object
    transport_type: object
    producer_type: object
    phase_type: object
    pod_type: object
    request_type: object
    plan_type: object
    preparation_type: object

    def require_original(self):
        self.decoder.require_original()
        for module, names in self.modules:
            if frozenset(name for name in module.__dict__ if not name.startswith('__')) != names:
                raise ExecutionAuthorityError('Canonical Runpod module changed')
        for module, name, value in self.namespace:
            if module.__dict__.get(name) is not value:
                raise ExecutionAuthorityError('Canonical Runpod binding changed')
        for cls, name, value, function, code in self.methods:
            if cls.__dict__.get(name) is not value or function.__code__ is not code:
                raise ExecutionAuthorityError('Canonical Runpod implementation changed')
        for module, name, function, code in self.programs:
            if (module.__dict__.get(name) is not function or function.__code__ is not code
                    or function.__globals__ is not module.__dict__):
                raise ExecutionAuthorityError('Canonical Runpod helper implementation changed')


def _canonical_runpod():
    global _CANONICAL
    if _CANONICAL is not None:
        _CANONICAL.require_original()
        return _CANONICAL
    try:
        clients = importlib.import_module('kestrel_cloud_runpod.clients')
        models = importlib.import_module('kestrel_cloud_runpod.models')
        cpu = importlib.import_module('kestrel_cloud_runpod.cpu_environment')
        prepared = importlib.import_module('kestrel_cloud_runpod.prepared_cpu')
        # Supported quote codec imports must complete BEFORE any namespace is
        # sealed. An ordinary import later must not invalidate an issued A
        # owner. Original acquisition-only vendors need not provide C1q.
        quote_modules = ()
        if importlib.util.find_spec('kestrel_cloud_runpod.cpu_environment_codec') is not None:
            codec = importlib.import_module('kestrel_cloud_runpod.cpu_environment_codec')
            catalog = importlib.import_module('kestrel_cloud_runpod.cpu_catalog')
            realized = importlib.import_module('kestrel_cloud_runpod.cpu_realized')
            quote_modules = (codec, catalog, realized)
        producer_type = clients.__dict__['_PreparedCpuAcquisitionProducer']
        factory = clients.RunpodControlPlaneClient.__dict__['prepare_cpu_acquisition_producer']
        phase_type = models.__dict__['_PodNonCostPhase']
    except (ImportError, KeyError, AttributeError):
        raise ResourceUnsupported('canonical Runpod acquisition producer is unavailable') from None
    if type(factory) is not types.FunctionType or not inspect.isclass(producer_type):
        raise ResourceUnsupported('canonical Runpod acquisition factory is unavailable')
    modules = (clients, models, cpu, prepared, *quote_modules)
    classes = (clients.RunpodControlPlaneClient, clients.RunpodTransport, producer_type,
        models.PodResource, phase_type, models.CpuPodCreateRequest,
        cpu.CpuEnvironmentAcquisitionPlan, prepared.PreparedCpuPodCreate)
    if quote_modules:
        classes += (cpu.CpuEnvironmentProfile, cpu.CpuNonComputeCostPolicy,
            cpu.CpuEnvironmentPlanner, catalog.CpuCatalogSnapshot, catalog.CpuPodCatalogMatch)
    methods = []
    sources = {module.__name__: Path(module.__file__).read_bytes() for module in modules}
    source_sha256 = tuple((name, hashlib.sha256(raw).hexdigest()) for name, raw in sources.items())
    programs = tuple(program for module in modules for program in _source_functions(module, sources[module.__name__]))
    for cls in classes:
        _require_source_class(cls, modules, sources)
        for name, value in cls.__dict__.items():
            function = _function(value)
            if type(function) is types.FunctionType:
                methods.append((cls, name, value, function, function.__code__))
    _CANONICAL = _CanonicalRunpod(
        tuple((module, frozenset(name for name in module.__dict__ if not name.startswith('__'))) for module in modules),
        tuple((module, name, value) for module in modules for name, value in module.__dict__.items() if not name.startswith('__')),
        tuple(methods), programs, source_sha256, _CanonicalJSONDecoder.capture(), clients.RunpodControlPlaneClient, clients.RunpodTransport,
        producer_type, phase_type, models.PodResource, models.CpuPodCreateRequest,
        cpu.CpuEnvironmentAcquisitionPlan, prepared.PreparedCpuPodCreate)
    return _CANONICAL


def _parts(request):
    if (type(request) is not httpx.Request or type(request.url) is not httpx.URL
            or type(getattr(request, '_content', None)) is not bytes
            or type(request.headers) is not httpx.Headers or type(request.extensions) is not dict
            or set(request.extensions) != {'timeout'} or type(request.extensions['timeout']) is not dict):
        raise ExecutionAuthorityError('Original native prepared request required')
    return (request.method, request.url.raw_scheme, request.url.raw_host, request.url.port,
        request.url.raw_path, tuple(request.headers.raw), tuple(sorted(request.extensions['timeout'].items())), request._content)


class _RunpodBinding:
    def __init__(self, attempt, client, pod_request, prepared, plan, preparation):
        from .storage_acquisition import _CoreAcquisitionAttempt
        from kestrel_sovereign.execution_sync_http import _configuration
        if type(self) is not _RunpodBinding or type(attempt) is not _CoreAcquisitionAttempt:
            raise ExecutionAuthorityError('Exact original acquisition attempt required')
        attempt.check_current()
        canonical = self._canonical = _canonical_runpod()
        if (type(client) is not canonical.client_type or type(client.transport) is not canonical.transport_type
                or type(pod_request) is not canonical.request_type or type(prepared) is not canonical.preparation_type
                or type(plan) is not canonical.plan_type
                or getattr(client.transport, '_prepared_cpu_acquisition', None) is not True):
            raise ResourceUnsupported('original dedicated Runpod acquisition profile required')
        self._root, self._attempt = attempt._result, attempt
        self._vendor, self._transport = client, client.transport
        self._client = client.transport._client
        _, self._pool = _configuration(self._client)
        if self._client.event_hooks and any(self._client.event_hooks.values()):
            raise ResourceUnsupported('acquisition refuses native event hooks')
        self._pod_request, self._prepared, self._plan = pod_request, prepared, plan
        self._preparation = EnvironmentAcquisitionPreparation.model_validate(preparation)
        self._body = client._prepared_body(pod_request, prepared, plan)
        current = self._root._value()
        if (self._body != self._preparation.create_payload.get_secret_value()
                or plan.request != current.request or current.quote.expires_at > plan.expires_at
                or current.quote.hourly_cost_usd != plan.all_in_hourly_usd
                or current.quote.estimated_total_cost_usd != plan.estimated_total_usd):
            raise ResourceUnsupported('original Runpod preparation differs from admitted request/quote')
        self._config = self._transport._prepared_configuration()
        if current.remote_environment is not None:
            from .runpod_quote_storage import _bytes
            projection = current.remote_environment
            modules = {module.__name__.rsplit('.', 1)[-1]: module for module, _ in canonical.modules}
            codec = modules.get('cpu_environment_codec')
            if codec is None:
                raise ResourceUnsupported('Original quote-plan codec required')
            original_plan = codec.deserialize_cpu_environment_plan(projection.plan_bytes,
                expected_storage_sha256=projection.plan_storage_sha256)
            if (codec.serialize_cpu_environment_plan(plan).get_secret_value() != projection.plan_bytes.get_secret_value()
                    or original_plan.digest() != projection.plan_digest
                    or hashlib.sha256(_bytes(dict(canonical.source_sha256))).hexdigest() != projection.binding.source_sha256
                    or self._config['account'] != projection.binding.account
                    or self._config['origin'] != projection.binding.base_url
                    or self._config['ca_data_sha256'] != projection.binding.tls_sha256):
                raise ResourceUnsupported('Original Runpod quote native binding differs')
            authorization = self._client.headers.get('authorization', '')
            if not authorization.startswith('Bearer '):
                raise ResourceUnsupported('Original Runpod credential binding absent')
            actual_fingerprint = hmac.new(self._root._key,
                b'runpod-current-key-v1\0'+authorization[7:].encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(actual_fingerprint, projection.binding.credential_fingerprint):
                raise ResourceUnsupported('Original Runpod credential binding differs')
            codec.require_cpu_environment_quote(original_plan, current.quote,
                capabilities=projection.capabilities, at=plan.planned_at)
            if hashlib.sha256(codec._json_bytes(current.quote.model_dump(mode='python'))).hexdigest() != projection.quote_binding_sha256:
                raise ResourceUnsupported('Original Runpod quote storage differs')
        self._headers = tuple(self._client.headers.raw)
        self._base = self._client.base_url
        if self._client.headers.get_list('accept-encoding') != ['identity']:
            raise ResourceUnsupported('original identity encoding profile required')
        self._producer = self._operation = self._native_request = self._request_parts = None
        self._binding_bytes = self._native_result_bytes = None
        checkpoint = attempt.checkpoint
        self._lease_facts = dict(lease_id='lease-'+checkpoint.attempt_id,
            request_id=current.request.request_id, quote_id=current.quote.quote_id,
            scope=current.request.scope, provider_id=current.request.provider_id,
            location=current.request.location, expires_at=current.request.expires_at)

    def _require_integrity(self):
        self._canonical.require_original()
        if (type(self) is not _RunpodBinding or self._vendor.transport is not self._transport
                or self._transport._client is not self._client or self._client._transport._pool is not self._pool
                or tuple(self._client.headers.raw) != self._headers or self._client.base_url != self._base):
            raise ExecutionAuthorityError('Original Runpod native binding changed')
        if self._producer is not None:
            if (type(self._producer) is not self._canonical.producer_type
                    or self._producer.operation is not self._operation
                    or self._producer._native_request is not self._native_request
                    or self._producer._control is not self._vendor
                    or self._producer._request is not self._pod_request
                    or self._producer._prepared is not self._prepared
                    or self._producer._plan is not self._plan
                    or self._producer._body is not self._body
                    or self._producer._original_acquisition is not self._root._native):
                raise ExecutionAuthorityError('Original Runpod native producer changed')

    def _require_current(self):
        self._require_integrity()
        self._root.check_current()

    def _bind(self, handle):
        self._require_current()
        producer = self._vendor.prepare_cpu_acquisition_producer(self._pod_request,
            prepared=self._prepared, plan=self._plan, _original_acquisition=handle)
        if type(producer) is not self._canonical.producer_type:
            raise ExecutionAuthorityError('Canonical original Runpod producer required')
        self._producer = producer
        self._operation, self._native_request = producer.operation, producer._native_request
        if type(self._operation) is not types.MethodType or self._operation.__self__ is not producer:
            raise ExecutionAuthorityError('Original canonical bound operation required')
        if not any(cls is self._canonical.producer_type and function is self._operation.__func__
                for cls, _, _, function, _ in self._canonical.methods):
            raise ExecutionAuthorityError('Original canonical producer function required')
        parts = self._request_parts = _parts(self._native_request)
        if parts[0] != 'POST' or parts[1] != b'https' or parts[-1] != self._body:
            raise ExecutionAuthorityError('Original Runpod prepared native request differs')
        self._binding_bytes = json.dumps(dict(schema_version=1,
            method=parts[0], scheme=parts[1].hex(), host=parts[2].hex(), port=parts[3],
            path=parts[4].hex(), headers=[(name.hex(), value.hex()) for name, value in parts[5]],
            timeout=parts[6], body_sha256=hashlib.sha256(parts[7]).hexdigest(),
            configuration=self._config, plan_sha256=self._plan.digest(),
            canonical_sources=self._canonical.source_sha256,
            canonical_decoders=self._canonical.decoder.source_sha256), sort_keys=True, separators=(',', ':')).encode()
        if len(self._binding_bytes) > 16384:
            raise ResourceUnsupported('original native preparation binding exceeds bounds')
        self._require_integrity()

    def _require_saved_preparation(self):
        from .repository import _AcquisitionEnvelope
        self._require_integrity()
        envelope = _AcquisitionEnvelope.from_private(json.loads(self._root._plain)['acquisition'])
        storage = self._root._storage()
        if (envelope.binding is None or envelope.binding.get_secret_value() != self._binding_bytes
                or storage.preparation is None or storage.preparation.digest() != self._preparation.digest()):
            raise ExecutionAuthorityError('Original saved native preparation differs')

    def _require_native_entry(self):
        from kestrel_sovereign.execution_sync_http import _configuration
        self._require_current()
        self._require_saved_preparation()
        _configuration(self._client, self._client._kestrel_sync_admission)
        if self._pool.connections or self._pool._requests:
            raise ExecutionAuthorityError('Original acquisition native pool is not pristine')
        if self._transport._prepared_configuration() != self._config:
            raise ExecutionAuthorityError('Original acquisition native configuration changed')

    def _require_request_write(self):
        self._require_integrity()
        if self._vendor._prepared_body(self._pod_request, self._prepared, self._plan) != self._body:
            raise ExecutionAuthorityError('Original saved acquisition bytes changed')

    def _require_actual_request(self, request):
        self._require_integrity()
        if _parts(request) != self._request_parts:
            raise ExecutionAuthorityError('Actual original native request differs from saved preparation')

    def _capture_phase(self, phase):
        self._root._native.check_parse_current()
        producer = self._producer
        if (type(phase) is not self._canonical.phase_type or producer._phase is not phase
                or phase._producer_token is not producer._producer_token):
            raise ExecutionAuthorityError('Original canonical non-cost phase required')
        handle = self._root._native
        if not handle._exhausted or not handle._response_closed or handle._response.status_code != 201:
            raise ExecutionAuthorityError('Original consumed native create response required')
        # These exact frozen slots have already passed the canonical bounded
        # parser. No caller mapping/property/cost continuation is invoked here.
        raw = json.dumps(dict(schema_version=1, id=phase.id, name=phase.name, status=phase.status,
            gpu_id=phase.gpu_id, gpu_count=phase.gpu_count, facts=phase.facts),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > 1024*1024:
            raise ExecutionAuthorityError('Canonical bounded native phase exceeded its reserved size')
        self._native_result_bytes = raw
        lease = EnvironmentLease(**self._lease_facts, provider_resource_id=phase.id,
            state='unknown', actual_cost_usd=None)
        lease_json = lease.model_dump_json().encode()
        self._root._native.check_parse_current()
        return phase, lease, lease_json, raw

    def _require_public_result(self):
        self._require_integrity()
        result, phase = self._producer._public_result, self._producer._phase
        if type(result) is not self._canonical.pod_type or phase is not self._root._native._phase:
            raise ExecutionAuthorityError('Original canonical public parse has not completed')
        if any(type(getattr(result, name)) is not type(getattr(phase, name))
                or getattr(result, name) != getattr(phase, name)
                for name in ('id', 'name', 'status', 'gpu_id', 'gpu_count')):
            raise ExecutionAuthorityError('Original canonical public result differs')


async def prepare_runpod_acquisition_producer(attempt, client, pod_request, *, prepared, plan, preparation):
    """Issue inactive, bind canonical native parts, then ACK original preparation.

    The caller holds its same original host operation through dispatch and
    final delivery. This helper never begins a native create or registers a
    provider. A restarted dispatching/unknown attempt gets no new handle.
    """
    from .storage_acquisition import _CoreAcquisitionAttempt
    from .acquisition_native import _NativeAcquisitionHandle
    if type(attempt) is not _CoreAcquisitionAttempt:
        raise ExecutionAuthorityError('Exact original acquisition attempt required')
    attempt.check_current()
    if attempt.checkpoint.state not in {'unprepared', 'prepared'}:
        raise ResourceOutcomeUnknown('original create already began; reconciliation only')
    binding = _RunpodBinding(attempt, client, pod_request, prepared, plan, preparation)
    handle = _NativeAcquisitionHandle(binding)
    attempt._result._attach_native(handle)
    binding._bind(handle)
    await attempt._result._prepare_native(handle, preparation)
    return binding._producer
