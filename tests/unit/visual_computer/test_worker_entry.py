"""The private image launcher accepts a fixed installed role, never a program."""
from pathlib import Path
import signal
import sys

import pytest

from kestrel_sovereign.visual_computer import worker_entry


@pytest.mark.parametrize("restoration_fails",[False,True])
def test_launcher_preserves_exec_error_and_actual_mask_restore_error(
    monkeypatch,tmp_path,process_signals,restoration_fails,
):
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    proxy,_,_=process_signals
    resource=tmp_path/"_worker_programs/broker_worker.py"
    resource.parent.mkdir();resource.write_text("# inert exec-error fixture\n")
    monkeypatch.setattr(worker_entry,"files",lambda _package:tmp_path)
    monkeypatch.setattr(sys,"argv",["worker_entry","broker"])
    original=OSError("scripted original exec failure")
    restoration=RuntimeError("scripted original mask restoration failure")
    def execute(*args):
        raise original
    def mask(how,numbers):
        if how==signal.SIG_SETMASK and restoration_fails:
            raise restoration
        return set()
    proxy.pthread_sigmask=mask
    monkeypatch.setattr(worker_entry.os,"execv",execute)
    with pytest.raises(BaseException)as caught:worker_entry.main()
    if restoration_fails:
        graph=execution_evidence_carriers(caught.value)
        assert original in graph and restoration in graph
    else:
        assert caught.value is original



@pytest.mark.asyncio
async def test_process_lifetime_preserves_original_body_and_terminal_mask_errors():
    from types import SimpleNamespace
    from kestrel_sovereign.execution_custody import execution_evidence_carriers
    original=ValueError("scripted original body error")
    closing=OSError("scripted original terminal mask error")
    async def body():
        raise original
    def detach():
        raise closing
    with pytest.raises(BaseException)as caught:
        await worker_entry.ProcessLifetime.run(SimpleNamespace(detach=detach),body())
    graph=execution_evidence_carriers(caught.value)
    assert original in graph and closing in graph


@pytest.mark.asyncio
async def test_process_lifetime_preserves_same_body_error_when_terminal_guard_succeeds():
    from types import SimpleNamespace
    original=ValueError("scripted original body")
    async def body():
        raise original
    with pytest.raises(ValueError)as caught:
        await worker_entry.ProcessLifetime.run(SimpleNamespace(detach=lambda:None),body())
    assert caught.value is original


def test_terminal_process_mask_is_retained_after_loop_detach(process_signals):
    from types import SimpleNamespace
    _,_,masks=process_signals
    lifetime=worker_entry.ProcessLifetime()
    lifetime.bind(SimpleNamespace(call_soon_threadsafe=lambda callback:None),lambda:None)
    lifetime.detach()
    assert masks==[(signal.SIG_UNBLOCK,worker_entry._RETIRE_SIGNALS),
                   (signal.SIG_BLOCK,worker_entry._RETIRE_SIGNALS)]


@pytest.mark.parametrize("role",["browser","broker"])
def test_closed_profile_command_uses_guarded_fixed_file(role):
    from kestrel_sovereign.visual_computer.worker_profile import LocalWorkerProfile,container_arguments
    profile=LocalWorkerProfile(browser_image="browser@sha256:"+"a"*64,
                               broker_image="broker@sha256:"+"b"*64)
    args=container_arguments(profile,role,"c"*64)
    assert args[-4:]==["python","-B","/opt/kestrel-worker-entry.py",role]



@pytest.fixture
def process_signals(monkeypatch):
    from types import SimpleNamespace
    handlers, masks = {}, []
    proxy = SimpleNamespace(SIG_BLOCK=signal.SIG_BLOCK, SIG_UNBLOCK=signal.SIG_UNBLOCK,
        SIG_SETMASK=signal.SIG_SETMASK, pending=False)
    def install(number, callback):
        handlers[number] = callback
    def mask(how, numbers):
        masks.append((how,tuple(numbers)))
        if how == signal.SIG_UNBLOCK and proxy.pending:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return set()
    proxy.signal, proxy.pthread_sigmask = install, mask
    monkeypatch.setattr(worker_entry, "signal", proxy)
    return proxy, handlers, masks


def test_process_lifetime_latches_once_and_only_schedules_loop_work(process_signals):
    from types import SimpleNamespace
    _, handlers, _ = process_signals
    queued, called = [], []
    lifetime = worker_entry.ProcessLifetime()
    lifetime.bind(SimpleNamespace(call_soon_threadsafe=queued.append), lambda:called.append(True))
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert lifetime.requested and not called and len(queued)==1
    queued.pop()()
    assert called==[True]
    lifetime.detach()
    assert all(callable(handlers[n])for n in (signal.SIGINT,signal.SIGTERM))


def test_pending_exec_withdrawal_is_not_lost_before_loop_binding(process_signals):
    from types import SimpleNamespace
    proxy, _, masks = process_signals
    proxy.pending = True
    lifetime = worker_entry.ProcessLifetime()
    assert lifetime.requested
    assert masks==[(signal.SIG_UNBLOCK,worker_entry._RETIRE_SIGNALS)]
    queued, called = [], []
    lifetime.bind(SimpleNamespace(call_soon_threadsafe=queued.append), lambda:called.append(True))
    assert not called and len(queued)==1
    queued[0]()
    assert called==[True]
    lifetime.detach()


def test_detached_process_lifetime_never_schedules_closed_loop(process_signals):
    from types import SimpleNamespace
    _, handlers, _ = process_signals
    def closed_loop(_callback):
        pytest.fail("terminal process signal reached a closed loop")
    lifetime = worker_entry.ProcessLifetime()
    loop = SimpleNamespace(call_soon_threadsafe=closed_loop)
    lifetime.bind(loop, lambda:pytest.fail("terminal signal reentered cleanup"))
    with pytest.raises(RuntimeError,match="already_bound"):
        lifetime.bind(loop,lambda:None)
    lifetime.detach()
    handlers[signal.SIGTERM](signal.SIGTERM,None)
    assert lifetime.requested and all(callable(handlers[n])for n in (signal.SIGINT,signal.SIGTERM))


@pytest.mark.parametrize("role",["normal","priority"])
def test_fixed_launcher_accepts_only_canonical_broker_lane(monkeypatch,tmp_path,role):
    resource=tmp_path/"_worker_programs/broker_worker.py"
    resource.parent.mkdir();resource.write_text("# inert canonical-path fixture\n")
    monkeypatch.setattr(worker_entry,"files",lambda _package:tmp_path)
    monkeypatch.setattr(sys,"argv",["worker_entry","broker","--lane",role])
    original_mask=signal.pthread_sigmask(signal.SIG_BLOCK,[])
    calls=[]
    class ExecObserved(BaseException):
        pass
    def execute(executable,args):
        assert set(worker_entry._RETIRE_SIGNALS)<=signal.pthread_sigmask(signal.SIG_BLOCK,[])
        calls.append((executable,args))
        raise ExecObserved
    monkeypatch.setattr(worker_entry.os,"execv",execute)
    with pytest.raises(ExecObserved):worker_entry.main()
    assert calls==[(sys.executable,[sys.executable,"-B",str(resource),"--lane",role])]
    assert signal.pthread_sigmask(signal.SIG_BLOCK,[])==original_mask


@pytest.mark.parametrize("arguments",[["browser","--lane","normal"],["broker","--lane","other"],
    ["broker","--lane","normal","extra"]])
def test_lane_launcher_refuses_other_arguments_before_resource_lookup(monkeypatch,arguments):
    monkeypatch.setattr(sys,"argv",["worker_entry",*arguments])
    monkeypatch.setattr(worker_entry,"files",lambda _package:pytest.fail("invalid lane reached resource"))
    with pytest.raises(SystemExit,match="exact private worker role"):worker_entry.main()



class ScriptedProcessLifetime:
    """Unit callback injection only; no process handler or native ACK."""
    requested = False

    def bind(self, loop, callback):
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(number, callback)

    def detach(self):
        pass


@pytest.mark.parametrize("arguments", [[], ["worker.py"], ["../worker.py"],
    ["browser", "--lane", "priority"], ["broker", "--command", "anything"],
    ["broker", "--lane"], ["browser", "--initialize", "anything"]])
def test_launcher_refuses_programs_and_forwarded_arguments(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["worker_entry", *arguments])

    def forbidden_resource(_package):
        pytest.fail("invalid role reached resource lookup")

    monkeypatch.setattr(worker_entry, "files", forbidden_resource)
    with pytest.raises(SystemExit, match="exact private worker role required"):
        worker_entry.main()


def test_missing_installed_program_has_no_source_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["worker_entry", "browser"])
    monkeypatch.setattr(worker_entry, "files", lambda _package: tmp_path)

    def forbidden_exec(*_args):
        pytest.fail("absent installed resource reached native exec")

    monkeypatch.setattr(worker_entry.os, "execv", forbidden_exec)
    with pytest.raises(FileNotFoundError):
        worker_entry.main()


@pytest.mark.parametrize("role,program", [
    ("browser", "two_domain_worker.py"), ("broker", "broker_worker.py"),
])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_launcher_replaces_itself_with_same_interpreter_fixed_resource(
    monkeypatch, tmp_path, role, program, bootstrap,
):
    resource = tmp_path / "_worker_programs" / program
    resource.parent.mkdir()
    resource.write_text("# inert launcher test fixture\n")
    arguments = ["--initialize"] if bootstrap else []
    monkeypatch.setattr(sys, "argv", ["worker_entry", role, *arguments])
    monkeypatch.setattr(worker_entry, "files", lambda _package: tmp_path)
    calls = []

    class ExecObserved(BaseException):
        pass

    def observe_exec(executable, arguments):
        calls.append((executable, arguments))
        raise ExecObserved

    monkeypatch.setattr(worker_entry.os, "execv", observe_exec)
    with pytest.raises(ExecObserved):
        worker_entry.main()
    assert calls == [(sys.executable, [sys.executable, "-B", str(resource), *arguments])]
    assert Path(calls[0][1][2]).is_relative_to(tmp_path)


# Filesystem regressions only: scripted bootstrap/role inputs do not establish
# Host admission, Linux UID/GID/mount identity, a TLS peer or a live worker.
@pytest.fixture
def canonical_private_worker_programs():
    import importlib.util
    from importlib.resources import files

    repository = Path(__file__).resolve().parents[3]
    if Path(worker_entry.__file__).resolve().parent == repository / "kestrel_sovereign/visual_computer":
        # Explicit source-CI test data, never a production launcher fallback.
        directory = repository / "docker/visual-browser"
    else:
        # Normal installed tests MUST use the actual packaged programs.
        directory = Path(str(files("kestrel_sovereign.visual_computer").joinpath("_worker_programs"))).resolve(strict=True)
        assert directory.parent == Path(worker_entry.__file__).resolve().parent
    names = ("controller", "wire", "sandbox_checks", "worker", "two_domain_worker", "broker_worker")
    previous = {name: sys.modules.get(name) for name in names}
    loaded = {}
    try:
        for name in names:
            path = directory / (name + ".py")
            assert path.is_file()
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            assert Path(module.__file__).resolve() == path.resolve()
            loaded[name] = module
        yield loaded
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@pytest.fixture
def private_file_bootstrap_data():
    from datetime import UTC, datetime, timedelta
    from kestrel_sovereign.visual_computer.worker_profile import PROFILE

    broker = {
        "profile": PROFILE, "identity": "a" * 64, "browser_incarnation": "b" * 64,
        "broker_incarnation": "c" * 64, "profile_sha256": "d" * 64,
        "expires_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        "client_ca": "SCRIPTED bootstrap-only CA, not a TLS identity",
        "client_sha256": "e" * 64,
        "roles": {role: {"certificate": "SCRIPTED bootstrap-only certificate",
            "key": "SCRIPTED bootstrap-only key", "sha256": "f" * 64}
            for role in ("normal", "priority")},
    }
    browser = {"schema_version": 1, "secret": "a" * 64, "incarnation": "b" * 64,
        "request": {"test_only_before_browser_engine": True},
        "document_html": "<p>filesystem unit data</p>", "network_profile": None,
        "profile_sha256": "d" * 64}
    return {"broker": broker, "browser": browser}


@pytest.mark.asyncio
@pytest.mark.parametrize("withdrawal_during_setup",[False,True])
async def test_browser_main_joins_close_after_five_seconds_before_original_manager_exit(
    monkeypatch, tmp_path, canonical_private_worker_programs, withdrawal_during_setup,
):
    """Scripted browser, real owned Tasks and elapsed time; not native proof.

    The unchanged real browser took7.96s to close in the lifecycle measurement.
    Its five-second wrapper cancelled a legitimate close. Exercise main's
    actual shutdown wiring with a six-second original acknowledgement.
    """
    import asyncio
    from contextlib import asynccontextmanager
    from datetime import UTC, datetime, timedelta
    import os
    import signal
    from types import SimpleNamespace
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task

    program=canonical_private_worker_programs["two_domain_worker"]
    wire=canonical_private_worker_programs["wire"]
    private,shared=tmp_path/"private",tmp_path/"shared"
    shared.mkdir()
    handlers,alarms,events={},[],[]
    loop=SimpleNamespace(add_signal_handler=lambda number, callback:handlers.setdefault(number,callback))
    monkeypatch.setattr(program,"asyncio",SimpleNamespace(get_running_loop=lambda:loop,
        Event=asyncio.Event,timeout=asyncio.timeout,sleep=asyncio.sleep))
    monkeypatch.setattr(program,"signal",SimpleNamespace(SIGINT=signal.SIGINT,SIGTERM=signal.SIGTERM,
        alarm=lambda seconds:alarms.append(seconds)))
    monkeypatch.setattr(program,"PRIVATE",str(private));monkeypatch.setattr(program,"IPC",str(shared))
    monkeypatch.setattr(program,"os",SimpleNamespace(getuid=lambda:10002,getgid=lambda:10003,
        umask=os.umask,open=os.open,fchmod=os.fchmod))
    monkeypatch.setattr(program,"process_evidence",lambda:{})
    context=SimpleNamespace()
    async def nothing(*args,**kwargs):return None
    page=SimpleNamespace(context=context,set_content=nothing)
    async def new_page():return page
    context.new_page=new_page;context.route_web_socket=nothing;context.add_init_script=nothing;context.route=nothing
    class Browser:
        async def new_context(self,**kwargs):
            if withdrawal_during_setup:handlers[signal.SIGTERM]()
            return context
        async def close(self):
            events.append("close-enter")
            await asyncio.sleep(6)
            events.append("original-close-ack")
    browser=Browser()
    async def launch(**kwargs):return browser
    async def evidence(_browser):return {}
    monkeypatch.setattr(program,"chromium_evidence",evidence)
    @asynccontextmanager
    async def manager():
        try:yield SimpleNamespace(chromium=SimpleNamespace(launch=launch))
        finally:events.append("original-manager-exit")
    monkeypatch.setattr(program,"original_playwright",manager)
    async def serve(controller,secret):
        handlers[signal.SIGTERM]()
        assert controller._session["state"]=="closing"
        assert alarms[-1]==40,"lifecycle withdrawal must arm the original shutdown watchdog"
        before=list(alarms)
        handlers[signal.SIGTERM]()
        assert alarms==before,"repeated withdrawal must not extend the original shutdown deadline"
    monkeypatch.setattr(program,"serve",serve)
    request={"schema_version":1,"session_id":"scripted-session","lease_id":"scripted-lease",
        "scope":{"schema_version":1,"tenant_id":"scripted","principal_id":"scripted","agent_id":None},
        "surface":"browser","required_capabilities":["screenshot"],
        "expires_at":(datetime.now(UTC)+timedelta(minutes=5)).isoformat()}
    previous_mask=os.umask(0o077)
    task=None
    try:
        task=create_owned_task(program.main(ScriptedProcessLifetime()),admission=OwnedTaskAdmission(),retain_originals=True)
        async with asyncio.timeout(15):
            while not(private/"incarnation").exists():await asyncio.sleep(.001)
            value={"schema_version":1,"secret":"a"*64,"incarnation":(private/"incarnation").read_text(),
                "request":request,"document_html":"<p>scripted</p>","network_profile":None,"profile_sha256":"d"*64}
            (private/"bootstrap.json").write_bytes(wire.encode(value))
            outcome=await await_owned_task(task)
    finally:
        if task is not None:await await_owned_task(task)
        os.umask(previous_mask)
    if outcome.error is not None:raise outcome.error
    if outcome.cancellation is not None:raise outcome.cancellation
    assert events==["close-enter","original-close-ack","original-manager-exit"]
    assert any(30<=seconds<=40 for seconds in alarms),"lifecycle withdrawal must arm a bounded shutdown watchdog"
    assert alarms.count(40)==1,"the close phase must not restart a withdrawal watchdog"


@pytest.mark.parametrize("role", ["broker", "browser"])
def test_private_bootstrap_files_are_exclusive_0600(
    monkeypatch, tmp_path, canonical_private_worker_programs, private_file_bootstrap_data, role,
):
    import os
    import stat

    modules = canonical_private_worker_programs
    program = modules["broker_worker" if role == "broker" else "two_domain_worker"]
    private = tmp_path / role
    private.mkdir(mode=0o700)
    monkeypatch.setattr(program, "PRIVATE", private)
    value = private_file_bootstrap_data[role]
    incarnation = value["broker_incarnation" if role == "broker" else "incarnation"]
    (private / "incarnation").write_text(incarnation)
    raw = modules["wire"].encode(value)
    # A permissive caller mask makes the writer's requested0600 load-bearing.
    previous_mask = os.umask(0)
    try:
        result = program.initialize(raw)
        original = (private / "bootstrap.json").read_bytes()
        assert modules["wire"].decode(result) == {"schema_version": 1, "submitted": True}
        assert original == raw
        assert stat.S_IMODE((private / "bootstrap.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((private / "initialized").stat().st_mode) == 0o600
        assert not (private / "bootstrap.pending").exists()
        with pytest.raises(FileExistsError):
            program.initialize(raw)
        assert (private / "bootstrap.json").read_bytes() == original
    finally:
        os.umask(previous_mask)


@pytest.mark.parametrize("role", ["broker", "browser"])
def test_private_bootstrap_existing_pending_is_not_overwritten(
    monkeypatch, tmp_path, canonical_private_worker_programs, private_file_bootstrap_data, role,
):
    import stat

    modules = canonical_private_worker_programs
    program = modules["broker_worker" if role == "broker" else "two_domain_worker"]
    private = tmp_path / role
    private.mkdir(mode=0o700)
    monkeypatch.setattr(program, "PRIVATE", private)
    value = private_file_bootstrap_data[role]
    (private / "incarnation").write_text(value["broker_incarnation" if role == "broker" else "incarnation"])
    pending = private / "bootstrap.pending"
    pending.write_bytes(b"original pending bytes")
    pending.chmod(0o600)
    with pytest.raises(FileExistsError):
        program.initialize(modules["wire"].encode(value))
    assert pending.read_bytes() == b"original pending bytes"
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600
    assert not (private / "bootstrap.json").exists()


@pytest.fixture
def real_private_file_tls_material():
    import hashlib
    from datetime import UTC, datetime, timedelta
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.now(UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "unit-filesystem-ca")])
    ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "unit-filesystem-server")]))
        .issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(days=1))
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(ca_key, hashes.SHA256()))
    certificate = leaf.public_bytes(serialization.Encoding.PEM).decode()
    private_key = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    return {"ca": ca.public_bytes(serialization.Encoding.PEM).decode(),
        "certificate": certificate, "key": private_key,
        "sha256": hashlib.sha256(leaf.public_bytes(serialization.Encoding.DER)).hexdigest()}


@pytest.mark.parametrize("role_name", ["normal", "priority"])
def test_broker_tls_private_files_are_loaded_0600_then_retired(
    monkeypatch, tmp_path, private_file_bootstrap_data, real_private_file_tls_material, role_name,
):
    import hashlib
    import os
    import ssl
    import stat
    import tempfile
    from types import SimpleNamespace
    from kestrel_sovereign.visual_computer import worker_broker

    material = real_private_file_tls_material
    value = dict(private_file_bootstrap_data["broker"], client_ca=material["ca"],
        roles={role_name: {k: material[k] for k in ("certificate", "key", "sha256")}})
    original_directory = tempfile.TemporaryDirectory
    original_load = ssl.SSLContext.load_cert_chain
    observed = []
    def own_directory(*, prefix, dir):
        assert prefix == "worker-role-" and dir == "/tmp"
        return original_directory(prefix=prefix, dir=tmp_path)
    def observe_native_load(context, certificate, key):
        paths = [Path(certificate), Path(key)]
        observed.append({"paths": paths,
            "modes": [stat.S_IMODE(p.stat().st_mode) for p in paths],
            "hashes": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]})
        return original_load(context, certificate, key)
    # Route only the private temporary directory; delegate to the actual native
    # SSL loader and actual TemporaryDirectory. No listener/peer is constructed.
    monkeypatch.setattr(worker_broker, "tempfile", SimpleNamespace(TemporaryDirectory=own_directory))
    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", observe_native_load)
    previous_mask = os.umask(0)
    try:
        context = worker_broker.server_context(value, worker_broker.LaneRole(role_name))
    finally:
        os.umask(previous_mask)
    assert type(context) is ssl.SSLContext and context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version == context.maximum_version == ssl.TLSVersion.TLSv1_3
    assert len(observed) == 1 and observed[0]["modes"] == [0o600, 0o600]
    assert observed[0]["hashes"] == [hashlib.sha256(material[k].encode()).hexdigest()
        for k in ("certificate", "key")]
    assert all(not p.exists() and not p.parent.exists() for p in observed[0]["paths"])


@pytest.mark.asyncio
@pytest.mark.parametrize("preexisting", [None, "incarnation", "profile"])
async def test_browser_main_shared_identity_files_survive_private_umask_without_overwrite(
    monkeypatch, tmp_path, canonical_private_worker_programs, private_file_bootstrap_data, preexisting,
):
    import asyncio
    from contextlib import asynccontextmanager
    import os
    import stat
    from types import SimpleNamespace
    from kestrel_sovereign._async_ownership import OwnedTaskAdmission, await_owned_task, create_owned_task

    program = canonical_private_worker_programs["two_domain_worker"]
    wire = canonical_private_worker_programs["wire"]
    private, shared = tmp_path / "private", tmp_path / "shared"
    shared.mkdir(mode=0o700)
    if preexisting is not None:
        (shared / preexisting).write_text("original identity file")
    monkeypatch.setattr(program, "PRIVATE", str(private))
    monkeypatch.setattr(program, "IPC", str(shared))
    monkeypatch.setattr(program, "process_evidence", lambda: {"SCRIPTED_pre_engine_only": True})
    signal_registrations = []
    scripted_loop = SimpleNamespace(add_signal_handler=lambda *args: signal_registrations.append(args))
    monkeypatch.setattr(program, "asyncio", SimpleNamespace(get_running_loop=lambda: scripted_loop,
        Event=asyncio.Event, timeout=asyncio.timeout, sleep=asyncio.sleep))
    # Unit-only role inputs; actual UID/GID/mount/Host authority is not claimed.
    monkeypatch.setattr(program, "os", SimpleNamespace(getuid=lambda: 10002,
        getgid=lambda: 10003, umask=os.umask, open=os.open, fchmod=os.fchmod))
    class BeforeBrowserEngine(Exception):
        pass
    stopped = BeforeBrowserEngine("test stops before any browser engine")
    reached_engine_boundary = []
    @asynccontextmanager
    async def stop_before_engine():
        reached_engine_boundary.append(True)
        raise stopped
        yield  # pragma: no cover — keeps the actual async-context shape.
    monkeypatch.setattr(program, "original_playwright", stop_before_engine)
    admission = OwnedTaskAdmission()
    previous_mask = os.umask(0o077)
    task = None
    try:
        task = create_owned_task(program.main(ScriptedProcessLifetime()), admission=admission,
            retirement_resource=private, retain_originals=True, name="unit-browser-private-file-producer")
        async with asyncio.timeout(5):
            while not (private / "incarnation").exists():
                await asyncio.sleep(0.001)
            incarnation = (private / "incarnation").read_text()
            value = dict(private_file_bootstrap_data["browser"], incarnation=incarnation)
            (private / "bootstrap.json").write_bytes(wire.encode(value))
            outcome = await await_owned_task(task)
    finally:
        try:
            if task is not None:
                if not task.done():
                    task.cancel()
                await await_owned_task(task)  # Actual original join, not Task.done ACK.
        finally:
            os.umask(previous_mask)
    if outcome.cancellation is not None:
        raise outcome.cancellation
    if preexisting is not None:
        if not isinstance(outcome.error, FileExistsError):
            if outcome.error is not None:
                raise outcome.error
            pytest.fail("original exclusive shared-file open did not refuse")
        assert not reached_engine_boundary
        assert (shared / preexisting).read_text() == "original identity file"
    else:
        if outcome.error is not stopped:
            if outcome.error is not None:
                raise outcome.error
            pytest.fail("original browser engine boundary was not reached")
        assert reached_engine_boundary == [True]
        assert len(signal_registrations) == 2
        assert (shared / "incarnation").read_text() == incarnation
        assert (shared / "profile").read_text() == value["profile_sha256"]
        assert all(stat.S_IMODE((shared / name).stat().st_mode) == 0o640
            for name in ("incarnation", "profile"))



@pytest.mark.parametrize("caller_umask", [0o000, 0o077], ids=["0000", "0077"])
def test_host_tls_material_private_files_are_loaded_and_retired(monkeypatch, caller_umask):
    """Real host TLS material/file lifecycle only; no listener, worker or Host admission."""
    from contextlib import contextmanager
    from datetime import UTC, datetime, timedelta
    import hashlib
    import os
    import ssl
    import stat
    import tempfile
    from types import SimpleNamespace
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.x509.oid import ExtendedKeyUsageOID
    from kestrel_sovereign.visual_computer import local_worker

    original_directory = tempfile.TemporaryDirectory
    original_load = ssl.SSLContext.load_cert_chain
    observations, observation_errors = [], []

    def public_key_hash(key):
        return hashlib.sha256(key.public_bytes(serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo)).hexdigest()

    @contextmanager
    def observe_directory(*args, **kwargs):
        assert args == () and kwargs == {"prefix": "original-worker-client-"}
        row = {}
        observations.append(row)
        try:
            with original_directory(*args, **kwargs) as directory:
                row["directory"] = Path(directory)
                paths = {name: Path(directory) / name for name in ("certificate", "key")}
                try:
                    yield directory
                finally:
                    # Descriptive only. A pre-write failure keeps its original
                    # exception; missing files here are not a teardown assertion.
                    try:
                        row["present"] = {name: path.is_file() for name, path in paths.items()}
                        row["modes"] = {name: stat.S_IMODE(path.stat().st_mode)
                            for name, path in paths.items() if path.is_file()}
                        if all(row["present"].values()):
                            row["certificate"] = x509.load_pem_x509_certificate(paths["certificate"].read_bytes())
                            key = serialization.load_pem_private_key(paths["key"].read_bytes(), password=None)
                            row["key_public_sha256"] = public_key_hash(key.public_key())
                    except BaseException as error:
                        observation_errors.append(error)
        finally:
            # Runs after the SAME original TemporaryDirectory.__exit__, even
            # when the original function raises before its first file write.
            try:
                directory = row.get("directory")
                row["original_directory_removed"] = directory is not None and not directory.exists()
            except BaseException as error:
                observation_errors.append(error)

    monkeypatch.setattr(local_worker, "tempfile", SimpleNamespace(TemporaryDirectory=observe_directory))
    expiry = datetime.now(UTC) + timedelta(minutes=5)
    primary = None
    previous_mask = os.umask(caller_umask)
    try:
        result = local_worker._tls_material(expiry)
    except BaseException as error:
        primary = error
    finally:
        os.umask(previous_mask)
    try:
        assert observations and all(row["original_directory_removed"] for row in observations)
        assert not observation_errors
        assert ssl.SSLContext.load_cert_chain is original_load
    except BaseException as audit_error:
        if primary is not None:
            primary.add_note("TLS filesystem observation also failed: " + type(audit_error).__name__)
            raise primary
        raise
    if primary is not None:
        raise primary  # Expected-success test: genuine pre-fix TypeError remains 1F.

    ca_pem, client_sha256, roles, contexts = result
    assert set(roles) == set(contexts) == {"normal", "priority"}
    assert len(observations) == 2
    ca = x509.load_pem_x509_certificate(ca_pem.encode())
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is True
    assert ca.not_valid_after_utc == expiry.replace(microsecond=0)
    ca_der = ca.public_bytes(serialization.Encoding.DER)
    for row in observations:
        assert row["present"] == {"certificate": True, "key": True}
        assert row["modes"] == {"certificate": 0o600, "key": 0o600}
        client = row["certificate"]
        client.verify_directly_issued_by(ca)
        assert hashlib.sha256(client.public_bytes(serialization.Encoding.DER)).hexdigest() == client_sha256
        assert public_key_hash(client.public_key()) == row["key_public_sha256"]
        assert ExtendedKeyUsageOID.CLIENT_AUTH in client.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    for role in ("normal", "priority"):
        material, context = roles[role], contexts[role]
        assert set(material) == {"certificate", "key", "sha256"}
        certificate = x509.load_pem_x509_certificate(material["certificate"].encode())
        certificate.verify_directly_issued_by(ca)
        role_key = serialization.load_pem_private_key(material["key"].encode(), password=None)
        assert public_key_hash(role_key.public_key()) == public_key_hash(certificate.public_key())
        assert hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest() == material["sha256"]
        assert ExtendedKeyUsageOID.SERVER_AUTH in certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        assert certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName) == [role + ".worker.kestrel.invalid"]
        assert certificate.not_valid_after_utc == expiry.replace(microsecond=0)
        assert type(context) is ssl.SSLContext and context.protocol == ssl.PROTOCOL_TLS_CLIENT
        assert context.minimum_version == context.maximum_version == ssl.TLSVersion.TLSv1_3
        assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname is True
        assert context.get_ca_certs(binary_form=True) == [ca_der]
    assert contexts["normal"] is not contexts["priority"]
    assert contexts["normal"].options == contexts["priority"].options
