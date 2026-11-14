"""Real process lifecycle controls; IPC/TLS/browser IO is explicitly scripted.

These component tests execute the canonical entry and broker main bodies.
They prove original OS process status and waits, not a Docker worker session.
"""
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

import pytest

from kestrel_sovereign._async_ownership import await_owned_operation, raise_owned_outcome

CHILD = r'''
import asyncio, importlib.util, io, json, os, signal, sys, time, traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
root=Path(sys.argv[3]);sys.path.insert(0,str(root))
mode,role=sys.argv[1:3]
signal.alarm(12)
original_import_handlers={n:signal.getsignal(n)for n in (signal.SIGINT,signal.SIGTERM)}
from kestrel_sovereign.visual_computer import worker_entry as entry
core=Path(entry.__file__).resolve().parent
directory=(root/"docker/visual-browser" if core==root/"kestrel_sovereign/visual_computer" else core/"_worker_programs")
assert directory.is_dir()
def ready():
 print("ORIGINAL-PROCESS-WINDOW",flush=True)
 time.sleep(4 if mode in {"lane-for-supervisor","lane-expiry-seven"} else 0.3)
def lane_main():
 sys.path.insert(0,str(directory))
 spec=importlib.util.spec_from_file_location("owned_canonical_broker_program",directory/"broker_worker.py")
 program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program)
 assert Path(program.__file__).resolve()==(directory/"broker_worker.py").resolve()
 from tests.unit.visual_computer.test_two_domain_worker_profile import settings
 value=settings();value["roles"]={"priority":value["roles"]["priority"]}
 program.sys=SimpleNamespace(argv=[str(directory/"broker_worker.py"),"--lane","priority"],stdin=SimpleNamespace(buffer=io.BytesIO(json.dumps(value).encode())))
 program.os=SimpleNamespace(getuid=lambda:10001,getgid=lambda:10003,umask=lambda mask:None)
 program.require_ipc=lambda *args:SimpleNamespace(close=lambda:None)
 program.server_context=lambda *args:None
 class Lane:
  def __init__(self,*args):self.withdrawn=asyncio.Event()
  def revoke(self):self.withdrawn.set()
  async def serve(self):
   if mode=="active-main":
    print("ORIGINAL-PROCESS-WINDOW",flush=True)
    await asyncio.wait_for(self.withdrawn.wait(),3)
   else:await asyncio.sleep(0)
 program.WorkerLane=Lane
 loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
 try:
  from kestrel_sovereign.visual_computer.worker_entry import ProcessLifetime
  loop.run_until_complete(program.main(ProcessLifetime()))
  if mode in {"post-main","lane-for-supervisor","lane-expiry-seven"}:ready()
 finally:
  loop.close();asyncio.set_event_loop(None)
 if mode=="post-loop":ready()
 if mode=="terminal-mask":
  blocked=signal.pthread_sigmask(signal.SIG_BLOCK,[])
  print("ORIGINAL-TERMINAL-MASK "+json.dumps({"SIGINT":signal.SIGINT in blocked,"SIGTERM":signal.SIGTERM in blocked}),flush=True)
 if mode in {"lane-early-zero","lane-early-seven"}:print("ORIGINAL-PROCESS-WINDOW",flush=True)
 if mode in {"lane-early-seven","lane-expiry-seven"}:raise SystemExit(7)
if mode=="lookup-window":
 class Resources:
  def joinpath(self,*parts):
   assert parts[0]=="_worker_programs"and len(parts)==2
   return directory/parts[1]
 def files(package):
  assert package=="kestrel_sovereign.visual_computer"
  ready()
  return Resources()
 entry.files=files
 sys.argv=[str(Path(entry.__file__).resolve()),*({"lane":["broker","--lane","priority"]}.get(role,[role]))]
 entry.main()
elif mode in {"post-main","post-loop","active-main","terminal-mask","lane-for-supervisor","lane-early-zero","lane-early-seven","lane-expiry-seven"}:
 lane_main()
elif mode in {"supervisor-expiry","supervisor-early-zero","supervisor-early-seven","supervisor-late-seven"}:
 from kestrel_sovereign.visual_computer import worker_broker as broker
 from tests.unit.visual_computer.test_two_domain_worker_profile import settings
 real_create=asyncio.create_subprocess_exec
 originals=[]
 async def launch(*argv,**kwargs):
  child_mode={"supervisor-expiry":"lane-for-supervisor","supervisor-early-zero":"lane-early-zero","supervisor-early-seven":"lane-early-seven","supervisor-late-seven":"lane-expiry-seven"}[mode]
  child=await real_create(sys.executable,"-B","-c",CHILD_TEXT,child_mode,"broker",str(root),
   stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env=os.environ.copy())
  originals.append(child)
  return child
 async def roles_ready(children,retirement):
  assert children==originals and len(children)==2
  for child in children:
   assert await asyncio.wait_for(child.stdout.readline(),5)==b"ORIGINAL-PROCESS-WINDOW\n"
 broker.asyncio.create_subprocess_exec=launch;broker._wait_roles_ready=roles_ready
 value=settings();value["expires_at"]=(datetime.now(UTC)+timedelta(seconds=3)).isoformat()
 async def run():
  from kestrel_sovereign.execution_custody import execution_terminal_error
  error=None
  secondary=[]
  try:
   await broker.supervise(value,Path(broker.__file__).with_name("_worker_programs")/"broker_worker.py")
  except BaseException as caught:error=caught
  rows=[]
  for child in originals:
   try:
    actual=await child.wait()
    assert type(actual)is int and actual==child.returncode
    rows.append({"pid":child.pid,"native_exit":actual,"same_original_Process":True,"actual_original_wait_ack":True})
   except BaseException as caught:
    secondary.append(caught)
    rows.append({"pid":child.pid,"native_exit":None,"same_original_Process":True,"actual_original_wait_ack":False})
  try:print("ORIGINAL-CHILD-WAITS "+json.dumps(rows),flush=True)
  except BaseException as caught:secondary.append(caught)
  if error is not None or secondary:raise execution_terminal_error(error,*secondary,retain_originals=True)
 asyncio.run(run())
elif mode=="import-only":
 before=original_import_handlers
 sys.path.insert(0,str(directory))
 for name in ("broker_worker","two_domain_worker"):
  spec=importlib.util.spec_from_file_location("owned_import_only_"+name,directory/(name+".py"))
  program=importlib.util.module_from_spec(spec);spec.loader.exec_module(program)
 assert all(signal.getsignal(n)is v for n,v in before.items())
else:raise AssertionError(mode)
'''

# Supply the same helper text to its supervised actual child fixtures.
CHILD = CHILD.replace("CHILD_TEXT", repr(CHILD))


async def original_process(tmp_path, mode, role, *, signal_at_window=True):
    from kestrel_sovereign.execution_custody import execution_terminal_error
    repository = Path(__file__).resolve().parents[3]
    env = dict(os.environ, PYTHONPATH=str(repository), PYTHONDONTWRITEBYTECODE="1",
               PYTHON_DOTENV_DISABLED="1")
    process, failure, native = None, None, None
    signal_api_returned = False
    first, stdout, stderr = b"", b"", b""
    errors = []
    argv = [sys.executable, "-B", "-c", CHILD, mode, role, str(repository)]
    async def launch():
        nonlocal process
        process = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=tmp_path, env=env,
        )  # Retain the original before fallible return/publication.
        return process
    try:
        original = raise_owned_outcome(
            await await_owned_operation(launch(), retirement_resource=argv),
            operation="owned lifecycle component child launch", retain_originals=True)
        assert process is original and process.pid > 0
        if signal_at_window:
            first = await asyncio.wait_for(original.stdout.readline(), 8)
            assert first == b"ORIGINAL-PROCESS-WINDOW\n", first
            original.send_signal(signal.SIGTERM)
            signal_api_returned = True  # API return, never a physical delivery ACK.
    except BaseException as error:
        failure = error
    finally:
        if process is not None:
            try:
                stdout, stderr = raise_owned_outcome(
                    await await_owned_operation(process.communicate(), retirement_resource=process),
                    operation="owned lifecycle component streams", retain_originals=True)
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    native = raise_owned_outcome(
                        await await_owned_operation(process.wait(), retirement_resource=process),
                        operation="owned lifecycle component original native wait", retain_originals=True)
                    assert type(native) is int and native == process.returncode
                except BaseException as error:
                    errors.append(error)
    facts = {"pid": None if process is None else process.pid, "native_exit":native,
             "actual_original_wait_ack":type(native) is int,
             "requested_signal":"SIGTERM" if signal_at_window else None,
             "signal_api_returned":signal_api_returned,
             "physical_signal_delivery_ack":False,
             "mode":mode,"role":role,"stdout":(first+stdout).decode(),"stderr":stderr.decode(),
             "no_Docker_TLS_or_worker_session_claim":True}
    try:
        (tmp_path/"original-process.json").write_text(json.dumps(facts,indent=2)+"\n")
    except BaseException as error:
        errors.append(error)
    if failure is not None or errors:
        raise execution_terminal_error(failure,*errors,retain_originals=True)
    return facts


@pytest.mark.asyncio
@pytest.mark.parametrize("role",["browser","broker","lane"])
async def test_pending_withdrawal_survives_resource_lookup_and_exec(tmp_path, role):
    facts=await original_process(tmp_path,"lookup-window",role)
    assert facts["native_exit"]==0, facts


@pytest.mark.asyncio
async def test_importing_programs_does_not_install_process_handlers(tmp_path):
    facts=await original_process(tmp_path,"import-only","broker",signal_at_window=False)
    assert facts["native_exit"]==0, facts


@pytest.mark.asyncio
async def test_live_lane_withdrawal_reaches_original_async_main(tmp_path):
    facts=await original_process(tmp_path,"active-main","broker")
    assert facts["native_exit"]==0, facts


@pytest.mark.asyncio
async def test_original_terminal_signal_mask_survives_loop_close(tmp_path):
    facts=await original_process(tmp_path,"terminal-mask","broker",signal_at_window=False)
    rows=[line for line in facts["stdout"].splitlines()if line.startswith("ORIGINAL-TERMINAL-MASK ")]
    assert len(rows)==1, facts
    assert json.loads(rows[0].removeprefix("ORIGINAL-TERMINAL-MASK "))=={"SIGINT":True,"SIGTERM":True}, facts
    assert facts["native_exit"]==0, facts


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,child_code",[("supervisor-early-zero",0),
    ("supervisor-early-seven",7),("supervisor-late-seven",7)])
async def test_actual_early_and_nonzero_child_statuses_stay_failures(tmp_path, mode, child_code):
    facts=await original_process(tmp_path,mode,"broker",signal_at_window=False)
    rows=[line for line in facts["stdout"].splitlines()if line.startswith("ORIGINAL-CHILD-WAITS ")]
    assert len(rows)==1, facts
    children=json.loads(rows[0].removeprefix("ORIGINAL-CHILD-WAITS "))
    assert len(children)==2 and all(row["actual_original_wait_ack"] and
        row["native_exit"]==child_code for row in children), facts
    assert facts["native_exit"]==1, facts


@pytest.mark.asyncio
@pytest.mark.parametrize("window",["post-main","post-loop"])
async def test_lane_withdrawal_handler_survives_main_and_loop_exit(tmp_path, window):
    facts=await original_process(tmp_path,window,"broker")
    assert facts["native_exit"]==0, facts


@pytest.mark.asyncio
async def test_natural_expiry_joins_real_original_children_without_signal_exit(tmp_path):
    facts=await original_process(tmp_path,"supervisor-expiry","broker",signal_at_window=False)
    rows=[line for line in facts["stdout"].splitlines()if line.startswith("ORIGINAL-CHILD-WAITS ")]
    assert len(rows)==1, facts
    children=json.loads(rows[0].removeprefix("ORIGINAL-CHILD-WAITS "))
    assert len(children)==2 and len({row["pid"]for row in children})==2
    assert all(row["same_original_Process"]and row["native_exit"]==0 for row in children), facts
    assert facts["native_exit"]==0, facts
