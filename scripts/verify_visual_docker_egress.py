#!/usr/bin/env python3
"""Real browser HTTP acceptance against two exclusively owned synthetic servers.

The test exception is an exact labelled Docker container and its inspected local
port. Normal adapter configuration cannot enable it. No external site is used.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
import secrets
from uuid import uuid4

from kestrel_sdk.computer_use import ActionRequest, Click, ControlRequest, EnvironmentRequest, Navigate, SessionRequest, TypeText
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope, ResourceUnsupported
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, DockerEnvironmentProvider, PROVIDER
from kestrel_sovereign.visual_computer.docker_egress import DockerEgressPolicy, OwnedTestServer, OWNED_SERVER_LABEL
from scripts.verify_visual_docker_adapters import run


# Static synthetic content only: URLs/requests never pass through application
# logs or audit output. The request journal stays in this owned container tmpfs.
SERVER = r"""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,signal,socket,sys,threading
from pathlib import Path
signal.alarm(180)
other=sys.argv[1]
lock=threading.Lock()
state={'paths':[],'slow_started':0,'slow_closed':0}
def persist():
 Path('/tmp/egress-state.json').write_text(json.dumps(state))
persist()
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  with lock:
   state['paths'].append(self.path);persist()
  if self.path=='/slow':
   with lock: state['slow_started']+=1;persist()
   self.connection.settimeout(40)
   try: closed=self.connection.recv(1)==b''
   except (OSError,TimeoutError): closed=False
   with lock: state['slow_closed']+=int(closed);persist()
   return
  if self.path=='/redirect':
   self.send_response(302);self.send_header('Location','/editor');self.end_headers();return
  if self.path=='/escape':
   self.send_response(302);self.send_header('Location',other+'/forbidden-redirect');self.end_headers();return
  body=b'OK';kind='text/plain'
  if self.path=='/editor':
   kind='text/html'
   body=('''<!doctype html><title>Governed synthetic editor</title>
<style>body{margin:0;font:24px sans-serif}#draft{height:70px;background:#ddf}button{height:60px;width:180px}</style>
<div id="draft" contenteditable="true">Governed browser draft </div><button onclick="fetch('/slow').catch(()=>{})">Save slowly</button><div id="channels"></div>
<script src="/subresource.js"></script><script>
let result=[];
fetch('''+json.dumps(other+'/forbidden-fetch')+''').catch(()=>{});
fetch('http://169.254.169.254/latest/meta-data/').catch(()=>{});
fetch('/post',{method:'POST',body:'synthetic'}).catch(()=>{});
fetch('/credentialed',{headers:{Authorization:'Synthetic fixture'}}).catch(()=>{});
try {new WebSocket('''+json.dumps(other.replace('http:','ws:')+'/socket')+''')}catch(e){result.push('ws-refused')}
try {new RTCPeerConnection();result.push('rtc-UNEXPECTED')}catch(e){result.push('rtc-refused')}
if(navigator.serviceWorker)navigator.serviceWorker.register('/service-worker.js').catch(()=>{});
document.getElementById('channels').textContent=result.join(',');
setTimeout(()=>fetch('/background').catch(()=>{}),5000);
</script>''').encode()
  elif self.path=='/subresource.js':
   kind='application/javascript';body=b'document.title="Subresource accepted";'
  elif self.path=='/service-worker.js':
   kind='application/javascript';body=b'self.addEventListener("fetch",()=>{});'
  self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.end_headers()
  self.wfile.write(body)
ThreadingHTTPServer(('0.0.0.0',8080),Handler).serve_forever()
"""


class OriginalCall:
    """Synthetic authority fixture; production authority belongs to host runtime."""
    def __init__(self):
        self.active = None
        self.tasks = set()

    @asynccontextmanager
    async def guard(self, session, action_id):
        binding = self.active
        if binding is None or binding[:3] != (session.session_id, session.control_epoch, action_id):
            raise ResourceConflict("synthetic_original_call_unavailable")
        task = asyncio.current_task()
        if task is binding[3]:
            raise AssertionError("origin task cannot borrow original authority")
        def current():
            if self.active is not binding or datetime.now(UTC) >= binding[4]:
                raise ResourceConflict("synthetic_original_call_revoked")
        self.tasks.add(task)
        try:
            current(); yield current; current()
        finally:
            self.tasks.discard(task)

    async def invoke(self, driver, session, request):
        if self.active is not None:
            raise AssertionError("overlapping synthetic host action")
        self.active = (session.session_id, session.control_epoch, request.action_id, asyncio.current_task(), request.not_after)
        try:
            return await driver.act(session, request)
        finally:
            self.active = None
            assert not self.tasks


def owned_server_arguments(image_id, label, program, *arguments):
    """One bounded resource profile for both controlled HTTP/TLS proofs."""
    return ["run", "--detach", "--label", OWNED_SERVER_LABEL+"="+label,
        "--name", "kestrel-http-proof-"+label, "--network", "bridge", "--publish", "127.0.0.1::8080",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=4m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true", "--memory", "128m", "--cpus", ".5", "--pids-limit", "32",
        "--log-driver", "none", "--entrypoint", "python", image_id, "-u", "-c", program, *arguments]


async def scenario(boundary, image, identity, output):
    servers, leases = [], []
    manifest = {"server_ids": [], "leases": [], "cleanup_confirmed": False}
    def record():
        (output / "http-custody.json").write_text(json.dumps(manifest, indent=2)+"\n")
    record()
    loop = asyncio.get_running_loop()
    resolver = loop.getaddrinfo
    async def no_runtime_dns(*args,**kwargs):
        raise AssertionError("pinned transport attempted runtime libc DNS")
    loop.getaddrinfo = no_runtime_dns
    async def server(other):
        label = identity + "-" + str(len(servers))
        cid = (await boundary.invoke(owned_server_arguments(image["Id"],label,SERVER,other))).decode().strip()
        servers.append((cid, label)); manifest["server_ids"].append(cid); record()
        inspected = await boundary.inspect(cid)
        port = int(inspected["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"])
        endpoint = OwnedTestServer(boundary, cid, label, port)
        await endpoint.verify()
        # Startup readiness is container-local; no extra host network endpoint.
        await boundary.invoke(["exec", cid, "python", "-c",
            "import socket,time\nfor _ in range(100):\n try:\n  s=socket.create_connection(('127.0.0.1',8080),.1);s.close();break\n except OSError:time.sleep(.03)\nelse:raise SystemExit(1)"])
        return endpoint
    async def stats(endpoint):
        return json.loads(await boundary.invoke(["exec", endpoint.container_id, "python", "-c",
            "from pathlib import Path;print(Path('/tmp/egress-state.json').read_text())"]))
    try:
        forbidden = await server("http://127.0.0.1:1")
        allowed = await server("http://127.0.0.1:"+str(forbidden.port))
        base = "http://127.0.0.1:"+str(allowed.port)
        policy = DockerEgressPolicy((base,), test_server=allowed)
        try:
            DockerEgressPolicy.from_settings({"origins": [base], "methods": ["GET", "HEAD"]})
        except ResourceUnsupported:
            pass
        else:
            raise AssertionError("normal settings admitted loopback exception")
        provider = DockerEnvironmentProvider(docker_socket=boundary.socket, image="local-worker@"+image["Id"],
            credential_key=secrets.token_bytes(32), network_policy=policy)
        authority = OriginalCall()
        driver = DockerComputerDriver(docker_socket=boundary.socket, network_policy=policy, egress_guard=authority.guard)
        async def new_session(suffix):
            now = datetime.now(UTC)
            scope = ResourceScope(tenant_id="synthetic", principal_id="http-proof")
            request = EnvironmentRequest(request_id=identity+suffix, scope=scope, provider_id=PROVIDER,
                profile_id=provider.profile_id, image=provider.image, surface="browser", location="local",
                required_capabilities=frozenset({"screenshot","navigate","click","type","takeover","files"}),
                requested_at=now, expires_at=now+timedelta(seconds=90), idle_ttl_seconds=80, ready_timeout_seconds=25,
                max_hourly_cost_usd=Decimal(0), max_total_cost_usd=Decimal(0))
            lease = await provider.acquire(request, await provider.quote(request))
            leases.append((provider,lease)); manifest["leases"].append(lease.model_dump(mode="json")); record()
            session = await driver.open(SessionRequest(session_id="http-"+identity+suffix, scope=scope,
                lease_id=lease.lease_id, surface="browser", required_capabilities=request.required_capabilities,
                expires_at=lease.expires_at), await provider.connect(scope,lease))
            assert (await provider.inspect(scope,lease)).state == "ready"
            return session,lease
        async def request(session,effect):
            observation = await driver.observe(session)
            return ActionRequest(action_id=uuid4().hex,session_id=session.session_id,
                observation_id=observation.frame.observation_id,control_epoch=session.control_epoch,
                not_after=datetime.now(UTC)+timedelta(seconds=10),action=effect)
        async def act(session,effect):
            action = await request(session,effect)
            receipt = await authority.invoke(driver,session,action)
            if receipt.outcome != "completed":
                (output/"http-failure.private.json").write_text(json.dumps({"allowed":await stats(allowed),
                    "forbidden":await stats(forbidden),"receipt":receipt.model_dump(mode="json")},indent=2)+"\n")
            assert receipt.outcome == "completed", receipt
            return action,receipt
        session,lease = await new_session("-journey")
        navigation,receipt = await act(session,Navigate(url=base+"/editor"))
        # The delayed timer has no autonomous authority after the original RPC.
        await asyncio.sleep(5.2)
        state = await stats(allowed)
        assert "/editor" in state["paths"] and "/subresource.js" in state["paths"]
        assert not any(path in state["paths"] for path in ("/background","/post","/credentialed","/service-worker.js"))
        assert (await stats(forbidden))["paths"] == []
        await act(session,Click(x=300,y=30))
        typed,typed_receipt = await act(session,TypeText(text="PINNED HTTP DRAFT"))
        assert await driver.action_status(session,typed.action_id) == typed_receipt
        pixels = (await driver.observe(session)).image.get_secret_value()
        (output/"http-browser.png").write_bytes(pixels)
        paused = await driver.control(session,ControlRequest(request_id="export-pause",session_id=session.session_id,
            expected_control_epoch=session.control_epoch,operation="pause"))
        from kestrel_sdk.workspaces import TransferLimits, WorkspaceExportRequest
        transfer = driver.workspace_transfer()
        snapshot = await transfer.prepare_export(paused,WorkspaceExportRequest(transfer_id="http-export",
            session_id=paused.session_id,lease_id=paused.lease_id,control_epoch=paused.control_epoch,
            workspace_id="browser",expires_at=paused.expires_at,limits=TransferLimits(max_files=1,max_file_bytes=128*1024,max_total_bytes=128*1024),
            paths=("document.html",),entrypoint="document.html"))
        document = (await transfer.read_export(paused,snapshot.snapshot_id,"document.html")).get_secret_value()
        (output/"http-document.html").write_bytes(document)
        channels = re.search(rb'<div id="channels">([^<]*)</div>',document).group(1)
        assert b"PINNED HTTP DRAFT" in document and b"rtc-refused" in channels and b"rtc-UNEXPECTED" not in channels
        await transfer.release_export(paused,snapshot.snapshot_id)
        session = await driver.control(paused,ControlRequest(request_id="export-resume",session_id=paused.session_id,
            expected_control_epoch=paused.control_epoch,operation="resume"))
        # Mouse dispatch completes; its resulting HTTP request remains pending.
        # Both Stop/pause and takeover must close accepted sockets before ack.
        cutovers = {}
        for count,operation in enumerate(("pause","takeover"),1):
            slow = await request(session,Click(x=40,y=100))
            pending = asyncio.create_task(authority.invoke(driver,session,slow))
            async with asyncio.timeout(8):
                while (await stats(allowed))["slow_started"] < count:
                    await asyncio.sleep(.05)
            authority.active = None  # Original caller grant revoked before Stop.
            started = asyncio.get_running_loop().time()
            taken = await driver.control(session,ControlRequest(request_id="http-"+operation,session_id=session.session_id,
                expected_control_epoch=session.control_epoch,operation=operation))
            cutover = asyncio.get_running_loop().time()-started
            cutovers[operation] = cutover
            result = await asyncio.gather(pending,return_exceptions=True)
            assert isinstance(result[0],(ResourceOutcomeUnknown,ResourceConflict,asyncio.CancelledError)), result
            assert taken.state == "paused" and taken.controller == ("human" if operation == "takeover" else "none") and cutover < 5
            assert not authority.tasks and not driver._pumps and not driver._borrowers
            assert (await stats(allowed))["slow_closed"] == count
            slow_receipt = await driver.action_status(taken,slow.action_id)
            assert slow_receipt.outcome == "unknown"
            if operation == "pause":
                session = await driver.control(taken,ControlRequest(request_id="stop-resume",session_id=taken.session_id,
                    expected_control_epoch=taken.control_epoch,operation="resume"))
        try:
            await authority.invoke(driver,session,slow)
        except ResourceConflict:
            pass
        else:
            raise AssertionError("stale epoch accepted input")
        # A separate session proves a redirect cannot bypass the exact origin.
        escape,escape_lease = await new_session("-redirect")
        escaped = await request(escape,Navigate(url=base+"/escape"))
        try:
            refused_receipt = await authority.invoke(driver,escape,escaped)
            assert refused_receipt.outcome == "unknown"
        except (ResourceNotFound,ResourceOutcomeUnknown):
            pass  # Browser retirement can close IPC before the unknown reply.
        assert (await stats(forbidden))["paths"] == []
        manifest["result"] = {"normal_loopback_profile_refused":True,"direct_navigation_and_same_origin_subresource":True,
            "no_background_post_credentials_service_worker_or_forbidden_server":True,"websocket_rtc_refused":True,
            "useful_browser_edit_and_export":True,"png_sha256":hashlib.sha256(pixels).hexdigest(),
            "priority_control_seconds":cutovers,"accepted_http_socket_closed_before_ack":True,
            "original_action_unknown":slow_receipt.model_dump(mode="json"),"stale_epoch_refused":True,
            "redirects_explicitly_unsupported":True,"forbidden_redirect_refused":True,"profile_id":policy.profile_id,"browser_network":"none"}
        manifest["result"]["runtime_libc_dns_never_called"] = True
        record()
        return manifest["result"]
    finally:
        loop.getaddrinfo = resolver
        manifest["released_leases"] = []
        for provider,lease in reversed(leases):
            released = await provider.release(lease.scope,lease)
            assert released.state == "released"
            manifest["released_leases"].append(released.model_dump(mode="json"));record()
        for cid,label in reversed(servers):
            inspected = await boundary.inspect(cid)
            if inspected["Id"] != cid or inspected["Config"].get("Labels",{}).get(OWNED_SERVER_LABEL) != label:
                raise AssertionError("refusing cleanup of replaced server")
            await boundary.invoke(["container","rm","--force",cid])
            try:
                await boundary.inspect(cid)
            except ResourceNotFound:
                pass
            else:
                raise AssertionError("owned server cleanup unconfirmed")
        manifest["cleanup_confirmed"] = True; record()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--docker-socket",required=True)
    args=parser.parse_args()
    result=asyncio.run(run(args.output.resolve(),args.docker_socket,scenario=scenario))
    print(json.dumps({"passed":"result" in result,"cleanup_confirmed":result["cleanup_confirmed"],"evidence":str(args.output)}))


if __name__ == "__main__":
    main()
