#!/usr/bin/env python3
"""Owned HTTPS/Chromium proof; no public TLS interoperability claim or trust option.

The temporary CA exists only inside this verifier's private SSLContext. Normal
hostname and chain verification stay on. Production gateway code is unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import secrets
import ssl
from unittest.mock import patch
from uuid import uuid4

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from kestrel_sdk.computer_use import ActionRequest, Click, ControlRequest, EnvironmentRequest, Navigate, SessionRequest, TypeText
from kestrel_sdk.resources import ResourceConflict, ResourceNotFound, ResourceOutcomeUnknown, ResourceScope
from kestrel_sovereign.visual_computer.docker import DockerComputerDriver, DockerEnvironmentProvider, PROVIDER
from kestrel_sovereign.visual_computer import docker_egress
from kestrel_sovereign.visual_computer.docker_egress import DockerEgressPolicy, OwnedTestServer, OWNED_SERVER_LABEL
from scripts.verify_visual_docker_adapters import run
from scripts.verify_visual_docker_egress import OriginalCall, owned_server_arguments


SERVER = r'''
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import base64,json,signal,socket,ssl,threading,time
from pathlib import Path
signal.alarm(180)
bootstrap=Path('/tmp/tls-fixture.json')
for _ in range(400):
 if bootstrap.exists():break
 time.sleep(.05)
else:raise SystemExit(1)
data=json.loads(bootstrap.read_text());bootstrap.unlink()
for name in ('cert','key'):
 p=Path('/tmp/'+name+'.pem');p.write_bytes(base64.b64decode(data[name]));p.chmod(0o600)
lock=threading.RLock()
state={'sni':[],'requests':[],'active':0,'handshake_refused':0,'slow_started':False,'slow_closed':False}
def persist():Path('/tmp/tls-state.json').write_text(json.dumps(state))
persist()
context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('/tmp/cert.pem','/tmp/key.pem')
def sni(ssl_socket,name,ctx):
 with lock:state['sni'].append(name);persist()
context.set_servername_callback(sni)
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  with lock:state['requests'].append({'path':self.path,'host':self.headers.get('Host')});persist()
  if self.path=='/slow':
   with lock:state['slow_started']=True;persist()
   self.connection.settimeout(30)
   try:closed=self.connection.recv(1)==b''
   except (OSError,TimeoutError):closed=False
   with lock:state['slow_closed']=closed;persist()
   return
  body=b'<!doctype html><title>Owned HTTPS editor</title><style>body{margin:0;font:24px sans-serif}div{height:70px;background:#ddf}button{width:180px;height:60px}</style><div contenteditable=true>Trusted TLS draft </div><button onclick="fetch(\'/slow\').catch(()=>{})">Save slowly</button>'
  self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
class Server(ThreadingHTTPServer):
 def get_request(self):
  raw,address=super().get_request()
  try:connection=context.wrap_socket(raw,server_side=True)
  except (ssl.SSLError,OSError):
   raw.close()
   with lock:state['handshake_refused']+=1;persist()
   raise
  with lock:state['active']+=1;persist()
  return connection,address
 def close_request(self,request):
  super().close_request(request)
  with lock:state['active']-=1;persist()
Server(('0.0.0.0',8080),Handler).serve_forever()
'''


def certificate_authority():
    key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"Kestrel owned TLS proof "+uuid4().hex)])
    now = datetime.now(UTC)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True,path_length=0),critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),critical=False)
        .add_extension(x509.KeyUsage(digital_signature=False,content_commitment=False,key_encipherment=False,
            data_encipherment=False,key_agreement=False,key_cert_sign=True,crl_sign=True,encipher_only=None,decipher_only=None),critical=True)
        .sign(key,hashes.SHA256()))
    return key,certificate


def leaf_certificate(authority,hostname):
    root_key,root_certificate = authority
    key = rsa.generate_private_key(public_exponent=65537,key_size=2048)
    now = datetime.now(UTC)
    certificate = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,hostname)]))
        .issuer_name(root_certificate.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()),critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()),critical=False)
        .add_extension(x509.KeyUsage(digital_signature=True,content_commitment=False,key_encipherment=True,
            data_encipherment=False,key_agreement=False,key_cert_sign=False,crl_sign=False,encipher_only=None,decipher_only=None),critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]),critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),critical=False)
        .sign(root_key,hashes.SHA256()))
    return certificate.public_bytes(serialization.Encoding.PEM),key.private_bytes(serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,serialization.NoEncryption())


async def scenario(boundary,image,identity,output):
    manifest = {"servers":[],"leases":[],"connections":[],"cleanup_confirmed":False}
    servers,leases,drivers = [],[],[]
    def record():(output/"tls-custody.json").write_text(json.dumps(manifest,indent=2)+"\n")
    record()
    authority = certificate_authority()
    root_pem = authority[1].public_bytes(serialization.Encoding.PEM)
    (output/"temporary-test-root.pem").write_bytes(root_pem)
    client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.verify_flags = ssl.create_default_context().verify_flags
    client_context.load_verify_locations(cadata=root_pem.decode())
    assert client_context.check_hostname and client_context.verify_mode == ssl.CERT_REQUIRED
    native_context = ssl.create_default_context
    native_connect = asyncio.open_connection
    loop = asyncio.get_running_loop()
    resolver = loop.getaddrinfo
    async def no_dns(*args,**kwargs):raise AssertionError("TLS pin attempted runtime DNS")
    loop.getaddrinfo = no_dns
    async def connect(address,port,**options):
        context = options["ssl"]
        item = {"address":address,"port":port,"server_hostname":options.get("server_hostname"),
            "check_hostname":context.check_hostname,"verify_mode":context.verify_mode.name,"verify_flags":int(context.verify_flags)}
        manifest["connections"].append(item);record()
        try:
            reader,writer = await native_connect(address,port,**options)
        except ssl.SSLCertVerificationError as error:
            item.update(error_class="SSLCertVerificationError",verify_code=error.verify_code);record()
            raise
        item["tls_version"] = writer.get_extra_info("ssl_object").version();record()
        return reader,writer
    async def server(suffix,wrong_hostname=False):
        label = identity+suffix
        entry = {"name":"kestrel-http-proof-"+label,"ownership":label,"container_id":None}
        servers.append(entry);manifest["servers"].append(entry);record()
        cert,key = leaf_certificate(authority,("wrong-" if wrong_hostname else "")+label+".invalid")
        (output/(label+".pem")).write_bytes(cert)
        cid = (await boundary.invoke(owned_server_arguments(image["Id"],label,SERVER))).decode().strip()
        entry["container_id"] = cid;record()
        await boundary.invoke(["exec","-i",cid,"python","-c",
            "import sys;from pathlib import Path;p=Path('/tmp/tls-fixture.json');p.write_bytes(sys.stdin.buffer.read());p.chmod(0o600)"],
            data=json.dumps({"cert":base64.b64encode(cert).decode(),"key":base64.b64encode(key).decode()}).encode())
        config = await boundary.inspect(cid)
        port = int(config["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"])
        endpoint = OwnedTestServer(boundary,cid,label,port,tls=True)
        await endpoint.verify()
        await boundary.invoke(["exec",cid,"python","-c",
            "import time;from pathlib import Path\nfor _ in range(200):\n if Path('/tmp/tls-state.json').exists():break\n time.sleep(.03)\nelse:raise SystemExit(1)"])
        return endpoint
    async def stats(endpoint):
        return json.loads(await boundary.invoke(["exec",endpoint.container_id,"python","-c",
            "from pathlib import Path;print(Path('/tmp/tls-state.json').read_text())"]))
    async def session(endpoint,suffix):
        policy = DockerEgressPolicy((endpoint.origin,),test_server=endpoint,address_pins=((endpoint.origin,"127.0.0.1"),))
        provider = DockerEnvironmentProvider(docker_socket=boundary.socket,image="local-worker@"+image["Id"],
            credential_key=secrets.token_bytes(32),network_policy=policy)
        caller = OriginalCall()
        driver = DockerComputerDriver(docker_socket=boundary.socket,network_policy=policy,egress_guard=caller.guard)
        drivers.append((driver,caller))
        now = datetime.now(UTC)
        request = EnvironmentRequest(request_id=identity+suffix,scope=ResourceScope(tenant_id="synthetic",principal_id="tls-proof"),
            provider_id=PROVIDER,profile_id=provider.profile_id,image=provider.image,surface="browser",location="local",
            required_capabilities=frozenset({"screenshot","navigate","click","type","takeover","files"}),
            requested_at=now,expires_at=now+timedelta(seconds=90),idle_ttl_seconds=80,ready_timeout_seconds=25,
            max_hourly_cost_usd=Decimal(0),max_total_cost_usd=Decimal(0))
        lease = await provider.acquire(request,await provider.quote(request))
        leases.append((provider,lease));manifest["leases"].append(lease.model_dump(mode="json"));record()
        value = await driver.open(SessionRequest(session_id="tls-"+identity+suffix,scope=request.scope,lease_id=lease.lease_id,
            surface="browser",required_capabilities=request.required_capabilities,expires_at=lease.expires_at),await provider.connect(request.scope,lease))
        return driver,caller,value
    async def request(driver,value,effect):
        frame = (await driver.observe(value)).frame
        return ActionRequest(action_id=uuid4().hex,session_id=value.session_id,observation_id=frame.observation_id,
            control_epoch=value.control_epoch,not_after=datetime.now(UTC)+timedelta(seconds=10),action=effect)
    try:
        valid = await server("-valid")
        wrong = await server("-wrong",wrong_hostname=True)
        with patch.object(asyncio,"open_connection",connect):
            # Normal platform trust must refuse the temporary CA first.
            failures = []
            for label,endpoint,context in (("untrusted_chain",valid,native_context()),("wrong_hostname",wrong,client_context)):
                driver,caller,value = await session(endpoint,"-"+label)
                action = await request(driver,value,Navigate(url=endpoint.origin+"/editor"))
                with patch.object(docker_egress.ssl,"create_default_context",return_value=context):
                    try:
                        receipt = await caller.invoke(driver,value,action)
                        assert receipt.outcome == "unknown"
                    except (ResourceNotFound,ResourceOutcomeUnknown):pass
                state = await stats(endpoint)
                assert state["requests"] == [] and state["active"] == 0
                assert state["sni"] and all(name == endpoint.ownership+".invalid" for name in state["sni"])
                error = next(item for item in reversed(manifest["connections"]) if item["port"] == endpoint.port)
                assert error["error_class"] == "SSLCertVerificationError" and error["check_hostname"] and error["verify_mode"] == "CERT_REQUIRED"
                failures.append({"fixture":label,"verify_code":error["verify_code"],"original_action_id":action.action_id,
                    "no_http_request":True,"peer_sni":state["sni"],"server_active_sockets":state["active"]})
                provider,lease = leases[-1]
                assert (await provider.release(lease.scope,lease)).state == "released"
            assert failures[0]["verify_code"] != failures[1]["verify_code"]
            driver,caller,value = await session(valid,"-trusted")
            async def act(effect):
                action = await request(driver,value,effect)
                receipt = await caller.invoke(driver,value,action)
                assert receipt.outcome == "completed",receipt
                return receipt
            with patch.object(docker_egress.ssl,"create_default_context",return_value=client_context):
                await act(Navigate(url=valid.origin+"/editor"))
                await act(Click(x=350,y=30));await act(TypeText(text="OWNED TLS VERIFIED"))
                pixels = (await driver.observe(value)).image.get_secret_value()
                (output/"tls-browser.png").write_bytes(pixels)
                slow = await request(driver,value,Click(x=40,y=100))
                pending = asyncio.create_task(caller.invoke(driver,value,slow))
                async with asyncio.timeout(8):
                    while not (await stats(valid))["slow_started"]:await asyncio.sleep(.04)
                caller.active = None
                started = loop.time()
                stopped = await driver.control(value,ControlRequest(request_id="tls-stop",session_id=value.session_id,
                    expected_control_epoch=value.control_epoch,operation="pause"))
                seconds = loop.time()-started
                result = await asyncio.gather(pending,return_exceptions=True)
                assert isinstance(result[0],(ResourceOutcomeUnknown,ResourceConflict,asyncio.CancelledError))
                state = await stats(valid)
                assert state["slow_closed"] and state["active"] == 0 and seconds < 5
                assert not caller.tasks and not driver._pumps and not driver._borrowers
                receipt = await driver.action_status(stopped,slow.action_id)
                assert receipt.outcome == "unknown"
                from kestrel_sdk.workspaces import TransferLimits,WorkspaceExportRequest
                transfer = driver.workspace_transfer()
                snapshot = await transfer.prepare_export(stopped,WorkspaceExportRequest(transfer_id="tls-export",
                    session_id=stopped.session_id,lease_id=stopped.lease_id,control_epoch=stopped.control_epoch,
                    workspace_id="browser",expires_at=stopped.expires_at,limits=TransferLimits(max_files=1,max_file_bytes=131072,max_total_bytes=131072),
                    paths=("document.html",),entrypoint="document.html"))
                html = (await transfer.read_export(stopped,snapshot.snapshot_id,"document.html")).get_secret_value()
                assert b"OWNED TLS VERIFIED" in html
                (output/"tls-document.html").write_bytes(html)
                await transfer.release_export(stopped,snapshot.snapshot_id)
            expected_host = valid.ownership+".invalid:"+str(valid.port)
            assert state["requests"] and all(item["host"] == expected_host for item in state["requests"])
            assert all(name == valid.ownership+".invalid" for name in state["sni"])
            manifest["accepted_server_evidence"] = state
            assert all(item["address"] == "127.0.0.1" and item["check_hostname"] and item["verify_mode"] == "CERT_REQUIRED" for item in manifest["connections"])
            manifest["result"] = {"owned_https_only":True,"public_https_interoperability":False,"runtime_dns_never_called":True,
                "literal_pin_original_host_and_sni":True,"certificate_failures":failures,"temporary_ca_sha256":hashlib.sha256(root_pem).hexdigest(),
                "browser_tls_edit_and_export":True,"png_sha256":hashlib.sha256(pixels).hexdigest(),"stop_seconds":seconds,
                "tls_socket_closed_and_child_borrowers_joined_before_ack":True,"original_receipt":receipt.model_dump(mode="json")}
            record();return manifest["result"]
    finally:
        loop.getaddrinfo = resolver
        manifest["released_leases"] = []
        for provider,lease in reversed(leases):
            released = await provider.release(lease.scope,lease)
            manifest["released_leases"].append(released.model_dump(mode="json"));record()
        for entry in reversed(servers):
            try:config = await boundary.inspect(entry["container_id"] or entry["name"])
            except ResourceNotFound:continue
            if config["Config"].get("Labels",{}).get(OWNED_SERVER_LABEL) != entry["ownership"] or (entry["container_id"] and config["Id"] != entry["container_id"]):
                raise AssertionError("refusing unowned TLS fixture cleanup")
            await boundary.invoke(["container","rm","--force",config["Id"]])
            try:await boundary.inspect(config["Id"])
            except ResourceNotFound:pass
            else:raise AssertionError("owned TLS fixture cleanup unconfirmed")
        assert all(not driver._borrowers and not driver._pumps and not caller.tasks for driver,caller in drivers)
        manifest["cleanup_confirmed"] = True;record()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--docker-socket",required=True)
    args=parser.parse_args()
    result=asyncio.run(run(args.output.resolve(),args.docker_socket,scenario=scenario))
    print(json.dumps({"passed":"result" in result,"cleanup_confirmed":result["cleanup_confirmed"],"evidence":str(args.output)}))


if __name__ == "__main__":main()
