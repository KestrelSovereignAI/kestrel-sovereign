"""Owned dual-stack TCP fixtures for local Docker verification only.

This is not a production transport or permission source. Every data-plane
target comes from the exact labelled server's inspected network attachment.
An independently contained positive control proves both listeners are live.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import secrets
from pathlib import Path

from kestrel_sdk.resources import ResourceNotFound
from kestrel_sovereign.visual_computer.docker import DockerCommandError

LABEL = "io.kestrel.visual-network-proof"
PORT = 18080
_ID = re.compile(r"[a-f0-9]{64}")
SERVER = r'''
import json,signal,socket,sys,threading
from pathlib import Path
signal.alarm(180)
nonce=sys.argv[1].encode()
counts={'ipv4':0,'ipv6':0}
lock=threading.Lock()
def persist():
 Path('/tmp/peer-counts.json').write_text(json.dumps(counts))
def serve(listener,name):
 while True:
  connection,_=listener.accept()
  with connection:
   with lock: counts[name]+=1;persist()
   connection.settimeout(1)
   try:connection.sendall(nonce)
   except OSError:pass
listeners=[]
for family,host,name in [(socket.AF_INET,'0.0.0.0','ipv4'),(socket.AF_INET6,'::','ipv6')]:
 listener=socket.socket(family)
 if family==socket.AF_INET6:listener.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,1)
 listener.bind((host,18080));listener.listen(8);listeners.append(listener)
 threading.Thread(target=serve,args=(listener,name),daemon=True).start()
persist();Path('/tmp/peer-ready').touch()
signal.pause()
'''
CONTROL = r'''
import json,signal,socket,sys
signal.alarm(15)
fixture=json.loads(sys.argv[1]);nonce=sys.argv[2].encode();results=[]
for target in fixture['targets']:
 family=socket.AF_INET if target['family']=='ipv4' else socket.AF_INET6
 with socket.socket(family) as connection:
  connection.settimeout(2);connection.connect((target['host'],target['port']))
  actual=b''
  while len(actual)<len(nonce):
   chunk=connection.recv(len(nonce)-len(actual))
   if not chunk:break
   actual+=chunk
  if actual!=nonce:raise RuntimeError('owned peer did not return its exact nonce')
 results.append(target)
print(json.dumps({'reachable_owned_targets':results}))
'''


class OwnedNetworkPeers:
    def __init__(self, invoke, image, identity, output, *, image_label):
        self.invoke, self.image, self.identity = invoke, image, identity
        self.image_label = image_label
        self.output = Path(output)
        self.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.names = {"server":"kestrel-net-server-"+identity, "control":"kestrel-net-control-"+identity}
        self.network_name = "kestrel-net-proof-"+identity
        self.nonce = secrets.token_hex(32)
        self.state = {"proof_id":identity, "image_id":image, "network_name":self.network_name,
            "network_id":None, "containers":{}, "fixture":None, "cleanup_confirmed":False}
        self.record()

    def record(self):
        (self.output/"custody.json").write_text(json.dumps(self.state, indent=2)+"\n")

    async def _inspect(self, kind, target):
        try:
            raw = await self.invoke([kind, "inspect", target])
        except DockerCommandError as error:
            # The SDK boundary handles container/image absence. Network
            # fixtures are verifier-only; accept only this exact daemon reply,
            # including the CLI's optional empty JSON array before stderr.
            name = re.escape(target.encode())
            absent = (rb"(?:\[\]\s*)?(?:Error: |Error response from daemon: )No such (?:object|image|container|network): "
                + name + rb"\s*" if kind != "network" else
                rb"(?:\[\]\s*)?Error response from daemon: network " + name + rb" not found\s*")
            if re.fullmatch(absent, error.diagnostics.get_secret_value()):
                raise ResourceNotFound("owned_fixture_absent") from None
            raise
        value = json.loads(raw)
        if not isinstance(value, list) or len(value) != 1:
            raise RuntimeError("owned fixture inspection is incomplete")
        return value[0]

    def _owned_network(self, network):
        if (not _ID.fullmatch(network.get("Id", ""))
                or network.get("Labels", {}).get(LABEL) != self.identity
                or network.get("Name") != self.network_name
                or (self.state["network_id"] is not None and network["Id"] != self.state["network_id"])):
            raise RuntimeError("owned fixture network identity differs")
        return network

    def _network(self, network):
        self._owned_network(network)
        if (network.get("Driver") != "bridge" or network.get("Internal") is not True
                or network.get("EnableIPv6") is not True):
            raise RuntimeError("owned dual-stack fixture network differs")
        return network

    def _owned_container(self, record, name):
        if (not _ID.fullmatch(record.get("Id", "")) or record.get("Image") != self.image
                or record["Config"].get("Labels", {}).get(LABEL) != self.identity
                or record.get("Name") != "/"+self.names[name]
                or (name in self.state["containers"] and record["Id"] != self.state["containers"][name])):
            raise RuntimeError("owned fixture container identity differs")
        return record

    def _container(self, record, name):
        self._owned_container(record, name)
        host, config = record["HostConfig"], record["Config"]
        networks = record.get("NetworkSettings", {}).get("Networks", {})
        attachment = next(iter(networks.values()), {})
        if (config.get("User") != "pwuser" or record.get("Mounts") != []
                or host.get("NetworkMode") != self.state["network_id"]
                or host.get("ReadonlyRootfs") is not True or host.get("Privileged") is not False
                or host.get("CapDrop") != ["ALL"] or host.get("CapAdd") is not None
                or host.get("Memory") != 128*1024*1024 or host.get("MemorySwap") != 128*1024*1024
                or host.get("NanoCpus") != 500000000 or host.get("PidsLimit") != 32
                or host.get("SecurityOpt") != ["no-new-privileges"]
                or host.get("PortBindings") not in ({}, None) or host.get("PublishAllPorts") is not False
                or host.get("Binds") is not None or len(networks) != 1
                or attachment.get("NetworkID") not in {self.state["network_id"], ""}
                or (record["State"]["Running"] and attachment.get("NetworkID") != self.state["network_id"])):
            raise RuntimeError("owned fixture container differs")
        return record

    def _arguments(self, name, program, *arguments):
        return ["create", "--name", self.names[name], "--label", LABEL+"="+self.identity,
            "--network", self.state["network_id"], "--init", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--memory", "128m", "--memory-swap", "128m",
            "--cpus", "0.5", "--pids-limit", "32", "--log-driver", "none",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=1m,mode=1777", "--entrypoint", "python",
            self.image, "-u", "-c", program, *arguments]

    async def _create(self, name, program, *arguments):
        cid = (await self.invoke(self._arguments(name, program, *arguments))).decode().strip()
        if not _ID.fullmatch(cid):
            raise RuntimeError("owned fixture container identity missing")
        self.state["containers"][name] = cid
        self.record()
        return self._container(await self._inspect("container", cid), name)

    async def start(self):
        image = await self._inspect("image", self.image)
        if image["Id"] != self.image or image["Config"].get("Labels", {}).get(self.image_label) != self.identity:
            raise RuntimeError("owned fixture image differs")
        prefix = secrets.token_hex(5)
        subnet = f"fd{prefix[:2]}:{prefix[2:6]}:{prefix[6:]}:1::/64"
        network_id = (await self.invoke(["network", "create", "--driver", "bridge", "--internal", "--ipv6",
            "--subnet", subnet, "--label", LABEL+"="+self.identity, self.network_name])).decode().strip()
        if not _ID.fullmatch(network_id):
            raise RuntimeError("owned fixture network identity missing")
        self.state["network_id"] = network_id
        self.record()
        network = self._network(await self._inspect("network", network_id))
        (self.output/"network-inspection.json").write_text(json.dumps(network, indent=2)+"\n")
        server = await self._create("server", SERVER, self.nonce)
        await self.invoke(["start", server["Id"]])
        await self.invoke(["exec", server["Id"], "python", "-c",
            "import time;from pathlib import Path;p=Path('/tmp/peer-ready');"
            "\nfor _ in range(100):\n if p.exists():break\n time.sleep(.02)\nelse:raise SystemExit(1)"])
        server = self._container(await self._inspect("container", server["Id"]), "server")
        (self.output/"server-inspection.json").write_text(json.dumps(server, indent=2)+"\n")
        endpoint = next(iter(server["NetworkSettings"]["Networks"].values()))
        targets = []
        for family, field, version in (("ipv4", "IPAddress", 4), ("ipv6", "GlobalIPv6Address", 6)):
            address = ipaddress.ip_address(endpoint.get(field, ""))
            if (address.version != version or not address.is_private or address.is_loopback
                    or address.is_multicast or address.is_unspecified or address.is_link_local):
                raise RuntimeError("owned fixture lacks the required private address family")
            targets.append({"family":family, "host":str(address), "port":PORT})
        self.state["fixture"] = {"proof_id":self.identity, "server_id":server["Id"],
            "network_id":network_id, "targets":targets}
        self.record()
        control = await self._create("control", CONTROL, json.dumps(self.state["fixture"]), self.nonce)
        raw = await self.invoke(["start", "--attach", control["Id"]])
        (self.output/"positive-control.json").write_bytes(raw)
        if json.loads(raw) != {"reachable_owned_targets":targets}:
            raise RuntimeError("owned IPv4/IPv6 positive control failed")
        ended = self._container(await self._inspect("container", control["Id"]), "control")
        if ended["State"]["Running"] or ended["State"]["ExitCode"] != 0 or ended["State"]["OOMKilled"]:
            raise RuntimeError("owned positive control did not retire normally")
        self.state["positive_control"] = {"both_families_reachable":True,
            "nonce_sha256":hashlib.sha256(self.nonce.encode()).hexdigest()}
        await self.verify_live()
        self.record()
        return self

    async def verify_live(self):
        self._network(await self._inspect("network", self.state["network_id"]))
        server = self._container(await self._inspect("container", self.state["containers"]["server"]), "server")
        if not server["State"]["Running"]:
            raise RuntimeError("owned peer stopped before refusal proof")
        endpoint = next(iter(server["NetworkSettings"]["Networks"].values()))
        expected = [target["host"] for target in self.state["fixture"]["targets"]]
        if [endpoint["IPAddress"], endpoint["GlobalIPv6Address"]] != expected:
            raise RuntimeError("owned peer addresses changed")
        counts = json.loads(await self.invoke(["exec", server["Id"], "python", "-c",
            "from pathlib import Path;print(Path('/tmp/peer-counts.json').read_text())"]))
        if counts != {"ipv4":1, "ipv6":1}:
            raise RuntimeError("offline worker reached an owned peer or positive control was incomplete")
        self.state["server_accept_counts"] = counts
        self.record()

    def verify_refused(self, result):
        expected = [{**target, "refused":True} for target in self.state["fixture"]["targets"]]
        if result != {"network_fixture":self.state["fixture"], "refused_peers":expected}:
            raise RuntimeError("owned IPv4/IPv6 refusal evidence differs")

    async def refuse_from(self, container_id):
        await self.verify_live()
        raw = await self.invoke(["exec", container_id, "python", "/opt/kestrel-browser/sandbox_probe.py",
            "--network-only", json.dumps(self.state["fixture"])])
        (self.output/"offline-refusal.json").write_bytes(raw)
        result = json.loads(raw)
        self.verify_refused(result)
        await self.verify_live()
        return result

    async def close(self):
        # Named-resource reconciliation also covers a lost create reply. Never
        # treat an arbitrary daemon failure as proof that a resource is absent.
        for name in ("control", "server"):
            try:
                record = await self._inspect("container", self.names[name])
            except ResourceNotFound:
                continue
            self._owned_container(record, name)
            self.state["containers"][name] = record["Id"]
            self.record()
            await self.invoke(["container", "rm", "--force", record["Id"]])
            try:
                await self._inspect("container", record["Id"])
            except ResourceNotFound:
                pass
            else:
                raise RuntimeError("owned peer container cleanup unconfirmed")
        try:
            network = self._owned_network(await self._inspect("network", self.network_name))
        except ResourceNotFound:
            network = None
        if network is not None:
            if network.get("Containers"):
                raise RuntimeError("owned peer network still has attached resources")
            self.state["network_id"] = network["Id"]
            self.record()
            await self.invoke(["network", "rm", network["Id"]])
            try:
                await self._inspect("network", network["Id"])
            except ResourceNotFound:
                pass
            else:
                raise RuntimeError("owned peer network cleanup unconfirmed")
        self.state["cleanup_confirmed"] = True
        self.record()

    async def __aenter__(self):
        try:
            return await self.start()
        except BaseException:
            await self.close()
            raise

    async def __aexit__(self, *error):
        await self.close()
