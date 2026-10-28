# Local visual browser substrate

This is the pinned browser image and sandbox verification for Core #3709,
part of #3703. It contains Chromium, a synthetic offline probe and a private persistent
browser worker, concrete SDK adapters, contained workspace transport and a
bounded active-call HTTP relay. Host authorization, durable custody and artifact
publication remain the canonical host's responsibility. The local proofs below
do not establish a completed visual-computer product or remote-provider support.

Build and verify on an available local Docker daemon:

```sh
python scripts/verify_visual_browser_sandbox.py --output /tmp/kestrel-browser-evidence-new
```

The evidence directory must not exist. The check builds only the image context
in this directory, labels its image/container with a unique ownership ID,
verifies Docker's actual configuration before starting the browser, and removes
only those exact resources afterward. It retains a custody manifest, build log,
browser diagnostics and a screenshot of synthetic content. An interrupted or
timed-out Docker operation keeps `cleanup_confirmed: false`; inspect the retained
resource identifiers and label before completing cleanup. Do not use global
Docker prune commands. The probe's in-container OS deadline is 90 seconds.

The verified configuration uses a non-root `pwuser`, read-only root filesystem,
no host mounts or Docker socket, no container capabilities, no privilege gain,
private process/IPC/UTS/cgroup namespaces, bounded CPU/memory/PIDs and temporary storage,
no swap, the default runc runtime and the reviewed proc/sys path masks,
and `network=none`. `host-config.json` is the complete reviewed HostConfig from
Docker 29.8.2, with local log retention disabled and realtime scheduling denied.
Its bytes are pinned alongside seccomp; extra, missing or changed daemon fields
are unsupported until reviewed. The verifier does not accept arbitrary daemon
defaults. The process independently measures all five Linux capability sets,
supplementary groups and RLIMIT_RTPRIO; root group and realtime priority fail.
These profiles preserve LF bytes on Windows checkouts. Chromium must report its PID/network namespace and
Seccomp-BPF sandboxes active. A synthetic form edit/click must persist its
result, and public-IP and metadata-address connection attempts must fail.
This is a real browser/container test; it does not exercise an agent, production
tools, reconnect custody, provider conformance or host-managed egress.

## Provenance and updates

The base is Microsoft's Playwright Python **1.63.0 Noble** OCI image index:
`sha256:72bd171a9ffc2b4b59532aaa6210e21014d07093120dc25528870c0b840da1f0`.
The index includes linux/amd64 and linux/arm64; current local evidence covers
linux/arm64 only. Playwright and transitive Python dependencies are version and
hash pinned in `requirements.txt`. The official image supplies browsers and
system libraries; the matching Python package is installed separately.

`seccomp.json` derives from [Microsoft's v1.63.0 profile](https://github.com/microsoft/playwright/blob/v1.63.0/utils/docker/seccomp_profile.json)
(upstream SHA-256 `cc3e61cabda6bbc1e53e54d27ba4d55a9d3be829b6dd1a596f4a7b31b1cc7849`),
licensed under Apache-2.0 in `LICENSE.playwright` with attribution in `NOTICE.playwright`. The sole syscall-policy
addition is `chroot` in the existing user-namespace rule. Chromium's sandbox
needs it inside its new user namespace when all outer container capabilities
are dropped. Kernel capability checks still apply; neither `SYS_ADMIN` nor
`SYS_CHROOT` is added to the container. The upstream default-deny policy remains. The verifier pins the complete local
profile SHA-256 (`38c15179be39d9bc1aadf19b06618bdb0dd9137f531435ff306c8dccf3b59f19`)
and refuses any change before building or starting a container.
The unmodified profile refused Chromium's sandbox chroot on the tested host;
the amended profile passed with the Chromium sandbox enabled. There is no
`--no-sandbox` fallback.

For an update, resolve and record a new official image digest, pin the matching
Playwright package, regenerate the hash lock with its recorded command, review
upstream seccomp/license/NOTICE changes, update the reviewed policy digest and
inspect-field expectations when warranted, and run both the unit checks and this real
Docker probe. A passing probe does not by itself authorize visiting untrusted
websites: that release additionally needs the driver, current grants, isolated
egress, authenticated transport, control fencing and live evidence in #3708,
#3709, #3710 and #3715. See the [official Docker guidance](https://playwright.dev/python/docs/docker).

## Private worker

Run `python scripts/verify_visual_browser_sandbox.py --worker --output /tmp/kestrel-worker-evidence-new`
to exercise the same admitted container with a persistent page. This check
creates synthetic offline content, operates it over separate authenticated Unix
socket exchanges and confirms physical expiry without a close request. It uses
the same unique ownership labels, policy verification and exact cleanup as the
substrate probe. Evidence records source revision/dirty state and actual build
context hashes and actual process/Chromium sandbox admission evidence; expiry acceptance requires a normal exit at or after the admitted
expiry, not merely a stopped container. Its screenshot contains only the synthetic draft.

The image's `worker.py` accepts one trusted-host bootstrap through
`docker exec -i <exact-owned-container-id> python /opt/kestrel-browser/rpc_client.py initialize`.
Bootstrap JSON travels on stdin, includes a new 256-bit secret, a SDK-shaped
SessionRequest and at most 64 KiB of initial offline HTML. It must be submitted
only after container admission. Bootstrap is one-shot; an unknown initialize
outcome is reconciled against that container, never replaced with new custody.
The temporary secret file is removed before browser startup; a marker prevents
reinitialization. The socket directory is private and there is no published port,
CDP endpoint or environment/argv credential. Local Docker access remains trusted
host authority, not a capability granted to an agent.

Subsequent `rpc_client.py` stdin exchanges require that secret and bind one owner,
lease, session and control epoch. A fresh transport reaches the existing page.
Actions use the latest produced frame, bounded age, deadline and action IDs.
Accepted effects retain completed/unknown receipts and are never automatically
replayed. A lost observation response requires observing again. Unknown transport
results require status reconciliation; control retries reconcile with session
status. If an accepted input times out or its waiter is cancelled, the worker
fences and retires the session: cancelling a Playwright wait cannot unsend a
Chromium input. That session cannot report a successful pause or be resumed.
Pause and close remain available when the control-history budget is exhausted;
resuming is refused. Transport has four connection slots with a 20-second bound;
the trusted host must avoid holding idle slots and retain physical cleanup
authority if control transport is unavailable. Authenticated manual control uses
the same worker endpoint, priority socket and shared input journal. Pause fences
queued input before draining accepted work, suspends observations and advances
the control epoch; resume advances it again. Logical takeover fences the agent
with a paused human controller. Published SDK 0.45 provides the Human control
contracts; this worker implements `HumanTransport`. The host must explicitly enable the manual channel and bind
its original issued consumer lease. The built-image and host acceptance gates
below still apply. A frame is an observation identity, not a promise that autonomous
page content is static.

The offline profile supports screenshot, click, text, a restricted key set,
scroll and wait. The explicit HTTP profile also supports agent navigation;
manual human navigation remains refused.
Waits send no browser input: a wait that cannot fit its deadline is refused,
and an interrupted wait has a known denial without retiring a healthy page.
The maximum 10-second wait has a separate bounded scheduling allowance.
Navigation requires the explicit HTTP profile below; file bytes use the distinct
negotiated workspace extension. Clipboard and unrestricted exec are refused. Sessions last at most
one hour, with wall and monotonic expiry, bounded receipts and IPC, an async
retirement loop and an OS process deadline (up to ten seconds of teardown grace).
With the reviewed no-restart Docker configuration, process termination retires
its child processes. Lost worker state is not recoverable accepted-action state;
a replacement is a new session. The host must retain custody until physical
container retirement is confirmed. None of this is the host's authorization,
privacy policy, egress gateway, inference adapter or SDK provider/driver factory.

## Local SDK adapters

`kestrel_sovereign.visual_computer.docker` supplies separate
`environment_provider_factory` and `computer_driver_factory` SDK factories.
The host supplies an explicit local Unix Docker socket; inherited Docker
contexts, proxy configuration and operator credentials are excluded. Provider
settings require a private 32-byte `credential_key`, a `profile_id` of
`offline-v1`, and an `image` such as `local-worker@sha256:<configuration digest>`.
For this local profile the digest is the exact locally built Docker image ID,
not a registry manifest or multi-platform index digest. Build the reviewed
worker first; the adapter never pulls an implicit replacement.

The canonical reviewed profiles now live in
[`docker_profile`](../../kestrel_sovereign/visual_computer/docker_profile/).
The substrate verifier and installed adapters share those exact files. The
measured Docker 29.8.2 cgroup-v2 API changes `OomKillDisable: false` to `null`
after start; the policy explicitly recognizes that serialization only in
running/exited states. Other changes to the complete HostConfig are refused.

A connection binds the full immutable container ID, image ID, PID, start time,
worker-generated startup nonce, owner and fixed expiry. One-shot bootstrap
checks the nonce before accepting the secret, closing the inspect/start race.
The private `control.sock` reserves two connections for status/action status
and control, independently of the four normal IPC slots. Takeover fences queued
input, drains accepted work, suspends agent observations and advances the epoch
to a paused human controller. It does not enable a public human-input or CDP
transport. An uncertain Chromium input still retires the browser and cannot
report successful takeover or resume.

The canonical host remains the only durable custody owner: encrypt
`DriverConnection`, retain unknown acquisition and action outcomes, enforce
released-request tombstones, and hold current authority/privacy guards through
all adapter calls and deliveries. Docker labels reconcile a retained request;
they do not authorize a caller or replace the host's released-request history.
Idle policy belongs to that host; this worker independently enforces its fixed
wall and monotonic expiry. No remote, x86_64, navigation, allowed egress,
clipboard or unrestricted exec capability is advertised by `offline-v1`.

Run the opt-in SDK acceptance with a new private evidence directory:

```sh
python -m scripts.verify_visual_docker_adapters \
  --docker-socket /absolute/path/to/local/docker.sock \
  --output /absolute/path/to/new/private-evidence
```

It drives the production adapters against the real daemon and Chromium,
reconnects to the same page, verifies actual pixels and deduplicated receipts,
occupies all normal transport slots before priority takeover, rejects stale
control, checks IPv4/IPv6 network denial, waits for physical autonomous expiry,
and verifies exact resource cleanup. Its screenshots are synthetic private
fixtures. This local proof is not a completed product or a remote-provider claim.

Network refusal checks use only a uniquely labelled server on an owned internal
dual-stack Docker network. A separate bounded peer proves both inspected IPv4
and IPv6 listeners are reachable before the unchanged `network=none` browser
attempts those exact addresses. The standalone sandbox verifier uses the same
fixture. No public, metadata-service or operator-LAN destination is probed.
Missing IPv6 fixture support fails the check; it does not silently reduce the
claim to IPv4. Peer containers have fixed process deadlines and resource bounds,
and cleanup verifies the original native container/network IDs and labels.


## Negotiated workspace transport

The worker pins published SDK `0.45.0` in its normal-index hash lock and requires
the `io.kestrel.visual-computer.worker-contract=sdk-0.45-human-v1` label. Core
declares SDK `>=0.45.0,<1`. The Dockerfile checks that the installed dependency
satisfies the worker pin and exposes the human contract. Source changes still
require a new built-image proof and complete host acceptance; a resolved SDK
lock alone does not establish that this Core successor is released.
Older images are refused rather than claiming their file-transfer capability.
The driver implements the SDK's `WorkspaceTransferProvider` extension;
`workspace_transfer()` returns a distinct transport with all standard prepare,
read/write, status, release/abort and commit methods. The host artifact service
must hold its exact current owner/privacy/session guard across every call.

The trusted host also injects `workspace_guard(session, original_transfer_id)`
from the same original-call borrower machinery, with a separate workspace
purpose. Every prepare/read/write/commit/status RPC borrows that retained checker
on an owned child task through async inspection, process launch, final stdin
write and private reply. Priority control cancels and joins workspace borrowers
before fencing the worker. A pre-dispatch revocation sends no new file command;
post-dispatch uncertainty retains the original transfer ID for reconciliation.
Release/abort can retire retained original IDs without fresh data authority;
that cleanup path cannot read or install new bytes. SDK signatures are unchanged.
Bare SDK callers still supply their own authorization responsibility.

Data requires a paused session with controller `none`, current epoch and live
transfer/session expiry. Takeover fences data; authenticated cleanup remains
available for retained partial state. The explicit `browser` workspace exports
`document.html`, a frozen HTML source snapshot of the current browser DOM.
HTML remains untrusted source, not an approved active renderer. Export IDs
freeze once; released IDs cannot export again. Imported workspaces are fresh
named targets; a complete identical retry returns the original receipt.

Every path uses the SDK's alias/path validation plus descriptor-relative
`O_NOFOLLOW` directory/file opens. Reads require owned regular files with one
hard link, same filesystem, stable metadata, size and digest checks. FIFO,
symlink, device and hard-link exports are refused. There is no archive or glob
expansion. Imports use exclusive file creation, verify the exact manifest tree,
and install through Linux `renameat2(RENAME_NOREPLACE)`; a racing target cannot
be replaced. An unknown installation receipt reconciles the original intent
and actual installed tree without another installation. Partial content never
becomes a live workspace and is removed by explicit abort or physical lease
retirement.

Limits are 32 selected files, 128 KiB per file, 2 MiB per transfer, path depth
16 and 1024 UTF-8 path bytes, 64 transfer IDs and 16 MiB reserved managed content
per worker. Completed workspaces retain their quota until lease retirement.
The outer tmpfs/resource limits still apply. This transport does not mount or
address ambient host files and does not publish an artifact itself.

## Bounded active-call HTTP profile

The offline profile remains the default. An explicit private `network_policy`
setting on both factories selects up to eight exact HTTP(S) origins and GET/HEAD
methods; the provider's computed `profile_id` binds that policy digest. Normal
settings contain `origins`, `methods` and an `address_pins` mapping from each
hostname origin to one explicit trusted public IPv4 address. For example,
`{"origins":["https://example.com"],"methods":["GET","HEAD"],"address_pins":{"https://example.com":"93.184.216.34"}}`
describes an operator-maintained pin, not a verified current service address.
Literal IPv4 origins pin themselves. Missing, extra or nonpublic pins and
unmeasured IPv6 egress are refused. Runtime libc DNS is never scheduled: its
accepted OS resolver work could outlive task cancellation. The gateway connects
only the admitted literal address, retaining the original HTTP Host, TLS SNI
and normal certificate validation. DNS changes require a newly admitted policy.
No ambient proxy,
cookie jar, authentication header, operator profile or retry is used.

The browser still has `network=none`. A private authenticated relay carries
bounded HTTP jobs through the exact session, control epoch and action ID to the
host gateway. The trusted host injects its `egress_guard(session, action_id)`
lexical callback. An owned child task holds the original operation's authority
and privacy lease across each job; synchronous current checks run before
connect, request send and response delivery. Another owned child retains this
same guard for every input, including offline input, through asynchronous
container inspection and the final stdin write. A guard rejection before that
write sends no input; transport loss or revocation afterward remains unknown.
The bare SDK adapter does not replace its caller's authorization responsibility.

Only the original active `act` RPC can admit jobs. Its closure cancels and joins
all gateway work and seals worker response handlers. Idle pages, waits, observe,
status and background requests acquire no HTTP authority. Stop and takeover
close accepted sockets and drain Chromium fulfillment handlers before a
successful fence. Cancelled CDP input or uncertain response fulfillment still
retires the worker; cancelling a waiter cannot unsend a browser effect.

This deliberately narrow profile refuses redirects: Playwright's routing API
does not intercept every redirect hop. It also refuses credentialed browser
requests, Set-Cookie, compressed responses, attachments, protocol upgrades and
informational responses. Service workers and downloads are disabled;
WebSocket routes close, WebRTC constructors refuse, and unsupported non-HTTP
channels have no gateway. Chromium's independent network namespace denies
direct TCP/UDP and DNS. No unauthenticated CDP or browser socket is published.
Per action limits are 16 HTTP requests, 4096 URL characters, 16 KiB response
headers, 128 KiB per response body and eight seconds per job, within the action
and fixed lease deadlines. Request URLs, query values, headers, bodies and
remote exception messages never enter normal gateway logs or audit output.

Run the explicit controlled test with a new private evidence directory:

```sh
python -m scripts.verify_visual_docker_egress \
  --docker-socket /absolute/path/to/local/docker.sock \
  --output /absolute/path/to/new/private-http-evidence
```

It creates two labelled, bounded synthetic Docker servers and verifies each
exact container ID and local port. A test-only Python object admits only the
one owned endpoint through the same literal-pin connection path; ordinary
configuration cannot enable this exception. A host resolver sentinel fails the
real proof if runtime libc DNS is called.
Actual Chromium navigates, fetches a subresource, edits pixels, exports its DOM,
refuses forbidden/background channels and redirects, and closes accepted slow
HTTP sockets before both pause and takeover acknowledgements. Original unknown
action IDs remain reconcilable; stale epochs fail. All owned containers and the
image are removed by exact identity. This proves local controlled HTTP; it does
not establish a public HTTPS interoperability, remote platform, cookie login,
redirect, manual human-input or general browsing capability.

The separate owned TLS proof runs with:

```sh
python -m scripts.verify_visual_docker_tls \
  --docker-socket /absolute/path/to/local/docker.sock \
  --output /absolute/path/to/new/private-tls-evidence
```

It creates a temporary CA and two owned certificate/server fixtures. Normal
platform trust must refuse that CA; a verifier-only SSLContext trusts it while
retaining normal verification flags, hostname checks and required certificate
validation. A leaf with the wrong SAN must still fail. The matching leaf enables
the actual browser edit/export journey over TLS, retaining original Host and
SNI while connecting to the exact fixture pin. Stop must close its accepted TLS
socket and join owned child borrowers before acknowledgement. The test-only TLS
origin derives from its unique Docker ownership label and cannot be configured
through normal settings. Production code has no TLS bypass, custom trust or
certificate option. This measures owned HTTPS behavior, not public HTTPS
interoperability.
