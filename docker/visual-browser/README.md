# Local visual browser substrate

This is the pinned browser image and sandbox verification for Core #3709,
part of #3703. It contains Chromium, a synthetic offline probe and a private persistent
browser worker. Host SDK-driver registration, current-grant authorization,
provider custody, artifact transfer, human takeover and governed network access
remain separate work. It is not a completed visual-computer capability.

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
authority if control transport is unavailable. Priority human control remains
part of host integration. Pause fences
queued input before draining accepted work, suspends observations and advances
the control epoch; resume advances it again. This is not human takeover. A frame
is an observation identity, not a promise that autonomous page content is static.

Only screenshot, click, text, a restricted key set, scroll and wait are supported.
Waits send no browser input: a wait that cannot fit its deadline is refused,
and an interrupted wait has a known denial without retiring a healthy page.
The maximum 10-second wait has a separate bounded scheduling allowance.
Navigation, clipboard, files, exec and takeover are refused. Sessions last at most
one hour, with wall and monotonic expiry, bounded receipts and IPC, an async
retirement loop and an OS process deadline (up to ten seconds of teardown grace).
With the reviewed no-restart Docker configuration, process termination retires
its child processes. Lost worker state is not recoverable accepted-action state;
a replacement is a new session. The host must retain custody until physical
container retirement is confirmed. None of this is the host's authorization,
privacy policy, egress gateway, inference adapter or SDK provider/driver factory.
