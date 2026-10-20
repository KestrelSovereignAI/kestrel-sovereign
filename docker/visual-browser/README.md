# Local visual browser substrate

This is the pinned browser image and sandbox verification for Core #3709,
part of #3703. It contains Chromium and a synthetic offline probe. It does
not yet contain a governed computer driver, a session API, artifact transfer,
takeover or network access. It is not a completed visual-computer capability.

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
and `network=none`. Chromium must report its PID/network namespace and
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
