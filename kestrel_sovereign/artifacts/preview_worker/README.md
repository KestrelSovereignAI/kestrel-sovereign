# Local artifact preview renderer

Build the immutable Python/Poppler/Pillow image from this directory, then
configure the host with its **local image ID**, never a mutable tag:

```sh
docker build --pull=false -t kestrel-artifact-renderer:local kestrel_sovereign/artifacts/preview_worker
docker image inspect kestrel-artifact-renderer:local --format '{{.Id}}'
export KESTREL_ARTIFACT_RENDERER_IMAGE=sha256:<the-64-hex-local-image-id>
```

The Dockerfile pins the multi-architecture Python base digest, dated Debian
snapshot, Poppler version, and CPython 3.13 Linux Pillow wheel hashes. Building
downloads those dependencies; request-time rendering never pulls an image.
The Docker CLI must address a local Unix socket serving a Linux daemon with
memory, swap, CPU quota and process-count cgroup support. Unsupported or absent
runtime configuration refuses preview while keeping authorized source download.

Read-only Docker context discovery can use the operator's CLI configuration.
Every engine command uses a fresh empty private `--config` directory retained
through the original CLI process and I/O join. The host also rejects proxy
environment settings in the created container before starting the decoder.

Every job has a unique native container identity. The host verifies its entire
sandbox configuration before sending source bytes: no network, host binds,
devices, capabilities or privilege escalation; read-only root; non-root user;
64 MiB noexec tmpfs; no container log driver. The full decoder process tree
shares 512 MiB memory with no swap, half a CPU and 16 processes. The app admits
at most two jobs, including source reads, without a waiter queue. Native create
or cleanup uncertainty retains capacity until exact identity reconciliation.
Cancellation joins the same accepted cleanup; it does not create another job.

Sources are at most 16 MiB. Raster input dimensions are at most 16 megapixels;
only the first frame or first PDF page is rendered, to a maximum 1024 by 1024
RGB image. Worker subprocess/supervisor deadlines are six/eight seconds and
the host attach deadline is nine seconds. The host bounds raw output and emits
fresh PNG IHDR/IDAT/IEND chunks without loading an image decoder. The browser
receives that verified derivative, with an aggregate two-image/two-megapixel
budget per document. HTML/SVG and original compressed image/PDF bytes are never
embedded as a preview. There is no host/browser decoding fallback.

The endpoint uses the artifact's exact immutable revision and original current
owner READ/privacy grant. Its erasure fence remains held through final transport
delivery; canonical deletion waits for that delivery. Derivatives have no second
repository, persistent cache, or independent authority.

Run the real local proofs using the installed immutable image ID:

```sh
ARTIFACT_TEST_RENDERER_IMAGE=sha256:<local-image-id> python -m pytest tests/integration/test_artifact_preview_docker.py tests/integration/test_artifact_preview_host.py tests/integration/test_artifact_product_browser.py
```

These prove native cleanup after cancellation/deadlines, a descendant aggregate
memory OOM probe, global admission, key revocation before decoder input, erasure
through actual ASGI send, and the shipped authenticated browser panel path.
