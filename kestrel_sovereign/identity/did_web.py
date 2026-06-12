"""
did:web producer + resolver — Wave 2 sub-PR 3 of Quantum Hardening (#921, #917).

Per W3C did:web Method Specification v1.0
(https://w3c-ccg.github.io/did-method-web/), a ``did:web`` identifier is
resolved by HTTPS-fetching a JSON document at a deterministic URL:

    did:web:example.com                  -> https://example.com/.well-known/did.json
    did:web:example.com:agent:meridian   -> https://example.com/agent/meridian/did.json
    did:web:example.com%3A8080:foo       -> https://example.com:8080/foo/did.json

Each ``:`` in the path tail becomes a ``/`` in the URL; a trailing path
segment yields a ``/did.json`` suffix; a tail-less DID lands at
``/.well-known/did.json``. Port numbers travel as percent-encoded ``%3A``
in the DID and decode to ``:`` in the URL host.

This module ships:

- ``build_did(domain, path_segments, port=None)`` — DID URI builder
- ``did_to_url(did)`` — DID URI -> resolution URL
- ``url_to_did(url)`` — inverse, for tests + tooling
- ``build_verification_methods(did, suite_pubkey_pairs)`` — produces
  W3C Multikey verification-method dicts that drop straight into the
  ``verificationMethod`` array of a DID document
- ``build_did_document(did, suite_pubkey_pairs, ...)`` — assembles the
  full W3C DID document (``@context`` + verification methods +
  authentication / assertionMethod relationships)
- ``parse_did_document(doc)`` — verifier-side: extract Multikey methods
  and resolve each back to a ``(suite, public_key)`` pair via the
  multikey registry
- ``resolve(did, *, fetcher)`` — pluggable fetcher for testability;
  defaults to ``urllib.request.urlopen`` with strict HTTPS

The resolver is intentionally library-light. Wave 2 sub-PR 4 wires
``inception_service`` to call ``build_did_document`` when a new agent
is hatched; the resolver side is exercised by the verifier paths in
Wave 3 (succession statements) and Wave 4 (CAR import).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote, unquote
from urllib.request import urlopen

from kestrel_sovereign.security.crypto_suite import CryptoSuite, CryptoSuiteError
from kestrel_sovereign.security.multikey import (
    multibase_to_public_key,
    public_key_to_multibase,
)


DID_WEB_PREFIX = "did:web:"
DID_DOCUMENT_FILENAME = "did.json"
WELL_KNOWN_PATH = ".well-known"

# W3C DID Core v1.1 + Multikey contexts. Resolvers MUST see these to
# interpret the document per spec.
DID_DOCUMENT_CONTEXTS = [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/multikey/v1",
]


class DidWebError(Exception):
    """Raised on malformed did:web URIs, parse failures, or fetch errors."""


# ---------------------------------------------------------------------------
# DID URI <-> URL conversions
# ---------------------------------------------------------------------------

def build_did(
    domain: str,
    path_segments: Sequence[str] = (),
    *,
    port: Optional[int] = None,
) -> str:
    """Build a ``did:web`` URI.

    ``domain`` is the host (no scheme, no port). ``port`` if given is
    encoded as ``%3A<port>`` per spec. ``path_segments`` are joined with
    ``:`` after the host (or host:port).

    Examples:
        >>> build_did("example.com")
        'did:web:example.com'
        >>> build_did("example.com", ["agent", "meridian"])
        'did:web:example.com:agent:meridian'
        >>> build_did("example.com", ["foo"], port=8080)
        'did:web:example.com%3A8080:foo'
    """
    if not domain or "/" in domain or ":" in domain:
        raise DidWebError(
            f"domain must be a bare host (no scheme/port/path); got {domain!r}"
        )
    if port is not None:
        if not (1 <= port <= 65535):
            raise DidWebError(f"port must be in 1..65535; got {port}")
        host = f"{domain}%3A{port}"
    else:
        host = domain

    # Each path segment is percent-encoded individually; ":" inside a
    # segment is illegal because ":" is the segment separator.
    encoded_segments: List[str] = []
    for seg in path_segments:
        if not seg:
            raise DidWebError("path segment must be non-empty")
        if ":" in seg:
            raise DidWebError(f"path segment must not contain ':'; got {seg!r}")
        encoded_segments.append(quote(seg, safe=""))

    if encoded_segments:
        return DID_WEB_PREFIX + host + ":" + ":".join(encoded_segments)
    return DID_WEB_PREFIX + host


def did_to_url(did: str) -> str:
    """Resolve a ``did:web`` URI to its HTTPS document URL.

    Per spec:
    - host-only → ``https://<host>/.well-known/did.json``
    - host + path → ``https://<host>/<path>/did.json``

    Percent-encoding in the host (e.g. ``%3A`` for the port colon) is
    decoded to its raw form.
    """
    if not did.startswith(DID_WEB_PREFIX):
        raise DidWebError(f"not a did:web URI: {did!r}")
    body = did[len(DID_WEB_PREFIX):]
    if not body:
        raise DidWebError(f"empty did:web body: {did!r}")

    parts = body.split(":")
    host = unquote(parts[0])
    path_segments = [unquote(p) for p in parts[1:]]

    if not host:
        raise DidWebError(f"did:web host is empty: {did!r}")

    if path_segments:
        path = "/".join(path_segments) + "/" + DID_DOCUMENT_FILENAME
    else:
        path = WELL_KNOWN_PATH + "/" + DID_DOCUMENT_FILENAME
    return f"https://{host}/{path}"


def url_to_did(url: str) -> str:
    """Inverse of :func:`did_to_url`. Useful for tests and tooling.

    Accepts only HTTPS URLs that end in ``/did.json``. The host (with
    optional port) becomes the DID host (port encoded as ``%3A``); the
    intermediate path segments become DID path segments. The well-known
    suffix produces a host-only DID.
    """
    if not url.startswith("https://"):
        raise DidWebError(f"did:web requires HTTPS; got {url!r}")
    rest = url[len("https://"):]
    if "/" not in rest:
        raise DidWebError(f"URL missing path: {url!r}")
    host, _, path = rest.partition("/")
    if not path.endswith("/" + DID_DOCUMENT_FILENAME) and path != DID_DOCUMENT_FILENAME:
        raise DidWebError(
            f"did:web URL must end with /{DID_DOCUMENT_FILENAME}: {url!r}"
        )
    path = path[: -(len(DID_DOCUMENT_FILENAME) + 1)] if path != DID_DOCUMENT_FILENAME else ""
    # Encode host's colon (port) per spec
    encoded_host = host.replace(":", "%3A")

    if path == WELL_KNOWN_PATH or path == "":
        return DID_WEB_PREFIX + encoded_host

    segments = path.split("/")
    return DID_WEB_PREFIX + encoded_host + ":" + ":".join(segments)


# ---------------------------------------------------------------------------
# Verification methods + DID document
# ---------------------------------------------------------------------------

def build_verification_methods(
    did: str,
    suite_pubkey_pairs: Iterable[Tuple[CryptoSuite, Any]],
    *,
    kid_prefix: str = "key",
) -> List[dict]:
    """Build W3C Multikey verification-method entries for a DID.

    Each entry has the shape::

        {
          "id": "did:web:example.com#key-1",
          "type": "Multikey",
          "controller": "did:web:example.com",
          "publicKeyMultibase": "z..."
        }

    The ``id`` fragment is ``#<kid_prefix>-<n>`` (1-indexed) so a hybrid
    identity gets ``#key-1`` (Ed25519) and ``#key-2`` (ML-DSA-65). Callers
    that want stable kids across rotations can post-process the list
    (e.g. set ``id`` to ``#ed25519`` / ``#ml-dsa-65``) before signing.
    """
    methods: List[dict] = []
    for i, (suite, pub) in enumerate(suite_pubkey_pairs, start=1):
        if not isinstance(suite, CryptoSuite):
            raise DidWebError(
                f"verification method #{i}: expected CryptoSuite, got "
                f"{type(suite).__name__}"
            )
        try:
            multibase = public_key_to_multibase(suite, pub)
        except CryptoSuiteError as e:
            raise DidWebError(
                f"verification method #{i} ({suite.alg_id}): {e}"
            ) from e
        methods.append({
            "id": f"{did}#{kid_prefix}-{i}",
            "type": "Multikey",
            "controller": did,
            "publicKeyMultibase": multibase,
        })
    return methods


def build_did_document(
    did: str,
    suite_pubkey_pairs: Sequence[Tuple[CryptoSuite, Any]],
    *,
    kid_prefix: str = "key",
    also_known_as: Optional[Sequence[str]] = None,
    services: Optional[Sequence[dict]] = None,
) -> dict:
    """Assemble a complete W3C DID document for a ``did:web`` identifier.

    All verification methods are listed under ``authentication`` and
    ``assertionMethod`` by reference (per W3C DID Core §5.3 / §5.4) so a
    consumer can verify both holder-binding and credential-issuer
    signatures using the same keys. Hybrid identities get every key in
    both relationships — verifier policy decides which subset is required
    (see :mod:`kestrel_sovereign.security.verify_policy`).

    ``services`` is passed through verbatim if provided. Wave 2 doesn't
    use it; Wave 4 (CAR / capsule sharing) will publish a service
    endpoint here.
    """
    if not did.startswith(DID_WEB_PREFIX):
        raise DidWebError(f"build_did_document: not a did:web URI: {did!r}")

    methods = build_verification_methods(
        did, suite_pubkey_pairs, kid_prefix=kid_prefix,
    )
    method_ids = [m["id"] for m in methods]

    doc: dict = {
        "@context": list(DID_DOCUMENT_CONTEXTS),
        "id": did,
        "verificationMethod": methods,
        "authentication": method_ids,
        "assertionMethod": method_ids,
    }
    if also_known_as:
        doc["alsoKnownAs"] = list(also_known_as)
    if services:
        doc["service"] = list(services)
    return doc


def parse_did_document(doc: dict) -> List[Tuple[str, CryptoSuite, Any]]:
    """Extract every Multikey verification method.

    Returns a list of ``(kid, suite, public_key)`` triples. Non-Multikey
    methods are ignored — DID documents may legitimately carry multiple
    types and we don't want to break on unknown ones. Methods with an
    unknown multicodec raise :class:`DidWebError` (vs. silent skip)
    because that signals a real interop break a caller should know about.
    """
    methods = doc.get("verificationMethod") or []
    if not isinstance(methods, list):
        raise DidWebError(
            f"verificationMethod must be a list; got {type(methods).__name__}"
        )

    out: List[Tuple[str, CryptoSuite, Any]] = []
    for i, m in enumerate(methods):
        if not isinstance(m, dict):
            raise DidWebError(f"verificationMethod[{i}] is not an object")
        if m.get("type") != "Multikey":
            continue
        kid = m.get("id")
        multibase = m.get("publicKeyMultibase")
        if not isinstance(kid, str) or not isinstance(multibase, str):
            raise DidWebError(
                f"verificationMethod[{i}] missing id or publicKeyMultibase"
            )
        try:
            suite, pub = multibase_to_public_key(multibase)
        except CryptoSuiteError as e:
            raise DidWebError(
                f"verificationMethod[{i}] (id={kid!r}): {e}"
            ) from e
        out.append((kid, suite, pub))
    return out


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

# A fetcher takes a URL and returns the raw response bytes. The default
# uses urllib with a tight timeout; tests inject a fake fetcher that
# returns canned bytes without going to the network.
Fetcher = Callable[[str], bytes]


def _default_fetcher(url: str) -> bytes:
    """HTTPS-only urllib fetcher with a 10-second timeout.

    The DID Core spec mandates HTTPS for ``did:web``. We refuse anything
    else at the resolver level — even the convenience of letting a
    caller pass an ``http://`` URL during development (use the fetcher
    arg with a stub if you need that).
    """
    if not url.startswith("https://"):
        raise DidWebError(f"did:web resolver requires HTTPS; got {url!r}")
    # SSRF guard (#1727): the host comes from an attacker-controllable sender DID
    # (e.g. did:web:169.254.169.254 → cloud metadata). Reject non-public targets
    # before the request fires. HTTPS-only is enforced above.
    from kestrel_sovereign.security.ssrf import validate_outbound_url, SSRFError
    try:
        validate_outbound_url(url, allowed_schemes=("https",))
    except SSRFError as e:
        raise DidWebError(f"did:web resolver refused non-public URL {url!r}: {e}") from e
    with urlopen(url, timeout=10) as resp:  # noqa: S310 (HTTPS-checked above)
        if resp.status != 200:
            raise DidWebError(f"GET {url} returned {resp.status}")
        return resp.read()


def resolve(
    did: str,
    *,
    fetcher: Optional[Fetcher] = None,
) -> dict:
    """Fetch and return the DID document for ``did``.

    The returned dict is the parsed JSON; verification of the document's
    contents (signatures, controllers, etc.) is the caller's job — this
    function only handles the resolution step. ``id`` in the document
    MUST equal the requested ``did`` per DID Core; we enforce that here
    so a misconfigured server can't pass off another agent's document.
    """
    url = did_to_url(did)
    fetch = fetcher or _default_fetcher
    raw = fetch(url)
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise DidWebError(f"DID document at {url} is not JSON: {e}") from e
    if not isinstance(doc, dict):
        raise DidWebError(
            f"DID document at {url} must be a JSON object; got "
            f"{type(doc).__name__}"
        )
    if doc.get("id") != did:
        raise DidWebError(
            f"DID document at {url} has id={doc.get('id')!r}, "
            f"expected {did!r} (impersonation guard)"
        )
    return doc
