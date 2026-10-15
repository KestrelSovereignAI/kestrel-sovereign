"""
The one verifier for v2 ``{alg, kid, sig}`` signature arrays (#2487).

Hybrid identity (:func:`identity.hybrid_keypair.verify_hybrid`), succession
statements (:mod:`identity.succession`), the succession-chain artifact
walker (:mod:`identity.succession_chain`) and data-access grants each pair
a ``signatures`` array with a list of W3C Multikey verification methods.
They used to carry four near-identical copies of that pairing, and the
copies disagreed on which malformed inputs they tolerated: an integer
``sig``, an integer or unhashable ``id``/``kid``, or a non-base58btc
``publicKeyMultibase`` escaped as ``TypeError``/``ValueError`` from some of
them. Signature arrays and verification methods are external material (a
peer's DID document, an archived succession statement), so a malformed
entry has to be a rejection, never an exception.

This module answers exactly two questions and nothing else:

- which signature entries cryptographically verify against which
  verification method, and
- what was structurally wrong with the material.

Policy stays with the callers: hybrid thresholds, succession DID binding and
adaptive predecessor policy, archival requirements and temporal cutoffs. The
:meth:`SignatureVerification.evaluate` adapter only feeds the verified
entries to :func:`verify_policy.evaluate_signatures` under a policy the
caller chooses.

Two kinds of finding
--------------------

**Fatal** errors are about the verification-method set or the shape of the
inputs as a whole: a container that is not a list of objects, a method whose
``id`` is not a string, or two methods with the same kid. Nothing verifies
when any fatal error is present, because a kid could otherwise resolve to an
attacker-chosen method (the duplicate-kid takeover).

**Rejected entries** are per-signature: an entry that is not an object,
lacks a string ``alg``/``kid``/``sig``, names an unknown kid, points at a
method whose key cannot be decoded, claims an algorithm other than its
method's key type, carries malformed hex, or fails the suite's
verification. A rejected entry is dropped; the rest are still evaluated.
Callers that need every entry to verify read :attr:`rejected` themselves.

Every external field is type-checked before it is normalized, used as a
dictionary key, multibase-decoded or hex-decoded. Results keep input order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from .crypto_suite import CryptoSuite, CryptoSuiteError
from .multikey import multibase_to_public_key
from .verify_policy import PolicyResult, VerifyPolicy, evaluate_signatures


# Attacker-controlled values are quoted into diagnostics; bound how much of
# each one, and how many findings one summary lists.
_DIAGNOSTIC_VALUE_CHARS = 80
_SUMMARIZED_FINDINGS = 8


def _quote(value: Any) -> str:
    """A bounded, never-raising rendering of an attacker-chosen value.

    Strings are sliced before ``repr`` so a huge one costs nothing. Any other
    type is named rather than rendered: ``repr`` of an arbitrary value can be
    huge or raise (a >4300-digit int does).
    """
    if not isinstance(value, str):
        return f"<{type(value).__name__}>"
    if len(value) > _DIAGNOSTIC_VALUE_CHARS:
        return repr(value[:_DIAGNOSTIC_VALUE_CHARS]) + "..."
    return repr(value)


def _bounded(text: str) -> str:
    limit = 2 * _DIAGNOSTIC_VALUE_CHARS
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize(findings: List[str]) -> str:
    shown = findings[:_SUMMARIZED_FINDINGS]
    hidden = len(findings) - len(shown)
    if hidden > 0:
        shown = shown + [f"... and {hidden} more"]
    return "; ".join(shown)


def _entries(value: Any, what: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    """Materialize a list-shaped input, or explain why it isn't one.

    A string, bytes or a single mapping is iterable but is never a list of
    entries; treating it as one would silently verify nothing (or the
    mapping's keys) instead of reporting the malformed container.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None, f"{what} must be a list; got {type(value).__name__}"
    try:
        return list(value), None
    except TypeError:
        return None, f"{what} must be a list; got {type(value).__name__}"


def verification_method_kid(vm_id: str) -> str:
    """The kid a signature entry uses to address a verification method.

    The fragment after the last ``#`` of the method's ``id``, or the whole
    ``id`` when it has no fragment. Signature entries carry this bare kid.
    """
    return vm_id.rsplit("#", 1)[-1] if "#" in vm_id else vm_id


@dataclass(frozen=True)
class VerificationMethodIndex:
    """Verification methods addressed by kid, plus structural findings.

    ``methods`` maps each kid to its method in input order. A method with
    no ``id`` (or an ``id`` with an empty fragment) is unaddressable and
    left out; it cannot be named by any signature entry.
    """

    methods: Mapping[str, Mapping[str, Any]]
    fatal: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.fatal

    def fatal_summary(self) -> str:
        return _summarize(list(self.fatal))


def index_verification_methods(verification_methods: Any) -> VerificationMethodIndex:
    """Index ``verification_methods`` by kid, failing closed on malformations.

    Fatal findings: the input is not a list; an entry is not an object; an
    ``id`` is present but not a string; two methods share a kid. W3C DID
    Core 5.1.1 requires every verification method id to be unique, and a
    duplicate would let a later attacker-supplied method shadow the
    legitimate one.
    """
    vms, container_error = _entries(verification_methods, "verification methods")
    if container_error is not None:
        return VerificationMethodIndex(methods={}, fatal=(container_error,))

    fatal: List[str] = []
    methods: dict = {}
    first_id_for_kid: dict = {}
    for i, vm in enumerate(vms):
        if not isinstance(vm, Mapping):
            fatal.append(
                f"verification method[{i}] is not an object "
                f"(got {type(vm).__name__})"
            )
            continue
        vm_id = vm.get("id")
        if vm_id is None:
            continue
        if not isinstance(vm_id, str):
            fatal.append(
                f"verification method[{i}] id must be a string; "
                f"got {type(vm_id).__name__}"
            )
            continue
        kid = verification_method_kid(vm_id)
        if not kid:
            continue
        if kid in first_id_for_kid:
            fatal.append(
                f"verification methods contain duplicate kid {_quote(kid)} "
                f"(ids: {_quote(first_id_for_kid[kid])} and {_quote(vm_id)}); "
                f"every verification method id MUST be unique (W3C DID Core "
                f"5.1.1). Refusing to silently overwrite — this is the "
                f"duplicate-kid takeover path."
            )
            continue
        first_id_for_kid[kid] = vm_id
        methods[kid] = vm
    return VerificationMethodIndex(methods=methods, fatal=tuple(fatal))


def decode_verification_method_key(
    vm: Mapping[str, Any],
) -> Tuple[Optional[Tuple[CryptoSuite, Any]], str]:
    """Decode a method's ``publicKeyMultibase`` to ``(suite, public_key)``.

    Returns ``(key, "")`` on success and ``(None, reason)`` when the field
    is missing, not a string, or not a decodable signing Multikey (a KEM
    key published alongside signing keys lands here too).
    """
    multibase = vm.get("publicKeyMultibase")
    if not isinstance(multibase, str):
        return None, "has no string publicKeyMultibase"
    try:
        return multibase_to_public_key(multibase), ""
    except CryptoSuiteError as e:
        return None, f"multikey decode failed: {_bounded(str(e))}"


@dataclass(frozen=True)
class RejectedSignatureEntry:
    """One signature entry that did not verify, and why."""

    index: int
    reason: str


@dataclass(frozen=True)
class SignatureVerification:
    """Which signature entries verified, and what was structurally wrong.

    ``verified`` holds copies of the verifying entries in input order.
    ``rejected`` explains every other entry, in input order. ``fatal`` is
    non-empty when the inputs as a whole were malformed; ``verified`` is
    then always empty.
    """

    verified: Tuple[dict, ...] = ()
    rejected: Tuple[RejectedSignatureEntry, ...] = ()
    fatal: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when the inputs were structurally sound (entries may still
        have been rejected)."""
        return not self.fatal

    def fatal_summary(self) -> str:
        return _summarize(list(self.fatal))

    def rejection_summary(self) -> str:
        return "rejected signature entries: " + _summarize(
            [f"signature[{r.index}]: {r.reason}" for r in self.rejected]
        )

    def evaluate(
        self,
        policy: VerifyPolicy,
        *,
        post_cutoff_classical_allowed: bool = True,
    ) -> PolicyResult:
        """Apply the caller's ``policy`` to the verified entries.

        A fatal finding fails outright with its own reason. A policy failure
        names the rejected entries, so "no signatures present" never hides
        that two malformed signatures were in fact supplied.
        """
        if self.fatal:
            return PolicyResult(
                ok=False,
                reason=self.fatal_summary(),
                alg_ids_seen=frozenset(),
            )
        result = evaluate_signatures(
            list(self.verified),
            policy,
            post_cutoff_classical_allowed=post_cutoff_classical_allowed,
        )
        if result.ok or not self.rejected:
            return result
        return replace(result, reason=f"{result.reason}; {self.rejection_summary()}")


def _check_entry(
    data: bytes,
    entry: Any,
    methods: Mapping[str, Mapping[str, Any]],
) -> str:
    """Return ``""`` if ``entry`` verifies over ``data``, else why not."""
    if not isinstance(entry, Mapping):
        return f"entry is not an object (got {type(entry).__name__})"
    alg = entry.get("alg")
    kid = entry.get("kid")
    sig_hex = entry.get("sig")
    for name, value in (("alg", alg), ("kid", kid), ("sig", sig_hex)):
        if not isinstance(value, str) or not value:
            return f"{name!r} must be a non-empty string; got {_quote(value)}"

    vm = methods.get(kid)
    if vm is None:
        return f"no verification method for kid {_quote(kid)}"
    key, decode_error = decode_verification_method_key(vm)
    if key is None:
        return f"verification method {_quote(kid)} {decode_error}"
    suite, public_key = key
    # The entry's alg must name the method's actual key type; otherwise an
    # Ed25519 signature could be tagged ml-dsa-65 to satisfy a PQ policy.
    if suite.alg_id != alg:
        return (
            f"alg {_quote(alg)} does not match verification method "
            f"{_quote(kid)} key type {suite.alg_id!r}"
        )
    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        return f"signature for kid {_quote(kid)} is not valid hex"
    if not suite.verify(data, sig_bytes, public_key):
        return f"signature for kid {_quote(kid)} ({alg}) does not verify"
    return ""


def verify_signature_entries(
    data: bytes,
    signatures: Iterable[Any],
    verification_methods: Any,
) -> SignatureVerification:
    """Cryptographically verify each ``signatures`` entry over ``data``.

    Each entry is paired with the verification method whose kid equals the
    entry's ``kid``, the method's Multikey is decoded, the entry's ``alg``
    must equal the decoded suite, and the hex signature must verify. Never
    raises for malformed signatures or verification methods; see the
    module docstring for which findings are fatal.
    """
    index = index_verification_methods(verification_methods)
    entries, container_error = _entries(signatures, "signatures")
    fatal = list(index.fatal)
    if container_error is not None:
        fatal.append(container_error)
    if fatal:
        return SignatureVerification(fatal=tuple(fatal))

    verified: List[dict] = []
    rejected: List[RejectedSignatureEntry] = []
    for i, entry in enumerate(entries):
        # Check and return the same plain copy, so a mapping that answers
        # differently through ``get`` and ``[]`` cannot verify as one
        # algorithm and be counted by the policy as another.
        snapshot = dict(entry) if isinstance(entry, Mapping) else entry
        reason = _check_entry(data, snapshot, index.methods)
        if reason:
            rejected.append(RejectedSignatureEntry(index=i, reason=reason))
        else:
            verified.append(snapshot)
    return SignatureVerification(
        verified=tuple(verified),
        rejected=tuple(rejected),
    )


__all__ = [
    "RejectedSignatureEntry",
    "SignatureVerification",
    "VerificationMethodIndex",
    "decode_verification_method_key",
    "index_verification_methods",
    "verification_method_kid",
    "verify_signature_entries",
]
