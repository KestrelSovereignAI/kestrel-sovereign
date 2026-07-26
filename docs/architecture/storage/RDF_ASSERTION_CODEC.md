---
type: Architecture Reference
title: RDF Assertion Codec Boundary
description: Lossless RDF 1.1 projection and capability-negotiated RDF 1.2 boundary for canonical semantic assertions.
resource: /docs/architecture/storage/RDF_ASSERTION_CODEC.md
tags:
- docs
- architecture
- storage
- semantic-knowledge
- rdf
timestamp: '2026-07-26T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# RDF assertion codec boundary

`kestrel_sovereign.knowledge.rdf_codec` is a projection boundary for the
canonical `Assertion` value model.  Assertions retain their deterministic,
tenant-scoped identity outside RDF; no serialized triple, blank-node label, or
RDF backend identifier is canonical storage identity.

## Stable RDF 1.1 projection

`RdfAssertionCodec.project(assertion)` emits `rdf11-reification` by default.
The assertion IRI (`urn:kestrel:assertion:sha256:…`) is a
`kestrel:SemanticAssertion` with `kestrel:hasRevision` pointing at an opaque,
deterministic revision resource.  That revision is both
`kestrel:AssertionRevision` and `rdf:Statement` and has:

- `rdf:subject`, `rdf:predicate`, and `rdf:object` for the canonical claim;
- typed confidence, epistemic state, assertion time, observed/valid intervals,
  lifecycle state, visibility, policy references, and identity profiles;
- a versioned ontology resource containing namespace, version, digest, and
  compatibility profile;
- ordered direct-source members, or a PROV activity with ordered derived-input
  members and complete rule/run lineage.

The statement resource carries metadata; it does not replace the assertion's
identity.  `serialize_rdf11_ntriples()` is deterministic N-Triples 1.1 output.
The packaged golden example is
`kestrel_sovereign/data/semantic/fixtures/rdf11-direct-language.nt`, so an
installed wheel—not a source checkout—contains the conformance artifact.

`projectedTenantId` and `projectedOwnerId` are audit labels only.  Import
requires `RdfImportOwnership` from the governed import boundary and rejects a
projection whose labels contradict it.  RDF content never establishes an
authoritative tenant or source owner.

## Experimental RDF 1.2 and SPARQL

The RDF 1.2 form is never selected as a fallback.  Construct the codec with an
`RdfCodecConfiguration` naming both the registry capability and its exact
version; e.g. the currently packaged draft capability is discovered from the
semantic registry rather than assumed by callers. The experimental projection
uses a typed `RdfTripleTerm`. The Kestrel revision resource is the reifier and
relates to that triple term solely through `rdf:reifies`; it does not emit or
require an `rdf:Reifier` class assertion. All Kestrel metadata remains attached
to the revision resource. A codec without that exact selection raises
`UnsupportedRdfCapabilityError` for an RDF 1.2 request instead of emitting an
RDF 1.1 shape.

`RdfCapabilityReport` returns the selected digest-pinned registry artifacts,
including the Kestrel ontology, PROV-O, and OWL-Time resources used by a
projection. Removing the experimental branch removes only this codec
representation; it does not require an assertion or storage migration. RDF 1.2
and SPARQL 1.2 draft syntax is not a durability promise.

SPARQL 1.2 has a separate `Sparql12AssertionReadAdapter`, which is available
only after the exact SPARQL 1.2 registry pin is selected. It forwards that pin
to the backend and compiles a distinct SPARQL 1.2 typed-read request; it never
silently falls back to the SPARQL 1.1 adapter. At that private adapter edge it
binds a projected claim through `rdf:reifies <<( subject predicate object )>>`;
it does not query the RDF 1.1 `rdf:subject`, `rdf:predicate`, or `rdf:object`
shape. Application callers still use only assertion-contract fields and cannot
provide raw draft triple-term syntax. The adapter also requires the exact RDF
1.2 triple-term/reifier projection capability, so a SPARQL 1.2 configuration
cannot accidentally query a stable RDF 1.1 projection with the draft shape.

Application reads use `RdfTypedQuery`, wrapping the existing bounded
`AssertionQuery`, and `RdfAssertionReadAdapter`.  A SPARQL implementation may
compile that typed request only at its private adapter edge
(`Sparql11AssertionReadAdapter` or the explicitly selected
`Sparql12AssertionReadAdapter`); raw SPARQL text is not an application API.
The adapter compiles subject, predicate, object, assertion IDs, lifecycle,
epistemic state, valid/observed interval containment, and the result limit.
Results have a deterministic assertion-ID/revision-ID order. `cursor` is
rejected before backend execution until a backend-neutral canonical cursor
encoding is defined. The private backend contract receives the registry pin,
must enforce its execution timeout, and must support cancellation; the adapter
materializes no more than `limit + 1` rows before it cancels and rejects an
over-producing backend.

## Import safety

The production `import_assertion()` boundary accepts raw RDF 1.1 N-Triples
bytes only. It invokes the maintained `RdfLibNTriplesParser` with a fixed,
supervised in-memory N-Triples stream before an internal
`RdfImportDocument` exists. There is no application-controlled syntax, base
URI, document location, loader, context, import, or parser-hook option.
`RdfImportLimits` rejects oversized byte streams before parsing. The streaming
sink rejects a statement-count overrun before retaining the next triple, and a
supervising parent terminates a parser worker that exceeds its deadline. The
codec then rejects term limits, excessive nesting/cycles, and parser time
overruns. N-Triples has no remote-context or import syntax; the worker receives
bytes rather than a location, so it cannot dereference a URL or file. RDF term
IRIs remain inert data at this boundary, so valid terms such as `mailto:` and
`tel:` round-trip without enabling a fetch. `import_projected_dataset()` is
limited to typed, in-process
projections (including the negotiated RDF 1.2 form), not external bytes.

The codec rejects rather than erases blank/local identifiers, directional
language literals, unsupported triple terms, datatype/language distinctions,
or incomplete lineage.  It is a read/write transport contract only and has no
database imports or direct persistence path.
