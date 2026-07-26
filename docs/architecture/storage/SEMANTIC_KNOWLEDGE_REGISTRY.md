---
type: Architecture Reference
title: Semantic Knowledge Registry
description: Offline, versioned ontology, shape, rule, and standards capability registry.
resource: /docs/architecture/storage/SEMANTIC_KNOWLEDGE_REGISTRY.md
tags:
- docs
- architecture
- storage
- semantic-knowledge
timestamp: '2026-07-26T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Semantic Knowledge Registry

`kestrel_sovereign.knowledge.registry` is the one source of truth for the
semantic-KB artifact contract. Its package-data manifest is
`kestrel_sovereign/data/semantic/registry.toml`; every entry has a stable
identifier, exact semantic version, local package-resource path, SHA-256,
dated URI, maturity, selected terms, and explicit imports.

The registry resolves imports locally in dependency-first order. It rejects a
missing import, cycle, version conflict, duplicate namespace, absent resource,
or digest mismatch. It does not dereference a standards URI, follow
`owl:imports`, choose a latest version, or use the network at startup, sleep,
import, inference, validation, or query time. Package resources are verified
through `importlib.resources`, so the same check operates from an installed
wheel.

`https://kestrel.ai/vocab/` is the Kestrel term namespace. Its ontology IRI is
versioned (`https://kestrel.ai/vocab/1.0.0`); this preserves stable term IRIs
such as `https://kestrel.ai/vocab/preferredDeployRegion` while pinning the
schema interpretation. The Kestrel ontology contains assertion/revision/source
and temporal/provenance schema only. It explicitly grants no permission, role,
approval, visibility change, or tool access.

The stable registry packages and pins the dated RFC 3986 artifact used by
the IRI profile, plus the dated W3C source documents for RDF 1.1 Concepts,
Turtle, N-Triples, N-Quads, TriG, XML Schema 1.1 Datatypes, RDF Dataset
Canonicalization, OWL 2 Profiles, SHACL Core, and SPARQL 1.1. Its local
Kestrel profiles import those sources explicitly. It also records the selected
RDFS, SKOS, PROV-O, and OWL-Time uses defined by the
[semantic knowledge architecture](SEMANTIC_KNOWLEDGE_ARCHITECTURE.md#standards-and-compatibility-matrix).
The RDF 1.2, SHACL 1.2, and SPARQL 1.2 dated snapshots are distinct
`experimental` resources. They require `allow_experimental=True`; they do not
share a capability identifier with the stable baseline and cannot become a
default merely because they are installed.

Semantic consumers receive `SemanticCapabilityContract` and `ArtifactPin`
objects, rather than inventing local version checks. This contract is
non-authorizing. The trusted constitution, approval, and privacy paths remain
the sole owners of authorization decisions.

## Operator and developer commands

Check the exact packaged closure without any network access:

```bash
python -m kestrel_sovereign.knowledge.registry check
```

Refreshing a pin is a deliberate source-maintenance action. Review and replace
the matching package resource first, consciously update the relevant version
and dated URI in the manifest, then run refresh with that exact resource.  The
command refuses a snapshot whose bytes differ from its package resource, so it
cannot write a manifest that fails the next load:

```bash
python -m kestrel_sovereign.knowledge.registry refresh \
  --manifest kestrel_sovereign/data/semantic/registry.toml \
  --resource kestrel-vocab --version 1.0.0 \
  --snapshot /reviewed/kestrel-vocab-1.0.0.ttl
```

The refresh command never downloads a resource and is never called by an
agent. A changed standard draft or recommendation must receive a new resource
identifier/version and compatibility assessment; it cannot overwrite or silently
replace an existing runtime pin.
