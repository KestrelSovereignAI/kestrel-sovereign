---
type: Architecture Spec
title: Identity Package Privacy Contract
description: Defines which conversation-derived data identity packages contain and excludes raw conversation history.
resource: /docs/architecture/security/IDENTITY_PACKAGE_PRIVACY.md
tags:
- docs
- architecture
- security
- privacy
timestamp: '2026-07-15T00:00:00Z'
status: active
owner: architecture
canonical: true
generated: false
privacy: public
---

# Identity Package Privacy Contract

## Contract

`AgentIdentityPackage` does **not** contain raw conversation history. Its v1 and
v2 schemas have no `conversations`, `conversation_history`, or `messages` field,
and the importer has no path that restores a raw transcript.

`IdentityExporter.export(include_conversations=False)` remains compatible with
older callers during a deprecation window, but the argument does not select a
payload variant. Given the same source state and export timestamp, the default
and explicit-`False` paths produce the same package version, canonical content,
and content hash. Passing `True` is rejected instead of returning a package that
falsely appears to contain a transcript. Callers should remove the argument
before the next major exporter API revision.

Adding raw history later requires a separately versioned schema and an explicit
contract for import, deletion, redaction, encryption, and sealed export. It must
not be smuggled into the current package format.

## Calibration examples are different

The personality fingerprint can contain `calibration_examples`. These are a
bounded, conversation-derived personality feature, not a conversation backup:

- at most 10 eligible adjacent user/assistant pairs are selected;
- each selected user input is truncated to 1,000 characters;
- each selected assistant response is truncated to 1,500 characters;
- retrieved-context wrappers are removed from the user input before export;
- the structured examples live under `personality.calibration_examples`, never
  in a raw transcript field; and
- the first five selected examples are also rendered into
  `system_prompt_template`, with user input truncated again to 300 characters
  and assistant output truncated again to 500 characters.

The bounds reduce scope; they do **not** anonymize or redact the selected text.
Calibration examples can still contain personal or sensitive content. Treat an
identity package as sensitive even though it excludes full conversation history:
signatures provide authenticity, not confidentiality. Keep local plaintext
exports protected, and use the encrypted/sealed paths when the package leaves a
trusted host.

## Other conversation-derived state

Episodes, reflection insights, relationships, vocabulary preferences, and other
portable identity fields may also be derived from prior interactions. Excluding
raw history is a schema boundary, not a claim that every exported field is free
of user-derived information.
