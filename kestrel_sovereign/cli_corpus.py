"""``kestrel corpus exclusions check`` -- the operator check for corpus exclusions (#3893).

Every governed corpus build reads the host's Sovereign-signed exclusion manifest
(:mod:`kestrel_sovereign.knowledge.corpus_exclusions`) and refuses without one.
This command reads it as the host's agents would, from the environment
``kestrel start`` launches them with, and verifies it against the same trust
root. It prints each agent's section with every excluded id, so an operator can
confirm that what must be held out is named in full, and says whether each
section is usable or would refuse its agent's builds.

``--agent-did`` names an agent whose section must be present. Exit status is
0 when the manifest verifies and every section, every named agent's included,
is usable; 1 otherwise. Nothing is written: the sequence a build admits is
recorded by that build.

The manifest module is imported when the command runs, so building the
``kestrel`` parser loads no signature machinery.
"""

from __future__ import annotations

import argparse
import json
import sys


def add_corpus_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "corpus", help="Governed learning corpus: inspect its exclusion manifest"
    )
    corpus_sub = parser.add_subparsers(dest="corpus_command")
    exclusions_p = corpus_sub.add_parser(
        "exclusions", help="The Sovereign-signed manifest of held-out material"
    )
    exclusions_sub = exclusions_p.add_subparsers(dest="exclusions_command")
    check_p = exclusions_sub.add_parser(
        "check",
        help="Verify the manifest corpus builds read and list what it excludes",
    )
    check_p.add_argument(
        "--agent-did",
        action="append",
        default=[],
        help="Require a usable section for this agent DID (repeatable)",
    )
    check_p.add_argument("--json", action="store_true", help="Print JSON instead of text")


def _section_report(manifest, agent_id: str) -> dict[str, object]:
    from kestrel_sovereign.knowledge.corpus_exclusions import (
        CorpusExclusionKind,
        CorpusExclusionManifestError,
    )

    section = manifest.sections.get(agent_id)
    if section is None:
        return {"agent_did": agent_id, "usable": False, "problem": "no section"}
    report: dict[str, object] = {
        "agent_did": agent_id,
        "ids": {kind.value: list(section.get(kind, ())) for kind in CorpusExclusionKind},
    }
    try:
        manifest.exclusions_for(agent_id)
    except CorpusExclusionManifestError as error:
        return {**report, "usable": False, "problem": str(error)}
    return {**report, "usable": True}


def _check(args: argparse.Namespace) -> int:
    from kestrel_sovereign.knowledge.corpus_exclusions import (
        CorpusExclusionManifestError,
        load_corpus_exclusions,
    )
    from kestrel_sovereign.paths import project_dir, spawned_agent_env

    environ = spawned_agent_env(project_dir())
    required = list(dict.fromkeys(args.agent_did))
    try:
        manifest = load_corpus_exclusions(environ=environ, agent_dids=frozenset(required))
    except CorpusExclusionManifestError as error:
        if args.json:
            print(json.dumps({"ok": False, "problem": str(error)}, indent=2))
        else:
            print(f"REFUSED: {error}", file=sys.stderr)
        return 1

    agents = sorted(set(manifest.sections).union(required))
    sections = [_section_report(manifest, agent_id) for agent_id in agents]
    ok = all(section["usable"] for section in sections)
    if args.json:
        print(json.dumps({
            "ok": ok,
            "path": manifest.path,
            "signer": manifest.signer,
            "verification": manifest.verification,
            "sequence": manifest.sequence,
            "issued_at": manifest.issued_at,
            "manifest_sha256": manifest.manifest_sha256,
            "reason": manifest.reason,
            "agents": sections,
        }, indent=2))
        return 0 if ok else 1

    print(f"Corpus exclusion manifest: {manifest.path}")
    print(f"  signer:    {manifest.signer} ({manifest.verification})")
    print(f"  sequence:  {manifest.sequence}")
    print(f"  issued at: {manifest.issued_at}")
    print(f"  digest:    {manifest.manifest_sha256}")
    if manifest.reason:
        print(f"  reason:    {manifest.reason}")
    for section in sections:
        state = "usable" if section["usable"] else f"REFUSED: {section['problem']}"
        print(f"Agent {section['agent_did']}: {state}")
        ids = section.get("ids", {})
        if ids and not any(ids.values()):
            print("  excludes nothing")
        for kind, values in ids.items():
            if values:
                print(f"  {kind} ({len(values)}):")
                for value in values:
                    print(f"    {value}")
    print("OK" if ok else "REFUSED: a corpus build for an agent above would refuse")
    return 0 if ok else 1


def run(args: argparse.Namespace) -> int:
    if (
        getattr(args, "corpus_command", None) != "exclusions"
        or getattr(args, "exclusions_command", None) != "check"
    ):
        print("usage: kestrel corpus exclusions check [--agent-did DID] [--json]",
              file=sys.stderr)
        return 2
    return _check(args)
