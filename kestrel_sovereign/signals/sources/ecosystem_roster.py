"""Durable ecosystem repo roster expansion for scheduled stale-work scans (#2269).

Scheduling ``scan_stale_work`` against an explicit repo list is brittle: a
wildcard like ``KestrelSovereignAI/kestrel-feature-*`` is not a real repo name,
so treating it literally either 404s or is silently dropped. This module turns a
declarative *roster spec* — any mix of orgs, an explicit allowlist, and repo
prefixes (including trailing-``*`` wildcards) — into a concrete, deduped list of
repositories to scan, resolved against the universe of repos actually accessible
to the agent's GitHub token.

Two boundaries are load-bearing:

* **tekspear repos are always excluded.** The roster is a *non-tekspear*
  KestrelSovereignAI ecosystem source; the exclusion is durable and hardcoded
  here rather than left to per-schedule config, so a recurring loop can never
  accidentally scan a tekspear repo (#2269 AC2).
* **inaccessible repos are explicit failures, never silent skips.** An explicit
  allowlist entry that isn't in the accessible universe — or a prefix/org that
  can't be listed because discovery failed — is reported as a ``failure`` with a
  reason, so a scan that couldn't see a repo is distinguishable from a scan that
  saw it and found nothing (#2269 AC3).

The expansion is a pure function of the spec plus the accessible-repo universe,
so it is fully testable without network access; the caller (the Talon
coordinator's ``scan_stale_work``) is responsible for fetching that universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

# Repos whose slug contains this marker are never included in an ecosystem
# roster, regardless of how they were named (org listing, allowlist, or prefix).
TEKSPEAR_MARKER = "tekspear"


def is_wildcard(entry: str) -> bool:
    """True when ``entry`` is a glob pattern rather than a literal repo slug."""
    return "*" in entry


def wildcard_prefix(entry: str) -> str:
    """The literal prefix of a wildcard entry (everything before the first ``*``)."""
    return entry.split("*", 1)[0]


def is_tekspear_repo(repo: str) -> bool:
    """True when ``repo`` is a tekspear repo that must be excluded from a roster."""
    return TEKSPEAR_MARKER in repo.lower()


def _clean_list(value: Any) -> list[str]:
    """Normalize a str / list / None into a list of non-empty trimmed strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


@dataclass(frozen=True)
class RosterSpec:
    """A declarative roster: orgs + allowlist + prefixes, minus explicit excludes.

    ``prefixes`` collects both explicit ``repo_prefix`` entries and the literal
    prefix of any wildcard that appeared in an org/allowlist field — so a
    wildcard string is never mistaken for a literal repo name.
    """

    orgs: tuple[str, ...] = ()
    allowlist: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.orgs or self.allowlist or self.prefixes)


def parse_roster_spec(
    *,
    org: Any = None,
    repos: Any = None,
    repo: Any = None,
    repo_prefix: Any = None,
    exclude_repos: Any = None,
) -> RosterSpec:
    """Parse loose scheduler args into a :class:`RosterSpec`.

    Wildcard entries are routed to ``prefixes`` no matter which field they came
    in on, so ``repos=["KestrelSovereignAI/kestrel-feature-*"]`` expands by
    prefix instead of being looked up as a literal (non-existent) repo (#2269 AC1).
    """
    orgs: list[str] = []
    allowlist: list[str] = []
    prefixes: list[str] = []

    for entry in _clean_list(org):
        (prefixes if is_wildcard(entry) else orgs).append(
            wildcard_prefix(entry) if is_wildcard(entry) else entry
        )

    for entry in _clean_list(repos) + _clean_list(repo):
        if is_wildcard(entry):
            prefixes.append(wildcard_prefix(entry))
        else:
            allowlist.append(entry)

    for entry in _clean_list(repo_prefix):
        # A trailing (or embedded) ``*`` on a prefix is redundant but harmless.
        prefixes.append(wildcard_prefix(entry) if is_wildcard(entry) else entry)

    def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for item in items:
            if item and item not in seen:
                seen[item] = None
        return tuple(seen)

    return RosterSpec(
        orgs=_dedupe(orgs),
        allowlist=_dedupe(allowlist),
        prefixes=_dedupe(p for p in prefixes if p),
        exclude=_dedupe(_clean_list(exclude_repos)),
    )


@dataclass(frozen=True)
class RosterExpansion:
    """Concrete result of expanding a roster against the accessible universe."""

    repos: tuple[str, ...] = ()
    failures: tuple[dict[str, str], ...] = ()
    excluded: tuple[str, ...] = ()


def _org_of(slug: str) -> str:
    return slug.split("/", 1)[0] if "/" in slug else ""


def expand_roster(
    spec: RosterSpec,
    *,
    accessible_repos: Optional[Iterable[str]],
    discovery_error: Optional[str] = None,
) -> RosterExpansion:
    """Expand ``spec`` against the ``accessible_repos`` universe.

    ``accessible_repos`` is the set of repo slugs the agent's token can actually
    see (from GitHub discovery). ``discovery_error`` is set when that listing
    could not be fetched; combined with an empty universe it means *nothing*
    could be verified, so every requested target is reported as an inaccessible
    failure rather than silently yielding zero repos (#2269 AC3).

    Returns the resolved repos (tekspear- and exclude-filtered, sorted, deduped),
    the failures (inaccessible explicit repos / un-listable orgs & prefixes), and
    the repos dropped by an exclusion rule.
    """
    accessible = set(accessible_repos or [])
    discovery_failed = discovery_error is not None and not accessible

    resolved: set[str] = set()
    failures: list[dict[str, str]] = []

    reason = discovery_error or "repository listing unavailable"

    for org in spec.orgs:
        if discovery_failed:
            failures.append({"scope": org, "reason": reason})
            continue
        matched = [r for r in accessible if _org_of(r).lower() == org.lower()]
        if matched:
            resolved.update(matched)
        else:
            failures.append(
                {"scope": org, "reason": "no accessible repos found for org"}
            )

    for prefix in spec.prefixes:
        if discovery_failed:
            failures.append({"pattern": f"{prefix}*", "reason": reason})
            continue
        matched = [r for r in accessible if r.startswith(prefix)]
        if matched:
            resolved.update(matched)
        else:
            failures.append(
                {"pattern": f"{prefix}*", "reason": "no accessible repos matched prefix"}
            )

    for slug in spec.allowlist:
        if slug in accessible:
            resolved.add(slug)
        else:
            failures.append({"repo": slug, "reason": reason if discovery_failed else "inaccessible"})

    exclude_set = set(spec.exclude)
    excluded: list[str] = []
    final: list[str] = []
    for slug in sorted(resolved, key=str.lower):
        if is_tekspear_repo(slug):
            excluded.append(slug)
            continue
        if slug in exclude_set:
            excluded.append(slug)
            continue
        final.append(slug)

    return RosterExpansion(
        repos=tuple(final),
        failures=tuple(failures),
        excluded=tuple(excluded),
    )
