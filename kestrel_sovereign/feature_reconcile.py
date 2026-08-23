"""Reconcile the host venv against the union of per-agent feature allowlists.

The per-agent ``[agents.*].features`` allowlist in ``multi_agent.toml`` decides
*which* features each agent loads, but an allowlist is a **filter**, not an
**installer**: naming a feature class an agent doesn't have installed in the
venv just makes it silently never load. This module closes that gap by deriving
what the *host* must provide from the union of all agent allowlists, and then
installing / updating exactly those packages.

Three distinct concerns (issue #1788):

1. **WHAT the host needs** = ``union(all [agents.*].features) + MANDATORY_FEATURES``,
   derived from ``multi_agent.toml`` — never hand-maintained.
2. **WHERE each comes from** = ``.kestrel-host-features.toml`` as a source map:
   one entry per package recording install mode — ``editable = "/path"`` or
   ``pypi = ">=x,<y"``.
3. **UPDATE mode** = per-feature, read from the source map: editable → ``git
   pull --ff-only`` the checkout; pypi → ``pip install --upgrade`` (respecting
   floor pins). ``--prefer-source`` / ``--prefer-pypi`` bulk-override the mode.

This module holds the **pure planning** logic (no subprocesses, no printing) so
it is trivially testable; ``cli_lifecycle.cmd_update`` owns execution + display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from packaging.utils import canonicalize_name

from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import MANDATORY_FEATURES

# The core distribution. Features whose registry ``package`` is this are bundled
# into kestrel-sovereign itself (step 1 of ``kestrel update`` pulls + reinstalls
# it) — reconcile must NOT try to pip-install them as separate extensions.
CORE_DISTRIBUTION = "kestrel-sovereign"


def canonical_package(name: str) -> str:
    """The one spelling of a distribution name this module keys everything on.

    Three sources name the same distribution three ways: the source map uses
    whatever the operator typed, the static registry uses hyphens, and live
    metadata (``ep.dist.name``) uses whatever the package's own build wrote —
    ``kestrel_feature_voice`` is a legal ``Name:`` for the project the registry
    calls ``kestrel-feature-voice``. PEP 503 says those are one distribution, so
    the reconcile pipeline resolves each to this form before it becomes a dict
    key; comparing raw spellings instead makes a declared source or pin
    invisible, and the package is then reconciled as if unmanaged (issue #2949).

    Safe as an install target too: pip and uv accept the canonical name, and
    ``importlib.metadata`` normalizes before it looks a distribution up.
    """
    return str(canonicalize_name(name))


@dataclass
class SourceEntry:
    """A resolved source-map entry: where a package comes from + how to update.

    ``editable`` and ``pypi`` are mutually exclusive (enforced by the manifest
    loader). ``pypi`` is a PEP 440 version spec; the empty string means "any
    version / latest". When neither is set the entry is legacy "install latest
    from PyPI" — preserved for back-compat with pre-#1788 manifests.
    """

    package: str
    editable: Optional[str] = None
    pypi: Optional[str] = None
    extras: List[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "editable" if self.editable else "pypi"


@dataclass
class ReconcileAction:
    """One planned reconcile operation for a single package."""

    package: str
    op: str  # "install" | "update" | "present"
    mode: str  # "editable" | "pypi"
    source: Optional[str]  # checkout path (editable) or version spec (pypi)
    required_by: List[str] = field(default_factory=list)
    current_version: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    # Editable only: the venv must be (re)linked via ``pip install -e`` because
    # the package is absent OR currently linked to a different checkout (or to a
    # non-editable PyPI build). A git pull alone would leave the venv stale.
    relink: bool = False
    # PyPI only: the source map declared where this package comes from, so the
    # executor must NOT substitute the registry's git URL on failure. That URL
    # installs repo HEAD — a different source, and an unpinned one — which would
    # move the package outside a declared window (``pypi = ">=x,<y"``) or off
    # the declared index entirely (``pypi = ""``, which pins no version but does
    # name a source). Only the legacy entry that declares neither may fall back.
    source_declared: bool = False
    # PyPI only: the package is CURRENTLY installed editable but the source map
    # now declares PyPI, so the executor must force a reinstall to replace the
    # editable link with the wheel — a plain ``--upgrade`` is a no-op when the
    # editable version already satisfies the spec, leaving it linked. The
    # executor scopes that reinstall to THIS package; a blanket
    # ``--force-reinstall`` would cascade into core (issue #2949).
    force_reinstall: bool = False
    note: str = ""


#: Fields of :class:`Provenance` that are NOT part of the source's identity.
#: Everything else is, automatically — see :attr:`Provenance.source_id`. Adding
#: a name here removes it from every source comparison in the guard, so it wants
#: a reason next to it.
_PROVENANCE_NON_IDENTITY = frozenset({
    # Whether the metadata could be read, not which source it names.
    "known",
})


@dataclass(frozen=True)
class Provenance:
    """Where a distribution came from, per PEP 610 — including "we can't tell".

    A distribution has exactly three provenances, and code that models only two
    keeps getting this wrong:

      * ``from_index``  — no ``direct_url.json``. The only positive evidence of
        an index resolution.
      * a direct URL    — VCS ref, local path, remote archive, editable
        checkout. The file names the source.
      * **unknown**     — the metadata exists but could not be read, parsed, or
        made sense of.

    The third is why this is a type rather than a tuple. It was previously a
    ``known`` flag every caller had to remember to consult, and within one
    review round two callers didn't: one read unknown as an index install, the
    other built an *unguarded* policy from it. A flag beside the data is the
    same shape as the bug this whole change exists to fix — a fact that must be
    checked at N sites, where any site can forget. Ask :attr:`is_from_index` or
    :attr:`editable_path` instead; both already account for unknown.
    """

    url: Optional[str] = None
    editable: bool = False
    known: bool = True
    #: The rest of the PEP 610 identity. ``url`` alone is not the source: two
    #: commits of one repository share a url, and two archives of one path share
    #: a url, so comparing urls calls a real replacement "no change". The
    #: revision lives in ``vcs_info``, the hash in ``archive_info``, and the
    #: subdirectory beside them — all outside ``url``, and all part of *which*
    #: source this is.
    revision: Optional[str] = None
    subdirectory: Optional[str] = None
    archive_hash: Optional[str] = None
    #: PEP 610 keeps the VCS KIND in ``vcs_info.vcs``, separate from ``url`` —
    #: the url is the bare transport address. The same address served by two
    #: different VCS at the same revision string is two different sources.
    vcs: Optional[str] = None

    @classmethod
    def unknown(cls) -> "Provenance":
        """Metadata that exists but would not read. Nothing may be inferred."""
        return cls(url=None, editable=False, known=False)

    @classmethod
    def from_index_install(cls) -> "Provenance":
        """No ``direct_url.json``: resolved from a package index."""
        return cls(url=None, editable=False, known=True)

    @classmethod
    def direct(
        cls,
        url: str,
        *,
        editable: bool = False,
        revision: Optional[str] = None,
        subdirectory: Optional[str] = None,
        archive_hash: Optional[str] = None,
        vcs: Optional[str] = None,
    ) -> "Provenance":
        """A source named directly — checkout, VCS ref, path, archive."""
        return cls(
            url=url, editable=editable, known=True, revision=revision,
            subdirectory=subdirectory, archive_hash=archive_hash, vcs=vcs,
        )

    @property
    def source_id(self) -> tuple:
        """The complete identity of this source, for comparison.

        Everything that distinguishes one direct source from another. Compare
        this, never ``url`` alone — a url match between two different commits is
        exactly the false "no change" a version pin already fails at, one field
        further in.

        ``editable`` belongs here as much as the revision does. The same path
        installed with and without ``-e`` is the same url at the same version
        but a materially different install: one is a copy, the other is a live
        checkout whose contents change under the running host.

        **Derived from the dataclass, never hand-listed.** Every hand-written
        version of this tuple has been incomplete — first the revision, hash and
        subdirectory, then ``editable`` — each time while its docstring claimed
        completeness. A list a human must remember to extend is a list that
        drifts from the thing it describes, and here the cost of drift is a real
        replacement reported as no change.

        So a new field on :class:`Provenance` joins the identity automatically.
        ``known`` is excluded deliberately and explicitly: it records whether we
        could READ the source, not which source it is. Anything else must opt
        out here, in the open, rather than be forgotten into invisibility.
        """
        import dataclasses

        return tuple(
            getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in _PROVENANCE_NON_IDENTITY
        )

    @property
    def is_from_index(self) -> bool:
        """Did this come from an index? Unknown answers **no**.

        Asked to decide whether a declared ``pypi`` source is satisfied, and
        "the metadata would not read" is not a yes. Failing closed costs at most
        a reinstall nobody needed; failing open lets the wrong source pass,
        which is the defect.
        """
        return self.known and self.url is None

    @property
    def editable_path(self) -> Optional[str]:
        """The checkout this is linked to, or None. Unknown is never a path."""
        return self.url if (self.known and self.editable) else None

    def describe(self) -> str:
        if self.editable_path:
            return f"editable → {self.editable_path}"
        if not self.known:
            return "unknown source (direct_url.json unreadable)"
        if self.url:
            # Name the revision when there is one: "the same repo" is not the
            # same source, and an operator comparing before/after needs to see
            # which part moved.
            extra = "".join(
                f" ({label} {value})"
                for label, value in (
                    ("vcs", self.vcs),
                    ("rev", self.revision),
                    ("subdir", self.subdirectory),
                    ("hash", self.archive_hash),
                )
                if value
            )
            return f"non-editable direct URL → {self.url}{extra}"
        return "non-editable (index wheel in site-packages)"


@dataclass
class CoreInstallShape:
    """How the core distribution is installed in this venv, right now.

    ``editable_path`` is the checkout recorded in PEP 660 ``direct_url.json``
    (None for a wheel copy in ``site-packages``). Captured *before* a batch of
    feature installs and compared *after* so a silent editable→wheel swap
    becomes a named failure rather than an invisible divergence (issue #2949).

    ``provenance`` is the whole of "where core came from", including the case
    where that cannot be determined. It is a :class:`Provenance` rather than
    loose fields precisely so no caller can read two thirds of it and miss the
    third — the mistake that let unknown metadata pass as an index install.
    """

    version: Optional[str] = None
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def editable_path(self) -> Optional[str]:
        return self.provenance.editable_path

    @property
    def direct_url(self) -> Optional[str]:
        return self.provenance.url

    @property
    def provenance_known(self) -> bool:
        return self.provenance.known

    @property
    def is_editable(self) -> bool:
        return bool(self.editable_path)

    @property
    def from_index(self) -> bool:
        """Was core resolved from a package index (rather than a direct URL)?

        Delegates, so this cannot answer differently from every other place the
        same question is asked. Unknown provenance answers False.
        """
        return self.provenance.is_from_index

    def describe(self) -> str:
        # A missing version is not evidence of a missing install: METADATA can
        # omit `Version` while `direct_url.json` proves the distribution is
        # right there. Saying "not installed" over provenance that names a
        # source is a diagnostic that contradicts itself.
        if self.version:
            state = self.version
        elif self.provenance.url or not self.provenance.known:
            state = "version unavailable"
        else:
            state = "not installed"
        return f"{self.provenance.describe()} ({state})"


@dataclass
class CoreSourcePolicy:
    """Where the core distribution must come from, for a batch of installs.

    At most one of the three is set:

      * ``editable`` — the checkout core must stay PEP 660-linked to.
      * ``pypi`` — the operator deliberately declared core comes from the index;
        the value is the declared PEP 440 spec (``""`` means "any version").
      * ``hold_version`` — nothing was declared and core did not come from an
        index: it is either a known direct URL (VCS, path, archive) or metadata
        that would not read. Nobody said where core *should* come from, so no
        source can be asserted — but the batch can refuse to let a feature move
        it. The installed version is pinned and any change is reported.
        ``hold_provenance`` records the provenance when it was readable, so the
        message an operator reads says which of the two cases they are in — and
        so the check compares the WHOLE source identity, not just a url.

    None set means core is a known, undeclared index wheel: nothing to protect,
    so the batch imposes no policy. That conclusion requires *knowing* core came
    from an index — reaching it from unreadable metadata is how a batch ran
    entirely unguarded over a core that might have been an editable link.
    """

    editable: Optional[str] = None
    pypi: Optional[str] = None
    hold_version: Optional[str] = None
    #: The provenance being held, when it was readable. Carries the WHOLE PEP
    #: 610 identity, not just a url — see :attr:`Provenance.source_id`.
    hold_provenance: Optional["Provenance"] = None

    @property
    def guarded(self) -> bool:
        """Does this policy constrain and verify anything at all?

        ``pypi is not None`` rather than a truth test: ``pypi = ""`` is an
        operator DECLARATION ("core comes from the index, any version"), and an
        empty version window is not an empty policy — the source half still has
        to hold. Reading it as unguarded let an editable (or absent) core pass
        a manifest that said the opposite (issue #2949).
        """
        return (
            bool(self.editable)
            or self.pypi is not None
            or self.hold_version is not None
            or self.hold_provenance is not None
        )

    @property
    def source_is_verifiable(self) -> bool:
        """Can this policy say where core *should* come from?

        False for ``hold_version``: it exists precisely because that is unknown,
        so a drift under it can be reported but never automatically repaired —
        there is no declared source to reinstall from, and guessing one is how
        an operator's core gets retargeted.
        """
        return self.hold_version is None and self.hold_provenance is None

    def describe_expected(self) -> str:
        if self.editable:
            return f"editable → {self.editable}"
        if self.pypi is not None:
            return f"{CORE_DISTRIBUTION}{self.pypi} from the index (non-editable)"
        if self.hold_version is not None or self.hold_provenance is not None:
            held = self.hold_provenance
            # The version half may be absent — METADATA can omit `Version` —
            # and reporting "unconstrained" for a policy that DID reject the
            # change tells the operator the opposite of what happened.
            at = (
                f"unchanged at {self.hold_version}"
                if self.hold_version is not None
                else "unchanged (installed version unavailable)"
            )
            if held is not None and held.known:
                return (
                    f"{at} ({held.describe()}, not an index — no declared "
                    "source to restore from)"
                )
            return f"{at} (source unverifiable — direct_url.json unreadable)"
        return "unconstrained"


def resolve_core_policy(
    source_index: Dict[str, SourceEntry],
    shape: "CoreInstallShape",
) -> CoreSourcePolicy:
    """Decide the :class:`CoreSourcePolicy` for a batch of feature installs.

    Every ``kestrel-feature-*`` depends on ``kestrel-sovereign``, so a feature
    install can resolve core from the index and replace whatever is installed.
    Guarding requires knowing where core is *supposed* to come from:

      1. An explicit ``kestrel-sovereign`` entry in the source map wins.
         ``editable`` names the checkout; ``pypi`` declares core is deliberately
         a wheel — and a non-empty spec is still a declaration the batch must
         honour, so a feature cannot drag core outside the operator's window
         (issue #2949).
      2. Otherwise the venv's own live editable link — captured *before* the
         installs run. When core is already an undeclared wheel there is no link
         to protect and no editable install is invented for the operator.
      3. When nothing is declared and core's provenance is UNREADABLE, there is
         no safe conclusion available: it may be an editable link, it may be a
         wheel. Hold the installed version instead of running unguarded.

    Takes the whole *shape* rather than a pre-extracted editable path so rule 3
    is reachable at all. Passing only ``editable_path`` threw away the one fact
    that distinguishes "core is a plain wheel, nothing to protect" from "we
    could not tell what core is" — and both arrived here as ``None``, so a
    damaged venv got no guard whatsoever.
    """
    src = source_index.get(CORE_DISTRIBUTION)
    if src is not None:
        if src.editable:
            return CoreSourcePolicy(editable=src.editable)
        if src.pypi is not None:
            return CoreSourcePolicy(pypi=src.pypi)
    if shape.editable_path:
        return CoreSourcePolicy(editable=shape.editable_path)
    if not shape.from_index:
        # Core did not come from an index, and nothing declared where it should
        # come from. One rule covers both remaining cases, because the relevant
        # fact is the same: this core was NOT resolved from an index, so it is
        # not the "plain wheel with nothing to protect" that ends this function.
        #
        #   * a known direct URL (VCS ref, local path, archive) — as deliberate
        #     an install as an editable link, and just as replaceable by a
        #     feature's index resolution;
        #   * unreadable provenance — it might be either, so assume the one that
        #     needs protecting.
        #
        # Splitting these on `provenance_known` guarded the second and let the
        # first fall through unguarded, which is the same fail-open one branch
        # further along.
        # `hold_version` may be None: METADATA can omit or empty `Version`, and
        # requiring one to guard at all meant a damaged direct-URL core fell
        # through to NO policy — unguarded, despite positive evidence it did not
        # come from an index. Detection does not need a version; only the
        # version PIN does, and losing the pin is not a reason to lose the check
        # as well.
        return CoreSourcePolicy(
            hold_version=shape.version or None, hold_provenance=shape.provenance,
        )
    return CoreSourcePolicy()


def core_install_constraints(
    shape: "CoreInstallShape", policy: "CoreSourcePolicy",
) -> List[str]:
    """Constraint lines that hold core inside *policy* for one install.

    ``uv pip`` is uv's *pip-compatible* layer: it resolves against a bare
    virtualenv and never reads ``uv.lock`` or the project config, so it cannot
    know the root is declared ``source = { editable = "." }``. Left unconstrained
    it treats ``kestrel-sovereign`` as an ordinary dependency and, whenever the
    installed version fails a feature's pin, fetches a wheel from the index —
    silently replacing the editable link (issue #2949).

    * Editable policy → ``kestrel-sovereign==<installed version>``. That bounds
      core's VERSION: either the feature's requirement is satisfiable by the
      checkout (install proceeds) or the resolve fails loudly with the real
      version skew. It does NOT bound core's SOURCE — the index publishes the
      same version the checkout builds, and such a wheel satisfies the pin
      exactly, so a reinstall of core still swaps the link for a copy. Holding
      the source is a separate job, done by scoping any force-reinstall to the
      one package being installed (``_extension_install_run(reinstall=...)``)
      and by verifying the shape afterwards. *shape* must be the state core is
      in when the install runs — re-snapshot after moving core, or the batch
      pins a version that is no longer installed.
    * PyPI policy → the operator's own declared spec, verbatim. Core may move
      inside the declared window (that is what declaring a range means) but a
      feature cannot drag it outside.

    Returns an empty list when there is nothing to hold.
    """
    if policy.editable:
        return [f"{CORE_DISTRIBUTION}=={shape.version}"] if shape.version else []
    if policy.pypi:
        if not spec_is_valid(policy.pypi):
            # `<pkg><garbage>` is not a constraint on <pkg> — for `banana` it
            # renders the package NAME `kestrel-sovereignbanana`, pinning
            # something that does not exist and leaving core free. Emit nothing
            # rather than a line that looks like protection and is not.
            return []
        return [f"{CORE_DISTRIBUTION}{policy.pypi}"]
    if policy.hold_version is not None:
        # Source unverifiable, so pin the version. That is the mechanism a
        # feature actually uses to move core — the resolver only reaches for the
        # index when the installed version fails a pin — so pinning it is real
        # protection even without knowing the source.
        return [f"{CORE_DISTRIBUTION}=={policy.hold_version}"]
    # A declared-but-empty spec (``pypi = ""``) has no version to constrain, so
    # there is no line to write — a bare package name would constrain nothing
    # and read as a pin in every message that quotes it. The policy is still
    # guarded (:attr:`CoreSourcePolicy.guarded`); its half is the SHAPE check,
    # which does not travel in a constraints file.
    return []


def version_is_valid(version: Optional[str]) -> bool:
    """Is *version* a parseable PEP 440 version?

    Asked separately from the spec because BOTH sides of the comparison can be
    unevaluable, and only checking one leaves the other failing open — which is
    how a malformed installed version came to satisfy every declared window.
    """
    if not version:
        return False
    try:
        from packaging.version import Version

        Version(version)
        return True
    except Exception:  # noqa: BLE001
        return False


def requested_extras(package_spec: str) -> Tuple[str, ...]:
    """The extras named in a spec like ``pkg[voice,web]>=1.2``."""
    if "[" not in package_spec or "]" not in package_spec:
        return ()
    inside = package_spec.split("[", 1)[1].split("]", 1)[0]
    return tuple(part.strip() for part in inside.split(",") if part.strip())


def requirement_applies(req, extras: Tuple[str, ...]) -> bool:
    """Whether *req* is active for this interpreter and these extras.

    An unmarked requirement always applies. A marked one applies if it
    evaluates true in the base environment or under any extra the caller
    requested — `packaging` evaluates ``extra == "x"`` to False when no extra is
    supplied, so the base environment alone would silently drop every
    extra-gated requirement.
    """
    if req.marker is None:
        return True
    for env in ({}, *({"extra": extra} for extra in extras)):
        try:
            if req.marker.evaluate(env):
                return True
        except Exception:  # noqa: BLE001
            # An undefined marker name tells us nothing either way; a later
            # environment may still resolve it.
            continue
    return False


@dataclass(frozen=True)
class UnmetRequirement:
    """One requirement of a distribution that this venv does not satisfy.

    Carries the requirement's CANONICAL name rather than leaving callers to
    read it back out of ``detail``: a name in metadata keeps whatever spelling
    it was written with, so ``Kestrel_Sovereign>=0.53`` and
    ``kestrel-sovereign-sdk`` are a miss and a false match respectively for
    anything matching on the rendered text.

    ``certain`` is False when the comparison could not actually be made — a
    malformed installed version, or a specifier that will not parse. That is a
    different fact from "this requirement is unmet", and the two callers want
    opposite things from it: the manifest gate retries (a needless reinstall
    costs a run), while the install endpoint does not upgrade a successful
    response to an error over metadata it could not read.
    """

    name: str
    detail: str
    certain: bool = True


def unsatisfied_requirements(
    dist_name: str, extras: Tuple[str, ...], requires, installed_version,
) -> Tuple[UnmetRequirement, ...]:
    """Every ACTIVE requirement of *dist_name* this venv does not satisfy.

    The other half of the question ``present`` has to answer. "Is it the
    declared version from the declared source" says where a distribution came
    from; this says whether the venv can actually load it. A package can be
    exactly what the manifest asked for and still be unusable because something
    it declares is missing or too old (issue #3080).

    Read from the INSTALLED metadata, because the requirement that matters is
    the one the artifact on disk declares — not the one the install command
    asked for, and not the index's idea of that version.

    Scoped deliberately to ONE distribution's own requirements rather than to
    the environment. ``feature sync``'s claim is that the venv matches the
    manifest, so it has no authority over a conflict between packages the
    manifest never names — and measured on a live host, an environment-wide
    check reports seven such conflicts among observability transitives and
    would refuse every update on a host that is working perfectly well.

    *requires* and *installed_version* are the venv readers, passed in so this
    stays a pure function of what they report.
    """
    from packaging.requirements import Requirement

    unmet: List[UnmetRequirement] = []
    for raw in requires(dist_name) or ():
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001 - unparseable metadata tells us nothing
            continue
        if not requirement_applies(req, extras):
            continue
        name = canonical_package(req.name)
        have = installed_version(req.name)
        if have is None:
            unmet.append(UnmetRequirement(
                name,
                f"{dist_name} requires {req.name}{req.specifier}, which is "
                "not installed",
            ))
            continue
        spec = str(req.specifier)
        if not spec:
            continue
        # `version_satisfies` fails OPEN by design — it was written for
        # reporting, where an unevaluable comparison must not manufacture
        # drift. A gate reading that as "satisfied" is the same failure one
        # layer up, so the two sides are asked separately and an unevaluable
        # comparison is reported as unmet-but-uncertain.
        evaluable = version_is_valid(have) and spec_is_valid(spec)
        if not evaluable:
            unmet.append(UnmetRequirement(
                name,
                f"{dist_name} requires {req.name}{spec}, and the installed "
                f"{have!r} cannot be compared against it",
                certain=False,
            ))
            continue
        if not version_satisfies(have, spec):
            unmet.append(UnmetRequirement(
                name,
                f"{dist_name} requires {req.name}{spec}, but {have} is installed",
            ))
    return tuple(unmet)


def spec_is_valid(spec: Optional[str]) -> bool:
    """Is *spec* a USABLE PEP 440 specifier? Empty counts as valid ("any").

    Stricter than "does packaging parse it", because parsing is not the bar the
    guard needs. ``===`` parses — it is arbitrary-equality carrying an EMPTY
    version — and then matches nothing at all. As a declared core window that is
    not a silent failure but a wedge: no version can ever satisfy it, so the
    guard reports drift forever and repairs forever. Rendered into a constraint
    it becomes ``kestrel-sovereign===``, which the installer rejects outright.

    A clause with no version to compare against is a typo, and it is far cheaper
    to say so at the manifest than to leave an operator with a core that can
    never conform.
    """
    if not spec:
        return True
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet

        parsed = SpecifierSet(spec)
        # `SpecifierSet` silently drops empty clauses, so ",", ",>=1" and
        # ">=1,,<2" all "parse" — and then the string we RENDER from them is a
        # requirement pip and uv both reject. Validating the rendered form is
        # the only check whose bar is the installer's bar; anything lower lets a
        # manifest typo through here and fails it later, far from the line that
        # caused it.
        Requirement(f"{CORE_DISTRIBUTION}{spec}")
    except Exception:  # noqa: BLE001
        return False
    return all(str(clause.version).strip() for clause in parsed)


def version_satisfies(version: Optional[str], spec: Optional[str]) -> bool:
    """Does *version* satisfy the PEP 440 *spec* (e.g. ``>=0.3,<0.4``)?

    An empty spec means "any version". A missing version never satisfies a
    non-empty spec. When the spec can't be evaluated (malformed value) we
    conservatively return True rather than manufacture a violation — safe for a
    *reporting* caller, which is what this was written for.

    It is NOT safe for the core guard, and that caller must not use it alone:
    "I could not evaluate the rule" reading as "the rule is met" is the
    fail-open this whole change exists to remove. :func:`core_install_matches`
    checks :func:`spec_is_valid` first for exactly that reason, and the manifest
    loader rejects a malformed spec before it ever gets here.
    """
    if not spec:
        return True
    if not version:
        return False
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        return Version(version) in SpecifierSet(spec)
    except Exception:  # noqa: BLE001
        return True


def core_install_matches(shape: "CoreInstallShape", policy: "CoreSourcePolicy") -> bool:
    """Is core installed the way *policy* requires?

    Editable policy: still linked to the declared checkout. Version drift on an
    intact link is normal (the checkout can be reinstalled) and is not a
    mismatch. PyPI policy: installed, not editable, *and* inside the declared
    window — a feature that pulled core past ``<0.53`` violates the manifest
    just as loudly as one that dropped an editable link.

    ``pypi = ""`` is asked the same questions bar the window: it declares the
    SOURCE and pins no version, so an editable or absent core still fails it.

    The PyPI branch asks for ``from_index`` rather than merely ``not
    is_editable``. Those are not the same question: a VCS, local-path or remote
    -archive install is non-editable too, and would otherwise satisfy a source
    declaration it plainly violates. Since ``pypi = ""`` exists precisely to
    declare a SOURCE and no version, testing the version-shaped proxy left it
    unable to tell sources apart at all.
    """
    if policy.editable:
        return _same_path(shape.editable_path, policy.editable)
    if policy.pypi is not None:
        if not spec_is_valid(policy.pypi):
            # A rule we cannot evaluate is not a rule that is met. The loader
            # rejects this shape, so reaching here means the policy came from
            # somewhere that skipped validation — fail closed rather than let
            # `version_satisfies`'s reporting-safe True stand in for a verdict.
            return False
        if policy.pypi and not version_is_valid(shape.version):
            # The other half of the same comparison. `Version(version)` raises on
            # malformed metadata and `version_satisfies` converts that into True,
            # so a core whose recorded version is garbage sat "inside" every
            # declared window and was never repaired. Validating only the spec
            # left this side open — both operands have to be evaluable before a
            # verdict about them means anything.
            return False
        return (
            shape.version is not None
            and not shape.is_editable
            and shape.from_index
            and version_satisfies(shape.version, policy.pypi)
        )
    if policy.hold_version is not None or policy.hold_provenance is not None:
        if policy.hold_version is not None and shape.version != policy.hold_version:
            return False
        held = policy.hold_provenance
        if held is None:
            return True
        if held.known:
            # The source WAS readable when the batch started, so it can be
            # checked — and must be. A version pin cannot see a same-version
            # swap: replacing a git-installed core with the index wheel that
            # publishes that same version passes every version test there is.
            # That is the exact defect this whole change exists to fix, and
            # asserting only the version here would have reintroduced it inside
            # the policy written to prevent it.
            return shape.provenance.source_id == held.source_id
        # Genuinely unknown source. Version stability is all that can be
        # asserted about WHERE it came from — but provenance becoming READABLE
        # is itself an observable change, and the overwhelmingly likely cause is
        # that something reinstalled core (a fresh install writes fresh
        # metadata). Accepting it because the version matched reopened exactly
        # the silent same-version replacement this guard exists to catch: an
        # unreadable editable link swapped for the index wheel publishing that
        # same version read as no drift.
        #
        # So the shape must still be unreadable. A transiently-unreadable file
        # that becomes readable is reported as drift, which is the safe
        # direction: the cost is a repair nobody needed, and the alternative is
        # missing a real replacement.
        return not shape.provenance.known
    return True


def describe_core_change(
    before: "CoreInstallShape",
    after: "CoreInstallShape",
    policy: "CoreSourcePolicy",
) -> Optional[str]:
    """Describe how core's install drifted from *policy*, or None if it matches."""
    if core_install_matches(after, policy):
        return None
    return (
        f"before: {before.describe()}\n"
        f"after:  {after.describe()}\n"
        f"expected: {policy.describe_expected()}"
    )


def compute_required_classes(multi_agent) -> Tuple[set, List[str]]:
    """Derive the set of feature classes the host must provide.

    ``required = union(each LOCAL agent's explicit [agents.*].features)
    + MANDATORY_FEATURES``.

    An agent with ``features = None`` (no allowlist) loads *every* installed
    feature and so imposes no *specific* requirement — the host cannot install
    "everything". Such agents are returned separately so the caller can note
    that the reconcile only guarantees the explicitly-named union.

    Returns ``(required_classes, load_all_agent_names)``.
    """
    required = set(MANDATORY_FEATURES)
    load_all: List[str] = []
    for name, cfg in multi_agent.get_local_agents().items():
        if cfg.features is None:
            load_all.append(name)
            continue
        required.update(cfg.features)
    return required, sorted(load_all)


def resolve_packages(
    required_classes,
    registry,
    entrypoint_dists: Optional[Dict[str, str]] = None,
    local_core_classes: Optional[set] = None,
) -> Tuple[Dict, Dict, List[str]]:
    """Resolve feature *class* names to *packages*.

    Allowlists name classes; install works on packages. Resolution consults, in
    order, the most authoritative source first — **live discovery is the source
    of truth**, the static catalog is only a fallback / metadata source. The
    order also mirrors the runtime loader's precedence (local features win over
    duplicate-named entry points):

      1. ``local_core_classes`` — in-tree bundled Feature classes. Checked
         first: the loader discovers them before entry-point features and skips
         duplicates, so a bundled class is what the agent actually loads.
         Provisioned by ``kestrel update`` step 1 (core reinstall) — no
         separate extension install.
      2. ``entrypoint_dists`` — installed external feature packages, mapped
         class → distribution from live entry-point metadata. The truth for
         what's actually loadable from a pip package right now.
      3. The static registry catalog — an external package that's *catalogued*
         but not yet installed (the legitimate "install the missing feature"
         case).
      4. ``MANDATORY_FEATURES`` — guaranteed by core even if uncatalogued.

    Anything that matches none of these is a class an agent allowlist names but
    the host cannot provide — the motivating silent-no-load bug — returned in
    ``unresolved`` as a hard error (no blind fallback).

    Returns ``(pkg_infos, class_to_pkg, unresolved)`` where ``pkg_infos`` is
    ``{package: FeaturePackageInfo}`` for every required, non-core, installable
    package. Package keys are :func:`canonical_package` form regardless of which
    of the three sources above named them, so the source index (also keyed that
    way, via :func:`package_for_label`) lines up.
    """
    entrypoint_dists = entrypoint_dists or {}
    local_core_classes = local_core_classes or set()

    reg_by_class: Dict[str, "FeaturePackageInfo"] = {}
    reg_by_package: Dict[str, "FeaturePackageInfo"] = {}
    for info in registry.values():
        if info.package:
            reg_by_package.setdefault(canonical_package(info.package), info)
        for cls in info.features:
            reg_by_class.setdefault(cls, info)

    pkg_infos: Dict = {}
    class_to_pkg: Dict[str, str] = {}
    unresolved: List[str] = []

    for cls in sorted(required_classes):
        # 1. Bundled in-tree class — no separate extension to provision. Checked
        #    FIRST to match the runtime loader, which discovers local features
        #    before entry-point ones and skips a duplicate-named entry point;
        #    so when a class is both bundled AND shipped by an installed
        #    package, the agent loads the bundled one and reconcile must not
        #    touch the external package (codex round 6 P2).
        if cls in local_core_classes:
            class_to_pkg[cls] = CORE_DISTRIBUTION
            continue

        # 2. Installed external package (live entry-point truth).
        #    ``ep.dist.name`` is whatever the package's own METADATA spells —
        #    a project built as ``kestrel_feature_voice`` reports the underscore
        #    form. Canonicalize it so the identity lines up with the source
        #    index and the registry, which spell it with hyphens: a mismatch
        #    silently drops the entry's declared source/pin (issue #2949).
        dist = entrypoint_dists.get(cls)
        if dist:
            dist = canonical_package(dist)
            class_to_pkg[cls] = dist
            if dist != CORE_DISTRIBUTION:
                # Prefer registry metadata (git URL, extras) when catalogued;
                # otherwise synthesize a minimal info from the live dist.
                pkg_infos[dist] = reg_by_package.get(dist) or FeaturePackageInfo(
                    name=dist, package=dist, git="", features=[cls],
                    description="", core=False,
                )
            continue

        # 3. Catalogued external package (maybe not yet installed → install).
        info = reg_by_class.get(cls)
        if info is not None:
            package = canonical_package(info.package) if info.package else ""
            class_to_pkg[cls] = package
            if package and package != CORE_DISTRIBUTION:
                pkg_infos[package] = info
            continue

        # 4. Mandatory features are guaranteed by core even when uncatalogued.
        if cls in MANDATORY_FEATURES:
            continue

        # 5. Named by an allowlist but the host can't provide it — the bug.
        unresolved.append(cls)

    return pkg_infos, class_to_pkg, unresolved


def build_source_index(manifest_entries, registry) -> Dict[str, SourceEntry]:
    """Map manifest entries to ``{package: SourceEntry}``.

    Manifest entries name a registry short-name ("voice") or a raw dist name
    ("kestrel-feature-voice"); both resolve to the package (dist) name so the
    index keys line up with :func:`resolve_packages` — and with the target the
    CLI installs, which is why both go through :func:`package_for_label`.
    """
    index: Dict[str, SourceEntry] = {}
    for entry in manifest_entries:
        package = package_for_label(entry["name"], registry)
        index[package] = SourceEntry(
            package=package,
            editable=entry.get("editable"),
            pypi=entry.get("pypi"),
            extras=list(entry.get("extras", []) or []),
        )
    return index


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    """True only when both are non-empty paths that resolve to the same place."""
    if not a or not b:
        return False
    try:
        from pathlib import Path as _Path

        return _Path(a).expanduser().resolve() == _Path(b).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return a == b


def _pypi_requirement(package: str, spec: Optional[str], extras: List[str]) -> str:
    """Render a pip requirement with extras BEFORE the version spec.

    pip requires ``pkg[extra]>=x,<y`` — ``pkg>=x,<y[extra]`` is rejected. Used
    for the PyPI source form so a pinned-with-extras feature installs cleanly.
    """
    base = f"{package}[{','.join(extras)}]" if extras else package
    return f"{base}{spec}" if spec else base


def resolve_registry_name(name: str, registry: dict) -> Optional[str]:
    """Resolve a user-provided name to a registry short name.

    Accepts a registry short name ("voice"), registered distribution name
    ("kestrel-feature-voice"), or feature/provider class name
    ("VoiceFeature"). Package-style names use Python's normalized-name rules,
    so case and runs of ``-``, ``_``, or ``.`` are equivalent.

    Lives here rather than in the CLI because the reconcile planner resolves the
    same names (see :func:`package_for_label`) and a second implementation is
    how the two answers drift apart. ``cli_features._resolve_feature_name`` is
    the CLI's name for this function.
    """
    # Preserve the cheapest and most common exact registry-key lookup.
    if name in registry:
        return name

    normalized = canonicalize_name(name)
    for pkg_name, info in registry.items():
        if (
            canonicalize_name(pkg_name) == normalized
            or canonicalize_name(info.package) == normalized
        ):
            return pkg_name

    # Feature lifecycle or provider implementation class-name match.
    for pkg_name, info in registry.items():
        components = getattr(info, "runtime_components", info.features)
        if name in components:
            return pkg_name

    return None


def package_for_label(label: str, registry) -> str:
    """The distribution (dist) name *label* installs — one answer per label.

    ``feature sync`` decides an entry IS core (``target == CORE_DISTRIBUTION``,
    so it is installed unconstrained and re-derives the batch pin) while
    :func:`build_source_index` decides which entry core's *policy* is read from.
    Those two questions must never get different answers, so both resolve the
    label here: registry lookup under Python's normalized-name rules, falling
    back to the normalized label itself when no registry row claims it.

    ``kestrel_sovereign`` and ``kestrel-sovereign`` are one distribution
    (PEP 503). Indexing the source policy under the raw spelling while executing
    the entry as core left the guard protecting — and restoring — a checkout the
    manifest never declared (issue #2949). The answer is always
    :func:`canonical_package` form — including when the registry supplied it —
    because :func:`resolve_packages` keys the packages this index is looked up
    with on exactly that.
    """
    key = resolve_registry_name(label, registry)
    if key and registry[key].package:
        return canonical_package(registry[key].package)
    return canonical_package(label)


def plan_reconcile(
    pkg_infos: Dict,
    source_index: Dict[str, SourceEntry],
    installed_versions: Dict[str, Optional[str]],
    editable_paths: Dict[str, Optional[str]],
    class_to_pkg: Dict[str, str],
    prefer: Optional[str] = None,
) -> Tuple[List[ReconcileAction], List[str]]:
    """Plan the reconcile for every required, installable package.

    Args:
        pkg_infos: required packages → FeaturePackageInfo (from resolve_packages).
        source_index: package → SourceEntry (from the source map).
        installed_versions: package → installed version (or None if absent).
        editable_paths: package → editable checkout path (or None) as the venv
            *currently* records it (PEP 660 ``direct_url.json``).
        class_to_pkg: class → package, to populate ``required_by``.
        prefer: ``"source"`` forces editable update mode where a checkout path
            is known; ``"pypi"`` forces pip-upgrade mode even for editable
            installs; ``None`` honours the source map.

    Returns ``(actions, no_source)`` where ``no_source`` lists required packages
    with neither a source-map entry nor a registry-resolvable install source —
    a hard error (no blind fallback).
    """
    # Invert class_to_pkg → package: [classes]
    required_by: Dict[str, List[str]] = {}
    for cls, pkg in class_to_pkg.items():
        required_by.setdefault(pkg, []).append(cls)

    actions: List[ReconcileAction] = []
    no_source: List[str] = []

    for package in sorted(pkg_infos):
        info = pkg_infos[package]
        src = source_index.get(package)
        current = installed_versions.get(package)
        detected_editable = editable_paths.get(package)

        # Decide the install/update MODE and SOURCE.
        editable_path = None
        pypi_spec = None
        extras: List[str] = []

        if src is not None:
            extras = src.extras
            if src.editable:
                editable_path = src.editable
            if src.pypi is not None:
                pypi_spec = src.pypi

        # An install already recorded as editable counts as a known checkout
        # even if the manifest didn't spell one out (keeps a hand-installed dev
        # checkout from being clobbered with a PyPI build).
        if editable_path is None and detected_editable:
            editable_path = detected_editable

        # Resolve mode, honouring the prefer override.
        if prefer == "pypi":
            mode = "pypi"
        elif prefer == "source" and editable_path:
            mode = "editable"
        elif editable_path and (src is None or src.mode == "editable"):
            mode = "editable"
        elif pypi_spec is not None or current is not None or info.package:
            mode = "pypi"
        else:
            mode = "pypi"

        relink = False
        source_declared = False
        force_reinstall = False
        op = "update" if current is not None else "install"
        if mode == "editable":
            if not editable_path:
                # prefer=source asked for editable but no checkout is known.
                no_source.append(package)
                continue
            source = editable_path
            # The venv must be re-linked via ``pip install -e`` (not just
            # git-pulled) when the package is absent, is installed from a
            # non-editable PyPI build, or is linked to a DIFFERENT checkout
            # than the source map now wants. Otherwise a git pull would leave
            # the venv pointing at the stale install (codex round 3 P2).
            relink = current is None or not _same_path(detected_editable, editable_path)
        else:
            # PyPI mode. A declared spec always wins. Otherwise we can only pip
            # from PyPI when the package has a known remote source — i.e. it is
            # catalogued in the registry (``info.git`` is set for every catalog
            # entry; synthesized live-only infos have it empty). Render extras
            # BEFORE the version spec (``pkg[extra]>=x``) — pip rejects the
            # reverse.
            # Switching a currently-editable install to PyPI requires a force
            # reinstall: a plain --upgrade is a no-op when the editable version
            # already satisfies the spec, leaving the checkout linked (codex
            # round 9 P2).
            force_reinstall = bool(detected_editable)
            if pypi_spec is not None:
                source = _pypi_requirement(package, pypi_spec, extras)
                # The entry named the index as this package's source. That holds
                # whether or not it also pinned a version: ``""`` means "any
                # version FROM THE INDEX", so the git-URL substitution below is
                # off the table either way (see ``source_declared``).
                source_declared = True
            elif info.git:
                source = _pypi_requirement(package, None, extras)
            elif current is not None:
                # Installed and loadable but no known remote source (a local /
                # private / direct-URL install with no source-map entry). Leave
                # it untouched — the allowlist requirement is already satisfied;
                # fabricating a PyPI upgrade would fail and abort the update
                # (codex round 4 P2). Add a source-map entry to manage it.
                op = "present"
                source = None
            else:
                no_source.append(package)
                continue

        actions.append(
            ReconcileAction(
                package=package,
                op=op,
                mode=mode,
                source=source,
                required_by=sorted(required_by.get(package, [])),
                current_version=current,
                extras=extras,
                relink=relink,
                source_declared=source_declared,
                force_reinstall=force_reinstall and op != "present",
            )
        )

    return actions, sorted(no_source)
