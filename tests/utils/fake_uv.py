"""A pretend venv + resolver, faithful to the core swap in issue #2949.

Shared by every surface that installs a feature package — ``feature sync`` /
``install`` / ``upgrade`` (``test_cli_feature_sync.py``), ``kestrel update``'s
reconcile (``test_cli_update_reconcile.py``), and the HTTP install endpoint
(``test_features_api.py``). One double, because the guard's whole claim is that
those surfaces behave identically; a per-file double is how they drift.

Substitutes at the ``subprocess.run`` seam rather than at
``_extension_install_run``, so the constraint-file plumbing itself stays under
test: delete the ``-c`` handling and the tests fail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CORE = "kestrel-sovereign"
CHECKOUT = "/src/kestrel-sovereign"

#: A dependency of core that is itself installed editable — the SDK on a dev
#: host. Core's own reinstall is the one that reaches it: ``--force-reinstall``
#: applies to every package the resolve touches, so repairing core with the
#: blanket flag drops this link for an index wheel. Modelled so "the repair
#: scopes its reinstall" is a fact about the venv rather than an assertion
#: about an argv.
SDK = "kestrel-sovereign-sdk"
SDK_CHECKOUT = "/src/kestrel-sovereign-sdk"


class UnboundedInstall(AssertionError):
    """An installer that never returns was started with ``timeout=None``.

    The honest model of a hung resolve is a subprocess that simply does not come
    back, which a test cannot sit through. So :class:`FakeUv` raises this
    instead — it is not a failure mode the production code should ever catch,
    it is the double refusing to pretend an unbounded wait terminated. A caller
    that passes a bound gets the real thing (``TimeoutExpired``).
    """


class FakeUv:
    """A venv whose state actually changes when something is installed.

    Models the two ways a feature install replaces core:

    * **version skew** — the installed core cannot satisfy a feature's
      requirement, so an *unconstrained* resolve fetches a core wheel from the
      index and drops the editable link. A constraints file holding core inside
      its declared window removes the offending versions as candidates, so the
      same resolve has no solution and fails loudly instead.
    * **reinstall cascade** — a blanket ``--force-reinstall`` reinstalls every
      resolved dependency, and core is one for every feature package. The
      version never changes, so the constraints file permits it exactly: the
      index copy lands on top of the editable link. Only scoping the reinstall
      to the feature package (``--reinstall-package``, or pip's
      ``--force-reinstall --no-deps`` pass) keeps core out of the reinstall set.
      The cascade is modelled one level further down, at :data:`SDK`, because
      installing CORE has collateral of its own.

    Installing core itself is modelled as faithfully as installing a feature:
    ``-e <path>`` links that checkout AND takes the version the checkout builds
    (``checkouts``), a spec takes the newest index version the spec allows. A
    double where editable installs never change the version cannot see a batch
    pinning the pre-switch version.

    ``honours_constraints=False`` models a path that bypasses the pin (a
    feature's own build step, a direct pip call) — prevention can't cover those,
    which is why the post-install check exists. ``repair_fails=True`` models the
    worse case: the swap happened AND core cannot be put back, which must never
    be reported as a clean install. ``repair_noops=True`` is the sneakier one:
    the repair exits 0 and changes nothing, so an exit code is not evidence the
    link came back. ``repair_hangs=True`` is the one an exit code cannot model
    at all: the repair's installer never returns, so only the caller's own
    timeout ends it (see :class:`UnboundedInstall`).

    The last two model the mirror image — an installer that ended badly *after*
    core was already home, so its exit status describes the installer and not
    the venv. ``core_write_pass_fails=True``: pip's scoped core install
    is a sequence and the first pass can already have landed core, so a failure
    in the WRITE pass is a nonzero exit over a conforming core — and one that
    stops the sequence before the pass which validates dependencies. ``core_resolve_refused=True`` is its
    opposite number: pip's LAST pass resolves the dependencies of the artifact
    the ``--no-deps`` pass installed, so a refusal there is also a nonzero exit
    over a conforming core — but this one means the host cannot load what it
    just installed, which re-reading core cannot see (issue #3047).
    ``repair_hangs_after_restore=True``:
    the write lands and the process is killed afterwards — a bound stops a hung
    installer, it does not undo what that installer had already done.
    """

    def __init__(
        self,
        *,
        core_version="0.52.0",
        core_checkout=CHECKOUT,
        sdk_checkout=SDK_CHECKOUT,
        checkouts=None,
        feature="kestrel-feature-voice",
        feature_version="0.4.0",
        feature_requires=">=0.53",
        feature_installed_requires=None,
        core_index=("0.52.0", "0.53.0"),
        honours_constraints=True,
        repair_fails=False,
        repair_noops=False,
        repair_hangs=False,
        core_write_pass_fails=False,
        core_resolve_refused=False,
        repair_hangs_after_restore=False,
        feature_install_fails=False,
        feature_install_times_out=False,
        feature_install_interrupted=False,
        repair_interrupted=False,
        direct_urls=None,
        unreadable_provenance=None,
    ):
        self.installed = {CORE: core_version}
        self.editable = {CORE: core_checkout} if core_checkout else {}
        if sdk_checkout:
            # An editable dependency OF core, not a modelled core checkout: it
            # is never an install target here, only collateral.
            self.installed[SDK] = "0.36.0"
            self.editable[SDK] = sdk_checkout
        self.checkouts = dict(checkouts or {})
        if core_checkout:
            self.checkouts.setdefault(core_checkout, core_version)
        self.feature = feature
        self.feature_version = feature_version
        #: What the INDEX artifact declares, and what a resolve reads whenever
        #: it builds its candidate from the index.
        self.feature_requires = feature_requires
        #: What the artifact currently ON DISK declares. Defaults to the index's
        #: own requirement — they differ only where a test says they do, which
        #: is the case in issue #3047: a checkout build and the wheel published
        #: at the same version are not obliged to declare the same dependencies.
        self.feature_installed_requires = feature_installed_requires or feature_requires
        self.core_index = list(core_index)
        self.honours_constraints = honours_constraints
        self.repair_fails = repair_fails
        self.repair_noops = repair_noops
        self.repair_hangs = repair_hangs
        self.core_write_pass_fails = core_write_pass_fails
        self.core_resolve_refused = core_resolve_refused
        self.repair_hangs_after_restore = repair_hangs_after_restore
        self.feature_install_fails = feature_install_fails
        self.feature_install_times_out = feature_install_times_out
        self.feature_install_interrupted = feature_install_interrupted
        self.repair_interrupted = repair_interrupted
        # Non-editable direct-URL installs, by dist: a VCS ref, local path or
        # remote archive. Distinct from `editable` (which is also a direct URL,
        # but flagged) and from absent (an index resolution).
        self.direct_urls = dict(direct_urls or {})
        # Dists whose direct_url.json exists but will not read/parse: provenance
        # UNKNOWN, which is a third state distinct from both of the above.
        self.unreadable_provenance = set(unreadable_provenance or ())
        self.commands = []
        self.pins = []  # the core pin seen per install (None = unconstrained)

    # -- venv state, read back by the code under test ------------------------

    def version(self, dist):
        import importlib.metadata as md

        if dist in self.installed:
            return self.installed[dist]
        raise md.PackageNotFoundError(dist)

    def editable_path(self, dist):
        return self.editable.get(dist)

    def direct_url_provenance(self, dist):
        """PEP 610 provenance for *dist* — a ``Provenance``.

        ``unreadable_provenance`` models the third state: metadata that exists
        but will not parse, so where the package came from is UNKNOWN. Without
        it the double could only express "index" and "direct URL", and a
        predicate that fails open on unknown would look correct in every test.

        Models what the metadata actually records, which is the distinction the
        guard now depends on: an editable install has a direct URL *and* the
        editable flag; ``direct_urls`` models a NON-editable direct install
        (VCS ref, local path, remote archive); anything else was resolved from
        an index and records no ``direct_url.json`` at all.

        Leaving this unmodelled is what made the old double unfaithful — it
        could not express "non-editable but not from the index", so the case
        that broke the ``pypi`` policy was invisible to every test.
        """
        from kestrel_sovereign.feature_reconcile import Provenance

        if dist in self.unreadable_provenance:
            return Provenance.unknown()
        editable = self.editable.get(dist)
        if editable:
            return Provenance.direct(editable, editable=True)
        url = self.direct_urls.get(dist)
        return Provenance.direct(url) if url else Provenance.from_index_install()

    # -- the resolver --------------------------------------------------------

    def run(self, cmd, capture_output=True, text=True, timeout=None):
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        self.commands.append(list(cmd))
        # Read the pin now: `_extension_install_run` deletes the file on return.
        pin = self._core_pin(cmd)
        self.pins.append(pin)

        target = str(cmd[-1])
        if self._is_core_target(cmd, target):
            return self._install_core(cmd, target, timeout)

        if self.feature_install_times_out:
            # A killed install is not a no-op: whatever pip had already written
            # stays written. Model the worst honest case — the dependency swap
            # landed before the timeout.
            self._swap_core_for_index_wheel(pin, self._resolved_feature_requires(cmd))
            raise subprocess.TimeoutExpired(cmd, timeout or 0)

        if self.feature_install_interrupted:
            # Ctrl-C. Same shape as the timeout above and for the same reason —
            # a killed installer keeps whatever it already wrote — but it
            # arrives as a BaseException that unwinds the caller instead of a
            # value it can inspect, which is how the post-check came to be
            # skipped (issue #2962).
            self._swap_core_for_index_wheel(pin, self._resolved_feature_requires(cmd))
            raise KeyboardInterrupt()

        if self._reinstalls_dependencies(cmd):
            self._reinstall_core_from_index(pin)
            self._reinstall_sdk_from_index()

        if "--no-deps" in cmd:
            # Dependencies are not resolved AT ALL: the installer looks at the
            # requested package and nothing else. No skew can surface here, and
            # nothing but the target can change — which is why the pass that
            # resolves dependencies has to run first, and why a double that
            # resolved them anyway hid the ordering bug (issue #2949).
            if self.feature_install_fails:
                return self._failed(cmd, f"x Failed to build `{self.feature}`")
            return self._install_feature(cmd)

        wanted = SpecifierSet(self._resolved_feature_requires(cmd))
        if Version(self.installed[CORE]) in wanted:
            # Core already satisfies the feature; a satisfied dependency is
            # left alone. Nothing to swap.
            if self.feature_install_fails:
                return self._failed(cmd, f"x Failed to build `{self.feature}`")
            return self._install_feature(cmd)

        candidates = [v for v in self._core_candidates(pin) if Version(v) in wanted]
        if not candidates:
            return self._failed(
                cmd,
                "x No solution found when resolving dependencies: "
                f"{self.feature}=={self.feature_version} depends on "
                f"{CORE}{self._resolved_feature_requires(cmd)}, but you require "
                f"{CORE}=={self.installed[CORE]}.",
            )

        # The swap: an index wheel replaces whatever core was. pip resolves and
        # installs dependencies BEFORE the requested package, so this lands even
        # when the feature's own build then fails.
        self.installed[CORE] = max(candidates, key=Version)
        self.editable.pop(CORE, None)
        if self.feature_install_fails:
            return self._failed(cmd, f"x Failed to build `{self.feature}`")
        return self._install_feature(cmd)

    # -- internals -----------------------------------------------------------

    def _failed(self, cmd, stderr):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=stderr)

    def _is_core_target(self, cmd, target) -> bool:
        """Is this install aimed at CORE itself, rather than a feature package?

        A spec names core outright (``kestrel-sovereign>=0.52,<0.53``); an
        editable install names a path, which is core's only when it is one of
        the modelled core checkouts. Everything else — including a git-URL
        fallback, which names neither — is a feature install.
        """
        if "-e" in cmd:
            return target in self.checkouts
        return target.startswith(CORE)

    def _install_feature(self, cmd):
        """Record the feature as installed, and drop its link if reinstalled.

        A feature already linked from a checkout is only replaced by the index
        wheel when the command actually reinstalls IT — which is the point of a
        source switch, and the reason a scoped reinstall must still reach the
        requested package.
        """
        landed = (
            self._reinstalls_package(cmd, self.feature)
            or self.installed.get(self.feature) != self.feature_version
        )
        self.installed[self.feature] = self.feature_version
        if self._reinstalls_package(cmd, self.feature):
            self.editable.pop(self.feature, None)
        if landed:
            # The index artifact is the one on disk now, so what the venv
            # declares as a dependency changed with the file that declared it.
            self.feature_installed_requires = self.feature_requires
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _resolved_feature_requires(self, cmd) -> str:
        """The feature requirement THIS command's resolve actually reads.

        pip builds its candidate for a name requirement from the installed
        distribution whenever that distribution already satisfies the request —
        modelled here as "the index would return the version already on disk".
        The candidate carries the INSTALLED artifact's metadata, so that is the
        dependency set the resolve enforces, not the index artifact's.

        A command that forces the package's own reinstall has no installed
        candidate to choose and reads the index artifact instead. That is the
        whole of issue #3047: pip's ``--force-reinstall --no-deps`` pass is the
        only one that reaches the index artifact, and it is the one pass that
        resolves nothing.
        """
        if (
            self.installed.get(self.feature) == self.feature_version
            and not self._reinstalls_package(cmd, self.feature)
        ):
            return self.feature_installed_requires
        return self.feature_requires

    def _reinstalls_package(self, cmd, package) -> bool:
        """Is *package* itself in this command's reinstall set?

        A blanket ``--force-reinstall`` covers everything the resolve touches;
        ``--reinstall-package X`` covers X alone.
        """
        if "--force-reinstall" in cmd:
            return True
        return (
            "--reinstall-package" in cmd
            and cmd[cmd.index("--reinstall-package") + 1] == package
        )

    def _reinstalls_dependencies(self, cmd) -> bool:
        """Would this command reinstall core as a resolved dependency?

        ``--force-reinstall`` applies to the whole resolve, so it reaches core
        unless dependencies are excluded (``--no-deps``) or the reinstall names
        the package it is for (``--reinstall-package <pkg>``, which uv restricts
        the reinstall set to). Neither the version pin nor ``--upgrade`` has any
        say: the reinstall happens at whatever version resolves, including the
        one already installed.
        """
        return (
            "--force-reinstall" in cmd
            and "--no-deps" not in cmd
            and "--reinstall-package" not in cmd
        )

    def _reinstall_core_from_index(self, pin):
        """Core comes back as an index wheel — same version is enough.

        The link is what is lost. A pin that permits the installed version
        permits the identical wheel the index publishes, so this is precisely
        the swap a constraints file cannot see.
        """
        from packaging.version import Version

        candidates = self._core_candidates(pin) & set(self.core_index)
        if not candidates:
            return  # nothing on the index the pin allows; core stays put
        self.installed[CORE] = max(candidates, key=Version)
        self.editable.pop(CORE, None)

    def _reinstall_sdk_from_index(self):
        """Core's editable dependency comes back as an index wheel too.

        The cascade does not stop at core: every package the resolve touches is
        in a blanket reinstall's set, so an editable SDK loses its link the same
        way. Modelled because the repair path installs CORE — where core itself
        is the target and the collateral is everything under it.
        """
        self.editable.pop(SDK, None)

    def _swap_core_for_index_wheel(self, pin, requires):
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        wanted = SpecifierSet(requires)
        candidates = [v for v in self._core_candidates(pin) if Version(v) in wanted]
        if candidates:
            self.installed[CORE] = max(candidates, key=Version)
            self.editable.pop(CORE, None)

    def _install_core(self, cmd, target, timeout=None):
        """Install core itself: ``-e <path>`` links that checkout's build, a
        spec takes the newest index wheel the spec allows."""
        if self.repair_hangs:
            # The resolve never completes, so nothing is written.
            self._never_returns(cmd, target, timeout)
        if self.repair_fails:
            return self._failed(cmd, f"x Failed to install {target}: no such checkout")
        if self.repair_interrupted:
            # Ctrl-C DURING the automatic restore. Nothing is written, but a
            # repair was unmistakably attempted — the distinction the interrupt
            # report has to get right (issue #2962).
            raise KeyboardInterrupt()
        if self.repair_noops:
            # Exit 0, venv unchanged — an installer that reported success and
            # left core exactly where it was.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if (
            self.core_resolve_refused
            and "--no-deps" not in cmd
            and "--upgrade" not in cmd
            and "--reinstall-package" not in cmd
        ):
            # pip's third pass: no force, no exclusion — it resolves the
            # dependencies of what the `--no-deps` pass installed, and core is
            # already home by the time it runs. Its refusal is a fact about the
            # dependency closure and about nothing else (issue #3047).
            return self._failed(
                cmd,
                "x No solution found when resolving dependencies: "
                f"{CORE}=={self.installed[CORE]} depends on "
                f"{SDK}>=0.99, but you require {SDK}==0.36.0.",
            )
        if self.core_write_pass_fails and "--no-deps" in cmd:
            # pip's destructive pass fails — but the resolve pass before it has
            # already put core back (it ran this same branch and wrote), so this
            # exit code describes the command, not the venv.
            return self._failed(cmd, f"x Failed to install {target}: connection reset")
        if self._reinstalls_dependencies(cmd):
            # Core is the target, so its OWN dependencies are the ones a blanket
            # reinstall drags off their checkouts.
            self._reinstall_sdk_from_index()
        result = self._land_core(cmd, target)
        if self.repair_hangs_after_restore:
            # Killed AFTER the write above. A timeout ends a process; it does
            # not roll back what that process had already done.
            self._never_returns(cmd, target, timeout)
        return result

    def _never_returns(self, cmd, target, timeout):
        """The installer stops coming back: killed by a bound, or hung forever.

        Without a bound the real subprocess would block indefinitely, which is
        the defect itself — refuse rather than fake a return.
        """
        if timeout is None:
            raise UnboundedInstall(
                f"installer for {target} never returns and was started with "
                "timeout=None — this call would block forever"
            )
        raise subprocess.TimeoutExpired(cmd, timeout)

    def _land_core(self, cmd, target):
        """Write the core install this command asks for."""
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        if "-e" in cmd:
            self.editable[CORE] = target
            # A checkout builds the version ITS pyproject declares — the whole
            # point of switching checkouts.
            self.installed[CORE] = self.checkouts[target]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        spec = SpecifierSet(target[len(CORE):])
        current = self.installed.get(CORE)
        if (
            current is not None
            and Version(current) in spec
            and not self._reinstalls_package(cmd, CORE)
        ):
            # pip and uv judge "already satisfied" by VERSION alone. A spec
            # install that is already satisfied writes nothing, so a core
            # sitting at the right version from the WRONG source survives
            # untouched unless the command scopes a reinstall of core. Modelling
            # this is what lets a test see a repair that exits 0 and fixes
            # nothing.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        self.installed[CORE] = max(
            (v for v in self.core_index if Version(v) in spec), key=Version,
        )
        self.editable.pop(CORE, None)
        # An index resolution records no PEP 610 provenance — landing one clears
        # whatever direct URL was there, and rewrites the dist-info, so damaged
        # provenance becomes readable-and-absent rather than staying unknown.
        self.direct_urls.pop(CORE, None)
        self.unreadable_provenance.discard(CORE)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def _core_candidates(self, pin):
        """Core versions the resolver may pick: the index, minus the pin."""
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        candidates = set(self.core_index) | {self.installed[CORE]}
        if pin and self.honours_constraints:
            allowed = SpecifierSet(pin)
            candidates = {v for v in candidates if Version(v) in allowed}
        return candidates

    def _core_pin(self, cmd):
        """The core pin read out of the ``-c <file>`` constraints file, if any."""
        if "-c" not in cmd:
            return None
        text = Path(cmd[cmd.index("-c") + 1]).read_text(encoding="utf-8")
        for line in (ln.strip() for ln in text.splitlines()):
            if line.startswith(CORE):
                return line[len(CORE):]
        return None


def use_fake_uv(monkeypatch, venv):
    """Point the venv-state reads and the installer at *venv*."""
    import importlib.metadata as md

    from kestrel_sovereign import cli

    monkeypatch.setattr(md, "version", venv.version)
    monkeypatch.setattr(cli, "_editable_install_path", venv.editable_path)
    monkeypatch.setattr(cli, "_direct_url_provenance", venv.direct_url_provenance)
    # The modelled host's declared checkouts EXIST — that is what makes them
    # declared. Without this the real `_editable_git_pull` runs against paths
    # like `/src/kestrel-sovereign` that are not on disk, fails "checkout does
    # not exist", and every test with an editable core entry inherits a failure
    # about the double rather than about the code. A test that wants a failing
    # pull patches this itself.
    from kestrel_sovereign import cli_lifecycle

    monkeypatch.setattr(
        cli_lifecycle, "_editable_git_pull",
        lambda checkout, allow_dirty: (0, "Already up to date."),
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(cli.subprocess, "run", venv.run)
