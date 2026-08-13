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


class FakeUv:
    """A venv whose state actually changes when something is installed.

    Models the one behaviour the fix exists for: when the installed core cannot
    satisfy a feature's requirement, an *unconstrained* resolve fetches a core
    wheel from the index and drops the editable link. A constraints file holding
    core inside its declared window removes the offending versions as
    candidates, so the same resolve has no solution and fails loudly instead.

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
    link came back.
    """

    def __init__(
        self,
        *,
        core_version="0.52.0",
        core_checkout=CHECKOUT,
        checkouts=None,
        feature="kestrel-feature-voice",
        feature_version="0.4.0",
        feature_requires=">=0.53",
        core_index=("0.52.0", "0.53.0"),
        honours_constraints=True,
        repair_fails=False,
        repair_noops=False,
        feature_install_fails=False,
        feature_install_times_out=False,
    ):
        self.installed = {CORE: core_version}
        self.editable = {CORE: core_checkout} if core_checkout else {}
        self.checkouts = dict(checkouts or {})
        if core_checkout:
            self.checkouts.setdefault(core_checkout, core_version)
        self.feature = feature
        self.feature_version = feature_version
        self.feature_requires = feature_requires
        self.core_index = list(core_index)
        self.honours_constraints = honours_constraints
        self.repair_fails = repair_fails
        self.repair_noops = repair_noops
        self.feature_install_fails = feature_install_fails
        self.feature_install_times_out = feature_install_times_out
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
            return self._install_core(cmd, target)

        if self.feature_install_times_out:
            # A killed install is not a no-op: whatever pip had already written
            # stays written. Model the worst honest case — the dependency swap
            # landed before the timeout.
            self._swap_core_for_index_wheel(pin)
            raise subprocess.TimeoutExpired(cmd, timeout or 0)

        wanted = SpecifierSet(self.feature_requires)
        if Version(self.installed[CORE]) in wanted:
            # Core already satisfies the feature; a satisfied dependency is
            # left alone. Nothing to swap.
            if self.feature_install_fails:
                return self._failed(cmd, f"x Failed to build `{self.feature}`")
            self.installed[self.feature] = self.feature_version
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        candidates = [v for v in self._core_candidates(pin) if Version(v) in wanted]
        if not candidates:
            return self._failed(
                cmd,
                "x No solution found when resolving dependencies: "
                f"{self.feature}=={self.feature_version} depends on "
                f"{CORE}{self.feature_requires}, but you require "
                f"{CORE}=={self.installed[CORE]}.",
            )

        # The swap: an index wheel replaces whatever core was. pip resolves and
        # installs dependencies BEFORE the requested package, so this lands even
        # when the feature's own build then fails.
        self.installed[CORE] = max(candidates, key=Version)
        self.editable.pop(CORE, None)
        if self.feature_install_fails:
            return self._failed(cmd, f"x Failed to build `{self.feature}`")
        self.installed[self.feature] = self.feature_version
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

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

    def _swap_core_for_index_wheel(self, pin):
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        wanted = SpecifierSet(self.feature_requires)
        candidates = [v for v in self._core_candidates(pin) if Version(v) in wanted]
        if candidates:
            self.installed[CORE] = max(candidates, key=Version)
            self.editable.pop(CORE, None)

    def _install_core(self, cmd, target):
        """Install core itself: ``-e <path>`` links that checkout's build, a
        spec takes the newest index wheel the spec allows."""
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        if self.repair_fails:
            return self._failed(cmd, f"x Failed to install {target}: no such checkout")
        if self.repair_noops:
            # Exit 0, venv unchanged — an installer that reported success and
            # left core exactly where it was.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if "-e" in cmd:
            self.editable[CORE] = target
            # A checkout builds the version ITS pyproject declares — the whole
            # point of switching checkouts.
            self.installed[CORE] = self.checkouts[target]
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        spec = SpecifierSet(target[len(CORE):])
        self.installed[CORE] = max(
            (v for v in self.core_index if Version(v) in spec), key=Version,
        )
        self.editable.pop(CORE, None)
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
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(cli.subprocess, "run", venv.run)
