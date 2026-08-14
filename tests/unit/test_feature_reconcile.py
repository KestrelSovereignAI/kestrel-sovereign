"""Tests for ``feature_reconcile`` — the pure planning half of #1788.

Pins the model: the host venv must hold ``union(per-agent allowlists) +
MANDATORY_FEATURES``; allowlists name *classes*, install works on *packages*;
the source map decides editable-vs-pypi update mode; an unresolvable class or a
package with no known source is a hard error (no blind fallback).
"""

from __future__ import annotations

import pytest

from kestrel_sovereign import feature_reconcile as fr
from kestrel_sovereign.feature_registry import FeaturePackageInfo
from kestrel_sovereign.multi_agent.config import (
    LocalAgentConfig,
    MultiAgentConfig,
    RemoteAgentConfig,
)


def _registry():
    """A small fake registry: short-name -> FeaturePackageInfo."""
    return {
        "identity": FeaturePackageInfo(
            name="identity", package="kestrel-sovereign", git="g",
            features=["IdentityFeature"], description="", core=True,
        ),
        "voice": FeaturePackageInfo(
            name="voice", package="kestrel-feature-voice",
            git="https://example/voice.git",
            features=["VoiceFeature"], description="", core=False,
        ),
        "reflection": FeaturePackageInfo(
            name="reflection", package="kestrel-feature-reflection",
            git="https://example/reflection.git",
            features=["ReflectionFeature"], description="", core=False,
        ),
    }


def _ma(**agents):
    return MultiAgentConfig(agents=agents)


# --------------------------------------------------------------------------
# compute_required_classes
# --------------------------------------------------------------------------

def test_required_classes_union_plus_mandatory():
    ma = _ma(
        a=LocalAgentConfig(data_dir="d/a", port=8801, features=["VoiceFeature"]),
        b=LocalAgentConfig(
            data_dir="d/b", port=8802,
            features=["VoiceFeature", "ReflectionFeature"],
        ),
    )
    required, load_all = fr.compute_required_classes(ma)
    assert "VoiceFeature" in required
    assert "ReflectionFeature" in required
    # Mandatory features always required regardless of allowlists.
    assert {"IdentityFeature", "SecurityFeature"} <= required
    assert load_all == []


def test_required_classes_flags_load_all_agents():
    """An agent with features=None loads everything and imposes no specific
    requirement — it must be surfaced, not silently treated as 'needs all'."""
    ma = _ma(
        named=LocalAgentConfig(data_dir="d/n", port=8801, features=["VoiceFeature"]),
        loadall=LocalAgentConfig(data_dir="d/l", port=8802, features=None),
    )
    required, load_all = fr.compute_required_classes(ma)
    assert "VoiceFeature" in required
    assert load_all == ["loadall"]


def test_required_classes_ignores_remote_agents():
    ma = _ma(
        local=LocalAgentConfig(data_dir="d/x", port=8801, features=["VoiceFeature"]),
        remote=RemoteAgentConfig(url="https://remote.example"),
    )
    required, load_all = fr.compute_required_classes(ma)
    assert "VoiceFeature" in required


# --------------------------------------------------------------------------
# resolve_packages
# --------------------------------------------------------------------------

def test_resolve_packages_maps_classes_and_skips_core():
    reg = _registry()
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"VoiceFeature", "IdentityFeature"}, reg,
    )
    # Voice resolves to its installable package.
    assert "kestrel-feature-voice" in pkg_infos
    assert class_to_pkg["VoiceFeature"] == "kestrel-feature-voice"
    # Identity resolves but is bundled in core -> NOT a reconcile target.
    assert "kestrel-sovereign" not in pkg_infos
    assert class_to_pkg["IdentityFeature"] == "kestrel-sovereign"
    assert unresolved == []


def test_resolve_packages_unknown_class_is_unresolved():
    reg = _registry()
    _, _, unresolved = fr.resolve_packages({"GhostFeature"}, reg)
    assert unresolved == ["GhostFeature"]


def test_resolve_packages_mandatory_missing_from_registry_is_not_error():
    """A mandatory class absent from the catalog lives in core — never an
    error."""
    reg = {"voice": _registry()["voice"]}  # no identity/security entries
    _, _, unresolved = fr.resolve_packages(
        {"SecurityFeature", "VoiceFeature"}, reg,
    )
    assert unresolved == []


def test_resolve_packages_uses_live_entrypoint_dist():
    """An installed external package (live entry point) resolves even when the
    static catalog hasn't listed it yet — discovery is the source of truth."""
    reg = {}  # empty catalog
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"ParametricSelfFeature"}, reg,
        entrypoint_dists={"ParametricSelfFeature": "kestrel-feature-parametric-self"},
    )
    assert unresolved == []
    assert "kestrel-feature-parametric-self" in pkg_infos
    assert class_to_pkg["ParametricSelfFeature"] == "kestrel-feature-parametric-self"


def test_resolve_packages_bundled_class_wins_over_duplicate_entrypoint():
    """When a class is BOTH bundled and shipped by an installed package, the
    loader loads the bundled one — reconcile must resolve to core, not the
    external dist (codex round 6 P2)."""
    reg = {}
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"DupeFeature"}, reg,
        entrypoint_dists={"DupeFeature": "kestrel-feature-dupe"},
        local_core_classes={"DupeFeature"},
    )
    assert unresolved == []
    assert pkg_infos == {}  # NOT the external package
    assert class_to_pkg["DupeFeature"] == fr.CORE_DISTRIBUTION


def test_resolve_packages_local_core_class_needs_no_install():
    """An in-tree bundled class resolves (no error) but is NOT a reconcile
    target — core reinstall provisions it."""
    reg = {}
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"AttachmentsFeature"}, reg,
        local_core_classes={"AttachmentsFeature"},
    )
    assert unresolved == []
    assert pkg_infos == {}
    assert class_to_pkg["AttachmentsFeature"] == fr.CORE_DISTRIBUTION


def test_resolve_packages_matches_a_catalog_row_spelled_differently():
    """Live metadata and the catalog may spell one distribution two ways.

    ``ep.dist.name`` reports whatever the installed package's METADATA says;
    the catalog row carries the git URL and extras the plan needs. Matched raw,
    the two miss each other and the package is planned from a synthesized
    sourceless info — no git URL, and (via the source index, keyed the same
    way) no declared source or pin either (#2949).
    """
    reg = {
        "reflection": FeaturePackageInfo(
            name="reflection", package="Kestrel_Feature_Reflection",
            git="https://example/reflection.git",
            features=["ReflectionFeature"], description="", core=False,
        ),
    }
    # 1. Installed: resolved from live metadata, matched to the catalog row.
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"ReflectionFeature"}, reg,
        entrypoint_dists={"ReflectionFeature": "kestrel_feature_reflection"},
    )
    assert unresolved == []
    assert class_to_pkg == {"ReflectionFeature": "kestrel-feature-reflection"}
    assert pkg_infos["kestrel-feature-reflection"].git == "https://example/reflection.git"

    # 2. Not installed: resolved from the catalog alone, same identity.
    pkg_infos, class_to_pkg, unresolved = fr.resolve_packages(
        {"ReflectionFeature"}, reg,
    )
    assert unresolved == []
    assert class_to_pkg == {"ReflectionFeature": "kestrel-feature-reflection"}
    assert list(pkg_infos) == ["kestrel-feature-reflection"]


def test_resolve_packages_uncatalogued_uninstalled_class_is_unresolved():
    """The motivating bug: an allowlist names a feature that is neither
    installed nor catalogued nor bundled -> hard error."""
    reg = {}
    _, _, unresolved = fr.resolve_packages(
        {"GhostFeature"}, reg,
        entrypoint_dists={"VoiceFeature": "kestrel-feature-voice"},
        local_core_classes={"AttachmentsFeature"},
    )
    assert unresolved == ["GhostFeature"]


# --------------------------------------------------------------------------
# build_source_index
# --------------------------------------------------------------------------

def test_build_source_index_resolves_names_and_dist_names():
    reg = _registry()
    entries = [
        {"name": "voice", "editable": "/co/voice", "pypi": None, "extras": []},
        {"name": "kestrel-feature-reflection", "editable": None,
         "pypi": ">=0.2,<0.3", "extras": []},
    ]
    idx = fr.build_source_index(entries, reg)
    assert idx["kestrel-feature-voice"].editable == "/co/voice"
    assert idx["kestrel-feature-voice"].mode == "editable"
    assert idx["kestrel-feature-reflection"].pypi == ">=0.2,<0.3"
    assert idx["kestrel-feature-reflection"].mode == "pypi"


def test_source_index_keys_are_canonical_whoever_spelled_the_name():
    """One key per distribution, whichever of the three sources named it.

    The operator's spelling, the catalog's ``package``, and live metadata all
    reach this index; :func:`resolve_packages` looks it up under
    :func:`canonical_package` form, so anything that keys it otherwise is a
    silently unmanaged package (#2949).
    """
    reg = dict(_registry())
    # A catalog row that spells its own distribution with underscores.
    reg["reflection"] = FeaturePackageInfo(
        name="reflection", package="Kestrel_Feature_Reflection",
        git="https://example/reflection.git",
        features=["ReflectionFeature"], description="", core=False,
    )
    entries = [
        # Operator's spelling differs from the catalog's, which differs again
        # from the canonical form. All three are one distribution (PEP 503).
        {"name": "KESTREL.FEATURE.REFLECTION", "editable": None,
         "pypi": ">=0.2,<0.3", "extras": []},
        {"name": "some_private_pkg", "editable": None, "pypi": "", "extras": []},
    ]
    idx = fr.build_source_index(entries, reg)
    assert idx["kestrel-feature-reflection"].pypi == ">=0.2,<0.3"
    # Unknown to the catalog: still canonical, so a live dist reporting
    # ``some-private-pkg`` finds it.
    assert idx["some-private-pkg"].pypi == ""


# --------------------------------------------------------------------------
# plan_reconcile
# --------------------------------------------------------------------------

def _voice_info():
    return _registry()["voice"]


def test_plan_installs_missing_editable_from_source_map():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", editable="/co/voice")}
    actions, no_source = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": None},
        editable_paths={"kestrel-feature-voice": None},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    assert no_source == []
    (a,) = actions
    assert a.op == "install" and a.mode == "editable" and a.source == "/co/voice"
    assert a.required_by == ["VoiceFeature"]


def test_plan_updates_present_pypi_package():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3,<0.4")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": None},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.op == "update" and a.mode == "pypi"
    assert a.source == "kestrel-feature-voice>=0.3,<0.4"


def test_plan_pypi_renders_extras_before_version_spec():
    """pip requires ``pkg[extra]>=x`` — extras must NOT trail the spec."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3,<0.4", extras=["local"])}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": None},
        editable_paths={"kestrel-feature-voice": None},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.source == "kestrel-feature-voice[local]>=0.3,<0.4"


def test_plan_detects_editable_install_without_manifest_entry():
    """A package already installed editable must be updated via git pull, not
    clobbered with a PyPI build, even when the manifest is silent."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    actions, no_source = fr.plan_reconcile(
        pkg_infos, source_index={},
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": "/dev/voice"},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    assert no_source == []
    (a,) = actions
    assert a.mode == "editable" and a.source == "/dev/voice" and a.op == "update"
    # Already linked to the same checkout -> git pull alone, no relink.
    assert a.relink is False


def test_plan_relinks_editable_when_installed_from_pypi():
    """Source map says editable but the package is a non-editable PyPI install
    -> must relink (pip install -e), not just git pull (codex round 3 P2)."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", editable="/co/voice")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},  # installed...
        editable_paths={"kestrel-feature-voice": None},          # ...but NOT editable
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.mode == "editable" and a.op == "update" and a.relink is True


def test_plan_relinks_editable_when_checkout_path_differs():
    """Editable but linked to a DIFFERENT checkout than the source map wants."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", editable="/co/voice")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": "/old/voice"},  # different path
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.source == "/co/voice" and a.relink is True


def test_plan_no_relink_when_editable_path_matches():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", editable="/co/voice")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": "/co/voice"},  # same path
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.relink is False


def test_plan_falls_back_to_registry_pypi_when_no_source():
    """No source-map entry + not editable -> install the registry package from
    PyPI (the registry IS a known source; not a blind fallback)."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    actions, no_source = fr.plan_reconcile(
        pkg_infos, source_index={},
        installed_versions={"kestrel-feature-voice": None},
        editable_paths={"kestrel-feature-voice": None},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    assert no_source == []
    (a,) = actions
    assert a.mode == "pypi" and a.source == "kestrel-feature-voice"


def test_plan_uncatalogued_installed_feature_is_present_not_pypi():
    """A live-discovered, uncatalogued, non-editable installed feature (local /
    private / direct-URL) has no remote source — leave it `present`, don't
    fabricate a PyPI upgrade that would fail (codex round 4 P2)."""
    # Synthesized info: no git URL (the catalog signal).
    synth = FeaturePackageInfo(
        name="kestrel-feature-private", package="kestrel-feature-private",
        git="", features=["PrivateFeature"], description="", core=False,
    )
    actions, no_source = fr.plan_reconcile(
        {"kestrel-feature-private": synth}, source_index={},
        installed_versions={"kestrel-feature-private": "1.0.0"},
        editable_paths={"kestrel-feature-private": None},
        class_to_pkg={"PrivateFeature": "kestrel-feature-private"},
    )
    assert no_source == []
    (a,) = actions
    assert a.op == "present" and a.source is None


def test_plan_uncatalogued_missing_feature_is_no_source():
    """The same uncatalogued package, but NOT installed, has no source at all
    -> hard error rather than a fabricated PyPI install."""
    synth = FeaturePackageInfo(
        name="kestrel-feature-private", package="kestrel-feature-private",
        git="", features=["PrivateFeature"], description="", core=False,
    )
    actions, no_source = fr.plan_reconcile(
        {"kestrel-feature-private": synth}, source_index={},
        installed_versions={"kestrel-feature-private": None},
        editable_paths={"kestrel-feature-private": None},
        class_to_pkg={"PrivateFeature": "kestrel-feature-private"},
    )
    assert actions == []
    assert no_source == ["kestrel-feature-private"]


def test_plan_force_reinstall_when_switching_editable_to_pypi():
    """Source map declares pypi but the install is editable -> force reinstall
    so the wheel replaces the checkout link (codex round 9 P2)."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3,<0.4")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},  # satisfies spec...
        editable_paths={"kestrel-feature-voice": "/dev/voice"},  # ...but editable
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.mode == "pypi" and a.force_reinstall is True


def test_plan_no_force_reinstall_for_normal_pypi_update():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3,<0.4")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": None},  # not editable
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
    )
    (a,) = actions
    assert a.force_reinstall is False


def test_plan_prefer_pypi_overrides_editable():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", editable="/co/voice")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": "/co/voice"},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
        prefer="pypi",
    )
    (a,) = actions
    assert a.mode == "pypi"


def test_plan_prefer_source_uses_editable_checkout():
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3")}
    actions, _ = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": "0.3.1"},
        editable_paths={"kestrel-feature-voice": "/dev/voice"},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
        prefer="source",
    )
    (a,) = actions
    assert a.mode == "editable" and a.source == "/dev/voice"


def test_plan_prefer_source_without_checkout_falls_back_to_pypi():
    """--prefer-source is a *preference*: with no known checkout it honours the
    only available source (PyPI) rather than erroring — PyPI is still a known
    source, not a blind provider swap."""
    pkg_infos = {"kestrel-feature-voice": _voice_info()}
    idx = {"kestrel-feature-voice": fr.SourceEntry(
        package="kestrel-feature-voice", pypi=">=0.3")}
    actions, no_source = fr.plan_reconcile(
        pkg_infos, idx,
        installed_versions={"kestrel-feature-voice": None},
        editable_paths={"kestrel-feature-voice": None},
        class_to_pkg={"VoiceFeature": "kestrel-feature-voice"},
        prefer="source",
    )
    assert no_source == []
    (a,) = actions
    assert a.mode == "pypi" and a.source == "kestrel-feature-voice>=0.3"


# --- core source policy (#2949) --------------------------------------------
#
# Every kestrel-feature-* depends on kestrel-sovereign, so a feature install can
# resolve core from the index. These pin the *policy*: where core is supposed to
# come from, what constraint that implies, and what counts as a violation.


_DERIVE = object()


def _shape(version=None, editable_path=None, direct_url=_DERIVE):
    """Build a CoreInstallShape the way real metadata would.

    An editable install records a PEP 610 direct URL *as well as* the editable
    flag, so ``direct_url`` defaults to ``editable_path`` — a shape with an
    editable path but no direct URL is one the metadata cannot produce, and
    asserting against it would prove nothing. Pass ``direct_url`` explicitly to
    model a NON-editable direct install (VCS, local path, remote archive):
    the case that ``not is_editable`` cannot distinguish from an index wheel.
    """
    if direct_url is _DERIVE:
        direct_url = editable_path
    return fr.CoreInstallShape(
        version=version, editable_path=editable_path, direct_url=direct_url,
    )


def test_core_policy_defaults_to_the_live_editable_link():
    """No manifest entry: protect the link the venv actually has."""
    policy = fr.resolve_core_policy({}, "/src/core")
    assert policy.editable == "/src/core" and policy.pypi is None
    assert policy.guarded


def test_core_policy_is_unguarded_for_an_undeclared_wheel():
    """Core already a wheel and nothing declared — there is no link to protect
    and no editable install is invented for the operator."""
    policy = fr.resolve_core_policy({}, None)
    assert not policy.guarded
    assert fr.core_install_constraints(_shape(version="0.52.0"), policy) == []


def test_core_policy_prefers_the_manifest_over_the_live_link():
    """An explicit entry is the operator moving core; the live link does not
    override their declaration."""
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, editable="/src/next")}
    policy = fr.resolve_core_policy(idx, "/src/old")
    assert policy.editable == "/src/next"


def test_core_policy_carries_a_declared_pypi_window():
    """A `pypi` entry declares a wheel — and a non-empty spec is still a
    declaration the batch must hold, not a waiver of the guard."""
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, pypi=">=0.52,<0.53")}
    policy = fr.resolve_core_policy(idx, "/src/old")
    assert policy.editable is None and policy.pypi == ">=0.52,<0.53"
    assert fr.core_install_constraints(_shape(version="0.52.0"), policy) == [
        "kestrel-sovereign>=0.52,<0.53"
    ]


def test_core_policy_with_an_empty_pypi_spec_still_holds_the_source():
    """`pypi = ""` is "any version FROM THE INDEX" — a declaration, not a waiver.

    Half of it is empty: there is no version window, so no constraint line to
    write. The other half is not: the entry names where core comes from, so an
    editable link (or a core that is not installed at all) violates it exactly
    as a version outside a window would. Reading the empty spec as "unguarded"
    let the live link the operator declared they were leaving pass the check
    (issue #2949).
    """
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, pypi="")}
    policy = fr.resolve_core_policy(idx, "/src/old")

    assert policy.guarded
    assert fr.core_install_constraints(_shape(version="0.52.0"), policy) == []
    assert fr.core_install_matches(_shape("0.52.0", None), policy)  # a wheel: fine
    assert not fr.core_install_matches(_shape("0.52.0", "/src/old"), policy)
    assert not fr.core_install_matches(_shape(None, None), policy)  # not installed
    assert policy.describe_expected() == (
        "kestrel-sovereign from the index (non-editable)"
    )


def test_editable_constraint_tracks_the_installed_version():
    """The pin is `==<what is installed now>`, so it must be derived from the
    shape at install time — re-snapshot after moving core."""
    policy = fr.resolve_core_policy({}, "/src/core")
    assert fr.core_install_constraints(_shape("0.52.0", "/src/core"), policy) == [
        "kestrel-sovereign==0.52.0"
    ]
    assert fr.core_install_constraints(_shape("0.53.0", "/src/core"), policy) == [
        "kestrel-sovereign==0.53.0"
    ]


def test_editable_policy_tolerates_version_drift_but_not_a_lost_link():
    """Reinstalling the checkout is normal; losing the link is the defect."""
    policy = fr.resolve_core_policy({}, "/src/core")
    assert fr.core_install_matches(_shape("0.99.0", "/src/core"), policy)
    assert not fr.core_install_matches(_shape("0.53.0", None), policy)
    assert not fr.core_install_matches(_shape("0.52.0", "/src/other"), policy)


def test_pypi_policy_is_violated_by_a_version_outside_the_window():
    """A feature that dragged core past `<0.53` violated the manifest just as
    loudly as one that dropped an editable link."""
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, pypi=">=0.52,<0.53")}
    policy = fr.resolve_core_policy(idx, None)
    assert fr.core_install_matches(_shape("0.52.1", None), policy)
    assert not fr.core_install_matches(_shape("0.53.0", None), policy)
    assert not fr.core_install_matches(_shape("0.52.1", "/src/core"), policy)


def test_pypi_policy_rejects_a_non_editable_direct_url_install():
    """`pypi` declares a SOURCE, and "not editable" is not that source.

    A core installed from a VCS ref, a local path or a remote archive is
    non-editable and can sit inside the declared version window, so a predicate
    built on `not is_editable` accepted it — silently satisfying a source
    declaration it plainly violates. PEP 610 records these installs with a
    `direct_url.json` that has no `dir_info.editable`, which is exactly the
    evidence the version-shaped proxy threw away (issue #2949).
    """
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, pypi=">=0.52,<0.54")}
    policy = fr.resolve_core_policy(idx, None)

    # An index wheel: no direct URL recorded at all. This is the only shape
    # that actually came from an index.
    assert fr.core_install_matches(_shape("0.53.0", None), policy)

    # Same version, same "not editable" — but installed from a git ref, and
    # from a local directory. Both violate a `pypi` source declaration.
    assert not fr.core_install_matches(
        _shape("0.53.0", None, direct_url="git+https://example.invalid/core@abc"),
        policy,
    )
    assert not fr.core_install_matches(
        _shape("0.53.0", None, direct_url="/src/core-checkout"), policy,
    )


def test_pypi_policy_with_no_version_window_still_rejects_a_direct_url():
    """`pypi = ""` is a pure SOURCE declaration — the case with no version
    opinion left to lean on, where testing the proxy could not distinguish
    anything at all."""
    idx = {fr.CORE_DISTRIBUTION: fr.SourceEntry(
        package=fr.CORE_DISTRIBUTION, pypi="")}
    policy = fr.resolve_core_policy(idx, None)
    assert fr.core_install_matches(_shape("0.53.0", None), policy)
    assert not fr.core_install_matches(
        _shape("0.53.0", None, direct_url="git+https://example.invalid/core"),
        policy,
    )


def test_shape_describes_a_direct_url_install_distinctly():
    """An operator reading the failure must be told which wrong source it is."""
    assert "index wheel" in _shape("0.53.0", None).describe()
    assert "editable" in _shape("0.53.0", "/src/core").describe()
    described = _shape("0.53.0", None, direct_url="git+https://x.invalid/c").describe()
    assert "direct URL" in described and "git+https://x.invalid/c" in described


def test_describe_core_change_names_before_after_and_expected():
    policy = fr.resolve_core_policy({}, "/src/core")
    described = fr.describe_core_change(
        _shape("0.52.0", "/src/core"), _shape("0.53.0", None), policy,
    )
    assert "before: editable → /src/core (0.52.0)" in described
    assert "after:  non-editable (index wheel in site-packages) (0.53.0)" in described
    assert "expected: editable → /src/core" in described
    assert fr.describe_core_change(
        _shape("0.52.0", "/src/core"), _shape("0.53.0", "/src/core"), policy,
    ) is None
