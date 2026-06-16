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
