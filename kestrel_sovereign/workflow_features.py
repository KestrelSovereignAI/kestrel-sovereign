"""Metadata-only ownership discovery for feature-contributed workflows.

Workflow packages advertise two entry points from the same distribution:

``kestrel_sovereign.workflow_features``
    The entry-point *name* is the workflow name and its value is the owning
    Feature class entry point.

``kestrel_sovereign.features``
    The ordinary Feature discovery entry point.  Its value must match the
    workflow ownership claim exactly.

This module deliberately never calls :meth:`importlib.metadata.EntryPoint.load`.
It is safe to use while constructing persisted feature ceilings, before any
third-party feature implementation has been imported.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


WORKFLOW_FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.workflow_features"
FEATURE_ENTRY_POINT_GROUP = "kestrel_sovereign.features"


class WorkflowFeatureResolutionError(RuntimeError):
    """A workflow ownership claim is absent, ambiguous, or invalid."""


@dataclass(frozen=True)
class WorkflowFeatureProvider:
    """Validated metadata for the Feature class that owns one workflow."""

    workflow_name: str
    feature_class_name: str
    entry_point: str
    distribution: str


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_name(entry_point: Any) -> str:
    distribution = getattr(entry_point, "dist", None)
    return str(getattr(distribution, "name", "") or "").strip()


def _select_entry_points(entry_points: Any, group: str) -> list[Any]:
    """Select ``group`` from modern, legacy, or explicitly injected metadata."""

    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group))
    if isinstance(entry_points, dict):
        return list(entry_points.get(group, []))
    return [
        entry_point
        for entry_point in entry_points
        if getattr(entry_point, "group", None) == group
    ]


def _feature_class_name(value: str) -> Optional[str]:
    """Return the terminal class name from a ``module:Class`` entry value."""

    target = value.split("[", 1)[0].strip()
    if ":" not in target:
        return None
    attribute = target.split(":", 1)[1].strip()
    if not attribute:
        return None
    class_name = attribute.split(".")[-1]
    return class_name if class_name.isidentifier() else None


def resolve_workflow_feature_provider(
    workflow_name: str,
    *,
    entry_points: Optional[Iterable[Any]] = None,
) -> WorkflowFeatureProvider:
    """Resolve and validate the Feature provider for ``workflow_name``.

    Resolution is fail-closed: exactly one workflow claim must exist, its
    distribution must be identifiable, and that same distribution must publish
    exactly one matching Feature entry point.  Entry-point values are inspected
    as strings only; third-party modules are never imported.
    """

    workflow_name = str(workflow_name or "").strip()
    if not workflow_name:
        raise WorkflowFeatureResolutionError("workflow name must not be empty")

    try:
        discovered = (
            importlib.metadata.entry_points()
            if entry_points is None
            else entry_points
        )
    except Exception as exc:  # noqa: BLE001 - controlled actionable boundary
        raise WorkflowFeatureResolutionError(
            "could not read installed workflow feature entry points"
        ) from exc
    if not hasattr(discovered, "select") and not isinstance(discovered, dict):
        # A caller may inject a one-shot iterable for deterministic tests. The
        # resolver selects two groups, so materialize once without loading any
        # entry-point targets.
        discovered = list(discovered)

    workflow_claims = [
        entry_point
        for entry_point in _select_entry_points(
            discovered, WORKFLOW_FEATURE_ENTRY_POINT_GROUP
        )
        if str(getattr(entry_point, "name", "") or "").strip() == workflow_name
    ]
    if not workflow_claims:
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} has no installed provider; install a "
            f"feature package that publishes {WORKFLOW_FEATURE_ENTRY_POINT_GROUP!r}"
        )
    if len(workflow_claims) != 1:
        owners = sorted(
            _distribution_name(entry_point) or "unknown distribution"
            for entry_point in workflow_claims
        )
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} has duplicate provider claims: "
            f"{', '.join(owners)}"
        )

    claim = workflow_claims[0]
    value = str(getattr(claim, "value", "") or "").strip()
    class_name = _feature_class_name(value)
    if class_name is None:
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} provider has invalid Feature entry-point "
            f"value {value!r}; expected 'module:FeatureClass'"
        )
    distribution = _distribution_name(claim)
    if not distribution:
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} provider has no owning distribution metadata"
        )

    distribution_key = _normalized_distribution_name(distribution)
    same_distribution_features = [
        entry_point
        for entry_point in _select_entry_points(discovered, FEATURE_ENTRY_POINT_GROUP)
        if _normalized_distribution_name(_distribution_name(entry_point))
        == distribution_key
        and str(getattr(entry_point, "value", "") or "").strip() == value
    ]
    if not same_distribution_features:
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} claims {value!r} from distribution "
            f"{distribution!r}, but that distribution does not publish the matching "
            f"{FEATURE_ENTRY_POINT_GROUP!r} entry point"
        )
    if len(same_distribution_features) != 1:
        names = sorted(
            str(getattr(entry_point, "name", "") or "<unnamed>")
            for entry_point in same_distribution_features
        )
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} has duplicate matching Feature entry "
            f"points in distribution {distribution!r}: {', '.join(names)}"
        )

    feature_entry_point = same_distribution_features[0]
    if str(getattr(feature_entry_point, "name", "") or "").strip() != class_name:
        raise WorkflowFeatureResolutionError(
            f"workflow {workflow_name!r} points to Feature class {class_name!r}, "
            f"but its same-distribution Feature entry point is named "
            f"{getattr(feature_entry_point, 'name', '')!r}"
        )

    return WorkflowFeatureProvider(
        workflow_name=workflow_name,
        feature_class_name=class_name,
        entry_point=value,
        distribution=distribution,
    )


def resolve_workflow_feature(
    workflow_name: str,
    *,
    entry_points: Optional[Iterable[Any]] = None,
) -> str:
    """Return the validated owning Feature class name for ``workflow_name``."""

    return resolve_workflow_feature_provider(
        workflow_name, entry_points=entry_points
    ).feature_class_name


__all__ = [
    "FEATURE_ENTRY_POINT_GROUP",
    "WORKFLOW_FEATURE_ENTRY_POINT_GROUP",
    "WorkflowFeatureProvider",
    "WorkflowFeatureResolutionError",
    "resolve_workflow_feature",
    "resolve_workflow_feature_provider",
]
