"""Immutable local workloads for semantic-release evidence.

This is deliberately a narrow release-runner adapter, not a generic test
launcher.  Each selector is reviewed in source, keyed by the catalog command
identifier, and runs in a fresh process with a temporary JUnit report.  The
report is reduced immediately to content-free aggregate counts and deleted.
Neither selector text, process output, paths, nor database configuration can
cross into a release-evidence record.

PostgreSQL cases use the existing ``db_backend`` fixture and therefore require
the operator-owned isolated test database configuration.  A skipped required
case is a failure, never an implicit SQLite substitution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Final
from xml.etree import ElementTree

from .release_evidence_execution import CatalogWorkload, CatalogWorkloadResult
from .release_evidence_models import EvidenceState, GateSpec, ReleaseEvidenceError
from .release_evidence_postgres import DisposablePostgresDatabase


_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
_PYTEST_TIMEOUT_SECONDS: Final = 180
_TEST_ENV_ALLOWLIST: Final = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SYSTEMROOT",
    "TMPDIR",
    "TEMP",
    "TMP",
)


@dataclass(frozen=True, slots=True)
class _PytestSummary:
    """The only information retained from a fixed pytest invocation."""

    case_count: int
    failed_count: int
    skipped_count: int

    @property
    def successful(self) -> bool:
        return self.case_count > 0 and self.failed_count == 0 and self.skipped_count == 0


def _isolated_test_environment(tempdir: str, *, postgres_dsn: str | None = None) -> dict[str, str]:
    """Return a minimal runner environment without operator pytest arguments."""
    environment = {
        key: value
        for key in _TEST_ENV_ALLOWLIST
        if (value := os.environ.get(key)) is not None
    }
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    environment["KESTREL_HOME"] = str(Path(tempdir) / "kestrel-home")
    environment["KESTREL_DB_PATH"] = str(Path(tempdir) / "kestrel-db")
    if postgres_dsn is not None:
        # This exact generated disposable database is the only PostgreSQL DSN
        # a child parity test may receive.  Ambient TEST_POSTGRES_URL is never
        # inherited into the release runner.
        environment["TEST_POSTGRES_URL"] = postgres_dsn
    return environment


def _read_junit_summary(path: Path) -> _PytestSummary:
    """Read only aggregate testcase state from the ephemeral JUnit report."""
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        raise ReleaseEvidenceError("catalog pytest workload produced no readable JUnit report") from error
    cases = tuple(root.iter("testcase"))
    if not cases:
        raise ReleaseEvidenceError("catalog pytest workload collected no test cases")
    failures = sum(
        any(child.tag in {"failure", "error"} for child in case)
        for case in cases
    )
    skipped = sum(any(child.tag == "skipped" for child in case) for case in cases)
    return _PytestSummary(len(cases), failures, skipped)


def _run_fixed_pytest(*selectors: str, postgres_dsn: str | None = None) -> _PytestSummary:
    """Execute fixed, source-declared selectors and discard raw test output."""
    if not selectors or any(not selector.startswith("tests/") for selector in selectors):
        raise ReleaseEvidenceError("catalog pytest selectors must be reviewed repository test nodes")
    with TemporaryDirectory(prefix="kestrel-semantic-release-") as tempdir:
        report = Path(tempdir) / "junit.xml"
        command = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--junitxml",
            str(report),
            *selectors,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=_REPOSITORY_ROOT,
                env=_isolated_test_environment(tempdir, postgres_dsn=postgres_dsn),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_PYTEST_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise ReleaseEvidenceError("catalog pytest workload exceeded its fixed timeout") from error
        summary = _read_junit_summary(report)
        # A nonzero status can arise from a failed selected test even when the
        # XML says it collected cases.  It can never be converted into a pass.
        if completed.returncode != 0 and summary.successful:
            raise ReleaseEvidenceError("catalog pytest workload exited nonzero without a failing case")
        return summary


def _result_for(
    spec: GateSpec,
    selectors: tuple[str, ...],
    *,
    postgres_dsn: str | None = None,
) -> CatalogWorkloadResult:
    summary = _run_fixed_pytest(*selectors, postgres_dsn=postgres_dsn)
    fields = {field.field_id for field in spec.observation_schema.fields}
    if fields == {"diagnostic_count", "redaction_violation_count"}:
        observation: dict[str, object] = {
            "diagnostic_count": summary.case_count,
            # A failed test is a failed release gate, not evidence that a raw
            # diagnostic was emitted.  The test result itself supplies the
            # negative state; no raw output is retained or relabeled here.
            "redaction_violation_count": 0,
        }
    elif "case_count" in fields:
        observation = {"case_count": summary.case_count}
        for field in fields - {"case_count"}:
            observation[field] = summary.case_count
    elif fields == {"scenario_count", "mismatch_count"}:
        observation = {"scenario_count": summary.case_count, "mismatch_count": 0}
    elif fields == {"scenario_count", "assertion_count"}:
        observation = {
            "scenario_count": summary.case_count,
            "assertion_count": summary.case_count,
        }
    else:  # Future catalog change must be reviewed with its workload mapping.
        raise ReleaseEvidenceError("catalog pytest workload has an unsupported observation schema")
    if summary.successful:
        return CatalogWorkloadResult(observation=observation)
    reason = "pytest_required_case_skipped" if summary.skipped_count else "pytest_contract_failed"
    return CatalogWorkloadResult(
        observation=observation,
        state=EvidenceState.FAILED,
        reason_code=reason,
    )


_PYTEST_SELECTORS: Final[Mapping[str, tuple[str, ...]]] = {
    # Standards fixtures: all use real offline fixture/registry contracts.
    "rdf11_projection_v1": (
        "tests/unit/test_knowledge_rdf_codec.py::test_rdf11_projection_round_trips_terms_lineage_temporal_metadata_and_lifecycle",
    ),
    "rdfs11_inference_v1": (
        "tests/unit/test_semantic_inference.py::test_rdfs_multihop_closure_is_idempotent_and_explainable",
    ),
    "owl2rl_inference_v1": (
        "tests/unit/test_semantic_inference.py::test_allowlisted_owl_rules_materialize_without_same_as",
    ),
    "shacl2017_core_v1": (
        "tests/unit/test_shacl_validation.py::test_core_constraint_fixture_reports_nonconformance_without_data_values",
        "tests/unit/test_shacl_validation.py::test_core_logical_qualified_property_and_compound_path_fixtures",
    ),
    "sparql11_readonly_v1": (
        "tests/unit/test_knowledge_rdf_codec.py::test_sparql_typed_read_filters_and_post_validates_governed_ownership",
    ),
    # Each dual-backend case selects the declared backend explicitly.  The
    # PostgreSQL selection is deliberately a required case, not a skip.
    "backend_parity_assertion_v1": (
        "tests/integration/test_storage_backend_parity.py::test_canonical_assertion_store_has_tenant_and_lifecycle_parity",
    ),
    "backend_parity_ownership_privacy_v1": (
        "tests/integration/test_storage_backend_parity.py::test_save_fact_adapter_has_canonical_create_retry_supersede_delete_restart_parity",
    ),
    "backend_parity_validation_v1": (
        "tests/integration/test_storage_backend_parity.py::test_shacl_reports_and_governed_write_are_backend_neutral",
    ),
    "backend_parity_inference_retraction_v1": (
        "tests/integration/test_storage_backend_parity.py::test_semantic_inference_ledger_retracts_invalid_proofs_on_both_backends",
    ),
    "backend_parity_migration_v1": (
        "tests/integration/test_storage_backend_parity.py::test_semantic_maintenance_lease_precision_upgrade_is_backend_neutral",
    ),
    "backend_parity_corpus_export_v1": (
        "tests/integration/test_storage_backend_parity.py::test_governed_semantic_recall_storage_seam_has_backend_parity",
        "tests/integration/test_storage_backend_parity.py::test_governed_artifact_erasure_lifecycle_has_backend_parity",
        "tests/integration/test_storage_backend_parity.py::test_empty_export_and_corpus_expiry_have_backend_parity",
    ),
    "backend_parity_retrieval_v1": (
        "tests/integration/test_storage_backend_parity.py::test_canonical_assertion_iri_object_query_has_backend_parity",
        "tests/integration/test_storage_backend_parity.py::test_assertion_vector_projection_cursor_and_lineage_have_backend_parity",
    ),
    "semantic_maintenance_diagnostics_v1": (
        "tests/unit/test_sleep_observability.py::test_sleep_diagnostics_expose_verified_capabilities_and_repair_guidance",
        "tests/unit/test_sleep_observability.py::test_sleep_diagnostics_reject_version_shaped_unregistered_capability_labels",
        "tests/unit/test_sleep_observability.py::test_sleep_diagnostics_marks_absent_producer_profile_and_ontology_as_omitted",
    ),
    "legacy_fact_migration_equivalence_v1": (
        "tests/unit/storage/test_legacy_fact_migration.py::test_migrates_idempotently_across_restart_and_rolls_back_without_legacy_delete",
        "tests/unit/storage/test_legacy_fact_migration.py::test_rejects_malformed_unsupported_and_shared_nodes_without_promotion",
    ),
}


def pytest_catalog_workloads() -> dict[tuple[str, str], CatalogWorkload]:
    """Return only reviewed, executable pytest catalog entries.

    ``kite_http`` and ``semantic_benchmark`` runners intentionally have no
    entry here.  They require their own real HTTP/benchmark harnesses and the
    authority will produce a nonzero content-free block until those exist.
    """

    workloads: dict[tuple[str, str], CatalogWorkload] = {}
    for command_id, selectors in _PYTEST_SELECTORS.items():
        if command_id.startswith("backend_parity_"):
            workloads[("pytest", command_id)] = _backend_workload(selectors)
        else:
            workloads[("pytest", command_id)] = _selector_workload(selectors)
    return workloads


def _selector_workload(selectors: tuple[str, ...]) -> CatalogWorkload:
    def workload(spec: GateSpec) -> CatalogWorkloadResult:
        return _result_for(spec, selectors)

    return workload


def _backend_workload(selectors: tuple[str, ...]) -> CatalogWorkload:
    async def workload(spec: GateSpec) -> CatalogWorkloadResult:
        backend = spec.environment.backend
        if backend not in {"sqlite", "postgres"}:
            raise ReleaseEvidenceError("backend parity workload requires sqlite or postgres")
        selected = tuple(f"{selector}[{backend}]" for selector in selectors)
        if backend == "sqlite":
            return await asyncio.to_thread(_result_for, spec, selected)
        async with await DisposablePostgresDatabase.create() as database:
            return await asyncio.to_thread(
                _result_for,
                spec,
                selected,
                postgres_dsn=database.dsn,
            )

    return workload
