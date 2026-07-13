"""On-disk persistence for the forge pipeline.

Each forged feature is a directory under a *forge root* that lives inside the
agent's data directory — deliberately **outside** ``kestrel_sovereign/features/``
and carrying no entry point, so feature discovery never sees it. That is what
makes a forged feature INERT until approved: it physically cannot be loaded from
the forge root. Approval is what later moves/loads it; until then it just sits
there as inert source plus a metadata record.

Layout::

    <forge_root>/<module_name>/
        forge_meta.json        # pipeline state + spec + verdict + timestamps
        pkg/                    # the scaffolded, inert feature package
            __init__.py
            feature.py
            test_<module>.py

The state machine (issue #2434)::

    draft -> validated -> pending_approval -> approved   (source approved)
                                           -> rejected   (explicit Sovereign denial)
                                           -> blocked    (no approver / timeout /
                                                          cancellation — recoverable,
                                                          NOT a user denial; #1542)

``approved`` means the *source* is approved. Actually loading/installing an
approved package into a running agent (``approved -> loaded``) is a separate,
operator-gated install step that is intentionally out of scope here — see the
follow-up in issue #2434. Nothing transitions a record to ``loaded`` yet, so an
approved forged feature stays inert source-on-disk until that step exists.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Ordered pipeline states. ``loaded`` is terminal-positive; ``rejected`` is
# terminal-negative (an explicit Sovereign denial). ``blocked`` is a
# *recoverable* non-terminal state for approval outcomes that are NOT a user
# denial (no approver attached, timeout, cancellation — #1542): the record can
# be re-registered rather than being permanently killed.
STATE_DRAFT = "draft"
STATE_VALIDATED = "validated"
STATE_PENDING = "pending_approval"
STATE_APPROVED = "approved"
STATE_REJECTED = "rejected"
STATE_BLOCKED = "blocked"
STATE_LOADED = "loaded"

VALID_STATES = frozenset({
    STATE_DRAFT,
    STATE_VALIDATED,
    STATE_PENDING,
    STATE_APPROVED,
    STATE_REJECTED,
    STATE_BLOCKED,
    STATE_LOADED,
})

META_FILENAME = "forge_meta.json"
PKG_DIRNAME = "pkg"


class ForgeStoreError(RuntimeError):
    """Raised on unrecoverable store I/O problems."""


@dataclass
class ForgeRecord:
    """The persisted state of one forged feature."""

    name: str                 # the module_name (directory key)
    display_name: str         # the name as supplied by the forge author
    class_name: str
    state: str
    spec: Dict[str, Any]
    verdict: Optional[Dict[str, Any]] = None
    files: Optional[List[str]] = None
    history: Optional[List[Dict[str, str]]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "class_name": self.class_name,
            "state": self.state,
            "spec": self.spec,
            "verdict": self.verdict,
            "files": self.files or [],
            "history": self.history or [],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForgeRecord":
        return cls(
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            class_name=data.get("class_name", ""),
            state=data.get("state", STATE_DRAFT),
            spec=data.get("spec", {}),
            verdict=data.get("verdict"),
            files=data.get("files", []),
            history=data.get("history", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class ForgeStore:
    """Filesystem-backed store for forged features under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    # -- paths -----------------------------------------------------------------

    def _feature_dir(self, name: str) -> Path:
        return self.root / name

    def _meta_path(self, name: str) -> Path:
        return self._feature_dir(name) / META_FILENAME

    def pkg_dir(self, name: str) -> Path:
        return self._feature_dir(name) / PKG_DIRNAME

    def exists(self, name: str) -> bool:
        return self._meta_path(name).exists()

    # -- reads -----------------------------------------------------------------

    def get(self, name: str) -> Optional[ForgeRecord]:
        path = self._meta_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ForgeStoreError(f"cannot read forge record {name!r}: {exc}") from exc
        return ForgeRecord.from_dict(data)

    def list(self) -> List[ForgeRecord]:
        records: List[ForgeRecord] = []
        if not self.root.exists():
            return records
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / META_FILENAME).exists():
                continue
            record = self.get(child.name)
            if record is not None:
                records.append(record)
        return records

    # -- writes ----------------------------------------------------------------

    def _write_meta(self, record: ForgeRecord) -> None:
        feature_dir = self._feature_dir(record.name)
        feature_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._meta_path(record.name).with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self._meta_path(record.name))

    def save_scaffold(
        self,
        record: ForgeRecord,
        files: Dict[str, str],
    ) -> List[str]:
        """Write the scaffolded package files and persist the record.

        The files land under ``<feature_dir>/pkg/`` — inert, since that path is
        never on the discovery route. Returns the list of relative file paths.
        """
        pkg = self.pkg_dir(record.name)
        pkg.mkdir(parents=True, exist_ok=True)
        written: List[str] = []
        for rel_path, content in files.items():
            target = pkg / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(rel_path)
        record.files = sorted(written)
        self._write_meta(record)
        return record.files

    def save(self, record: ForgeRecord) -> None:
        self._write_meta(record)
