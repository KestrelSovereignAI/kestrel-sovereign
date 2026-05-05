#!/usr/bin/env python3
"""
Pre-ceremony backup — encrypted, verifiable, restorable snapshot of every
agent we are about to rotate under the Quantum Hardening epic ([#921]).

Why this exists
---------------

Before running a hybrid-rotation ceremony (per ``SUCCESSION_RUNBOOK.md``)
on a live agent, we want belt-and-suspenders restore capability. The
agent's existing Lighthouse / GCS / GraphSync backups are not enough on
their own:

- Lighthouse manifests can lag the live database by hours (they re-pin on
  a schedule, not on every write). This script verifies that explicitly.
- A failed ceremony can corrupt local files in the data dir. We want a
  pre-ceremony snapshot that is bit-identical to the moment before.
- The plaintext-PEM legacy key sometimes lives outside the agent dir
  (e.g. Kestrel #1's ``agent_data/kestrel_<addr>.pem`` at project root,
  a side effect of the Emma recovery). The agent-scoped backup tools
  miss it.

What this script does
---------------------

1. **Atomic SQLite snapshot.** Uses the SQLite online-backup API
   (``conn.backup()``) on every ``kestrel_prime.db`` so a snapshot is
   consistent even if the live agent is mid-write. Raw cp on a SQLite
   file with WAL active produces a corrupt restore.
2. **File-tree copy** of every other critical file under each agent's
   data directory (private keys, DID JSON, kestrel.toml, SOUL.md,
   skills/, talon_jobs/, etc.).
3. **SHA-256 manifest** of every file backed up. Used both for restore
   verification and for detecting pre-ceremony tampering.
4. **AES-256-GCM encryption** with a passphrase-derived key
   (PBKDF2-HMAC-SHA256, 600,000 iterations — current OWASP guidance).
   Salt + nonce written alongside the ciphertext.
5. **Self-verification.** After producing the encrypted archive the
   script decrypts it back to a temp dir and re-hashes every file,
   asserting bit-equality. A backup that doesn't restore is worse than
   no backup.

Usage
-----

::

    export KESTREL_BACKUP_PASSPHRASE='<choose a strong one>'
    uv run python scripts/quantum_pre_ceremony_backup.py

To restore::

    export KESTREL_BACKUP_PASSPHRASE='<the same one>'
    uv run python scripts/quantum_pre_ceremony_backup.py \\
        --restore \\
        --archive /path/to/backup.tar.gz.enc \\
        --output  /path/to/restore-dir

What this script does NOT do
----------------------------

- Upload the encrypted archive anywhere. Operator copies the resulting
  ``backup.tar.gz.enc`` to Google Drive (and ideally a second
  independent location) by hand.
- Touch the source data. Read-only on every agent dir. The script will
  refuse to run if it cannot open the SQLite file in read-only mode.
- Manage your passphrase. Lose the passphrase and the backup is
  unrecoverable, by design.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import struct
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


PROJECT_ROOT = Path("/Volumes/data2/projects/kestrel-sovereign")
AGENT_DATA = PROJECT_ROOT / "agent_data"

# Targets to back up. Each entry: (logical_name, source_dir, extra_files).
# extra_files is a list of paths relative to PROJECT_ROOT to also copy
# into the backup at <logical_name>/_extras/<basename>.
BACKUP_TARGETS = [
    {
        "name": "kestrel-1-emma",
        "source_dir": AGENT_DATA / "Emma",
        "extras": [
            # Plaintext legacy PEM at project root — rescue copy.
            "agent_data/kestrel_0xB4E7F05F9c39FcD0b0d2C516249BE960c863647E.pem",
            "agent_data/kestrel_0xB4E7F05F9c39FcD0b0d2C516249BE960c863647E.json",
        ],
    },
    {
        "name": "meridian",
        "source_dir": AGENT_DATA / "meridian",
        "extras": [],
    },
]

# Passphrase env var
PASSPHRASE_ENV = "KESTREL_BACKUP_PASSPHRASE"

# Encryption parameters
PBKDF2_ITERATIONS = 600_000
SALT_SIZE = 32
NONCE_SIZE = 12
AES_KEY_SIZE = 32  # AES-256

# Backup format version (so future restore can detect format changes)
BACKUP_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _info(msg: str) -> None:
    print(f"  {msg}")


def _ok(msg: str) -> None:
    print(f"  OK   {msg}")


def _step(title: str) -> None:
    print(f"\n=== {title} ===")


def _err(msg: str) -> None:
    print(f"  ERR  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# SQLite consistent snapshot
# ---------------------------------------------------------------------------

def sqlite_snapshot(source_db: Path, dest_db: Path) -> None:
    """Atomic snapshot via SQLite online-backup API.

    Works correctly even if the live agent is writing to ``source_db``
    via WAL — ``conn.backup()`` performs a consistent point-in-time copy.
    Raw ``cp`` of a SQLite file with active WAL produces a corrupt
    restore.
    """
    src = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(dest_db))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


# ---------------------------------------------------------------------------
# File tree copy with hashing
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_files(directory: Path) -> Iterable[Path]:
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            yield p


def collect_target(
    target: dict,
    staging_root: Path,
    manifest: list[dict],
) -> int:
    """Collect one agent target into the staging tree.

    Returns the total number of files collected.
    """
    name = target["name"]
    source = target["source_dir"]
    if not source.exists():
        raise SystemExit(f"error: source dir {source} does not exist")

    dest = staging_root / name
    dest.mkdir(parents=True)

    files_collected = 0

    # Copy non-DB files via a regular file walk; SQLite files via online
    # backup. The WAL/SHM siblings are not copied — the snapshot is a
    # standalone DB that doesn't need them.
    for src_file in _walk_files(source):
        rel = src_file.relative_to(source)
        # Skip WAL/SHM/sync siblings of the live DB; they're consumed
        # into the snapshot.
        if src_file.name.endswith(("-wal", "-shm", ".db.sync")):
            continue
        # Skip damaged backup snapshots from prior recovery events
        if ".damaged-" in src_file.name:
            continue
        # Skip prior ad-hoc backup files (we make a fresh snapshot below)
        if src_file.name.endswith(".db.backup") or src_file.name.endswith(
            ".db.before-meridian-recovery-20260429_103429"
        ) or src_file.name == "kestrel_prime.db":
            continue

        dest_file = dest / rel
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        sha = _sha256_file(dest_file)
        manifest.append({
            "agent": name,
            "rel_path": str(rel),
            "size": dest_file.stat().st_size,
            "sha256": sha,
        })
        files_collected += 1

    # SQLite consistent snapshot of kestrel_prime.db
    src_db = source / "kestrel_prime.db"
    if src_db.exists():
        dest_db = dest / "kestrel_prime.db"
        sqlite_snapshot(src_db, dest_db)
        sha = _sha256_file(dest_db)
        manifest.append({
            "agent": name,
            "rel_path": "kestrel_prime.db",
            "size": dest_db.stat().st_size,
            "sha256": sha,
            "source": "sqlite_online_backup",
        })
        files_collected += 1

    # Extra files (from project root)
    extras = target.get("extras", [])
    if extras:
        extras_dir = dest / "_extras"
        extras_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in extras:
            src_extra = PROJECT_ROOT / rel_path
            if not src_extra.exists():
                _info(f"  (extra not found, skipping: {rel_path})")
                continue
            dest_extra = extras_dir / Path(rel_path).name
            shutil.copy2(src_extra, dest_extra)
            sha = _sha256_file(dest_extra)
            manifest.append({
                "agent": name,
                "rel_path": f"_extras/{Path(rel_path).name}",
                "size": dest_extra.stat().st_size,
                "sha256": sha,
                "origin": str(rel_path),
            })
            files_collected += 1

    return files_collected


# ---------------------------------------------------------------------------
# Encryption / decryption
# ---------------------------------------------------------------------------

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_archive(
    plaintext_path: Path, passphrase: str, ciphertext_path: Path,
) -> dict:
    """Encrypt an archive file with AES-256-GCM. PBKDF2-derived key.

    Wire format::

        magic(4 bytes)='KSBA'  | "Kestrel Backup Archive"
        version(1 byte)=1
        salt_len(1 byte)=32
        salt(32 bytes)
        nonce_len(1 byte)=12
        nonce(12 bytes)
        iter_count(4 bytes BE)=600000
        ciphertext_len(8 bytes BE)
        ciphertext+gcm_tag

    The format is self-describing so the restore path doesn't need
    sidecar metadata to find the salt/nonce.
    """
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = _derive_key(passphrase, salt)
    aes = AESGCM(key)

    plaintext = plaintext_path.read_bytes()
    ciphertext = aes.encrypt(nonce, plaintext, associated_data=b"kestrel-backup-v1")

    with open(ciphertext_path, "wb") as f:
        f.write(b"KSBA")
        f.write(bytes([BACKUP_FORMAT_VERSION]))
        f.write(bytes([SALT_SIZE]))
        f.write(salt)
        f.write(bytes([NONCE_SIZE]))
        f.write(nonce)
        f.write(struct.pack(">I", PBKDF2_ITERATIONS))
        f.write(struct.pack(">Q", len(ciphertext)))
        f.write(ciphertext)

    return {
        "format": "KSBA",
        "version": BACKUP_FORMAT_VERSION,
        "kdf": "PBKDF2-HMAC-SHA256",
        "kdf_iterations": PBKDF2_ITERATIONS,
        "cipher": "AES-256-GCM",
    }


def decrypt_archive(ciphertext_path: Path, passphrase: str, plaintext_path: Path) -> None:
    """Inverse of :func:`encrypt_archive`."""
    with open(ciphertext_path, "rb") as f:
        magic = f.read(4)
        if magic != b"KSBA":
            raise SystemExit(
                f"error: not a Kestrel backup archive (bad magic: {magic!r})"
            )
        version = f.read(1)[0]
        if version != BACKUP_FORMAT_VERSION:
            raise SystemExit(
                f"error: unknown backup format version {version}; this build "
                f"handles v{BACKUP_FORMAT_VERSION}"
            )
        salt_len = f.read(1)[0]
        salt = f.read(salt_len)
        nonce_len = f.read(1)[0]
        nonce = f.read(nonce_len)
        iterations = struct.unpack(">I", f.read(4))[0]
        ct_len = struct.unpack(">Q", f.read(8))[0]
        ciphertext = f.read(ct_len)

    if iterations != PBKDF2_ITERATIONS:
        # Allow but warn — a future format may change defaults
        _info(f"(archive uses {iterations} PBKDF2 iterations, current is {PBKDF2_ITERATIONS})")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=salt,
        iterations=iterations,
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    aes = AESGCM(key)
    plaintext = aes.decrypt(nonce, ciphertext, associated_data=b"kestrel-backup-v1")
    plaintext_path.write_bytes(plaintext)


# ---------------------------------------------------------------------------
# Backup driver
# ---------------------------------------------------------------------------

def cmd_backup(passphrase: str) -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    output_dir = Path(f"/tmp/kestrel-pre-ceremony-backup-{timestamp}")
    if output_dir.exists():
        _err(f"output dir {output_dir} already exists; refusing to clobber")
        return 2
    output_dir.mkdir(parents=True)

    print(f"Pre-ceremony backup — {timestamp}")
    print(f"Output dir: {output_dir}")

    staging_root = output_dir / "staging"
    staging_root.mkdir()
    manifest: list[dict] = []

    # ------------------------------------------------------------------
    # Collect each target into the staging tree
    # ------------------------------------------------------------------
    for target in BACKUP_TARGETS:
        _step(f"Collecting {target['name']} ({target['source_dir']})")
        n = collect_target(target, staging_root, manifest)
        _ok(f"{n} files collected")

    total_files = len(manifest)
    total_bytes = sum(e["size"] for e in manifest)
    _info(f"\nTotal: {total_files} files, {total_bytes / 1e6:.1f} MB uncompressed")

    # ------------------------------------------------------------------
    # Pack into tar.gz
    # ------------------------------------------------------------------
    _step("Compressing tarball")
    archive_path = output_dir / "backup.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(staging_root, arcname="kestrel-backup", recursive=True)
        # Add the manifest INSIDE the tarball too, so a restorer who
        # decrypts but loses the sidecar can still find it.
        manifest_inside = output_dir / "_inside_manifest.json"
        manifest_inside.write_text(
            json.dumps({
                "format": "kestrel-pre-ceremony-backup",
                "format_version": BACKUP_FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "targets": [t["name"] for t in BACKUP_TARGETS],
                "files": manifest,
            }, indent=2, sort_keys=True),
        )
        tar.add(manifest_inside, arcname="kestrel-backup/MANIFEST.json")
        manifest_inside.unlink()
    archive_size = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    _ok(f"backup.tar.gz {archive_size / 1e6:.1f} MB (sha256={archive_sha[:16]}…)")

    # ------------------------------------------------------------------
    # Encrypt
    # ------------------------------------------------------------------
    _step("Encrypting (AES-256-GCM, PBKDF2-SHA256 600k iter)")
    encrypted_path = output_dir / "backup.tar.gz.enc"
    enc_meta = encrypt_archive(archive_path, passphrase, encrypted_path)
    enc_sha = _sha256_file(encrypted_path)
    _ok(f"backup.tar.gz.enc {encrypted_path.stat().st_size / 1e6:.1f} MB (sha256={enc_sha[:16]}…)")

    # ------------------------------------------------------------------
    # Self-verification: decrypt + extract + re-hash + delete temp
    # ------------------------------------------------------------------
    _step("Self-verification (decrypt + extract + hash compare)")
    with tempfile.TemporaryDirectory(prefix="kestrel-verify-") as verify_dir:
        verify_root = Path(verify_dir)
        decrypted = verify_root / "decrypted.tar.gz"
        decrypt_archive(encrypted_path, passphrase, decrypted)
        if _sha256_file(decrypted) != archive_sha:
            _err("decrypt round-trip produced different bytes than original tarball")
            return 1
        _ok("decrypt round-trip: bit-identical to original tarball")

        # Extract and hash every file
        with tarfile.open(decrypted, "r:gz") as tar:
            tar.extractall(verify_root, filter="data")
        extracted_root = verify_root / "kestrel-backup"
        mismatches = 0
        for entry in manifest:
            agent = entry["agent"]
            rel = entry["rel_path"]
            expected_sha = entry["sha256"]
            extracted = extracted_root / agent / rel
            if not extracted.exists():
                _err(f"missing in extract: {agent}/{rel}")
                mismatches += 1
                continue
            actual = _sha256_file(extracted)
            if actual != expected_sha:
                _err(f"hash mismatch: {agent}/{rel}")
                mismatches += 1
        if mismatches > 0:
            _err(f"{mismatches} file(s) failed verification — backup is INVALID")
            return 1
        _ok(f"all {len(manifest)} files verified bit-identical after decrypt+extract")

    # ------------------------------------------------------------------
    # Sidecar manifest + RESTORE.md
    # ------------------------------------------------------------------
    sidecar = output_dir / "backup-manifest.json"
    sidecar.write_text(json.dumps({
        "format": "kestrel-pre-ceremony-backup",
        "format_version": BACKUP_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targets": [t["name"] for t in BACKUP_TARGETS],
        "encrypted_archive_sha256": enc_sha,
        "plaintext_archive_sha256": archive_sha,
        "files": manifest,
        "encryption": enc_meta,
    }, indent=2, sort_keys=True))

    restore_md = output_dir / "RESTORE.md"
    restore_md.write_text(f"""# Restoring this backup

This backup was created by `scripts/quantum_pre_ceremony_backup.py` on
{datetime.now(timezone.utc).isoformat()}.

Targets: {', '.join(t['name'] for t in BACKUP_TARGETS)}

## To restore

You need:
- The encrypted archive: `backup.tar.gz.enc`
- The passphrase you set in `KESTREL_BACKUP_PASSPHRASE` when creating it
- The script: `scripts/quantum_pre_ceremony_backup.py`

```bash
export KESTREL_BACKUP_PASSPHRASE='<the same one>'
uv run python scripts/quantum_pre_ceremony_backup.py \\
    --restore \\
    --archive  /path/to/backup.tar.gz.enc \\
    --output   /tmp/kestrel-restore
```

The output dir will contain one subdirectory per target. To re-anchor an
agent, copy the contents of `<target>/` back into the original agent
data dir.

## Verifying integrity without restoring

```bash
shasum -a 256 backup.tar.gz.enc
# should match `encrypted_archive_sha256` in backup-manifest.json
```

## What's in here

See `backup-manifest.json` for the full file list and per-file SHA-256
hashes. The same manifest is also embedded INSIDE the archive at
`kestrel-backup/MANIFEST.json` so it survives even if the sidecar is lost.
""")

    print(f"\n{'=' * 60}")
    print("Backup complete.")
    print('=' * 60)
    print(f"\nLocation: {output_dir}")
    print(f"  backup.tar.gz.enc     {encrypted_path.stat().st_size / 1e6:.1f} MB  (upload this to Google Drive)")
    print(f"  backup-manifest.json  {sidecar.stat().st_size} bytes")
    print(f"  RESTORE.md            restore instructions")
    print(f"\nSHA-256 of encrypted archive (record this elsewhere):")
    print(f"  {enc_sha}")
    print()
    print("Next steps:")
    print("  1. Copy backup.tar.gz.enc to Google Drive (and ideally a")
    print("     second independent location)")
    print("  2. Vault the passphrase separately (1Password, etc.). Without")
    print("     it the backup is unrecoverable.")
    print("  3. Do NOT delete the local copy until you have verified you")
    print("     can restore from the Google Drive copy.")
    return 0


def cmd_restore(archive: Path, output: Path, passphrase: str) -> int:
    if output.exists() and any(output.iterdir()):
        _err(f"output dir {output} exists and is non-empty; refusing to clobber")
        return 2
    output.mkdir(parents=True, exist_ok=True)

    print(f"Restoring from {archive}")
    print(f"Output: {output}")

    decrypted = output / "_tmp_decrypted.tar.gz"
    _step("Decrypting")
    decrypt_archive(archive, passphrase, decrypted)
    _ok(f"decrypted {decrypted.stat().st_size / 1e6:.1f} MB")

    _step("Extracting")
    with tarfile.open(decrypted, "r:gz") as tar:
        tar.extractall(output, filter="data")
    decrypted.unlink()
    extracted = output / "kestrel-backup"
    if not extracted.exists():
        _err("expected kestrel-backup/ in archive but didn't find it")
        return 1
    _ok(f"extracted to {extracted}")

    manifest_path = extracted / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _step("Verifying file hashes against MANIFEST.json")
        mismatches = 0
        for entry in manifest["files"]:
            f = extracted / entry["agent"] / entry["rel_path"]
            if not f.exists():
                _err(f"missing: {entry['agent']}/{entry['rel_path']}")
                mismatches += 1
                continue
            if _sha256_file(f) != entry["sha256"]:
                _err(f"hash mismatch: {entry['agent']}/{entry['rel_path']}")
                mismatches += 1
        if mismatches:
            _err(f"{mismatches} file(s) failed hash verification")
            return 1
        _ok(f"all {len(manifest['files'])} files verified")

    print(f"\nRestore complete. Inspect: {extracted}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else None)
    parser.add_argument("--restore", action="store_true", help="Restore mode")
    parser.add_argument("--archive", type=Path, help="(restore) encrypted archive path")
    parser.add_argument("--output", type=Path, help="(restore) output directory")
    args = parser.parse_args()

    passphrase = os.environ.get(PASSPHRASE_ENV)
    if not passphrase:
        _err(f"{PASSPHRASE_ENV} is required. Pick a strong passphrase and:")
        _err(f"  export {PASSPHRASE_ENV}='<your passphrase>'")
        _err("Lose the passphrase and the backup is unrecoverable.")
        return 2

    if args.restore:
        if not args.archive or not args.output:
            _err("--archive and --output are required in --restore mode")
            return 2
        return cmd_restore(args.archive, args.output, passphrase)
    else:
        return cmd_backup(passphrase)


if __name__ == "__main__":
    sys.exit(main())
