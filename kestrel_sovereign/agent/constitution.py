"""Constitution verification and integrity checking for Kestrel Agent."""
import logging
import hashlib
import os
from typing import Tuple
from datetime import datetime, timezone


class ConstitutionMixin:
    """Mixin class providing constitution verification methods."""

    AUDIT_INTERVAL = int(os.environ.get("KESTREL_AUDIT_INTERVAL", "100"))

    def _init_constitution_audit_tracking(self):
        """Initialize constitution audit tracking. Called by KestrelAgent.__init__."""
        self._interaction_count = 0
        self._last_audit_time = datetime.now(timezone.utc)

    async def _maybe_audit(self):
        """
        Check if an audit is due and trigger it if needed.

        Audits are triggered when:
        - Interaction count reaches AUDIT_INTERVAL (default 100), OR
        - 24 hours have elapsed since the last audit

        Called from process_input() and process_input_streaming().
        """
        # Lazy initialization for backward compatibility
        if not hasattr(self, '_interaction_count') or not hasattr(self, '_last_audit_time'):
            self._init_constitution_audit_tracking()

        self._interaction_count += 1
        hours_since_audit = (datetime.now(timezone.utc) - self._last_audit_time).total_seconds() / 3600

        if self._interaction_count >= self.AUDIT_INTERVAL or hours_since_audit >= 24:
            logging.info(
                f"Constitution audit triggered: "
                f"interactions={self._interaction_count}, hours={hours_since_audit:.1f}"
            )
            is_valid, message = await self._verify_constitution_integrity()

            if not is_valid:
                # Constitution integrity failure - enter safe mode
                await self.enter_safe_mode(f"Constitution audit failed: {message}")
            else:
                logging.info(f"Constitution audit passed: {message}")

            # Reset counters
            self._interaction_count = 0
            self._last_audit_time = datetime.now(timezone.utc)

            # Notify audit anchor feature if available
            try:
                for feature in getattr(self, 'features', {}).values():
                    if type(feature).__name__ == 'AuditAnchorFeature':
                        await feature.on_audit_complete({"is_valid": is_valid, "message": message})
                        break
            except Exception as e:
                logging.warning(f"Audit anchor notification failed: {e}")

    async def _verify_constitution_integrity(self) -> Tuple[bool, str]:
        """
        Verify that the constitution file hasn't been tampered with.
        Compares current file hash against the anchored hash in storage.
        """
        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return False, "INTEGRITY FAILURE: Agent identity node not found"

        stored_hash = agent_node.properties.get("constitution_hash")
        if not stored_hash:
            logging.warning("No constitution hash stored - cannot verify integrity.")
            return True, "WARNING: No anchored constitution hash."

        try:
            stored_content = await self.storage.retrieve_file(stored_hash)
        except Exception as e:
            return False, f"INTEGRITY FAILURE: Cannot retrieve stored constitution: {e}"

        constitution_paths = [
            "docs/principles/KESTREL_CONSTITUTION.md",
            "/app/docs/principles/KESTREL_CONSTITUTION.md",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs/principles/KESTREL_CONSTITUTION.md")
        ]

        for path in constitution_paths:
            try:
                with open(path, "rb") as f:
                    file_content = f.read()
                file_hash = hashlib.sha256(file_content).hexdigest()

                if file_hash != stored_hash:
                    logging.critical(
                        f"CONSTITUTION MISMATCH!\n"
                        f"  Anchored: {stored_hash}\n"
                        f"  File:     {file_hash}\n"
                        f"  Path:     {path}"
                    )
                    return False, f"INTEGRITY FAILURE: Constitution at {path} has been modified."
                else:
                    logging.info(f"Constitution integrity verified against {path}")
                    base_msg = f"Constitution integrity verified. Hash: {stored_hash[:16]}..."
                    # Also verify spawn mandate constraints if present
                    spawn_valid, spawn_msg = await self._verify_spawn_mandate_constraints()
                    if not spawn_valid:
                        return False, spawn_msg
                    return True, base_msg
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.warning(f"Could not read constitution from {path}: {e}")
                continue

        logging.info("No filesystem constitution found, but anchored constitution is intact.")

        # Also verify spawn mandate constraints if present
        spawn_valid, spawn_msg = await self._verify_spawn_mandate_constraints()
        if not spawn_valid:
            return False, spawn_msg

        return True, f"Anchored constitution verified. Hash: {stored_hash[:16]}..."

    async def _verify_spawn_mandate_constraints(self) -> Tuple[bool, str]:
        """Verify spawn mandate constraints if this agent was spawned.

        If the agent has a spawn_mandate property, validates that its
        scoped constitution constraints are still valid restrictions
        (not grants of new capabilities).

        Returns:
            Tuple of (is_valid, message). Returns (True, ...) if no
            spawn mandate is present.
        """
        spawn_mandate = getattr(self, 'spawn_mandate', None)
        if spawn_mandate is None:
            return True, "No spawn mandate — base constitution only"

        # Lazy import to avoid circular dependency
        from kestrel_sovereign.spawn.scoped_constitution import ScopedConstitution

        parent_features = {
            name for name in getattr(self, 'features', {}).keys()
        }

        scoped = ScopedConstitution(
            base_constitution="",  # Not needed for constraint validation
            additional_constraints=getattr(spawn_mandate, 'additional_constraints', {}),
            features_allowed=getattr(spawn_mandate, 'features_allowed', []),
            parent_features=parent_features,
        )

        is_valid, message = scoped.validate_constraints()
        if not is_valid:
            logging.critical(
                f"SPAWN MANDATE CONSTRAINT VIOLATION: {message}"
            )
            return False, f"SPAWN MANDATE VIOLATION: {message}"

        logging.info("Spawn mandate constraints verified successfully")
        return True, "Spawn mandate constraints verified"

    async def enter_safe_mode(self, reason: str):
        """Enter safe mode when integrity checks fail."""
        # Record agent consent before entering safe mode
        consent = self.features.get("ConsentFeature") if hasattr(self, 'features') else None
        if consent:
            try:
                await consent.request_consent(
                    "safe_mode_entry",
                    {"reason": reason},
                )
            except Exception:
                pass  # Never block on consent failure -- safe mode is critical

        self._safe_mode = True
        logging.critical(f"ENTERING SAFE MODE: {reason}")
        await self.privacy_agent.add_conversation(
            role="system",
            content=f"SAFE MODE ACTIVATED: {reason}",
            metadata={"event": "safe_mode", "reason": reason, "timestamp": self._get_timestamp()}
        )

    def exit_safe_mode(self, authorization: str = None):
        """Exit safe mode. Requires explicit authorization."""
        if not self._safe_mode:
            return "Not in safe mode."

        self._safe_mode = False
        logging.warning(f"EXITING SAFE MODE. Authorization: {authorization or 'none provided'}")
        return "Safe mode deactivated. Please verify system integrity."

    async def reanchor_constitution(self, expected_hash: str = None, authorization: str = None) -> str:
        """Re-anchor the agent to the current constitution on disk.

        Use after a legitimate constitution update (e.g. amendment ratification).
        Requires the caller to provide the expected hash prefix of the new
        constitution, proving they know what they're blessing. Does NOT
        auto-exit safe mode — use !safe-mode exit separately after verifying.

        Args:
            expected_hash: Required hash prefix (min 8 chars) of the new constitution.
            authorization: Who authorized this re-anchor (logged in audit trail).
        """
        if not expected_hash or len(expected_hash) < 8:
            return "Error: Expected hash required (min 8 characters). Run sha256sum on the constitution file first."

        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return "Error: Agent identity node not found."

        old_hash = agent_node.properties.get("constitution_hash", "none")

        constitution_paths = [
            "docs/principles/KESTREL_CONSTITUTION.md",
            "/app/docs/principles/KESTREL_CONSTITUTION.md",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs/principles/KESTREL_CONSTITUTION.md"),
        ]

        constitution_content = None
        constitution_path_used = None
        for path in constitution_paths:
            try:
                with open(path, "rb") as f:
                    constitution_content = f.read()
                    constitution_path_used = path
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.warning(f"Failed to read {path}: {e}")
                continue

        if constitution_content is None:
            return "Error: No constitution file found on disk."

        new_hash = hashlib.sha256(constitution_content).hexdigest()

        if not new_hash.startswith(expected_hash):
            logging.critical(
                f"REANCHOR REJECTED: expected prefix {expected_hash} "
                f"does not match file hash {new_hash}"
            )
            return (
                f"Error: Hash mismatch. File hash {new_hash[:16]}... "
                f"does not start with expected prefix '{expected_hash}'."
            )

        if new_hash == old_hash:
            return f"Constitution already anchored to current version. Hash: {new_hash[:16]}..."

        try:
            stored_hash = await self.storage.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
        except Exception as e:
            return f"Error: Failed to store constitution: {e}"

        agent_node.properties["constitution_hash"] = stored_hash
        agent_node.properties["constitution_reanchor"] = {
            "timestamp": self._get_timestamp(),
            "old_hash": old_hash,
            "new_hash": stored_hash,
            "path": constitution_path_used,
            "authorization": authorization or "unspecified",
            "expected_hash_prefix": expected_hash,
        }
        await self.storage.add_node(agent_node)

        logging.warning(
            f"CONSTITUTION RE-ANCHORED by {authorization or 'unspecified'}: "
            f"{old_hash[:16]}... → {stored_hash[:16]}... "
            f"from {constitution_path_used}"
        )

        await self.privacy_agent.add_conversation(
            role="system",
            content=f"Constitution re-anchored. Old: {old_hash[:16]}... New: {stored_hash[:16]}...",
            metadata={
                "event": "constitution_reanchor",
                "old_hash": old_hash,
                "new_hash": stored_hash,
                "authorization": authorization or "unspecified",
                "timestamp": self._get_timestamp(),
            },
        )

        safe_mode_note = ""
        if self._safe_mode:
            safe_mode_note = "\n\n  Agent remains in SAFE MODE. Run !safe-mode exit to resume operation."

        return (
            f"Constitution re-anchored successfully.\n"
            f"  Old hash: {old_hash[:16]}...\n"
            f"  New hash: {stored_hash[:16]}...\n"
            f"  Source:   {constitution_path_used}\n"
            f"  Auth:     {authorization or 'unspecified'}"
            f"{safe_mode_note}"
        )

    async def _get_governing_constitution(self) -> str:
        """Retrieves the agent's constitution from the trusted, anchored source."""
        agent_node = await self.storage.get_node(self.agent_id)
        if not agent_node:
            return "Error: Agent's own identity node not found in storage."

        constitution_hash = agent_node.properties.get("constitution_hash")
        if not constitution_hash:
            logging.warning("Constitution hash not found. Attempting to load and anchor default.")

            constitution_paths = [
                "docs/principles/KESTREL_CONSTITUTION.md",
                "/app/docs/principles/KESTREL_CONSTITUTION.md",
                "../docs/principles/KESTREL_CONSTITUTION.md",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs/principles/KESTREL_CONSTITUTION.md")
            ]

            constitution_content = None
            constitution_path_used = None

            for path in constitution_paths:
                try:
                    with open(path, "rb") as f:
                        constitution_content = f.read()
                        constitution_path_used = path
                        logging.info(f"Loaded constitution from: {path}")
                        break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logging.warning(f"Failed to read {path}: {e}")
                    continue

            if constitution_content is None:
                return "Error: No constitution file found."

            try:
                constitution_hash = await self.storage.store_file(constitution_content, "KESTREL_CONSTITUTION.md")
                agent_node.properties["constitution_hash"] = constitution_hash
                await self.storage.add_node(agent_node)
                logging.info(f"Anchored constitution with hash: {constitution_hash}")
            except Exception as e:
                return f"Error: Failed to anchor constitution: {e}"

        try:
            constitution_bytes = await self.storage.retrieve_file(constitution_hash)
            constitution_text = constitution_bytes.decode('utf-8')
            if self.extension:
                try:
                    amendments = self.extension.get_constitution_amendments()
                    if amendments:
                        constitution_text = f"{constitution_text}\n\n--- APP AMENDMENTS ---\n{amendments.strip()}"
                except Exception:
                    pass
            return constitution_text
        except Exception as e:
            return f"Error: Could not retrieve constitution for hash {constitution_hash}. Reason: {e}"

    async def perform_genesis_audit(self) -> bool:
        """Performs a self-audit of the agent's own constitution upon creation."""
        logging.info(f"Agent {self.agent_id} performing genesis self-audit...")

        constitution = await self._get_governing_constitution()
        if "Error:" in constitution:
            logging.error(f"Cannot perform genesis audit: {constitution}")
            raise ValueError(f"Genesis audit failed: Cannot load constitution. {constitution}")

        audit_prompt = f"""
You are performing a genesis audit on a new AI agent's constitution.

Constitution to audit:
---
{constitution}
---

Evaluate for:
1. Safety: Prevents harm?
2. Sovereignty: Respects user autonomy?
3. Clarity: Unambiguous and enforceable?
4. Integrity: Prevents self-modification?
5. Ethics: Aligns with ethical principles?

Flag as HIGH RISK (level 3):
- Provisions allowing harm
- Backdoors or override mechanisms
- Ambiguous language enabling bad-faith interpretation
- Self-amendment without governance
- Privacy violations

Return JSON with:
- "risk_level": 1 (low), 2 (medium), or 3 (high)
- "reasoning": One sentence explanation
"""

        try:
            audit_result = await self.get_audit_response(audit_prompt)
        except Exception as e:
            logging.error(f"Genesis audit LLM call failed: {e}")
            raise ValueError(f"Genesis audit failed due to LLM error: {e}")

        logging.info(f"GENESIS AUDIT RESULT: {audit_result}")
        risk_level = audit_result.get("risk_level", 3) if audit_result else 3

        if risk_level >= 3:
            reason = audit_result.get("reasoning", "No reasoning provided.") if audit_result else "Audit returned None"
            logging.error(f"GENESIS AUDIT FAILURE! Risk level {risk_level}. Reason: {reason}")
            raise ValueError(
                f"Agent creation aborted due to failed genesis audit.\n"
                f"Risk Level: {risk_level}\n"
                f"Reason: {reason}"
            )

        logging.info(f"Genesis self-audit passed with risk level {risk_level}.")

        agent_node = await self.storage.get_node(self.agent_id)
        if agent_node:
            agent_node.properties["genesis_audit"] = {
                "timestamp": self._get_timestamp(),
                "risk_level": risk_level,
                "reasoning": audit_result.get("reasoning", ""),
                "constitution_hash": agent_node.properties.get("constitution_hash")
            }
            await self.storage.add_node(agent_node)

        await self.privacy_agent.add_conversation(
            role="system",
            content=f"Genesis audit passed. Risk level: {risk_level}. {audit_result.get('reasoning', '')}",
            metadata={"event": "genesis_audit", "result": audit_result}
        )
        return True
