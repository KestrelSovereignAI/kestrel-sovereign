"""Constitution verification and integrity checking for Kestrel Agent."""
import logging
import hashlib
import os
from typing import Tuple


class ConstitutionMixin:
    """Mixin class providing constitution verification methods."""

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
                    return True, f"Constitution integrity verified. Hash: {stored_hash[:16]}..."
            except FileNotFoundError:
                continue
            except Exception as e:
                logging.warning(f"Could not read constitution from {path}: {e}")
                continue

        logging.info("No filesystem constitution found, but anchored constitution is intact.")
        return True, f"Anchored constitution verified. Hash: {stored_hash[:16]}..."

    async def enter_safe_mode(self, reason: str):
        """Enter safe mode when integrity checks fail."""
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
