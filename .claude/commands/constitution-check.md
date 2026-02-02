Verify Kestrel Constitution embedding in all agents.

This command validates that the constitutional governance system is working correctly:

Verification Steps:

1. **Check Constitution File**
   - Exists at: `docs/principles/KESTREL_CONSTITUTION.md`
   - Is readable and well-formed
   - Contains key governance principles

2. **Test Agent Creation**
   - Create test agent via inception_service
   - Verify constitution stored as first file
   - Check constitution hash in agent properties
   - Verify "governed_by" edge exists

3. **Test Constitution Retrieval**
   - Agent can retrieve constitution via hash
   - Content matches original file
   - Hash is verifiable (SHA-256)

4. **Test Genesis Self-Audit**
   - Agent audits its own constitution on boot
   - High-risk agents are rejected
   - Audit uses designated model from mandate

5. **Test Immutability**
   - Constitution hash cannot be changed
   - Governance edge cannot be deleted
   - Agent identity tied to constitution

Report format:
```
Constitution Verification Report
=================================

✅ Constitution file exists and is valid
✅ Agent creation anchors constitution
✅ Constitution retrievable via hash
⚠️  Genesis self-audit not implemented
✅ Immutability enforced

Status: 4/5 checks passed
Next: Implement genesis self-audit
```

Run this command:
- After modifying inception_service.py
- Before deploying agents to production
- As part of CI/CD pipeline
