Test all 5 Kestrel privacy modes end-to-end.

This command validates the complete privacy system:

Test Each Mode:

**1. EPHEMERAL Mode**
- Create agent with EPHEMERAL mode
- Send test messages
- Verify NOTHING stored in database
- Verify NOTHING stored in files
- Verify only local LLM used
- Test session clear on exit

**2. ISOLATED Mode**
- Create agent with ISOLATED mode
- Send test messages
- Verify stored in temporary database
- Verify temp storage cleared on session end
- Test backup to cache-only
- Test promote to ANONYMOUS

**3. ANONYMOUS Mode**
- Create agent with ANONYMOUS mode
- Send messages with PII (names, addresses, emails)
- Verify PII filtered/redacted before storage
- Verify cloud LLM allowed
- Test encrypted backups required

**4. NORMAL Mode**
- Create agent with NORMAL mode
- Verify full storage working
- Verify all features enabled
- Test backups working
- Verify cloud LLM allowed

**5. PUBLIC Mode**
- Create agent with PUBLIC mode
- Verify sharing features enabled
- Test export functionality
- Verify public accessibility

Mode Transitions:
- Test switching from EPHEMERAL → NORMAL
- Test switching from NORMAL → EPHEMERAL (with warning)
- Test switching from PUBLIC → EPHEMERAL (requires confirmation)
- Verify previous messages preserved during mode switch

UI Testing:
- Verify privacy indicator shows correct mode
- Test mode selector dropdown
- Verify color coding (EPHEMERAL=red, PUBLIC=blue)
- Test privacy status command

Report format:
```
Privacy System Test Report
===========================

Mode Testing:
✅ EPHEMERAL: No storage (verified)
✅ ISOLATED: Temporary storage (verified)
⚠️  ANONYMOUS: PII filtering needs improvement
✅ NORMAL: Full features (verified)
✅ PUBLIC: Sharing enabled (verified)

Mode Transitions:
✅ All transitions working
✅ Warnings displayed correctly

UI Components:
✅ Privacy indicator functional
✅ Mode selector working
✅ Color coding correct

Status: 12/13 tests passed (92%)
Issue: ANONYMOUS mode detects emails but misses phone numbers
```

Run with: /privacy-test
