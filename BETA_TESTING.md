# Kestrel Beta Testing Guide

## 🎯 Who Should Beta Test

### ✅ Perfect Beta Testers:
- **AI/Cryptography Researchers** - Novel constitutional model and DID system
- **Privacy-First Developers** - Building apps with strong sovereignty requirements
- **Autonomous Agent Builders** - Need A2A protocol and self-improvement systems
- **Security Researchers** - Want to audit/break the DID and constitutional layers

### ⚠️ Not Ideal Beta Testers:
- General consumers expecting polished applications
- Developers wanting messenger bot features (use OpenClaw)
- Teams requiring production-ready, stable APIs
- Anyone needing voice/multi-channel/browser automation

---

## 🧪 What to Test

### High Priority (Core Value Props):
1. **Constitutional AI Framework**
   - Create an agent via `inception_service.py`
   - Observe the genesis self-audit process
   - Try to create an agent that violates constitutional principles
   - Test hierarchical permissions (DENY → RESTRICTED → APPROVED → UNRESTRICTED)

2. **Privacy Modes**
   - Test all 5 privacy levels (EPHEMERAL, ISOLATED, ANONYMOUS, NORMAL, PUBLIC)
   - Verify ephemeral mode doesn't store conversations
   - Check that encrypted storage works when enabled
   - Test privacy mode transitions

3. **Cryptographic Identity (DIDs)**
   - Generate agent DIDs
   - Export/import agent identity packages
   - Verify portable identity across systems
   - **Note**: Verification layer is WIP - test what works, report what doesn't

4. **Multi-LLM Support**
   - Test fallback between local (Ollama) and cloud providers
   - Try different provider combinations (Anthropic, OpenAI, Gemini)
   - Report any model selection issues

5. **Agent Economics**
   - Create agent wallets
   - Test multi-currency support (FIL, USDC, USDT, ETH)
   - Explore autonomous funding concepts
   - **Note**: This is infrastructure - no marketplace yet

### Medium Priority (Nice to Have):
- A2A protocol (agent-to-agent communication)
- Knowledge graph storage and retrieval
- RAG pipeline for document Q&A
- Reflection system (self-improvement)
- Feature discovery and loading

---

## ❌ What NOT to Test

### Don't Waste Time On:
- **Production Deployment** - This is beta, not production-ready
- **Mission-Critical Systems** - Don't use for anything that can't fail
- **Multi-Channel Integration** - Not implemented (that's OpenClaw's domain)
- **Voice/Browser Features** - Not on roadmap for core framework
- **Perfect Documentation** - We know docs are incomplete, focus on functionality

---

## 🐛 How to Report Issues

### Good Bug Reports Include:
1. **Environment**: OS, Python version, uv version
2. **Steps to Reproduce**: Exact commands you ran
3. **Expected Behavior**: What you thought would happen
4. **Actual Behavior**: What actually happened
5. **Logs**: Relevant error messages or stack traces
6. **Severity**: Blocker / Major / Minor / Enhancement

### GitHub Issue Template:
```markdown
**Environment**:
- OS: macOS 14.2 / Ubuntu 22.04 / Windows 11
- Python: 3.11.5
- Kestrel version: v0.1.8

**Steps to Reproduce**:
1. uv run python inception_service.py
2. [your steps]

**Expected**: Agent should create successfully

**Actual**: Genesis audit failed with error X

**Logs**:
```
[paste relevant logs]
```

**Severity**: Major (blocking basic functionality)
```

### Where to Report:
- **Bugs**: [GitHub Issues](https://github.com/KestrelSovereignAI/kestrel-sovereign/issues)
- **Feature Requests**: GitHub Discussions
- **Security Issues**: Email maintainers directly (see SECURITY.md)

---

## 💡 Beta Testing Best Practices

### 1. Start with Quick Start
Follow the README quick start guide before diving into complex features. Verify basic functionality first.

### 2. Test One Thing at a Time
Isolate variables. If you change multiple configs and something breaks, you won't know what caused it.

### 3. Document Your Setup
Keep notes on your configuration, especially `llm_config.toml` and any environment variables.

### 4. Check Existing Issues First
Before filing a bug, search GitHub issues to see if it's already reported.

### 5. Provide Context
"It doesn't work" is not helpful. Explain what you were trying to achieve and why.

### 6. Be Patient with Responses
We're a small team. Complex issues may take time to investigate.

---

## 🛠️ Common Issues & Troubleshooting

### Issue: "Cannot connect to Ollama"
**Solution**: Make sure `ollama serve` is running and accessible at `http://localhost:11434`

### Issue: "API key errors with cloud providers"
**Solution**: Check `llm_config.toml` and ensure API keys are valid. Consider using `.env` file instead of hardcoding.

### Issue: "Database locked" errors
**Solution**: Close other processes accessing the same SQLite database. Use PostgreSQL for multi-process scenarios.

### Issue: "Genesis audit fails"
**Solution**: This is expected if agent violates constitutional principles. Check `KESTREL_CONSTITUTION.md` for rules.

### Issue: "DID verification fails"
**Solution**: Known limitation - verification layer is WIP. Identity generation works, verification is incomplete.

### Issue: "Import errors for optional dependencies"
**Solution**: Run `uv sync` to install all dependencies. Some features require extras (e.g., `[full]`).

---

## 📊 What We're Looking For

### Valuable Feedback:
✅ "Constitutional audit rejected my agent for reason X - is this expected?"
✅ "Privacy mode Y doesn't work as documented - here's how to reproduce"
✅ "DID generation works but feature Z is missing - here's my use case"
✅ "Integration with framework X would be valuable - here's why"

### Less Valuable Feedback:
❌ "This doesn't work" (no details)
❌ "Why isn't feature X implemented?" (without proposing to help)
❌ "When will this be production-ready?" (no timeline commitments)
❌ "Can you add voice/multi-channel?" (use OpenClaw for that)

---

## 🤝 Contributing Beyond Testing

### We're Seeking:
- **Code Contributors**: PRs for bug fixes, features, documentation
- **Technical Co-Maintainers**: Hardcore devs to lead OS development
- **Integration Partners**: Want to combine Kestrel with OpenClaw or other frameworks
- **Research Collaborators**: Academic papers on constitutional AI, agent economics

### How to Contribute:
1. Read `CONTRIBUTING.md` for guidelines
2. Start with small PRs (docs, tests, bug fixes)
3. Discuss major changes in GitHub Discussions first
4. Follow code review feedback
5. Be patient - quality > speed

---

## 🎓 Learning Resources

### Understanding Kestrel:
- **Constitution**: `docs/principles/KESTREL_CONSTITUTION.md`
- **Architecture**: `docs/architecture/AGENT_ECOSYSTEM.md`
- **Privacy Modes**: `docs/architecture/PRIVACY_MODES.md`
- **DIDs**: `docs/architecture/CRYPTOGRAPHIC_ANCHORING.md`

### External Resources:
- [W3C DIDs Spec](https://www.w3.org/TR/did-core/) - Decentralized Identifiers standard
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073) - Original research
- [OpenClaw Docs](https://github.com/openclaw/openclaw) - Complementary framework

---

## ⏱️ Beta Timeline

### Phase 1 (Current): Early Access
- Goal: Find critical bugs, validate core features
- Focus: Constitutional AI, privacy modes, DIDs
- Duration: 4-6 weeks

### Phase 2: Stability & Integration
- Goal: API stabilization, OpenClaw integration
- Focus: A2A protocol, multi-agent coordination
- Duration: TBD based on Phase 1 feedback

### Phase 3: Production Hardening
- Goal: Documentation, security audits, v1.0 prep
- Focus: Enterprise governance, compliance features
- Duration: TBD

**Note**: These are aspirational timelines. Actual progress depends on community engagement and resources.

---

## ❓ FAQ

**Q: Is Kestrel production-ready?**
A: No. v0.1.8 is beta. Use for research, experimentation, learning. Not for production apps.

**Q: Will APIs change?**
A: Yes. We'll document breaking changes but can't guarantee stability before v1.0.

**Q: Can I build commercial apps on Kestrel?**
A: Yes (Apache 2.0 license). But understand beta risks - APIs may shift, features may break.

**Q: How does Kestrel relate to OpenClaw?**
A: Complementary. OpenClaw = connectivity (channels, voice, devices). Kestrel = sovereignty (identity, governance, economics). Use together.

**Q: Why not just contribute to OpenClaw?**
A: Different layer of the stack. OpenClaw is Node.js/TypeScript, Kestrel is Python. Different architectural goals.

**Q: What's the business model?**
A: Framework stays open source (Apache 2.0). Exploring hosting, companion apps. Think MongoDB model - open core, commercial support.

**Q: Can I become a maintainer?**
A: Yes! Prove yourself with quality contributions, then we'll discuss. We're actively seeking technical co-maintainers.

**Q: What if I find a security vulnerability?**
A: Please report responsibly via email (see SECURITY.md). We'll acknowledge within 48 hours and coordinate disclosure.

---

## 📞 Getting Help

### Community Support:
- **GitHub Discussions**: General questions, architecture discussions
- **GitHub Issues**: Bug reports, feature requests
- **Discord** (coming soon): Real-time chat with other beta testers

### Direct Contact:
- **Security**: See SECURITY.md for responsible disclosure
- **Partnerships**: Open a GitHub Discussion
- **Press/Media**: Open a GitHub Discussion

---

## 🙏 Thank You

Beta testing is hard work. You're helping build public infrastructure for sovereign AI agents. Your feedback directly shapes the roadmap.

**Remember:**
- Be patient with bugs
- Be thorough with reports
- Be collaborative with maintainers
- Be honest with feedback

Together, we're building something that matters.

---

*Last Updated: February 3, 2026*
*Kestrel v0.1.8 Beta*
