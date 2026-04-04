"""
Evidence Compiler for Constitutional Council

Gathers all relevant information for council deliberation:
- Git diffs and commit history
- Test results
- Security assessments
- Architecture documentation
- Known risks
- Previous council decisions
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Evidence

logger = logging.getLogger(__name__)

# Project root for Kestrel
PROJECT_ROOT = Path(__file__).parent.parent.parent


async def compile_evidence(
    target: str = "general",
    since_commit: Optional[str] = None,
    include_full_docs: bool = False,
) -> Evidence:
    """
    Compile an evidence package for council deliberation.

    Args:
        target: What this evidence is for (e.g., "emma_genesis")
        since_commit: Only include changes since this commit
        include_full_docs: Include full doc contents (verbose)

    Returns:
        Evidence package ready for council review
    """
    evidence = Evidence(target=target)

    # Run gathering tasks in parallel
    tasks = [
        _gather_git_changes(since_commit),
        _gather_test_results(),
        _gather_security_assessment(target),
        _gather_architecture_docs(include_full_docs),
        _gather_risks(target),
        _gather_previous_decisions(),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Unpack results
    if not isinstance(results[0], Exception):
        evidence.code_changes, evidence.related_commits = results[0]
    else:
        logger.warning(f"Failed to gather git changes: {results[0]}")

    if not isinstance(results[1], Exception):
        test_data = results[1]
        evidence.test_results = test_data
        evidence.test_count = test_data.get("total", 0)
        evidence.test_passed = test_data.get("passed", 0)
        evidence.test_failed = test_data.get("failed", 0)
    else:
        logger.warning(f"Failed to gather test results: {results[1]}")

    if not isinstance(results[2], Exception):
        evidence.security_assessment = results[2]
    else:
        logger.warning(f"Failed to gather security assessment: {results[2]}")

    if not isinstance(results[3], Exception):
        evidence.architecture_docs, evidence.source_files = results[3]
    else:
        logger.warning(f"Failed to gather architecture docs: {results[3]}")

    if not isinstance(results[4], Exception):
        evidence.risks = results[4]
    else:
        logger.warning(f"Failed to gather risks: {results[4]}")

    if not isinstance(results[5], Exception):
        evidence.previous_decisions = results[5]
    else:
        logger.warning(f"Failed to gather previous decisions: {results[5]}")

    evidence.compiled_at = datetime.utcnow()
    return evidence


async def _gather_git_changes(
    since_commit: Optional[str] = None
) -> tuple[List[str], List[str]]:
    """Gather git diffs and commit history."""
    changes = []
    commits = []

    try:
        # Get recent commits
        if since_commit:
            cmd = ["git", "log", f"{since_commit}..HEAD", "--oneline", "-20"]
        else:
            cmd = ["git", "log", "--oneline", "-20"]

        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        commits = stdout.decode().strip().split("\n") if stdout else []

        # Get diff summary
        if since_commit:
            diff_cmd = ["git", "diff", "--stat", since_commit]
        else:
            diff_cmd = ["git", "diff", "--stat", "HEAD~10"]

        result = await asyncio.create_subprocess_exec(
            *diff_cmd,
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        if stdout:
            changes.append(stdout.decode().strip())

        # Get specific file changes for key files
        key_files = [
            "storage/encryption.py",
            "kestrel_agent.py",
            "inception_service.py",
        ]
        for filepath in key_files:
            if since_commit:
                file_diff_cmd = ["git", "diff", since_commit, "--", filepath]
            else:
                file_diff_cmd = ["git", "diff", "HEAD~5", "--", filepath]

            result = await asyncio.create_subprocess_exec(
                *file_diff_cmd,
                cwd=PROJECT_ROOT,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if stdout and len(stdout) < 5000:  # Only include if not too large
                changes.append(f"# {filepath}\n{stdout.decode().strip()}")

    except FileNotFoundError as e:
        logger.error(f"Git command not found: {e}", exc_info=True)
    except subprocess.SubprocessError as e:
        logger.error(f"Git subprocess error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Git gathering failed: {e}", exc_info=True)

    return changes, commits


async def _gather_test_results() -> Dict[str, Any]:
    """Run pytest and gather results."""
    try:
        # Run pytest with json output
        result = await asyncio.create_subprocess_exec(
            "uv", "run", "pytest",
            "--tb=no", "-q", "--co", "-q",  # Just count tests
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        # Count tests from collection output
        output = stdout.decode() if stdout else ""
        test_lines = [l for l in output.split("\n") if l.strip() and "::" in l]
        total_tests = len(test_lines)

        # Try to get pass/fail from last run
        # Check for pytest cache
        cache_file = PROJECT_ROOT / ".pytest_cache" / "v" / "cache" / "lastfailed"
        failed_count = 0
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    failed_data = json.load(f)
                    failed_count = len(failed_data) if failed_data else 0
            except (OSError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to read pytest lastfailed cache: {e}")
            except Exception as e:
                logger.debug(f"Unexpected error reading pytest cache: {e}", exc_info=True)

        return {
            "total": total_tests,
            "passed": total_tests - failed_count,
            "failed": failed_count,
            "summary": f"{total_tests} tests collected",
        }

    except FileNotFoundError as e:
        logger.error(f"uv or pytest not found: {e}", exc_info=True)
        return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}
    except subprocess.SubprocessError as e:
        logger.error(f"Test subprocess error: {e}", exc_info=True)
        return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}
    except OSError as e:
        logger.error(f"Failed to read pytest cache: {e}", exc_info=True)
        return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse pytest cache: {e}", exc_info=True)
        return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}
    except Exception as e:
        logger.error(f"Test gathering failed: {e}", exc_info=True)
        return {"total": 0, "passed": 0, "failed": 0, "error": str(e)}


async def _gather_security_assessment(target: str) -> str:
    """Gather security assessment information."""
    assessment_parts = []

    # Check for security review document
    review_path = PROJECT_ROOT / "docs" / "reviews" / f"{target.upper()}_REVIEW.md"
    if review_path.exists():
        try:
            content = review_path.read_text()
            # Extract summary section if available
            if "## Summary" in content:
                summary_start = content.find("## Summary")
                summary_end = content.find("\n## ", summary_start + 1)
                if summary_end == -1:
                    summary_end = len(content)
                assessment_parts.append(content[summary_start:summary_end])
            else:
                assessment_parts.append(content[:2000])  # First 2000 chars
        except OSError as e:
            logger.warning(f"Could not read security review: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Could not read security review: {e}", exc_info=True)

    # Check for general security review
    general_review = PROJECT_ROOT / "docs" / "CRITICAL_CODE_REVIEW.md"
    if general_review.exists():
        try:
            content = general_review.read_text()
            # Extract key findings
            if "Critical Security" in content or "Key Findings" in content:
                lines = content.split("\n")
                key_lines = []
                in_section = False
                for line in lines:
                    if "Critical" in line or "Key Findings" in line:
                        in_section = True
                    elif in_section:
                        if line.startswith("## "):
                            break
                        key_lines.append(line)
                if key_lines:
                    assessment_parts.append("\n".join(key_lines[:20]))
        except OSError as e:
            logger.warning(f"Could not read critical review: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Could not read critical review: {e}", exc_info=True)

    # Check recent security-related commits
    try:
        result = await asyncio.create_subprocess_exec(
            "git", "log", "--oneline", "--grep=security", "-5",
            cwd=PROJECT_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        if stdout:
            assessment_parts.append(
                "Recent security commits:\n" + stdout.decode().strip()
            )
    except FileNotFoundError as e:
        logger.debug(f"Git command not found: {e}")
    except subprocess.SubprocessError as e:
        logger.debug(f"Git subprocess error: {e}")
    except Exception as e:
        logger.debug(f"Unexpected error gathering security commits: {e}", exc_info=True)

    return "\n\n".join(assessment_parts) if assessment_parts else "No security assessment available."


async def _gather_architecture_docs(
    include_full: bool = False
) -> tuple[List[str], List[str]]:
    """Gather relevant architecture documentation."""
    docs = []
    source_files = []

    # Key architecture docs
    doc_paths = [
        "docs/architecture/FEATURE_AGENT_FRAMEWORK.md",
        "docs/architecture/PRIVACY_MODES.md",
        "docs/architecture/CRYPTOGRAPHIC_ANCHORING.md",
        "docs/principles/KESTREL_CONSTITUTION.md",
    ]

    for doc_path in doc_paths:
        full_path = PROJECT_ROOT / doc_path
        if full_path.exists():
            source_files.append(doc_path)
            try:
                content = full_path.read_text()
                if include_full:
                    docs.append(f"# {doc_path}\n\n{content}")
                else:
                    # Just include title and first paragraph
                    lines = content.split("\n")
                    summary_lines = []
                    for i, line in enumerate(lines[:20]):
                        summary_lines.append(line)
                        if i > 5 and line.strip() == "":
                            break
                    docs.append("\n".join(summary_lines))
            except OSError as e:
                logger.warning(f"Could not read {doc_path}: {e}", exc_info=True)
            except Exception as e:
                logger.warning(f"Could not read {doc_path}: {e}", exc_info=True)

    return docs, source_files


async def _gather_risks(target: str) -> List[str]:
    """Gather known risks from project status and reviews."""
    risks = []

    # Check PROJECT_STATUS.md for risks
    status_path = PROJECT_ROOT / "PROJECT_STATUS.md"
    if status_path.exists():
        try:
            content = status_path.read_text()
            # Look for risk-related sections
            lines = content.split("\n")
            in_risk_section = False
            for line in lines:
                if "Risk" in line or "Warning" in line or "Issue" in line:
                    in_risk_section = True
                    risks.append(line.strip())
                elif in_risk_section:
                    if line.startswith("## ") or line.startswith("# "):
                        in_risk_section = False
                    elif line.strip().startswith("- "):
                        risks.append(line.strip()[2:])
        except OSError as e:
            logger.warning(f"Could not read PROJECT_STATUS.md: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Could not read PROJECT_STATUS.md: {e}", exc_info=True)

    # Check specific review for risks
    review_path = PROJECT_ROOT / "docs" / "reviews" / f"{target.upper()}_REVIEW.md"
    if review_path.exists():
        try:
            content = review_path.read_text()
            if "High Risk" in content:
                risks.append("High Risk: Key rotation mechanism not implemented")
            if "risk" in content.lower():
                lines = content.split("\n")
                for line in lines:
                    if "risk" in line.lower() and len(line) < 200:
                        risks.append(line.strip())
        except OSError as e:
            logger.debug(f"Could not read review file: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error reading review: {e}", exc_info=True)

    # Deduplicate
    return list(dict.fromkeys(risks))[:10]  # Max 10 risks


async def _gather_previous_decisions() -> List[str]:
    """Gather previous council decisions from storage."""
    decisions = []

    # Check for council session files
    sessions_dir = PROJECT_ROOT / "data" / "council_sessions"
    if sessions_dir.exists():
        try:
            for session_file in sorted(sessions_dir.glob("*.json"))[-5:]:
                with open(session_file, encoding="utf-8") as f:
                    session = json.load(f)
                    decisions.append(
                        f"{session.get('created_at', 'Unknown')}: "
                        f"{session.get('question', 'Unknown')[:100]} -> "
                        f"{session.get('outcome', 'Unknown')}"
                    )
        except OSError as e:
            logger.warning(f"Could not read previous sessions: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse session JSON: {e}", exc_info=True)
        except Exception as e:
            logger.warning(f"Could not read previous sessions: {e}", exc_info=True)

    return decisions


# Convenience function for specific targets
async def compile_emma_genesis_evidence() -> Evidence:
    """Compile evidence specifically for Emma Genesis decision."""
    evidence = await compile_evidence(
        target="emma_genesis",
        include_full_docs=False,
    )

    # Add Emma-specific context
    evidence.risks.extend([
        "No key rotation mechanism - loss of key means loss of agent",
        "Docker environment variable exposure risk for KESTREL_DATA_KEY",
        "First permanent agent - no prior experience with long-term operation",
    ])

    return evidence
