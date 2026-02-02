"""
Kestrel Compute Feature - Script Analyzer.

Security analysis of scripts using pattern matching and risk scoring.
Identifies dangerous patterns, rewritable operations, and potential concerns.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .models import ComputeScript, SecurityFinding, SuggestedFix, calculate_risk_score

logger = logging.getLogger(__name__)


# ==============================================================================
# Security Patterns
# ==============================================================================

# Critical patterns - Auto-DENY, cannot be approved or rewritten
CRITICAL_PATTERNS: Dict[str, List[Tuple[str, str, str]]] = {
    # (pattern, category, description)
    "bash": [
        (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", "fork_bomb", "Fork bomb pattern - spawns infinite processes, will crash system"),
        (r"mkfs\.\w+", "disk_format", "Filesystem formatting command - destroys all data on target"),
        (r"dd\s+if=.*\s+of=/dev/sd[a-z]", "disk_overwrite", "Raw disk overwrite - destroys partition data"),
        (r">\s*/dev/sd[a-z]", "disk_write", "Direct write to disk device - corrupts filesystem"),
        (r"/etc/(passwd|shadow|sudoers)", "credential_access", "Access to system credential files"),
        (r"curl\s+[^\|]*\|\s*(sh|bash)", "rce", "Remote code execution via curl pipe to shell"),
        (r"wget\s+[^\|]*\|\s*(sh|bash)", "rce", "Remote code execution via wget pipe to shell"),
        (r"eval\s+\"\$\(curl", "rce", "Remote code execution via eval of downloaded content"),
    ],
    "python": [
        (r"ctypes\.(c_char_p|CDLL|windll|cdll)", "native_code", "Loading native libraries - potential for memory corruption"),
        (r"__builtins__\.__dict__", "sandbox_escape", "Builtins manipulation - sandbox escape attempt"),
        (r"__import__\s*\(\s*['\"]os['\"]\s*\)\.system", "rce", "Dynamic import of os.system - obfuscated shell access"),
    ],
}

# Rewritable patterns - Get transformed, not blocked
REWRITABLE_PATTERNS: Dict[str, List[Tuple[str, str, str]]] = {
    # (pattern, category, description)
    "bash": [
        (r"\brm\s+(-[rfivI]+\s+)?", "file_delete", "File deletion - will be rewritten to move to trash"),
    ],
    "python": [
        (r"os\.remove\s*\(", "file_delete", "File deletion - will be wrapped with safe_remove"),
        (r"os\.unlink\s*\(", "file_delete", "File deletion - will be wrapped with safe_remove"),
        (r"shutil\.rmtree\s*\(", "dir_delete", "Directory deletion - will be wrapped with safe_remove"),
        (r"Path\([^)]*\)\.unlink\s*\(", "file_delete", "Path unlink - will be wrapped with safe_remove"),
    ],
}

# Warning patterns - Flagged but allowed with user approval
WARNING_PATTERNS: Dict[str, List[Tuple[str, str, str]]] = {
    # (pattern, category, description)
    "bash": [
        (r"chmod\s+[0-7]*7", "permissions", "World-writable permissions - security risk"),
        (r"sudo\s+", "privilege", "Privilege escalation attempt (will fail in sandbox)"),
        (r"chown\s+", "ownership", "Ownership change - may fail in sandbox"),
        (r">\s*/etc/", "system_write", "Writing to system config directory"),
        (r"curl\s+[^\|]*-o\s+", "download", "File download - verify source"),
        (r"wget\s+", "download", "File download - verify source"),
        (r"nc\s+(-[a-z]+\s+)*\d+", "network", "Netcat connection - potential data exfiltration"),
    ],
    "python": [
        (r"eval\s*\(", "code_exec", "Dynamic code execution - verify input is trusted"),
        (r"exec\s*\(", "code_exec", "Dynamic code execution - verify input is trusted"),
        (r"subprocess\.[^(]*\(.*shell\s*=\s*True", "shell_injection", "Shell=True enables injection attacks"),
        (r"os\.system\s*\(", "shell_exec", "Shell command execution - prefer subprocess"),
        (r"pickle\.(load|loads)\s*\(", "deserialization", "Pickle deserialization - only use with trusted data"),
        (r"socket\.(socket|create_connection)", "network", "Network socket creation"),
        (r"requests\.(get|post|put|delete)\s*\(", "network", "HTTP request - verify URL"),
        (r"urllib\.request\.urlopen", "network", "HTTP request - verify URL"),
    ],
}

# Data exfiltration patterns - Sensitive path access
DATA_EXFIL_PATTERNS: Dict[str, List[Tuple[str, str, str]]] = {
    # (pattern, category, description)
    "bash": [
        (r"~\/\.ssh", "ssh_keys", "SSH key directory - high sensitivity"),
        (r"~\/\.aws", "cloud_creds", "AWS credentials directory - high sensitivity"),
        (r"~\/\.gcloud", "cloud_creds", "Google Cloud credentials - high sensitivity"),
        (r"~\/\.azure", "cloud_creds", "Azure credentials - high sensitivity"),
        (r"~\/\.config\/gh", "api_tokens", "GitHub CLI tokens - medium sensitivity"),
        (r"~\/\.kube\/config", "k8s_creds", "Kubernetes config - high sensitivity"),
        (r"~\/\.docker\/config", "docker_creds", "Docker registry credentials - high sensitivity"),
        (r"\$KESTREL_DATA_KEY", "master_key", "Master encryption key access - critical"),
    ],
    "python": [
        (r"expanduser\(['\"]~\/\.ssh", "ssh_keys", "SSH key directory access"),
        (r"expanduser\(['\"]~\/\.aws", "cloud_creds", "AWS credentials access"),
        (r"environ\s*\[\s*['\"]KESTREL_DATA_KEY", "master_key", "Master encryption key access"),
        (r"environ\.get\s*\(\s*['\"]KESTREL_DATA_KEY", "master_key", "Master encryption key access"),
    ],
}


@dataclass
class AnalysisResult:
    """Result of script security analysis."""
    findings: List[SecurityFinding]
    has_critical: bool
    has_rewritable: bool
    rewritable_patterns: List[str]  # Patterns that need rewriting
    risk_score: int
    
    @property
    def can_proceed(self) -> bool:
        """Whether the script can proceed (no critical issues)."""
        return not self.has_critical


class ScriptAnalyzer:
    """
    Analyze scripts for security concerns.
    
    Detects:
    - Critical patterns (auto-deny)
    - Rewritable patterns (will be transformed)
    - Warning patterns (require user approval)
    - Data exfiltration attempts
    
    Example:
        analyzer = ScriptAnalyzer()
        result = analyzer.analyze(script)
        
        if result.has_critical:
            # Auto-deny
            pass
        elif result.has_rewritable:
            # Transform the script
            pass
        else:
            # Queue for approval with warnings
            pass
    """
    
    def __init__(
        self,
        custom_critical: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
        custom_warning: Optional[Dict[str, List[Tuple[str, str, str]]]] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            custom_critical: Additional critical patterns by language
            custom_warning: Additional warning patterns by language
        """
        self.critical_patterns = {**CRITICAL_PATTERNS}
        self.rewritable_patterns = {**REWRITABLE_PATTERNS}
        self.warning_patterns = {**WARNING_PATTERNS}
        self.exfil_patterns = {**DATA_EXFIL_PATTERNS}
        
        if custom_critical:
            for lang, patterns in custom_critical.items():
                self.critical_patterns.setdefault(lang, []).extend(patterns)
        
        if custom_warning:
            for lang, patterns in custom_warning.items():
                self.warning_patterns.setdefault(lang, []).extend(patterns)
        
        # Compile patterns for efficiency
        self._compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}
    
    def _get_compiled(self, language: str, category: str) -> List[Tuple[re.Pattern, str, str]]:
        """Get compiled regex patterns for a language and category."""
        cache_key = f"{language}:{category}"
        
        if cache_key not in self._compiled:
            patterns_map = {
                "critical": self.critical_patterns,
                "rewritable": self.rewritable_patterns,
                "warning": self.warning_patterns,
                "exfil": self.exfil_patterns,
            }
            
            raw_patterns = patterns_map.get(category, {}).get(language, [])
            compiled = []
            
            for pattern, cat, desc in raw_patterns:
                try:
                    compiled.append((re.compile(pattern, re.MULTILINE), cat, desc))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            
            self._compiled[cache_key] = compiled
        
        return self._compiled.get(cache_key, [])
    
    def analyze(self, script: ComputeScript) -> AnalysisResult:
        """
        Analyze a script for security concerns.
        
        Args:
            script: The script to analyze
            
        Returns:
            AnalysisResult with findings and recommendations
        """
        findings: List[SecurityFinding] = []
        rewritable_found: List[str] = []
        has_critical = False
        
        content = script.content
        lines = content.split('\n')
        
        # Check critical patterns
        for compiled, category, description in self._get_compiled(script.language, "critical"):
            for match in compiled.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                findings.append(SecurityFinding(
                    severity="critical",
                    category=category,
                    description=description,
                    pattern_matched=match.group(),
                    recommendation="This pattern is blocked and cannot be approved.",
                    line_number=line_num,
                ))
                has_critical = True
        
        # Check rewritable patterns
        for compiled, category, description in self._get_compiled(script.language, "rewritable"):
            for match in compiled.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                findings.append(SecurityFinding(
                    severity="info",
                    category=category,
                    description=f"{description} - Pattern will be safely rewritten",
                    pattern_matched=match.group(),
                    recommendation="This operation will be transformed to use trash folder instead of permanent deletion.",
                    line_number=line_num,
                ))
                rewritable_found.append(match.group())
        
        # Check warning patterns
        for compiled, category, description in self._get_compiled(script.language, "warning"):
            for match in compiled.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                
                # Determine severity based on category
                severity = "high" if category in ("shell_injection", "code_exec", "rce") else "medium"
                
                findings.append(SecurityFinding(
                    severity=severity,
                    category=category,
                    description=description,
                    pattern_matched=match.group(),
                    recommendation=self._get_recommendation(category),
                    line_number=line_num,
                ))
        
        # Check data exfiltration patterns
        for compiled, category, description in self._get_compiled(script.language, "exfil"):
            for match in compiled.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                
                severity = "critical" if category == "master_key" else "high"
                if severity == "critical":
                    has_critical = True
                
                findings.append(SecurityFinding(
                    severity=severity,
                    category=category,
                    description=description,
                    pattern_matched=match.group(),
                    recommendation="Access to sensitive credentials detected. Review carefully.",
                    line_number=line_num,
                ))
        
        risk_score = calculate_risk_score(findings)
        
        return AnalysisResult(
            findings=findings,
            has_critical=has_critical,
            has_rewritable=bool(rewritable_found),
            rewritable_patterns=rewritable_found,
            risk_score=risk_score,
        )
    
    def _get_recommendation(self, category: str) -> str:
        """Get a recommendation for a pattern category."""
        recommendations = {
            "shell_injection": "Use subprocess with shell=False and pass arguments as a list.",
            "code_exec": "Avoid eval/exec on user input. Use ast.literal_eval for simple data.",
            "network": "Verify the URL/host is trusted before allowing network access.",
            "download": "Verify the download source is trusted and the file will be validated.",
            "permissions": "Consider using more restrictive permissions (e.g., 755 instead of 777).",
            "privilege": "Script will run in a sandbox without sudo privileges.",
            "deserialization": "Only unpickle data from trusted sources.",
        }
        return recommendations.get(category, "Review this pattern carefully before approval.")
    
    def get_suggested_fixes(self, script: ComputeScript) -> List[SuggestedFix]:
        """
        Get suggested fixes for issues in a script.
        
        Args:
            script: The script to analyze
            
        Returns:
            List of suggested fixes
        """
        result = self.analyze(script)
        fixes: List[SuggestedFix] = []
        
        for finding in result.findings:
            if finding.severity == "critical":
                # Critical patterns need removal or complete rewrite
                if finding.category == "rce":
                    fixes.append(SuggestedFix(
                        type="split_script",
                        description="Download file first, review it, then execute separately",
                        original=finding.pattern_matched,
                        replacement="# Download: curl -o /tmp/script.sh <url>\n# Review: cat /tmp/script.sh\n# Then approve execution separately",
                    ))
                else:
                    fixes.append(SuggestedFix(
                        type="remove_pattern",
                        description=f"Remove {finding.category} pattern",
                        original=finding.pattern_matched,
                        replacement="# REMOVED: dangerous pattern",
                    ))
            
            elif finding.category == "shell_injection":
                fixes.append(SuggestedFix(
                    type="rewrite_pattern",
                    description="Use subprocess without shell=True",
                    original="subprocess.run(cmd, shell=True)",
                    replacement="subprocess.run(cmd.split())",
                ))
            
            elif finding.category == "code_exec":
                fixes.append(SuggestedFix(
                    type="rewrite_pattern",
                    description="Use ast.literal_eval for safe evaluation",
                    original="eval(user_input)",
                    replacement="import ast; ast.literal_eval(user_input)",
                ))
        
        return fixes


def analyze_script(script: ComputeScript) -> AnalysisResult:
    """
    Convenience function to analyze a script.
    
    Args:
        script: The script to analyze
        
    Returns:
        AnalysisResult with findings
    """
    analyzer = ScriptAnalyzer()
    return analyzer.analyze(script)
