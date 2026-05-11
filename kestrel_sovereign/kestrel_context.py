#!/usr/bin/env python3
"""
Kestrel Context Loader for Claude Code
Preloads all necessary context files into Claude's initial prompt.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from datetime import datetime

logger = logging.getLogger(__name__)


class KestrelContextLoader:
    """Loads Kestrel project context for Claude Code sessions."""
    
    # Files to load in priority order
    STATIC_CONTEXT_FILES = [
        "AGENTS.md",  # Static memory - project detection and references
        "PROJECT_STATUS.md",  # Dynamic memory - current state
    ]

    # Essential docs referenced in AGENTS.md
    ESSENTIAL_DOCS = [
        "docs/PROJECT_VISION.md",
        "docs/principles/KESTREL_CONSTITUTION.md",
        "docs/principles/US_CONSTITUTION.md",
        "docs/architecture/core/FEATURE_AGENT_FRAMEWORK.md",
        "docs/architecture/security/PRIVACY_MODES.md",
        "docs/architecture/security/CRYPTOGRAPHIC_ANCHORING.md",
        "docs/GETTING_STARTED.md",
    ]

    # Historical "platform extension" support was present in the initial
    # import, but its condition and document list were missing. Keep the mode
    # explicit so callers can opt in with a concrete list instead of relying on
    # accidental path detection.
    EXTENSION_DOCS: Sequence[str] = ()

    def __init__(
        self,
        project_path: str = ".",
        *,
        platform_extension: bool = False,
        extension_docs: Optional[Sequence[str]] = None,
    ):
        self.project_path = Path(project_path).resolve()
        self.platform_extension = platform_extension
        self.extension_docs = tuple(
            self.EXTENSION_DOCS if extension_docs is None else extension_docs
        )
        self.context = {}
        self.files_loaded = []
        
    def load_file(self, filepath: str) -> Optional[str]:
        """Load a single file if it exists."""
        path = self.project_path / filepath
        if path.exists():
            try:
                content = path.read_text()
                self.files_loaded.append(filepath)
                return content
            except OSError as e:
                logger.warning(f"Error reading {filepath}: {e}")
        return None
    
    def load_context(self) -> Dict[str, str]:
        """Load all context files in proper order."""
        print("🔍 Loading Kestrel context files...", file=sys.stderr)
        
        # Load static and dynamic memory files
        for filename in self.STATIC_CONTEXT_FILES:
            content = self.load_file(filename)
            if content:
                self.context[filename] = content
                print(f"✅ Loaded {filename} ({len(content)} chars)", file=sys.stderr)
        
        # Load essential documentation
        for doc_path in self.ESSENTIAL_DOCS:
            content = self.load_file(doc_path)
            if content:
                self.context[doc_path] = content
                print(f"✅ Loaded {doc_path} ({len(content)} chars)", file=sys.stderr)
        
        if self.platform_extension:
            print(
                "🎯 Platform extension context enabled - loading extension documentation",
                file=sys.stderr,
            )
            for doc_path in self.extension_docs:
                content = self.load_file(doc_path)
                if content:
                    self.context[doc_path] = content
                    print(f"✅ Loaded {doc_path} ({len(content)} chars)", file=sys.stderr)
        
        # Look for README.md if not already loaded
        if "README.md" not in self.context:
            content = self.load_file("README.md")
            if content:
                self.context["README.md"] = content
                print(f"✅ Loaded README.md ({len(content)} chars)", file=sys.stderr)
        
        return self.context
    
    def extract_mcp_config(self) -> Optional[Dict]:
        """Extract MCP configuration from AGENTS.md if present."""
        project_agent = self.context.get("AGENTS.md", "")
        if not project_agent:
            return None
        
        # Look for JSON blocks containing mcpServers
        import re
        json_blocks = re.findall(r'```json\n(.*?)```', project_agent, re.DOTALL)
        
        for block in json_blocks:
            try:
                config = json.loads(block.strip())
                if 'mcpServers' in config:
                    print(f"🔧 Found MCP configuration with {len(config['mcpServers'])} servers", file=sys.stderr)
                    return config
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON block in AGENTS.md: {e}")
                continue
        
        return None
    
    def create_prompt(self, task: Optional[str] = None) -> str:
        """Create the initial prompt with all context."""
        prompt_parts = []
        
        # Add header
        prompt_parts.append("# Kestrel Project Context")
        prompt_parts.append(f"## Session Started: {datetime.now().isoformat()}")
        prompt_parts.append(f"## Working Directory: {self.project_path}")
        if self.platform_extension:
            prompt_parts.append("## Mode: Platform Extension")
        
        prompt_parts.append("")
        
        # Add task if specified
        if task:
            prompt_parts.append(f"## Task")
            prompt_parts.append(task)
            prompt_parts.append("")
        
        # Add loaded context files
        prompt_parts.append("## Loaded Context Files")
        prompt_parts.append(f"Total files loaded: {len(self.files_loaded)}")
        prompt_parts.append("")
        
        # Add each context file
        for filepath, content in self.context.items():
            prompt_parts.append(f"### File: {filepath}")
            prompt_parts.append("```markdown")
            prompt_parts.append(content)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        # Add instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append("You have been provided with the complete Kestrel project context above.")
        prompt_parts.append("Key points:")
        prompt_parts.append("- AGENTS.md contains static references and project configuration")
        prompt_parts.append("- PROJECT_STATUS.md contains current dynamic state and issues")
        prompt_parts.append("- The constitutional documents provide governance principles")
        prompt_parts.append("- Architecture docs explain the technical implementation")
        if self.platform_extension:
            prompt_parts.append("- You are working on the platform extension")
            prompt_parts.append("- Extension PRD and database schema have been loaded")
        
        prompt_parts.append("")
        prompt_parts.append("Please proceed with the task or await user instructions.")
        
        return "\n".join(prompt_parts)
    
    def launch_claude(self, task: Optional[str] = None, dry_run: bool = False):
        """Launch Claude Code with preloaded context."""
        # Load all context
        self.load_context()
        
        # Create the prompt
        prompt = self.create_prompt(task)
        
        if dry_run:
            print("\n" + "="*60, file=sys.stderr)
            print("DRY RUN - Would send this prompt to Claude:", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
            print("="*60, file=sys.stderr)
            print(f"Total prompt size: {len(prompt)} characters", file=sys.stderr)
            print(f"Files included: {', '.join(self.files_loaded)}", file=sys.stderr)
            return
        
        # Extract MCP config if present
        mcp_config = self.extract_mcp_config()
        mcp_config_file = None
        
        if mcp_config:
            # Create temporary MCP config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(mcp_config, f, indent=2)
                mcp_config_file = f.name
                print(f"📝 Created MCP config file: {mcp_config_file}", file=sys.stderr)
        
        # Build Claude command
        claude_cmd = ["claude"]
        
        if mcp_config_file:
            claude_cmd.extend(["--mcp-config", mcp_config_file])
        
        # Add the prompt as initial message
        claude_cmd.append(prompt)
        
        print("🚀 Launching Claude Code with preloaded context...", file=sys.stderr)
        print(f"📊 Total context size: {sum(len(c) for c in self.context.values())} characters", file=sys.stderr)
        print(f"📁 Files loaded: {len(self.files_loaded)}", file=sys.stderr)
        
        try:
            # Launch Claude
            subprocess.run(claude_cmd)
        except FileNotFoundError:
            print("❌ Error: 'claude' command not found. Please ensure Claude Code CLI is installed.", file=sys.stderr)
            print("   Visit: https://docs.anthropic.com/claude/docs/claude-code", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error launching Claude: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Clean up temp file
            if mcp_config_file:
                Path(mcp_config_file).unlink(missing_ok=True)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch Claude Code with Kestrel project context preloaded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kestrel_sovereign/kestrel_context.py
  python kestrel_sovereign/kestrel_context.py --dry-run
  python kestrel_sovereign/kestrel_context.py "fix the bug in storage.py"
        """
    )
    
    parser.add_argument(
        "task",
        nargs="?",
        help="Optional task to include in the initial prompt"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode - show what would be loaded without launching Claude"
    )
    
    parser.add_argument(
        "--path",
        default=".",
        help="Project path (default: current directory)"
    )

    parser.add_argument(
        "--platform-extension",
        action="store_true",
        help="Include platform-extension mode markers and extension docs"
    )

    parser.add_argument(
        "--extension-doc",
        action="append",
        default=[],
        help="Extension documentation path to include; may be repeated"
    )

    args = parser.parse_args()

    # Create loader
    loader = KestrelContextLoader(
        args.path,
        platform_extension=args.platform_extension,
        extension_docs=args.extension_doc,
    )

    # Launch Claude
    loader.launch_claude(task=args.task, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
