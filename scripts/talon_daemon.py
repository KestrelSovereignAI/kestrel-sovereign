#!/usr/bin/env python3
"""
Talon Daemon - Continuous GitHub Issue Processing.

Polls configured repos for claimable issues and processes them using
kestrel-talon. Supports both Claude (Max OAuth) and OpenCode (local Kimi)
backends.

Usage:
    # First-time setup (config is gitignored — copy from the example):
    cp scripts/talon_daemon.example.toml scripts/talon_daemon.toml
    $EDITOR scripts/talon_daemon.toml      # set repo_dir paths for your machine

    python scripts/talon_daemon.py
    python scripts/talon_daemon.py --config scripts/talon_daemon.toml
    python scripts/talon_daemon.py --backend codex --model gpt-5.5
    python scripts/talon_daemon.py --backend opencode --opencode-model kimi-local/kimi-k2.5
    python scripts/talon_daemon.py --dry-run

Environment Variables:
    GITHUB_TOKEN                    Required (or uses `gh auth token --user <gh_user>`)
    ANTHROPIC_API_KEY               Used only with backend=claude and auth_lane=api_key
    KESTREL_TALON_DIR               Override location of the kestrel-talon checkout
                                    (default: sibling of this repo)
    KESTREL_TALON_WORKTREE_BASE     Override worktree base directory
                                    (default: parent of this repo)
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kestrel_sovereign.features.talon.runtime import (
    TalonExecution,
    TalonPolicy,
    TalonPreference,
    TalonRuntimeError,
    TalonRuntimeRequest,
    build_talon_invocation,
    normalize_auth_lane,
    normalize_backend,
)

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python <3.11

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("talon_daemon")

# Defaults assume kestrel-talon is checked out as a sibling of this repo and
# worktrees live alongside it. Override with env vars on other layouts.
_PROJECTS_DIR = _REPO_ROOT.parent
TALON_DIR = Path(os.environ.get("KESTREL_TALON_DIR", _PROJECTS_DIR / "kestrel-talon"))
WORKTREE_BASE = Path(os.environ.get("KESTREL_TALON_WORKTREE_BASE", _PROJECTS_DIR))
DEFAULT_CONFIG = Path(__file__).parent / "talon_daemon.toml"


@dataclass
class RepoConfig:
    """Configuration for a single repo to monitor."""

    repo: str  # owner/repo
    repo_dir: str  # local checkout path
    label: str = "enhancement"
    quality_checks: list[str] = field(default_factory=list)
    max_iterations: int = 1
    max_turns: int = 50


@dataclass
class DaemonConfig:
    """Top-level daemon configuration."""

    repos: list[RepoConfig] = field(default_factory=list)
    poll_interval: int = 300  # seconds between polls
    cooldown: int = 60  # seconds between issue processing
    backend: str = "claude"  # "claude", "codex", or "opencode"
    model: str = "opus"  # Backend-specific model.
    auth_lane: str = "oauth"  # oauth, api_key, or provider_config
    opencode_model: str = "kimi-local/kimi-k2.5"  # Legacy CLI/config alias
    gh_user: str = "UncleSaurus"
    pause_file: str = "/tmp/talon-daemon-pause"
    max_daily_issues: int = 50
    worktree: bool = True
    skip_clarification: bool = True
    self_review: bool = True
    verbose: bool = False


@dataclass
class DaemonStats:
    """Runtime statistics."""

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issues_processed: int = 0
    issues_succeeded: int = 0
    issues_failed: int = 0
    issues_today: int = 0
    last_issue: Optional[str] = None
    last_poll: Optional[datetime] = None
    current_day: Optional[str] = None

    def new_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.current_day != today:
            self.current_day = today
            self.issues_today = 0

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "issues_processed": self.issues_processed,
            "issues_succeeded": self.issues_succeeded,
            "issues_failed": self.issues_failed,
            "issues_today": self.issues_today,
            "last_issue": self.last_issue,
            "last_poll": self.last_poll.isoformat() if self.last_poll else None,
            "uptime_hours": round(
                (datetime.now(timezone.utc) - self.started_at).total_seconds() / 3600, 1
            ),
        }


def load_config(path: Path, overrides: argparse.Namespace) -> DaemonConfig:
    """Load config from TOML file with CLI overrides."""
    config = DaemonConfig()

    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)

        daemon = data.get("daemon", {})
        config.poll_interval = daemon.get("poll_interval", config.poll_interval)
        config.cooldown = daemon.get("cooldown", config.cooldown)
        config.backend = daemon.get("backend", config.backend)
        config.model = daemon.get("model", config.model)
        config.auth_lane = daemon.get("auth_lane", config.auth_lane)
        config.opencode_model = daemon.get("opencode_model", config.opencode_model)
        config.gh_user = daemon.get("gh_user", config.gh_user)
        config.max_daily_issues = daemon.get("max_daily_issues", config.max_daily_issues)
        config.worktree = daemon.get("worktree", config.worktree)
        config.skip_clarification = daemon.get("skip_clarification", config.skip_clarification)
        config.self_review = daemon.get("self_review", config.self_review)
        config.verbose = daemon.get("verbose", config.verbose)

        for repo_data in data.get("repos", []):
            config.repos.append(
                RepoConfig(
                    repo=repo_data["repo"],
                    repo_dir=repo_data.get("repo_dir", ""),
                    label=repo_data.get("label", "enhancement"),
                    quality_checks=repo_data.get("quality_checks", []),
                    max_iterations=repo_data.get("max_iterations", 1),
                    max_turns=repo_data.get("max_turns", 50),
                )
            )
    else:
        example = path.with_name(path.stem + ".example" + path.suffix)
        if path == DEFAULT_CONFIG and example.exists():
            # Default config is gitignored, so a fresh checkout has only the
            # tracked .example file. Fail fast with a non-zero exit so any
            # supervising process (workload_manager.py, launchd) doesn't enter
            # a tight restart loop on an empty config.
            raise SystemExit(
                f"FATAL: {path} not found.\n"
                f"Bootstrap with: cp {example} {path} && $EDITOR {path}"
            )
        logger.warning(f"Config file not found: {path}, using defaults")

    # CLI overrides
    if overrides.backend:
        config.backend = overrides.backend
    if overrides.opencode_model:
        config.opencode_model = overrides.opencode_model
        if (overrides.backend or config.backend) == "opencode":
            config.model = overrides.opencode_model
    if overrides.model:
        config.model = overrides.model
    if getattr(overrides, "auth_lane", None):
        config.auth_lane = overrides.auth_lane
    if overrides.verbose:
        config.verbose = True
    if overrides.poll_interval:
        config.poll_interval = overrides.poll_interval

    return config


def get_gh_token(gh_user: str) -> str:
    """Get GitHub token for the specified user."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token", "--user", gh_user],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("No GitHub token available")
    return token


def list_claimable_issues(repo: str, label: str, gh_token: str) -> list[dict]:
    """List claimable issues for a repo using gh CLI."""
    try:
        result = subprocess.run(
            [
                "gh", "issue", "list",
                "--repo", repo,
                "--label", label,
                "--state", "open",
                "--json", "number,title,labels",
                "--limit", "20",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "GH_TOKEN": gh_token},
        )
        if result.returncode != 0:
            logger.error(f"gh issue list failed for {repo}: {result.stderr}")
            return []

        issues = json.loads(result.stdout)
        # Filter out already-claimed issues
        claimed_labels = {"agent-claimed", "agent-analyzing", "agent-blocked", "agent-complete", "agent-failed"}
        claimable = []
        for issue in issues:
            issue_labels = {l["name"] for l in issue.get("labels", [])}
            if not issue_labels & claimed_labels:
                claimable.append(issue)
        return claimable

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error listing issues for {repo}: {e}")
        return []


def build_talon_command(
    repo_config: RepoConfig,
    issue_number: int,
    daemon_config: DaemonConfig,
) -> list[str]:
    """Build the kestrel-talon claim command."""
    backend = normalize_backend(daemon_config.backend)
    if backend is None:
        raise TalonRuntimeError("Talon daemon backend is required")
    auth_lane = normalize_auth_lane(daemon_config.auth_lane)
    model = (
        daemon_config.opencode_model
        if backend == "opencode" and daemon_config.opencode_model
        else daemon_config.model
    )
    execution = TalonExecution(
        repo=repo_config.repo,
        issue=issue_number,
        repo_dir=Path(repo_config.repo_dir) if repo_config.repo_dir else Path.cwd(),
        worktree_base=WORKTREE_BASE,
        worktree=daemon_config.worktree,
        max_iterations=repo_config.max_iterations,
        max_turns=repo_config.max_turns,
        skip_clarification=daemon_config.skip_clarification,
        self_review=daemon_config.self_review,
        quality_checks=tuple(repo_config.quality_checks),
    )
    preference = TalonPreference(
        default_backend=backend,
        default_model=model,
        default_auth_lane=auth_lane,
        max_iterations=repo_config.max_iterations,
        max_turns=repo_config.max_turns,
        skip_clarification=daemon_config.skip_clarification,
        self_review=daemon_config.self_review,
    )
    invocation = build_talon_invocation(
        TalonRuntimeRequest(backend=backend, model=model, auth_lane=auth_lane),
        execution,
        policy=TalonPolicy(
            require_worktree=daemon_config.worktree,
            allow_api_billing=(auth_lane == "api_key"),
        ),
        preference=preference,
        base_env={
            "GITHUB_TOKEN": "placeholder-for-command-build",
            **(
                {"ANTHROPIC_API_KEY": "placeholder-for-command-build"}
                if auth_lane == "api_key"
                else {}
            ),
        },
    )
    cmd = ["uv", "run", "kestrel-talon"] + invocation.argv
    if daemon_config.verbose:
        cmd.append("--verbose")
    return cmd


async def process_issue(
    repo_config: RepoConfig,
    issue: dict,
    daemon_config: DaemonConfig,
    gh_token: str,
) -> bool:
    """Process a single issue. Returns True on success."""
    issue_num = issue["number"]
    issue_title = issue["title"]
    logger.info(f"Processing {repo_config.repo}#{issue_num}: {issue_title}")

    env = {
        **os.environ,
        "GH_TOKEN": gh_token,
        "GITHUB_TOKEN": gh_token,
    }
    try:
        cmd = build_talon_command(repo_config, issue_num, daemon_config)
        backend = normalize_backend(daemon_config.backend) or "claude"
        auth_lane = normalize_auth_lane(daemon_config.auth_lane) or "oauth"
        invocation_env = build_talon_invocation(
            TalonRuntimeRequest(
                backend=backend,
                model=(
                    daemon_config.opencode_model
                    if backend == "opencode" and daemon_config.opencode_model
                    else daemon_config.model
                ),
                auth_lane=auth_lane,
            ),
            TalonExecution(
                repo=repo_config.repo,
                issue=issue_num,
                repo_dir=Path(repo_config.repo_dir) if repo_config.repo_dir else Path.cwd(),
                worktree_base=WORKTREE_BASE,
                worktree=daemon_config.worktree,
                max_iterations=repo_config.max_iterations,
                max_turns=repo_config.max_turns,
                skip_clarification=daemon_config.skip_clarification,
                self_review=daemon_config.self_review,
                quality_checks=tuple(repo_config.quality_checks),
            ),
            policy=TalonPolicy(
                require_worktree=daemon_config.worktree,
                allow_api_billing=(auth_lane == "api_key"),
            ),
            preference=TalonPreference(
                default_backend=backend,
                default_model=daemon_config.model,
                default_auth_lane=auth_lane,
            ),
            base_env=env,
        ).env
    except TalonRuntimeError as e:
        logger.error(f"Invalid Talon runtime for {repo_config.repo}#{issue_num}: {e}")
        return False

    logger.info(f"Command: cd {TALON_DIR} && {' '.join(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=TALON_DIR,
            env=invocation_env,
            stdout=asyncio.subprocess.PIPE if not daemon_config.verbose else None,
            stderr=asyncio.subprocess.PIPE if not daemon_config.verbose else None,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            logger.info(f"SUCCESS: {repo_config.repo}#{issue_num}")
            return True
        else:
            error_msg = stderr.decode() if stderr else "unknown error"
            logger.error(
                f"FAILED: {repo_config.repo}#{issue_num} (exit {proc.returncode}): {error_msg[:500]}"
            )
            return False

    except asyncio.CancelledError:
        logger.warning(f"Cancelled processing {repo_config.repo}#{issue_num}")
        raise
    except Exception as e:
        logger.error(f"Error processing {repo_config.repo}#{issue_num}: {e}")
        return False


def is_paused(pause_file: str) -> bool:
    """Check if daemon is paused via pause file."""
    return Path(pause_file).exists()


def check_memory_pressure() -> str:
    """Check macOS memory pressure level. Returns 'green', 'yellow', or 'red'."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            level = int(result.stdout.strip())
            if level >= 4:
                return "red"
            elif level >= 2:
                return "yellow"
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return "green"


async def daemon_loop(config: DaemonConfig):
    """Main daemon loop."""
    stats = DaemonStats()
    gh_token = get_gh_token(config.gh_user)

    if not gh_token:
        logger.error("No GitHub token, exiting")
        return

    if not config.repos:
        logger.error("No repos configured, exiting")
        return

    logger.info(f"Talon daemon started: {len(config.repos)} repos, backend={config.backend}")
    logger.info(f"Poll interval: {config.poll_interval}s, cooldown: {config.cooldown}s")
    for repo in config.repos:
        logger.info(f"  Watching: {repo.repo} (label={repo.label})")

    while True:
        try:
            stats.new_day()

            # Check pause file
            if is_paused(config.pause_file):
                logger.info("Paused (touch /tmp/talon-daemon-pause to pause, rm to resume)")
                await asyncio.sleep(30)
                continue

            # Check memory pressure
            pressure = check_memory_pressure()
            if pressure == "red":
                logger.warning("Memory pressure RED — pausing processing")
                await asyncio.sleep(60)
                continue
            elif pressure == "yellow":
                logger.info("Memory pressure YELLOW — slowing down")
                await asyncio.sleep(config.poll_interval * 2)

            # Check daily limit
            if stats.issues_today >= config.max_daily_issues:
                logger.info(f"Daily limit reached ({config.max_daily_issues}), waiting until tomorrow")
                await asyncio.sleep(3600)
                continue

            stats.last_poll = datetime.now(timezone.utc)

            # Scan all repos for claimable issues
            found_work = False
            for repo_config in config.repos:
                issues = list_claimable_issues(repo_config.repo, repo_config.label, gh_token)

                if not issues:
                    logger.debug(f"No claimable issues in {repo_config.repo}")
                    continue

                logger.info(f"Found {len(issues)} claimable issue(s) in {repo_config.repo}")

                # Process one issue at a time (sequential)
                issue = issues[0]
                found_work = True

                success = await process_issue(repo_config, issue, config, gh_token)

                stats.issues_processed += 1
                stats.issues_today += 1
                stats.last_issue = f"{repo_config.repo}#{issue['number']}"

                if success:
                    stats.issues_succeeded += 1
                else:
                    stats.issues_failed += 1

                # Log stats
                logger.info(
                    f"Stats: {stats.issues_processed} processed, "
                    f"{stats.issues_succeeded} succeeded, "
                    f"{stats.issues_failed} failed, "
                    f"{stats.issues_today} today"
                )

                # Cooldown between issues
                logger.info(f"Cooldown: {config.cooldown}s")
                await asyncio.sleep(config.cooldown)

                # Only process one issue per poll cycle (sequential)
                break

            if not found_work:
                logger.info(f"No work found. Sleeping {config.poll_interval}s...")

            await asyncio.sleep(config.poll_interval)

        except asyncio.CancelledError:
            logger.info("Daemon cancelled, shutting down")
            break
        except Exception as e:
            logger.error(f"Unexpected error in daemon loop: {e}", exc_info=True)
            await asyncio.sleep(60)

    # Write final stats
    logger.info(f"Final stats: {json.dumps(stats.to_dict(), indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Talon Daemon - Continuous Issue Processing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    parser.add_argument("--backend", choices=["claude", "opencode", "codex"], help="Override backend")
    parser.add_argument("--opencode-model", help="Override opencode model")
    parser.add_argument("--model", help="Override backend model")
    parser.add_argument("--auth-lane", choices=["oauth", "api_key", "provider_config"], help="Override auth lane")
    parser.add_argument("--verbose", action="store_true", help="Stream agent output")
    parser.add_argument("--poll-interval", type=int, help="Override poll interval (seconds)")
    parser.add_argument("--dry-run", action="store_true", help="List issues without processing")
    args = parser.parse_args()

    config = load_config(args.config, args)

    if args.dry_run:
        gh_token = get_gh_token(config.gh_user)
        for repo_config in config.repos:
            issues = list_claimable_issues(repo_config.repo, repo_config.label, gh_token)
            print(f"\n{repo_config.repo} ({repo_config.label}):")
            if not issues:
                print("  No claimable issues")
            for issue in issues:
                print(f"  #{issue['number']}: {issue['title']}")
        return

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()
    task = None

    def shutdown(sig, frame):
        logger.info(f"Received {signal.Signals(sig).name}, shutting down...")
        if task:
            task.cancel()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        task = loop.create_task(daemon_loop(config))
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
