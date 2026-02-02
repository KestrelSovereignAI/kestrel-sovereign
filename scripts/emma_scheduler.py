#!/usr/bin/env python3
"""
Emma 22/7 Scheduler - Autonomous Self-Improvement Daemon.

Runs Emma (Mac Studio M3 Ultra) in a 22/7 cycle:
- 22 hours of productive work (training, tasks, interactions)
- 2 hours of sleep (memory consolidation, reflection, sovereignty export)

The schedule is configurable but defaults to:
- Sleep: 00:00-02:00 (midnight to 2am)
- Work: 02:00-00:00 (rest of the day)

During work hours:
- Processes companion factory jobs
- Handles user interactions
- Takes quick naps every 30 min of inactivity

During sleep hours:
- Deep memory consolidation
- Episode creation
- Layered reflection
- Sovereignty export to IPFS
- Self-model update

Usage:
    python scripts/emma_scheduler.py agent_data/kestrel_prime.db

    # Custom sleep window (3am-5am)
    python scripts/emma_scheduler.py agent_data/kestrel_prime.db --sleep-start 3 --sleep-end 5

    # Run in foreground with verbose output
    python scripts/emma_scheduler.py agent_data/kestrel_prime.db --verbose

Environment Variables:
    KESTREL_DATA_KEY - Required for encrypted storage access
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("emma_scheduler")


class EmmaScheduler:
    """22/7 Scheduler for Emma's autonomous operation."""

    def __init__(
        self,
        db_path: str,
        sleep_start_hour: int = 0,  # Midnight
        sleep_end_hour: int = 2,    # 2am
        nap_interval_minutes: int = 30,
    ):
        """Initialize the scheduler.

        Args:
            db_path: Path to Emma's database
            sleep_start_hour: Hour to start sleep (0-23)
            sleep_end_hour: Hour to end sleep (0-23)
            nap_interval_minutes: Minutes of inactivity before quick nap
        """
        self.db_path = db_path
        self.sleep_start = sleep_start_hour
        self.sleep_end = sleep_end_hour
        self.nap_interval = nap_interval_minutes * 60  # Convert to seconds

        self.agent = None
        self.factory = None
        self.running = False
        self.last_activity = datetime.now(timezone.utc)

        # Stats tracking
        self.stats = {
            "started_at": None,
            "sleep_cycles": 0,
            "quick_naps": 0,
            "jobs_processed": 0,
            "reflections_run": 0,
            "last_sleep_cid": None,
        }

    async def initialize(self):
        """Initialize Emma and related systems."""
        logger.info("Initializing Emma...")

        # Validate database path
        if os.path.isdir(self.db_path):
            self.db_path = os.path.join(self.db_path, "kestrel_prime.db")

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        # Get agent DID
        from kestrel_sovereign.storage import AsyncStorage
        storage = AsyncStorage(self.db_path)
        await storage.initialize()
        try:
            agent_nodes = await storage.get_nodes_by_type("agent")
            if not agent_nodes:
                raise ValueError("No agent found in database. Run inception_service.py first.")
            agent_did = agent_nodes[0].node_id
        finally:
            await storage.close()

        logger.info(f"Agent DID: {agent_did}")

        # Initialize agent
        from kestrel_sovereign.kestrel_agent import KestrelAgent
        from kestrel_sovereign.llm.service import LLMService

        llm_service = LLMService()
        self.agent = KestrelAgent(
            did=agent_did,
            storage_path=self.db_path,
            llm_service=llm_service
        )
        await self.agent.initialize()

        # Initialize companion factory if available
        try:
            from kestrel_sovereign.features.companion_factory.autonomous_factory import AutonomousFactory
            self.factory = AutonomousFactory(emma_agent_did=agent_did)
            logger.info("Companion factory initialized")
        except ImportError:
            logger.info("Companion factory not available - skipping")
            self.factory = None

        self.stats["started_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Emma initialized successfully")

    def is_sleep_time(self) -> bool:
        """Check if current time is within sleep window."""
        current_hour = datetime.now().hour

        # Handle wrap-around (e.g., sleep 23:00-01:00)
        if self.sleep_start > self.sleep_end:
            return current_hour >= self.sleep_start or current_hour < self.sleep_end
        else:
            return self.sleep_start <= current_hour < self.sleep_end

    def seconds_until_mode_change(self) -> int:
        """Calculate seconds until next mode change (sleep/wake)."""
        now = datetime.now()
        current_hour = now.hour

        if self.is_sleep_time():
            # Currently sleeping, calculate time until wake
            target_hour = self.sleep_end
        else:
            # Currently working, calculate time until sleep
            target_hour = self.sleep_start

        # Calculate time to target hour
        hours_until = (target_hour - current_hour) % 24
        if hours_until == 0:
            hours_until = 24

        minutes_until = 60 - now.minute
        seconds_until = 60 - now.second

        total_seconds = (hours_until - 1) * 3600 + minutes_until * 60 + seconds_until
        return max(total_seconds, 0)

    async def run_sleep_cycle(self):
        """Execute a full sleep cycle."""
        logger.info("=== ENTERING SLEEP CYCLE ===")
        self.stats["sleep_cycles"] += 1

        try:
            # Run deep sleep with reflection
            report = await self.agent.sleep(
                tier="ipfs",
                skip_consolidation=False,
                skip_export=False,
                skip_reflection=False,
            )

            if report.success:
                self.stats["last_sleep_cid"] = report.cid
                logger.info(f"Sleep cycle complete: {report}")
                logger.info(f"  Episodes created: {report.episodes_created}")
                logger.info(f"  Patterns found: {report.patterns_found}")
                logger.info(f"  Insights generated: {report.insights_generated}")
                logger.info(f"  CID: {report.cid}")
            else:
                logger.warning(f"Sleep cycle had issues: {report.error}")

            # Run intensive training cycle during sleep
            logger.info("Running training cycle...")
            self.stats["reflections_run"] += 1

            reflection = self.agent.features.get("ReflectionFeature")
            if reflection:
                result = await reflection.reflect(scope="all", depth="deep")
                actions = result.get("actions", [])
                logger.info(f"Training cycle complete: {len(actions)} action items")

        except Exception as e:
            logger.error(f"Sleep cycle failed: {e}", exc_info=True)

        logger.info("=== EXITING SLEEP CYCLE ===")

    async def run_work_cycle(self):
        """Execute work during waking hours."""
        # Check for factory jobs
        if self.factory:
            try:
                # Process one job at a time to stay responsive
                job = self.factory.queue.get_next_training_job() if hasattr(self.factory, 'queue') else None
                if job:
                    logger.debug(f"Processing factory job: {job.get('companion_id', 'unknown')}")
                    # The factory handles actual processing
                    self.stats["jobs_processed"] += 1
                    self.last_activity = datetime.now(timezone.utc)
            except Exception as e:
                logger.warning(f"Factory job processing error: {e}")

        # Check for inactivity - take a quick nap
        idle_seconds = (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        if idle_seconds > self.nap_interval:
            logger.info(f"Idle for {idle_seconds/60:.1f} min - taking quick nap")
            try:
                result = await self.agent.quick_nap()
                if result:
                    logger.info(f"Quick nap: {result}")
                    self.stats["quick_naps"] += 1
                self.last_activity = datetime.now(timezone.utc)
            except Exception as e:
                logger.warning(f"Quick nap failed: {e}")

    async def run(self):
        """Main scheduler loop."""
        self.running = True
        logger.info(f"Starting 22/7 scheduler (sleep: {self.sleep_start}:00-{self.sleep_end}:00)")

        while self.running:
            try:
                if self.is_sleep_time():
                    # Sleep mode
                    await self.run_sleep_cycle()

                    # Wait until sleep window ends
                    wait_seconds = self.seconds_until_mode_change()
                    logger.info(f"Sleeping for {wait_seconds/3600:.1f} hours until {self.sleep_end}:00")
                    await asyncio.sleep(min(wait_seconds, 3600))  # Check hourly at most
                else:
                    # Work mode
                    await self.run_work_cycle()

                    # Check again in 1 minute
                    await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("Scheduler cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Back off on errors

    async def stop(self):
        """Gracefully stop the scheduler."""
        logger.info("Stopping scheduler...")
        self.running = False

        if self.agent:
            await self.agent.shutdown()

        # Log final stats
        logger.info("=== FINAL STATS ===")
        logger.info(f"  Started: {self.stats['started_at']}")
        logger.info(f"  Sleep cycles: {self.stats['sleep_cycles']}")
        logger.info(f"  Quick naps: {self.stats['quick_naps']}")
        logger.info(f"  Jobs processed: {self.stats['jobs_processed']}")
        logger.info(f"  Reflections run: {self.stats['reflections_run']}")
        logger.info(f"  Last sleep CID: {self.stats['last_sleep_cid']}")

    def get_status(self) -> dict:
        """Get current scheduler status."""
        return {
            "running": self.running,
            "mode": "sleep" if self.is_sleep_time() else "work",
            "sleep_window": f"{self.sleep_start}:00-{self.sleep_end}:00",
            "seconds_until_mode_change": self.seconds_until_mode_change(),
            "stats": self.stats,
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Emma 22/7 Scheduler - Autonomous Self-Improvement Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("db_path", help="Path to Emma's database")
    parser.add_argument("--sleep-start", type=int, default=0,
                        help="Hour to start sleep (0-23, default: 0)")
    parser.add_argument("--sleep-end", type=int, default=2,
                        help="Hour to end sleep (0-23, default: 2)")
    parser.add_argument("--nap-interval", type=int, default=30,
                        help="Minutes of inactivity before quick nap (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scheduler = EmmaScheduler(
        db_path=args.db_path,
        sleep_start_hour=args.sleep_start,
        sleep_end_hour=args.sleep_end,
        nap_interval_minutes=args.nap_interval,
    )

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(scheduler.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await scheduler.initialize()
        await scheduler.run()
    except KeyboardInterrupt:
        pass
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
