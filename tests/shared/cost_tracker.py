"""
Cost tracking for cloud resource usage during tests.

Tracks:
- RunPod GPU instance hours
- Estimated costs per test run
- Warns when costs exceed thresholds
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path

COST_LOG_FILE = Path("/tmp/kestrel_test_costs.json")
COST_WARNING_THRESHOLD = 5.0  # USD - warn if test run costs exceed this


@dataclass
class ResourceUsage:
    """Track usage of a single cloud resource."""
    resource_type: str
    resource_id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    cost_per_hour: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        """Calculate duration in hours."""
        end = self.ended_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() / 3600

    @property
    def estimated_cost(self) -> float:
        """Calculate estimated cost in USD."""
        return self.duration_hours * self.cost_per_hour

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'cost_per_hour': self.cost_per_hour,
            'duration_hours': self.duration_hours,
            'estimated_cost': self.estimated_cost,
            'metadata': self.metadata
        }


class CostTracker:
    """
    Tracks costs across test runs.

    Singleton that accumulates resource usage and warns
    when costs exceed configured thresholds.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self):
        """Initialize the tracker."""
        if self._initialized:
            return

        self._usages: List[ResourceUsage] = []
        self._session_start = datetime.now(timezone.utc)
        self._warning_threshold = COST_WARNING_THRESHOLD
        self._initialized = True

    def set_warning_threshold(self, threshold: float):
        """Set the cost warning threshold in USD."""
        self._initialize()
        self._warning_threshold = threshold

    def start_usage(
        self,
        resource_type: str,
        resource_id: str,
        cost_per_hour: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ResourceUsage:
        """
        Record start of resource usage.

        Args:
            resource_type: Type of resource (e.g., "runpod", "cloud_sql")
            resource_id: Unique identifier for the resource
            cost_per_hour: Hourly cost in USD
            metadata: Optional additional data

        Returns:
            ResourceUsage object to track this usage
        """
        self._initialize()
        usage = ResourceUsage(
            resource_type=resource_type,
            resource_id=resource_id,
            started_at=datetime.now(timezone.utc),
            cost_per_hour=cost_per_hour,
            metadata=metadata or {}
        )
        self._usages.append(usage)
        return usage

    def end_usage(self, usage: ResourceUsage):
        """
        Record end of resource usage.

        Args:
            usage: The ResourceUsage object returned by start_usage()
        """
        usage.ended_at = datetime.now(timezone.utc)
        self._check_cost_warning()

    def _check_cost_warning(self):
        """Warn if costs are high."""
        total = self.total_cost
        if total > self._warning_threshold:
            print(f"\n⚠️  [COST WARNING] Test run has cost ${total:.2f} "
                  f"(threshold: ${self._warning_threshold:.2f})")

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all tracked resources."""
        self._initialize()
        return sum(u.estimated_cost for u in self._usages)

    @property
    def active_usages(self) -> List[ResourceUsage]:
        """Get currently active (not ended) usages."""
        self._initialize()
        return [u for u in self._usages if u.ended_at is None]

    def report(self) -> str:
        """Generate a formatted cost report."""
        self._initialize()
        lines = [
            "",
            "=" * 60,
            "  TEST RUN COST REPORT",
            "=" * 60,
            f"  Session started: {self._session_start.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  Total estimated cost: ${self.total_cost:.2f}",
            "",
        ]

        if self._usages:
            lines.append("  Resource breakdown:")
            for usage in self._usages:
                status = "✓" if usage.ended_at else "⏳"
                lines.append(
                    f"    {status} {usage.resource_type}:{usage.resource_id}: "
                    f"{usage.duration_hours:.2f}h × ${usage.cost_per_hour:.2f}/h = "
                    f"${usage.estimated_cost:.2f}"
                )
        else:
            lines.append("  No cloud resources used")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self):
        """Save cost report to file."""
        self._initialize()
        data = {
            'session_start': self._session_start.isoformat(),
            'session_end': datetime.now(timezone.utc).isoformat(),
            'total_cost': self.total_cost,
            'usages': [u.to_dict() for u in self._usages]
        }
        try:
            COST_LOG_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[COST] Failed to save cost report: {e}")

    def reset(self):
        """Reset the tracker for a new session."""
        self._usages = []
        self._session_start = datetime.now(timezone.utc)


# Global singleton
cost_tracker = CostTracker()


# Convenience context manager for tracking usage
class track_cost:
    """
    Context manager for tracking resource costs.

    Usage:
        with track_cost("runpod", pod_id, cost_per_hour=0.50) as usage:
            # use the resource
            pass
        # cost is automatically tracked
    """

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        cost_per_hour: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.cost_per_hour = cost_per_hour
        self.metadata = metadata
        self.usage: Optional[ResourceUsage] = None

    def __enter__(self) -> ResourceUsage:
        self.usage = cost_tracker.start_usage(
            self.resource_type,
            self.resource_id,
            self.cost_per_hour,
            self.metadata
        )
        return self.usage

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.usage:
            cost_tracker.end_usage(self.usage)
        return False  # Don't suppress exceptions
