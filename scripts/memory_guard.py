#!/usr/bin/env python3
"""
Memory Guard - macOS memory pressure monitor.

Monitors system memory pressure and provides a simple API for other
scripts to check whether they should pause, slow down, or stop.

Can be used standalone or imported by workload_manager.py.

Usage:
    # Standalone monitor
    python scripts/memory_guard.py

    # As library
    from memory_guard import MemoryGuard, PressureLevel
    guard = MemoryGuard()
    level = guard.check()
    if level == PressureLevel.RED:
        stop_work()
"""

import enum
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class PressureLevel(enum.Enum):
    GREEN = "green"    # <80% — all workloads allowed
    YELLOW = "yellow"  # 80-90% — pause heavy workloads (LoRA, new batches)
    RED = "red"        # >90% — kill LoRA, pause talon, only Kimi + embeddings


@dataclass
class MemoryStatus:
    """Current memory state."""
    pressure: PressureLevel
    total_gb: float
    used_gb: float
    available_gb: float
    usage_pct: float
    swap_used_gb: float
    kernel_pressure: Optional[int] = None  # macOS kern.memorystatus_vm_pressure_level

    def __str__(self) -> str:
        return (
            f"Memory: {self.usage_pct:.0f}% ({self.used_gb:.1f}/{self.total_gb:.1f}GB) "
            f"avail={self.available_gb:.1f}GB swap={self.swap_used_gb:.1f}GB "
            f"pressure={self.pressure.value}"
        )


class MemoryGuard:
    """Monitors macOS memory pressure and provides workload guidance."""

    def __init__(
        self,
        yellow_threshold: float = 80.0,
        red_threshold: float = 90.0,
        poll_interval: float = 10.0,
    ):
        self.yellow_threshold = yellow_threshold
        self.red_threshold = red_threshold
        self.poll_interval = poll_interval
        self._last_status: Optional[MemoryStatus] = None

    def check(self) -> PressureLevel:
        """Quick check — returns pressure level."""
        status = self.get_status()
        return status.pressure

    def get_status(self) -> MemoryStatus:
        """Get detailed memory status."""
        total_gb, used_gb, available_gb, swap_gb = self._read_memory_stats()
        usage_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
        kernel_pressure = self._read_kernel_pressure()

        # Determine pressure level
        # Prefer kernel pressure if available (more accurate than simple percentage)
        if kernel_pressure is not None and kernel_pressure >= 4:
            pressure = PressureLevel.RED
        elif kernel_pressure is not None and kernel_pressure >= 2:
            pressure = PressureLevel.YELLOW
        elif usage_pct >= self.red_threshold:
            pressure = PressureLevel.RED
        elif usage_pct >= self.yellow_threshold:
            pressure = PressureLevel.YELLOW
        else:
            pressure = PressureLevel.GREEN

        status = MemoryStatus(
            pressure=pressure,
            total_gb=total_gb,
            used_gb=used_gb,
            available_gb=available_gb,
            usage_pct=usage_pct,
            swap_used_gb=swap_gb,
            kernel_pressure=kernel_pressure,
        )

        # Log level changes
        if self._last_status and self._last_status.pressure != status.pressure:
            if status.pressure == PressureLevel.RED:
                logger.warning(f"Memory pressure -> RED: {status}")
            elif status.pressure == PressureLevel.YELLOW:
                logger.warning(f"Memory pressure -> YELLOW: {status}")
            else:
                logger.info(f"Memory pressure -> GREEN: {status}")

        self._last_status = status
        return status

    def _read_memory_stats(self) -> tuple[float, float, float, float]:
        """Read memory stats using sysctl and vm_stat."""
        total_gb = 0.0
        used_gb = 0.0
        available_gb = 0.0
        swap_gb = 0.0

        # Total memory
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                total_gb = int(result.stdout.strip()) / (1024 ** 3)
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass

        # vm_stat for detailed breakdown
        try:
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                stats = {}
                for line in result.stdout.strip().split("\n"):
                    if ":" in line:
                        key, val = line.split(":", 1)
                        val = val.strip().rstrip(".")
                        try:
                            stats[key.strip()] = int(val)
                        except ValueError:
                            pass

                page_size = 16384  # Apple Silicon uses 16KB pages
                # Try to get actual page size
                try:
                    ps_result = subprocess.run(
                        ["sysctl", "-n", "hw.pagesize"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if ps_result.returncode == 0:
                        page_size = int(ps_result.stdout.strip())
                except (subprocess.TimeoutExpired, ValueError):
                    pass

                free_pages = stats.get("Pages free", 0)
                inactive_pages = stats.get("Pages inactive", 0)
                speculative_pages = stats.get("Pages speculative", 0)
                purgeable_pages = stats.get("Pages purgeable", 0)

                available_pages = free_pages + inactive_pages + speculative_pages + purgeable_pages
                available_gb = available_pages * page_size / (1024 ** 3)
                used_gb = total_gb - available_gb

                # Swap
                swapins = stats.get("Swapins", 0)
                swapouts = stats.get("Swapouts", 0)
                # vm_stat doesn't directly report swap usage, use sysctl
                try:
                    swap_result = subprocess.run(
                        ["sysctl", "-n", "vm.swapusage"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if swap_result.returncode == 0:
                        # Format: "total = 0.00M  used = 0.00M  free = 0.00M"
                        for part in swap_result.stdout.split():
                            if part.endswith("M") and "used" in swap_result.stdout.split("used")[0]:
                                # Get the value after "used ="
                                pass
                        parts = swap_result.stdout.split()
                        for i, p in enumerate(parts):
                            if p == "used" and i + 2 < len(parts):
                                val = parts[i + 2].rstrip("M").rstrip("G")
                                try:
                                    swap_val = float(val)
                                    if parts[i + 2].endswith("G"):
                                        swap_gb = swap_val
                                    else:
                                        swap_gb = swap_val / 1024
                                except ValueError:
                                    pass
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return total_gb, used_gb, available_gb, swap_gb

    def _read_kernel_pressure(self) -> Optional[int]:
        """Read macOS kernel memory pressure level."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    def should_allow(self, workload: str) -> bool:
        """Check if a workload should be allowed to run.

        Workload priority (highest to lowest):
        - kimi: Always allowed (never touch)
        - embeddings: Allowed in GREEN and YELLOW
        - talon: Allowed in GREEN only
        - lora: Allowed in GREEN only
        """
        level = self.check()

        if workload == "kimi":
            return True
        elif workload == "embeddings":
            return level != PressureLevel.RED
        elif workload in ("talon", "lora"):
            return level == PressureLevel.GREEN
        else:
            return level == PressureLevel.GREEN


def main():
    """Standalone monitor — prints status every 10 seconds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    guard = MemoryGuard()
    print("Memory Guard — monitoring (Ctrl+C to stop)")
    print("-" * 60)

    try:
        while True:
            status = guard.get_status()
            print(status)
            time.sleep(guard.poll_interval)
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
