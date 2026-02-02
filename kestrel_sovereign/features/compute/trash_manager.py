"""
Kestrel Compute Feature - Trash Manager.

Manages the trash folder for deleted files, providing restore and cleanup
operations.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# Default trash directory
DEFAULT_TRASH_DIR = Path(os.environ.get(
    "KESTREL_TRASH_DIR",
    os.path.expanduser("~/.kestrel/trash")
))


@dataclass
class TrashItem:
    """An item in the trash folder."""
    name: str                    # Original filename
    path: Path                   # Current path in trash
    original_path: Optional[str] # Original location (if known)
    deleted_at: datetime         # When it was deleted
    size_bytes: int             # File/folder size
    is_dir: bool                # Whether it's a directory
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "original_path": self.original_path,
            "deleted_at": self.deleted_at.isoformat(),
            "size_bytes": self.size_bytes,
            "is_dir": self.is_dir,
        }


class TrashManager:
    """
    Manage the trash folder for safe file deletion.
    
    The trash folder contains timestamped subdirectories with deleted files.
    This allows restore operations and cleanup of old items.
    
    Example:
        manager = TrashManager()
        
        # List recent trash items
        items = manager.list_items(days=7)
        
        # Restore an item
        manager.restore(item.path, "/original/location")
        
        # Empty old trash
        deleted_count = manager.empty(older_than_days=30)
    """
    
    def __init__(self, trash_dir: Optional[Path] = None):
        """
        Initialize the trash manager.
        
        Args:
            trash_dir: Path to trash directory (default: ~/.kestrel/trash)
        """
        self.trash_dir = trash_dir or DEFAULT_TRASH_DIR
    
    def ensure_trash_dir(self) -> None:
        """Create trash directory if it doesn't exist."""
        self.trash_dir.mkdir(parents=True, exist_ok=True)
    
    def list_items(
        self,
        days: Optional[int] = None,
        limit: int = 100,
    ) -> List[TrashItem]:
        """
        List items in the trash folder.
        
        Args:
            days: Only show items from the last N days (None = all)
            limit: Maximum number of items to return
            
        Returns:
            List of TrashItems, newest first
        """
        items: List[TrashItem] = []
        
        if not self.trash_dir.exists():
            return items
        
        cutoff = None
        if days is not None:
            cutoff = datetime.now() - timedelta(days=days)
        
        # Iterate through timestamped subdirectories
        subdirs = sorted(self.trash_dir.iterdir(), reverse=True)
        
        for subdir in subdirs:
            if not subdir.is_dir():
                continue
            
            # Parse timestamp from directory name
            # Format: YYYYMMDD_HHMMSS_microseconds
            try:
                # Try with microseconds first
                if '_' in subdir.name and len(subdir.name) > 15:
                    parts = subdir.name.split('_')
                    if len(parts) >= 2:
                        timestamp_str = f"{parts[0]}_{parts[1]}"
                        deleted_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    else:
                        deleted_at = datetime.fromtimestamp(subdir.stat().st_mtime)
                else:
                    deleted_at = datetime.fromtimestamp(subdir.stat().st_mtime)
            except (ValueError, OSError):
                # Fall back to modification time
                try:
                    deleted_at = datetime.fromtimestamp(subdir.stat().st_mtime)
                except OSError:
                    continue
            
            if cutoff and deleted_at < cutoff:
                continue
            
            # List files in this trash subdirectory
            try:
                for item_path in subdir.iterdir():
                    try:
                        stat = item_path.stat()
                        is_dir = item_path.is_dir()
                        
                        if is_dir:
                            size = self._get_dir_size(item_path)
                        else:
                            size = stat.st_size
                        
                        items.append(TrashItem(
                            name=item_path.name,
                            path=item_path,
                            original_path=None,  # We don't track original path currently
                            deleted_at=deleted_at,
                            size_bytes=size,
                            is_dir=is_dir,
                        ))
                        
                        if len(items) >= limit:
                            return items
                            
                    except OSError as e:
                        logger.warning(f"Error reading trash item {item_path}: {e}")
                        continue
                        
            except OSError as e:
                logger.warning(f"Error reading trash subdir {subdir}: {e}")
                continue
        
        return items
    
    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        pass
        except OSError:
            pass
        return total
    
    def restore(
        self,
        trash_path: Path,
        destination: Optional[str] = None,
    ) -> Path:
        """
        Restore an item from trash.
        
        Args:
            trash_path: Path to the item in trash
            destination: Where to restore (default: current directory)
            
        Returns:
            Path to the restored item
            
        Raises:
            FileNotFoundError: If trash item doesn't exist
            FileExistsError: If destination already exists
        """
        if not trash_path.exists():
            raise FileNotFoundError(f"Trash item not found: {trash_path}")
        
        # Determine destination
        if destination:
            dest_path = Path(destination)
        else:
            dest_path = Path.cwd() / trash_path.name
        
        # Check destination doesn't exist
        if dest_path.exists():
            raise FileExistsError(f"Destination already exists: {dest_path}")
        
        # Create parent directory if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move from trash to destination
        shutil.move(str(trash_path), str(dest_path))
        
        logger.info(f"Restored {trash_path.name} to {dest_path}")
        
        # Clean up empty trash subdirectory
        parent = trash_path.parent
        if parent.exists() and not any(parent.iterdir()):
            try:
                parent.rmdir()
            except OSError:
                pass
        
        return dest_path
    
    def empty(
        self,
        older_than_days: int = 30,
        dry_run: bool = False,
    ) -> int:
        """
        Permanently delete old trash items.
        
        Args:
            older_than_days: Delete items older than this many days
            dry_run: If True, don't actually delete, just count
            
        Returns:
            Number of items deleted
        """
        if not self.trash_dir.exists():
            return 0
        
        cutoff = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        
        # Iterate through timestamped subdirectories
        for subdir in list(self.trash_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            # Parse timestamp from directory name
            try:
                parts = subdir.name.split('_')
                if len(parts) >= 2:
                    timestamp_str = f"{parts[0]}_{parts[1]}"
                    deleted_at = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                else:
                    deleted_at = datetime.fromtimestamp(subdir.stat().st_mtime)
            except (ValueError, OSError):
                try:
                    deleted_at = datetime.fromtimestamp(subdir.stat().st_mtime)
                except OSError:
                    continue
            
            if deleted_at < cutoff:
                if dry_run:
                    # Count items in directory
                    try:
                        deleted_count += sum(1 for _ in subdir.iterdir())
                    except OSError:
                        deleted_count += 1
                else:
                    try:
                        item_count = sum(1 for _ in subdir.iterdir())
                        shutil.rmtree(subdir)
                        deleted_count += item_count
                        logger.info(f"Deleted trash directory: {subdir.name}")
                    except OSError as e:
                        logger.warning(f"Error deleting trash directory {subdir}: {e}")
        
        return deleted_count
    
    def get_stats(self) -> dict:
        """
        Get statistics about the trash folder.
        
        Returns:
            Dictionary with item_count, total_size_bytes, oldest_item_age_days
        """
        if not self.trash_dir.exists():
            return {
                "item_count": 0,
                "total_size_bytes": 0,
                "oldest_item_age_days": None,
            }
        
        items = self.list_items(limit=10000)
        
        if not items:
            return {
                "item_count": 0,
                "total_size_bytes": 0,
                "oldest_item_age_days": None,
            }
        
        total_size = sum(item.size_bytes for item in items)
        oldest = min(items, key=lambda x: x.deleted_at)
        age_days = (datetime.now() - oldest.deleted_at).days
        
        return {
            "item_count": len(items),
            "total_size_bytes": total_size,
            "oldest_item_age_days": age_days,
        }
    
    def format_size(self, size_bytes: int) -> str:
        """Format bytes as human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


# Global instance for convenience
_trash_manager: Optional[TrashManager] = None


def get_trash_manager() -> TrashManager:
    """Get the global trash manager instance."""
    global _trash_manager
    if _trash_manager is None:
        _trash_manager = TrashManager()
    return _trash_manager
