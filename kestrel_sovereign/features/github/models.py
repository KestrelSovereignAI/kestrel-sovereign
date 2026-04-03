"""Data models for GitHub feature."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FileType(Enum):
    """Type of file in repository."""
    FILE = "file"
    DIR = "dir"
    SYMLINK = "symlink"
    SUBMODULE = "submodule"


@dataclass
class RepoFile:
    """A file or directory in a repository."""
    path: str
    name: str
    type: FileType
    size: int = 0
    sha: str = ""
    download_url: Optional[str] = None
    
    def is_dir(self) -> bool:
        return self.type == FileType.DIR
    
    def is_file(self) -> bool:
        return self.type == FileType.FILE


@dataclass
class FileContent:
    """Content of a file from repository."""
    path: str
    content: str
    sha: str
    size: int
    encoding: str = "utf-8"
    repo: str = ""
    ref: str = "main"
    cached_at: Optional[datetime] = None


@dataclass
class SearchResult:
    """A search result from GitHub code search."""
    path: str
    repo: str
    name: str
    sha: str
    score: float = 0.0
    html_url: str = ""
    text_matches: list[dict] = field(default_factory=list)


@dataclass
class CodeDefinition:
    """A code definition (function, class, etc.) extracted from source."""
    name: str
    type: str  # "function", "class", "method", "variable"
    path: str
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    source: str = ""


@dataclass
class ComponentManifest:
    """A component.yaml manifest for a feature."""
    feature_name: str
    description: str
    version: str = "1.0.0"
    entry_point: str = "feature.py"
    files: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict, feature_name: str) -> "ComponentManifest":
        """Create from dictionary (parsed YAML).

        Handles both formats for tools:
        - Simple: ["tool1", "tool2"]
        - Dict: [{"name": "tool1", "description": "..."}, ...]
        """
        # Normalize tools to list of strings
        raw_tools = data.get("tools", [])
        tools = []
        for t in raw_tools:
            if isinstance(t, dict):
                tools.append(t.get("name", str(t)))
            else:
                tools.append(str(t))

        return cls(
            feature_name=feature_name,
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            entry_point=data.get("entry_point", "feature.py"),
            files=data.get("files", []),
            tools=tools,
            dependencies=data.get("dependencies", []),
        )
