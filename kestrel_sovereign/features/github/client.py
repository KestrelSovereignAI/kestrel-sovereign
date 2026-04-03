"""GitHub API client - re-exported from kestrel-feature-github package.

This module exists for backward compatibility. The canonical implementation
lives in kestrel_feature_github.client.
"""
from kestrel_feature_github.client import GitHubClient, GitHubClientError

__all__ = ["GitHubClient", "GitHubClientError"]
