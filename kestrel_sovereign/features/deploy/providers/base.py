"""
Deploy Provider Abstract Base Class.

Re-exports from kestrel_sdk.deploy.base for backward compatibility.
Feature packages should import from kestrel_sdk.deploy.base directly.
"""

# Re-export everything from kestrel_sdk
from kestrel_sdk.deploy.base import DeployProvider  # noqa: F401
