"""Local compatibility overlays for SDK modules.

The real SDK remains an external dependency. This package extends the SDK
namespace so Kestrel can ship a narrowly patched ``payer_policy`` module while
other ``kestrel_sdk.*`` imports continue to resolve from the installed SDK.
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
