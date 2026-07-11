"""Sovereign consumes and compatibility-exports SDK extension contracts."""

from kestrel_sdk.extensions import AppExtension as SdkAppExtension
from kestrel_sdk.features.ui import UIContributions as SdkUIContributions
from kestrel_sovereign.extensions.app_extension import AppExtension
from kestrel_sovereign.features.base import UIContributions


def test_ui_contributions_is_the_sdk_type():
    assert UIContributions is SdkUIContributions


def test_app_extension_compatibility_path_is_the_sdk_type():
    assert AppExtension is SdkAppExtension
