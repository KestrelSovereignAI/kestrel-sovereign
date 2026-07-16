"""Unit tests for Azure Container Apps deployment provider."""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "azure.mgmt.appcontainers",
    reason="azure-mgmt-appcontainers not installed (cloud extras)",
)

from azure.core.exceptions import ClientAuthenticationError, ResourceNotFoundError

from kestrel_sovereign.features.deploy.models import (
    DeploymentProfile,
    DeployManagerError,
    DeployProviderType,
)
from kestrel_sovereign.features.deploy.providers.azure_container import (
    AzureContainerProvider,
)


@pytest.fixture
def deployment_profile() -> DeploymentProfile:
    return DeploymentProfile(
        provider=DeployProviderType.AZURE_CONTAINER_APPS,
        service_name="kestrel-dev",
        region="eastus2",
    )


@pytest.fixture
def provider_and_client() -> tuple[AzureContainerProvider, MagicMock]:
    provider = AzureContainerProvider(
        subscription_id="test-subscription",
        resource_group="test-resource-group",
    )
    client = MagicMock()
    provider._client = client
    provider._get_or_create_environment = MagicMock(
        return_value="/subscriptions/test/environments/kestrel-env"
    )

    result = MagicMock()
    result.configuration.ingress.fqdn = "kestrel-dev.example"
    result.latest_revision_name = "kestrel-dev--rev-1"
    poller = MagicMock()
    poller.result.return_value = result
    client.container_apps.begin_create_or_update.return_value = poller
    return provider, client


async def test_deploy_logs_create_only_for_resource_not_found(
    deployment_profile: DeploymentProfile,
    provider_and_client: tuple[AzureContainerProvider, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, client = provider_and_client
    client.container_apps.get.side_effect = ResourceNotFoundError("app absent")

    with caplog.at_level(
        "INFO",
        logger="kestrel_sovereign.features.deploy.providers.azure_container",
    ):
        result = await provider.deploy(
            image="registry.example/kestrel:tag",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

    assert result["status"] == "active"
    client.container_apps.get.assert_called_once_with(
        resource_group_name="test-resource-group",
        container_app_name="kestrel-dev",
    )
    client.container_apps.begin_create_or_update.assert_called_once()
    assert "Creating new Container App: kestrel-dev" in caplog.text


async def test_deploy_logs_update_when_resource_exists(
    deployment_profile: DeploymentProfile,
    provider_and_client: tuple[AzureContainerProvider, MagicMock],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, client = provider_and_client
    client.container_apps.get.return_value = MagicMock()

    with caplog.at_level(
        "INFO",
        logger="kestrel_sovereign.features.deploy.providers.azure_container",
    ):
        result = await provider.deploy(
            image="registry.example/kestrel:tag",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

    assert result["status"] == "active"
    client.container_apps.get.assert_called_once()
    client.container_apps.begin_create_or_update.assert_called_once()
    assert "Updating existing Container App: kestrel-dev" in caplog.text


async def test_deploy_does_not_misclassify_lookup_failure_as_absent(
    deployment_profile: DeploymentProfile,
    provider_and_client: tuple[AzureContainerProvider, MagicMock],
) -> None:
    provider, client = provider_and_client
    client.container_apps.get.side_effect = ClientAuthenticationError("GET forbidden")

    with pytest.raises(DeployManagerError, match="GET forbidden"):
        await provider.deploy(
            image="registry.example/kestrel:tag",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

    client.container_apps.begin_create_or_update.assert_not_called()


async def test_deploy_lookup_preserves_caller_cancellation(
    deployment_profile: DeploymentProfile,
    provider_and_client: tuple[AzureContainerProvider, MagicMock],
) -> None:
    provider, client = provider_and_client
    client.container_apps.get.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await provider.deploy(
            image="registry.example/kestrel:tag",
            service_name="kestrel-dev",
            profile=deployment_profile,
        )

    client.container_apps.begin_create_or_update.assert_not_called()
