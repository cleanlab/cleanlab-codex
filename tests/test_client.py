# ruff: noqa: DTZ005

import uuid
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from codex import AuthenticationError
from codex.types.project_return_schema import Config, ProjectReturnSchema
from codex.types.users.myself.user_organizations_schema import UserOrganizationsSchema

from cleanlab_codex.client import Client
from cleanlab_codex.project import MissingProjectError
from cleanlab_codex.types.organization import Organization
from cleanlab_codex.types.project import ProjectConfig, ProjectReturnConfig

FAKE_PROJECT_ID = str(uuid.uuid4())
FAKE_USER_ID = "Test User"
FAKE_ORGANIZATION_ID = "Test Organization"
FAKE_PROJECT_NAME = "Test Project"
FAKE_PROJECT_DESCRIPTION = "Test Description"
DEFAULT_PROJECT_CONFIG = ProjectConfig()
DUMMY_API_KEY = "GP0FzPfA7wYy5L64luII2YaRT2JoSXkae7WEo7dH6Bw"


def test_client_uses_default_organization(mock_client_from_api_key: MagicMock) -> None:
    """Test that client uses first organization when none specified"""
    default_org_id = "default-org-id"
    mock_client_from_api_key.users.myself.organizations.list.return_value = UserOrganizationsSchema(
        organizations=[
            Organization(
                organization_id=default_org_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id=FAKE_USER_ID,
            )
        ],
    )
    client = Client(DUMMY_API_KEY)  # no organization_id provided
    assert client.organization_id == default_org_id


def test_client_uses_specified_organization(mock_client_from_api_key: MagicMock) -> None:
    """Test that client uses specified organization ID"""
    specified_org_id = "specified-org-id"
    client = Client(DUMMY_API_KEY, organization_id=specified_org_id)
    assert client.organization_id == specified_org_id
    # Verify we don't unnecessarily call list_organizations
    mock_client_from_api_key.users.myself.organizations.list.assert_not_called()


def test_create_project_without_description(mock_client_from_api_key: MagicMock) -> None:
    """Test creating project with no description"""
    mock_client_from_api_key.projects.create.return_value = ProjectReturnSchema(
        id=FAKE_PROJECT_ID,
        config=ProjectReturnConfig(),
        created_at=datetime.now(),
        created_by_user_id=FAKE_USER_ID,
        name=FAKE_PROJECT_NAME,
        organization_id=FAKE_ORGANIZATION_ID,
        updated_at=datetime.now(),
        description=None,
    )
    client = Client(DUMMY_API_KEY, organization_id=FAKE_ORGANIZATION_ID)
    project = client.create_project(FAKE_PROJECT_NAME)  # no description
    mock_client_from_api_key.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=None,
    )
    assert project.id == FAKE_PROJECT_ID


def test_client_authentication_error() -> None:
    """Test handling of invalid API key"""
    mock_error = Mock(response=Mock(status=401), body={"error": "Expired"})

    with patch("cleanlab_codex.client.client_from_api_key") as mock_client_from_api_key:
        mock_client_from_api_key.side_effect = AuthenticationError(
            "test", response=mock_error.response, body=mock_error.body
        )

        with pytest.raises(AuthenticationError):
            Client(DUMMY_API_KEY)


def test_get_project_not_found(mock_client_from_api_key: MagicMock) -> None:
    """Test getting a non-existent project"""
    mock_client_from_api_key.projects.retrieve.return_value = None
    client = Client(DUMMY_API_KEY)
    with pytest.raises(MissingProjectError):
        client.get_project("non-existent-id")
    assert mock_client_from_api_key.projects.retrieve.call_count == 1


def test_list_organizations(mock_client_from_api_key: MagicMock) -> None:
    mock_client_from_api_key.users.myself.organizations.list.return_value = UserOrganizationsSchema(
        organizations=[
            Organization(
                organization_id=FAKE_ORGANIZATION_ID,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                user_id=FAKE_USER_ID,
            )
        ],
    )
    client = Client("")
    organizations = client.list_organizations()
    assert len(organizations) == 1
    assert organizations[0].organization_id == FAKE_ORGANIZATION_ID
    assert organizations[0].user_id == FAKE_USER_ID


def test_create_project(mock_client_from_api_key: MagicMock) -> None:
    mock_client_from_api_key.projects.create.return_value = ProjectReturnSchema(
        id=FAKE_PROJECT_ID,
        config=Config(),
        created_at=datetime.now(),
        created_by_user_id=FAKE_USER_ID,
        name=FAKE_PROJECT_NAME,
        organization_id=FAKE_ORGANIZATION_ID,
        updated_at=datetime.now(),
        description=FAKE_PROJECT_DESCRIPTION,
    )
    mock_client_from_api_key.organization_id = FAKE_ORGANIZATION_ID
    codex = Client("", organization_id=FAKE_ORGANIZATION_ID)
    project = codex.create_project(FAKE_PROJECT_NAME, FAKE_PROJECT_DESCRIPTION)
    mock_client_from_api_key.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=FAKE_PROJECT_DESCRIPTION,
    )
    assert project.id == FAKE_PROJECT_ID
    assert mock_client_from_api_key.projects.retrieve.call_count == 0


def test_get_project(mock_client_from_api_key: MagicMock) -> None:
    mock_client_from_api_key.projects.retrieve.return_value = ProjectReturnSchema(
        id=FAKE_PROJECT_ID,
        config=Config(),
        created_at=datetime.now(),
        created_by_user_id=FAKE_USER_ID,
        name=FAKE_PROJECT_NAME,
        organization_id=FAKE_ORGANIZATION_ID,
        updated_at=datetime.now(),
        description=FAKE_PROJECT_DESCRIPTION,
    )

    project = Client("").get_project(FAKE_PROJECT_ID)
    assert project.id == FAKE_PROJECT_ID

    assert mock_client_from_api_key.projects.retrieve.call_count == 1
    assert mock_client_from_api_key.projects.retrieve.call_args[0][0] == FAKE_PROJECT_ID
