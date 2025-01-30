# ruff: noqa: DTZ005

import uuid
from datetime import datetime
from unittest.mock import MagicMock

from codex.types.project_return_schema import Config, ProjectReturnSchema
from codex.types.users.myself.user_organizations_schema import UserOrganizationsSchema

from cleanlab_codex.client import Client
from cleanlab_codex.types.organization import Organization
from cleanlab_codex.types.project import ProjectConfig

FAKE_PROJECT_ID = str(uuid.uuid4())
FAKE_USER_ID = "Test User"
FAKE_ORGANIZATION_ID = "Test Organization"
FAKE_PROJECT_NAME = "Test Project"
FAKE_PROJECT_DESCRIPTION = "Test Description"
DEFAULT_PROJECT_CONFIG = ProjectConfig()


def test_list_organizations(mock_client: MagicMock) -> None:
    mock_client.users.myself.organizations.list.return_value = UserOrganizationsSchema(
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


def test_create_project(mock_client: MagicMock) -> None:
    mock_client.projects.create.return_value = ProjectReturnSchema(
        id=FAKE_PROJECT_ID,
        config=Config(),
        created_at=datetime.now(),
        created_by_user_id=FAKE_USER_ID,
        name=FAKE_PROJECT_NAME,
        organization_id=FAKE_ORGANIZATION_ID,
        updated_at=datetime.now(),
        description=FAKE_PROJECT_DESCRIPTION,
    )
    codex = Client("", organization_id=FAKE_ORGANIZATION_ID)
    project = codex.create_project(FAKE_PROJECT_NAME, FAKE_PROJECT_DESCRIPTION)
    mock_client.projects.create.assert_called_once_with(
        config=DEFAULT_PROJECT_CONFIG,
        organization_id=FAKE_ORGANIZATION_ID,
        name=FAKE_PROJECT_NAME,
        description=FAKE_PROJECT_DESCRIPTION,
    )
    assert project.project_id == FAKE_PROJECT_ID
