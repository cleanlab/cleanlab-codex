from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cleanlab_codex.internal.utils import init_codex_client
from cleanlab_codex.project import Project

if TYPE_CHECKING:
    from cleanlab_codex.types.organization import Organization

from cleanlab_codex.types.project import ProjectConfig


class Client:
    def __init__(self, api_key: str, organization_id: Optional[str] = None):
        """Initialize the Codex client.

        Args:
            api_key (str): The key to authenticate with Cleanlab Codex. Can either be a user-level API Key or a project-level Access Key. (TODO: link to docs on what these are).
            organization_id (str): The ID of the organization to create the project in. If not provided, the user's default organization will be used.
        Returns:
            Client: The authenticated Codex Client.

        Raises:
            AuthenticationError: If the key is invalid.
        """
        self.api_key = api_key
        self._client = init_codex_client(api_key)

        self.organization_id = organization_id if organization_id else self.list_organizations()[0].organization_id

    def get_project(self, project_id: str) -> Project:
        return Project(self._client, project_id)

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new Codex project for the authenticated user.

        Args:
            name (str): The name of the project.
            description (:obj:`str`, optional): The description of the project.

        Returns:
            Project: The created project.
        """

        project_id = self._client.projects.create(
            config=ProjectConfig(),
            organization_id=self.organization_id,
            name=name,
            description=description,
        ).id

        return Project(self._client, project_id)

    def list_organizations(self) -> list[Organization]:
        """List the organizations the authenticated user is a member of.

        Returns:
            list[Organization]: A list of organizations the authenticated user is a member of.

        Raises:
            AuthenticationError: If the client is not authenticated with a user-level API Key.
        """
        return self._client.users.myself.organizations.list().organizations
