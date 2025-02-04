from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cleanlab_codex.internal.utils import client_from_api_key
from cleanlab_codex.project import Project

if TYPE_CHECKING:
    from cleanlab_codex.types.organization import Organization


class Client:
    def __init__(self, api_key: str, organization_id: Optional[str] = None):
        """Initialize user-level access to the Codex SDK.

        Args:
            api_key (str): The API key for authenticating the user. (TODO: link to docs on what this means)
            organization_id (str): The ID of the organization to create projects in. If not provided, the user's default organization will be used.
        Returns:
            Client: The authenticated Codex Client.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        self.api_key = api_key
        self._client = client_from_api_key(api_key)

        self.organization_id = (
            organization_id if organization_id is not None else self.list_organizations()[0].organization_id
        )

    def get_project(self, project_id: str) -> Project:
        """Get a project by ID. Must be accessible by the authenticated user.

        Args:
            project_id (str): The ID of the project to get.

        Returns:
            Project: The project.
        """
        return Project(self._client, project_id)

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new Codex project for the authenticated user.

        Args:
            name (str): The name of the project.
            description (:obj:`str`, optional): The description of the project.

        Returns:
            Project: The created project.
        """

        return Project.create(self._client, self.organization_id, name, description)

    def list_organizations(self) -> list[Organization]:
        """List the organizations the authenticated user is a member of.

        Returns:
            list[Organization]: A list of organizations the authenticated user is a member of.
        """
        return self._client.users.myself.organizations.list().organizations
