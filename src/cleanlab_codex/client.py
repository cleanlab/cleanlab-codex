"""Client for interacting with Cleanlab Codex."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Optional

from cleanlab_codex.internal.organization import list_organizations
from cleanlab_codex.internal.sdk_client import client_from_api_key
from cleanlab_codex.project import Project

if _TYPE_CHECKING:
    from cleanlab_codex.types.organization import Organization


class Client:
    """
    Client for interacting with Cleanlab Codex. In order to use this client, make sure you have an account at [codex.cleanlab.ai](https://codex.cleanlab.ai).

    We recommend using the [Web UI](https://codex.cleanlab.ai) to [set up Codex projects](/codex/web_tutorials/create_project), but you can also use this client to programmatically set up Codex projects.
    """

    def __init__(self, api_key: str | None = None, organization_id: Optional[str] = None):
        """Initialize the Codex client.

        Args:
            api_key (str, optional): The API key for authenticating the user. If not provided, the client will attempt to use the API key from the environment variable `CODEX_API_KEY`. You can find your API key at [codex.cleanlab.ai/account](https://codex.cleanlab.ai/account).
            organization_id (str, optional): The ID of the organization the client should use. If not provided, the user's default organization will be used.
        Returns:
            Client: The authenticated Codex Client.

        Raises:
            AuthenticationError: If the API key is invalid.
        """
        self.api_key = api_key
        self._client = client_from_api_key(api_key)

        self._organization_id = (
            organization_id if organization_id is not None else self.list_organizations()[0].organization_id
        )

    @property
    def organization_id(self) -> str:
        """The organization ID the client is using."""
        return self._organization_id

    def get_project(self, project_id: str) -> Project:
        """Get a project by ID. Must be accessible by the authenticated user.

        Args:
            project_id (str): The ID of the project to get.

        Returns:
            Project: The project.
        """
        return Project(self._client, project_id)

    def create_project(self, name: str, description: Optional[str] = None) -> Project:
        """Create a new Codex project. Project will be created in the organization the client is using.

        Args:
            name (str): The name of the project.
            description (str, optional): The description of the project.

        Returns:
            Project: The created project.
        """

        return Project.create(self._client, self._organization_id, name, description)

    def list_organizations(self) -> list[Organization]:
        """List the organizations the authenticated user is a member of.

        Returns:
            list[Organization]: A list of organizations the authenticated user is a member of.
            See [`Organization`](/codex/api/python/types.organization#class-organization) for more information.
        """
        return list_organizations(self._client)
