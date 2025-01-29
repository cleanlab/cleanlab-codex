from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cleanlab_codex.internal.project import create_project, query_project
from cleanlab_codex.internal.utils import init_codex_client

if TYPE_CHECKING:
    from cleanlab_codex.types.entry import Entry, EntryCreate
    from cleanlab_codex.types.organization import Organization


class Codex:
    """
    A client to interact with Cleanlab Codex.
    """

    def __init__(self, key: str | None = None):
        """Initialize the Codex client.

        Args:
            key (str): The key to authenticate with Cleanlab Codex. Can either be a user-level API Key or a project-level Access Key. (TODO: link to docs on what these are).

        Returns:
            Codex: The authenticated Codex client.

        Raises:
            AuthenticationError: If the key is invalid.
        """
        self.key = key
        self._client = init_codex_client(key)

    def list_organizations(self) -> list[Organization]:
        """List the organizations the authenticated user is a member of.

        Returns:
            list[Organization]: A list of organizations the authenticated user is a member of.

        Raises:
            AuthenticationError: If the client is not authenticated with a user-level API Key.
        """
        return self._client.users.myself.organizations.list().organizations

    def create_project(self, name: str, organization_id: str, description: Optional[str] = None) -> str:
        """Create a new Codex project.

        Args:
            name (str): The name of the project.
            organization_id (str): The ID of the organization to create the project in. Must be authenticated as a member of this organization.
            description (:obj:`str`, optional): The description of the project.

        Returns:
            int: The ID of the created project.
        """
        return create_project(
            client=self._client,
            name=name,
            organization_id=organization_id,
            description=description,
        )

    def add_entries(self, entries: list[EntryCreate], project_id: str) -> None:
        """Add a list of entries to the Codex project.

        Args:
            entries (list[EntryCreate]): The entries to add to the Codex project.
            project_id (int): The ID of the project to add the entries to.

        Raises:
            AuthenticationError: If the client is not authenticated with a user-level API Key.
        """
        # TODO: implement batch creation of entries in backend and update this function
        for entry in entries:
            self._client.projects.entries.create(project_id, question=entry["question"], answer=entry.get("answer"))

    def create_project_access_key(
        self,
        project_id: str,
        access_key_name: str,
        access_key_description: Optional[str] = None,
    ) -> str:
        """Create a new access key for a project.

        Args:
            project_id (int): The ID of the project to create the access key for.
            access_key_name (str): The name of the access key.
            access_key_description (:obj:`str`, optional): The description of the access key.

        Returns:
            str: The access key token.
        """
        return self._client.projects.access_keys.create(
            project_id=project_id,
            name=access_key_name,
            description=access_key_description,
        ).token

    def query(
        self,
        question: str,
        *,
        project_id: Optional[str] = None,  # TODO: update to uuid once project IDs are changed to UUIDs
        fallback_answer: Optional[str] = None,
        read_only: bool = False,
    ) -> tuple[Optional[str], Optional[Entry]]:
        """Query Codex to check if the Codex project contains an answer to this question and add the question to the Codex project for SME review if it does not.

        Args:
            question (str): The question to ask the Codex API.
            project_id (:obj:`int`, optional): The ID of the project to query.
                If the client is authenticated with a user-level API Key, this is required.
                If the client is authenticated with a project-level Access Key, this is optional. The client will use the Access Key's project ID by default.
            fallback_answer (:obj:`str`, optional): Optional fallback answer to return if Codex is unable to answer the question.
            read_only (:obj:`bool`, optional): Whether to query the Codex API in read-only mode. If True, the question will not be added to the Codex project for SME review.
                This can be useful for testing purposes before when setting up your project configuration.

        Returns:
            tuple[Optional[str], Optional[Entry]]: A tuple representing the answer for the query and the existing or new entry in the Codex project.
                If Codex is able to answer the question, the first element will be the answer returned by Codex and the second element will be the existing entry in the Codex project.
                If Codex is unable to answer the question, the first element will be `fallback_answer` if provided, otherwise None, and the second element will be a new entry in the Codex project.
        """
        return query_project(
            client=self._client,
            question=question,
            project_id=project_id,
            fallback_answer=fallback_answer,
            read_only=read_only,
        )
