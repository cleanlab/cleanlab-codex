"""Module for interacting with a Codex project."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Optional

from codex import AuthenticationError

from cleanlab_codex.internal.project import query_project
from cleanlab_codex.internal.sdk_client import client_from_access_key
from cleanlab_codex.types.project import ProjectConfig

if _TYPE_CHECKING:
    from datetime import datetime

    from codex import Codex as _Codex

    from cleanlab_codex.types.entry import Entry, EntryCreate

_ERROR_CREATE_ACCESS_KEY = (
    "Failed to create access key. Please ensure you have the necessary permissions "
    "and are using a user-level API key, not a project access key. "
    "See cleanlab_codex.Client.get_project."
)

_ERROR_ADD_ENTRIES = (
    "Failed to add entries. Please ensure you have the necessary permissions "
    "and are using a user-level API key, not a project access key. "
    "See cleanlab_codex.Client.get_project."
)


class MissingProjectError(Exception):
    """Raised when the project ID or access key does not match any existing project."""

    def __str__(self) -> str:
        return "valid project ID or access key is required to authenticate access"


class Project:
    """Represents a Codex project.

    To integrate a Codex project into your RAG/Agentic system, we recommend using one of our abstractions such as [`CodexTool`](/reference/python/codex_tool).
    The [`query`](#method-query) method can also be used directly if none of our existing abstractions are sufficient for your use case.
    """

    def __init__(self, sdk_client: _Codex, project_id: str, *, verify_existence: bool = True):
        """Initialize the Project. This method is not meant to be used directly.
        Instead, use the [`Client.get_project()`](/reference/python/client#method-get_project),
        [`Client.create_project()`](/reference/python/client#method-create_project), or [`Project.from_access_key()`](/reference/python/project#classmethod-from_access_key) methods.

        Args:
            sdk_client (`Codex`): The Codex SDK client to use to interact with the project.
            project_id (`str`): The ID of the project.
            verify_existence (`bool`, optional): Whether to verify that the project exists.
        """
        self._sdk_client = sdk_client
        self._id = project_id

        # make sure the project exists
        if verify_existence and sdk_client.projects.retrieve(project_id) is None:
            raise MissingProjectError

    @property
    def id(self) -> str:
        """The ID of the project."""
        return self._id

    @classmethod
    def from_access_key(cls, access_key: str) -> Project:
        """Initialize a Project from a [project-level access key](/codex/sme_tutorials/getting_started/#access-keys).

        Args:
            access_key (`str`): The access key for authenticating project access.

        Returns:
            Project: The project associated with the access key.
        """
        sdk_client = client_from_access_key(access_key)

        try:
            project_id = sdk_client.projects.access_keys.retrieve_project_id().project_id
        except Exception as e:
            raise MissingProjectError from e

        return Project(sdk_client, project_id, verify_existence=False)

    @classmethod
    def create(cls, sdk_client: _Codex, organization_id: str, name: str, description: str | None = None) -> Project:
        """Create a new Codex project. This method is not meant to be used directly. Instead, use the [`create_project`](/reference/python/client#method-create_project) method on the `Client` class.

        Args:
            sdk_client (`Codex`): The Codex SDK client to use to create the project. This client must be authenticated with a user-level API key.
            organization_id (`str`): The ID of the organization to create the project in.
            name (`str`): The name of the project.
            description (`str`, optional): The description of the project.

        Returns:
            Project: The created project.

        Raises:
            AuthenticationError: If the SDK client is not authenticated with a user-level API key.
        """
        project_id = sdk_client.projects.create(
            config=ProjectConfig(),
            organization_id=organization_id,
            name=name,
            description=description,
        ).id

        return Project(sdk_client, project_id, verify_existence=False)

    def create_access_key(self, name: str, description: str | None = None, expiration: datetime | None = None) -> str:
        """Create a new access key for this project. Must be authenticated with a user-level API key to use this method.
        See [`Client.create_project()`](/reference/python/client#method-create_project) or [`Client.get_project()`](/reference/python/client#method-get_project).

        Args:
            name (`str`): The name of the access key.
            description (`str`, optional): The description of the access key.
            expiration (`datetime`, optional): The expiration date of the access key. If not provided, the access key will not expire.

        Returns:
            str: The access key token.

        Raises:
            AuthenticationError: If the Project was created from a project-level access key instead of a [Client instance](/reference/python/client#class-client).
        """
        try:
            return self._sdk_client.projects.access_keys.create(
                project_id=self.id, name=name, description=description, expires_at=expiration
            ).token
        except AuthenticationError as e:
            raise AuthenticationError(_ERROR_CREATE_ACCESS_KEY, response=e.response, body=e.body) from e

    def add_entries(self, entries: list[EntryCreate]) -> None:
        """Add a list of entries to this Codex project. Must be authenticated with a user-level API key to use this method.
        See [`Client.create_project()`](/reference/python/client#method-create_project) or [`Client.get_project()`](/reference/python/client#method-get_project).

        Args:
            entries (list[EntryCreate]): The entries to add to this project. See [`EntryCreate`](/reference/python/types.entry#class-entrycreate).

        Raises:
            AuthenticationError: If the Project was created from a project-level access key instead of a [Client instance](/reference/python/client#class-client).
        """
        try:
            # TODO: implement batch creation of entries in backend and update this function
            for entry in entries:
                self._sdk_client.projects.entries.create(
                    self.id, question=entry["question"], answer=entry.get("answer")
                )
        except AuthenticationError as e:
            raise AuthenticationError(_ERROR_ADD_ENTRIES, response=e.response, body=e.body) from e

    def query(
        self,
        question: str,
        *,
        fallback_answer: Optional[str] = None,
        read_only: bool = False,
    ) -> tuple[Optional[str], Optional[Entry]]:
        """Query Codex to check if this project contains an answer to the question. Add the question to the project for SME review if it does not.

        Args:
            question (`str`): The question to ask the Codex API.
            fallback_answer (`str`, optional): Optional fallback answer to return if Codex is unable to answer the question.
            read_only (`bool`, optional): Whether to query the Codex API in read-only mode. If True, the question will not be added to the Codex project for SME review.
                This can be useful for testing purposes when setting up your project configuration.

        Returns:
            tuple[Optional[str], Optional[Entry]]: A tuple representing the answer for the query and the existing or new entry in the Codex project.
                If Codex is able to answer the question, the first element will be the answer returned by Codex and the second element will be the existing [`Entry`](/reference/python/types.entry#class-entry) in the Codex project.
                If Codex is unable to answer the question, the first element will be `fallback_answer` if provided, otherwise None. The second element will be a new [`Entry`](/reference/python/types.entry#class-entry) in the Codex project.
        """
        return query_project(
            client=self._sdk_client,
            question=question,
            project_id=self.id,
            fallback_answer=fallback_answer,
            read_only=read_only,
        )
