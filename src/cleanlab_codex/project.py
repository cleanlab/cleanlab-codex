from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from cleanlab_codex.internal.utils import client_from_access_key

if TYPE_CHECKING:
    from datetime import datetime

    from codex import Codex as _Codex

    from cleanlab_codex.types.entry import Entry, EntryCreate


class MissingProjectIdError(Exception):
    """Raised when the project ID is not provided."""

    def __str__(self) -> str:
        return "project_id is required when authenticating with a user-level API Key"


class Project:
    def __init__(self, sdk_client: _Codex, project_id: str):
        self.sdk_client = sdk_client
        self.project_id = project_id

        # make sure the project exists
        self._project = sdk_client.projects.retrieve(project_id)

        if self._project is None:
            raise MissingProjectIdError


    @classmethod
    def from_access_key(cls, access_key: str):
        """Initialize project-level access to the Codex SDK.

        Args:
            access_key (str): The access key for authenticating the project. (TODO: link to docs on what this means).

        Returns:
            Project: The project associated with the access key.
        """
        sdk_client = client_from_access_key(access_key)
        project_id = sdk_client.projects.access_keys.retrieve_project_id().project_id

        if project_id is None:
            raise MissingProjectIdError

        return Project(sdk_client, project_id)

    def create_access_key(self, name: str, description: str | None = None, expiration: datetime | None = None):
        """Create a new access key for this project.

        Args:
            name (str): The name of the access key.
            description (:obj:`str`, optional): The description of the access key.

        Returns:
            str: The access key token.
        """
        return self.sdk_client.projects.access_keys.create(
            project_id=self.project_id, name=name, description=description, expires_at=expiration
        ).token

    def add_entries(self, entries: list[EntryCreate]) -> None:
        """Add a list of entries to this Codex project.

        Args:
            entries (list[EntryCreate]): The entries to add to this project.

        Raises:
            AuthenticationError: If the client is not authenticated with a user-level API Key.
        """
        # TODO: implement batch creation of entries in backend and update this function
        for entry in entries:
            self.sdk_client.projects.entries.create(
                self.project_id, question=entry["question"], answer=entry.get("answer")
            )

    def query(
        self,
        question: str,
        *,
        fallback_answer: Optional[str] = None,
        read_only: bool = False,
    ) -> tuple[Optional[str], Optional[Entry]]:
        """Query Codex to check if this project contains an answer to this question and add the question to the Codex project for SME review if it does not.

        Args:
            question (str): The question to ask the Codex API.
            fallback_answer (:obj:`str`, optional): Optional fallback answer to return if Codex is unable to answer the question.
            read_only (:obj:`bool`, optional): Whether to query the Codex API in read-only mode. If True, the question will not be added to the Codex project for SME review.
                This can be useful for testing purposes before when setting up your project configuration.

        Returns:
            tuple[Optional[str], Optional[Entry]]: A tuple representing the answer for the query and the existing or new entry in the Codex project.
                If Codex is able to answer the question, the first element will be the answer returned by Codex and the second element will be the existing entry in the Codex project.
                If Codex is unable to answer the question, the first element will be `fallback_answer` if provided, otherwise None, and the second element will be a new entry in the Codex project.
        """
        query_res = self.sdk_client.projects.entries.query(self.project_id, question=question)
        if query_res is not None:
            if query_res.answer is not None:
                return query_res.answer, query_res

            return fallback_answer, query_res

        if not read_only:
            created_entry = self.sdk_client.projects.entries.add_question(self.project_id, question=question)
            return fallback_answer, created_entry

        return fallback_answer, None
