"""Module for interacting with a Codex project."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Dict, Literal, Optional

from codex import AuthenticationError

from cleanlab_codex.internal.analytics import _AnalyticsMetadata
from cleanlab_codex.internal.sdk_client import client_from_access_key
from cleanlab_codex.types.project import ProjectConfig

if _TYPE_CHECKING:
    from datetime import datetime

    from codex import Codex as _Codex
    from codex.types.project_validate_response import ProjectValidateResponse


_ERROR_CREATE_ACCESS_KEY = (
    "Failed to create access key. Please ensure you have the necessary permissions "
    "and are using a user-level API key, not a project access key. "
    "See cleanlab_codex.Client.get_project."
)


class MissingProjectError(Exception):
    """Raised when the project ID or access key does not match any existing project."""

    def __str__(self) -> str:
        return "valid project ID or access key is required to authenticate access"


class Project:
    """Represents a Codex project.

    To integrate a Codex project into your RAG/Agentic system, we recommend using one of our abstractions such as [`Validator`](/codex/api/python/validator).
    """

    def __init__(self, sdk_client: _Codex, project_id: str, *, verify_existence: bool = True):
        """Initialize the Project. This method is not meant to be used directly.
        Instead, use the [`Client.get_project()`](/codex/api/python/client#method-get_project),
        [`Client.create_project()`](/codex/api/python/client#method-create_project), or [`Project.from_access_key()`](/codex/api/python/project#classmethod-from_access_key) methods.

        Args:
            sdk_client (Codex): The Codex SDK client to use to interact with the project.
            project_id (str): The ID of the project.
            verify_existence (bool, optional): Whether to verify that the project exists.
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
        """Initialize a Project from a [project-level access key](/codex/web_tutorials/create_project/#access-keys).

        Args:
            access_key (str): The access key for authenticating project access.

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
    def create(
        cls,
        sdk_client: _Codex,
        organization_id: str,
        name: str,
        description: str | None = None,
    ) -> Project:
        """Create a new Codex project. This method is not meant to be used directly. Instead, use the [`create_project`](/codex/api/python/client#method-create_project) method on the `Client` class.

        Args:
            sdk_client (Codex): The Codex SDK client to use to create the project. This client must be authenticated with a user-level API key.
            organization_id (str): The ID of the organization to create the project in.
            name (str): The name of the project.
            description (str, optional): The description of the project.

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
            extra_headers=_AnalyticsMetadata().to_headers(),
        ).id

        return Project(sdk_client, project_id, verify_existence=False)

    def create_access_key(
        self,
        name: str,
        description: str | None = None,
        expiration: datetime | None = None,
    ) -> str:
        """Create a new access key for this project. Must be authenticated with a user-level API key to use this method.
        See [`Client.create_project()`](/codex/api/python/client#method-create_project) or [`Client.get_project()`](/codex/api/python/client#method-get_project).

        Args:
            name (str): The name of the access key.
            description (str, optional): The description of the access key.
            expiration (datetime, optional): The expiration date of the access key. If not provided, the access key will not expire.

        Returns:
            str: The access key token.

        Raises:
            AuthenticationError: If the Project was created from a project-level access key instead of a [Client instance](/codex/api/python/client#class-client).
        """
        try:
            return self._sdk_client.projects.access_keys.create(
                project_id=self.id,
                name=name,
                description=description,
                expires_at=expiration,
                extra_headers=_AnalyticsMetadata().to_headers(),
            ).token
        except AuthenticationError as e:
            raise AuthenticationError(_ERROR_CREATE_ACCESS_KEY, response=e.response, body=e.body) from e

    def validate(
        self,
        context: str,
        prompt: str,
        query: str,
        response: str,
        *,
        custom_metadata: Optional[object] = None,
        eval_scores: Optional[Dict[str, float]] = None,
        eval_thresholds: Optional[Dict[str, float]] = None,
        quality_preset: Literal["best", "high", "medium", "low", "base"] = "medium",
    ) -> ProjectValidateResponse:
        """Run validation on a query to an AI system.

        Args:
            context (str): The context used by the AI system to generate a response for the query.
            prompt (str): The full prompt (including system instructions, context, and the original query) used by the AI system to generate a response for the query.
            query (str): The original user input to the AI system.
            response (str): The response generated by the AI system for the query.
            custom_metadata (object, optional): Custom metadata to log in Codex for the query.
            eval_scores (Dict[str, float], optional): Optional scores to use for the query. When provided, Codex will skip running TrustworthyRAG evaluations on the query and use the provided scores instead.
            eval_thresholds (Dict[str, float], optional): Optional thresholds to use for evaluating the query. We recommend configuring thresholds on the Project instead and using the same thresholds for all queries.
            quality_preset (Literal["best", "high", "medium", "low", "base"], optional): The quality preset to use for the query.

        Returns:
            ProjectValidateResponse: The response from the validation.
        """
        return self._sdk_client.projects.validate(
            self._id,
            context=context,
            prompt=prompt,
            query=query,
            response=response,
            custom_eval_thresholds=eval_thresholds,
            custom_metadata=custom_metadata,
            eval_scores=eval_scores,
            quality_preset=quality_preset,
        )

    def add_remediation(self, question: str, answer: str | None = None) -> None:
        """Add a remediation to the project. A remediation represents a question and answer pair that is expert verified
        and should be used to answer future queries to the AI system that are similar to the question.

        Args:
            question (str): The question to add to the project.
            answer (str, optional): The expert answer for the question. If not provided, the question will be added to the project without an expert answer.
        """
        self._sdk_client.projects.remediations.create(
            project_id=self.id,
            question=question,
            answer=answer,
            extra_headers=_AnalyticsMetadata().to_headers(),
        )
