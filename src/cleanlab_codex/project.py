"""Module for interacting with a Codex project."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Dict, Optional, Union, cast

from codex import AuthenticationError
from codex.types.project_validate_params import Response

from cleanlab_codex.internal.analytics import _AnalyticsMetadata
from cleanlab_codex.internal.sdk_client import client_from_access_key
from cleanlab_codex.types.project import ProjectConfig

if _TYPE_CHECKING:
    from datetime import datetime

    from codex import Codex as _Codex
    from codex.types.project_validate_response import ProjectValidateResponse
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


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

    To integrate a Codex project into your RAG/Agentic system, we recommend using the [`Project.validate()` method](/codex/api/python/project#method-validate).
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
        *,
        messages: list[ChatCompletionMessageParam],
        response: Union[ChatCompletion, str],
        query: str,
        context: str,
        rewritten_query: Optional[str] = None,
        metadata: Optional[object] = None,
        eval_scores: Optional[Dict[str, float]] = None,
    ) -> ProjectValidateResponse:
        """Evaluate the quality of an AI-generated response using the structured message history, query, and retrieved context.

        This method runs validation on an AI response using the full `messages` history (formatted as OpenAI-style chat messages),
        which should include the latest user query and any preceding system or assistant messages.

        **For single-turn Q&A apps, `messages` can be a minimal list with one user message. For multi-turn conversations, provide the full dialog
        leading up to the final response.

        The function assesses the trustworthiness and quality of the AI `response` in light of the provided `context` and
        `query`, which should align with the most recent user message in `messages`.

        If your AI response is flagged as problematic, then this method will:
            - return an expert answer if one was previously provided for a similar query
            - otherwise log this query for future SME review (to consider providing an expert answer) in the Web interface.

        Args:
            messages (list[ChatCompletionMessageParam]): The full message history from the AI conversation, formatted for OpenAI-style chat completion.
                This must include the final user message that triggered the AI response. All other arguments—`query`, `context`, and `response`—
                must correspond specifically to this final user message.
            response (ChatCompletion | str): Your AI-response that was generated based on the given `messages`. This is the response being evaluated, and should not appear in the `messages`.
            query (str): The user query that the `response` is answering. This query should be in the latest user message in `messages`.
            context (str): The retrieved context (e.g., from your RAG system) that was supplied to the AI when generating the `response` to the final user query in `messages`.
            rewritten_query (str, optional): An optional reformulation of `query` (e.g. made self-contained w.r.t multi-turn conversations) to improve retrieval quality.
            metadata (object, optional): Arbitrary metadata to associate with this validation for logging or analysis inside the Codex project.
            eval_scores (dict[str, float], optional): Precomputed evaluation scores to bypass automatic scoring. Providing `eval_scores` for specific evaluations bypasses automated scoring and uses the supplied scores instead. Consider providing these scores if you already have them precomputed to reduce runtime.

        Returns:
            ProjectValidateResponse: A structured object with the following fields:

                - should_guardrail (bool): True if the AI system should suppress or modify the response before returning it to the user. When True, the response is considered problematic and may require further review or modification.
                - escalated_to_sme (bool): True if the query should be escalated to SME for review. When True, the query is logged and may be answered by an expert.
                - eval_scores (dict[str, ThresholdedEvalScore]): Evaluation scores for different response attributes (e.g., trustworthiness, helpfulness, ...). Each includes a numeric score and a `failed` flag indicating whether the score falls below threshold.
                - expert_answer (str | None): If it was auto-determined that this query should be escalated to SME, and a prior SME answer for a similar query was found, then this will return that expert answer.  Otherwise, it is None.

                When available, consider swapping your AI response with the expert answer before serving the response to your user.
        """

        return self._sdk_client.projects.validate(
            self._id,
            messages=messages,
            response=cast(Response, response),
            context=context,
            query=query,
            rewritten_question=rewritten_query,
            custom_metadata=metadata,
            eval_scores=eval_scores,
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
