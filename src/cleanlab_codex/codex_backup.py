from __future__ import annotations

import os
from functools import wraps
from typing import Any, Optional, Protocol, Sequence, Union, cast, Self

import requests

from cleanlab_codex.project import Project
from cleanlab_codex.validation import is_bad_response


def handle_backup_default(codex_response: str, primary_system: Any) -> None:  # noqa: ARG001
    """Default implementation is a no-op."""
    return None


class _TLM(Protocol):
    def get_trustworthiness_score(
        self,
        query: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
        /,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


class _TemporaryTLM(_TLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        self.api_base_url = api_base_url.rstrip("/") if api_base_url else os.getenv("CODEX_API_BASE_URL")
        assert self.api_base_url is not None, "CODEX_API_BASE_URL is not set"
        self._headers = {
            "X-API-Key": api_key or os.getenv("CODEX_API_KEY"),
            "Content-Type": "application/json",
        }

    def _make_request(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make a request to the TLM API."""
        url = f"{self.api_base_url}/api/tlm/{endpoint}"
        response = requests.post(
            url,
            json=data,
            headers=self._headers,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def get_trustworthiness_score(
        self, query: Union[str, Sequence[str]], response: Union[str, Sequence[str]], **kwargs: Any
    ) -> dict[str, Any]:
        """Get trustworthiness score for a query-response pair."""
        data = {"prompt": query, "response": response, **kwargs}
        return self._make_request("score", data)

    def prompt(self, prompt: Union[str, Sequence[str]], /, **kwargs: Any) -> dict[str, Any]:
        """Send a prompt to the TLM API."""
        data = {"prompt": prompt, **kwargs}
        return self._make_request("prompt", data)


class BackupHandler(Protocol):
    """Protocol defining how to handle backup responses from Codex.

    This protocol defines a callable interface for processing Codex responses that are
    retrieved when the primary response system (e.g., a RAG system) fails to provide
    an adequate answer. Implementations of this protocol can be used to:

    - Update the primary system's context or knowledge base
    - Log Codex responses for analysis
    - Trigger system improvements or retraining
    - Perform any other necessary side effects

    Args:
        codex_response (str): The response received from Codex
        primary_system (Any): The instance of the primary RAG system that
            generated the inadequate response. This allows the handler to
            update or modify the primary system if needed.

    Returns:
        None: The handler performs side effects but doesn't return a value
    """

    def __call__(self, codex_response: str, primary_system: Any) -> None: ...


class CodexBackup:
    """A backup decorator that connects to a Codex project to answer questions that
    cannot be adequately answered by the existing agent.
    
    Args:
        project: The Codex project to use for backup responses
        fallback_answer: The fallback answer to use if the primary system fails
        backup_handler: A callback function that processes Codex's response and updates the primary RAG system. This handler is called whenever Codex provides a backup response after the primary system fails. By default, the backup handler is a no-op.
        primary_system: The existing RAG system that needs to be backed up by Codex
        is_bad_response_kwargs: Additional keyword arguments to pass to the is_bad_response function, for detecting inadequate responses from the primary system
    """

    DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."

    def __init__(
        self,
        *,
        project: Project,
        fallback_answer: str = DEFAULT_FALLBACK_ANSWER,
        backup_handler: BackupHandler = handle_backup_default,
        primary_system: Optional[Any] = None,
        is_bad_response_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._project = project
        self._fallback_answer = fallback_answer
        self._backup_handler = backup_handler
        self._tlm = _TemporaryTLM()  # TODO: Improve integration
        self._primary_system: Optional[Any] = primary_system
        self._is_bad_response_kwargs = is_bad_response_kwargs

    @classmethod
    def from_project(cls, project: Project, **kwargs: Any) -> Self:
        return cls(project=project, **kwargs)

    @property
    def primary_system(self) -> Any:
        if self._primary_system is None:
            raise ValueError("Primary system not set. Please set a primary system using the `add_primary_system` method.")
        return self._primary_system

    @primary_system.setter
    def primary_system(self, primary_system: Any) -> None:
        """Set the primary RAG system that will be used to generate responses."""
        self._primary_system = primary_system

    @property
    def is_bad_response_kwargs(self) -> dict[str, Any]:
        return self._is_bad_response_kwargs or {}

    @is_bad_response_kwargs.setter
    def is_bad_response_kwargs(self, is_bad_response_kwargs: dict[str, Any]) -> None:
        self._is_bad_response_kwargs = is_bad_response_kwargs

    def run(
        self,
        response: str,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """Check if a response is adequate and provide a backup from Codex if needed.

        Args:
            primary_system: The system that generated the original response
            response: The response to evaluate
            query: The original query that generated the response
            context: Optional context used to generate the response
            
        Returns:
            str: Either the original response if adequate, or a backup response from Codex
        """

        _is_bad_response_kwargs = self.is_bad_response_kwargs
        if not is_bad_response(
            response,
            query=query,
            context=context,
            tlm=self._tlm,
            fallback_answer=self._fallback_answer,
            **_is_bad_response_kwargs,
        ):
            return response

        cache_result = self._project.query(query, fallback_answer=self._fallback_answer)[0]
        if not cache_result:
            return response

        if self._primary_system is not None:
            self._backup_handler(
                codex_response=cache_result,
                primary_system=self._primary_system,
            )
        return cache_result
