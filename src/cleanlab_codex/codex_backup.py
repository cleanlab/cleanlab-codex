"""Enables connecting RAG applications to Codex as a Backup system.

This module provides functionality to use Codex as a fallback when a primary
RAG (Retrieval-Augmented Generation) system fails to provide adequate responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from cleanlab_codex.response_validation import BadResponseDetectionConfig, is_bad_response

if TYPE_CHECKING:
    from cleanlab_codex.project import Project
    from cleanlab_codex.types.backup import BackupHandler
    from cleanlab_codex.types.tlm import TLM


def handle_backup_default(codex_response: str, primary_system: Any) -> None:  # noqa: ARG001
    """Default implementation is a no-op."""
    return None


class CodexBackup:
    """A backup decorator that connects to a Codex project to answer questions that
    cannot be adequately answered by the existing agent.

    Args:
        project: The Codex project to use for backup responses
        fallback_answer: The fallback answer to use if the primary system fails to provide an adequate response
        backup_handler: A callback function that processes Codex's response and updates the primary RAG system. This handler is called whenever Codex provides a backup response after the primary system fails. By default, the backup handler is a no-op.
        primary_system: The existing RAG system that needs to be backed up by Codex
        tlm: The client for the Trustworthy Language Model, which evaluates the quality of responses from the primary system
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
        tlm: Optional[TLM] = None,
        is_bad_response_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._project = project
        self._fallback_answer = fallback_answer
        self._backup_handler = backup_handler
        self._primary_system: Optional[Any] = primary_system
        self._tlm = tlm
        self._is_bad_response_kwargs = is_bad_response_kwargs or {}

    @classmethod
    def from_project(cls, project: Project, **kwargs: Any) -> CodexBackup:
        return cls(project=project, **kwargs)

    @property
    def primary_system(self) -> Any:
        if self._primary_system is None:
            error_message = "Primary system not set. Please set a primary system using the `add_primary_system` method."
            raise ValueError(error_message)
        return self._primary_system

    @primary_system.setter
    def primary_system(self, primary_system: Any) -> None:
        """Set the primary RAG system that will be used to generate responses."""
        self._primary_system = primary_system

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

        is_bad = is_bad_response(
            response,
            query=query,
            context=context,
            config=BadResponseDetectionConfig.model_validate(
                {
                    "tlm": self._tlm,
                    "fallback_answer": self._fallback_answer,
                    **self._is_bad_response_kwargs,
                },
            ),
        )
        if not is_bad:
            return response

        codex_response = self._project.query(query, fallback_answer=self._fallback_answer)[0]
        if not codex_response:
            return response

        if self._primary_system is not None:
            self._backup_handler(
                codex_response=codex_response,
                primary_system=self._primary_system,
            )
        return codex_response
