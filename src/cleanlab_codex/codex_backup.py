from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional

from cleanlab_codex.codex import Codex
from cleanlab_codex.validation import is_bad_response


def handle_backup_default(backup_response: str, decorated_instance: Any) -> None:  # noqa: ARG001
    """Default implementation is a no-op."""
    return None


class CodexBackup:
    """A backup decorator that connects to a Codex project to answer questions that
    cannot be adequately answered by the existing agent.
    """

    DEFAULT_FALLBACK_ANSWER = "Based on the available information, I cannot provide a complete answer to this question."

    def __init__(
        self,
        codex_client: Codex,
        *,
        project_id: Optional[str] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
        backup_handler: Callable[[str, Any], None] = handle_backup_default,
    ):
        self._codex_client = codex_client
        self._project_id = project_id
        self._fallback_answer = fallback_answer
        self._backup_handler = backup_handler

    @classmethod
    def from_access_key(
        cls,
        access_key: str,
        *,
        project_id: Optional[str] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
        backup_handler: Callable[[str, Any], None] = handle_backup_default,
    ) -> CodexBackup:
        """Creates a CodexBackup from an access key. The project ID that the CodexBackup will use is the one that is associated with the access key."""
        return cls(
            codex_client=Codex(key=access_key),
            project_id=project_id,
            fallback_answer=fallback_answer,
            backup_handler=backup_handler,
        )

    @classmethod
    def from_client(
        cls,
        codex_client: Codex,
        *,
        project_id: Optional[str] = None,
        fallback_answer: Optional[str] = DEFAULT_FALLBACK_ANSWER,
        backup_handler: Callable[[str, Any], None] = handle_backup_default,
    ) -> CodexBackup:
        """Creates a CodexBackup from a Codex client.
        If the Codex client is initialized with a project access key, the CodexBackup will use the project ID that is associated with the access key.
        If the Codex client is initialized with a user API key, a project ID must be provided.
        """
        return cls(
            codex_client=codex_client,
            project_id=project_id,
            fallback_answer=fallback_answer,
            backup_handler=backup_handler,
        )

    def to_decorator(self):
        """Factory that creates a backup decorator using the provided Codex client"""

        def decorator(chat_method):
            """
            Decorator for RAG chat methods that adds backup response handling.

            If the original chat method returns an inadequate response, attempts to get
            a backup response from Codex. Returns the backup response if available,
            otherwise returns the original response.

            Args:
                chat_method: Method with signature (self, user_message: str) -> str
                    where 'self' refers to the instance being decorated, not an instance of CodexBackup.
            """

            @wraps(chat_method)
            def wrapper(decorated_instance, user_message):
                # Call the original chat method
                assistant_response = chat_method(decorated_instance, user_message)

                # Return original response if it's adequate
                if not is_bad_response(assistant_response):
                    return assistant_response

                # Query Codex for a backup response
                cache_result = self._codex_client.query(user_message)[0]
                if not cache_result:
                    return assistant_response

                # Handle backup response if handler exists
                self._backup_handler(
                    backup_response=cache_result,
                    decorated_instance=decorated_instance,
                )
                return cache_result

            return wrapper

        return decorator
