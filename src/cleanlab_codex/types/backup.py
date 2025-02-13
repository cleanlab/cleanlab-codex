"""Types for Codex Backup."""

from __future__ import annotations

from typing import Any, Protocol


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
