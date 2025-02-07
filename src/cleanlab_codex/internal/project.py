from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from codex import Codex as _Codex

from cleanlab_codex.types.entry import Entry


class MissingProjectIdError(Exception):
    """Raised when the project ID is not provided."""

    def __str__(self) -> str:
        return "project_id is required when authenticating with a user-level API Key"


def query_project(
    client: _Codex,
    question: str,
    project_id: str,
    *,
    fallback_answer: Optional[str] = None,
    read_only: bool = False,
) -> tuple[Optional[str], Optional[Entry]]:
    maybe_entry = client.projects.entries.query(project_id, question=question)
    if maybe_entry is not None:
        entry = Entry.model_validate(maybe_entry.model_dump())
        if entry.answer is not None:
            return entry.answer, entry

        return fallback_answer, entry

    if not read_only:
        created_entry = Entry.model_validate(
            client.projects.entries.add_question(project_id, question=question).model_dump()
        )
        return fallback_answer, created_entry

    return fallback_answer, None
