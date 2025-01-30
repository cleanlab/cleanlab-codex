from __future__ import annotations

from beartype.typing import Optional, Tuple
from codex import Codex as _Codex

from cleanlab_codex.types.entry import Entry
from cleanlab_codex.types.project import ProjectConfig


class MissingProjectIdError(Exception):
    """Raised when the project ID is not provided."""

    def __str__(self) -> str:
        return "project_id is required when authenticating with a user-level API Key"


def create_project(client: _Codex, name: str, organization_id: str, description: Optional[str] = None) -> str:
    project = client.projects.create(
        config=ProjectConfig(),
        organization_id=organization_id,
        name=name,
        description=description,
    )
    return project.id


def query_project(
    client: _Codex,
    question: str,
    *,
    project_id: Optional[str] = None,
    fallback_answer: Optional[str] = None,
    read_only: bool = False,
) -> Tuple[Optional[str], Optional[Entry]]:
    if client.access_key is not None:
        project_id = client.projects.access_keys.retrieve_project_id().project_id
    elif project_id is None:
        raise MissingProjectIdError

    query_res = client.projects.entries.query(project_id, question=question)
    if query_res is not None:
        if query_res.answer is not None:
            return query_res.answer, query_res

        return fallback_answer, query_res

    if not read_only:
        created_entry = client.projects.entries.add_question(project_id, question=question)
        return fallback_answer, created_entry

    return fallback_answer, None
