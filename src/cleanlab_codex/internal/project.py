from typing import Optional
from codex import Codex as _Codex

from cleanlab_codex.types.entry import Entry
from cleanlab_codex.types.project import ProjectConfig


def create_project(
    client: _Codex, name: str, organization_id: str, description: Optional[str] = None
) -> int:
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
    project_id: Optional[int] = None,
    fallback_answer: Optional[str] = None,
    read_only: bool = False,
) -> tuple[Optional[str], Optional[Entry]]:
    if client.access_key is not None:
        project_id = client.projects.access_keys.retrieve_project_id()
    elif project_id is None:
        raise ValueError(
            "project_id is required when authenticating with a user-level API Key"
        )

    query_res = client.projects.knowledge.query(project_id, question=question)
    if query_res is not None:
        if query_res.answer is not None:
            return query_res.answer, query_res
        else:
            return fallback_answer, query_res
    elif not read_only:
        created_entry = client.projects.knowledge.add_question(
            project_id, question=question
        )
        return fallback_answer, created_entry
    else:
        return fallback_answer, None
